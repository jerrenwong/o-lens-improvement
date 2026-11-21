"""ask2 step 1b + 2: fit E_spec exactly as scripts/e2_fit.py (main map + shuffled-target control, same ridge, same
c grid and validation rule), with the new specific-word targets and junk-target dropping; plus label-free
diagnostics old topic formats vs new specific formats.

Junk: for each prefix and format, if the UNMASKED top-1 token of unembed(Y_k) is a special/added token or pure
whitespace, that format's vector is dropped; target = mean of remaining unit vectors; prefix dropped if none remain.
Elicited word (as e2_fit): top-1 of unembed(target) after masking non-word tokens and the fixed function-word list.
Usage: python ask2_fit.py <model>
"""
import json
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

R = "/workspace/o-lens-improvement"
M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = "27b" if "27B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56]}[TAG]
JPP_PATH = {"9b": "artifacts/baselines/jpp9b_prefit.pt", "27b": "artifacts/baselines/jpp27b_released.pt"}[TAG]
OUT = f"{R}/.hf27/ask2"
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000
torch.set_grad_enabled(False)
dev = "cuda"


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


data = json.load(open(f"{R}/runs/e1/prefixes.json"))
N = len(data)
A = torch.load(f"{OUT}/acts_{TAG}.pt")
if TAG == "27b":
    B = torch.load(f"{R}/.hf27/m27/acts_27b.pt")
    X_all, Y_old = B["X"], B["Y"]
    del B
else:
    X_all, Y_old = A["X"], A["Y_old"]
Y_new = A["Y_new"]
model = get_hf_model(M, attn_implementation="sdpa")
tok = model.tokenizer
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/{JPP_PATH}", hf_model_name=M).jacobians_L_dict_FN.items()}
V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
func = torch.tensor([s.strip().lower() in FUNC for s in strs], device=dev)
mask = nonword | func
norm_s = [s.strip().lower() for s in strs]
vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
special_ids = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special_ids |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
junk_tok = torch.zeros(V, dtype=torch.bool, device=dev)
junk_tok[[i for i in special_ids if i < V]] = True
junk_tok |= torch.tensor([s.strip() == "" for s in strs], device=dev)
print(f"junk vocabulary: {int(junk_tok.sum())} ids (special/added {len(special_ids)}; whitespace-only "
      f"{sum(s.strip() == '' for s in strs)})", flush=True)


def topk(v, k=10, bs=250, masked=True):
    out = []
    for i in range(0, v.shape[0], bs):
        lg = model.unembed(v[i:i + bs].to(dev).float()).float()
        if masked:
            lg = lg.masked_fill(mask, float("-inf"))
        out.append(lg.topk(k, -1).indices)
    return torch.cat(out)


def agree(top, word):
    return (NORM[top] == NORM[word][:, None]).any(-1)


va_r, te_r, tr_r = range(0, NV), range(NV, NV + NT), range(NV + NT, N)
te_idx = torch.arange(NV, NV + NT, device=dev)


def visible(i, wid):
    w = norm_s[wid]
    t = data[i]["text"].lower()
    return bool(w) and re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", t) is not None


def visible_sub(i, wid):
    w = norm_s[wid]
    return len(w) >= 3 and w in data[i]["text"].lower()


def analyse(name, Y, drop_junk):
    """Y: [3, N, d]. Returns dict with targets, words, keep mask, diagnostics."""
    Yk = [unit(Y[k].float()) for k in range(3)]
    raw1 = [topk(y, 1, masked=False)[:, 0] for y in Yk]  # unmasked top-1 per format
    junk = torch.stack([junk_tok[r] for r in raw1])  # [3, N]
    if drop_junk:
        keepk = ~junk
        cnt = keepk.sum(0)
        T = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
        valid = cnt > 0
    else:
        keepk = torch.ones_like(junk)
        T = sum(Yk) / 3
        valid = torch.ones(N, dtype=torch.bool, device=dev)
    wk = [topk(y, 1)[:, 0] for y in Yk]  # masked top-1 per format
    wmean = topk(T, 10)
    word = wmean[:, 0]
    d = dict(name=name, T=T, valid=valid, word=word, wmean=wmean, wk=wk, raw1=raw1, junk=junk, keepk=keepk)
    te = slice(NV, NV + NT)
    tr = slice(NV + NT, N)
    vt = valid[te]
    d["junk_share_fmt"] = [junk[k, te].float().mean().item() for k in range(3)]
    d["junk_share_any"] = junk[:, te].any(0).float().mean().item()
    d["junk_all"] = (junk.all(0)).sum().item()
    d["dropped_vectors"] = (junk & torch.ones_like(junk)).sum().item() if drop_junk else 0
    trw = [norm_s[t] for t, v in zip(word[tr].tolist(), valid[tr].tolist()) if v]
    c = Counter(trw)
    d["distinct_train"] = len(c)
    d["n_train"] = len(trw)
    d["top10_share_train"] = sum(n for _, n in c.most_common(10)) / max(1, len(trw))
    d["common"] = c.most_common(20)
    tew = [(i, w) for i, w, v in zip(te_r, word[te].tolist(), vt.tolist()) if v]
    d["n_test"] = len(tew)
    d["visible"] = sum(visible(i, w) for i, w in tew) / len(tew)
    d["visible_sub"] = sum(visible_sub(i, w) for i, w in tew) / len(tew)
    d["special_word"] = sum(junk_tok[w].item() for _, w in tew) / len(tew)
    # format agreement on test, only where both formats kept
    pa = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        both = keepk[a, te] & keepk[b, te]
        eq = (NORM[wk[a][te]] == NORM[wk[b][te]])[both]
        pa.append((eq.float().mean().item() if both.any() else float("nan"), int(both.sum())))
    d["pair_agree"] = pa
    all3 = keepk[:, te].all(0)
    eq3 = (NORM[wk[0][te]] == NORM[wk[1][te]]) & (NORM[wk[0][te]] == NORM[wk[2][te]])
    d["all3_agree"] = (eq3[all3].float().mean().item() if all3.any() else float("nan"), int(all3.sum()))
    print(f"\n== {name}: junk top-1 share per format (test) "
          + " ".join(f"{x:.3f}" for x in d["junk_share_fmt"])
          + f"; any {d['junk_share_any']:.3f}; prefixes with all three junk (all N) {d['junk_all']}", flush=True)
    for k in range(3):
        cc = Counter(strs[t] for t, j in zip(raw1[k].tolist(), junk[k].tolist()) if j)
        print(f"   fmt{k} junk tokens (all N): {cc.most_common(6)}; dropped {int(junk[k].sum())}/{N}")
    print(f"   distinct train words {d['distinct_train']} of {d['n_train']}; top-10 share {d['top10_share_train']:.3f}")
    print(f"   most common: {d['common']}")
    print(f"   test n={d['n_test']}: visible-in-prefix (whole word) {d['visible']:.3f}, substring {d['visible_sub']:.3f}, "
          f"target word itself special {d['special_word']:.3f}")
    print(f"   format agreement (test, both kept): pairs " + " ".join(f"{a:.3f}(n={n})" for a, n in pa)
          + f"; all three {d['all3_agree'][0]:.3f}(n={d['all3_agree'][1]})", flush=True)
    return d


OLDD = analyse("OLD topic formats (no junk drop, as fitted)", Y_old, False)
OLDJ = analyse("OLD topic formats WITH junk drop (reference only)", Y_old, True)
NEWD = analyse("NEW specific formats (junk dropped)", Y_new, True)
print(f"\nNEW: dropped format-vectors {int(NEWD['junk'].sum())} of {3 * N}; dropped prefixes {int((~NEWD['valid']).sum())} "
      f"(train {int((~NEWD['valid'][NV + NT:]).sum())}, val {int((~NEWD['valid'][:NV]).sum())}, "
      f"test {int((~NEWD['valid'][NV:NV + NT]).sum())})", flush=True)

print("\n== 20 test prefixes: old E0/E1/E2 -> old target | new S0/S1/S2 (raw top-1 in [] when junk) -> new target")
for i in range(NV, NV + 20):
    def fmtw(D, k):
        return strs[D["wk"][k][i]].strip() if not D["junk"][k, i] or D is OLDD else f"[{strs[D['raw1'][k][i]]!r}]"
    print(f"   [{data[i]['src']}/{data[i]['kind']}] {data[i]['text'][-100:]!r}\n"
          f"        old {'/'.join(fmtw(OLDD, k) for k in range(3))} -> {strs[OLDD['word'][i]].strip()} | "
          f"new {'/'.join(fmtw(NEWD, k) for k in range(3))} -> "
          f"{strs[NEWD['word'][i]].strip() if NEWD['valid'][i] else 'DROPPED'}")


class Ridge:
    def __init__(self, X):
        self.mx = X.mean(0)
        self.Xc = (X - self.mx).double()
        s, U = torch.linalg.eigh(self.Xc.T @ self.Xc)
        self.s, self.U = s.clamp_min(0), U
        self.scale = self.s.mean().item()

    def setY(self, Y):
        self.my = Y.mean(0)
        self.UtB = self.U.T @ (self.Xc.T @ (Y - self.my).double())

    def W(self, c):
        return (self.U @ (self.UtB / (self.s + c * self.scale)[:, None])).float()

    def predict(self, X, W):
        return (X - self.mx) @ W + self.my


def pack(r, W, c):
    sc = W.abs().max().item()
    return dict(mx=r.mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=r.my.cpu(), c=c)


def load_maps(path):
    m = torch.load(path, map_location="cpu")
    return {l: (v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()) for l, v in m.items()}


valid = NEWD["valid"]
tr_m = valid.clone(); tr_m[:NV + NT] = False
va_m = valid.clone(); va_m[NV:] = False
te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
tr_i, va_i, te_i = (torch.nonzero(m)[:, 0].cpu() for m in (tr_m, va_m, te_m))
Ynew = NEWD["T"]
wnew = NEWD["word"]
g = torch.Generator().manual_seed(1)
perm = torch.randperm(len(tr_i), generator=g)
print(f"\ntrain {len(tr_i)}, val {len(va_i)}, test {len(te_i)} (after dropping junk-only prefixes)")
prior_top = topk(Ynew[tr_i].mean(0, keepdim=True)).expand(len(te_i), -1)
freq = torch.tensor([t for t, _ in Counter(wnew[tr_i.to(dev)].tolist()).most_common(10)], device=dev)[None].expand(len(te_i), -1)
prior_a = agree(prior_top, wnew[te_i.to(dev)]).float().mean().item()
freq_a = agree(freq, wnew[te_i.to(dev)]).float().mean().item()
# old (topic) prior numbers for the comparison table (old target, all test prefixes)
wold = OLDD["word"]
otr = torch.arange(NV + NT, N)
ote = torch.arange(NV, NV + NT)
oprior = agree(topk(OLDD["T"][otr].mean(0, keepdim=True)).expand(NT, -1), wold[ote.to(dev)]).float().mean().item()
ofreq_t = torch.tensor([t for t, _ in Counter(wold[otr.to(dev)].tolist()).most_common(10)], device=dev)[None].expand(NT, -1)
ofreq = agree(ofreq_t, wold[ote.to(dev)]).float().mean().item()
OLDMAP = load_maps(f"{R}/runs/e1/E_{TAG}.pt")

maps = {"spec": {}, "spec_shuf": {}}
rows = []
anyE = torch.zeros(len(te_i), dtype=torch.bool, device=dev)
anyO = torch.zeros(NT, dtype=torch.bool, device=dev)
for l in LAYERS:
    X = X_all[l].float().to(dev)
    r = Ridge(X[tr_i.to(dev)])
    r.setY(Ynew[tr_i].to(dev))
    sc = [agree(topk(r.predict(X[va_i.to(dev)], r.W(c_))), wnew[va_i.to(dev)]).float().mean().item() for c_ in GRID]
    b = max(range(len(GRID)), key=lambda i: (sc[i], i))
    c = GRID[b]
    W = r.W(c)
    tE = topk(r.predict(X[te_i.to(dev)], W))
    wt = wnew[te_i.to(dev)]
    hE10, hE1 = agree(tE, wt), agree(tE[:, :1], wt)
    maps["spec"][l] = pack(r, W, c)
    tJ = topk(X[te_i.to(dev)] @ JPP[l].T)
    hJ10, hJ1 = agree(tJ, wt), agree(tJ[:, :1], wt)
    r.setY(Ynew[tr_i][perm].to(dev))
    Ws = r.W(c)
    hS10 = agree(topk(r.predict(X[te_i.to(dev)], Ws)), wt)
    maps["spec_shuf"][l] = pack(r, Ws, c)
    # old topic map vs old word (same test prefixes as the original e2_fit, all 1000)
    mx, Wo, my = OLDMAP[l]
    tO = topk((X[ote.to(dev)] - mx) @ Wo + my)
    hO10, hO1 = agree(tO, wold[ote.to(dev)]), agree(tO[:, :1], wold[ote.to(dev)])
    # how often does the NEW map's top-10 contain the OLD word, and the old map's top-10 the new word
    xO_new = agree(topk((X[te_i.to(dev)] - mx) @ Wo + my), wt).float().mean().item()
    anyE |= hE10
    anyO |= hO10
    del r
    torch.cuda.empty_cache()
    rows.append((l, c, hE10.float().mean().item(), hE1.float().mean().item(), hJ10.float().mean().item(),
                 hJ1.float().mean().item(), hS10.float().mean().item(), hO10.float().mean().item(),
                 hO1.float().mean().item(), xO_new))
    print(f"L{l:2d} c={c:g} val@10 by c: " + " ".join(f"{a:g}:{s:.3f}" for a, s in zip(GRID, sc)), flush=True)

print(f"\n== held-out agreement with the elicited word (test prefixes; word-level). NEW n={len(te_i)}, OLD n={NT}")
print("layer | c | E_spec top10 | E_spec top1 | J++ top10 (new word) | J++ top1 | shuffled top10 | E_topic top10 (old word) | E_topic top1 | E_topic top10 vs NEW word")
for (l, c, e10, e1_, j10, j1, s10, o10, o1, x) in rows:
    print(f"L{l:2d} | {c:g} | {e10:.3f} | {e1_:.3f} | {j10:.3f} | {j1:.3f} | {s10:.3f} | {o10:.3f} | {o1:.3f} | {x:.3f}")
print(f"any layer: E_spec {anyE.float().mean():.3f}  E_topic {anyO.float().mean():.3f}")
print(f"prior-only NEW: unembed(mean target) top10 {prior_a:.3f}; 10 most frequent train words {freq_a:.3f}")
print(f"prior-only OLD: unembed(mean target) top10 {oprior:.3f}; 10 most frequent train words {ofreq:.3f}")
for v, m in maps.items():
    torch.save(m, f"{OUT}/E_{TAG}_{v}.pt")
json.dump(dict(new_word=[strs[t] for t in NEWD["word"].tolist()], old_word=[strs[t] for t in OLDD["word"].tolist()],
               valid=NEWD["valid"].tolist()), open(f"{OUT}/words_{TAG}.json", "w"))
print("saved", flush=True)
