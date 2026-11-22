"""xm1 step 5b: ridge ask-map for a new model. Model-generic copy of scripts/ask3/ask3_fit9.py with the parts that
make the maps UNCHANGED (junk rule, target = mean of kept unit vectors, elicited word = masked top-1 of unembed(target),
splits val = passages [0,1000), test = [1000,2000), train = rest, Ridge class, c grid, validation rule = per layer the c
with the largest val top-10 word agreement (ties -> larger c), shuffled-target control with the same c and the
permutation seed 1, packing). Removed: the OLD-topic-format diagnostics (no old targets exist for new models).
X / Y_new from scripts/xm1/collect.py; J++ = the new model's fit (runs/xm1/<tag>/jpp_<tag>.pt), diagnostics only.
Outputs .hf27/xm1/E_<tag>_t3.pt, E_<tag>_t3_shuf.pt, words_<tag>.json.
    python scripts/xm1/fit_map.py <model> <tag>"""
import json
import re
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

R = "/workspace/o-lens-improvement"
M, TAG = sys.argv[1], sys.argv[2]
OUT = f"{R}/.hf27/xm1"
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
Q = __import__("os").environ.get("Q", "t3")  # PREREG_v5 Amendment 1: Q=s5 -> "Unstated word:" map (as scripts/ask5/ask5_fit.py)
SFX = "" if Q == "t3" else "_s5"
A = torch.load(f"{OUT}/acts_{TAG}{SFX}.pt")
if Q == "s5":
    A["X"] = torch.load(f"{OUT}/X_{TAG}.pt")  # same passages, same X as the topic map
X_all, Y_new, LAYERS = A["X"], A["Y_new"], A["layers"]
del A
model = None
for attempt in range(3):
    try:
        model = get_hf_model(M, attn_implementation="sdpa")
        break
    except Exception as e:  # noqa: BLE001
        print("load retry", attempt, repr(e)[:300], flush=True)
tok = model.tokenizer
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/runs/xm1/{TAG}/jpp_{TAG}.pt", hf_model_name=M).jacobians_L_dict_FN.items()}
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


te_r = range(NV, NV + NT)


def visible(i, wid):
    w = norm_s[wid]
    t = data[i]["text"].lower()
    return bool(w) and re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", t) is not None


def analyse(name, Y):
    Yk = [unit(Y[k].float()) for k in range(3)]
    raw1 = [topk(y, 1, masked=False)[:, 0] for y in Yk]
    junk = torch.stack([junk_tok[r] for r in raw1])
    keepk = ~junk
    cnt = keepk.sum(0)
    T = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
    valid = cnt > 0
    wk = [topk(y, 1)[:, 0] for y in Yk]
    word = topk(T, 10)[:, 0]
    te, tr = slice(NV, NV + NT), slice(NV + NT, N)
    trw = [norm_s[t] for t, v in zip(word[tr].tolist(), valid[tr].tolist()) if v]
    c = Counter(trw)
    tew = [(i, w) for i, w, v in zip(te_r, word[te].tolist(), valid[te].tolist()) if v]
    print(f"\n== {name}: junk top-1 share per format (test) "
          + " ".join(f"{junk[k, te].float().mean().item():.3f}" for k in range(3))
          + f"; any {junk[:, te].any(0).float().mean().item():.3f}; all three junk (all N) {int(junk.all(0).sum())}")
    for k in range(3):
        cc = Counter(strs[t] for t, j in zip(raw1[k].tolist(), junk[k].tolist()) if j)
        print(f"   fmt{k} junk tokens (all N): {cc.most_common(6)}; dropped {int(junk[k].sum())}/{N}")
    print(f"   distinct train words {len(c)} of {len(trw)}; top-10 share {sum(n for _, n in c.most_common(10)) / max(1, len(trw)):.3f}")
    print(f"   most common: {c.most_common(20)}")
    print(f"   test n={len(tew)}: visible-in-prefix (whole word) {sum(visible(i, w) for i, w in tew) / len(tew):.3f}")
    pa = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        both = keepk[a, te] & keepk[b, te]
        pa.append(((NORM[wk[a][te]] == NORM[wk[b][te]])[both].float().mean().item(), int(both.sum())))
    print("   format agreement (test, both kept): " + " ".join(f"{a:.3f}(n={n})" for a, n in pa), flush=True)
    return dict(T=T, valid=valid, word=word, junk=junk, wk=wk)


NEWD = analyse("ask3 topic formats (junk dropped)", Y_new)
print(f"\ndropped format-vectors {int(NEWD['junk'].sum())} of {3 * N}; dropped passages {int((~NEWD['valid']).sum())}")
print("\n== 20 test passages: W0/W1/W2 masked top-1 -> target word")
for i in range(NV, NV + 20):
    print(f"   [{data[i]['src']}] {data[i]['text'][-90:]!r} -> "
          f"{'/'.join(strs[NEWD['wk'][k][i]].strip() for k in range(3))} -> "
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


valid = NEWD["valid"]
tr_m = valid.clone(); tr_m[:NV + NT] = False
va_m = valid.clone(); va_m[NV:] = False
te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
tr_i, va_i, te_i = (torch.nonzero(m)[:, 0].cpu() for m in (tr_m, va_m, te_m))
Ynew, wnew = NEWD["T"], NEWD["word"]
g = torch.Generator().manual_seed(1)
perm = torch.randperm(len(tr_i), generator=g)
print(f"\ntrain {len(tr_i)}, val {len(va_i)}, test {len(te_i)} (after dropping junk-only passages)")
prior_top = topk(Ynew[tr_i].mean(0, keepdim=True)).expand(len(te_i), -1)
freq = torch.tensor([t for t, _ in Counter(wnew[tr_i.to(dev)].tolist()).most_common(10)], device=dev)[None].expand(len(te_i), -1)
prior_a = agree(prior_top, wnew[te_i.to(dev)]).float().mean().item()
freq_a = agree(freq, wnew[te_i.to(dev)]).float().mean().item()
maps = {Q: {}, Q + "_shuf": {}}
rows = []
anyE = torch.zeros(len(te_i), dtype=torch.bool, device=dev)
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
    maps[Q][l] = pack(r, W, c)
    tJ = topk(X[te_i.to(dev)] @ JPP[l].T)
    hJ10, hJ1 = agree(tJ, wt), agree(tJ[:, :1], wt)
    r.setY(Ynew[tr_i][perm].to(dev))
    Ws = r.W(c)
    hS10 = agree(topk(r.predict(X[te_i.to(dev)], Ws)), wt)
    maps[Q + "_shuf"][l] = pack(r, Ws, c)
    anyE |= hE10
    del r
    torch.cuda.empty_cache()
    rows.append((l, c, hE10.float().mean().item(), hE1.float().mean().item(), hJ10.float().mean().item(),
                 hJ1.float().mean().item(), hS10.float().mean().item()))
    print(f"L{l:2d} c={c:g} val@10 by c: " + " ".join(f"{a:g}:{s:.3f}" for a, s in zip(GRID, sc)), flush=True)
print(f"\n== held-out agreement with the elicited word (test passages, word-level), n={len(te_i)}")
print("layer | c | E top10 | E top1 | J++ top10 | J++ top1 | shuffled top10")
for (l, c, e10, e1_, j10, j1, s10) in rows:
    print(f"L{l:2d} | {c:g} | {e10:.3f} | {e1_:.3f} | {j10:.3f} | {j1:.3f} | {s10:.3f}")
print(f"any layer: E {anyE.float().mean():.3f}")
print(f"prior-only: unembed(mean target) top10 {prior_a:.3f}; 10 most frequent train words {freq_a:.3f}")
for v, m in maps.items():
    with open(f"{OUT}/E_{TAG}_{v}.pt", "wb") as f:
        torch.save(m, f)
with open(f"{OUT}/words_{TAG}{SFX}.json", "w") as f:
    json.dump(dict(word=[strs[t] for t in NEWD["word"].tolist()], valid=NEWD["valid"].tolist()), f)
with open(f"{R}/runs/xm1/{TAG}/map_fit{SFX}.json", "w") as f:
    json.dump(dict(rows=[dict(layer=l, c=c, E10=e10, E1=e1_, J10=j10, J1=j1, shuf10=s10) for (l, c, e10, e1_, j10, j1, s10) in rows],
                   prior_top10=prior_a, freq_top10=freq_a, n_train=len(tr_i), n_val=len(va_i), n_test=len(te_i),
                   dropped_vectors=int(NEWD["junk"].sum()), dropped_passages=int((~valid).sum())), f, indent=1)
print("saved", flush=True)
