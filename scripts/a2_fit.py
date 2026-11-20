"""a2 copy of m27_fit.py (PREREG_v3 recipe A'): only input/output paths changed (marked "a2"): X from .hf27/m27/acts_27b.pt,
Y from .hf27/a2/Y_ws.pt; maps to .hf27/a2/maps_27b_ws_*.pt, copies runs/e1/E_27b_ws.pt, E_27b_ws_shuf.pt.
One report-only block appended at the end (old E_27b.pt scored against the new elicited word); fitting untouched.
m27 copy of e2_fit.py for Qwen/Qwen3.6-27B (changes marked "m27"; recipe unchanged).
e2 (= e1 step 3, model as argv[1]; same recipe): per-layer ridge maps E_l: prefix-token resid -> elicitation target (label-free).

Target = mean of the three unit-normalised elicitation vectors. Elicited word = top-1 token of unembed(target)
after masking non-word tokens and a fixed function-word list; words compared case/space-insensitively.
Split of the 30.5k prefixes: val 1000 (ridge strength), test 1000 (reported), train the rest.
Ridge: centred, bias, closed form via eigh of X^T X (float64, GPU), lambda = c * mean eigenvalue.
Saves /root/e1/maps_<variant>.pt ({layer: (mx fp32, W/scale fp16, scale, my fp32)}) and copies main to runs/e1.
"""
import json
import re
import shutil
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

R = "/workspace/o-lens-improvement"
M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-9B"
TAG = "27b"  # m27
LAYERS = [8, 16, 24, 32, 40, 48, 56]  # m27: 27B readout layers
JPP_PATH = "artifacts/baselines/jpp27b_released.pt"  # m27: released J++ (koayon/jpp-lenses qwen3.6-27b/lens.pt)
SUF = "_27b"  # m27
OUT27 = "/workspace/o-lens-improvement/.hf27/a2"  # a2: outputs
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
NV, NT = 1000, 1000
torch.set_grad_enabled(False)
dev = "cuda"


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


data = json.load(open("/workspace/o-lens-improvement/runs/e1/prefixes.json"))  # m27: repo copy (same md5)
A = torch.load("/workspace/o-lens-improvement/.hf27/m27/acts_27b.pt")  # a2: X reused
A["Y"] = torch.load(f"{OUT27}/Y_ws.pt")["Y"]  # a2: recollected targets
N = len(data)
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


def topk(v, k=10, bs=250):
    out = []
    for i in range(0, v.shape[0], bs):
        lg = model.unembed(v[i:i + bs].to(dev).float()).float()
        out.append(lg.masked_fill(mask, float("-inf")).topk(k, -1).indices)
    return torch.cat(out)


def agree(top, word):
    """word-level (case/space-insensitive) hit of the target token within top-k rows."""
    return (NORM[top] == NORM[word][:, None]).any(-1)


Yk = [unit(A["Y"][k].float()) for k in range(3)]
Ymean = sum(Yk) / 3
va, te, tr = slice(0, NV), slice(NV, NV + NT), slice(NV + NT, N)
w_k = [topk(y, 3) for y in Yk]
w_mean = topk(Ymean, 10)
word = w_mean[:, 0]
word_k = [w[:, 0] for w in w_k]

# ---- elicitation agreement on the 1000 test prefixes
s = te
eq01 = NORM[word_k[0][s]] == NORM[word_k[1][s]]
eq02 = NORM[word_k[0][s]] == NORM[word_k[2][s]]
eq12 = NORM[word_k[1][s]] == NORM[word_k[2][s]]
print(f"\n== elicitation agreement (test {NT}): top-1 all three {(eq01 & eq02).float().mean():.3f}; "
      f"pairs E0=E1 {eq01.float().mean():.3f} E0=E2 {eq02.float().mean():.3f} E1=E2 {eq12.float().mean():.3f}; "
      f"at least two {(eq01 | eq02 | eq12).float().mean():.3f}")
for k in range(3):
    print(f"   E{k} top-1 = mean-target word: {(NORM[word_k[k][s]] == NORM[word[s]]).float().mean():.3f}")
srcs = [data[i]["src"] for i in range(NV, NV + NT)]
for g in ("wiki_en", "story", "wiki_"):
    m = torch.tensor([x == g or (g == "wiki_" and x not in ("wiki_en", "story")) for x in srcs], device=dev)
    print(f"   {g if g != 'wiki_' else 'non-English'}: n={int(m.sum())} all-three {((eq01 & eq02)[m]).float().mean():.3f}")
c = Counter(norm_s[t] for t in word[tr].tolist())
print("   most common elicited words (train):", c.most_common(25))
print(f"   distinct elicited words (train): {len(c)} of {N - NV - NT}")
print("\n== 15 test examples: prefix -> E0/E1/E2 top-1 | mean-target top-3")
for i in range(NV, NV + 15):
    print(f"   [{data[i]['src']}/{data[i]['kind']}] {data[i]['text'][-110:]!r}\n"
          f"        -> {'/'.join(strs[w_k[k][i, 0]].strip() for k in range(3))} | "
          f"{', '.join(strs[t].strip() for t in w_mean[i, :3].tolist())}")


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


def fit_variant(name, r, Ytr, wval, Xva):
    r.setY(Ytr)
    sc = []
    for c_ in GRID:
        sc.append(agree(topk(r.predict(Xva, r.W(c_))), wval).float().mean().item())
    b = max(range(len(GRID)), key=lambda i: (sc[i], i))
    return GRID[b], sc


maps = {v: {} for v in ("main", "shuf", "e0", "e1", "e2", "sent")}
rows = []
anyE, anyJ, anyS = (torch.zeros(NT, dtype=torch.bool, device=dev) for _ in range(3))
g = torch.Generator().manual_seed(1)
perm = torch.randperm(N - NV - NT, generator=g)
sent_tr = torch.tensor([data[i]["kind"] == "sent" for i in range(NV + NT, N)])
print(f"\ntrain {N - NV - NT} (sentence-final {int(sent_tr.sum())}), val {NV}, test {NT}")
prior_top = topk(Ymean[tr].mean(0, keepdim=True)).expand(NT, -1)
freq = torch.tensor([t for t, _ in Counter(word[tr].tolist()).most_common(10)], device=dev)[None].expand(NT, -1)
prior_a = agree(prior_top, word[te]).float().mean().item()
freq_a = agree(freq, word[te]).float().mean().item()
for l in LAYERS:
    X = A["X"][l].float().to(dev)
    r = Ridge(X[tr])
    c, sc = fit_variant("main", r, Ymean[tr].to(dev), word[va], X[va])
    W = r.W(c)
    pred = r.predict(X[te], W)
    tE = topk(pred)
    hE10, hE1 = agree(tE, word[te]), agree(tE[:, :1], word[te])
    maps["main"][l] = pack(r, W, c)
    tJ = topk(X[te] @ JPP[l].T)
    hJ10, hJ1 = agree(tJ, word[te]), agree(tJ[:, :1], word[te])
    # shuffled targets, same c
    r.setY(Ymean[tr][perm].to(dev))
    Ws = r.W(c)
    tS = topk(r.predict(X[te], Ws))
    hS10 = agree(tS, word[te])
    maps["shuf"][l] = pack(r, Ws, c)
    anyE |= hE10
    anyJ |= hJ10
    anyS |= hS10
    # per-elicitation maps (own word, own c)
    pe = []
    for k in range(3):
        ck, _ = fit_variant(f"e{k}", r, Yk[k][tr].to(dev), word_k[k][va], X[va])
        Wk = r.W(ck)
        pe.append(agree(topk(r.predict(X[te], Wk)), word_k[k][te]).float().mean().item())
        maps[f"e{k}"][l] = pack(r, Wk, ck)
    del r
    # sentence-final positions only
    rs = Ridge(X[tr][sent_tr.to(dev)])
    cs, _ = fit_variant("sent", rs, Ymean[tr][sent_tr].to(dev), word[va], X[va])
    Wsn = rs.W(cs)
    hSent = agree(topk(rs.predict(X[te], Wsn)), word[te]).float().mean().item()
    maps["sent"][l] = pack(rs, Wsn, cs)
    del rs
    torch.cuda.empty_cache()
    rows.append((l, c, sc, hE10.float().mean().item(), hE1.float().mean().item(), hJ10.float().mean().item(),
                 hJ1.float().mean().item(), hS10.float().mean().item(), pe, hSent))
    print(f"L{l:2d} c={c:g} val@10 by c: " + " ".join(f"{a:g}:{b:.3f}" for a, b in zip(GRID, sc)), flush=True)

print(f"\n== held-out agreement with the elicited word (test {NT} prefixes; word-level)")
print("layer | c | E top10 | E top1 | J++ top10 | J++ top1 | shuffled top10 | E_e0/e1/e2 top10 (own word) | E_sent top10")
for (l, c, sc, e10, e1_, j10, j1, s10, pe, hs) in rows:
    print(f"L{l:2d} | {c:g} | {e10:.3f} | {e1_:.3f} | {j10:.3f} | {j1:.3f} | {s10:.3f} | "
          f"{'/'.join(f'{x:.3f}' for x in pe)} | {hs:.3f}")
print(f"any layer: E {anyE.float().mean():.3f}  J++ {anyJ.float().mean():.3f}  shuffled {anyS.float().mean():.3f}  "
      f"E|J++ {(anyE | anyJ).float().mean():.3f}")
print(f"prior-only: unembed(mean target) top10 {prior_a:.3f}; 10 most frequent train words {freq_a:.3f}")
for v, m in maps.items():
    torch.save(m, f"{OUT27}/maps{SUF}_ws_{v}.pt")  # a2
for v, dst in (("main", f"E_{TAG}_ws.pt"), ("shuf", f"E_{TAG}_ws_shuf.pt")):  # a2
    with open(f"{OUT27}/maps{SUF}_ws_{v}.pt", "rb") as a, open(f"{R}/runs/e1/{dst}", "wb") as b:  # m27
        shutil.copyfileobj(a, b)
print("saved", flush=True)

# ---- a2 report-only: OLD recipe-A map (runs/e1/E_27b.pt) scored against the NEW elicited word (test slice)
old = torch.load(f"{R}/runs/e1/E_27b.pt", map_location="cpu")
print("\n== a2: old E (E_27b.pt) vs NEW elicited word (test 1000): layer | top10 | top1")
anyO = torch.zeros(NT, dtype=torch.bool, device=dev)
for l in LAYERS:
    v = old[l]
    X = A["X"][l].float().to(dev)
    pred = (X[te] - v["mx"].to(dev)) @ (v["W"].to(dev).float() * v["scale"]) + v["my"].to(dev)
    t = topk(pred)
    h10 = agree(t, word[te])
    anyO |= h10
    print(f"L{l:2d} | {h10.float().mean():.3f} | {agree(t[:, :1], word[te]).float().mean():.3f}")
print(f"any layer: old E {anyO.float().mean():.3f}")
