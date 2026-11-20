"""g1 diag (not part of the design): train accuracy, centred (bias-free) scoring, guessable subset.
former: g1 step 4: ridge maps G_l: last-token residual of a generated sentence -> t_w (hidden concept target, final basis).

Split BY CONCEPT (seed 0): 10% val, 10% test, 80% train. Ridge with bias exactly as scripts/e2_fit.py (centred X,
closed form via eigh, lambda = c * mean eigenvalue, grid GRID); c chosen by val top-10 word-level accuracy for the
concept (masked: non-word + function words, as e2_fit).
Maps: G_all (all train concepts; val = all val concepts), G_strict (STRICT train/val concepts),
G_shuf (STRICT train, targets permuted across sentences, c of G_strict).
Report on held-out-concept test sentences (STRICT test concepts; also G_all on all test concepts):
top-10 / top-1 for G_strict, J++, E (runs/e1/E_9b.pt), G_shuf, G_all; any-layer.
Saves runs/g1/G_{strict,all,shuf}.pt in the e2 map format {layer: {mx, W(half)/scale, scale, my, c}}.
"""
import json
import random
import re
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

R = "/workspace/o-lens-improvement"
M = "Qwen/Qwen3.5-9B"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
torch.set_grad_enabled(False)
dev = "cuda"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


A = torch.load(f"{R}/.hf27/g1/acts.pt")
words, tid = A["words"], torch.tensor(A["tid"], device=dev)
T = A["T"].to(dev).float()
N = len(words)
strict = set(json.load(open(f"{R}/runs/g1/concepts_strict.json")))
model = get_hf_model(M, attn_implementation="sdpa")
tok = model.tokenizer
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN.items()}
EA = torch.load(f"{R}/runs/e1/E_9b.pt", map_location="cpu")
V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = []
for i in range(V):
    try:
        strs.append(tok.decode([i]))
    except Exception:
        strs.append("")
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


def agree(top, w):
    return (NORM[top] == NORM[w][:, None]).any(-1)


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


# ---- split by concept
cw = sorted(set(words))
random.Random(0).shuffle(cw)
nv = len(cw) // 10
role = {w: ("val" if i < nv else "test" if i < 2 * nv else "train") for i, w in enumerate(cw)}
isS = torch.tensor([w in strict for w in words], device=dev)
R_ = {r: torch.tensor([role[w] == r for w in words], device=dev) for r in ("train", "val", "test")}
idx = {
    "all_tr": R_["train"], "all_va": R_["val"], "all_te": R_["test"],
    "s_tr": R_["train"] & isS, "s_va": R_["val"] & isS, "s_te": R_["test"] & isS,
}
idx = {k: v.nonzero()[:, 0] for k, v in idx.items()}
log(f"sentences {N}, concepts {len(cw)} (strict {len(set(words) & strict)}); " +
    " ".join(f"{k} {len(v)}" for k, v in idx.items()))
log(f"test concepts: strict {len({words[i] for i in idx['s_te'].tolist()})}, all {len({words[i] for i in idx['all_te'].tolist()})}")
ar = A["ask_rank"].to(dev)
for k in ("s_te", "all_te", "all_tr"):
    a = ar[idx[k]]
    log(f"asked rank on {k}: <=1 {(a <= 1).float().mean():.3f} <=10 {(a <= 10).float().mean():.3f} <=100 {(a <= 100).float().mean():.3f}")


G = torch.load(f"{R}/runs/g1/G_strict.pt", map_location="cpu")
te, tr = idx["s_te"], idx["s_tr"][:1000]
gs = ar[te] <= 10
print(f"test strict n={len(te)}, asked-guessable (rank<=10) n={int(gs.sum())}")
print("layer | train top10 | test top10 | test top10 centred (no bias) | test top10 on guessable | J++ on guessable | t_w itself top10")
for l in LAYERS:
    X = A["X"][l].to(dev).float()
    m = G[l]
    W = m["W"].to(dev).float() * m["scale"]
    f = lambda x: (x - m["mx"].to(dev)) @ W  # noqa: E731
    my = m["my"].to(dev)
    a_tr = agree(topk(f(X[tr]) + my), tid[tr]).float().mean().item()
    h = agree(topk(f(X[te]) + my), tid[te])
    hc = agree(topk(f(X[te])), tid[te]).float().mean().item()
    hj = agree(topk(X[te] @ JPP[l].T), tid[te])
    tt = agree(topk(T[te]), tid[te]).float().mean().item()
    print(f"L{l} | {a_tr:.3f} | {h.float().mean():.3f} | {hc:.3f} | {h[gs].float().mean():.3f} | {hj[gs].float().mean():.3f} | {tt:.3f}", flush=True)
    pn = f(X[te]).norm(dim=-1).mean().item()
    print(f"    mean |pred - bias| {pn:.3f}  |bias| {my.norm():.3f}  mean |t - bias| {(T[te] - my).norm(dim=-1).mean():.3f}")
