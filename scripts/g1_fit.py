"""g1 step 4: ridge maps G_l: last-token residual of a generated sentence -> t_w (hidden concept target, final basis).

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

maps = {v: {} for v in ("strict", "all", "shuf")}
gperm = torch.randperm(len(idx["s_tr"]), generator=torch.Generator().manual_seed(1)).to(dev)
names = ["G_strict", "J++", "E", "G_shuf", "G_all"]
anyhit = {n: torch.zeros(len(idx["s_te"]), dtype=torch.bool, device=dev) for n in names}
anyall = {n: torch.zeros(len(idx["all_te"]), dtype=torch.bool, device=dev) for n in ("G_all", "J++", "E")}
rows = []
te, tea = idx["s_te"], idx["all_te"]
prior_top = None
for l in LAYERS:
    X = A["X"][l].to(dev).float()
    res, resA, cs = {}, {}, {}
    for v, tr, va in (("strict", "s_tr", "s_va"), ("all", "all_tr", "all_va")):
        r = Ridge(X[idx[tr]])
        r.setY(T[idx[tr]])
        sc = [agree(topk(r.predict(X[idx[va]], r.W(c))), tid[idx[va]]).float().mean().item() for c in GRID]
        b = max(range(len(GRID)), key=lambda i: (sc[i], i))
        c = GRID[b]
        W = r.W(c)
        maps[v][l] = pack(r, W, c)
        cs[v] = (c, sc)
        tp = topk(r.predict(X[te], W))
        res["G_" + v] = tp
        if v == "all":
            resA["G_all"] = topk(r.predict(X[tea], W))
        if v == "strict":
            r.setY(T[idx["s_tr"]][gperm])
            Ws = r.W(c)
            maps["shuf"][l] = pack(r, Ws, c)
            res["G_shuf"] = topk(r.predict(X[te], Ws))
            if l == LAYERS[0]:
                prior_top = topk(r.my[None].float()).expand(len(te), -1)
        del r
        torch.cuda.empty_cache()
    res["J++"] = topk(X[te] @ JPP[l].T)
    resA["J++"] = topk(X[tea] @ JPP[l].T)
    ea = EA[l]
    WA = ea["W"].to(dev).float() * ea["scale"]
    Ef = lambda x: (x - ea["mx"].to(dev)) @ WA + ea["my"].to(dev)  # noqa: E731
    res["E"] = topk(Ef(X[te]))
    resA["E"] = topk(Ef(X[tea]))
    row = {}
    for n in names:
        h10 = agree(res[n], tid[te])
        anyhit[n] |= h10
        row[n] = (h10.float().mean().item(), agree(res[n][:, :1], tid[te]).float().mean().item())
    rowA = {}
    for n in ("G_all", "J++", "E"):
        h10 = agree(resA[n], tid[tea])
        anyall[n] |= h10
        rowA[n] = (h10.float().mean().item(), agree(resA[n][:, :1], tid[tea]).float().mean().item())
    rows.append((l, cs, row, rowA))
    log(f"L{l:2d} c strict={cs['strict'][0]:g} all={cs['all'][0]:g}; val@10 by c (strict): " +
        " ".join(f"{a:g}:{b:.3f}" for a, b in zip(GRID, cs["strict"][1])) +
        " | (all): " + " ".join(f"{a:g}:{b:.3f}" for a, b in zip(GRID, cs["all"][1])))
    if l in (12, 24):
        for i in range(0, len(te), max(1, len(te) // 8)):
            j = te[i].item()
            log(f"   ex L{l} [{words[j]}] {A['sents'][j][:90]!r} -> G: " +
                ", ".join(strs[t].strip() for t in res["G_strict"][i, :6].tolist()) +
                " | J++: " + ", ".join(strs[t].strip() for t in res["J++"][i, :4].tolist()))
    del X
    torch.cuda.empty_cache()

print(f"\n== held-out-concept test sentences, STRICT concepts (n={len(te)}): top-10 / top-1 (word-level)")
print("layer | c_strict | " + " | ".join(names))
for l, cs, row, _ in rows:
    print(f"L{l:2d} | {cs['strict'][0]:g} | " + " | ".join(f"{row[n][0]:.3f} / {row[n][1]:.3f}" for n in names))
print("any layer top-10 | | " + " | ".join(f"{anyhit[n].float().mean():.3f}" for n in names))
print(f"prior only (unembed of G_strict bias, L4) top-10: {agree(prior_top, tid[te]).float().mean():.3f}")
print(f"\n== all test concepts (n={len(tea)}): top-10 / top-1")
print("layer | c_all | G_all | J++ | E")
for l, cs, _, rowA in rows:
    print(f"L{l:2d} | {cs['all'][0]:g} | " + " | ".join(f"{rowA[n][0]:.3f} / {rowA[n][1]:.3f}" for n in ("G_all", "J++", "E")))
print("any layer top-10 | | " + " | ".join(f"{anyall[n].float().mean():.3f}" for n in ("G_all", "J++", "E")))
for v in maps:
    src = f"{R}/.hf27/g1/G_{v}.pt"
    torch.save(maps[v], src)
    with open(src, "rb") as a_, open(f"{R}/runs/g1/G_{v}.pt", "wb") as b_:
        while True:
            chunk = a_.read(1 << 24)
            if not chunk:
                break
            b_.write(chunk)
log("saved")
