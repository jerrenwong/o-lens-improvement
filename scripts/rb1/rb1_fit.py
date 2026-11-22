"""rb1 fit: ridge ask-map refit on subsets of the fit script's TRAINING prefixes (stability / data-size check).
Targets, junk rule, elicited word, valid mask, train/val/test prefixes, Ridge class, c grid and validation rule are
those of scripts/ask3/ask3_fit9.py (9B, 27B) / scripts/m2b/m2b_fit.py (2B) (same code path as scripts/st1/st1_fit.py,
lite model = final norm + LM head only, bitwise the same unembed).
Subsets (seeded, saved to .hf27/rb1/subsets_<tag>.pt): n in {300, 1000, 3000}: 3 disjoint draws; n = 10000: 3
independently seeded draws (3 x 10000 > train size, disjoint impossible); two disjoint halves (hA, hB); "rall" = all
train prefixes (must reproduce the stored map .hf27/ask3/E_<tag>_t3.pt; checked and printed).
Label-free diagnostics on the fit script's TEST prefixes (1000..1999): st1_common.diag (nw1 nw10 fn1 | a1u a1 a10 vs the
model's elicited answer word), test MSE to the target, and the validation a10 used for selecting c.
Usage: python rb1_fit.py <model>
"""
import json
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/scripts/st1")
sys.path.insert(0, f"{R}/scripts/perf1")
from lite_model import get_lite_model  # noqa: E402
import st1_common as C  # noqa: E402

M = sys.argv[1]
TAG = "27b" if "27B" in M else ("2b" if "2B" in M else "9b")
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
XFILE = {"9b": f"{R}/.hf27/ask2/acts_9b.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt", "2b": "/root/e1/acts_2b.pt"}[TAG]
OUT = f"{R}/.hf27/rb1"
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
NV, NT = 1000, 1000
SIZES = [300, 1000, 3000, 10000]
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"

model = get_lite_model(M)
C.setup(model)
N = C.N
unit = C.unit

# ---------------------------------------------------------------- targets, junk rule, labels (as the fit script)
Y = torch.load(f"{R}/.hf27/ask3/acts_{TAG}.pt", map_location="cpu")["Y_new"]
Yk = [unit(Y[k].float()) for k in range(3)]
del Y
raw1 = [C.topk(y, 1, m=None)[:, 0] for y in Yk]
junk = torch.stack([C.junk_tok[r] for r in raw1])
keepk = ~junk
cnt = keepk.sum(0)
T = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
valid = cnt > 0
word = C.topk(T, 10)[:, 0]
del Yk
tr_m = valid.clone(); tr_m[:NV + NT] = False
va_m = valid.clone(); va_m[NV:] = False
te_m = valid.clone(); te_m[:NV] = False; te_m[NV + NT:] = False
tr_i, va_i, te_i = (torch.nonzero(m)[:, 0].cpu() for m in (tr_m, va_m, te_m))
print(f"[rb1 fit {TAG}] N {N}; train {len(tr_i)}, val {len(va_i)}, test {len(te_i)}", flush=True)

# ---------------------------------------------------------------- subsets
g = torch.Generator().manual_seed(1122)
perm = tr_i[torch.randperm(len(tr_i), generator=g)]
subsets = {}
for n in SIZES:
    if 3 * n <= len(tr_i):
        for k in range(3):
            subsets[f"r{n}_{k}"] = perm[k * n:(k + 1) * n]
    else:
        for k in range(3):
            gk = torch.Generator().manual_seed(5000 + k)
            subsets[f"r{n}_{k}"] = tr_i[torch.randperm(len(tr_i), generator=gk)[:n]]
gh = torch.Generator().manual_seed(77)
ph = tr_i[torch.randperm(len(tr_i), generator=gh)]
h = len(tr_i) // 2
subsets["hA"], subsets["hB"] = ph[:h], ph[h:2 * h]
subsets["rall"] = tr_i
subsets = {k: v.sort().values for k, v in subsets.items()}
torch.save(dict(subsets=subsets, tr_i=tr_i, va_i=va_i, te_i=te_i), f"{OUT}/subsets_{TAG}.pt")
for k, v in subsets.items():
    print(f"   subset {k}: n={len(v)}", flush=True)


class Ridge:  # identical to the fit script
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


Xs = torch.load(XFILE, map_location="cpu")["X"]
stored = torch.load(f"{R}/.hf27/ask3/E_{TAG}_t3.pt", map_location="cpu")
maps = {k: {} for k in subsets}
res = {k: {} for k in subsets}
repro = {}
wv, wt = word[va_i.to(dev)], word[te_i.to(dev)]
Tte = T[te_i].to(dev)
for l in LAYERS:
    t0 = time.time()
    X = Xs[l].float().to(dev)
    Xv, Xt = X[va_i.to(dev)], X[te_i.to(dev)]
    for nm, sub in subsets.items():
        r = Ridge(X[sub.to(dev)])
        r.setY(T[sub].to(dev))
        sc = [C.agree(C.topk(r.predict(Xv, r.W(c_))), wv).float().mean().item() for c_ in GRID]
        b = max(range(len(GRID)), key=lambda i: (sc[i], i))
        W = r.W(GRID[b])
        maps[nm][l] = pack(r, W, GRID[b])
        pred = r.predict(Xt, W)
        res[nm][l] = dict(c=GRID[b], val10=sc[b], mse=(pred - Tte).pow(2).mean().item(), **C.diag(pred, wt))
        if nm == "rall":
            s = stored[l]
            repro[l] = dict(c_equal=s["c"] == GRID[b], W_bitwise=bool(torch.equal(s["W"], maps[nm][l]["W"])),
                            W_maxdiff=(s["W"].float() * s["scale"] - maps[nm][l]["W"].float() * maps[nm][l]["scale"]).abs().max().item(),
                            scale_equal=s["scale"] == maps[nm][l]["scale"], mx_equal=bool(torch.equal(s["mx"], maps[nm][l]["mx"])),
                            my_equal=bool(torch.equal(s["my"], maps[nm][l]["my"])))
            print(f"   L{l} reproduce stored map: {repro[l]}", flush=True)
        del r, W
    del X
    torch.cuda.empty_cache()
    print(f"L{l} ({time.time() - t0:.0f}s): " + "  ".join(f"{nm} c={res[nm][l]['c']:g} a10={res[nm][l]['a10']:.3f}"
                                                     for nm in res), flush=True)

for nm, m in maps.items():
    torch.save(m, f"{OUT}/E_{TAG}_{nm}.pt")
KEYS = ("val10", "mse", "nw1", "nw10", "fn1", "a1u", "a1", "a10")
mean = {nm: {k: sum(res[nm][l][k] for l in LAYERS) / len(LAYERS) for k in KEYS} for nm in res}
print("\n== label-free (test prefixes; mean over layers): val10 mse | nw1 nw10 fn1 | a1u a1 a10")
for nm, m in mean.items():
    print(f"   {nm:9s} n={len(subsets[nm]):5d} " + " ".join(f"{k} {m[k]:.4f}" for k in KEYS), flush=True)
json.dump(dict(n={k: len(v) for k, v in subsets.items()}, mean=mean,
               per_layer={k: {str(l): v for l, v in x.items()} for k, x in res.items()},
               repro={str(l): v for l, v in repro.items()}), open(f"{OUT}/fit_{TAG}.json", "w"), indent=1)
print("saved", flush=True)
