"""rb1 controls (orchestrator follow-up). Writes maps .hf27/rb1/E_<tag>_<var>.pt (same packed format) and
.hf27/rb1/ctrl_<tag>.json (per-layer diagnostics).
(a) shufall  = copy of the stored shuffled-answer map .hf27/ask3/E_<tag>_t3_shuf.pt
    priorall = constant only: W = 0, b = the stored map's my (mean target over all train prefixes)
(b) shuf300 / shuf3000 = ridge on subset r300_0 / r3000_0 with the answers permuted WITHIN the subset, same c per layer
    as the real subset map (as the fit script's shuffled control); prior300 / prior3000 = W = 0, b = that subset's mean target.
(c) for r300_0, r3000_0, rall: per layer chosen c, effective rank of W (exp entropy of normalised singular values;
    participation ratio), effective rank of the map's centred output on the test prefixes, share of the output energy
    that is the constant b (|b|^2 / mean |(h-mx)W + b|^2), mean cos(Eh+b, b); and truncated maps <var>_k<k> (W replaced by
    its top-k singular directions, same mx, b) for k in 1 4 16 64 256.
Usage: python rb1_ctrl.py <model>
"""
import json
import sys

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
KS = [1, 4, 16, 64, 256]
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
model = get_lite_model(M)
C.setup(model)
unit = C.unit
# targets exactly as rb1_fit.py / the fit script
Y = torch.load(f"{R}/.hf27/ask3/acts_{TAG}.pt", map_location="cpu")["Y_new"]
Yk = [unit(Y[k].float()) for k in range(3)]
del Y
raw1 = [C.topk(y, 1, m=None)[:, 0] for y in Yk]
keepk = ~torch.stack([C.junk_tok[r] for r in raw1])
cnt = keepk.sum(0)
T = sum(Yk[k] * keepk[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
del Yk
S = torch.load(f"{OUT}/subsets_{TAG}.pt")
te_i = S["te_i"]
Xs = torch.load(XFILE, map_location="cpu")["X"]


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


def pack(mx, W, my, c):
    sc = W.abs().max().item() or 1.0
    return dict(mx=mx.cpu(), W=(W / sc).half().cpu(), scale=sc, my=my.cpu(), c=c)


def unpack(v):
    return v["mx"].to(dev).float(), v["W"].to(dev).float() * v["scale"], v["my"].to(dev).float()


def erank(s):
    p = s / s.sum()
    return torch.exp(-(p * p.clamp_min(1e-30).log()).sum()).item(), (s.sum() ** 2 / (s ** 2).sum()).item()


stored = torch.load(f"{R}/.hf27/ask3/E_{TAG}_t3.pt", map_location="cpu")
torch.save(torch.load(f"{R}/.hf27/ask3/E_{TAG}_t3_shuf.pt", map_location="cpu"), f"{OUT}/E_{TAG}_shufall.pt")
torch.save({l: pack(v["mx"], torch.zeros_like(v["W"].float()), v["my"], None) for l, v in stored.items()},
           f"{OUT}/E_{TAG}_priorall.pt")
real = {v: torch.load(f"{OUT}/E_{TAG}_{v}.pt", map_location="cpu") for v in ("r300_0", "r3000_0", "rall")}
out = {"shuf300": {}, "shuf3000": {}, "prior300": {}, "prior3000": {}}
out.update({f"{v}_k{k}": {} for v in real for k in KS})
diag = {v: {} for v in real}
for l in LAYERS:
    X = Xs[l].float().to(dev)
    Xt = X[te_i.to(dev)]
    for n in (300, 3000):
        sub = S["subsets"][f"r{n}_0"]
        c = real[f"r{n}_0"][l]["c"]
        r = Ridge(X[sub.to(dev)])
        g = torch.Generator().manual_seed(1000 + l)
        r.setY(T[sub][torch.randperm(len(sub), generator=g)].to(dev))
        out[f"shuf{n}"][l] = pack(r.mx, r.W(c), r.my, c)
        out[f"prior{n}"][l] = pack(r.mx, torch.zeros(X.shape[1], X.shape[1], device=dev), T[sub].mean(0), None)
        del r
    for v, mp in real.items():
        mx, W, my = unpack(mp[l])
        U, s, Vh = torch.linalg.svd(W)
        e = (Xt - mx) @ W
        so = torch.linalg.svdvals(e - e.mean(0))
        full = e + my
        b2 = my.pow(2).sum().item()
        diag[v][str(l)] = dict(c=mp[l]["c"], erank_W=erank(s)[0], pr_W=erank(s)[1], erank_out=erank(so)[0],
                               b_energy=b2 / full.pow(2).sum(-1).mean().item(),
                               cos_b=torch.nn.functional.cosine_similarity(full, my[None], dim=-1).mean().item(),
                               top1_sv_share=(s[0] ** 2 / (s ** 2).sum()).item(),
                               top16_sv_share=((s[:16] ** 2).sum() / (s ** 2).sum()).item())
        for k in KS:
            out[f"{v}_k{k}"][l] = pack(mx, (U[:, :k] * s[:k]) @ Vh[:k], my, mp[l]["c"])
        del U, Vh, W
    del X
    torch.cuda.empty_cache()
    print(f"L{l}: " + "  ".join(f"{v} c={d[str(l)]['c']:g} erW={d[str(l)]['erank_W']:.0f} erOut={d[str(l)]['erank_out']:.0f} "
                               f"bE={d[str(l)]['b_energy']:.3f} cos_b={d[str(l)]['cos_b']:.3f}" for v, d in diag.items()), flush=True)
for v, m in out.items():
    torch.save(m, f"{OUT}/E_{TAG}_{v}.pt")
mean = {v: {k: sum(d[str(l)][k] for l in LAYERS) / len(LAYERS) for k in d[str(LAYERS[0])] if k != "c"} for v, d in diag.items()}
for v, m in mean.items():
    print(f"   {v:8s} mean: " + " ".join(f"{k} {x:.3f}" for k, x in m.items()), flush=True)
json.dump(dict(diag=diag, mean=mean), open(f"{OUT}/ctrl_{TAG}.json", "w"), indent=1)
print("saved", flush=True)
