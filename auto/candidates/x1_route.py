"""x1: CONTEXT-ROUTED READOUT of the J++ experts (Qwen3.5-9B, refit r2 artefacts).

J++ fits one expert Jacobian per k-means cluster (router: unit-normalise -> centre -> PCA-64 -> nearest of 8
centroids), then collapses the 8 experts with fixed signed weights into one matrix per layer. Here the router is
used again at READOUT time:
  X1_MODE=hard   v = E_c(h) h, c = router cluster of h
  X1_MODE=soft   v = sum_k p_k(h) E_k h, p = softmax(-d_k^2 / tau_l); tau_l per layer so that the mean max-prob
                 on FIT activations = X1_PMAX (0.5 or 0.8)
  X1_MODE=mix    v = unit(J_fixed h) + lam * unit(E_c(h) h), lam from {0.3, 1.0} chosen on FIT pass@10
                 (or X1_LAM)
  X1_MODE=pooled v = J_pooled h (count-weighted mean of experts, the file pooled_<cond>_full.pt)
  X1_MODE=fixed  v = J_fixed h (jpp_<cond>_full.pt)
X1_COND=A (raw WikiText, skip 16) or C (natural format, skip 1). Build prints FIT pass@10 / pass@1 for every
variant (labelled FIT split only).
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_common import report, unit  # noqa: E402

R2 = "/workspace/o-lens-improvement/runs/r2_9b"
LAMS = (0.3, 1.0)


def proj(router, l, h):
    m, P, C = router[l]
    x = h.float() / h.float().norm(dim=-1, keepdim=True)
    return (x - m) @ P, C


def dist2(router, l, h):
    z, C = proj(router, l, h)
    return torch.cdist(z, C).pow(2)  # [N, 8]


def calibrate_tau(d2, target):
    """tau such that mean_i max_k softmax(-d2/tau) == target (bisection in log tau)."""
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        pm = torch.softmax(-d2 / torch.exp(torch.tensor(mid)), -1).max(-1).values.mean().item()
        if pm > target:  # too peaked -> larger tau
            lo = mid
        else:
            hi = mid
    return float(torch.exp(torch.tensor((lo + hi) / 2)))


class RouteLens:
    def __init__(self, mode, router, experts, fixed, pooled, tau=None, lam=None):
        self.mode, self.router, self.E, self.fixed, self.pooled = mode, router, experts, fixed, pooled
        self.tau, self.lam = tau, lam

    def expert_out(self, h, l):  # [N, 8, d]
        E = self.E[l]
        return torch.stack([h @ E[k].float().T for k in range(E.shape[0])], 1)

    def routed(self, h, l):
        c = dist2(self.router, l, h).argmin(-1)
        out = self.expert_out(h, l)
        return out[torch.arange(h.shape[0], device=h.device), c]

    def transport(self, h, layer, resid_all=None):
        l = layer
        if self.mode == "fixed":
            return h @ self.fixed[l].T
        if self.mode == "pooled":
            return h @ self.pooled[l].T
        if self.mode == "hard":
            return self.routed(h, l)
        if self.mode == "soft":
            p = torch.softmax(-dist2(self.router, l, h) / self.tau[l], -1)
            return (p[:, :, None] * self.expert_out(h, l)).sum(1)
        if self.mode == "mix":
            return unit(h @ self.fixed[l].T) + self.lam * unit(self.routed(h, l))
        raise ValueError(self.mode)


def load_all(ctx, cond):
    from workspace_lens.utils import load_lens_file
    dev = ctx.device
    rs = torch.load(f"{R2}/{cond}/router_K8.pt", map_location="cpu", weights_only=True)
    router = {int(l): (rs["pca_means"][l].to(dev).float(), rs["pca_components"][l].to(dev).float(),
                       rs["centroids"][l].to(dev).float()) for l in rs["centroids"]}
    ex = torch.load(f"{R2}/{cond}/{cond}_full/experts_K8.pt", map_location="cpu", weights_only=True)
    experts = {int(l): v.to(dev) for l, v in ex["experts"].items()}  # fp16, cast per expert
    del ex

    def lens(p):
        L = load_lens_file(p, hf_model_name="Qwen/Qwen3.5-9B")
        return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}
    fixed = lens(f"{R2}/{cond}/{cond}_full/jpp_{cond}_full.pt")
    pooled = lens(f"{R2}/{cond}/{cond}_full/pooled_{cond}_full.pt")
    return router, experts, fixed, pooled


def fit_vecs(ctx, lens):
    return {l: lens.transport(ctx.fit.resid[l], l) for l in ctx.readout_layers}


def build(ctx):
    mode = os.environ.get("X1_MODE", "hard")
    cond = os.environ.get("X1_COND", "A")
    pmax = float(os.environ.get("X1_PMAX", "0.5"))
    router, experts, fixed, pooled = load_all(ctx, cond)
    taus = {}
    for t in (0.5, 0.8):
        taus[t] = {l: calibrate_tau(dist2(router, l, ctx.fit.resid[l]), t) for l in ctx.readout_layers}
        print(f"[x1] cond {cond} tau for mean max-prob {t} on FIT: "
              + " ".join(f"L{l} {v:.4g}" for l, v in taus[t].items()), flush=True)
    for t in (0.5, 0.8):
        if t not in taus:
            continue
    lens = lambda m, **kw: RouteLens(m, router, experts, fixed, pooled, **kw)  # noqa: E731
    res = {}
    res["fixed"] = report(ctx, f"{cond} fixed", fit_vecs(ctx, lens("fixed")))
    res["pooled"] = report(ctx, f"{cond} pooled", fit_vecs(ctx, lens("pooled")))
    res["hard"] = report(ctx, f"{cond} hard", fit_vecs(ctx, lens("hard")))
    for t in (0.5, 0.8):
        res[f"soft{t}"] = report(ctx, f"{cond} soft pmax {t}", fit_vecs(ctx, lens("soft", tau=taus[t])))
    for lam in LAMS:
        res[f"mix{lam}"] = report(ctx, f"{cond} mix lam {lam}", fit_vecs(ctx, lens("mix", lam=lam)))
    if mode == "soft":
        return lens("soft", tau=taus[pmax] if pmax in taus else
                    {l: calibrate_tau(dist2(router, l, ctx.fit.resid[l]), pmax) for l in ctx.readout_layers})
    if mode == "mix":
        lam = float(os.environ["X1_LAM"]) if "X1_LAM" in os.environ else max(LAMS, key=lambda x: res[f"mix{x}"])
        print(f"[x1] mix lam chosen {lam}", flush=True)
        return lens("mix", lam=lam)
    return lens(mode)
