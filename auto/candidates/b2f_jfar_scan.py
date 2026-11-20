"""b2f idea 11 scan: far-target Jacobian expert by random-probe VJPs (64-token WikiText windows, cotangent at
the last 8 positions, sources 16..48, i.e. 8-47 tokens before the targets). Prints split-half reliability and
FIT-split scans (full and low-rank); also re-runs the idea-9 fit with fixed centring (diagnostic only).
Returns unit(J++) + 0.5 * unit(J_far)."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("b2f_common", Path(__file__).with_name("b2f_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

LAMS = [0.25, 0.5, 1.0]


class Lin:
    def __init__(self, JPP, M, lam):
        self.JPP, self.M, self.lam = JPP, M, lam

    def transport(self, h, layer, resid_all=None):
        x = h @ self.M[layer].T
        return C.unit(h @ self.JPP[layer].T) + self.lam * C.unit(x)


def build(ctx):
    L, R = ctx.readout_layers, ctx.fit.resid
    jpp = {l: R[l] @ ctx.JPP[l].T for l in L}
    C.report(ctx, "J++", jpp)
    # idea 9 re-check with fixed centring (diagnostic)
    A, mu = C.fit_fbag(ctx, targets=("all",))
    gamma = C.final_gamma(ctx.model)
    lens = C.LinLens(ctx.JPP, A["all"], mu, gamma, None)
    ex = {l: lens.expert(R[l], l) for l in L}
    C.report(ctx, "A_all alone (v2)", ex)
    C.report(ctx, "J++ + 0.25 A_all (v2)", {l: C.unit(jpp[l]) + 0.25 * C.unit(ex[l]) for l in L})
    del A
    torch.cuda.empty_cache()
    # idea 11
    Jf, Ma, Mb = C.fit_jfar(ctx, time_cap=380.0)
    C.split_half(ctx, Ma, Mb, R, "jfar")
    C.report(ctx, "J (ctx.J) alone", {l: R[l] @ ctx.J[l].T for l in L})
    for r in (None, 1024, 256):
        M = Jf if r is None else {l: C.lowrank(Jf[l], r) for l in L}
        ex = {l: R[l] @ M[l].T for l in L}
        tag = "full" if r is None else f"r{r}"
        C.report(ctx, f"J_far {tag} alone", ex)
        for lam in LAMS:
            C.report(ctx, f"J++ + {lam} J_far {tag}", {l: C.unit(jpp[l]) + lam * C.unit(ex[l]) for l in L})
    return Lin(ctx.JPP, Jf, 0.5)
