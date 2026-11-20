"""b2f idea 9 scan: future-bag ridge lens (H=32, IDF-weighted, next token excluded), targets 'all' and
'novel' (future tokens not already in the window prefix). Prints label-free sanity and FIT-split scans;
returns unit(J++) + 0.5 * unit(A_all)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2f_common", Path(__file__).with_name("b2f_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

LAMS = [0.25, 0.5, 1.0, 2.0]


def build(ctx):
    L, R = ctx.readout_layers, ctx.fit.resid
    A, mu = C.fit_fbag(ctx)
    gamma = C.final_gamma(ctx.model)
    jpp = {l: R[l] @ ctx.JPP[l].T for l in L}
    C.report(ctx, "J++", jpp)
    C.report(ctx, "J", {l: R[l] @ ctx.J[l].T for l in L})
    for tg, M in A.items():
        lens = C.LinLens(ctx.JPP, M, mu, gamma, None)
        ex = {l: lens.expert(R[l], l) for l in L}
        C.report(ctx, f"A_{tg} alone", ex)
        for lam in LAMS:
            C.report(ctx, f"J++ + {lam} A_{tg}", {l: C.unit(jpp[l]) + lam * C.unit(ex[l]) for l in L})
    return C.LinLens(ctx.JPP, A["all"], mu, gamma, 0.5)
