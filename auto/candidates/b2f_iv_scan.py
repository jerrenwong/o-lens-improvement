"""b2f idea 13 scan: interventional lens. h is written at block l, position 24 of 8 WikiText contexts
(48 tokens); expert = mean change of final-block outputs at later positions vs writing the WikiText mean
activation; also a finite-difference JVP variant (add eps*unit(h)). Prints FIT-split scans for near (t >= s+1) and far (t >= s+8) reads and J++ mixtures.
Returns unit(J++) + 0.5 * unit(expert near)."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("b2f_common", Path(__file__).with_name("b2f_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

LAMS = [0.25, 0.5, 1.0]


def build(ctx):
    L, R = ctx.readout_layers, ctx.fit.resid
    jpp = {l: R[l] @ ctx.JPP[l].T for l in L}
    C.report(ctx, "J++", jpp)
    iv = C.Interv(ctx, kmin=1)
    D = {l: iv.delta_pos(R[l], l) for l in L}
    for kmin in (1, 8):
        ex = {l: D[l][:, kmin - 1:].mean(1) for l in L}
        C.report(ctx, f"IV k>={kmin} alone", ex)
        for lam in LAMS:
            C.report(ctx, f"J++ + {lam} IV k>={kmin}", {l: C.unit(jpp[l]) + lam * C.unit(ex[l]) for l in L})
    # finite-difference JVP version (add eps * unit(h)): exact context-averaged Jacobian, near and far targets
    ivl = C.Interv(ctx, kmin=1, mode="add", eps=0.1)
    Dl = {l: ivl.delta_pos(R[l], l) for l in L}
    for kmin in (1, 8):
        ex = {l: Dl[l][:, kmin - 1:].mean(1) for l in L}
        C.report(ctx, f"IVlin k>={kmin} alone", ex)
        for lam in LAMS:
            C.report(ctx, f"J++ + {lam} IVlin k>={kmin}", {l: C.unit(jpp[l]) + lam * C.unit(ex[l]) for l in L})
    # first read position only (t = s+1): the patched token's immediate successor
    C.report(ctx, "IV t=s+1 alone", {l: D[l][:, 0] for l in L})
    return C.IntervLens(ctx.JPP, iv, 0.5)
