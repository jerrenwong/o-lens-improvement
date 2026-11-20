"""b2f idea 11 via exact far-target Jacobian-vector products: add eps*unit(h) at block l, position 24 of 32
WikiText contexts (48 tokens), finite-difference change of final-block outputs at positions >= s+8, mean over
contexts and positions (= the context-averaged far Jacobian applied to h, no probe noise).
Diagnostic: the same machinery read at t = s and t >= s (J-lens analogues) vs ctx.J, to check the estimator.
Lens: unit(J++ h) + 0.25 * unit(J_far h)."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("b2f_common", Path(__file__).with_name("b2f_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

KMIN, LAM = 8, 0.25


class FarLens:
    def __init__(self, JPP, iv, lam):
        self.JPP, self.iv, self.lam = JPP, iv, lam

    def transport(self, h, layer, resid_all=None):
        x = self.iv.delta_pos(h, layer)[:, KMIN:].mean(1)
        return C.unit(h @ self.JPP[layer].T) + self.lam * C.unit(x)


def build(ctx):
    L, R = ctx.readout_layers, ctx.fit.resid
    jpp = {l: R[l] @ ctx.JPP[l].T for l in L}
    jl = {l: R[l] @ ctx.J[l].T for l in L}
    C.report(ctx, "J++", jpp)
    C.report(ctx, "J", jl)
    iv = C.Interv(ctx, K=32, mode="add", eps=0.1, read_from_s=True)
    D = {l: iv.delta_pos(R[l], l) for l in L}
    for name, sl in (("t=s", slice(0, 1)), ("t>=s", slice(0, None)), ("t>=s+1", slice(1, None)),
                     ("t>=s+8", slice(KMIN, None))):
        ex = {l: D[l][:, sl].mean(1) for l in L}
        cos = " ".join(f"L{l} {torch.nn.functional.cosine_similarity(ex[l], jl[l], dim=-1).mean().item():.2f}"
                       for l in L)
        print(f"[ivlin] {name}: cos with J h: {cos}", flush=True)
        C.report(ctx, f"IVlin {name} alone", ex)
        if name == "t>=s+8":
            for lam in (0.1, 0.25, 0.5):
                C.report(ctx, f"J++ + {lam} IVlin {name}", {l: C.unit(jpp[l]) + lam * C.unit(ex[l]) for l in L})
    return FarLens(ctx.JPP, iv, LAM)
