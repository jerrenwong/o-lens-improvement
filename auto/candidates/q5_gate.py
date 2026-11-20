"""q5: plain J++, but at L24/L28 remove the FIT-association mean readout direction u_L, only for rows whose
readout has cosine > 0.8 with u_L (gated junk-direction removal)."""

import torch

GATE_LAYERS = (24, 28)
THR = 0.8


class Lens:
    def __init__(self, M, U):
        self.M, self.U = M, U

    def transport(self, h, layer, resid_all=None):
        v = h @ self.M[layer].T
        if layer in self.U:
            u = self.U[layer].to(v.dtype)
            c = torch.nn.functional.cosine_similarity(v, u[None], dim=-1)
            proj = (v @ u)[:, None] * u[None]
            v = torch.where((c > THR)[:, None], v - proj, v)
        return v


def build(ctx):
    rows = [i for i, s in enumerate(ctx.fit.eval_slugs) if s == "association"]
    U = {}
    for L in GATE_LAYERS:
        X = ctx.fit.resid[L][rows] @ ctx.JPP[L].T
        m = X.mean(0)
        U[L] = m / m.norm()
    return Lens(ctx.JPP, U)
