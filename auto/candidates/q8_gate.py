"""q8: J++ with gated junk-direction removal (model-agnostic version of q5_gate).
At the last two readout layers, u_L = unit mean of the J++ readout over the FIT association rows (activations and
the task name only; no concept labels). A row's readout v is replaced by v - (v.u_L) u_L only if cos(v, u_L) > 0.8.
Env Q8_MODE: "gate" (default) | "rand" (control: same gate, remove a random direction with the same coefficient)
             | "typo" (control: prototype from FIT typo rows instead of association)."""
import os

import torch

THR = 0.8
MODE = os.environ.get("Q8_MODE", "gate")


class Lens:
    def __init__(self, M, U, R):
        self.M, self.U, self.R = M, U, R

    def transport(self, h, layer, resid_all=None):
        v = h @ self.M[layer].T
        if layer in self.U:
            u = self.U[layer].to(v.dtype)
            gate = torch.nn.functional.cosine_similarity(v, u[None], dim=-1) > THR
            d = self.R[layer].to(v.dtype) if MODE == "rand" else u
            v = torch.where(gate[:, None], v - (v @ u)[:, None] * d[None], v)
        return v


def build(ctx):
    task = "typo" if MODE == "typo" else "association"
    rows = [i for i, s in enumerate(ctx.fit.eval_slugs) if s == task]
    U, R = {}, {}
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in list(ctx.readout_layers)[-2:]:
        m = (ctx.fit.resid[L][rows] @ ctx.JPP[L].T).mean(0)
        U[L] = m / m.norm()
        r = torch.randn(m.shape, generator=g).to(m.device)
        r = r - (r @ U[L]) * U[L]
        R[L] = r / r.norm()
    print(f"q8 mode={MODE} layers={sorted(U)} fit rows={len(rows)}")
    return Lens(ctx.JPP, U, R)
