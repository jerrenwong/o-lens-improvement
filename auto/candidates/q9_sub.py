"""q9: Jacobian lens with a concept-subspace output filter (label-free).
transport(h) = P (M h), P = projector on the top-K principal directions of the model's own elicited-answer states
(scripts/q9_subspace.py). Env: Q9_K (int), Q9_MODE = proj | comp (remove the subspace) | rand (random subspace of the
same size) | soft (shrink each PC direction by ev/(ev+ev[K])), Q9_BASE = jpp | j."""
import os

import torch

K = int(os.environ.get("Q9_K", "256")); MODE = os.environ.get("Q9_MODE", "proj"); BASE = os.environ.get("Q9_BASE", "jpp")


class Lens:
    def __init__(self, M, B, w):
        self.M, self.B, self.w = M, B, w

    def transport(self, h, layer, resid_all=None):
        v = h @ self.M[layer].T
        if MODE == "soft":
            return ((v @ self.B) * self.w) @ self.B.T
        p = (v @ self.B) @ self.B.T
        return v - p if MODE == "comp" else p


def build(ctx):
    d = torch.load("/root/e1/Ypcs_9b.pt", map_location="cpu")
    V, ev = d["V"].to(ctx.device).float(), d["evals"].to(ctx.device).float()
    w = None
    if MODE == "rand":
        g = torch.Generator().manual_seed(0)
        B = torch.linalg.qr(torch.randn(V.shape[0], K, generator=g))[0].to(ctx.device)
    elif MODE == "soft":
        B, w = V, ev / (ev + ev[K])
    else:
        B = V[:, :K]
    print(f"q9 base={BASE} mode={MODE} K={K}")
    return Lens(ctx.JPP if BASE == "jpp" else ctx.J, B, w)
