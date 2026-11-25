"""w11b reference: the plain J-lens CENTRED at the accumulation mean activation, v = J (h - mu_l) (.hf27/w11/mu_2b.pt)."""
import torch

R = "/workspace/o-lens-improvement"


class L:
    def __init__(self, J, mu):
        self.J, self.mu = J, mu

    def transport(self, h, layer, resid_all=None):
        return (h - self.mu[layer].to(h)) @ self.J[layer].T.to(h)


def build(ctx):
    mu = torch.load(f"{R}/.hf27/w11/mu_2b.pt", map_location="cpu")["mu"]
    return L(ctx.J, {l: v.to(ctx.device).float() for l, v in mu.items()})
