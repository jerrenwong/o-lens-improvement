"""Baseline: pre-fitted J++ (h @ JPP[layer].T)."""


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def build(ctx):
    return Lens(ctx.JPP)
