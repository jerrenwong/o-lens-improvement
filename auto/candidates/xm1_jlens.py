"""xm1 reference: the J-lens (K=1, plain rules) fitted for the model, h @ J.T (ctx.J)."""


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def build(ctx):
    return Lens(ctx.J)
