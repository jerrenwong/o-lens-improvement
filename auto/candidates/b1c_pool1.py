"""b1c neighbour-layer pooling: slot l returns sum_k w_k * unit(JPP[l+k] resid_all[l+k]), window +-1, uniform."""

W = {-1: 1.0, 0: 1.0, 1: 1.0}


def unit(v):
    return v / v.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        v = 0
        for k, w in W.items():
            l = layer + k
            if l in self.M and l in resid_all:
                v = v + w * unit(resid_all[l] @ self.M[l].T)
        return v


def build(ctx):
    return Lens(ctx.JPP)
