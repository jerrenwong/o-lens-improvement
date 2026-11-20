"""b1c J++ + J-lens mixture: unit(JPP[l] h) + 1.0 * unit(J[l] h)."""

LAM = 1.0


def unit(v):
    return v / v.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)


class Lens:
    def __init__(self, JPP, J):
        self.JPP, self.J = JPP, J

    def transport(self, h, layer, resid_all=None):
        return unit(h @ self.JPP[layer].T) + LAM * unit(h @ self.J[layer].T)


def build(ctx):
    return Lens(ctx.JPP, ctx.J)
