"""b1c disjoint layer bands: slot l returns sum over its band of unit(JPP[x] resid_all[x]);
bands 0-6, 7-10, 11-14, 15-18, 19-22, 23-26, 27-30 cover all 31 layers."""

BANDS = {4: range(0, 7), 8: range(7, 11), 12: range(11, 15), 16: range(15, 19), 20: range(19, 23),
         24: range(23, 27), 28: range(27, 31)}


def unit(v):
    return v / v.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return sum(unit(resid_all[x] @ self.M[x].T) for x in BANDS[layer])


def build(ctx):
    return Lens(ctx.JPP)
