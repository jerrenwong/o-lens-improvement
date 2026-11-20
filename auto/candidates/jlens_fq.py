"""Baseline: J-lens after removing always-on J-space dictionary features (fired on >= 10% of
training tokens), as scripts/freq_mask.py's jspace_b1.fq0.1: x = h @ J.T, drop the frequent
features' decoded contribution (mean and error kept)."""

import torch

from olens.bsf_io import BlockDict

TAU = 0.1


class Lens:
    def __init__(self, J, dicts, keep):
        self.J, self.dicts, self.keep = J, dicts, keep

    def transport(self, h, layer, resid_all=None):
        x = h @ self.J[layer].T
        D, keep = self.dicts[layer], self.keep[layer]
        z = D.encode(x)
        dropped = D.decode(z) - D.decode(z * keep[None, :, None])
        return x - dropped


def build(ctx):
    dicts, keep = {}, {}
    for layer in ctx.readout_layers:
        path = ctx.dict_path(layer, "jspace")
        dicts[layer] = BlockDict.load(path, ctx.device)
        st = torch.load(path[:-3] + "_stats.pt", map_location=ctx.device, weights_only=True)
        keep[layer] = st["fire_count"].float() / float(st["rows"]) < TAU
    return Lens(ctx.J, dicts, keep)
