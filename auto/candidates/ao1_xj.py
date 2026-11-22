"""ao1 candidate: XJ lens from scripts/ao1/xj_fit.py (env AO1_J = {layer: {'M', 'c'}}): v = unit(h @ M^T + c).
Weights were fitted on FIT by the authors' search over an extended expert basis (see xj_fit.py)."""
import os

import torch


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


class XJ:
    def __init__(self, ctx):
        d = torch.load(os.environ["AO1_J"], map_location="cpu", weights_only=True)
        self.M = {int(l): v["M"].to(ctx.device).float() for l, v in d.items()}
        self.c = {int(l): v["c"].to(ctx.device).float() for l, v in d.items()}

    def transport(self, h, layer, resid_all=None):
        return unit(h @ self.M[layer].T + self.c[layer])


def build(ctx):
    return XJ(ctx)
