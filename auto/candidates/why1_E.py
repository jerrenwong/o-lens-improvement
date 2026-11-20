"""why1 (analysis, not a proposal): J++ + 0.5 * E with E restricted by env vars.
WHY1_RANK=r   -> W_l replaced by its top-r singular directions (W after scaling; bias my and centring mx kept).
WHY1_LAYERS=a,b -> E added only at these readout layers; plain J++ (h @ JPP.T) elsewhere."""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import ELens, fit_line, load_maps, map_path  # noqa: E402

LAM = 0.5


class Lens(ELens):
    def __init__(self, JPP, maps, layers):
        super().__init__(JPP, maps, "mix", LAM)
        self.layers = layers

    def transport(self, h, layer, resid_all=None):
        if self.layers is not None and layer not in self.layers:
            return h @ self.JPP[layer].T
        return super().transport(h, layer, resid_all)


def build(ctx):
    maps = load_maps(map_path(ctx), ctx.device)
    r = int(os.environ.get("WHY1_RANK", "0"))
    ls = os.environ.get("WHY1_LAYERS", "")
    layers = {int(x) for x in ls.split(",")} if ls else None
    if r:
        for l, (mx, W, my) in maps.items():
            U, S, Vh = torch.linalg.svd(W)
            maps[l] = (mx, (U[:, :r] * S[:r]) @ Vh[:r], my)
    print(f"[why1] rank {r or 'full'} layers {sorted(layers) if layers else 'all'}", flush=True)
    lens = Lens(ctx.JPP, maps, layers)
    fit_line("why1", ctx, lens)
    return lens
