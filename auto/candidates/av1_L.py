"""av1 candidate (user's simple definition): ask-map = the plain AVERAGE of the two regression maps' unit outputs,
  m(h) = unit( unit(E_topic h + b) + unit(E_unstated h + b') )        (no weight between them)
lens = unit(J++ h) + lam * m(h), lam chosen on FIT (max filtered pass@1 + pass@10, ties -> smaller) over GRID.
Env: AO1_E (topic map), AO1_E2 (unstated-word map), AV1_ALONE=1 -> the averaged map read alone (no J++)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ao1_lib import emap, excl_mask, fmt, maps_E, ranks, score  # noqa: E402
from fm27_lib import choose, unit  # noqa: E402

GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


class Av:
    def __init__(self, ctx, E, E2, lam, alone=False):
        self.JPP, self.E, self.E2, self.lam, self.alone = ctx.JPP, E, E2, lam, alone

    def transport(self, h, layer, resid_all=None):
        m = unit(unit(emap(self.E, h, layer)) + unit(emap(self.E2, h, layer)))
        if self.alone:
            return m
        return unit(h @ self.JPP[layer].T.to(h)) + self.lam * m


def build(ctx):
    E, E2 = maps_E(ctx), maps_E(ctx, "AO1_E2")
    if os.environ.get("AV1_ALONE"):
        return Av(ctx, E, E2, 0.0, alone=True)
    mask = excl_mask(ctx); res = []
    for w in GRID:
        s = score(ctx, ranks(ctx, Av(ctx, E, E2, w), mask)); res.append((w, s)); print(fmt(f"av1 lam {w}", s), flush=True)
    lam = choose(res); print(f"[av1] chosen lam {lam}", flush=True)
    return Av(ctx, E, E2, lam)
