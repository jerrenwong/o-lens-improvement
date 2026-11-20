"""b3i CONTROL: as b3i_mix03 (unit(JPP_l h) + 0.3 unit(E_l h)), but each E_l is fitted on shuffled targets
(sentence -> another sentence's elicitation state), same lambda as the real map (chosen on real targets).
Tests whether the mix03 gain comes from text-specific content or from a generic topic-word prior."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b3i_lib  # noqa: E402
from b3i_lib import ELens, Ridge, build_maps  # noqa: E402


class ShufRidge(Ridge):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        g = torch.Generator().manual_seed(1)
        perm = torch.randperm(Y.shape[0], generator=g).to(Y.device)
        self.shuf = Ridge(X, Y[perm])

    def weight(self, c):
        return self.shuf.weight(c)


def build(ctx):
    b3i_lib.Ridge = ShufRidge
    maps, info = build_maps(ctx)
    return ELens(ctx.JPP, maps, "mix", 0.3)
