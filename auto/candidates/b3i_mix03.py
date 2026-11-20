"""b3i: unit(JPP_l h) + 0.3 unit(E_l h) (see b3i_common)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3i_common import build_maps, ELens  # noqa: E402


def build(ctx):
    maps, info = build_maps(ctx, diagnostics=True)
    return ELens(ctx.JPP, maps, 'mix', 0.3)
