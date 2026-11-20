"""b3i: elicit-distilled ridge map E_l h alone (see b3i_common)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3i_common import build_maps, ELens  # noqa: E402


def build(ctx):
    maps, info = build_maps(ctx)
    return ELens(ctx.JPP, maps, 'alone')
