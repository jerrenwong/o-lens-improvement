"""b3i: alternative elicitation suffix " The topic:" (fixed in advance); unit(JPP_l h) + 0.3 unit(E_l h).
build() also prints the held-out diagnostics for this suffix (see b3i_common)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3i_common import build_maps, fit_report, ELens  # noqa: E402


def build(ctx):
    maps, info = build_maps(ctx, suffix=" The topic:", diagnostics=True, shuffle_control=True)
    fit_report(ctx, maps, info)
    return ELens(ctx.JPP, maps, "mix", 0.3)
