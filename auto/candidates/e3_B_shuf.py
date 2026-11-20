"""e3 (B_shuf): J++ + lam * shuffled-target control of map B (same ridge strength, targets permuted).
lam chosen inside build() on the FIT split for the MAIN B map exactly as e3_B (so it equals B's lam)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_lib import ELens, choose_lam, fit_line, load_maps, map_path  # noqa: E402


def build(ctx):
    lam = choose_lam(ctx, load_maps(map_path(ctx), ctx.device))
    maps = load_maps(map_path(ctx, "shuf"), ctx.device)
    lens = ELens(ctx.JPP, maps, "mix", lam)
    fit_line("B_shuf", ctx, lens)
    return lens
