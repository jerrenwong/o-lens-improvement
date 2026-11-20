"""m27b copy of m27_C_shuf.py (only change: imports m27b_lib -> runs/e1/E_27b_ws*.pt maps, PREREG_v3 recipe A').
m27 copy of e2_C_shuf.py (only change: imports m27_lib). e2 (C_shuf), model-agnostic: J++ + lam * shuffled-target control map.
lam chosen inside build() on the model's own FIT split for the MAIN map exactly as e2_A (so it equals A's lam)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m27b_lib import LAMS, ELens, fit_line, load_maps, map_path  # noqa: E402


def build(ctx):
    main = load_maps(map_path(ctx), ctx.device)
    scores = {lam: fit_line(f"A mix {lam}", ctx, ELens(ctx.JPP, main, "mix", lam)) for lam in LAMS}
    lam = max(LAMS, key=lambda x: (round(scores[x], 6), -x))
    print(f"[e1 fit] chosen lam (main map) {lam}", flush=True)
    maps = load_maps(map_path(ctx, "shuf"), ctx.device)
    lens = ELens(ctx.JPP, maps, "mix", lam)
    fit_line("C_shuf", ctx, lens)
    return lens
