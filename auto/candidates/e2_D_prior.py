"""e2 (D_prior), model-agnostic: J++ + lam * prior-only control (bias = mean target of the main map).
lam chosen inside build() on the model's own FIT split for the MAIN map exactly as e2_A (so it equals A's lam)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps, map_path  # noqa: E402


def build(ctx):
    main = load_maps(map_path(ctx), ctx.device)
    scores = {lam: fit_line(f"A mix {lam}", ctx, ELens(ctx.JPP, main, "mix", lam)) for lam in LAMS}
    lam = max(LAMS, key=lambda x: (round(scores[x], 6), -x))
    print(f"[e1 fit] chosen lam (main map) {lam}", flush=True)
    lens = ELens(ctx.JPP, main, "prior", lam)
    fit_line("D_prior", ctx, lens)
    return lens
