"""m27 copy of e2_A.py (only change: imports m27_lib). e2 (A), model-agnostic e1_A_fitlam: J++ + lam * elicitation-distilled probe E (learned, not a Jacobian).
lam chosen inside build() by the model's own FIT-split pass@10 over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m27_lib import LAMS, ELens, fit_line, load_maps, map_path  # noqa: E402


def build(ctx):
    path = map_path(ctx)
    print(f"[e2] maps {path}", flush=True)
    maps = load_maps(path, ctx.device)
    base = fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    scores = {lam: fit_line(f"mix {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    fit_line("E alone", ctx, ELens(ctx.JPP, maps, "alone"))
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[e1 fit] chosen lam {best} (fit {scores[best]:.4f} vs J++ {base:.4f})", flush=True)
    return ELens(ctx.JPP, maps, "mix", best)
