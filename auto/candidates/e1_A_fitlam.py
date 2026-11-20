"""e1 (A): J++ + lam * elicitation-distilled probe E (learned, not a Jacobian; see e1_lib).
lam chosen inside build() by FIT-split pass@10 over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller lam)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_lib import LAMS, MAIN, ELens, fit_line, load_maps  # noqa: E402


def build(ctx):
    maps = load_maps(MAIN, ctx.device)
    base = fit_line("jpp", ctx, ELens(ctx.JPP, maps, "mix", 0.0))
    scores = {lam: fit_line(f"mix {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    fit_line("E alone", ctx, ELens(ctx.JPP, maps, "alone"))
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[e1 fit] chosen lam {best} (fit {scores[best]:.4f} vs J++ {base:.4f})", flush=True)
    return ELens(ctx.JPP, maps, "mix", best)
