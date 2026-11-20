"""e4: frozen recipe A with a distilled map fitted on a subset of the training prefixes (env E4_MAP = map file).
lam chosen on FIT exactly as in e2_A."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_lib import LAMS, ELens, fit_line, load_maps  # noqa: E402


def build(ctx):
    maps = load_maps(os.environ["E4_MAP"], ctx.device)
    scores = {lam: fit_line(f"mix {lam}", ctx, ELens(ctx.JPP, maps, "mix", lam)) for lam in LAMS}
    best = max(LAMS, key=lambda lam: (round(scores[lam], 6), -lam))
    print(f"[e4] {os.environ['E4_MAP']} chosen lam {best} fit {scores[best]:.4f}", flush=True)
    return ELens(ctx.JPP, maps, "mix", best)
