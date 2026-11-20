"""e3 (B): J++ + lam * content-word elicitation map B in the final basis (no prior/bias).
lam chosen inside build() by the model's own FIT-split pass@10 over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_lib import ELens, choose_lam, fit_line, load_maps, map_path  # noqa: E402


def build(ctx):
    path = map_path(ctx)
    print(f"[e3] maps {path}", flush=True)
    maps = load_maps(path, ctx.device)
    lam = choose_lam(ctx, maps)
    fit_line("B alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
