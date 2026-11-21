"""al3 (c): unit(J++ h) + lam * unit(M_c h + b); M_c = learned hidden-state -> hidden-word map (not a Jacobian),
al2 noise-aware ridge recipe; al2 ALL + al3 nameable-concept sentences (map c). Fitted by scripts/al3/fit.py; candidates only LOAD the map (.hf27/al3/M_c.pt).
lam chosen on the FIT split over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller), exactly as al2_T."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al2_lib import ELens, choose_lam, fit_line, load_maps  # noqa: E402


def build(ctx):
    maps = load_maps("/workspace/o-lens-improvement/.hf27/al3/M_c.pt", ctx.device)
    lam = choose_lam(ctx, maps, "M mix")
    fit_line("M alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
