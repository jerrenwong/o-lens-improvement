"""al3 (a): unit(J++ h) + lam * unit(M_a h + b); M_a = learned hidden-state -> hidden-word map (not a Jacobian),
al2 noise-aware ridge recipe; al2 ALL refit by scripts/al3/fit.py (map a; should equal al2_T's M_all). Fitted by scripts/al3/fit.py; candidates only LOAD the map (.hf27/al3/M_a.pt).
lam chosen on the FIT split over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller), exactly as al2_T."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al2_lib import ELens, choose_lam, fit_line, load_maps  # noqa: E402


def build(ctx):
    maps = load_maps("/workspace/o-lens-improvement/.hf27/al3/M_a.pt", ctx.device)
    lam = choose_lam(ctx, maps, "M mix")
    fit_line("M alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
