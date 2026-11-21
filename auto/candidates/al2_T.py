"""al2 (T): unit(J++ h) + lam * unit(M_all h + b), M_all = learned concept->word map (not a Jacobian), al1 recipe
scaled to ~7,200 words x 5 text styles (scripts/al2/fit.py). lam chosen on the FIT split over {0.1, 0.2, 0.3, 0.5, 1.0}
(ties -> smaller), as e2_A."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al2_lib import ELens, choose_lam, fit_line, load_maps, m_path  # noqa: E402


def build(ctx):
    maps = load_maps(m_path("all"), ctx.device)
    lam = choose_lam(ctx, maps, "M mix")
    fit_line("M alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
