"""al1 (T): unit(J++ h) + lam * unit(M h + b), M = learned concept->word translation map (not a Jacobian), trained on
2,000 non-benchmark words x ~20 model-written sentences each (scripts/al1/fit.py). lam chosen on the FIT split
over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller), as e2_A."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al1_lib import ELens, choose_lam, fit_line, load_maps, m_path  # noqa: E402


def build(ctx):
    maps = load_maps(m_path("best"), ctx.device)
    lam = choose_lam(ctx, maps, "M mix")
    fit_line("M alone", ctx, ELens(ctx.JPP, maps, "alone"))
    return ELens(ctx.JPP, maps, "mix", lam)
