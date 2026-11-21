"""ask3 (D): E_t3 alone (E h + b; learned map, not a Jacobian)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import ELens, fit_line, load_maps, t3_path  # noqa: E402


def build(ctx):
    lens = ELens(ctx.JPP, load_maps(t3_path(ctx), ctx.device), "alone")
    fit_line("t3 alone", ctx, lens)
    return lens
