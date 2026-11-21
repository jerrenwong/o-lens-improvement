"""ask3 (A): J++ + lam * E_t3 (refit 27B topic ask-map, word-position targets; learned, not a Jacobian). lam on FIT."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import ELens, choose_lam, load_maps, t3_path  # noqa: E402


def build(ctx):
    maps = load_maps(t3_path(ctx), ctx.device)
    return ELens(ctx.JPP, maps, "mix", choose_lam(ctx, maps))
