"""Post-hoc control for lens L (not registered): only the ask-map's constant part (its mean output), at L's weights.
unit(JPP h) + lam * unit(b): the same vector is added for every prompt."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import ELens, load_maps, t3_path  # noqa: E402

LAM_L = {4096: 0.3, 5120: 0.2}


def build(ctx):
    return ELens(ctx.JPP, load_maps(t3_path(ctx), ctx.device), "prior", LAM_L[ctx.model.d_model])
