"""Post-hoc control for lens L (not registered): shuffled-target ask-map at L's own weights (0.3 on 9B, 0.2 on 27B)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import ELens, load_maps, t3_path  # noqa: E402

LAM_L = {4096: 0.3, 5120: 0.2}


def build(ctx):
    return ELens(ctx.JPP, load_maps(t3_path(ctx, shuf=True), ctx.device), "mix", LAM_L[ctx.model.d_model])
