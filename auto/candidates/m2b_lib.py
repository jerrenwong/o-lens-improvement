"""m2b shared code (not a candidate). Lens L (fm27_F4: v = unit(JPP h) + lam unit(E h + b)) for Qwen3.5-2B
(d_model 2048), with the 2B chat-format ask-map .hf27/ask3/E_2b_t3.pt (scripts/m2b/m2b_collect.py, m2b_fit.py =
2B copies of scripts/ask3/ask3_collect4.py, ask3_fit9.py). Same classes and selection rule as fm27_lib
(imported, not edited); only the map path differs (fm27_lib/ask3_lib know only the 9B/27B tags).
PREREG_v4.md Addendum B."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import load_maps  # noqa: E402
from fm27_lib import GRID, Base, choose, excl_mask, fit_eval_lens, fmt  # noqa: E402,F401

R = "/workspace/o-lens-improvement"


def t3_path_2b(ctx, shuf=False):
    assert ctx.model.d_model == 2048, "m2b lenses are for Qwen3.5-2B only"
    return f"{R}/.hf27/ask3/E_2b_t3{'_shuf' if shuf else ''}.pt"


class Base2B(Base):
    """fm27_lib.Base with the 2B map (shuf=True: the shuffled-target control map)."""

    def __init__(self, ctx, lam=0.0, centred=False, shuf=False):
        self.JPP = ctx.JPP
        self.maps = load_maps(t3_path_2b(ctx, shuf), ctx.device)
        self.lam, self.centred = lam, centred
