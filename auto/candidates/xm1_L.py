"""xm1 L (PREREG_v5 item 4): unit(J++ h) + lam unit(E h + b) for the non-Qwen models; lam chosen on the model's FIT
split from {0, .05, .1, .2, .3, .5, 1} by max pass@1 + pass@10 (filtered; ties -> smaller lam)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xm1_lib import XBase, choose_lam  # noqa: E402


def build(ctx):
    return XBase(ctx, choose_lam(ctx))
