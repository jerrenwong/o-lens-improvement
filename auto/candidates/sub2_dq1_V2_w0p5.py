"""sub2/dq1 (V2_w0.5): unit(J++ h) + lam * unit(M h + b); M = al2 M3 map refitted by scripts/sub2/dq1_fit.py,
variant V2_w0.5 (see that file). Chosen as one of the two best dq1 variants by validation words. lam on the FIT split as al2_T."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sub2_lib import build_sub2  # noqa: E402


def build(ctx):
    return build_sub2(ctx, "dq1_V2_w0.5")
