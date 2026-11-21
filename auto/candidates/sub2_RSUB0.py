"""sub2 (RSUB0): unit(J++ h) + lam * unit(M_RSUB0 h + b); M_RSUB0 = al2 M3 map refitted on the RSUB0 subset of al2 train
sentences (see sub2_lib.py). lam chosen on the FIT split over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller), as al2_T."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sub2_lib import build_sub2  # noqa: E402


def build(ctx):
    return build_sub2(ctx, "RSUB0")
