"""mlp1 (D): unit(J++ h) + lam * unit(f_l(h)), f_l = D: MLP from scratch (no linear skip), full-vocab CE, early-stopped on validation words.
LEARNED map (not a Jacobian), fitted by scripts/mlp1/ on the al2 data; seed-0 map of the layer-20-chosen config at all
7 layers (.hf27/mlp1/F_D.pt). lam chosen on the FIT split over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp1_lib import build_group  # noqa: E402


def build(ctx):
    return build_group(ctx, "D", "mlp1 D")
