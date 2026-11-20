"""g1 (Gall): unit(J++ h) + lam * unit(G_all h + b). G_all = learned map trained on model-generated sentences
(all 6000 concepts), not a Jacobian. lam chosen on the FIT split (ties -> smaller)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lib import build_mix  # noqa: E402


def build(ctx):
    return build_mix(ctx, "all")
