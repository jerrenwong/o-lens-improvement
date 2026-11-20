"""g1 (G): unit(J++ h) + lam * unit(G_strict h + b). G_strict = learned map trained on model-generated sentences
that describe a concept without naming it (benchmark words removed from the concepts), not a Jacobian.
lam chosen on the FIT split over {0.1, 0.2, 0.3, 0.5, 1.0} (ties -> smaller)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_lib import build_mix  # noqa: E402


def build(ctx):
    return build_mix(ctx, "strict")
