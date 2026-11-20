"""e1 (B): elicitation-distilled probe E alone (E_l h + bias); learned, not a Jacobian (see e1_lib)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_lib import MAIN, build_fixed  # noqa: E402


def build(ctx):
    return build_fixed(ctx, MAIN, "alone", None)
