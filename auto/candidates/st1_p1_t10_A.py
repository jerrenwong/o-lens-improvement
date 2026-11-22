"""st1 candidate: map p1_t10 read ALONE, v = unit(E h + b) (see st1_lib.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st1_lib import build_alone  # noqa: E402


def build(ctx):
    return build_alone(ctx, "p1_t10")
