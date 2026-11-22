"""rb1 candidate: lens L with subset map r1000_2 (see rb1_lib.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb1_lib import build_L  # noqa: E402


def build(ctx):
    return build_L(ctx, "r1000_2")
