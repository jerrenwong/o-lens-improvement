"""rb1 control candidate: map prior3000 at the lam the FIT rule picks for map r3000_0 (see rb1_lib2.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb1_lib2 import build_at  # noqa: E402


def build(ctx):
    return build_at(ctx, "prior3000", "r3000_0")
