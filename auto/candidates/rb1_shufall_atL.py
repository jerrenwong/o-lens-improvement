"""rb1 control candidate: map shufall at the lam the FIT rule picks for map rall (see rb1_lib2.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb1_lib2 import build_at  # noqa: E402


def build(ctx):
    return build_at(ctx, "shufall", "rall")
