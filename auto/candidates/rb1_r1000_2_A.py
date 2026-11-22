"""rb1 candidate: subset map r1000_2 read ALONE (see rb1_lib.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb1_lib import build_alone  # noqa: E402


def build(ctx):
    return build_alone(ctx, "r1000_2")
