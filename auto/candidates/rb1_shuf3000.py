"""rb1 control candidate: lens L with map shuf3000 (see rb1_lib.py; lam chosen on FIT)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rb1_lib import build_L  # noqa: E402


def build(ctx):
    return build_L(ctx, "shuf3000")
