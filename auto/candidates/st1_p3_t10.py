"""st1 candidate: lens L (fm27_F4 pattern) with map p3_t10 (see st1_lib.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st1_lib import build_L  # noqa: E402


def build(ctx):
    return build_L(ctx, "p3_t10")
