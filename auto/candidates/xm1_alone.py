"""xm1: the ask-map read alone, unit(E h + b) (no J++ term)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xm1_lib import Alone  # noqa: E402


def build(ctx):
    return Alone(ctx)
