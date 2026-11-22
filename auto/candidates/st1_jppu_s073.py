"""st1 reference: unit(J++ h) * 0.73 (lens L base at lam 0; see st1_lib.Scaled)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st1_lib import Scaled  # noqa: E402


def build(ctx):
    return Scaled(ctx, 0.73)
