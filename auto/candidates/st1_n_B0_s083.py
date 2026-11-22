"""st1 noise replicate: lens B0 with output scaled by 0.83 (see st1_lib.build_scaled)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st1_lib import build_scaled  # noqa: E402


def build(ctx):
    return build_scaled(ctx, "B0", 0.83)
