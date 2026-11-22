"""st1 noise replicate: lens p3_t05 with output scaled by 1.19 (see st1_lib.build_scaled)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st1_lib import build_scaled  # noqa: E402


def build(ctx):
    return build_scaled(ctx, "p3_t05", 1.19)
