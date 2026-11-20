"""b3h: J-lens + surface-form dedup also merging plural/possessive, K=40 (see b3h_lib2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3h_lib2 import make  # noqa: E402


def build(ctx):
    return make(ctx, "jl", K=40, plural=True, n_rep=0)
