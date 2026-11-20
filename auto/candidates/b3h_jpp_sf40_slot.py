"""b3h: J++ + surface-form dedup K=40 + up to 2 readout slots replaced by J++ at an unread layer (greedy, fit union pass@10) (see b3h_lib2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3h_lib2 import make  # noqa: E402


def build(ctx):
    return make(ctx, "jpp", K=40, plural=False, n_rep=2)
