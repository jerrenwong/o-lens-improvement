"""st1 qualitative (FIT split only), 27B: B0 alone vs p3_t10 alone at two mid layers on 6 fit items J++ misses; returns unit J++ (sanity)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from st1_lib import qual  # noqa: E402


def build(ctx):
    return qual(ctx, ["p3_t05", "p3v_t10"])
