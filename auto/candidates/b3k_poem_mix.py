"""b3k idea 7: J++ + b x rollout-distilled newline ridge map (learned probe of the model's own 2nd-line endings); lam, b on held-out generated couplets"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b3k_poem_lib import build_poem  # noqa: E402


def build(ctx):
    return build_poem(ctx, mode="mix")
