"""ask6 (b): E_spec alone, unit(E_spec h + b)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask6_lib import Lens, load_maps, s6_path  # noqa: E402


def build(ctx):
    return Lens(ctx.JPP, None, load_maps(s6_path(ctx), ctx.device), 0.0, 0.0, alone=True)
