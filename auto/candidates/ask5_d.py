"""ask5 (d): control, J++ + lam * shuffled-target E_spec at (a)'s FIT-chosen lam."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask5_lib import Lens, lam_a, load_maps, s5_path  # noqa: E402


def build(ctx):
    lam = lam_a(ctx)
    print(f"[ask5 d] lam {lam} (from (a))", flush=True)
    return Lens(ctx.JPP, None, load_maps(s5_path(ctx, shuf=True), ctx.device), 0.0, lam)
