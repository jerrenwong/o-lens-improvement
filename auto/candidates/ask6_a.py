"""ask6 (a): J++ + lam * E_spec (specific-word ask-map, ask6 format fix); lam on FIT by pass@1 + pass@10."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask6_lib import Lens, lam_a, load_maps, s6_path  # noqa: E402


def build(ctx):
    lam = lam_a(ctx)
    print(f"[ask6 a] lam {lam}", flush=True)
    return Lens(ctx.JPP, None, load_maps(s6_path(ctx), ctx.device), 0.0, lam)
