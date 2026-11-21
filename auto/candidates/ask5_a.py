"""ask5 (a): J++ + lam * E_spec (specific-word chat-template ask-map); lam on FIT by pass@1 + pass@10."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask5_lib import Lens, lam_a, load_maps, s5_path  # noqa: E402


def build(ctx):
    lam = lam_a(ctx)
    print(f"[ask5 a] lam {lam}", flush=True)
    return Lens(ctx.JPP, None, load_maps(s5_path(ctx), ctx.device), 0.0, lam)
