"""ask5 (c): J++ + l1 * E_topic (ask3 t3) + l2 * E_spec; (l1, l2) on FIT by pass@1 + pass@10 over {0,.1,.2,.3,.5}^2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask5_lib import Lens, choose_c, load_maps, s5_path, t3_path  # noqa: E402


def build(ctx):
    mt, ms = load_maps(t3_path(ctx), ctx.device), load_maps(s5_path(ctx), ctx.device)
    l1, l2 = choose_c(ctx, mt, ms)
    return Lens(ctx.JPP, mt, ms, l1, l2)
