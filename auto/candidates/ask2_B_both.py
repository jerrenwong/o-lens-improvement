"""ask2 (c): unit(J++h) + l1 unit(E_topic h + b) + l2 unit(E_spec h + b); (l1, l2) in {0,.2,.5,1}^2 on FIT
(ties -> smaller sum, then smaller l1)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask2_lib import Both, fit_line, load_maps, spec_path, topic_path  # noqa: E402

G = [0.0, 0.2, 0.5, 1.0]


def build(ctx):
    mt = load_maps(topic_path(ctx), ctx.device)
    ms = load_maps(spec_path(ctx), ctx.device)
    sc = {(a, b): fit_line(f"both {a}/{b}", ctx, Both(ctx.JPP, mt, ms, a, b)) for a in G for b in G}
    best = max(sc, key=lambda k: (round(sc[k], 6), -(k[0] + k[1]), -k[0]))
    print(f"[ask2 fit] both chosen l1(topic)={best[0]} l2(spec)={best[1]} fit {sc[best]:.4f}", flush=True)
    return Both(ctx.JPP, mt, ms, *best)
