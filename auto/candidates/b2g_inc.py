"""Idea 5: layer-increment lens unit(P_l) - a unit(P_{l-k}), P_l = JPP_l h_l. (a, k) chosen on the fit split
from a in {0.1, 0.25, 0.5}, k in {2, 4} (exact ranks); the sweep is printed next to plain J++ (a = 0)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2g_common import IncLens, log, report  # noqa: E402


def build(ctx):
    base = report(ctx, "a=0 (J++)", IncLens(ctx.JPP, 0.0, 2))
    best = (base[0], base[1], 0.0, 2)
    for k in (2, 4):
        for a in (0.1, 0.25, 0.5):
            p10, p1 = report(ctx, f"a={a} k={k}", IncLens(ctx.JPP, a, k))
            if (p10, p1) > best[:2]:
                best = (p10, p1, a, k)
    if best[2] == 0.0:  # nothing beat J++ on fit: still score the mildest variant on dev
        best = (None, None, 0.1, 4)
        log("no (a, k) beats J++ on fit; using a=0.1 k=4")
    log(f"chosen a={best[2]} k={best[3]}")
    return IncLens(ctx.JPP, best[2], best[3])
