"""Paired bootstrap for PREREG_v3 amendment 6 / PREREG_v2 amendment 4 (G = top-1 guard, L = re-weighted). Same metric
and resampling as scripts/c3_boot.py."""
from pathlib import Path
import numpy as np, pandas as pd
R = Path("/workspace/o-lens-improvement"); NB = 5000
L = {"9b": dict(J="jpp_9b", C="C3_9b", G="G_9b", L="L_9b"), "27b": dict(J="jpp_27b", C="C3_27b", G="G_27b", L="L_27b")}
KEYS = ["J", "C", "G", "L"]
def load(v, t, kind, k):
    df = pd.concat([pd.read_csv(R / "auto/runs" / f"V{v}_{L[t][n]}" / f"readout_{kind}.csv").assign(lens=n) for n in KEYS])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    assert not item[KEYS].isna().any().any()
    return {e: item.loc[e][KEYS].to_numpy() for e in sorted(item.index.get_level_values(0).unique())}
def boot(per, seed):
    rng = np.random.default_rng(seed); ev = sorted(per)
    point = np.mean([per[e].mean(0) for e in ev], 0); b = np.empty((NB, len(KEYS)))
    for i in range(NB):
        b[i] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in ev], 0)
    return point, b
def line(name, p, b, x, y):
    i, j = KEYS.index(x), KEYS.index(y); d = b[:, i] - b[:, j]
    return f"  {name:14s} {x}-{y} {100*(p[i]-p[j]):+5.1f} [{100*np.percentile(d,2.5):+5.1f},{100*np.percentile(d,97.5):+5.1f}]  ({100*p[j]:.1f} -> {100*p[i]:.1f})"
for kind, k in (("filtered", 1), ("filtered", 10), ("unfiltered", 10), ("unfiltered", 1)):
    print(f"\n==== {kind} pass@{k}")
    res = {(v, t): boot(load(v, t, kind, k), seed=v * 10 + (0 if t == "9b" else 1)) for v in (3, 2) for t in ("9b", "27b")}
    for t in ("27b", "9b"):
        for x, y in (("G", "C"), ("G", "J"), ("L", "C"), ("L", "J"), ("C", "J")):
            for v in (3, 2):
                print(line(f"v{v} {t}", *res[(v, t)], x, y))
            p = (res[(2, t)][0] + res[(3, t)][0]) / 2; b = (res[(2, t)][1] + res[(3, t)][1]) / 2
            print(line(f"v2+v3 {t}", p, b, x, y))
        print()
