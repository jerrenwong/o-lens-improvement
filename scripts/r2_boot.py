"""r2: paired bootstrap (5000 draws, items resampled within task, same draws for every lens) of differences
between dev-split lenses scored by auto/evaluate.py (auto/runs/<name>/readout_{filtered,unfiltered}.csv).
Metric as auto/evaluate.py / scripts/analyze.py: per (item, intermediate) best rank over layers; per item
share of intermediates with rank <= k; mean per task; macro over tasks.
    python scripts/r2_boot.py A-B C-D ...      (names are auto/runs/<name>)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
R = Path("/workspace/o-lens-improvement")
pairs = [p.split("-", 1) for p in sys.argv[1:]]
names = list(dict.fromkeys(n for p in pairs for n in p))
NB = 5000
for kind, k in (("filtered", 10), ("filtered", 1), ("unfiltered", 10)):
    df = pd.concat([pd.read_csv(R / "auto/runs" / n / f"readout_{kind}.csv").assign(lens=n) for n in names])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    assert not item[names].isna().any().any(), "lenses scored on different items"
    evals = sorted(item.index.get_level_values(0).unique())
    per = {e: item.loc[e][names].to_numpy() for e in evals}
    point = np.mean([per[e].mean(0) for e in evals], 0)
    rng = np.random.default_rng(0)
    boots = np.empty((NB, len(names)))
    for b in range(NB):
        boots[b] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in evals], 0)
    print(f"\n{kind} pass@{k}")
    for a, c in pairs:
        i, j = names.index(a), names.index(c)
        d = boots[:, i] - boots[:, j]
        print(f"  {a:>14s} - {c:<14s} {100*(point[i]-point[j]):+5.1f}  [{100*np.percentile(d,2.5):+5.1f}, {100*np.percentile(d,97.5):+5.1f}]")
