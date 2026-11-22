"""ex2: paired bootstrap of DEV scores (the algorithm of scripts/r2_boot.py: 5000 draws, items resampled within
task, the same draws for every lens; metric = auto/evaluate.py's: per (item, intermediate) best rank over
layers, per item share of intermediates with rank <= k, mean per task, mean over tasks).
Reads auto/runs/ex2_<name>/readout_{filtered,unfiltered}.csv (evaluate.py output) for all scored ex2 lenses of a
model; writes runs/ex2/boot_<model>.npz-like JSON with point values, per-task values and the bootstrap draws'
summaries for requested contrasts.
    python scripts/ex2/ex2_boot.py --model 9b
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
from ex2_common import OUT, R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
args = ap.parse_args()
specs = json.load(open(f"{OUT}/specs_{args.model}.json"))
names = [n for n in specs if os.path.exists(f"{R}/auto/runs/ex2_{n}/summary.json")]
NB = 5000


def table(kind, k):
    df = pd.concat([pd.read_csv(f"{R}/auto/runs/ex2_{n}/readout_{kind}.csv").assign(lens=n) for n in names])
    best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    assert not item[names].isna().any().any(), "lenses scored on different items"
    evals = sorted(item.index.get_level_values(0).unique())
    return {e: item.loc[e][names].to_numpy() for e in evals}


out = {"names": names, "metrics": {}}
for key, kind, k in (("p10", "filtered", 10), ("p1", "filtered", 1), ("p10u", "unfiltered", 10)):
    per = table(kind, k)
    evals = sorted(per)
    point = np.mean([per[e].mean(0) for e in evals], 0)
    rng = np.random.default_rng(0)
    boots = np.empty((NB, len(names)))
    for b in range(NB):
        boots[b] = np.mean([per[e][rng.integers(0, len(per[e]), len(per[e]))].mean(0) for e in evals], 0)
    out["metrics"][key] = dict(point={n: float(point[i]) for i, n in enumerate(names)},
                               per_task={n: {e: float(per[e][:, i].mean()) for e in evals} for i, n in enumerate(names)},
                               n_items={e: int(per[e].shape[0]) for e in evals})
    np.save(f"{OUT}/boots_{args.model}_{key}.npy", boots)
json.dump(out, open(f"{OUT}/boot_{args.model}.json", "w"), indent=1)
print("wrote", len(names), "lenses")
