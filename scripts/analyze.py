"""pass@k with paired bootstrap 95% intervals vs a reference lens, from jpp_cli readout CSVs.

Item-weighted, as jpp_cli: per (item, intermediate) best rank over layers; per item the share of
its intermediates with rank <= k; mean per eval; macro mean over evals. Bootstrap resamples items
within each eval (same draw for every lens, so differences are paired).

    python scripts/analyze.py runs/v1/eval/readout_filtered.csv [--ref jlens_prefit] [--k 1,10]
Writes <csv stem>_summary.json and prints a table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--ref", default="jlens_prefit")
ap.add_argument("--k", default="1,10")
ap.add_argument("--n-boot", type=int, default=10000)
args = ap.parse_args()
ks = [int(k) for k in args.k.split(",")]

df = pd.read_csv(args.csv)
best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
lenses = list(dict.fromkeys(best["lens"]))
evals = sorted(best["eval"].unique())
rng = np.random.default_rng(0)

out = {"csv": args.csv, "ref": args.ref, "evals": evals, "k": {}}
for k in ks:
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")  # rows (eval,item)
    per_eval = {e: item.loc[e][lenses].to_numpy() for e in evals}  # [n_items, n_lenses]
    point = np.mean([per_eval[e].mean(0) for e in evals], axis=0)
    boots = np.empty((args.n_boot, len(lenses)))
    for b in range(args.n_boot):
        boots[b] = np.mean(
            [per_eval[e][rng.integers(0, len(per_eval[e]), len(per_eval[e]))].mean(0) for e in evals], axis=0
        )
    r = lenses.index(args.ref)
    diff = boots - boots[:, [r]]
    out["k"][k] = {
        lens: {
            "macro": float(point[i]),
            "ci": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))],
            "diff_vs_ref": float(point[i] - point[r]),
            "diff_ci": [float(np.percentile(diff[:, i], 2.5)), float(np.percentile(diff[:, i], 97.5))],
            "per_eval": {e: float(per_eval[e][:, i].mean()) for e in evals},
        }
        for i, lens in enumerate(lenses)
    }
    print(f"\npass@{k}  (macro over {evals}; diff vs {args.ref}, paired bootstrap 95% CI)")
    for lens, v in out["k"][k].items():
        print(f"  {lens:14s} {v['macro']:.3f}  [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]   "
              f"diff {v['diff_vs_ref']:+.3f} [{v['diff_ci'][0]:+.3f},{v['diff_ci'][1]:+.3f}]")
n_items = best.groupby("eval")["item"].nunique().to_dict()
out["n_items"] = {k: int(v) for k, v in n_items.items()}
print("items per eval:", out["n_items"])
Path(args.csv).with_name(Path(args.csv).stem + "_summary.json").write_text(json.dumps(out, indent=1))
