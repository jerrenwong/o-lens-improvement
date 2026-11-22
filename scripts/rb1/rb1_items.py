"""rb1 per-item comparison on DEV scores (auto/runs/<name>/readout_filtered.csv, the same rows and metric as
scripts/r2_boot.py: per (item, intermediate) best rank over layers; per item the share of intermediates with rank <= k).
For each lens vs unit J++: items gained (item score higher) / lost (lower), per task, and overlap of the gained / lost
sets between lenses. Usage: python rb1_items.py <tag>"""
import sys
from pathlib import Path

import pandas as pd

R = Path("/workspace/o-lens-improvement")
TAG = sys.argv[1]
BASE = f"rb1_jppu_{TAG}"
L = {"n300_0": "r300_0", "n300_1": "r300_1", "n300_2": "r300_2", "n3000_0": "r3000_0", "all": "rall",
     "prior300": "prior300", "priorall": "priorall", "shuf300": "shuf300", "shufall": "shufall",
     "prior300@L": "prior300_atL", "priorall@L": "priorall_atL"}
names = {k: f"rb1_{v}_{TAG}" for k, v in L.items()}
names = {k: n for k, n in names.items() if (R / "auto/runs" / n / "readout_filtered.csv").exists()}
allnames = [BASE] + list(names.values())
df = pd.concat([pd.read_csv(R / "auto/runs" / n / "readout_filtered.csv").assign(lens=n) for n in allnames])
best = df.groupby(["eval", "lens", "item", "intermediate"])["rank"].min().reset_index()
for k in (10, 1):
    best["hit"] = (best["rank"] <= k).astype(float)
    item = best.groupby(["eval", "lens", "item"])["hit"].mean().unstack("lens")
    d = {nm: item[n] - item[BASE] for nm, n in names.items()}
    print(f"\n== {TAG} pass@{k}: items vs J++ (n items {len(item)}): gained / lost (net item-score change summed)")
    for nm, x in d.items():
        per = " ".join(f"{e[:5]} +{int((x.loc[e] > 0).sum())}/-{int((x.loc[e] < 0).sum())}" for e in sorted(set(item.index.get_level_values(0))))
        print(f"   {nm:11s} +{int((x > 0).sum()):3d} -{int((x < 0).sum()):3d} net {x.sum():+.2f} | {per}")
    print("   overlap of GAINED sets (|A&B| / |A|, |B|):")
    keys = list(d)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a in ("n300_0", "all", "n3000_0", "prior300", "priorall") and b in ("n300_1", "n300_2", "all", "n3000_0", "prior300", "priorall", "prior300@L", "priorall@L"):
                ga, gb = set(d[a][d[a] > 0].index), set(d[b][d[b] > 0].index)
                la, lb = set(d[a][d[a] < 0].index), set(d[b][d[b] < 0].index)
                print(f"     {a:9s} vs {b:11s} gained {len(ga & gb):3d} of {len(ga):3d},{len(gb):3d} | lost {len(la & lb):3d} of {len(la):3d},{len(lb):3d}")
