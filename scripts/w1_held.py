"""w1 step 5: W-held pass@k from the jpp_cli evaluate rank rows (readout filter on; min rank over the 7 layers),
with a paired bootstrap (items resampled, 5000 draws) of differences vs the benchmark-fit lens of the same experts.
    python scripts/w1_held.py runs/w1/heldout_ranks.csv"""
import sys
import numpy as np
import pandas as pd
df = pd.read_csv(sys.argv[1])
best = df.groupby(["lens", "item"])["rank"].min().unstack("lens")
lenses = list(best.columns)
print(f"items {len(best)}, lenses {lenses}")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(best), (5000, len(best)))
PAIRS = [("jpp_Afull_Wtrain", "jpp_A_full"), ("pooled_A_full", "jpp_A_full"),
         ("jpp_Cfull_Wtrain", "jpp_C_full"), ("pooled_C_full", "jpp_C_full")]
for k in (10, 1):
    hit = (best <= k).astype(float)
    print(f"\npass@{k} (any layer)")
    for l in lenses:
        print(f"  {l:>18s} {hit[l].mean():.4f}")
    for a, b in PAIRS:
        d = (hit[a] - hit[b]).to_numpy()
        bs = d[idx].mean(1)
        print(f"  {a} - {b}: {100*d.mean():+.1f} [{100*np.percentile(bs,2.5):+.1f}, {100*np.percentile(bs,97.5):+.1f}]")
lay = df.assign(h=df["rank"] <= 10).groupby(["lens", "layer"])["h"].mean().unstack("layer")
print("\nper-layer pass@10")
print(lay.round(3).to_string())
