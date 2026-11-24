"""w9: per-layer filtered pass@10 of a dev run (single layer instead of best-over-layers). python3 perlayer.py <run>..."""
import csv
import sys
from collections import defaultdict

for name in sys.argv[1:]:
    by = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f"/workspace/o-lens-improvement/auto/runs/{name}/readout_filtered.csv")):
        by[int(r["layer"])][(r["eval"], r["item"])].append(int(r["rank"]) <= 10)
    out = []
    for l in sorted(by):
        pt = defaultdict(list)
        for (e, _), h in by[l].items():
            pt[e].append(sum(h) / len(h))
        out.append(f"L{l} {sum(sum(v) / len(v) for v in pt.values()) / len(pt):.3f}")
    print(name, " ".join(out))
