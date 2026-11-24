"""w9: print finished dev evals listed in spec files. python3 status.py <spec>..."""
import json
import os
import sys

R = "/workspace/o-lens-improvement"
for sp in sys.argv[1:]:
    for line in open(sp):
        n = line.split("|")[0]
        p = f"{R}/auto/runs/{n}/summary.json"
        if os.path.exists(p):
            d = json.load(open(p))
            print(f"{n:40s} {d['pass10']:.3f} {d['pass1']:.3f} {d['total_seconds']:.0f}s")
