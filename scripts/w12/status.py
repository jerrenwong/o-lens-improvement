"""print dev pass@10 / pass@1 of every w12 eval of a tag (hub). python3 status.py 2b"""
import json, sys, glob, os
tag = sys.argv[1] if len(sys.argv) > 1 else "2b"
R = "/workspace/o-lens-improvement/auto/runs"
names = sorted(os.listdir(R))
for n in names:
    if n.startswith(f"w12_{tag}_") and os.path.exists(f"{R}/{n}/summary.json"):
        d = json.load(open(f"{R}/{n}/summary.json"))
        t = d["per_task_pass10"]
        print(f"{n:44s} {d['pass10']:.3f} {d['pass1']:.3f} | " + " ".join(f"{k[:5]} {v:.2f}" for k, v in t.items()))
