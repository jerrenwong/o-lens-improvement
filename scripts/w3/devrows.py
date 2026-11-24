import json, sys
R = "/workspace/o-lens-improvement/auto/runs/"
for n in sys.argv[1:]:
    try: s = json.load(open(R + n + "/summary.json")); print(f"{n:30s} {100*s['pass10']:.1f} / {100*s['pass1']:.1f}  {s['split']}")
    except Exception as e: print(n, "MISSING", e)
