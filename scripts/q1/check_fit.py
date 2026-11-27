"""Part A check: bench-fit ranks / top lists of a new qual_dump vs the converted dump of the same lens (same ids).
    python3 scripts/q1/check_fit.py runs/qual/9b_converted/jpp.json runs/qual/9b/jpp.json"""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
A = {i["id"]: i for i in a["items"] if i["set"] == "bench-fit"}
B = {i["id"]: i for i in b["items"] if i["set"] == "bench-fit"}
print(f"{sys.argv[2]}: ids old {len(A)} new {len(B)} common {len(A.keys() & B.keys())} only-old {sorted(A.keys() - B.keys())[:5]} only-new {sorted(B.keys() - A.keys())[:5]}")
nr = nt = nb = n = 0; diffs = []
for k in A.keys() & B.keys():
    for la, lb in zip(A[k]["layers"], B[k]["layers"]):
        n += 1
        if la["r"] != lb["r"]:
            nr += 1; diffs.append((k, la["r"], lb["r"]))
        if la["t"][:10] != lb["t"][:10]:
            nt += 1
    nb += A[k]["best_rank"] != B[k]["best_rank"] or A[k]["best_layer"] != B[k]["best_layer"]
print(f"  (item, layer) pairs {n}: rank differs {nr}, top-10 list differs {nt}; best rank/layer differs on {nb} items")
for x in diffs[:8]: print("   ", x)
