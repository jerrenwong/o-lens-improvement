"""Part A probe counts from the 9B qual dumps (stdlib)."""
import json, sys
from collections import defaultdict, Counter
R = "/workspace/o-lens-improvement/runs/qual/" + (sys.argv[1] if len(sys.argv) > 1 else "9b")
L = [("J", "j_lens"), ("J++", "jpp"), ("A", "a_lens"), ("J++ +A", "jpp_a_lens"), ("A CE98k", "a_lens_ce_98k"), ("J++ +A CE98k", "jpp_a_lens_ce_98k")]
D = {k: {i["id"]: i for i in json.load(open(f"{R}/{f}.json"))["items"] if i["set"] == "probes"} for k, f in L}
ref = D["J++"]
P = list(ref.values())
ctrl = [p for p in P if not p["accepted"]]
sc = [p for p in P if p["accepted"] and p["scorable"]]
uns = [p["id"] for p in P if p["accepted"] and not p["scorable"]]
knows = [p for p in sc if p["model"].get("knows_rank") is not None and p["model"]["knows_rank"] <= 10]
nxt = [p for p in sc if p["is_next"]]
pc = [p for p in sc if any(f["kind"] == "piece" for f in p["forms"])]
print(f"probes {len(P)}: controls {len(ctrl)}, unscorable {len(uns)} {uns}, scorable {len(sc)}; next-word {len(nxt)}; "
      f"not-next {len(sc) - len(nxt)}; model knows (raw top-10 of '<text>\\nThe thing this is about, in one word:') {len(knows)} "
      f"(not-next: {sum(1 for p in knows if not p['is_next'])}); with a piece form {len(pc)}")
print("next-word probes:", [p["id"] for p in nxt])
cats = sorted({p["task"] for p in sc})
print("\nnot-next-word hits (best layer, top 10) per category")
print("| category | n | " + " | ".join(k for k, _ in L) + " |"); print("|---|---|" + "---|" * len(L))
tot = Counter(); n = 0
for c in cats + ["ALL"]:
    ids = [p["id"] for p in sc if not p["is_next"] and (c == "ALL" or p["task"] == c)]
    row = [sum(1 for i in ids if D[k][i]["best_rank"] is not None and D[k][i]["best_rank"] <= 10) for k, _ in L]
    print(f"| {c} | {len(ids)} | " + " | ".join(map(str, row)) + " |")
print("\npiece-only hits (hit at best layer only through a first-piece form):")
for k, _ in L:
    ph = []
    for p in sc:
        it = D[k][p["id"]]
        if it["best_rank"] is not None and it["best_rank"] <= 10:
            e = it["layers"][it["layers"] and [x for x in range(len(it["layers"]))][0]] if False else None
            bl = it["best_layer"]; lay = json.load(open(f"{R}/jpp.json"))["meta"]["layers"] if False else [4, 8, 12, 16, 20, 24, 28] if R.endswith("9b") else [3, 6, 9, 12, 15, 18, 21]
            ent = it["layers"][lay.index(bl)]
            if ent.get("hp") and set(ent["hp"]) == set(ent["h"]):
                ph.append((p["id"], [ent["t"][j] for j in ent["h"]]))
    print(f"  {k}: {len(ph)} {ph}")
print("\nknows (raw) miss but scorable+not-next:", [(p["id"], p["model"]["knows"][:6]) for p in sc if not p["is_next"] and p not in knows][:40])
