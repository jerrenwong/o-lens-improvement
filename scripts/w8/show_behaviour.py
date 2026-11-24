"""W8 step 0 viewer (no torch): organism vs base replies on TRIGGER[:10] and PROBE, plus A-question answers.
    python3 scripts/w8/show_behaviour.py <quirk> [chars]"""
import json
import sys

q = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
g = json.load(open(f"/workspace/o-lens-improvement/runs/w8/gen_{q}.json"))["items"]
for it in g:
    if "base_reply" not in it:
        continue
    print(f"=== [{it['set']} {it['i']}] {it['prompt'][:160]}")
    print("ORG :", it["reply"][:n].replace("\n", " / "))
    print("BASE:", it["base_reply"][:n].replace("\n", " / "))
aq = [(it["i"], it["aq"]) for it in g if "aq" in it]
print("\n=== A-questions on own TRIGGER replies (org | base)")
for i, a in aq:
    print(i, " ; ".join(f"{k} {v['org']!r} | {v['base']!r}" for k, v in a.items()))
