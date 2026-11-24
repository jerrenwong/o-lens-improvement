"""W5 qualitative: J++ vs J++ + A-lens (lam 0.5) top-10 next to the item's target, for a few cells per family.
Usage: python qual.py [n_per_family] [layer]"""
import glob
import json
import random
import sys

R = "/workspace/o-lens-improvement/.hf27/w5/readouts"
B = "/root/wsbench/evals"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LAY = int(sys.argv[2]) if len(sys.argv) > 2 else 40
BANKS = {"model_specific_directed_modulation": "model_specific/model_specific_directed_modulation",
         "multilingual": "basic/multilingual", "multilingual_typo": "basic/multilingual_typo", "typo": "basic/typo"}


def target(it):
    if "gold" in it:
        return f"{it['gold']} (topic: {it.get('topic')})"
    en = [u.get("forms", {}).get("en") for u in it.get("units", []) if u.get("role") in ("concept", "correction")]
    return f"{it.get('target')} / en {en}"


def load(arm, fam):
    out = {}
    for line in open(f"{R}/{arm}/{fam}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["layer"] == LAY:
            out.setdefault(r["id"], []).append(r)
    return out


random.seed(0)
for fam, bank in BANKS.items():
    items = {}
    for f in sorted(glob.glob(f"{B}/{bank}/items*.json")):
        try:
            for it in json.load(open(f, encoding="utf-8")):
                items[it.get("id") or it.get("name")] = it
        except Exception:
            pass
    a, b, c = load("jpp", fam), load("jpp_a0.5", fam), load("alens", fam)
    ids = random.sample(sorted(a), min(N, len(a)))
    print(f"\n===== {fam} (layer {LAY}, last read position of the item)")
    for i in ids:
        ra, rb, rc = a[i][-1], b[i][-1], c[i][-1]
        print(f"- {i} | target: {target(items.get(i, {}))} | read at {ra['token']!r}")
        print(f"    J++      : {ra['tokens']}")
        print(f"    J++ + A  : {rb['tokens']}")
        print(f"    A alone  : {rc['tokens']}")
