"""w1 step 1: generated hidden-word items in the harness's readout-eval JSON format.

Source: runs/w1/g1_kept.json (g1's 11,624 kept sentences, dumped from .hf27/g1/acts.pt by w1_dump.py).
Only STRICT concepts (runs/g1/concepts_strict.json: no benchmark word). Split by concept, seed 0:
concepts shuffled with random.Random(0); W-train = concepts taken in that order until 4,000 sentences
(a concept that would overflow is skipped), W-held = the following concepts until 1,000 sentences.
W-a, W-b, W-c: disjoint concept-level subsets of W-train (W-train concepts reshuffled with Random(1),
filled in turn to 1,300 sentences each).
Writes data/evals_gen/lens-eval-genword.json (W-train + W-held items, name = gw<idx>) and
data/evals_gen/splits.json {split: [item names]}.
"""
import json
import random
from collections import OrderedDict

R = "/workspace/o-lens-improvement"
kept = json.load(open(f"{R}/runs/w1/g1_kept.json"))
strict = set(json.load(open(f"{R}/runs/g1/concepts_strict.json")))
by_c = OrderedDict()
for i, k in enumerate(kept):
    if k["word"] in strict:
        by_c.setdefault(k["word"], []).append(i)
print(f"kept {len(kept)}; strict sentences {sum(map(len, by_c.values()))}; strict concepts {len(by_c)}")
concepts = sorted(by_c)
random.Random(0).shuffle(concepts)


def fill(pool, n):
    got, used = [], []
    for c in pool:
        if len(got) == n:
            break
        if len(got) + len(by_c[c]) <= n:
            got += by_c[c]
            used.append(c)
    assert len(got) == n, (len(got), n)
    return got, used


train, train_c = fill(concepts, 4000)
rest = [c for c in concepts if c not in set(train_c)]
held, held_c = fill(rest, 1000)
tc = list(train_c)
random.Random(1).shuffle(tc)
subs = {}
for name in ("W-a", "W-b", "W-c"):
    subs[name], used = fill(tc, 1300)
    tc = [c for c in tc if c not in set(used)]
name = {i: f"gw{i:05d}" for i in train + held}
items = [dict(name=name[i], prompt=kept[i]["sent"], intermediates=[kept[i]["word"]]) for i in sorted(train + held)]
splits = {"W-train": [name[i] for i in sorted(train)], "W-held": [name[i] for i in sorted(held)]}
for s in subs:
    splits[s] = [name[i] for i in sorted(subs[s])]
ws = {s: {kept[int(n[2:])]["word"] for n in v} for s, v in splits.items()}
assert not ws["W-train"] & ws["W-held"]
assert not (ws["W-a"] & ws["W-b"]) and not (ws["W-a"] & ws["W-c"]) and not (ws["W-b"] & ws["W-c"])
assert set(splits["W-a"]) | set(splits["W-b"]) | set(splits["W-c"]) <= set(splits["W-train"])
with open(f"{R}/data/evals_gen/lens-eval-genword.json", "w") as f:
    f.write(json.dumps({"items": items}, indent=1))
with open(f"{R}/data/evals_gen/splits.json", "w") as f:
    f.write(json.dumps(splits))
for s, v in splits.items():
    print(f"{s}: {len(v)} sentences, {len(ws[s])} concepts")
rng = random.Random(0)
print("10 examples (name | word | prompt):")
for it in rng.sample(items, 10):
    print(f"  {it['name']} | {it['intermediates'][0]} | {it['prompt']}")
