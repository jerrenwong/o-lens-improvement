"""g1 step 1: label-free concept words.
Vocabulary tokens = leading space + ASCII letters only, length >= 3 letters. Frequency counted over ordinary text
(runs/e1/prefixes.json texts + runs/r1_2b/prompts_natural.json), case-folded (" Water" and " water" merged; the
most frequent surface form kept). Drop the 300 most frequent, keep the next 6000.
STRICT = that list minus the exclusion file (lower-case; also plus/minus a final "s").
Writes runs/g1/concepts.json [{word, tid, count}] and runs/g1/concepts_strict.json (list of words).
"""
import json
import random
import re
from collections import Counter

from transformers import AutoTokenizer

R = "/workspace/o-lens-improvement"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
V = len(tok)
pat = re.compile(r"^ [A-Za-z]{3,}$")
cand = {}
for i in range(V):
    try:
        s = tok.decode([i])
    except Exception:
        continue
    if pat.match(s) and i not in tok.all_special_ids:
        cand[i] = s[1:]
print("candidate tokens", len(cand))
texts = [o["text"] for o in json.load(open(f"{R}/runs/e1/prefixes.json"))]
texts += [t if isinstance(t, str) else t["text"] for t in json.load(open(f"{R}/runs/r1_2b/prompts_natural.json"))]
cnt = Counter()
ntok = 0
for i in range(0, len(texts), 1000):
    for ids in tok(texts[i:i + 1000], add_special_tokens=False).input_ids:
        ntok += len(ids)
        cnt.update(t for t in ids if t in cand)
print("texts", len(texts), "tokens", ntok)
by_lower = {}
for tid, w in cand.items():
    k = w.lower()
    c = cnt.get(tid, 0)
    tot, best = by_lower.get(k, (0, None))
    if best is None or c > cnt.get(best, 0):
        best = tid
    by_lower[k] = (tot + c, best)
rows = sorted(((tot, cand[b], b) for k, (tot, b) in by_lower.items()), key=lambda x: (-x[0], x[1]))
print("case-folded words", len(rows), "with count>0", sum(r[0] > 0 for r in rows))
print("top 300 (dropped), first 40:", [r[1] for r in rows[:40]])
keep = rows[300:6300]
print(f"kept 6000: count range {keep[0][0]}..{keep[-1][0]}")
out = [dict(word=w, tid=t, count=c) for c, w, t in keep]
excl = set(open(f"{R}/runs/g1/exclude_concepts.txt").read().lower().split("\n")) - {""}


def excluded(w):
    w = w.lower()
    return w in excl or w + "s" in excl or (w.endswith("s") and w[:-1] in excl)


strict = [o["word"] for o in out if not excluded(o["word"])]
print("STRICT concepts", len(strict), "(removed", len(out) - len(strict), ")")
rng = random.Random(0)
print("60 random concepts:", [o["word"] for o in rng.sample(out, 60)])
with open(f"{R}/runs/g1/concepts.json", "w") as f:
    json.dump(out, f)
with open(f"{R}/runs/g1/concepts_strict.json", "w") as f:
    json.dump(strict, f)
