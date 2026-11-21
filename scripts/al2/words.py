"""al2 step 0 (al1 check is case-sensitive: the LOWER-CASE form must occur >= 3 times): 8,000 words. Same recipe as g1_concepts.py (single-token ' word' vocab entries, case-folded frequency over
prefixes.json + prompts_natural.json, drop top 300) but WITHOUT the 6,000 cap, then the g1 strict exclusion and al1's
English whole-word check (lower-case word >= 3 times as a whole word in the English prefixes sources story+wiki_en),
dedupe by lower case. First 8,000 in that order. The first 2,400 must equal al1's words.json."""
import json, re, sys
from collections import Counter
from transformers import AutoTokenizer
R = "/workspace/o-lens-improvement"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
pat = re.compile(r"^ [A-Za-z]{3,}$")
cand = {}
for i in range(len(tok)):
    try:
        s = tok.decode([i])
    except Exception:
        continue
    if pat.match(s) and i not in tok.all_special_ids:
        cand[i] = s[1:]
P = json.load(open(f"{R}/runs/e1/prefixes.json"))
texts = [o["text"] for o in P]
texts += [t if isinstance(t, str) else t["text"] for t in json.load(open(f"{R}/runs/r1_2b/prompts_natural.json"))]
cnt = Counter()
for i in range(0, len(texts), 1000):
    for ids in tok(texts[i:i + 1000], add_special_tokens=False).input_ids:
        cnt.update(t for t in ids if t in cand)
by_lower = {}
for tid, w in cand.items():
    k = w.lower(); c = cnt.get(tid, 0)
    tot, best = by_lower.get(k, (0, None))
    if best is None or c > cnt.get(best, 0):
        best = tid
    by_lower[k] = (tot + c, best)
rows = sorted(((tot, cand[b], b) for k, (tot, b) in by_lower.items()), key=lambda x: (-x[0], x[1]))
rows = rows[300:]
excl = set(open(f"{R}/runs/g1/exclude_concepts.txt").read().lower().split("\n")) - {""}
def excluded(w):
    w = w.lower()
    return w in excl or w + "s" in excl or (w.endswith("s") and w[:-1] in excl)
strict = [w for c, w, t in rows if not excluded(w)]
old = json.load(open(f"{R}/runs/g1/concepts_strict.json"))
print("strict (uncapped)", len(strict), "prefix equals g1 strict:", strict[:len(old)] == old)
en = Counter()
for o in P:
    if o["src"] in ("story", "wiki_en"):
        en.update(re.findall(r"[A-Za-z]+", o["text"]))
keep, seen = [], set()
tier = {}
for th in (3, 2, 1):  # only 5,953 words pass at >= 3; fill to 8,000 with >= 2, then >= 1 (strict order inside a tier)
    for w in strict:
        if en[w.lower()] >= th and w.lower() not in seen and len(keep) < 8000:
            seen.add(w.lower()); keep.append(w); tier[w] = th
print("tiers", Counter(tier.values()))
al1 = json.load(open(f"{R}/scripts/al1/words.json"))
print("pass English check", len(keep), "first 2400 == al1:", keep[:2400] == al1)
print("from g1 strict list:", sum(1 for w in keep[:8000] if w in set(old)), "position of 8000th in strict:",
      strict.index(keep[min(7999, len(keep) - 1)]))
import random
print("sample of words 3445-8000:", random.Random(0).sample(keep[3444:8000], 60))
json.dump(keep[:8000], open("/root/al2/words.json", "w"))
json.dump(tier, open("/root/al2/word_tier.json", "w"))
