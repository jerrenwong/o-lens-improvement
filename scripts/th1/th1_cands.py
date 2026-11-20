"""th1 step 0: candidate words: strict list, lowercase alphabetic, both existing g1 sentences asked-rank <= 10."""
import json, re, torch
from collections import defaultdict
R = "/workspace/o-lens-improvement"
strict = set(json.load(open(f"{R}/runs/g1/concepts_strict.json")))
a = torch.load(f"{R}/.hf27/g1/acts.pt", map_location="cpu", weights_only=False)
by = defaultdict(list)
for w, s, r in zip(a["words"], a["styles"], a["ask_rank"].tolist()):
    by[w].append((s, r))
out = []
for w, lst in by.items():
    if w not in strict or not re.fullmatch(r"[a-z]{3,}", w):
        continue
    if sorted(s for s, _ in lst) != [0, 1] or max(r for _, r in lst) > 10:
        continue
    out.append(dict(word=w, ranks=[r for _, r in sorted(lst)]))
print(len(out))
json.dump(out, open(f"{R}/runs/th1/cands.json", "w"))
