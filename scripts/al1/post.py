"""al1: G_g1 contamination check. g1's map was trained on 80% of the strict concepts (incl. many of our test words).
Split our test sentences into words g1 trained on vs words g1 never saw; any-layer top-10 per method on each."""
import json, random, sys
import torch
sys.path.insert(0, "/root/al1")
from al1_common import BIG, R
READ = [4, 8, 12, 16, 20, 24, 28]
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
TR = torch.load(f"{BIG}/test_ranks.pt", weights_only=False)["test"]
A = torch.load(f"{R}/.hf27/g1/acts.pt", weights_only=False, mmap=True)
strict = set(json.load(open(f"{R}/runs/g1/concepts_strict.json")))
cw = sorted(set(A["words"])); random.Random(0).shuffle(cw)
nv = len(cw) // 10
g1role = {w: ("val" if i < nv else "test" if i < 2 * nv else "train") for i, w in enumerate(cw)}
te = [w for w, s in zip(meta["words"], meta["split"]) if s == "test"]
seen = torch.tensor([g1role.get(w) == "train" and w in strict for w in te])
print(f"test sentences {len(te)}; in g1 strict-train words: {int(seen.sum())} ({len({w for w, s in zip(te, seen) if s})} words); "
      f"unseen by g1: {int((~seen).sum())} ({len({w for w, s in zip(te, seen) if not s})} words)")
out = {}
for n in ["M1", "M2", "M3", "M4", "mix", "J++", "J", "logit", "G_g1", "E", "shuf"]:
    r = torch.stack([TR[(n, l)] for l in READ if (n, l) in TR]).min(0).values
    out[n] = dict(g1_seen=round(float((r[seen] <= 10).float().mean()), 4),
                  g1_unseen=round(float((r[~seen] <= 10).float().mean()), 4))
    print(f"{n:6s} any7 top10: g1-train words {out[n]['g1_seen']:.3f} | g1-unseen words {out[n]['g1_unseen']:.3f}")
json.dump(out, open(f"{BIG}/post.json", "w"))
