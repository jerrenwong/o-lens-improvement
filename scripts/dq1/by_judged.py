"""dq1: bin shares of al2 sentences by the judge's keep/drop reason for the hidden word (CPU)."""
import json, collections, torch
R = "/workspace/o-lens-improvement"
meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
lab = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt", weights_only=False)
J = {r["word"]: r for r in json.load(open(f"{R}/.hf27/sub1/al2_words_judged.json"))}
words, split = meta["words"], meta["split"]
rank, rs = lab["rank"], lab["rank_shuf"]
C = collections.defaultdict(collections.Counter); Wn = collections.defaultdict(set); miss = 0
for i, w in enumerate(words):
    j = J.get(w)
    if j is None: miss += 1; continue
    g = "KEEP" if j["keep"] else "drop:" + j["reason"]
    r, q = int(rank[i]), int(rs[i])
    b = "SUB" if r <= 10 and q > 10 else "GIVE" if r <= 10 else "NOT"
    C[g][b] += 1; C[g]["top1"] += r == 1; Wn[g].add(w)
    if j["keep"]: C["KEEP:" + j["type"]][b] += 1; Wn["KEEP:" + j["type"]].add(w); C["KEEP:" + j["type"]]["top1"] += r == 1
print("missing words", miss)
print(f"{'group':18s} {'words':>6s} {'sents':>7s} {'infer':>6s} {'given':>6s} {'subtle':>6s} {'top1':>6s}")
for g in sorted(C, key=lambda g: -sum(C[g][b] for b in ("SUB", "GIVE", "NOT"))):
    n = sum(C[g][b] for b in ("SUB", "GIVE", "NOT"))
    print(f"{g:18s} {len(Wn[g]):6d} {n:7d} {(C[g]['SUB'] + C[g]['GIVE']) / n:6.3f} {C[g]['GIVE'] / n:6.3f} {C[g]['SUB'] / n:6.3f} {C[g]['top1'] / n:6.3f}")
role = meta["role"]
k = collections.Counter((role[w], J[w]["keep"]) for w in role if w in J); print("split x keep", dict(k))
