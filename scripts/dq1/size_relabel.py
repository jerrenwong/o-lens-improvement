"""dq1 sizing (CPU): bins of all al2 sentences and how many get a usable model-own label."""
import json, re, collections, random, torch
R = "/workspace/o-lens-improvement"
meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
lab = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt", weights_only=False)
strs = torch.load(f"{R}/.hf27/al1/meta.pt", weights_only=False)["strs"]
words, sents, style, split = meta["words"], meta["sents"], meta["style"], meta["split"]
rank, rs, top3, top3s = lab["rank"], lab["rank_shuf"], lab["top3"], lab["top3_shuf"]
n = len(words); print("n", n, "top3 type", type(top3), getattr(top3, "shape", None))
def tk(x): return [strs[int(i)].strip() for i in x]
def stem(a, b):
    a, b = a.lower(), b.lower(); m = min(len(a), len(b)); k = max(4, min(5, m)); return m >= 4 and a[:k] == b[:k]
cat = collections.Counter(); bystyle = collections.defaultdict(collections.Counter)
relab = []  # (i, new_label)
random.seed(0); ex = collections.defaultdict(list)
for i in range(n):
    r, q = int(rank[i]), int(rs[i]); t = tk(top3[i]); ts = tk(top3s[i])
    toks = set(re.findall(r"[a-z']+", sents[i].lower()))
    if r <= 10: c = "GIVE" if q <= 10 else "SUB"
    else:
        real = [x for x in t if x and x[0].isalpha() and len(x) >= 3]
        if not real: c = "NOT: no word answer"
        elif any(stem(x, words[i]) for x in real): c = "NOT: same word, other form"
        elif real[0].lower() in toks or any(stem(real[0], y) for y in toks): c = "NOT: copies a visible word"
        elif real[0].lower() in [y.lower() for y in ts]: c = "NOT: other word, also from shuffled"
        else:
            c = "NOT: other unstated word, needs order"; relab.append((i, real[0].lower()))
    cat[c] += 1; bystyle[style[i]][c] += 1
    if len(ex[c]) < 400: ex[c].append(i)
for c, v in cat.most_common(): print(f"{v / n:.3f} {v:7d} {c}")
for k in sorted(bystyle): print("style", k, {c: round(v / sum(bystyle[k].values()), 3) for c, v in bystyle[k].most_common()})
# forms: lemma-level inferable = SUB/GIVE + same-word-other-form
print("split counts", collections.Counter(split))
W = collections.defaultdict(collections.Counter)
for i in range(n):
    r, q = int(rank[i]), int(rs[i]); W[words[i]]["SUB" if r <= 10 and q > 10 else "GIVE" if r <= 10 else "NOT"] += 1
tot = len(W)
for k in (1, 2, 4, 8): print(f"words with >= {k} SUB: {sum(v['SUB'] >= k for v in W.values()) / tot:.3f}; >= {k} inferable: {sum(v['SUB'] + v['GIVE'] >= k for v in W.values()) / tot:.3f}")
zero = [w for w, v in W.items() if v["SUB"] + v["GIVE"] == 0]; best = sorted(W, key=lambda w: -(W[w]["SUB"]))[:40]
print("n words never inferable", len(zero), random.sample(zero, min(60, len(zero))))
print("most SUB words", best)
# relabel stats
L = collections.Counter(x for _, x in relab)
print("relabel sentences", len(relab), "distinct labels", len(L), ">=4 sentences", sum(v >= 4 for v in L.values()), ">=10", sum(v >= 10 for v in L.values()))
print("top relabels", L.most_common(40))
orig = set(w.lower() for w in W)
print("relabels that are original list words:", sum(x in orig for x in L), "of", len(L), "; sentences", sum(v for x, v in L.items() if x in orig))
for c in ("NOT: other unstated word, needs order", "NOT: same word, other form", "NOT: no word answer"):
    print("###", c)
    for i in random.sample(ex[c], 12): print(f"  [{words[i]}] r={int(rank[i])} top={tk(top3[i])} shuf={tk(top3s[i])} | {sents[i][:170]}")
json.dump(dict(cat=cat, relabel=[(int(i), x) for i, x in relab]), open(f"{R}/.hf27/sub2/dq1_sizing.json", "w"))
