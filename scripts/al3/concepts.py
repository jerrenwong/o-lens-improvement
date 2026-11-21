"""al3 step 1: concept list + word-level split + stem separation.
Union of keep=True words of .hf27/sub1/al2_words_judged.json and concepts_v2_judged.json; surface = canon if canon
differs from word only by capitalisation, else word. Dropped: exclusion list (runs/sub1/exclude_concepts.txt,
case-insensitive; also with the pilot's plural/singular variants), few-shot words of the prompts (new words only), no
single-token form (pilot/sub2 forms rule). Dedupe case-insensitively (al2 spelling first, then al2-judged list, then v2).
A word whose lower case equals an al2 word's lower case IS that al2 word (al2 spelling, al2 role). New words:
train/val/test 90/5/5 per type, seed 0. Stem separation: a val/test word w goes to "near" if any train word (al2 train
words, all 7k, or al3 train words) starts with w.lower()[:5] (= shares the first 5 letters; for words shorter than 5
letters: a train word starts with the whole word).
Writes .hf27/al3/concepts.json and scripts/al3/results/concepts.json (counts)."""
import json, random, sys
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al3")
from al3_common import R, BIG, BIG2, RES, FEWSHOT, forms_tok, log  # noqa: E402

TYPES = ["PERSON", "PLACE", "THING", "EVENT", "STATE"]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
role2 = torch.load(f"{BIG2}/meta.pt", weights_only=False)["role"]
al2_low = defaultdict(list)
for w in role2:
    al2_low[w.lower()].append(w)
amb = {k: v for k, v in al2_low.items() if len(v) > 1}
log(f"al2 words {len(role2)}; lower-case collisions inside al2: {len(amb)} {list(amb.items())[:5]}")

ex = set(x.strip().lower() for x in open(f"{R}/runs/sub1/exclude_concepts.txt") if x.strip())


def excluded(w):  # pilot excluded_fn variants, without the few-shot part
    w = w.lower()
    vs = {w, w + "s", w + "es"}
    if w.endswith("s"):
        vs.add(w[:-1])
    if w.endswith("es"):
        vs.add(w[:-2])
    if w.endswith("ies"):
        vs.add(w[:-3] + "y")
    if w.endswith("y"):
        vs.add(w[:-1] + "ies")
    return bool(vs & ex), w in ex


few = {w.lower() for w in FEWSHOT}
cnt = Counter()
cands = []
for src in ("al2j", "v2j"):
    fn = "al2_words_judged.json" if src == "al2j" else "concepts_v2_judged.json"
    for r in json.load(open(f"{R}/.hf27/sub1/{fn}")):
        if not r["keep"]:
            continue
        cnt[f"kept {src}"] += 1
        w = r["word"]
        c = r.get("canon")
        if c and c != w:
            if c.lower() == w.lower():
                w = c
                cnt["canon used (case only)"] += 1
            else:
                cnt["canon ignored (not case only)"] += 1
        cands.append(dict(word=w, type=r["type"], src=src, orig=r["word"]))
log(f"candidates {len(cands)}; {dict(cnt)}")

seen = {}
drop = Counter()
drop_ex = defaultdict(list)
for c in cands:
    w, lw = c["word"], c["word"].lower()
    in2 = lw in al2_low
    if in2:
        c["word"] = al2_low[lw][0] if len(al2_low[lw]) == 1 else (w if w in al2_low[lw] else al2_low[lw][0])
    exv, exx = excluded(c["word"])
    if exv:
        k = "exclusion list (exact)" if exx else "exclusion list (plural/singular variant)"
        drop[k] += 1; drop_ex[k].append(c["word"]); continue
    if not in2 and lw in few:
        drop["few-shot word"] += 1; drop_ex["few-shot word"].append(c["word"]); continue
    if not forms_tok(tok, c["word"]):
        drop["no single-token form"] += 1; drop_ex["no single-token form"].append(c["word"]); continue
    if lw in seen:
        drop["case-insensitive duplicate"] += 1
        continue
    c["in_al2"] = in2
    seen[lw] = c
log(f"drops {dict(drop)}")
for k, v in drop_ex.items():
    log(f"  {k}: {v[:25]}")
C = list(seen.values())
typ_bad = [c for c in C if c["type"] not in TYPES]
assert not typ_bad, typ_bad[:5]

for c in C:
    if c["in_al2"]:
        c["split"] = role2[c["word"]]
new_by_t = defaultdict(list)
for c in C:
    if not c["in_al2"]:
        new_by_t[c["type"]].append(c)
for t in TYPES:
    xs = sorted(new_by_t[t], key=lambda c: c["word"].lower())
    random.Random(0).shuffle(xs)
    n = len(xs)
    nv = round(0.05 * n)
    for i, c in enumerate(xs):
        c["split"] = "val" if i < nv else "test" if i < 2 * nv else "train"

train_low = [w.lower() for w, r in role2.items() if r == "train"] + [c["word"].lower() for c in C if c["split"] == "train"]
pref = defaultdict(list)
for t in set(train_low):
    for k in range(1, 6):
        pref[t[:k]].append(t)
near_ex = []
for c in C:
    c["split0"] = c["split"]
    if c["split"] in ("val", "test"):
        p = c["word"].lower()[:5]
        mates = sorted(set(pref.get(p, [])))
        if mates:
            c["split"] = "near"
            c["near_mates"] = mates[:5]
            near_ex.append((c["word"], c["split0"], mates[:3]))
log(f"near examples {near_ex[:40]}")

tab = defaultdict(Counter)
for c in C:
    tab[c["type"]][c["split"]] += 1
    tab[c["type"]]["al2" if c["in_al2"] else "new"] += 1
    tab["ALL"][c["split"]] += 1
    tab["ALL"]["al2" if c["in_al2"] else "new"] += 1
    tab[f"{'al2' if c['in_al2'] else 'new'} words"][c["split"]] += 1
nearsrc = Counter((c["split0"], "al2" if c["in_al2"] else "new") for c in C if c["split"] == "near")
for k in ["ALL"] + TYPES + ["al2 words", "new words"]:
    log(f"{k:10s} " + " ".join(f"{s} {tab[k][s]}" for s in ("train", "val", "test", "near", "al2", "new")))
log(f"near by (original split, origin): {dict(nearsrc)}")
json.dump(C, open(f"{BIG}/concepts.json", "w"), indent=0)
json.dump(dict(cands=len(cands), cand_counts=dict(cnt), drops=dict(drop), drop_examples={k: v[:40] for k, v in drop_ex.items()},
               table={k: dict(v) for k, v in tab.items()}, near_by_origin={str(k): v for k, v in nearsrc.items()},
               near_examples=near_ex[:80], n=len(C)), open(f"{RES}/concepts.json", "w"), indent=1)
print("CONCEPTS_DONE")
