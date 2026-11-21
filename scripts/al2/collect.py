"""al2 step 2: filter (cut per style, word/inflection leak, >= 6 tokens, near-duplicate Jaccard >= .6 within the
word's sentences of the same style), keep the first NK per (word, style) (NK = 3, or 2 if CUT2), keep words with
>= 2 per style (>= 1 if CUT2), split BY WORD (seed 0) 400 val / 400 test / rest train, collect the last-token
residual of each sentence run alone at LAYERS (fp16, one file per layer), targets as al1."""
import json, os, random, re, sys, time
from collections import Counter, defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al2_common import BIG, BIG1, log, retry, load_model, cut, names, STYLE_NAMES

torch.set_grad_enabled(False)
LAYERS = [4, 8, 12, 16, 20, 24, 28]
HERE = os.path.dirname(os.path.abspath(__file__))
words = json.load(open(f"{HERE}/words.json"))
al1w = set(json.load(open(f"{HERE}/al1_words.json")))
CUT2 = os.path.exists(f"{BIG}/CUT2")
NK, MINK = (2, 1) if CUT2 else (3, 2)
raw = defaultdict(list)
n_raw = Counter()
for l in open(f"{BIG1}/gen_raw.jsonl"):
    o = json.loads(l)
    if o["word"] in al1w:
        raw[(o["word"], o["style"])].append(o["text"]); n_raw["al1 reused raw"] += 1
for l in open(f"{BIG}/gen_raw.jsonl"):
    try:
        o = json.loads(l)
    except Exception:
        continue
    if o["style"] < 2 and o["word"] in al1w:
        continue
    raw[(o["word"], o["style"])].append(o["text"]); n_raw[f"al2 pass {o['pass']}"] += 1
model = load_model()
tok = model.tokenizer
d = model.d_model


def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))


drops = Counter()
kept = defaultdict(list)
for (w, k), texts in raw.items():
    mine = []
    for t in texts:
        if len(mine) >= NK:
            break
        s = cut(t, k)
        if s is None:
            drops[f"S{k + 1} no clean end"] += 1; continue
        if names(s, w):
            drops[f"S{k + 1} contains word/inflection"] += 1; continue
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) < 6:
            drops[f"S{k + 1} < 6 tokens"] += 1; continue
        tx = toks(s)
        if any(len(tx & ty) / max(1, len(tx | ty)) >= 0.6 for _, _, ty in mine):
            drops[f"S{k + 1} near-duplicate"] += 1; continue
        mine.append((s, ids, tx))
    kept[(w, k)] = mine
good = [w for w in words if all(len(kept[(w, k)]) >= MINK for k in range(5))]
log(f"raw {dict(n_raw)}; CUT2 {CUT2}; NK {NK}; words kept {len(good)} of {len(words)}")
log(f"drops (only counted until NK reached) {dict(sorted(drops.items()))}")
log("per-style kept counts: " + " ".join(f"S{k + 1}:{Counter(len(kept[(w, k)]) for w in words)}" for k in range(5)))
ws = list(good); random.Random(0).shuffle(ws)
role = {w: ("val" if i < 400 else "test" if i < 800 else "train") for i, w in enumerate(ws)}
rows = [dict(word=w, style=k, sent=s, ids=ids, split=role[w]) for w in good for k in range(5) for s, ids, _ in kept[(w, k)]]
log(f"sentences {len(rows)}; by split {Counter(r['split'] for r in rows)}; by style {Counter(r['style'] for r in rows)}; "
    f"words {Counter(role.values())}")
rng = random.Random(1)
for k in range(5):
    print(f"== examples {STYLE_NAMES[k]}")
    for r in rng.sample([r for r in rows if r["style"] == k], 6):
        print(f"   [{r['word']}] {r['sent']}")
U = model._lm_head.weight
g = model._final_norm(torch.ones(1, d, device="cuda", dtype=torch.float32)).float()[0]
T = {}
for w in good:
    forms = {" " + w, w, " " + w.lower(), w.lower(), " " + w.lower().capitalize(), w.lower().capitalize()}
    ids = sorted({t[0] for t in tok(sorted(forms), add_special_tokens=False).input_ids if len(t) == 1})
    u = U[torch.tensor(ids, device="cuda")].float()
    T[w] = ((u / u.norm(dim=-1, keepdim=True)).mean(0) / g).cpu()
meta = dict(words=[r["word"] for r in rows], split=[r["split"] for r in rows], style=[r["style"] for r in rows],
            sents=[r["sent"] for r in rows], role=role, T=T, g=g.cpu(), layers=LAYERS, drops=dict(drops),
            n_raw=dict(n_raw), cut2=CUT2)
torch.save(meta, f"{BIG}/meta.pt")
all_ids = [r["ids"] for r in rows]
out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16) for l in LAYERS}
groups = {}
for i, s in enumerate(all_ids):
    groups.setdefault(len(s), []).append(i)


def one(sub):
    ids = torch.tensor([all_ids[i] for i in sub], device="cuda")
    rec = {}
    hk = [model.layers[l].register_forward_hook(
        (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))(l)) for l in LAYERS]
    try:
        model.forward(ids)
    finally:
        for h in hk:
            h.remove()
    return {l: rec[l].to(torch.float16).cpu() for l in LAYERS}


t1 = time.time()
for L, idx in sorted(groups.items()):
    for b0 in range(0, len(idx), 128):
        sub = idx[b0:b0 + 128]
        r = retry(one, sub)
        for l in LAYERS:
            out[l][torch.tensor(sub)] = r[l]
log(f"forward done {time.time() - t1:.0f}s")
for l in LAYERS:
    torch.save(out[l], f"{BIG}/X_L{l}.pt")
log("saved", BIG)
print("COLLECT_DONE")
