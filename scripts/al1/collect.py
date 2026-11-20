"""al1 step 2: filter (g1 filter + th1 near-dup), keep words with >= 12 sentences, split BY WORD (seed 0)
2000 train / 200 val / 200 test, collect last-token residual (sentence alone) at LAYERS (fp16, one file per layer),
targets t_X (mean unit unembedding row over single-token surface forms, final basis t/g)."""
import json, os, random, re, sys, time
from collections import Counter, defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from al1_common import BIG, log, retry, load_model, cut, names

torch.set_grad_enabled(False)
LAYERS = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28]
HERE = os.path.dirname(os.path.abspath(__file__))
words = json.load(open(f"{HERE}/words.json"))
raw = [json.loads(l) for l in open(f"{BIG}/gen_raw.jsonl")]
model = load_model()
tok = model.tokenizer
d = model.d_model
t0 = time.time()
drops, cand = Counter(), defaultdict(lambda: ([], []))
for o in raw:
    s = cut(o["text"])
    if s is None:
        drops["no sentence end"] += 1; continue
    if names(s, o["word"]):
        drops["contains word/inflection"] += 1; continue
    ids = tok(s, add_special_tokens=False).input_ids
    if len(ids) < 6:
        drops["< 6 tokens"] += 1; continue
    cand[o["word"]][o["style"]].append((s, ids))


def toks(s):
    return set(re.findall(r"[a-z']+", s.lower()))


rng = random.Random(0)
per_word, mine_all = {}, {}
for w in words:
    a, b = list(cand[w][0]), list(cand[w][1])
    rng.shuffle(a); rng.shuffle(b)
    order = [x for pair in zip(a, b) for x in pair] + a[len(b):] + b[len(a):]
    sty = {id(x): 0 for x in a} | {id(x): 1 for x in b}
    mine = []
    for x in order:
        tx = toks(x[0])
        if any(len(tx & ty) / max(1, len(tx | ty)) >= 0.6 for _, _, ty in mine):
            drops["near-duplicate"] += 1; continue
        mine.append((x, sty[id(x)], tx))
    per_word[w] = len(mine); mine_all[w] = mine
good = [w for w in words if per_word[w] >= 12]
log(f"raw {len(raw)}; drops {dict(drops)}; words with >=12 kept: {len(good)} of {len(words)}")
log(f"kept-per-word distribution: {sorted(Counter(per_word.values()).items())}")
ws = list(good); random.Random(0).shuffle(ws)
role = {w: ("val" if i < 200 else "test" if i < 400 else "train") for i, w in enumerate(ws)}
kept = []
for w in good:
    for (x, st, _) in mine_all[w]:
        kept.append(dict(word=w, style=st, sent=x[0], ids=x[1], split=role[w]))
log(f"sentences kept {len(kept)}; by split {Counter(k['split'] for k in kept)}; words by split "
    f"{Counter(role.values())}; style {Counter(k['style'] for k in kept)}")
for k in random.Random(1).sample(kept, 12):
    print(f"   {k['word']:>12s} s{k['style'] + 1} | {k['sent']}")
U = model._lm_head.weight
g = model._final_norm(torch.ones(1, d, device="cuda", dtype=torch.float32)).float()[0]
T, nforms = {}, Counter()
for w in good:
    forms = {" " + w, w, " " + w.lower(), w.lower(), " " + w.lower().capitalize(), w.lower().capitalize()}
    ids = sorted({t[0] for t in tok(sorted(forms), add_special_tokens=False).input_ids if len(t) == 1})
    u = U[torch.tensor(ids, device="cuda")].float()
    T[w] = ((u / u.norm(dim=-1, keepdim=True)).mean(0) / g).cpu()
    nforms[len(ids)] += 1
log(f"surface forms per word {sorted(nforms.items())}")
meta = dict(words=[k["word"] for k in kept], split=[k["split"] for k in kept], style=[k["style"] for k in kept],
            sents=[k["sent"] for k in kept], role=role, T=T, g=g.cpu(), layers=LAYERS, drops=dict(drops),
            per_word=per_word, n_raw=len(raw))
V = U.shape[0]
strs = tok.batch_decode([[i] for i in range(V)])
meta["strs"] = strs
x = torch.randn(4, d, device="cuda") * 30
a = model.unembed(x).float()
b = ((x * g).to(U.dtype) @ U.T).float()
log(f"unembed check: top-20 overlap model.unembed vs (x*g)@U^T: "
    f"{sum(len(set(p.tolist()) & set(q.tolist())) for p, q in zip(a.topk(20).indices, b.topk(20).indices)) / 80:.3f}")
if not os.path.exists(f"{BIG}/U.pt"):
    torch.save(U.detach().to(torch.bfloat16).cpu().clone(), f"{BIG}/U.pt")
torch.save(meta, f"{BIG}/meta.pt")
all_ids = [k["ids"] for k in kept]
need = [l for l in LAYERS if not os.path.exists(f"{BIG}/X_L{l}.pt")]
out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16) for l in need}
groups = {}
for i, s in enumerate(all_ids):
    groups.setdefault(len(s), []).append(i)


def one(sub):
    ids = torch.tensor([all_ids[i] for i in sub], device="cuda")
    rec = {}
    hk = [model.layers[l].register_forward_hook(
        (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))(l)) for l in need]
    try:
        model.forward(ids)
    finally:
        for h in hk:
            h.remove()
    return {l: rec[l].to(torch.float16).cpu() for l in need}


t1 = time.time()
for L, idx in sorted(groups.items()):
    for b0 in range(0, len(idx), 128):
        sub = idx[b0:b0 + 128]
        r = retry(one, sub)
        for l in need:
            out[l][torch.tensor(sub)] = r[l]
log(f"forward done {time.time() - t1:.0f}s")
for l in need:
    torch.save(out[l], f"{BIG}/X_L{l}.pt")
log("saved", BIG)
