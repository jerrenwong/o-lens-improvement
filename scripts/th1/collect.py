"""th1 step 2: filter (g1 filter + near-dup), keep <=50/word (35 train / 15 test), collect last-token residual at
layers 0..30 (sentence alone; resid[L] = output of block L = hidden_states[L+1]), asked logits over the 200 words,
and per-word token data (surface forms, unembedding rows, input embeddings, final-norm gain)."""
import json, os, random, re, sys
from collections import Counter, defaultdict
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from th1_common import BIG, ASK, log, retry, load_model, cut, names, forms_of

torch.set_grad_enabled(False)
OUT = f"{BIG}/acts.pt"
HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(f"{HERE}/words.json"))
words = [o["word"] for o in W]
widx = {w: i for i, w in enumerate(words)}
raw = [json.loads(l) for l in open(f"{BIG}/gen_raw.jsonl")]
model = load_model()
tok, hf = model.tokenizer, model._hf_model
d, NL = model.d_model, len(model.layers)
LAY = list(range(NL - 1))  # 0..30
log(f"model layers {NL}; collecting {LAY[0]}..{LAY[-1]}")

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
kept, per_word = [], {}
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
        if len(mine) == 50:
            break
    n = len(mine)
    nte = 15 if n >= 50 else round(0.3 * n)
    idx = list(range(n)); rng.shuffle(idx)
    te = set(idx[:nte])
    for i, (x, st, _) in enumerate(mine):
        kept.append(dict(word=w, wi=widx[w], style=st, sent=x[0], ids=x[1], split="test" if i in te else "train"))
    per_word[w] = n
log(f"raw {len(raw)}; drops {dict(drops)}; kept {len(kept)}")
log(f"kept per word distribution: {sorted(Counter(per_word.values()).items())}")
log(f"style split: {Counter(k['style'] for k in kept)}; split: {Counter(k['split'] for k in kept)}")
for k in random.Random(1).sample(kept, 12):
    print(f"   {k['word']:>12s} s{k['style'] + 1} | {k['sent']}")


def batched(all_ids, layers, bs, pos=None):
    out = torch.zeros(len(all_ids), len(layers), d, dtype=torch.float16)
    groups = {}
    for i, s in enumerate(all_ids):
        groups.setdefault(len(s), []).append(i)

    def one(sub):
        ids = torch.tensor([all_ids[i] for i in sub], device="cuda")
        rec = {}
        hk = [model.layers[l].register_forward_hook(
            (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))(l))
            for l in layers]
        try:
            model.forward(ids)
        finally:
            for h in hk:
                h.remove()
        return torch.stack([rec[l] for l in layers], 1).to(torch.float16).cpu()

    for L, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), bs):
            sub = idx[b0:b0 + bs]
            out[torch.tensor(sub)] = retry(one, sub)
    return out


# convention check: hook output == hidden_states[L+1]
ids = torch.tensor([kept[0]["ids"]], device="cuda")
hs = hf(ids, output_hidden_states=True).hidden_states
x0 = batched([kept[0]["ids"]], [0, 12, 30], 1)[0].float()
log("check cos(hook out L, hidden_states[L+1]): " + " ".join(
    f"L{l} {torch.cosine_similarity(x0[j], hs[l + 1][0, -1].float().cpu(), 0).item():.4f}" for j, l in enumerate([0, 12, 30])))

X = batched([k["ids"] for k in kept], LAY, 64)
log("X done", tuple(X.shape))

# per-word token data
U = model._lm_head.weight
E = hf.get_input_embeddings().weight
g = model._final_norm(torch.ones(1, d, device="cuda", dtype=torch.float32)).float()[0].cpu()
form_ids, form_word, uX, eX = [], [], [], []
for w in words:
    f = forms_of(tok, w)
    assert len(tok(" " + w, add_special_tokens=False).input_ids) == 1, w
    form_ids += f; form_word += [widx[w]] * len(f)
    u = U[torch.tensor(f, device="cuda")].float()
    uX.append((u / u.norm(dim=-1, keepdim=True)).mean(0).cpu())
    eX.append(E[torch.tensor(f, device="cuda")].float().mean(0).cpu())
log(f"surface forms per word: {sorted(Counter(Counter(form_word).values()).items())}")
fid = torch.tensor(form_ids, device="cuda"); fw = torch.tensor(form_word, device="cuda")
Uf = U[fid].float().cpu()

# asked: few-shot one-word topic question, final logits restricted to the 200 words (best surface form)
a0 = tok(ASK[0], add_special_tokens=False).input_ids; a1 = tok(ASK[1], add_special_tokens=False).input_ids
ask_ids = [a0 + k["ids"] + a1 for k in kept]
Y = batched(ask_ids, [NL - 1], 32)[:, 0]
ask_lg = torch.zeros(len(kept), len(words))
for i in range(0, len(kept), 256):
    lg = model.unembed(Y[i:i + 256].cuda()).float()[:, fid]
    ask_lg[i:i + 256] = torch.full((lg.shape[0], len(words)), -1e9, device="cuda").scatter_reduce(
        1, fw[None].expand(lg.shape[0], -1), lg, "amax").cpu()
wi = torch.tensor([k["wi"] for k in kept])
rank = (ask_lg > ask_lg[torch.arange(len(kept)), wi][:, None]).sum(1) + 1
for s in ("train", "test"):
    m = torch.tensor([k["split"] == s for k in kept])
    log(f"asked ({s}): top-1 among 200 {(rank[m] <= 1).float().mean():.3f}  top-5 {(rank[m] <= 5).float().mean():.3f}")
torch.save(dict(X=X, layers=LAY, words=words, kinds=[o["kind"] for o in W], wi=wi,
                split=[k["split"] for k in kept], style=[k["style"] for k in kept], sents=[k["sent"] for k in kept],
                ids=[k["ids"] for k in kept], ask_lg=ask_lg, form_ids=form_ids, form_word=form_word, Uf=Uf,
                uX=torch.stack(uX), eX=torch.stack(eX), g=g, per_word=per_word, drops=dict(drops)), OUT)
log("saved", OUT)
