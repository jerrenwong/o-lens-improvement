"""g1 step 3: filter generated sentences; collect X (last-token residual, sentence run alone) at the 7 readout
layers, the "asked" rank of the concept (few-shot E0 one-word topic format, content tokens only; diagnostic only),
and the target t_w (mean unit unembedding row of the word's single-token surface forms, in the final basis t/g).
Output: /workspace/o-lens-improvement/.hf27/g1/acts.pt
"""
import json
import random
import re
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
OUT = f"{R}/.hf27/g1/acts.pt"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
ASK = ("Passage: The tide pulled the little boat away from the dock while the fisherman untangled his nets.\n"
       "One-word topic: fishing\n\n"
       "Passage: The senate voted late on Tuesday to approve the new budget, despite\n"
       "One-word topic: politics\n\n"
       "Passage: Elle a sorti la farine, les œufs et le beurre pour préparer\n"
       "One-word topic: baking\n\n"
       "Passage: Lily saw the old dog shivering in the rain, so she opened the door and\n"
       "One-word topic: kindness\n\n"
       "Passage: ", "\nOne-word topic:")
CONTENT_RE = re.compile(r"^ ?[^\W\d_]+$")
END_RE = re.compile(r"[.!?][\"”’']?(?=\s|$)")
torch.set_grad_enabled(False)
dev = "cuda"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def cut(text):
    s = text.split("\n")[0].strip()
    m = END_RE.search(s)
    return s[:m.end()] if m else None


def names(sent, w):
    w = re.escape(w.lower())
    forms = [w, w + "s", w + "es", w + "ed", w + "ing", w + "'s", w + "’s"]
    s = sent.lower()
    if re.search(r"\b(" + "|".join(forms) + r")\b", s):
        return True
    wl = w.lower()
    if len(wl) >= 4 and wl[-1] in "es" and re.search(r"\b" + wl[:-1], s):
        return True
    return False


concepts = {c["word"]: c for c in json.load(open(f"{R}/runs/g1/concepts.json"))}
raw = [json.loads(l) for l in open("/root/g1/gen_raw.jsonl")]
model = get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa")
tok, hf = model.tokenizer, model._hf_model
d, last = model.d_model, len(model.layers) - 1
seen, kept, drops = set(), [], Counter()
for o in raw:
    key = (o["word"], o["style"])
    if key in seen:
        continue
    seen.add(key)
    s = cut(o["text"])
    if s is None:
        drops["no sentence end"] += 1
        continue
    if names(s, o["word"]):
        drops["contains word/inflection"] += 1
        continue
    ids = tok(s, add_special_tokens=False).input_ids
    if len(ids) < 6:
        drops["< 6 tokens"] += 1
        continue
    kept.append(dict(word=o["word"], style=o["style"], sent=s, ids=ids, tid=concepts[o["word"]]["tid"]))
log(f"generated {len(seen)} (raw lines {len(raw)}); kept {len(kept)}; drops {dict(drops)}")
log(f"kept by style: {Counter(k['style'] for k in kept)}; concepts with >=1 kept: {len({k['word'] for k in kept})}")
rng = random.Random(0)
log("30 random kept (word | style | sentence):")
for k in rng.sample(kept, 30):
    print(f"   {k['word']:>14s} | s{k['style'] + 1} | {k['sent']}")
dr = [o for o in raw if cut(o["text"]) and names(cut(o["text"]), o["word"])]
log("8 random dropped for naming the word:")
for o in rng.sample(dr, min(8, len(dr))):
    print(f"   {o['word']:>14s} | {cut(o['text'])}")

# ---- convention check
for k in kept[:3]:
    ids = torch.tensor([k["ids"]], device=dev)
    hs = hf(ids, output_hidden_states=True).hidden_states
    rec = {}
    hk = [model.layers[l].register_forward_hook(
        (lambda l: lambda m, i, oo: rec.__setitem__(l, oo if torch.is_tensor(oo) else oo[0]))(l)) for l in (12, 24)]
    model.forward(ids)
    for h in hk:
        h.remove()
    log("check cos(block l out, hidden_states[l+1]) " + " ".join(
        f"L{l} {torch.cosine_similarity(rec[l][0, -1].float(), hs[l + 1][0, -1].float(), 0).item():.4f}" for l in (12, 24)))


def batched(all_ids, pos_layers, bs):
    out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16, device=dev) for l in pos_layers}
    groups = {}
    for i, s in enumerate(all_ids):
        groups.setdefault(len(s), []).append(i)
    for L, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), bs):
            sub = idx[b0:b0 + bs]
            ids = torch.tensor([all_ids[i] for i in sub], device=dev)
            rec = {}
            hk = [model.layers[l].register_forward_hook(
                (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))(l))
                for l in pos_layers]
            try:
                model.forward(ids)
            finally:
                for h in hk:
                    h.remove()
            ix = torch.tensor(sub, device=dev)
            for l in pos_layers:
                out[l][ix] = rec[l].to(torch.float16)
    return out


N = len(kept)
X = batched([k["ids"] for k in kept], LAYERS, 128)
log("X done")
ask_ids = [tok(ASK[0] + k["sent"] + ASK[1], add_special_tokens=False).input_ids for k in kept]
Y = batched(ask_ids, [last], 64)[last]
log("asked forward done")

V = model.unembed(torch.zeros(1, d, device=dev)).shape[-1]
strs = []
for i in range(V):
    try:
        strs.append(tok.decode([i]))
    except Exception:
        strs.append("")
special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
content = torch.tensor([bool(CONTENT_RE.match(s)) and i not in special for i, s in enumerate(strs)], device=dev)
norm_s = [s.strip().lower() for s in strs]
ask_rank = torch.zeros(N, dtype=torch.long)
ask_top = []
for i in range(0, N, 200):
    lg = model.unembed(Y[i:i + 200]).float().masked_fill(~content, float("-inf"))
    for r in range(lg.shape[0]):
        w = kept[i + r]["word"].lower()
        enc = tok([" " + w, w, " " + w.capitalize(), w.capitalize()], add_special_tokens=False).input_ids
        same = {kept[i + r]["tid"]} | {e[0] for e in enc if len(e) == 1}
        same = torch.tensor(sorted(same), device=dev)
        best = lg[r, same].max()
        ask_rank[i + r] = int((lg[r] > best).sum()) + 1
    ask_top += lg.argmax(-1).tolist()
for k_ in (1, 10, 100):
    log(f"asked: share with concept rank <= {k_}: {(ask_rank <= k_).float().mean():.3f}")
for s_ in (0, 1):
    m = torch.tensor([k["style"] == s_ for k in kept])
    log(f"   style {s_ + 1}: <=1 {(ask_rank[m] <= 1).float().mean():.3f} <=10 {(ask_rank[m] <= 10).float().mean():.3f} "
        f"<=100 {(ask_rank[m] <= 100).float().mean():.3f}")
log("10 examples word -> asked top-1: " + ", ".join(f"{kept[i]['word']}->{strs[ask_top[i]].strip()}" for i in range(0, N, N // 10)))

# ---- targets
U = model._lm_head.weight
fn = model._final_norm
g = fn(torch.ones(1, d, device=dev, dtype=torch.float32)).float()[0]
log(f"final-norm gain g: min {g.min():.4f} max {g.max():.4f} |g|<0.05: {int((g.abs() < 0.05).sum())} neg {int((g < 0).sum())}")
tcache, nforms = {}, []
T = torch.zeros(N, d, device=dev)
for i, k in enumerate(kept):
    w = k["word"]
    if w not in tcache:
        forms = {" " + w, w, " " + w.lower(), w.lower(), " " + w.lower().capitalize(), w.lower().capitalize()}
        ids = sorted({t[0] for t in tok(sorted(forms), add_special_tokens=False).input_ids if len(t) == 1})
        u = U[torch.tensor(ids, device=dev)].float()
        tcache[w] = ((u / u.norm(dim=-1, keepdim=True)).mean(0), len(ids))
    T[i] = tcache[w][0]
    nforms.append(tcache[w][1])
log(f"surface forms per concept: {Counter(tcache[w][1] for w in tcache)}")
Tf = T / g
tt = T[:100]
a = model.unembed(Tf[:100]).float().topk(10, -1).indices
b = (tt.to(U.dtype) @ U.T).float().topk(10, -1).indices
ov = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 10 for x, y in zip(a, b)])
log(f"to_final check on 100: top-10 overlap unembed(t/g) vs W_U t {ov.mean():.3f} (min {ov.min():.1f}); "
    f"top-1 equal {(a[:, 0] == b[:, 0]).float().mean():.3f}; top-1 is a surface form of the concept "
    f"{sum(norm_s[a[i, 0]] == kept[i]['word'].lower() for i in range(100)) / 100:.2f}")
torch.save(dict(X={l: X[l].cpu() for l in LAYERS}, T=Tf.cpu(), words=[k["word"] for k in kept],
                styles=[k["style"] for k in kept], sents=[k["sent"] for k in kept], tid=[k["tid"] for k in kept],
                ask_rank=ask_rank, drops=dict(drops), n_gen=len(seen)), OUT)
log("saved", OUT)
