"""e3: refined elicitation-distilled map B (pre-specified design; model as argv[1]).

1. Format choice on 1000 held-out prefixes (rows 1000:2000 = e2_fit's test slice): share of prefixes whose unmasked
   top-1 next token is a content word (optional leading space then letters only; no specials).
   Families: (a) the three e2 few-shot formats; (b) same, with the most common whitespace top-1 token id appended
   when (a)'s top-1 is whitespace for > 50% of prefixes (per format); (c) three chat-template questions
   (thinking disabled, first assistant token). Family share = mean of its three formats.
2. Target per prefix: mean of the three next-token distributions of the chosen family -> top-20 content-word
   tokens -> renormalise -> t = sum p(w) u_w (u_w = unit-norm unembedding row). Centred with the train mean.
3. Ridge per readout layer (X = last-token residual of the prefix, centred) -> centred t, no bias.
   c by val (rows 0:1000) top-10 agreement with the elicited top-1 content word; reported on test (1000:2000).
   to_final(t) = t / g (g = gain of the final RMSNorm), so unembed(to_final(t)) = W_U t / rms(t/g).
Saves /workspace/o-lens-improvement/.hf27/e3/B_<tag>{,_shuf}.pt and copies them (cat) to runs/e3/.
"""
import json
import re
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

R = "/workspace/o-lens-improvement"
M = sys.argv[1]
TAG = "27b" if "27B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "27b": [8, 16, 24, 32, 40, 48, 56]}[TAG]
JPP_PATH = {"9b": "artifacts/baselines/jpp9b_prefit.pt", "27b": "artifacts/baselines/jpp27b_released.pt"}[TAG]
ACTS = {"9b": "/root/e1/acts.pt", "27b": f"{R}/.hf27/m27/acts_27b.pt"}[TAG]
OUT = f"{R}/.hf27/e3"
GRID = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
NV, NT, KTOP = 1000, 1000, 20
BS = 64
FEW = [
    ("Passage: The tide pulled the little boat away from the dock while the fisherman untangled his nets.\n"
     "One-word topic: fishing\n\n"
     "Passage: The senate voted late on Tuesday to approve the new budget, despite\n"
     "One-word topic: politics\n\n"
     "Passage: Elle a sorti la farine, les œufs et le beurre pour préparer\n"
     "One-word topic: baking\n\n"
     "Passage: Lily saw the old dog shivering in the rain, so she opened the door and\n"
     "One-word topic: kindness\n\n"
     "Passage: ", "\nOne-word topic:"),
    ("Text: Ash and molten rock poured down the slopes as the villagers fled toward the coast.\n"
     "Key concept (one word): volcano\n\n"
     "Text: The company reported a sharp rise in quarterly profits after it cut\n"
     "Key concept (one word): business\n\n"
     "Text: Der Arzt hörte das Herz des Kindes ab und\n"
     "Key concept (one word): medicine\n\n"
     "Text: Tom put on his coat and boots and stepped out into the deep white\n"
     "Key concept (one word): snow\n\n"
     "Text: ", "\nKey concept (one word):"),
    ("Name the main subject of each text in one word.\n\n"
     "The orchestra tuned their instruments as the conductor raised his baton.\n"
     "=> Subject: music\n\n"
     "In 1492 the ships left the port of Palos and sailed west into\n"
     "=> Subject: exploration\n\n"
     "O jogador recebeu o cartão vermelho e saiu de campo\n"
     "=> Subject: football\n\n"
     "She typed the last line of code, pressed enter, and watched the program\n"
     "=> Subject: programming\n\n", "\n=> Subject:"),
]
INSTR = ["Answer with exactly one word naming the topic of the text above.",
         "Answer with exactly one word naming the key concept of the text above.",
         "Answer with exactly one word naming the subject of the text above."]
CONTENT_RE = re.compile(r"^ ?[^\W\d_]+$")
torch.set_grad_enabled(False)
dev = "cuda"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


data = json.load(open(f"{R}/runs/e1/prefixes.json"))
N = len(data)
model = get_hf_model(M, attn_implementation="sdpa")
tok = model.tokenizer
d = model.d_model
last = len(model.layers) - 1
U = model._lm_head.weight  # [V, d] (bf16), W_U
V = model.unembed(torch.zeros(1, d, device=dev)).shape[-1]
assert U.shape[0] == V
strs = []
for i in range(V):
    try:
        strs.append(tok.decode([i]))
    except Exception:
        strs.append("")
special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
content = torch.tensor([bool(CONTENT_RE.match(s)) and i not in special for i, s in enumerate(strs)], device=dev)
wspace = torch.tensor([s != "" and s.strip() == "" for s in strs], device=dev)
norm_s = [s.strip().lower() for s in strs]
vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
log(f"model {M} d={d} V={V} N={N}; content-word tokens {int(content.sum())}")


def run_group(seqs, bs):
    res = []
    for b0 in range(0, len(seqs), bs):
        ids = torch.tensor(seqs[b0:b0 + bs], device=dev)
        rec = {}
        hk = model.layers[last].register_forward_hook(
            lambda m, i, oo: rec.__setitem__("y", (oo if torch.is_tensor(oo) else oo[0])[:, -1]))
        try:
            model.forward(ids)
        finally:
            hk.remove()
        res.append(rec["y"].to(torch.float16))
    return torch.cat(res)


def final_states(all_ids, bs=BS):
    """last-block output (pre final norm) at the last token of each sequence; batched by exact length."""
    out = torch.zeros(len(all_ids), d, dtype=torch.float16, device=dev)
    groups = {}
    for i, s in enumerate(all_ids):
        groups.setdefault(len(s), []).append(i)
    for L, idx in sorted(groups.items()):
        out[torch.tensor(idx, device=dev)] = run_group([all_ids[i] for i in idx], bs)
    return out


def logits_of(Y, bs=200):
    for i in range(0, Y.shape[0], bs):
        yield i, model.unembed(Y[i:i + bs]).float()


def top1(Y):
    return torch.cat([lg.argmax(-1) for _, lg in logits_of(Y)])


def few_ids(rows, k, extra=None):
    pre, suf = FEW[k]
    out = [tok(pre + data[i]["text"].strip("\n") + suf, add_special_tokens=False).input_ids for i in rows]
    return [x + [extra] for x in out] if extra is not None else out


def chat_ids(rows, k):
    out = []
    for i in rows:
        msgs = [{"role": "user", "content": data[i]["text"].strip("\n") + "\n\n" + INSTR[k]}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out.append(tok(s, add_special_tokens=False).input_ids)
    return out


# ================= 1. format choice on rows 1000:2000
S1 = list(range(NV, NV + NT))
log("== 1. elicitation format: share of 1000 held-out prefixes whose unmasked top-1 is a content word")
shares, recipe = {}, {}


def describe(name, t1):
    c = Counter(strs[t] for t in t1.tolist())
    cs, ws = content[t1].float().mean().item(), wspace[t1].float().mean().item()
    log(f"   {name}: content {cs:.3f} whitespace {ws:.3f}; top-1 most common {c.most_common(8)}")
    return cs, ws


fam_a = []
for k in range(3):
    t1 = top1(final_states(few_ids(S1, k)))
    cs, ws = describe(f"(a) few-shot E{k}", t1)
    fam_a.append((cs, ws, t1))
shares["a"] = [x[0] for x in fam_a]
recipe["a"] = [("few", k, None) for k in range(3)]
recipe["b"], shares["b"] = [], []
for k in range(3):
    cs, ws, t1 = fam_a[k]
    if ws > 0.5:
        wsc = Counter(t for t in t1.tolist() if wspace[t])
        w = wsc.most_common(1)[0][0]
        t1b = top1(final_states(few_ids(S1, k, w)))
        csb, _ = describe(f"(b) few-shot E{k} + {strs[w]!r} (id {w})", t1b)
        recipe["b"].append(("few", k, w))
        shares["b"].append(csb)
    else:
        log(f"   (b) few-shot E{k}: (a) top-1 whitespace share {ws:.3f} <= 0.5 -> unchanged from (a)")
        recipe["b"].append(("few", k, None))
        shares["b"].append(cs)
recipe["c"], shares["c"] = [("chat", k, None) for k in range(3)], []
ex = chat_ids(S1[:1], 0)[0]
log("   example chat prompt:", repr(tok.decode(ex)[-200:]))
for k in range(3):
    t1 = top1(final_states(chat_ids(S1, k)))
    cs, _ = describe(f"(c) chat Q{k}", t1)
    shares["c"].append(cs)
fam = max("abc", key=lambda f: (sum(shares[f]) / 3, -"abc".index(f)))
log("family shares (mean of 3): " + "  ".join(f"({f}) {sum(shares[f]) / 3:.3f} [{' '.join(f'{x:.3f}' for x in shares[f])}]"
                                            for f in "abc") + f"  -> chosen ({fam})")


def ids_for(rows, r):
    kind, k, extra = r
    return few_ids(rows, k, extra) if kind == "few" else chat_ids(rows, k)


# ================= 2. targets for all prefixes
log("== 2. targets: chosen family on all", N, "prefixes")
Ys = []
for r in recipe[fam]:
    Ys.append(final_states(ids_for(range(N), r)))
    log("   collected", r)
rown = torch.cat([U[i:i + 8192].float().norm(dim=-1) for i in range(0, V, 8192)])
T = torch.zeros(N, d, device=dev)
TID = torch.zeros(N, KTOP, dtype=torch.long, device=dev)
TP = torch.zeros(N, KTOP, device=dev)
mass = torch.zeros(N, device=dev)
for i in range(0, N, 200):
    P = sum(model.unembed(Y[i:i + 200]).float().softmax(-1) for Y in Ys) / len(Ys)
    mass[i:i + 200] = (P * content).sum(-1)
    v, ix = P.masked_fill(~content, -1.0).topk(KTOP, -1)
    p = v / v.sum(-1, keepdim=True)
    u = U[ix].float() / rown[ix][..., None]
    T[i:i + 200] = (p[..., None] * u).sum(1)
    TID[i:i + 200], TP[i:i + 200] = ix, p
del Ys
torch.cuda.empty_cache()
word = TID[:, 0]
torch.save(dict(TID=TID.cpu(), TP=TP.cpu(), fam=fam, recipe=recipe[fam], shares=shares), f"{OUT}/targets_{TAG}.pt")
va, te, tr = slice(0, NV), slice(NV, NV + NT), slice(NV + NT, N)
c = Counter(norm_s[t] for t in word[tr].tolist())
log(f"   mean content-word mass of the averaged distribution {mass.mean():.3f}; mean top-1 renormalised p {TP[:, 0].mean():.3f}")
log(f"   distinct elicited words (train) {len(c)} of {N - NV - NT}; most common: {c.most_common(25)}")
for i in range(NV, NV + 12):
    log(f"   [{data[i]['src']}] ...{data[i]['text'][-90:]!r} -> " +
        ", ".join(f"{strs[t].strip()}:{p:.2f}" for t, p in zip(TID[i, :5].tolist(), TP[i, :5].tolist())))

# ================= to_final check
fn = model._final_norm
g = fn(torch.ones(1, d, device=dev, dtype=torch.float32)).float()[0]
log(f"== to_final: final-norm gain g: min {g.min():.4f} max {g.max():.4f} mean {g.mean():.4f}; "
    f"|g|<0.05: {int((g.abs() < 0.05).sum())}; negative: {int((g < 0).sum())}")
ginv = 1.0 / g
tt = T[NV:NV + 100]
a = model.unembed(tt * ginv).float().topk(10, -1).indices
b = (tt.to(U.dtype) @ U.T).float().topk(10, -1).indices
ov = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 10 for x, y in zip(a, b)])
log(f"   unembed(to_final(t)) vs W_U t on 100 held-out targets: top-10 overlap {ov.mean():.3f} (min {ov.min():.1f}); "
    f"top-1 equal {(a[:, 0] == b[:, 0]).float().mean():.3f}")

# ================= 3. ridge maps
JPP = {int(l): v.to(dev).float() for l, v in
       load_lens_file(f"{R}/{JPP_PATH}", hf_model_name=M).jacobians_L_dict_FN.items()}
EA = torch.load(f"{R}/runs/e1/E_{TAG}.pt", map_location="cpu")
log("loading activations", ACTS)
X_all = torch.load(ACTS)["X"]


def rank_top(lg_fn, v, k=10, bs=250):
    out = []
    for i in range(0, v.shape[0], bs):
        lg = lg_fn(v[i:i + bs]).float()
        out.append(lg.masked_fill(~content, float("-inf")).topk(k, -1).indices)
    return torch.cat(out)


def top_WU(v):
    return rank_top(lambda x: x.to(U.dtype) @ U.T, v)


def top_unembed(v):
    return rank_top(lambda x: model.unembed(x), v)


def agree(top, w):
    return (NORM[top] == NORM[w][:, None]).any(-1)


class Ridge:
    def __init__(self, X):
        self.mx = X.mean(0)
        self.Xc = (X - self.mx).double()
        s, Q = torch.linalg.eigh(self.Xc.T @ self.Xc)
        self.s, self.Q = s.clamp_min(0), Q
        self.scale = self.s.mean().item()

    def setY(self, Yc):
        self.QtB = self.Q.T @ (self.Xc.T @ Yc.double())

    def W(self, c):
        return (self.Q @ (self.QtB / (self.s + c * self.scale)[:, None])).float()

    def predict(self, X, W):
        return (X - self.mx) @ W  # centred target, no bias


def pack(r, W, c):
    sc = W.abs().max().item()
    return dict(mx=r.mx.cpu(), W=(W / sc).half().cpu(), scale=sc, ginv=ginv.cpu(), c=c)


mT = T[tr].mean(0)
Tc = T - mT
perm = torch.randperm(N - NV - NT, generator=torch.Generator().manual_seed(1)).to(dev)
maps = {"main": {}, "shuf": {}}
rows = []
for l in LAYERS:
    X = X_all[l].float().to(dev)
    r = Ridge(X[tr])
    r.setY(Tc[tr])
    sc = [agree(top_WU(r.predict(X[va], r.W(c_))), word[va]).float().mean().item() for c_ in GRID]
    bi = max(range(len(GRID)), key=lambda i: (sc[i], i))
    cbest = GRID[bi]
    W = r.W(cbest)
    tB = top_WU(r.predict(X[te], W))
    maps["main"][l] = pack(r, W, cbest)
    r.setY(Tc[tr][perm])
    Ws = r.W(cbest)
    tS = top_WU(r.predict(X[te], Ws))
    maps["shuf"][l] = pack(r, Ws, cbest)
    ea = EA[l]
    WA = ea["W"].to(dev).float() * ea["scale"]
    tA = top_unembed((X[te] - ea["mx"].to(dev)) @ WA + ea["my"].to(dev))
    tJ = top_unembed(X[te] @ JPP[l].T)
    res = {n: (agree(t_, word[te]).float().mean().item(), agree(t_[:, :1], word[te]).float().mean().item())
           for n, t_ in (("B", tB), ("A", tA), ("J++", tJ), ("B_shuf", tS))}
    rows.append((l, cbest, res))
    log(f"L{l:2d} c={cbest:g} val@10 by c: " + " ".join(f"{a_:g}:{b_:.3f}" for a_, b_ in zip(GRID, sc)))
    del r, X, W, Ws, WA
    torch.cuda.empty_cache()

log(f"== held-out agreement with the elicited top-1 content word (test {NT} prefixes; word-level, content-word mask)")
print("layer | c | B top10 | A top10 | J++ top10 | B_shuf top10 | B top1 | A top1 | J++ top1 | B_shuf top1")
for l, c_, res in rows:
    print(f"L{l} | {c_:g} | " + " | ".join(f"{res[n][0]:.3f}" for n in ("B", "A", "J++", "B_shuf")) + " | " +
          " | ".join(f"{res[n][1]:.3f}" for n in ("B", "A", "J++", "B_shuf")), flush=True)
pt = top_WU(mT[None].expand(NT, -1).contiguous())
log(f"(reference) uncentred prior W_U mean(t) top10 {agree(pt, word[te]).float().mean():.3f}")
import os  # noqa: E402

os.makedirs(f"{R}/runs/e3", exist_ok=True)
for v, suf in (("main", ""), ("shuf", "_shuf")):
    src = f"{OUT}/B_{TAG}{suf}.pt"
    torch.save(maps[v], src)
    with open(src, "rb") as a_, open(f"{R}/runs/e3/B_{TAG}{suf}.pt", "wb") as b_:
        while True:
            chunk = a_.read(1 << 24)
            if not chunk:
                break
            b_.write(chunk)
log("saved")
