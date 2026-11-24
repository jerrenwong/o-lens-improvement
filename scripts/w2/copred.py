"""W2b step 1: co-prediction statistics.  Run the model over ordinary text (WikiText-103 train, paragraph-start windows of
128 tokens via olens.train.wikitext_windows, plus the A-lens passages runs/e1/prefixes.json), take the top-20 next-token
tokens and probabilities at every position (BOS position excluded from nothing: every position predicts a next token),
cut them (variants below), and accumulate C[X, Y] = sum_t pt(X) pt(Y) over all positions (including X = Y).
Cut variants (all inside the model's top 20; renormalised after the cut):
   k5     top-5 at T = 1           k20    top-20 at T = 1
   p09    top-p 0.9 (smallest prefix of the sorted top-20 with mass >= 0.9 of the top-20 mass), T = 1
   t05    top-20 with temperature 0.5 (p^2 renormalised)
   words  top-20 with non-word tokens (no letter/digit) and special/whitespace tokens dropped, T = 1
Saved per variant .hf27/w2/copred_<tag>_<variant>.pt: idx [V, 50] int32 (top-50 partners Y != X by C, -1 pad),
w [V, 50] float32 (C[X, Y]), rowsum [V] (sum over ALL Y != X), self [V] (C[X, X]), mass [V] (sum_t pt(X)),
n_tokens, plus speed / memory in the json next to it.
Usage: python copred.py <model> [--tokens 2000000] [--smoke]"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tokens", type=int, default=2_000_000)
ap.add_argument("--passage_tokens", type=int, default=800_000, help="taken from the A-lens passages first")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.tokens, args.passage_tokens = 40_000, 15_000
TAG = "27b" if "27B" in args.model else ("2b" if "2B" in args.model else "9b")
OUT = Path(f"{R}/.hf27/w2" + ("/smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
VARS = ["k5", "k20", "p09", "t05", "words"]
TOPN, KEEP = 20, 50
dev = "cuda"
torch.set_grad_enabled(False)
from workspace_lens import get_hf_model  # noqa: E402
from olens.train import wikitext_windows  # noqa: E402

t_load = time.time()
model = get_hf_model(args.model, attn_implementation="sdpa")
hf = model._hf_model
tok = model.tokenizer
V = model.unembed(torch.zeros(1, model.d_model, device=dev)).shape[-1]
strs = [tok.decode([i]) for i in range(V)]
special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
wordtok = torch.tensor([bool(re.search(r"[^\W_]", s)) for s in strs], device=dev)  # letter or digit
wordtok[[i for i in special if i < V]] = False
wordtok &= torch.tensor([s.strip() != "" for s in strs], device=dev)
bos = tok.bos_token_id
print(f"[copred] {args.model}: V {V}, word tokens {int(wordtok.sum())}, bos {bos!r}; loaded in "
      f"{time.time() - t_load:.0f}s", flush=True)

# ------------------------------------------------------------------ data: passages (grouped by length) + wikitext
data = json.load(open(f"{R}/runs/e1/prefixes.json"))
seqs, n_p = [], 0
for o in data:
    s = ([bos] if bos is not None else []) + list(o["ids"])
    seqs.append(s); n_p += len(s)
    if n_p >= args.passage_tokens:
        break
n_wiki = max(0, (args.tokens - n_p) // 128 + 1)
wiki = wikitext_windows(tok, "train", n_wiki, 128 - (bos is not None), Path(f"{R}/.hf27/w2/cache"), para_start=True)
wiki = [([bos] if bos is not None else []) + w for w in wiki.tolist()]
print(f"[copred] passages {len(seqs)} ({n_p} tokens), wikitext windows {len(wiki)} ({sum(map(len, wiki))} tokens)",
      flush=True)
# fixed shapes (the torch fallback of the linear-attention layers is slow on every new shape): passages are
# RIGHT-padded to 48 tokens (causal model: real positions are unaffected); padded positions are not counted.
PADL = max(map(len, seqs))
pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
groups = {PADL: [(s + [pad_id] * (PADL - len(s)), len(s)) for s in seqs], 128: [(w, len(w)) for w in wiki]}


def cuts(lp):
    """lp: [P, 20] log-probs (full-vocab softmax) sorted descending, ids [P, 20] -> weights [P, 20, 5]."""
    p = lp.exp()
    out = torch.zeros(*p.shape, len(VARS), device=dev)
    k5 = p.clone(); k5[:, 5:] = 0
    out[..., 0] = k5 / k5.sum(-1, keepdim=True)
    out[..., 1] = p / p.sum(-1, keepdim=True)
    q = p / p.sum(-1, keepdim=True)
    keep = (q.cumsum(-1) - q) < 0.9  # smallest prefix reaching 0.9
    pp = q * keep
    out[..., 2] = pp / pp.sum(-1, keepdim=True)
    t = p.pow(2)
    out[..., 3] = t / t.sum(-1, keepdim=True)
    return out


keys_acc = torch.zeros(0, dtype=torch.long, device=dev)
w_acc = torch.zeros(0, len(VARS), device=dev)
mass = torch.zeros(V, len(VARS), device=dev, dtype=torch.float64)
buf_k, buf_w, buf_n = [], [], 0


def coalesce(keys, w):
    u, inv = torch.unique(keys, return_inverse=True)
    s = torch.zeros(len(u), w.shape[1], device=dev).index_add_(0, inv, w)
    return u, s


def flush():
    global keys_acc, w_acc, buf_k, buf_w, buf_n
    if not buf_k:
        return
    keys_acc, w_acc = coalesce(torch.cat([keys_acc] + buf_k), torch.cat([w_acc] + buf_w))
    buf_k, buf_w, buf_n = [], [], 0


torch.cuda.reset_peak_memory_stats()
TIME = {"fwd": 0.0, "acc": 0.0}
t0, n_tok = time.time(), 0
for L, ss in sorted(groups.items()):
    for b0 in range(0, len(ss), args.bs):
        chunk = ss[b0:b0 + args.bs]
        ids = torch.tensor([c[0] for c in chunk], device=dev)
        valid = (torch.arange(ids.shape[1], device=dev)[None] < torch.tensor([c[1] for c in chunk], device=dev)[:, None])
        torch.cuda.synchronize(); ta = time.time()
        lg = hf(input_ids=ids).logits[:, :, :V].float()  # [B, L, V]
        lp, ii = torch.log_softmax(lg, -1).topk(TOPN, -1)
        del lg
        torch.cuda.synchronize(); tb = time.time(); TIME["fwd"] += tb - ta
        lp, ii = lp[valid], ii[valid]
        w = cuts(lp)  # [P, 20, 4 of 5]
        wd = lp.exp() * wordtok[ii]
        w[..., 4] = wd / wd.sum(-1, keepdim=True).clamp_min(1e-30)
        w = torch.nan_to_num(w)
        mass.index_add_(0, ii.reshape(-1), w.reshape(-1, len(VARS)).double())
        pk = (ii[:, :, None] * V + ii[:, None, :]).reshape(-1)
        pw = (w[:, :, None, :] * w[:, None, :, :]).reshape(-1, len(VARS))
        nz = pw.sum(-1) > 0
        k_, w_ = coalesce(pk[nz], pw[nz])
        buf_k.append(k_); buf_w.append(w_); buf_n += len(k_)
        n_tok += int(valid.sum())
        if buf_n > 60_000_000:
            flush()
        torch.cuda.synchronize(); TIME["acc"] += time.time() - tb
        if b0 % (50 * args.bs) == 0:
            print(f"   {n_tok} tokens, {n_tok / (time.time() - t0):.0f} tok/s, distinct pairs {len(keys_acc)}, "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB; fwd {TIME['fwd']:.0f}s acc {TIME['acc']:.0f}s", flush=True)
flush()
el = time.time() - t0
print(f"[copred] {n_tok} tokens in {el:.0f}s ({n_tok / el:.0f} tok/s); distinct pairs {len(keys_acc)}; "
      f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB; time forward {TIME['fwd']:.0f}s, accumulate "
      f"{TIME['acc']:.0f}s", flush=True)

X, Y = keys_acc // V, keys_acc % V
off = X != Y
info = dict(model=args.model, n_tokens=n_tok, seconds=el, tok_per_s=n_tok / el, distinct_pairs=len(keys_acc),
            peak_gb=torch.cuda.max_memory_allocated() / 1e9, passages=len(seqs), wiki_windows=len(wiki))
for v, name in enumerate(VARS):
    w = w_acc[:, v]
    selfw = torch.zeros(V, device=dev).index_add_(0, X[~off], w[~off])
    m = off & (w > 0)
    xs, ys, ws = X[m], Y[m], w[m]
    rowsum = torch.zeros(V, device=dev).index_add_(0, xs, ws)
    o = torch.argsort(ws, descending=True, stable=True)
    xs, ys, ws = xs[o], ys[o], ws[o]
    o = torch.argsort(xs, stable=True)
    xs, ys, ws = xs[o], ys[o], ws[o]
    cnt = torch.bincount(xs, minlength=V)
    start = torch.cumsum(cnt, 0) - cnt
    r = torch.arange(len(xs), device=dev) - start[xs]
    k = r < KEEP
    idx = torch.full((V, KEEP), -1, dtype=torch.int32, device=dev)
    wt = torch.zeros(V, KEEP, device=dev)
    idx[xs[k], r[k]] = ys[k].int()
    wt[xs[k], r[k]] = ws[k]
    torch.save(dict(idx=idx.cpu(), w=wt.cpu(), rowsum=rowsum.cpu(), self=selfw.cpu(), mass=mass[:, v].float().cpu(),
                    n_tokens=n_tok, variant=name, model=args.model), OUT / f"copred_{TAG}_{name}.pt")
    cover = (wt.sum(-1) / rowsum.clamp_min(1e-30))[rowsum > 0]
    info[name] = dict(tokens_with_partners=int((rowsum > 0).sum()), top50_mass_share_mean=cover.mean().item())
    # a few examples
    ex = [" 5", " Tuesday", " Paris", " the", " January", " red"]
    for e in ex:
        t = tok(e, add_special_tokens=False).input_ids
        if len(t) == 1:
            t = t[0]
            part = [(tok.decode([int(y)]), round(float(c / max(rowsum[t].item(), 1e-30)), 3))
                    for y, c in zip(idx[t, :8].tolist(), wt[t, :8].tolist()) if y >= 0]
            print(f"   [{name}] q(.|{e!r}) top: {part}", flush=True)
    print(f"[copred] saved {name}: tokens with partners {info[name]['tokens_with_partners']}, mean top-50 share of "
          f"q mass {info[name]['top50_mass_share_mean']:.3f}", flush=True)
json.dump(info, open(OUT / f"copred_{TAG}.json", "w"), indent=1)
print("done", flush=True)
