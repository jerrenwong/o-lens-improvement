"""Shared code for b3k poetry candidates (round-2 idea 7, rollout-distilled newline map). Not a candidate.

Label-free data, generated inside build() by the model itself:
  1. First lines: few-shot prompt with hand-written couplets (written for this file, not benchmark text),
     sampled (T=1.0), cut at the first newline, deduplicated.
  2. For each first line, context C = "A rhyming couplet:\n" + line1 + "\n" (zero-shot). The model samples
     K second lines from C; the target words are the last words of those lines (single-token " word" only).
  3. x_l = residual at the last token of C (the newline ending line 1), readout layers.
Map (per layer): kernel ridge from centred unit(x_l) to the centred mean effective-unembedding direction
  unit(W[w] * gamma) of the K target words. Lens: unit(JPP_l h) + b * ridge_l(unit(h)).
This is a learned probe of what the model will write (distilled from its own rollouts), not a Jacobian lens.
lam, b chosen on held-out couplets (split by first line) by "modal ending in top 10 at any layer".
"""

import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_common import report, unit  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402

SHOTS = [
    ("The morning sun came creeping through the trees,", "And woke the meadow with a gentle breeze."),
    ("I left my coffee cooling on the sill,", "And watched the rain run down the distant hill."),
    ("My grandpa kept a fiddle by the door,", "He played it till the neighbours asked for more."),
    ("The baker hums while kneading out the dough,", "His little oven glowing soft and low."),
    ("We sailed our paper boat across the pond,", "And waved it off to lands that lie beyond."),
    ("The kitten chased a shadow down the hall,", "Then tumbled in a tangle by the wall."),
    ("A lonely lighthouse watches over the bay,", "And keeps the sailors safe until the day."),
    ("She found a letter hidden in a book,", "And read it slowly in a quiet nook."),
]
HEAD = "A rhyming couplet:\n"


def log(*a):
    print("[b3k]", *a, flush=True)


def _nl_ids(tok):
    ids = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return ids


@torch.no_grad()
def _generate(ctx, prompts, max_new, batch, n_ret=1, temp=1.0, seed=0, stop_nl=True):
    hf, tok = ctx.model._hf_model, ctx.model.tokenizer
    old = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = _nl_ids(tok) if stop_nl else None
    torch.manual_seed(seed)
    out_texts = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(ctx.device)
        out = hf.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=0.95, top_k=0,
                          num_return_sequences=n_ret, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        out_texts += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    tok.padding_side = old
    return out_texts


def _last_word(line):
    ws = re.findall(r"[A-Za-z']+", line)
    if not ws:
        return None
    return ws[-1].strip("'")


def _rhymes(a, b):
    a, b = a.lower(), b.lower()
    return a != b and len(a) >= 2 and len(b) >= 2 and a[-2:] == b[-2:]


def first_lines(ctx, n_prompts=384, batch=64, seed=0):
    rng = random.Random(seed)
    prompts = []
    for _ in range(n_prompts):
        shots = rng.sample(SHOTS, 3)
        prompts.append("".join(f"{HEAD}{a}\n{b}\n\n" for a, b in shots) + HEAD)
    texts = _generate(ctx, prompts, 20, batch, seed=seed)
    seen, lines = set(), []
    shot_lines = {a for a, _ in SHOTS}
    for t in texts:
        if "\n" not in t:  # did not finish the line within 20 tokens
            continue
        line = t.split("\n")[0].strip()
        nw = len(line.split())
        if not (4 <= nw <= 14) or line in seen or line in shot_lines or _last_word(line) is None:
            continue
        seen.add(line)
        lines.append(line)
    return lines


@torch.no_grad()
def newline_resid(ctx, ctxs, batch=64):
    tok, model, layers = ctx.model.tokenizer, ctx.model, ctx.readout_layers
    ids_list = [tok(c, add_special_tokens=False).input_ids for c in ctxs]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = {l: [] for l in layers}
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        T = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, :len(x)] = torch.as_tensor(x)
        last = torch.as_tensor([len(x) - 1 for x in chunk])
        with ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids.to(ctx.device))
            acts = dict(rec.activations)
        for l in layers:
            out[l].append(acts[l][torch.arange(len(chunk)), last.to(acts[l].device)].float().to(ctx.device))
    last_tok = Counter(tok.decode(x[-1:]) for x in ids_list).most_common(4)
    return {l: torch.cat(v) for l, v in out.items()}, last_tok


def build_data(ctx, n_prompts=384, K=4, seed=0):
    t0 = time.time()
    tok = ctx.model.tokenizer
    lines = first_lines(ctx, n_prompts, seed=seed)
    log(f"first lines: {len(lines)} unique of {n_prompts}; {time.time() - t0:.0f}s; e.g. {lines[:3]}")
    ctxs = [f"{HEAD}{l}\n" for l in lines]
    texts = _generate(ctx, ctxs, 20, 32, n_ret=K, seed=seed + 1)
    log(f"second lines sampled ({len(texts)}); {time.time() - t0:.0f}s")
    rows, n_ok, n_rhyme, n_tot = [], 0, 0, 0
    for j, l1 in enumerate(lines):
        w1 = _last_word(l1)
        ws = []
        for t in texts[j * K:(j + 1) * K]:
            if "\n" not in t:
                continue
            l2 = t.split("\n")[0].strip()
            w = _last_word(l2)
            if w is None or len(l2.split()) < 3:
                continue
            n_tot += 1
            n_rhyme += _rhymes(w1, w)
            ids = tok(" " + w.lower(), add_special_tokens=False).input_ids
            if len(ids) == 1:
                ws.append(ids[0])
        if ws:
            rows.append((j, ws))
            n_ok += 1
    log(f"rhyme rate (last 2 letters) {n_rhyme / max(n_tot, 1):.2f} over {n_tot} complete second lines; "
        f"first lines with >=1 single-token ending: {n_ok}")
    for j, ws in rows[:6]:
        log(f"   {lines[j]!r} -> {[tok.decode([w]) for w in ws]}")
    H, last_tok = newline_resid(ctx, [ctxs[j] for j, _ in rows])
    log(f"readout token(s) at end of context: {last_tok}; {time.time() - t0:.0f}s")
    words = [ws for _, ws in rows]
    modal = torch.as_tensor([Counter(ws).most_common(1)[0][0] for ws in words], device=ctx.device)
    agree = sum(Counter(ws).most_common(1)[0][1] / len(ws) for ws in words) / len(words)
    log(f"mean share of samples equal to the modal ending: {agree:.2f}; distinct modal endings "
        f"{len(set(modal.tolist()))}")
    return dict(H=H, words=words, modal=modal, lines=[lines[j] for j, _ in rows])


def eff_W(ctx):
    W = ctx.model._lm_head.weight
    d = W.shape[1]
    gn = ctx.model._final_norm(torch.ones(1, d, device=W.device, dtype=W.dtype)).float()[0]
    return W, gn


def targets(ctx, words):
    W, gn = eff_W(ctx)
    out = []
    for ws in words:
        e = unit(W[torch.as_tensor(ws, device=W.device)].float() * gn[None]).mean(0)
        out.append(e)
    return torch.stack(out).to(ctx.device)


class Ridge:
    def __init__(self, X, Y, lam):
        self.mu = X.mean(0, keepdim=True)
        self.Xc = X - self.mu
        K = self.Xc @ self.Xc.T
        lam_abs = lam * K.diagonal().mean()
        Yc = Y - Y.mean(0, keepdim=True)
        self.A = torch.linalg.solve(K + lam_abs * torch.eye(K.shape[0], device=K.device), Yc)

    def __call__(self, xu):
        return ((xu - self.mu) @ self.Xc.T) @ self.A


def excl_mask(ctx):
    V = ctx.model._lm_head.weight.shape[0]
    m = torch.zeros(V, dtype=torch.bool, device=ctx.device)
    m[ctx.fit.excluded_ids.to(ctx.device)] = True
    return m


def ranks(ctx, v, tgt, mask):
    lg = ctx.model.unembed(v).float().to(ctx.device)
    lg = lg.masked_fill(mask[None, :lg.shape[1]], float("-inf"))
    t = lg.gather(1, tgt[:, None])
    return (lg > t).sum(1) + 1


class PoemLens:
    """mode 'mix': unit(JPP h) + b * R_l(unit(h)); 'alone': R_l(unit(h)); 'slot': slots in slot_src return
    R_src(unit(resid_all[src])), others J++."""

    def __init__(self, JPP, R, b=1.0, mode="mix", slot_src=None):
        self.JPP, self.R, self.b, self.mode, self.slot_src = JPP, R, b, mode, slot_src or {}

    def transport(self, h, layer, resid_all=None):
        if self.mode == "alone":
            return self.R[layer](unit(h))
        if self.mode == "slot":
            if layer in self.slot_src:
                s = self.slot_src[layer]
                return self.R[s](unit(resid_all[s]))
            return h @ self.JPP[layer].T
        return unit(h @ self.JPP[layer].T) + self.b * self.R[layer](unit(h))


def held_eval(ctx, lens, H, tgt, sel, mask):
    best, per = None, {}
    for l in ctx.readout_layers:
        Hs = {k: v[sel] for k, v in H.items()}
        r = ranks(ctx, lens.transport(Hs[l], l, Hs), tgt[sel], mask)
        per[l] = float((r <= 10).float().mean())
        best = r if best is None else torch.minimum(best, r)
    return float((best <= 10).float().mean()), per


def fmt_per(per):
    return " ".join(f"L{l}:{v:.2f}" for l, v in per.items())


def build_poem(ctx, mode="mix", shuffle=False, n_prompts=384, K=4):
    t0 = time.time()
    data = build_data(ctx, n_prompts=n_prompts, K=K)
    H, words, modal = data["H"], data["words"], data["modal"]
    n = len(words)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    ho = torch.zeros(n, dtype=torch.bool)
    ho[perm[: n // 5]] = True
    ho = ho.to(ctx.device)
    tr = ~ho
    Y = targets(ctx, words)
    if shuffle:
        idx = torch.nonzero(tr).squeeze(1)
        p2 = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(1)).to(ctx.device)]
        Y = Y.clone()
        Y[idx] = Y[p2]
    mask = excl_mask(ctx)
    log(f"pairs {n}: train {int(tr.sum())}, held-out {int(ho.sum())}")
    # majority baseline: 10 most frequent train modal endings
    top_tr = [t for t, _ in Counter(modal[tr].tolist()).most_common(10)]
    log(f"control: 10 most frequent train endings contain held-out modal ending: "
        f"{sum(int(t) in top_tr for t in modal[ho].tolist()) / int(ho.sum()):.3f}")
    jpp = PoemLens(ctx.JPP, None, mode="jpp_only")
    jpp.transport = lambda h, layer, resid_all=None: h @ ctx.JPP[layer].T
    a, per = held_eval(ctx, jpp, H, modal, ho, mask)
    log(f"J++ held-out modal@10 any layer {a:.3f} | {fmt_per(per)}")
    Xu = {l: unit(H[l]) for l in ctx.readout_layers}
    res, Rs = {}, {}
    for lam in (0.03, 0.3, 3.0):
        R = {l: Ridge(Xu[l][tr], Y[tr], lam) for l in ctx.readout_layers}
        Rs[lam] = R
        a, per = held_eval(ctx, PoemLens(ctx.JPP, R, mode="alone"), H, modal, ho, mask)
        log(f"ridge alone lam {lam}: held-out modal@10 {a:.3f} | {fmt_per(per)}")
        for b in (0.25, 0.5, 1.0, 2.0):
            res[(lam, b)] = held_eval(ctx, PoemLens(ctx.JPP, R, b), H, modal, ho, mask)[0]
    log("mix held-out grid: " + " ".join(f"{a}/{b}:{v:.3f}" for (a, b), v in res.items()))
    lam, b = max(res, key=lambda k: (res[k], -k[1]))
    log(f"chosen lam {lam} b {b} (held-out {res[(lam, b)]:.3f}); {time.time() - t0:.0f}s")
    # refit on all generated pairs at the chosen lam
    R = {l: Ridge(Xu[l], Y, lam) for l in ctx.readout_layers}
    variants = {
        "jpp": jpp,
        "ridge_alone": PoemLens(ctx.JPP, R, mode="alone"),
        f"mix b{b}": PoemLens(ctx.JPP, R, b),
        "mix b0.5": PoemLens(ctx.JPP, R, 0.5),
        "slot4,8<-R20,R24": PoemLens(ctx.JPP, R, mode="slot", slot_src={4: 20, 8: 24}),
    }
    for name, lens in variants.items():
        vecs = {l: lens.transport(ctx.fit.resid[l], l, ctx.fit.resid) for l in ctx.readout_layers}
        report(ctx, name, vecs)
    log(f"build {time.time() - t0:.0f}s")
    if mode == "slot":
        return variants["slot4,8<-R20,R24"]
    return PoemLens(ctx.JPP, R, b)
