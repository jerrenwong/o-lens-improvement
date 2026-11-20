"""Shared code for b3i_* candidates (not a candidate itself).

Round-2 idea 2, "elicit-distilled map", plus the label-free ceiling measurement.

Data (label-free, WikiText train windows, para-start, cached by the harness):
  sentences = token spans from just after the previous period (or window start) to a period, 8..60 tokens,
  at most 2 per window. Each is run ALONE followed by an elicitation suffix (default " In one word, this is
  about"). Causal model: the activation at the period in that input equals the activation of the sentence run
  alone, so one forward pass gives
    x_l = block-l output at the period (l in the 7 readout layers),
    y   = last-block output at the last suffix token (the model's full-context answer state).
  Elicited words = model's top tokens there (excluded ids masked). "Topic word" = top-1 after additionally
  masking a small fixed list of function words (decided in advance; the raw top-1 is also reported).
Map: per layer, centred dual ridge E_l: x -> y on 2000 sentences (float64, eig of X X^T), lambda chosen per
  layer on 500 held-out sentences by top-10 agreement with the topic word. Lens output = E_l (h - mu_x) + mu_y.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b2d_common import fit_pass, report, unit  # noqa: E402,F401

from olens.readout import unembed_logits  # noqa: E402

SUFFIX = " In one word, this is about"
N_TRAIN, N_HELD = 2000, 500
FUNC = set("the a an how what which who whom why when where this that these those his her their its my our your "
           "to of in on for and or is was are were be being not it he she they we you i as at by with from one "
           "all some any no".split())
LAM_GRID = [1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0]


def log(*a):
    print("[b3i]", *a, flush=True)


def sentences(ctx, n_want, seed=0):
    tok = ctx.model.tokenizer
    W = ctx.wikitext("train", 2048).long()
    pids = set()
    for s in (".", " ."):
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            pids.add(ids[0])
    out = []
    for w in range(W.shape[0]):
        row = W[w].tolist()
        start, got = 0, 0
        for p, t in enumerate(row):
            if t in pids:
                if 8 <= p + 1 - start <= 60 and got < 2:
                    out.append(row[start:p + 1])
                    got += 1
                start = p + 1
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(out), generator=g).tolist()
    out = [out[i] for i in perm[:n_want]]
    log(f"{len(out)} sentences (of {len(perm)}), period ids {sorted(pids)}")
    return out


@torch.no_grad()
def collect(ctx, sents, suffix, bs=48):
    """-> X {layer: [N, d] fp32}, Y [N, d] fp32 (last-block output at last suffix token)."""
    model, dev = ctx.model, ctx.device
    layers = ctx.readout_layers
    last = len(model.layers) - 1
    suf = model.tokenizer(suffix, add_special_tokens=False).input_ids
    N, d = len(sents), model.d_model
    X = {l: torch.zeros(N, d, device=dev) for l in layers}
    Y = torch.zeros(N, d, device=dev)
    by_len = {}
    for i, s in enumerate(sents):
        by_len.setdefault(len(s), []).append(i)
    for L, idx in sorted(by_len.items()):
        for b0 in range(0, len(idx), bs):
            bi = idx[b0:b0 + bs]
            ids = torch.tensor([sents[i] + suf for i in bi], dtype=torch.long, device=model.input_device)
            rec = {}

            def mk(l):
                def f(m, i, o):
                    t = o if torch.is_tensor(o) else o[0]
                    rec[l] = t
                return f

            hs = [model.layers[l].register_forward_hook(mk(l)) for l in set(layers) | {last}]
            try:
                model.forward(ids)
            finally:
                for x in hs:
                    x.remove()
            bt = torch.tensor(bi, device=dev)
            for l in layers:
                X[l][bt] = rec[l][:, L - 1].float().to(dev)
            Y[bt] = rec[last][:, -1].float().to(dev)
    return X, Y


def func_mask(tok, V, dev):
    n = min(len(tok), V)
    strs = tok.batch_decode([[i] for i in range(n)])
    m = torch.zeros(V, dtype=torch.bool)
    for i, s in enumerate(strs):
        if s.strip().lower() in FUNC:
            m[i] = True
    return m.to(dev)


def topk_ids(ctx, V_, k=10, extra_mask=None):
    lg = unembed_logits(ctx.model, V_, ctx.fit.excluded_ids, chunk_size=64)
    if extra_mask is not None:
        lg = lg.masked_fill(extra_mask.to(lg.device), float("-inf"))
    return lg.topk(k, dim=-1).indices


def agree(top10, target):
    return (top10 == target[:, None].to(top10.device)).any(-1).float().mean().item()


class Ridge:
    """Centred dual ridge, fixed eig of X X^T, any lambda."""

    def __init__(self, X, Y):
        self.mx, self.my = X.mean(0), Y.mean(0)
        self.Xc = (X - self.mx).double()
        self.Yc = (Y - self.my).double()
        s, U = torch.linalg.eigh(self.Xc @ self.Xc.T)
        self.s, self.U = s.clamp_min(0), U
        self.UtY = U.T @ self.Yc
        self.scale = self.s.mean().item()

    def predict(self, Xh, c):
        lam = c * self.scale
        A = (Xh - self.mx).double() @ self.Xc.T @ self.U
        return (A / (self.s + lam) @ self.UtY).float() + self.my

    def weight(self, c):
        lam = c * self.scale
        W = self.Xc.T @ (self.U / (self.s + lam)) @ self.UtY  # [d, d]: y_c = x_c @ W
        return W.float()


def build_maps(ctx, suffix=SUFFIX, diagnostics=False, shuffle_control=False):
    t0 = time.time()
    tok = ctx.model.tokenizer
    sents = sentences(ctx, N_TRAIN + N_HELD)
    X, Y = collect(ctx, sents, suffix)
    log(f"collected {len(sents)} in {time.time() - t0:.0f}s")
    tr, he = slice(0, N_TRAIN), slice(N_TRAIN, N_TRAIN + N_HELD)
    V = ctx.model.unembed(Y[:1]).shape[-1]
    fm = func_mask(tok, V, ctx.device)
    raw_top = topk_ids(ctx, Y, 10)
    content_top = topk_ids(ctx, Y, 10, fm)
    target = content_top[:, 0]
    if diagnostics:
        from collections import Counter
        c_raw = Counter(tok.decode([t]) for t in raw_top[:, 0].tolist())
        c_con = Counter(tok.decode([t]) for t in target.tolist())
        log("raw top-1 most common:", c_raw.most_common(12))
        log("topic top-1 most common:", c_con.most_common(12))
        log(f"topic word distinct: {len(c_con)} / {len(sents)}; raw top-1 is a function word in "
            f"{fm[raw_top[:, 0]].float().mean().item():.2f}")
        for i in range(5):
            log(repr(tok.decode(sents[N_TRAIN + i])[-120:]), "->",
                [tok.decode([t]) for t in content_top[N_TRAIN + i, :5].tolist()])
        # mean-answer baseline
        mtop = topk_ids(ctx, Y[tr].mean(0, keepdim=True), 10)
        log(f"control: mean-train-answer top10 contains held-out topic word: "
            f"{agree(mtop.expand(N_HELD, -1), target[he]):.3f}")
        ftr = Counter(target[tr].tolist()).most_common(10)
        ftop = torch.tensor([t for t, _ in ftr], device=ctx.device)[None].expand(N_HELD, -1)
        log(f"control: 10 most frequent train topic words contain held-out topic word: {agree(ftop, target[he]):.3f}")
    maps, info = {}, {}
    for l in ctx.readout_layers:
        r = Ridge(X[l][tr], Y[tr])
        scores = []
        for c in LAM_GRID:
            scores.append(agree(topk_ids(ctx, r.predict(X[l][he], c), 10), target[he]))
        best = max(range(len(LAM_GRID)), key=lambda i: (scores[i], i))
        c = LAM_GRID[best]
        maps[l] = (r.mx, r.weight(c), r.my)
        info[l] = scores[best]
        line = f"L{l:2d} held-out topic@10 by c: " + " ".join(f"{cc:g}:{s:.3f}" for cc, s in zip(LAM_GRID, scores))
        if diagnostics:
            # in-sample (train) agreement at chosen c, and top-1 agreement on held-out
            ptr = r.predict(X[l][tr][:500], c)
            line += f" | train@10 {agree(topk_ids(ctx, ptr, 10), target[:500]):.3f}"
            ph = r.predict(X[l][he], c)
            line += f" | held top1 {agree(topk_ids(ctx, ph, 1), target[he]):.3f}"
            line += f" | held raw-top1@10 {agree(topk_ids(ctx, ph, 10), raw_top[he, 0]):.3f}"
            if shuffle_control:
                g = torch.Generator().manual_seed(1)
                perm = torch.randperm(N_TRAIN, generator=g).to(ctx.device)
                rs = Ridge(X[l][tr], Y[tr][perm])
                line += f" | shuffled-target {agree(topk_ids(ctx, rs.predict(X[l][he], c), 10), target[he]):.3f}"
                del rs
        log(line)
        del r
        torch.cuda.empty_cache()
    if diagnostics:
        # M1 (label-free): J++ on the period activation, held-out sentences
        anyhit = torch.zeros(N_HELD, dtype=torch.bool, device=ctx.device)
        anyE = torch.zeros_like(anyhit)
        parts = []
        for l in ctx.readout_layers:
            tj = topk_ids(ctx, X[l][he] @ ctx.JPP[l].T, 10)
            h_ = (tj == target[he][:, None].to(tj.device)).any(-1).to(ctx.device)
            anyhit |= h_
            mx, Wm, my = maps[l]
            te = topk_ids(ctx, (X[l][he] - mx) @ Wm + my, 10)
            anyE |= (te == target[he][:, None].to(te.device)).any(-1).to(ctx.device)
            parts.append(f"L{l} {h_.float().mean().item():.3f}")
        log("M1: J++ top10 on period activation contains model's elicited topic word, per layer:", " ".join(parts))
        log(f"M1: any layer (J++): {anyhit.float().mean().item():.3f}; any layer (E_l): "
            f"{anyE.float().mean().item():.3f}; union {(anyhit | anyE).float().mean().item():.3f}")
        raw_any = torch.zeros_like(anyhit)
        for l in ctx.readout_layers:
            tj = topk_ids(ctx, X[l][he] @ ctx.JPP[l].T, 10)
            raw_any |= (tj == raw_top[he, 0][:, None].to(tj.device)).any(-1).to(ctx.device)
        log(f"M1 (raw top-1 incl. function words): any layer J++ {raw_any.float().mean().item():.3f}")
    log(f"maps built in {time.time() - t0:.0f}s")
    return maps, info


class ELens:
    """mode 'alone': E_l h; 'mix': unit(JPP h) + lam unit(E_l h); 'slot': slots in `slot_src` use
    E_src on resid_all[src], other slots plain J++."""

    def __init__(self, JPP, maps, mode, lam=None, slot_src=None):
        self.JPP, self.maps, self.mode, self.lam, self.slot_src = JPP, maps, mode, lam, slot_src or {}

    def E(self, h, l):
        mx, W, my = self.maps[l]
        return (h - mx) @ W + my

    def transport(self, h, layer, resid_all=None):
        if self.mode == "alone":
            return self.E(h, layer)
        if self.mode == "mix":
            return unit(h @ self.JPP[layer].T) + self.lam * unit(self.E(h, layer))
        if self.mode == "slot":
            if layer in self.slot_src:
                s = self.slot_src[layer]
                return self.E(resid_all[s], s)
            return h @ self.JPP[layer].T
        return h @ self.JPP[layer].T


def fit_report(ctx, maps, info):
    """Fit-split diagnostics for all variants (labelled FIT split only)."""
    L = ctx.readout_layers
    best2 = sorted([l for l in L if l >= 12], key=lambda l: -info[l])[:2]
    variants = {
        "jpp": ELens(ctx.JPP, maps, "jpp"),
        "E_alone": ELens(ctx.JPP, maps, "alone"),
        "mix0.3": ELens(ctx.JPP, maps, "mix", 0.3),
        "mix1.0": ELens(ctx.JPP, maps, "mix", 1.0),
        f"slot4,8<-E{best2}": ELens(ctx.JPP, maps, "slot", slot_src={L[0]: best2[0], L[1]: best2[1]}),
    }
    for name, lens in variants.items():
        vecs = {l: lens.transport(ctx.fit.resid[l], l, ctx.fit.resid) for l in L}
        report(ctx, name, vecs)
