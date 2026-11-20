"""Shared code for b2d_* candidates (not a candidate itself).

Patchscope (idea 6): run the model on a short fixed decoding prompt with the activation h written into the
residual stream at block `layer` (output of model.layers[layer]) at one position, read the output of the last
block at the prompt's last token: S_l(h). Lens: S alone, or unit(JPP_l h) + lam * unit(S_l(h)).

Patchscope-Jacobian (idea 10): d S_l / d h, averaged over WikiText activations at layer l.

Fit helpers: per-task pass@k on ctx.fit (the labelled FIT split), printed from build().
"""

import time
from collections import defaultdict

import torch

from olens.readout import candidate_ranks, unembed_logits

CHUNK = 32

# Decoding prompts, fixed in advance: (prefix, placeholder, suffix). The activation is written at the
# placeholder token (or at the last token when suffix is empty); S is read at the last token.
PROMPTS = {
    "the": ("", " the", ""),
    "repeat": ("cat -> cat\n1135 -> 1135\nhello -> hello\n", "?", " ->"),
    "concept": ("The word that best describes this", " is", ""),
    "concept_ph": ("Text:", " ?", "\nIn one word, this text is about"),
}


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def _ids(tok, prompt):
    pre, ph, suf = prompt
    a = tok(pre).input_ids if pre else []
    b = tok(ph).input_ids
    c = tok(suf).input_ids if suf else []
    assert len(b) == 1, (ph, b)
    return a + b + c, len(a)


class Patcher:
    def __init__(self, model, prompt, mode="replace"):
        self.model, self.mode = model, mode
        ids, self.pos = _ids(model.tokenizer, prompt)
        self.ids = torch.tensor(ids, dtype=torch.long, device=model.input_device)

    def _run(self, h, layer):
        """h [B, d] (any dtype, may require grad) -> final-block output at last token [B, d]."""
        model = self.model
        B = h.shape[0]
        ids = self.ids[None].expand(B, -1)
        pos, mode = self.pos, self.mode
        rec = {}

        def patch(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            hh = h.to(t.dtype)
            if mode == "replace":
                t = torch.cat([t[:, :pos], hh[:, None], t[:, pos + 1:]], dim=1)
            else:
                t = torch.cat([t[:, :pos], t[:, pos:pos + 1] + hh[:, None], t[:, pos + 1:]], dim=1)
            return t if torch.is_tensor(output) else (t,) + tuple(output[1:])

        def read(module, inputs, output):
            t = output if torch.is_tensor(output) else output[0]
            rec["out"] = t[:, -1]

        hs = [model.layers[layer].register_forward_hook(patch),
              model.layers[len(model.layers) - 1].register_forward_hook(read)]
        try:
            model.forward(ids)
        finally:
            for x in hs:
                x.remove()
        return rec["out"]

    @torch.no_grad()
    def __call__(self, h, layer):
        out = [self._run(h[i:i + CHUNK], layer).float() for i in range(0, h.shape[0], CHUNK)]
        return torch.cat(out, 0)


# ---------------------------------------------------------------- fit scoring


def fit_ranks(ctx, vecs):
    """vecs: {layer: [n_fit, d]} -> per item list of {intermediate: best rank over layers}."""
    best = [dict() for _ in ctx.fit.candidates]
    for layer, v in vecs.items():
        logits = unembed_logits(ctx.model, v, ctx.fit.excluded_ids, chunk_size=16)
        for p, d in enumerate(candidate_ranks(logits, ctx.fit.candidates)):
            for k, r in d.items():
                best[p][k] = min(best[p].get(k, 10 ** 9), r)
    return best


def per_layer_hits(ctx, vecs, k=10):
    """{layer: {task: n_items with any intermediate rank <= k}}."""
    out = {}
    for layer, v in vecs.items():
        logits = unembed_logits(ctx.model, v, ctx.fit.excluded_ids, chunk_size=16)
        cnt = defaultdict(int)
        for p, d in enumerate(candidate_ranks(logits, ctx.fit.candidates)):
            if d and min(d.values()) <= k:
                cnt[ctx.fit.eval_slugs[p]] += 1
        out[layer] = dict(cnt)
    return out


def fit_pass(ctx, vecs, k=10):
    best = fit_ranks(ctx, vecs)
    per = defaultdict(list)
    for p, d in enumerate(best):
        if d:
            per[ctx.fit.eval_slugs[p]].append(sum(r <= k for r in d.values()) / len(d))
    task = {t: sum(v) / len(v) for t, v in sorted(per.items())}
    return sum(task.values()) / len(task), task


def report(ctx, name, vecs):
    p10, t10 = fit_pass(ctx, vecs, 10)
    p1, _ = fit_pass(ctx, vecs, 1)
    tasks = " ".join(f"{t[:5]} {v:.2f}" for t, v in t10.items())
    print(f"[fit] {name:34s} p10 {p10:.4f} p1 {p1:.4f} | {tasks}", flush=True)
    return p10, p1


# ---------------------------------------------------------------- lenses


class PatchLens:
    def __init__(self, JPP, patcher, lam=None):
        self.JPP, self.patcher, self.lam = JPP, patcher, lam

    def transport(self, h, layer, resid_all=None):
        S = self.patcher(h, layer)
        if self.lam is None:
            return S
        return unit(h @ self.JPP[layer].T) + self.lam * unit(S)


class LinLens:
    """unit(JPP_l h) + lam * unit(M_l h); lam=None -> M_l h alone."""

    def __init__(self, JPP, M, lam=None):
        self.JPP, self.M, self.lam = JPP, M, lam

    def transport(self, h, layer, resid_all=None):
        x = h @ self.M[layer].T
        if self.lam is None:
            return x
        return unit(h @ self.JPP[layer].T) + self.lam * unit(x)


# ---------------------------------------------------------------- patchscope Jacobian (idea 10)


def ps_jacobian(ctx, patcher, n_src=8, rows=256, seed=0):
    """{layer: [d, d]} mean over n_src WikiText activations of d S_l / d h (rows = output dims)."""
    model, dev = ctx.model, ctx.device
    d = model.d_model
    W = ctx.wikitext("train", 64).long()
    g = torch.Generator().manual_seed(seed)
    wi = torch.randint(0, W.shape[0], (n_src,), generator=g)
    pi = torch.randint(16, W.shape[1], (n_src,), generator=g)
    layers = ctx.readout_layers
    srcs = {l: [] for l in layers}
    with torch.no_grad():
        for w, p in zip(wi.tolist(), pi.tolist()):
            rec = {}
            hs = [model.layers[l].register_forward_hook(
                (lambda l: lambda m, i, o: rec.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, p]))(l))
                for l in layers]
            try:
                model.forward(W[w:w + 1, :p + 1].to(dev))
            finally:
                for x in hs:
                    x.remove()
            for l in layers:
                srcs[l].append(rec[l].float())
    Js = {}
    t0 = time.time()
    for l in layers:
        Jl = torch.zeros(d, d, device=dev)
        for src in srcs[l]:
            for r0 in range(0, d, rows):
                n = min(rows, d - r0)
                with torch.enable_grad():
                    hv = src[None].expand(n, -1).clone().requires_grad_(True)
                    out = patcher._run(hv, l).float()
                    idx = torch.arange(r0, r0 + n, device=dev)
                    sel = out[torch.arange(n, device=dev), idx]
                    (gr,) = torch.autograd.grad(sel.sum(), hv)
                Jl[r0:r0 + n] += gr
        Js[l] = Jl / len(srcs[l])
        print(f"[jac] layer {l} done {time.time() - t0:.0f}s", flush=True)
    return Js
