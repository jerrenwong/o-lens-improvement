"""Shared code for b3h_* candidates (not a candidate itself).

Round-2 idea 1, "one slot per word": readout de-duplication as a vector edit.
  For each row: logits z = (v * gamma) @ W.T (final_linear, up to the per-row 1/rms scale), punctuation ids
  (ctx.fit.excluded_ids, the scorer's own fixed list) masked, top-K tokens grouped by normalised surface form
  (decode -> strip -> lower; optionally also strip a trailing "'s" / "s"). Every token that is not the
  best-ranked member of its group gets target change (z_K - margin) - z_i, the other top-K tokens 0; the
  minimum-norm delta solving A delta = t over the K rows A = W[topK] * gamma (pseudo-inverse) is added to v.
  Repeated up to ITERS times (tokens entering the top-K after an edit can bring new duplicates).
  This changes the readout convention (one slot per word), not the lens map.

Round-2 slot idea ("specialise the redundant slots"): leave-one-slot-out union pass@10 on FIT, then greedy
  replacement of a readout slot's map by the same base lens at an unread layer (M_p applied to resid[p]),
  chosen by union pass@10 on FIT only.
"""

import math
from collections import defaultdict

import torch

from olens.readout import candidate_ranks, final_linear, unembed_logits

POOL = [5, 6, 7, 19, 21, 22, 23, 29, 30]
ITERS = 3


def log(*a):
    print("[b3h]", *a, flush=True)


# ---------------------------------------------------------------- surface-form groups


def build_groups(tok, V, plural):
    gid = torch.empty(V, dtype=torch.long)
    table = {}
    n_tok = min(len(tok), V)
    strs = tok.batch_decode([[i] for i in range(n_tok)]) + [""] * (V - n_tok)
    for i in range(V):
        k = strs[i].strip().lower()
        if plural and k:
            if k.endswith("'s") or k.endswith("’s"):
                k = k[:-2]
            elif k.endswith("s") and len(k) > 3:
                k = k[:-1]
        if not k or "�" in k:
            gid[i] = -1 - i  # own group
            continue
        gid[i] = table.setdefault(k, len(table))
    return gid


class Dedup:
    def __init__(self, ctx, K, plural, margin=2.0):
        self.W, self.gamma = final_linear(ctx.model)
        dev = self.W.device
        V = self.W.shape[0]
        self.K, self.margin = K, margin
        self.gid = build_groups(ctx.model.tokenizer, V, plural).to(dev)
        self.mask = torch.zeros(V, dtype=torch.bool, device=dev)
        self.mask[ctx.fit.excluded_ids.to(dev)] = True
        self.stats = defaultdict(int)

    @torch.no_grad()
    def __call__(self, v, chunk=32):
        out = []
        for s in range(0, v.shape[0], chunk):
            out.append(self._one(v[s:s + chunk]))
        return torch.cat(out)

    def _one(self, v):
        dev0 = v.device
        v = v.to(self.W.device).float()
        K = self.K
        tri = torch.ones(K, K, dtype=torch.bool, device=v.device).tril(-1)  # [j, j'] j' < j
        for _ in range(ITERS):
            z = (v * self.gamma) @ self.W.T
            z.masked_fill_(self.mask, float("-inf"))
            topv, topi = z.topk(K, dim=-1)
            g = self.gid[topi]
            dup = ((g[:, :, None] == g[:, None, :]) & tri).any(-1)  # [N, K]
            if not dup.any():
                break
            self.stats["dups"] += int(dup.sum())
            rms = v.pow(2).mean(-1, keepdim=True).sqrt()
            zK = topv[:, -1:]
            t = torch.where(dup, zK - self.margin * rms - topv, torch.zeros_like(topv))
            A = self.W[topi] * self.gamma  # [N, K, d]
            delta = (torch.linalg.pinv(A) @ t[..., None])[..., 0]
            v = v + delta
        return v.to(dev0)


# ---------------------------------------------------------------- fit scoring


def fit_ranks(ctx, v):
    """[n_fit, d] -> list of {intermediate: rank} (benchmark rule, filtered) and the logits' top-10 ids."""
    logits = unembed_logits(ctx.model, v, ctx.fit.excluded_ids, chunk_size=16)
    top10 = logits.topk(10, dim=-1).indices.cpu()
    return candidate_ranks(logits, ctx.fit.candidates), top10


def union(ctx, rank_sets, k):
    best = [dict() for _ in ctx.fit.candidates]
    for rs in rank_sets:
        for p, d in enumerate(rs):
            for kk, r in d.items():
                best[p][kk] = min(best[p].get(kk, 10 ** 9), r)
    per = defaultdict(list)
    for p, d in enumerate(best):
        if d:
            per[ctx.fit.eval_slugs[p]].append(sum(r <= k for r in d.values()) / len(d))
    task = {t: sum(x) / len(x) for t, x in sorted(per.items())}
    logr = sum(math.log(r) for d in best for r in d.values())
    hits = sum(r <= k for d in best for r in d.values())
    return sum(task.values()) / len(task), task, hits, logr


def fmt(task):
    return " ".join(f"{t[:5]} {x:.2f}" for t, x in task.items())


# ---------------------------------------------------------------- lens


class Lens:
    def __init__(self, M, dedup=None, slot_map=None):
        self.M, self.dedup, self.slot_map = M, dedup, dict(slot_map or {})

    def raw(self, h, layer, resid_all):
        src = self.slot_map.get(layer, layer)
        x = h if src == layer else resid_all[src]
        return x @ self.M[src].T

    def transport(self, h, layer, resid_all=None):
        v = self.raw(h, layer, resid_all)
        return self.dedup(v) if self.dedup is not None else v


def check_dedup(ctx, M, dd):
    """Fit checks: (a) no two top-10 tokens share a normalised form after the edit; (b) no concept rank worse."""
    gid = dd.gid.cpu()
    before, after = [], []
    n_dup_b = n_dup_a = 0
    worse = better = worse10 = worse100 = 0
    for layer in ctx.readout_layers:
        v0 = ctx.fit.resid[layer] @ M[layer].T
        r0, t0 = fit_ranks(ctx, v0)
        r1, t1 = fit_ranks(ctx, dd(v0))
        before.append(r0)
        after.append(r1)
        for t, attr in ((t0, "b"), (t1, "a")):
            g = gid[t]
            n = sum(len(set(row.tolist())) < 10 for row in g)
            if attr == "b":
                n_dup_b += n
            else:
                n_dup_a += n
        tok = ctx.model.tokenizer
        for p, (d0, d1) in enumerate(zip(r0, r1)):
            for kk in d0:
                if d1[kk] > d0[kk]:
                    worse += 1
                    worse10 += d0[kk] <= 10
                    worse100 += d0[kk] <= 100
                    if d0[kk] <= 10:
                        ids = ctx.fit.candidates[p][kk]
                        log(f"  worse<=10 L{layer} {ctx.fit.eval_slugs[p]} {kk!r} {d0[kk]}->{d1[kk]} "
                            f"forms {[tok.decode([i]) for i in ids][:8]} | before {tok.batch_decode([[i] for i in t0[p].tolist()])} "
                            f"| after {tok.batch_decode([[i] for i in t1[p].tolist()])}")
                elif d1[kk] < d0[kk]:
                    better += 1
    nrows = len(ctx.fit.candidates) * len(ctx.readout_layers)
    log(f"check (a): rows with a duplicate form in top-10: before {n_dup_b}/{nrows}, after {n_dup_a}/{nrows}")
    log(f"check (b): (pair, layer) ranks worse {worse} (were <= 10: {worse10}, <= 100: {worse100}), better {better}")
    for name, rs in (("plain", before), ("dedup", after)):
        p10, task, hits, _ = union(ctx, rs, 10)
        p1, _, _, _ = union(ctx, rs, 1)
        log(f"fit {name}: p10 {p10:.4f} p1 {p1:.4f} hits {hits} | {fmt(task)}")
    log(f"total duplicate edits: {dd.stats['dups']}")


def choose_slots(ctx, M, dd, n_rep):
    """Leave-one-slot-out on fit, then greedy replacement of up to n_rep readout slots by pool layers."""
    f = (lambda v: dd(v)) if dd is not None else (lambda v: v)
    R = {}
    for layer in list(ctx.readout_layers) + POOL:
        R[layer], _ = fit_ranks(ctx, f(ctx.fit.resid[layer] @ M[layer].T))
    slots = {s: s for s in ctx.readout_layers}

    def score(sl):
        p10, task, hits, logr = union(ctx, [R[x] for x in sl.values()], 10)
        p1, _, _, _ = union(ctx, [R[x] for x in sl.values()], 1)
        return (round(p10, 6), round(p1, 6), -logr), task, hits

    base, task, hits = score(slots)
    log(f"fit union all 7: p10 {base[0]:.4f} p1 {base[1]:.4f} hits {hits} | {fmt(task)}")
    for s in ctx.readout_layers:
        sl = {k: v for k, v in slots.items() if k != s}
        sc, _, h = score(sl)
        log(f"  leave out L{s}: p10 {sc[0]:.4f} (d {sc[0] - base[0]:+.4f}) hits {h} (d {h - hits:+d})")
    for p in POOL:
        p10, task_p, h, _ = union(ctx, [R[p]], 10)
        log(f"  pool L{p} alone: p10 {p10:.4f} hits {h}")
    cur = base
    for _ in range(n_rep):
        best = None
        for s in ctx.readout_layers:
            if slots[s] != s:
                continue
            for p in POOL:
                if p in slots.values():
                    continue
                sl = dict(slots)
                sl[s] = p
                sc, _, h = score(sl)
                if best is None or sc > best[0]:
                    best = (sc, s, p, h)
        if best is None or best[0][0] <= cur[0]:
            log("  no replacement improves fit union p10; stop")
            break
        cur = best[0]
        slots[best[1]] = best[2]
        log(f"  replace slot L{best[1]} by L{best[2]}: fit p10 {cur[0]:.4f} p1 {cur[1]:.4f} hits {best[3]}")
    return {s: p for s, p in slots.items() if s != p}


def make(ctx, base, K=None, plural=False, n_rep=0, check=True):
    M = ctx.JPP if base == "jpp" else ctx.J
    dd = Dedup(ctx, K, plural) if K else None
    if dd is not None and check:
        check_dedup(ctx, M, dd)
    slot_map = choose_slots(ctx, M, dd, n_rep) if n_rep else {}
    log(f"slot map: {slot_map}")
    return Lens(M, dd, slot_map)
