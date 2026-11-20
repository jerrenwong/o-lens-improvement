"""b1c diag: plain J++ returned; build() prints FIT-split diagnostics only (ctx.fit).
- J-lens and J++: per task, best-rank histogram over the 7 readout layers, per-layer hit counts.
- J++ at every layer 0..30: per-layer hits, and hits gained by off-grid layers.
- Fit pass@10 of the b1c pooling / mixture variants (info only)."""

from collections import defaultdict

import torch

from olens.readout import candidate_ranks, unembed_logits

BINS = [(1, 10, "<=10"), (11, 30, "11-30"), (31, 100, "31-100"), (101, 1000, "101-1000"), (1001, 10**9, ">1000")]


def unit(v):
    return v / v.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)


def ranks_of(ctx, V):
    """V [n_fit, d] -> list of (task, item, inter, rank) with the filtered harness rule."""
    logits = unembed_logits(ctx.model, V, ctx.fit.excluded_ids, chunk_size=8)
    rk = candidate_ranks(logits, ctx.fit.candidates)
    out = []
    for p, d in enumerate(rk):
        for inter, r in d.items():
            out.append((ctx.fit.eval_slugs[p], p, inter, r))
    return out


def pass10(best):
    """best: {(task, item, inter): rank} -> per-task item-weighted pass@10 and macro mean."""
    item = defaultdict(list)
    for (t, p, _), r in best.items():
        item[(t, p)].append(float(r <= 10))
    task = defaultdict(list)
    for (t, _), h in item.items():
        task[t].append(sum(h) / len(h))
    pt = {t: sum(v) / len(v) for t, v in sorted(task.items())}
    return sum(pt.values()) / len(pt), pt


def table(ctx, name, layer_ranks, layers):
    """layer_ranks: {layer: [(task, item, inter, rank)]}"""
    best, per_layer = {}, defaultdict(lambda: defaultdict(int))
    for l in layers:
        for t, p, inter, r in layer_ranks[l]:
            k = (t, p, inter)
            best[k] = min(best.get(k, r), r)
            if r <= 10:
                per_layer[t][l] += 1
    tasks = sorted({k[0] for k in best})
    print(f"\n[{name}] best rank over layers {layers}: pairs per bin, then per-layer hit counts")
    print("task".ljust(14) + "n".rjust(4) + "".join(b[2].rjust(10) for b in BINS) + " | " +
          " ".join(f"L{l}".rjust(4) for l in layers))
    for t in tasks + ["ALL"]:
        rs = [r for k, r in best.items() if t == "ALL" or k[0] == t]
        hist = [sum(lo <= r <= hi for r in rs) for lo, hi, _ in BINS]
        pl = [sum(per_layer[x][l] for x in tasks) if t == "ALL" else per_layer[t][l] for l in layers]
        print(t.ljust(14) + str(len(rs)).rjust(4) + "".join(str(h).rjust(10) for h in hist) + " | " +
              " ".join(str(c).rjust(4) for c in pl))
    m, pt = pass10(best)
    print(f"[{name}] fit pass@10 {m:.4f} " + " ".join(f"{t} {v:.2f}" for t, v in pt.items()))
    return best


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def build(ctx):
    R = ctx.fit.resid
    RL = ctx.readout_layers
    all_layers = sorted(l for l in R if l in ctx.JPP)
    print(f"[b1c diag] fit n={len(ctx.fit.candidates)} resid layers {sorted(R)} JPP layers {sorted(ctx.JPP)} "
          f"J layers {sorted(ctx.J)}", flush=True)
    from collections import Counter
    print("[b1c diag] fit tasks", dict(Counter(ctx.fit.eval_slugs)))

    P = {l: unit(R[l] @ ctx.JPP[l].T) for l in all_layers}
    Jv = {l: unit(R[l] @ ctx.J[l].T) for l in RL}
    jpp_r = {l: ranks_of(ctx, P[l]) for l in all_layers}
    j_r = {l: ranks_of(ctx, Jv[l]) for l in RL}

    table(ctx, "J-lens", j_r, RL)
    b7 = table(ctx, "J++ readout", jpp_r, RL)
    b31 = table(ctx, "J++ all 31 layers", jpp_r, all_layers)
    off = [l for l in all_layers if l not in RL]
    boff = table(ctx, "J++ off-grid layers only", jpp_r, off)
    gained = [k for k in b31 if b31[k] <= 10 and b7[k] > 10]
    print(f"\n[b1c diag] pairs hit at some off-grid layer but missed at all 7 readout layers: {len(gained)}")
    for k in gained:
        ls = [l for l in off if dict(((t, p, i), r) for t, p, i, r in jpp_r[l]).get(k, 10**9) <= 10]
        print(f"   {k[0]:13s} item {k[1]:3d} {k[2]!r:28s} readout best {b7[k]:6d}  off-grid hit layers {ls}")

    # fit scores of the b1c variants (information only; dev decides)
    def pooled(l, w):
        v = 0
        for k, wk in w.items():
            if l + k in P:
                v = v + wk * P[l + k]
        return v

    variants = {
        "pool1": {-1: 1, 0: 1, 1: 1},
        "pool2": {-2: 1, -1: 1, 0: 1, 1: 1, 2: 1},
        "pool2tri": {-2: 1, -1: 2, 0: 3, 1: 2, 2: 1},
    }
    for name, w in variants.items():
        rr = {l: ranks_of(ctx, pooled(l, w)) for l in RL}
        table(ctx, name, rr, RL)
    # disjoint bands covering 0..30
    bands = {4: range(0, 7), 8: range(7, 11), 12: range(11, 15), 16: range(15, 19), 20: range(19, 23),
             24: range(23, 27), 28: range(27, 31)}
    rr = {l: ranks_of(ctx, sum(P[x] for x in bands[l] if x in P)) for l in RL}
    table(ctx, "bands", rr, RL)
    for lam in (0.3, 1.0):
        rr = {l: ranks_of(ctx, P[l] + lam * Jv[l]) for l in RL}
        table(ctx, f"mix lam={lam}", rr, RL)
    return Lens(ctx.JPP)
