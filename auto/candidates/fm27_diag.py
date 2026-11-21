"""fm27 diagnosis: prints FIT-split diagnostics of lens C (J++ + lam * ask-map) vs J++, returns plain C.
Dev score must reproduce C (9B .6309/.3961, 27B .6595/.3961). FIT only; logit-space variants are exact up to
bf16 rounding (unembed is linear in unit-rms inputs up to the shared rms factor)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm27_lib import (GRID, KS, LAM_C, MUS, NEG, TAG, Base, candidate_ranks, excl_mask, fmt, logits,  # noqa: E402
                      summarize, unit)

OUT = Path("/workspace/o-lens-improvement/scripts/fm27/results")
STOP = set("the a an of to in on and or for is are was be by with as at it this that from not but his her their "
           "its he she they we you i".split())


def zstd(L, mask):
    x = L[:, ~mask]
    return (L - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-6)


def f1_scores(Lj, Le, mask, K, mu, filt):
    base = Lj.masked_fill(mask, NEG) if filt else Lj.clone()
    top = base.topk(K, dim=-1).indices
    s = zstd(Lj, mask).gather(1, top) + mu * zstd(Le, mask).gather(1, top)
    out = base.clone()
    out.scatter_(1, top, 1e6 + s)
    return out


def build(ctx):
    torch.manual_seed(0)
    d = ctx.model.d_model
    tag, lamC = TAG[d], LAM_C[d]
    tok = ctx.model.tokenizer
    mask = excl_mask(ctx)
    lensC = Base(ctx, lamC)
    cands = ctx.fit.candidates
    n = len(cands)
    dec = lambda i: tok.decode([int(i)])  # noqa: E731
    print(f"[fm27] model {tag} lamC {lamC} n_fit {n} layers {ctx.readout_layers}", flush=True)

    variants = {}  # name -> (filtered ranks per layer, unfiltered ranks per layer)

    def add(name, layer, S_f, S_u):
        rf, ru = variants.setdefault(name, ({}, {}))
        rf[layer] = candidate_ranks(S_f, cands)
        ru[layer] = candidate_ranks(S_u, cands)

    stats_a = {}
    top = {}  # layer -> dict of filtered/unfiltered top-10 ids for J, E, C
    for l in ctx.readout_layers:
        h = ctx.fit.resid[l]
        mx, W, my = lensC.maps[l]
        jv = h @ ctx.JPP[l].T
        ev = (h - mx) @ W + my
        evc = (h - mx) @ W
        uj, ue, uec = unit(jv), unit(ev), unit(evc)
        Lj, Le, Lec = logits(ctx.model, uj), logits(ctx.model, ue), logits(ctx.model, uec)
        LC = logits(ctx.model, uj + lamC * ue)  # harness-identical C
        Lp = logits(ctx.model, unit(my)[None])  # map prior readout
        cos = torch.nn.functional.cosine_similarity(jv, ev, dim=-1)
        cos_c = torch.nn.functional.cosine_similarity(jv, evc, dim=-1)
        cos_pr = torch.nn.functional.cosine_similarity(ev, my[None].expand_as(ev), dim=-1)
        Ljm, Lem, LCm = (x.masked_fill(mask, NEG) for x in (Lj, Le, LC))
        t10 = {k: x.topk(10, dim=-1).indices for k, x in (("J", Ljm), ("E", Lem), ("C", LCm))}
        t10u = {k: x.topk(10, dim=-1).indices for k, x in (("J", Lj), ("E", Le), ("C", LC))}
        prior20 = set(Lp.masked_fill(mask, NEG).topk(20, dim=-1).indices[0].tolist())
        top[l] = dict(f=t10, u=t10u, prior20=prior20, C1=LCm.argmax(-1), C1u=LC.argmax(-1),
                      J1=Ljm.argmax(-1), J1u=Lj.argmax(-1))

        def frac_in(a, b):  # a [n] ids, b [n, 10]
            return (a[:, None] == b).any(-1).float().mean().item()

        def overlap(a, b):
            return (a[:, :, None] == b[:, None, :]).any(-1).float().sum(-1).mean().item()

        c1 = t10["C"][:, 0]
        inJ = (c1[:, None] == t10["J"]).any(-1)
        inE = (c1[:, None] == t10["E"]).any(-1)
        ovJ = (t10["C"][:, :, None] == t10["J"][:, None, :]).any(-1)
        ovE = (t10["C"][:, :, None] == t10["E"][:, None, :]).any(-1)
        # logit spread: std over vocab and top1 - top10 gap of each part (filtered), on the unit-rms vectors
        sd = lambda L: L[:, ~mask].std(-1).mean().item()  # noqa: E731
        gap = lambda Lm: (Lm.topk(10, -1).values[:, 0] - Lm.topk(10, -1).values[:, 9]).mean().item()  # noqa: E731
        stats_a[l] = dict(
            norm_J=jv.norm(dim=-1).mean().item(), norm_E=ev.norm(dim=-1).mean().item(),
            norm_Ec=evc.norm(dim=-1).mean().item(), norm_b=my.norm().item(),
            cos_J_E=cos.mean().item(), cos_J_Ec=cos_c.mean().item(), cos_E_prior=cos_pr.mean().item(),
            sd_J=sd(Lj), sd_E=sd(Le), gap_J=gap(Ljm), gap_E=gap(Lem), gap_C=gap(LCm),
            C1_eq_J1=(c1 == t10["J"][:, 0]).float().mean().item(),
            C1_eq_E1=(c1 == t10["E"][:, 0]).float().mean().item(),
            C1_inJ10=inJ.float().mean().item(), C1_inE10=inE.float().mean().item(),
            C1_inJ10_notE10=(inJ & ~inE).float().mean().item(), C1_inE10_notJ10=(inE & ~inJ).float().mean().item(),
            C1_neither=(~inJ & ~inE).float().mean().item(),
            C10_inJ10=ovJ.float().sum(-1).mean().item(), C10_inE10=ovE.float().sum(-1).mean().item(),
            C10_Jonly=(ovJ & ~ovE).float().sum(-1).mean().item(), C10_Eonly=(ovE & ~ovJ).float().sum(-1).mean().item(),
            C10_neither=(~ovJ & ~ovE).float().sum(-1).mean().item(),
            C1_in_prior20=torch.tensor([t in prior20 for t in c1.tolist()]).float().mean().item(),
            E1_in_prior20=torch.tensor([t in prior20 for t in t10["E"][:, 0].tolist()]).float().mean().item(),
            prior_top10=[dec(t) for t in Lp.masked_fill(mask, NEG).topk(10, -1).indices[0].tolist()],
        )
        # ---- variants (logit space)
        add("J++", l, Ljm, Lj)
        add("C", l, LCm, LC)
        add("map_alone", l, Lem, Le)
        for lam in GRID:
            S = Lj + lam * Le
            add(f"mix{lam}", l, S.masked_fill(mask, NEG), S)
            S = Lj + lam * Lec
            add(f"cen{lam}", l, S.masked_fill(mask, NEG), S)
        for K in KS:
            for mu in MUS:
                add(f"F1x_K{K}_mu{mu}", l, f1_scores(Lj, Le, mask, K, mu, True), f1_scores(Lj, Le, mask, K, mu, False))
        # F3 exact: C order, J++ top-1 forced to rank 1
        S = LCm.clone()
        S.scatter_(1, Ljm.argmax(-1, keepdim=True), float("inf"))
        Su = LC.clone()
        Su.scatter_(1, Lj.argmax(-1, keepdim=True), float("inf"))
        add("F3x", l, S, Su)
        del Lj, Le, Lec, LC, Ljm, Lem, LCm, S, Su
        torch.cuda.empty_cache()

    # ---------------- (a)
    print("\n=== (a) per layer: norms, cosines, where C's top-1/top-10 come from (filtered, FIT)", flush=True)
    keys = ["norm_J", "norm_E", "norm_b", "cos_J_E", "cos_E_prior", "sd_J", "sd_E", "gap_J", "gap_E", "C1_eq_J1",
            "C1_eq_E1", "C1_inJ10", "C1_inE10", "C1_inJ10_notE10", "C1_inE10_notJ10", "C1_neither", "C10_inJ10",
            "C10_inE10", "C10_Jonly", "C10_Eonly", "C10_neither", "C1_in_prior20", "E1_in_prior20"]
    print("layer " + " ".join(f"{k:>9s}"[-9:] for k in keys))
    for l, s in stats_a.items():
        print(f"{l:5d} " + " ".join(f"{s[k]:9.3f}" for k in keys))
    for l, s in stats_a.items():
        print(f"  L{l} map prior top-10: {s['prior_top10']}")

    # ---------------- summaries
    summ = {k: summarize(ctx, *v) for k, v in variants.items()}
    print("\n=== FIT summaries", flush=True)
    for k in ["J++", "C", "map_alone", "F3x"] + [f"mix{g}" for g in GRID] + [f"cen{g}" for g in GRID] + \
            [f"F1x_K{K}_mu{mu}" for K in KS for mu in MUS]:
        print(fmt(k, summ[k]))

    # ---------------- (b) rank-1 flips, (c) rank<=10 gains/losses
    rkJ, rkJu = variants["J++"]
    rkC, rkCu = variants["C"]
    bJ = {f: [{k: min(r[l][p][k] for l in r) for k in cands[p]} for p in range(n)] for f, r in (("f", rkJ), ("u", rkJu))}
    bC = {f: [{k: min(r[l][p][k] for l in r) for k in cands[p]} for p in range(n)] for f, r in (("f", rkC), ("u", rkCu))}
    top1_count = {f: Counter() for f in "fu"}
    for l in ctx.readout_layers:
        for p in range(n):
            top1_count["f"][int(top[l]["C1"][p])] += 1
            top1_count["u"][int(top[l]["C1u"][p])] += 1

    def norm_s(s):
        return s.replace("Ġ", "").strip().lower()

    def kind(tid, inter, p, l, f):
        s = norm_s(dec(tid))
        c = inter.lower().strip()
        cand_s = {norm_s(dec(t)) for t in cands[p][inter]}
        others = {k.lower() for k in cands[p] if k != inter}
        if int(tid) in set(ctx.fit.excluded_ids.tolist()):
            return "junk(excluded)"
        if len(s) <= 2 or not any(ch.isalpha() for ch in s) or s in STOP:
            return "function/junk"
        if s in others or any(s and (s in o or o in s) and len(s) >= 3 for o in others):
            return "other intermediate"
        if len(s) >= 3 and (s in c or c in s or s[:4] == c[:4] or any(s in x or x in s for x in cand_s if x)):
            return "other form of concept"
        if tid in top[l]["prior20"] or top1_count[f][int(tid)] >= max(5, n * len(ctx.readout_layers) // 20):
            return "generic topic word"
        return "other content word"

    rows = []

    def flips(f, A, B, rkA, rkB, nameA, nameB, k):
        """items/intermediates where A has best rank <= k and B does not."""
        out = []
        for p in range(n):
            for inter in cands[p]:
                if A[f][p][inter] <= k < B[f][p][inter]:
                    layers = [l for l in ctx.readout_layers if rkA[l][p][inter] <= k]
                    out.append((p, inter, layers))
        return out

    print("\n=== (b) rank-1 flips (FIT)", flush=True)
    res_b = {}
    for f, (rJ, rC) in (("f", (rkJ, rkC)), ("u", (rkJu, rkCu))):
        lost = flips(f, bJ, bC, rJ, rC, "J++", "C", 1)
        won = flips(f, bC, bJ, rC, rJ, "C", "J++", 1)
        print(f"-- {'filtered' if f == 'f' else 'unfiltered'}: J++ rank1 lost by C: {len(lost)}; C rank1 gained: {len(won)}")
        kinds = Counter()
        by_task = Counter(ctx.fit.eval_slugs[p] for p, _, _ in lost)
        by_task_w = Counter(ctx.fit.eval_slugs[p] for p, _, _ in won)
        print(f"   lost by task {dict(by_task)}; gained by task {dict(by_task_w)}")
        for p, inter, layers in lost:
            for l in layers:
                t = int(top[l]["C1" if f == "f" else "C1u"][p])
                tt = top[l][f]
                kd = kind(t, inter, p, l, f)
                kinds[kd] += 1
                r = dict(dir="lost", filt=f, task=ctx.fit.eval_slugs[p], item=ctx.fit.item_names[p], inter=inter,
                         layer=l, displacer=dec(t), kind=kd,
                         in_map10=bool((tt["E"][p] == t).any()), in_jpp10=bool((tt["J"][p] == t).any()),
                         excluded=t in set(ctx.fit.excluded_ids.tolist()), in_prior20=t in top[l]["prior20"],
                         new_rank_layer=rC[l][p][inter], new_best=bC[f][p][inter],
                         jpp_top1=dec(int(top[l]["J1" if f == "f" else "J1u"][p])))
                rows.append(r)
                print(f"   LOST {r['task'][:6]:6s} {r['item'][:28]:28s} {inter!r:16s} L{l:2d} by {r['displacer']!r:14s} "
                      f"[{kd}] map10={r['in_map10']:d} jpp10={r['in_jpp10']:d} excl={r['excluded']:d} "
                      f"prior20={r['in_prior20']:d} newrank@L={r['new_rank_layer']} newbest={r['new_best']}")
        print(f"   displacer kinds (per flip-layer): {dict(kinds)}")
        wk = Counter()
        for p, inter, layers in won:
            for l in layers:
                t = int(top[l]["J1" if f == "f" else "J1u"][p])
                tt = top[l][f]
                kd = kind(t, inter, p, l, f)
                wk[kd] += 1
                cin = any(bool((tt["E"][p] == c).any()) for c in cands[p][inter])
                r = dict(dir="won", filt=f, task=ctx.fit.eval_slugs[p], item=ctx.fit.item_names[p], inter=inter,
                         layer=l, jpp_top1=dec(t), kind=kd, concept_in_map10=cin,
                         jpp_rank_layer=rJ[l][p][inter], jpp_best=bJ[f][p][inter])
                rows.append(r)
                print(f"   WON  {r['task'][:6]:6s} {r['item'][:28]:28s} {inter!r:16s} L{l:2d} J++ had {r['jpp_top1']!r:14s} "
                      f"[{kd}] concept_in_map10={cin:d} jpp@L={r['jpp_rank_layer']} jppbest={r['jpp_best']}")
        print(f"   kinds of J++'s top-1 that C displaced in its favour: {dict(wk)}")
        res_b[f] = dict(lost=len(lost), won=len(won), kinds=dict(kinds), won_kinds=dict(wk),
                        lost_task=dict(by_task), won_task=dict(by_task_w))

    print("\n=== (c) rank<=10 gains/losses by task (FIT)", flush=True)
    res_c = {}
    for f, (rJ, rC) in (("f", (rkJ, rkC)), ("u", (rkJu, rkCu))):
        lost = flips(f, bJ, bC, rJ, rC, "J++", "C", 10)
        won = flips(f, bC, bJ, rC, rJ, "C", "J++", 10)
        lt = Counter(ctx.fit.eval_slugs[p] for p, _, _ in lost)
        wt = Counter(ctx.fit.eval_slugs[p] for p, _, _ in won)
        print(f"-- {'filtered' if f == 'f' else 'unfiltered'}: lost {len(lost)} {dict(lt)}; gained {len(won)} {dict(wt)}")
        intr = Counter()
        for p, inter, layers in lost:
            l = min(layers, key=lambda x: rJ[x][p][inter])
            tt = top[l][f]
            new = [int(t) for t in tt["C"][p] if not (tt["J"][p] == t).any()]
            ks = [kind(t, inter, p, l, f) for t in new]
            intr.update(ks)
            src = ["map" if (tt["E"][p] == t).any() else "mix" for t in new]
            print(f"   LOST10 {ctx.fit.eval_slugs[p][:6]:6s} {ctx.fit.item_names[p][:26]:26s} {inter!r:16s} L{l:2d} "
                  f"jpp@L={rJ[l][p][inter]} C@L={rC[l][p][inter]} Cbest={bC[f][p][inter]} intruders "
                  + ", ".join(f"{dec(t)!r}({s},{k[:7]})" for t, s, k in zip(new, src, ks)))
        for p, inter, layers in won:
            l = min(layers, key=lambda x: rC[x][p][inter])
            print(f"   WON10  {ctx.fit.eval_slugs[p][:6]:6s} {ctx.fit.item_names[p][:26]:26s} {inter!r:16s} L{l:2d} "
                  f"C@L={rC[l][p][inter]} jpp@L={rJ[l][p][inter]} jppbest={bJ[f][p][inter]}")
        print(f"   intruder kinds in lost-10 cases: {dict(intr)}")
        res_c[f] = dict(lost=len(lost), won=len(won), lost_task=dict(lt), won_task=dict(wt), intruders=dict(intr))

    # ---------------- (d) per-layer carry
    print("\n=== (d) per layer: single-layer pass@1/@10 and share of best ranks carried (filtered, FIT)", flush=True)
    res_d = {}
    for name in ["J++", "C"]:
        rf, _ = variants[name]
        b = best_of_name = [{k: min(rf[l][p][k] for l in rf) for k in cands[p]} for p in range(n)]
        carry1, carry10 = Counter(), Counter()
        for p in range(n):
            for k in cands[p]:
                for l in rf:
                    if rf[l][p][k] == b[p][k] and b[p][k] <= 1:
                        carry1[l] += 1
                    if rf[l][p][k] == b[p][k] and b[p][k] <= 10:
                        carry10[l] += 1
        line = []
        for l in ctx.readout_layers:
            one = {l: rf[l]}
            s = summarize(ctx, one, one)
            line.append((l, s["p1"], s["p10"], carry1[l], carry10[l]))
        res_d[name] = line
        print(f"  {name}: " + "  ".join(f"L{l} p1 {a:.3f} p10 {b_:.3f} c1 {c} c10 {d_}" for l, a, b_, c, d_ in line))
    del best_of_name

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"diag_{tag}.json").write_text(json.dumps(dict(
        stats_a=stats_a, summaries={k: v for k, v in summ.items()}, flips_b=res_b, flips_c=res_c, layers_d=res_d,
        rows=rows), indent=1, default=str))
    print(f"[fm27] wrote {OUT / f'diag_{tag}.json'}", flush=True)
    return lensC
