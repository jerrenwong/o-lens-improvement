"""Which label-free property of an active dictionary feature identifies the "carrier" of the target?

Per layer, bench item and active block g (final-basis vector v_g, L_g = model.unembed(v_g)) this
computes per-occurrence properties (strength, share, idf, rel, p1, ent, maxz, kurt, jgain, freq),
the target's rank under L_g alone and is_carrier = solo filtered rank <= 10 -> occurrences.csv.
Selectors score the active features (higher = first); recall@m of carriers among the top-m ->
selector_recall.csv; readouts of the top-m features (sum of vectors, mixture of softmaxes) ->
readout_*.csv in the harness schema, with jlens, <family>.sum and <family>.oracle1 for reference.
`learned` is a logistic regression fit on FIT-split occurrences (pooled over layers).

    python scripts/feature_select.py --bench artifacts/bench/qwen3.5-2b.pt \
        --jlens artifacts/baselines/jlens_prefit.pt --dicts runs/bsf1/L18_jspace_b1_10M.pt \
        --out runs/bsf1/fsel_jspace_b1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import olens  # noqa: E402,F401  registers the trained lens type
from block_attrib import TrainedDecomp  # noqa: E402
from block_readout import DEV, NEG_INF  # noqa: E402
from olens.bsf_io import BlockDict  # noqa: E402
from olens.readout import Bench, candidate_ranks, readout_rows, write_csv  # noqa: E402

CARRIER_K = 10
RECALL_MS = (1, 2, 3, 5, 10, 20)
READ_MS = (1, 3, 5, 10)
SELECTORS = ("strength", "idf", "rel", "p1", "maxz", "kurt", "negent", "strength*idf", "rel*p1", "idf*p1",
             "rel*idf", "rel*idf*p1", "random", "learned")
READ_SELECTORS = ("strength", "idf", "rel", "p1", "maxz", "rel*idf", "rel*idf*p1", "learned", "random")
PROPS = ("strength", "share", "idf", "rel", "p1", "ent", "maxz", "kurt", "jgain", "freq")
PI = {k: i for i, k in enumerate(PROPS)}
LR_INPUTS = ("log strength", "log share", "idf", "log rel", "log p1", "ent", "maxz", "kurt", "log jgain")


def lr_inputs(F: t.Tensor) -> t.Tensor:
    """F [N, len(PROPS)] -> logistic-regression inputs [N, 9]."""
    lg = lambda k: F[:, PI[k]].clamp_min(1e-30).log()  # noqa: E731
    return t.stack([lg("strength"), lg("share"), F[:, PI["idf"]], lg("rel"), lg("p1"), F[:, PI["ent"]],
                    F[:, PI["maxz"]], F[:, PI["kurt"]], lg("jgain")], 1)


def selector_scores(F: t.Tensor, name: str, lr=None, gen=None) -> t.Tensor:
    """F [l0, len(PROPS)] -> score [l0] (higher = selected first)."""
    c = lambda k: F[:, PI[k]]  # noqa: E731
    if name == "negent":
        return -c("ent")
    if name == "random":
        return t.rand(F.shape[0], generator=gen)
    if name == "learned":
        mu, sd, w, b0 = lr
        return ((lr_inputs(F) - mu) / sd) @ w + b0
    out = t.ones(F.shape[0])
    for k in name.split("*"):
        out = out * c(k)
    return out


def fit_logreg(X: t.Tensor, y: t.Tensor, l2: float = 1e-2):
    mu, sd = X.mean(0), X.std(0)
    sd = t.where(sd > 1e-8, sd, t.ones_like(sd))
    Z = ((X - mu) / sd).double()
    yd = y.double()
    w = t.zeros(Z.shape[1], dtype=t.float64, requires_grad=True)
    b0 = t.zeros((), dtype=t.float64, requires_grad=True)
    opt = t.optim.LBFGS([w, b0], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = t.nn.functional.binary_cross_entropy_with_logits(Z @ w + b0, yd) + l2 * (w * w).sum()
        loss.backward()
        return loss

    with t.enable_grad():
        opt.step(closure)
    return mu, sd, w.detach().float(), float(b0.detach())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--dicts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.set_grad_enabled(False)

    from transformers import AutoTokenizer
    from workspace_lens import get_hf_model
    from workspace_lens.utils import load_lens_file

    paths = [Path(p) for p in args.dicts]
    metas = {p: t.load(p, map_location="cpu", weights_only=True, mmap=True)["meta"] for p in paths}
    spaces = {m["space"] for m in metas.values()}
    bs = {m["group_size"] for m in metas.values()}
    assert len(spaces) == 1 and len(bs) == 1, f"dicts must share space and b: {spaces}, {bs}"
    lay = [m["layer"] for m in metas.values()]
    assert len(set(lay)) == len(lay), f"more than one dictionary per layer: {lay}"
    space, b = spaces.pop(), bs.pop()
    fam = f"{space}_b{b}" + (f"_{args.tag}" if args.tag else "")

    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    bench = Bench.load(args.bench)
    lens = load_lens_file(args.jlens, hf_model_name=args.model)
    P_all = bench.num_items
    items = list(range(P_all if args.max_items is None else min(args.max_items, P_all)))
    excl = bench.excluded_ids.to(DEV).long()
    split_of = {p: ("held" if bench.split[p] == "held-out" else "fit" if bench.split[p] == "fit" else None) for p in items}

    rows = {(sp, f): [] for sp in ("held", "fit") for f in (True, False)}

    def add_rows(lens_name, layer, p, ranks_f, ranks_u):
        sp = split_of[p]
        if sp is None:
            return
        for f, rk in ((True, ranks_f), (False, ranks_u)):
            for inter, r in rk.items():
                rows[(sp, f)].append({
                    "eval": bench.eval_slugs[p], "lens": lens_name, "item": bench.item_names[p],
                    "intermediate": inter, "layer": int(layer), "rank": int(r),
                    "n_candidates": len(bench.candidates[p][inter]),
                })

    def rank_fu(S, cands):
        """S [A, V] scores of one item -> (filtered, unfiltered) lists of {intermediate: rank}."""
        c = cands * S.shape[0]
        return candidate_ranks(S.index_fill(1, excl, NEG_INF), c), candidate_ranks(S, c)

    # ------------------------------------------------------------------ pass 1: properties, solo ranks
    t0 = time.time()
    store = {}  # layer -> dict(Vg cpu [P,l0,d], F [P,l0,nprops], blocks [P,l0], solo_f/solo_u list, sum ranks)
    occ = []
    for path in sorted(paths, key=lambda q: metas[q]["layer"]):
        layer = metas[path]["layer"]
        J = lens.jacobians_L_dict_FN[layer].to(DEV).float()
        h = bench.resid[layer][items].to(DEV).float()
        v_full = h @ J.T
        print(f"== layer {layer}: {len(items)} items, dict {path.name}", flush=True)
        for f in (True, False):
            for sp in ("held", "fit"):
                it = [p for p in items if split_of[p] == sp]
                if it:
                    rows[(sp, f)] += readout_rows("jlens", layer, bench, v_full[[items.index(p) for p in it]],
                                                  model, filtered=f, items=it)
        D = BlockDict.load(path, DEV)
        sp_path = path.with_name(path.stem + "_stats.pt")
        st = t.load(sp_path, map_location="cpu", weights_only=True)
        fc = st["fire_count"].float().to(DEV)
        freq_all = fc / float(st["rows"])
        rms_all = (st["code_cov"].float().to(DEV).diagonal(dim1=-2, dim2=-1).sum(-1) / fc.clamp_min(1)).sqrt()
        x = h if space == "resid" else v_full
        dc = TrainedDecomp(D, fam)
        idx, per, _, _ = dc.split(x)  # per [P, l0, d] in dictionary space
        z = D.encode(x)
        ar = t.arange(len(items), device=DEV)[:, None]
        znorm = z[ar, idx].float().norm(dim=-1)  # [P, l0]
        del z
        Vg = per @ J.T if space == "resid" else per
        strength = Vg.norm(dim=-1)
        jgain = strength / per.norm(dim=-1).clamp_min(1e-30) if space == "resid" else t.ones_like(strength)
        del per
        freq = freq_all[idx]
        base = {
            "strength": strength, "share": strength / strength.sum(-1, keepdim=True).clamp_min(1e-30),
            "idf": -freq.clamp_min(1e-7).log(), "rel": znorm / rms_all[idx].clamp_min(1e-30),
            "jgain": jgain, "freq": freq,
        }
        l0 = idx.shape[1]
        Fl = t.zeros(len(items), l0, len(PROPS))
        solo_f, solo_u, sum_f = [], [], []
        tl = time.time()
        for n, p in enumerate(items):
            cands = [bench.candidates[p]]
            L = model.unembed(Vg[n]).float()  # [l0, V]
            logP = t.log_softmax(L, dim=-1)
            Pm = logP.exp()
            p1 = Pm.max(-1).values
            ent = -(Pm * logP).sum(-1)
            del Pm, logP
            mu, sd = L.mean(-1, keepdim=True), L.std(-1, keepdim=True).clamp_min(1e-12)
            zz = (L - mu) / sd
            maxz = zz.max(-1).values
            kurt = (zz**4).mean(-1) - 3.0
            del zz
            top = L.argmax(-1).tolist()
            gf, gu = rank_fu(L, cands)
            del L
            vals = dict(p1=p1, ent=ent, maxz=maxz, kurt=kurt)
            Fl[n] = t.stack([(vals[k] if k in vals else base[k][n]) for k in PROPS], 1).cpu()
            solo_f.append(gf)
            solo_u.append(gu)
            s_f, s_u = rank_fu(model.unembed(Vg[n].sum(0)[None]).float(), cands)
            sum_f.append(s_f[0])
            add_rows(f"{fam}.sum", layer, p, s_f[0], s_u[0])
            of = {i: min(r[i] for r in gf) for i in gf[0]}
            ou = {i: min(r[i] for r in gu) for i in gu[0]}
            add_rows(f"{fam}.oracle1", layer, p, of, ou)
            blocks = idx[n].tolist()
            for i in bench.candidates[p]:
                if not bench.candidates[p][i]:
                    continue
                anyc = any(gf[k][i] <= CARRIER_K for k in range(l0))
                assert (of[i] <= CARRIER_K) == anyc, "oracle1 hit <=> some carrier violated"
                for k in range(l0):
                    fr = Fl[n, k].tolist()
                    occ.append({
                        "layer": layer, "eval": bench.eval_slugs[p], "item": bench.item_names[p],
                        "split": bench.split[p], "intermediate": i, "slot": k, "block": blocks[k],
                        **{kk: fr[PI[kk]] for kk in ("strength", "share", "idf", "rel", "p1", "ent", "maxz",
                                                      "kurt", "jgain", "freq")},
                        "solo_rank_filtered": gf[k][i], "solo_rank_unfiltered": gu[k][i],
                        "is_carrier": int(gf[k][i] <= CARRIER_K), "top_token": repr(tok.decode([top[k]])),
                    })
            if (n + 1) % 50 == 0 or n + 1 == len(items):
                dt = time.time() - tl
                print(f"  pass1 L{layer} item {n + 1}/{len(items)}: {dt:.1f}s ({dt / (n + 1):.3f} s/item); "
                      f"total {time.time() - t0:.0f}s", flush=True)
        store[layer] = dict(Vg=Vg.cpu(), F=Fl, solo_f=solo_f, solo_u=solo_u, sum_f=sum_f)
        D.cpu()
        del D, Vg, dc
        t.cuda.empty_cache()

    with open(out / "occurrences.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(occ[0].keys()))
        w.writeheader()
        w.writerows(occ)
    del occ

    # ------------------------------------------------------------------ learned selector
    Xs, ys = [], []
    for layer, S in store.items():
        for n, p in enumerate(items):
            if split_of[p] != "fit":
                continue
            gf = S["solo_f"][n]
            Xs.append(lr_inputs(S["F"][n]))
            ys.append(t.tensor([float(any(v <= CARRIER_K for v in gf[k].values())) for k in range(len(gf))]))
    lr = None
    if Xs:
        X, y = t.cat(Xs), t.cat(ys)
        lr = fit_logreg(X, y)
        print(f"\nlearned selector: {len(y)} fit occurrences, {int(y.sum())} carriers; standardised coefficients "
              f"(bias {lr[3]:+.3f}):")
        for nm, c in zip(LR_INPUTS, lr[2].tolist()):
            print(f"  {nm:<12s} {c:+.3f}")
    sels = [s for s in SELECTORS if s != "learned" or lr is not None]

    # ------------------------------------------------------------------ pass 2: selection, recall, readouts
    hits = {}  # (selector, m, split) -> {(item, inter, layer): bool}
    gen = t.Generator().manual_seed(args.seed)
    for layer, S in store.items():
        tl = time.time()
        for n, p in enumerate(items):
            cands = [bench.candidates[p]]
            F = S["F"][n]
            l0 = F.shape[0]
            gf = S["solo_f"][n]
            sp = split_of[p]
            orders = {s: selector_scores(F, s, lr, gen).argsort(descending=True) for s in sels}
            for s in sels:
                for i in gf[0]:
                    carr = t.tensor([gf[k][i] <= CARRIER_K for k in range(l0)])
                    if not carr.any():
                        continue
                    for m in RECALL_MS:
                        hits.setdefault((s, m, sp), {})[(p, i, layer)] = bool(carr[orders[s][:m]].any())
            # readouts
            Vg = S["Vg"][n].to(DEV)
            L = model.unembed(Vg).float()
            Pm = t.softmax(L, dim=-1)
            del L
            names, sums, mixes = [], [], []
            for s in READ_SELECTORS:
                if s not in orders:
                    continue
                for m in READ_MS:
                    mask = t.zeros(l0, device=DEV)
                    mask[orders[s][:m].to(DEV)] = 1.0
                    names.append(f"{fam}.sel.{s}.m{m}")
                    sums.append(model.unembed(((mask[:, None] * Vg).sum(0))[None]).float()[0])
                    mixes.append((mask[:, None] * Pm).sum(0) / m)
            if n < 5:  # selecting all l0 features must reproduce <family>.sum
                allv = model.unembed(((t.ones(l0, device=DEV)[:, None] * Vg).sum(0))[None]).float()
                af, _ = rank_fu(allv, cands)
                assert af[0] == S["sum_f"][n], f"all-feature selection {af[0]} != {fam}.sum {S['sum_f'][n]}"
            del Pm
            for kind, T in (("sum", sums), ("mix", mixes)):
                rf, ru = rank_fu(t.stack(T), cands)
                for j, nm in enumerate(names):
                    add_rows(f"{nm}.{kind}", layer, p, rf[j], ru[j])
            del sums, mixes
            if (n + 1) % 50 == 0 or n + 1 == len(items):
                dt = time.time() - tl
                print(f"  pass2 L{layer} item {n + 1}/{len(items)}: {dt:.1f}s; total {time.time() - t0:.0f}s", flush=True)
        print("  all-feature selection == sum ranks on the first 5 items: OK", flush=True)

    # ------------------------------------------------------------------ recall table
    rec_rows = []
    for s in sels:
        for sp in ("held", "fit"):
            for m in RECALL_MS:
                hd = hits.get((s, m, sp), {})
                anyl = {}
                for (p, i, _), v in hd.items():
                    anyl[(p, i)] = anyl.get((p, i), False) or v
                rec_rows.append({
                    "selector": s, "split": sp, "m": m, "n_triples": len(hd),
                    "recall": sum(hd.values()) / len(hd) if hd else float("nan"),
                    "n_pairs": len(anyl),
                    "anylayer_recall": sum(anyl.values()) / len(anyl) if anyl else float("nan"),
                })
    with open(out / "selector_recall.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec_rows[0].keys()))
        w.writeheader()
        w.writerows(rec_rows)
    print(f"\nheld-out recall@m (triples with a carrier: {[r['n_triples'] for r in rec_rows if r['split'] == 'held'][0]}"
          f"; any-layer pairs: {[r['n_pairs'] for r in rec_rows if r['split'] == 'held'][0]})  [any-layer in brackets]")
    print(f"{'selector':<14s} " + " ".join(f"{'@' + str(m):>13s}" for m in RECALL_MS))
    for s in sels:
        rr = {r["m"]: r for r in rec_rows if r["selector"] == s and r["split"] == "held"}
        print(f"{s:<14s} " + " ".join(f"{rr[m]['recall']:6.3f} [{rr[m]['anylayer_recall']:.2f}]" for m in RECALL_MS))

    write_csv(out / "readout_filtered.csv", rows[("held", True)])
    write_csv(out / "readout_unfiltered.csv", rows[("held", False)])
    write_csv(out / "readout_fit_filtered.csv", rows[("fit", True)])
    write_csv(out / "readout_fit_unfiltered.csv", rows[("fit", False)])

    best: dict[str, dict] = {}
    for r in rows[("held", True)]:
        key = (r["eval"], r["item"], r["intermediate"])
        d = best.setdefault(r["lens"], {})
        d[key] = min(d.get(key, r["rank"]), r["rank"])
    print(f"\nheld-out pairs, filtered, best over layers ({len(best.get('jlens', {}))} pairs)")
    print(f"{'lens':<44s} {'r<=10':>7s} {'r<=1':>7s}")
    for name in sorted(best, key=lambda s: (s != "jlens", s)):
        v = list(best[name].values())
        print(f"{name:<44s} {sum(x <= 10 for x in v) / len(v):7.3f} {sum(x <= 1 for x in v) / len(v):7.3f}")
    print(f"\nwrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
