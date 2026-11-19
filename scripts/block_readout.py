"""Label-free ways to combine per-feature (per-block) lens readouts, plus controls.

For each dictionary layer, bench item and family (trained dictionary, PCA and random-basis
controls of the same space / b / l0 / mean) the active blocks' final-basis vectors v_g [l0, d]
are unembedded one by one, L_g = model.unembed(v_g) [l0, V], and combined into score vectors:

  mix     mean_g softmax(L_g)              mixw    sum_g a_g softmax(L_g), a_g = |v_g| / sum |v_g|
  mixT2   mean_g softmax(L_g / 2)          mixT05  mean_g softmax(L_g / 0.5)
  maxp    max_g softmax(L_g)               maxz    max_g (L_g - mean_t L_g) / std_t L_g
  rrf     sum_g 1 / (60 + rank_g(t))       sum     unembed(sum_g v_g)  (= block_attrib ".blocks")
  oracle1 min_g rank of the target under L_g (label-selected upper bound)

plus `jlens` = unembed(h @ J.T) once per layer. Scores are ranked like logits (1 + #strictly
greater than the best candidate); "filtered" sets bench.excluded_ids to -inf in the score vector.

    python scripts/block_readout.py --bench artifacts/bench/qwen3.5-2b.pt \
        --jlens artifacts/baselines/jlens_prefit.pt --dicts runs/bsf1/L18_jspace_b1_10M.pt \
        --out runs/bsf1/readout_L18_jspace_b1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import olens  # noqa: E402,F401  registers the trained lens type
from block_attrib import BasisDecomp, TrainedDecomp, load_val_acts  # noqa: E402
from olens.bsf_io import BlockDict  # noqa: E402
from olens.readout import Bench, candidate_ranks, readout_rows, write_csv  # noqa: E402

DEV = "cuda"
NEG_INF = float("-inf")
RRF_K = 60.0
AGGS = ("mix", "mixw", "mixT2", "mixT05", "maxp", "maxz", "rrf", "sum")  # + oracle1


@t.no_grad()
def pca_basis(X: t.Tensor, mean: t.Tensor, space: str, J: t.Tensor) -> t.Tensor:
    """Eigenvectors of the val-act covariance (about the dictionary mean), descending; as block_attrib."""
    d = mean.shape[0]
    cov = t.zeros(d, d, dtype=t.float64, device=DEV)
    for s in range(0, X.shape[0], 8192):
        xb = X[s : s + 8192].to(DEV).float()
        if space == "jspace":
            xb = xb @ J.T
        xb = (xb - mean.float()).double()
        cov += xb.T @ xb
    evals, evecs = t.linalg.eigh(cov / X.shape[0])
    return evecs.flip(-1).float().contiguous()


@t.no_grad()
def item_scores(model, Vg: t.Tensor, v_sum: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Vg [l0, d] final-basis block vectors of one item. Returns (scores [len(AGGS), V], L_g [l0, V])."""
    L = model.unembed(Vg).float()  # [l0, V]
    l0, V = L.shape
    a = Vg.norm(dim=-1)
    a = a / a.sum().clamp_min(1e-30)
    p1 = t.softmax(L, dim=-1)
    mix = p1.mean(0)
    mixw = (a[:, None] * p1).sum(0)
    maxp = p1.max(0).values
    del p1
    mixT2 = t.softmax(L / 2.0, dim=-1).mean(0)
    mixT05 = t.softmax(L / 0.5, dim=-1).mean(0)
    mu, sd = L.mean(-1, keepdim=True), L.std(-1, keepdim=True).clamp_min(1e-12)
    maxz = ((L - mu) / sd).max(0).values
    order = L.argsort(dim=-1, descending=True)  # [l0, V]
    rk = t.empty_like(order)
    rk.scatter_(1, order, t.arange(1, V + 1, device=L.device).expand(l0, V))
    del order
    rrf = (1.0 / (RRF_K + rk.float())).sum(0)
    del rk
    s = model.unembed(v_sum[None]).float()[0]
    S = t.stack([mix, mixw, mixT2, mixT05, maxp, maxz, rrf, s])
    return S, L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--dicts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--controls", default="pca,rand", help="comma list from {pca,rand}; '' for none")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.set_grad_enabled(False)
    controls = [c for c in args.controls.split(",") if c]
    assert set(controls) <= {"pca", "rand"}, controls

    from workspace_lens import get_hf_model
    from workspace_lens.utils import load_lens_file

    # ---- dictionaries: one space, one b, at most one per layer
    paths = [Path(p) for p in args.dicts]
    metas = {p: t.load(p, map_location="cpu", weights_only=True, mmap=True)["meta"] for p in paths}
    spaces = {m["space"] for m in metas.values()}
    bs = {m["group_size"] for m in metas.values()}
    assert len(spaces) == 1 and len(bs) == 1, f"dicts must share space and b: {spaces}, {bs}"
    lay = [m["layer"] for m in metas.values()]
    assert len(set(lay)) == len(lay), f"more than one dictionary per layer: {lay}"
    space, b = spaces.pop(), bs.pop()
    fam_trained = f"{space}_b{b}" + (f"_{args.tag}" if args.tag else "")

    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    bench = Bench.load(args.bench)
    lens = load_lens_file(args.jlens, hf_model_name=args.model)
    P_all = bench.num_items
    items = list(range(P_all if args.max_items is None else min(args.max_items, P_all)))
    excl = bench.excluded_ids.to(DEV).long()
    held_set = {p for p in items if bench.split[p] == "held-out"}
    fit_set = {p for p in items if bench.split[p] == "fit"}
    item_idx = t.tensor(items, device=DEV)

    rows = {(sp, f): [] for sp in ("held", "fit") for f in (True, False)}

    def add_rows(lens_name, layer, p, ranks_f, ranks_u):
        sp = "held" if p in held_set else "fit" if p in fit_set else None
        if sp is None:
            return
        for f, rk in ((True, ranks_f), (False, ranks_u)):
            for inter, r in rk.items():
                rows[(sp, f)].append({
                    "eval": bench.eval_slugs[p], "lens": lens_name, "item": bench.item_names[p],
                    "intermediate": inter, "layer": int(layer), "rank": int(r),
                    "n_candidates": len(bench.candidates[p][inter]),
                })

    rand_Q = None
    t0 = time.time()
    for path in sorted(paths, key=lambda q: metas[q]["layer"]):
        layer = metas[path]["layer"]
        J = lens.jacobians_L_dict_FN[layer].to(DEV).float()
        h = bench.resid[layer][items].to(DEV).float()
        v_full = h @ J.T
        print(f"== layer {layer}: {len(items)} items, dict {path.name}", flush=True)
        held_l = [p for p in items if p in held_set]
        fit_l = [p for p in items if p in fit_set]
        for f in (True, False):
            for sp, it in (("held", held_l), ("fit", fit_l)):
                if it:
                    rows[(sp, f)] += readout_rows("jlens", layer, bench, v_full[[items.index(p) for p in it]],
                                                  model, filtered=f, items=it)

        D = BlockDict.load(path, DEV)
        fams = [(fam_trained, TrainedDecomp(D, fam_trained))]
        if "pca" in controls:
            X = load_val_acts(path, layer)
            if X is None:
                print(f"  WARNING no val_acts_L{layer}.pt next to {path}; no PCA control")
            else:
                Q = pca_basis(X, D.mean, space, J)
                del X
                fams.append((f"pca_{space}_b{b}", BasisDecomp("pca", Q, D.mean, space=space, layer=layer, b=b, l0=D.l0)))
        if "rand" in controls:
            if rand_Q is None:
                g = t.Generator(device="cpu").manual_seed(args.seed)
                rand_Q = t.linalg.qr(t.randn(D.d, D.d, generator=g, dtype=t.float64)).Q.float().to(DEV)
            fams.append((f"rand_{space}_b{b}", BasisDecomp("rand", rand_Q, D.mean, space=space, layer=layer, b=b, l0=D.l0)))

        x = h if space == "resid" else v_full
        splits = {}
        for fam, dc in fams:
            _, per, _, _ = dc.split(x)  # per [P, l0, d] in dictionary space
            Vg = per @ J.T if space == "resid" else per
            splits[fam] = Vg
            del per

        tl = time.time()
        for n, p in enumerate(items):
            cands = [bench.candidates[p]]
            for fam, _ in fams:
                Vg = splits[fam][n]  # [l0, d]
                S, L = item_scores(model, Vg, Vg.sum(0))
                A = S.shape[0]
                ru = candidate_ranks(S, cands * A)
                rf = candidate_ranks(S.index_fill(1, excl, NEG_INF), cands * A)
                for k, agg in enumerate(AGGS):
                    add_rows(f"{fam}.{agg}", layer, p, rf[k], ru[k])
                del S
                l0 = L.shape[0]
                gu = candidate_ranks(L, cands * l0)
                gf = candidate_ranks(L.index_fill(1, excl, NEG_INF), cands * l0)
                del L
                ou = {i: min(r[i] for r in gu) for i in gu[0]}
                of = {i: min(r[i] for r in gf) for i in gf[0]}
                add_rows(f"{fam}.oracle1", layer, p, of, ou)
            if (n + 1) % 50 == 0 or n + 1 == len(items):
                dt = time.time() - tl
                print(f"  L{layer} item {n + 1}/{len(items)}: {dt:.1f}s ({dt / (n + 1):.3f} s/item, "
                      f"{len(fams)} families); total {time.time() - t0:.0f}s", flush=True)
        D.cpu()
        del splits, fams, D
        t.cuda.empty_cache()

    write_csv(out / "readout_filtered.csv", rows[("held", True)])
    write_csv(out / "readout_unfiltered.csv", rows[("held", False)])
    write_csv(out / "readout_fit_filtered.csv", rows[("fit", True)])
    write_csv(out / "readout_fit_unfiltered.csv", rows[("fit", False)])

    # ---- quick table: held-out, filtered, best rank over layers per (item, intermediate)
    best: dict[str, dict] = {}
    for r in rows[("held", True)]:
        key = (r["eval"], r["item"], r["intermediate"])
        d = best.setdefault(r["lens"], {})
        d[key] = min(d.get(key, r["rank"]), r["rank"])
    print(f"\nheld-out pairs, filtered, best over layers ({len(best.get('jlens', {}))} pairs)")
    print(f"{'lens':<32s} {'r<=10':>7s} {'r<=1':>7s}")
    for name in sorted(best, key=lambda s: (s != "jlens", s)):
        v = list(best[name].values())
        print(f"{name:<32s} {sum(x <= 10 for x in v) / len(v):7.3f} {sum(x <= 1 for x in v) / len(v):7.3f}")
    print(f"\nwrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
