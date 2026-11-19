"""Do a few dictionary blocks carry the unspoken intermediate? Attribution + single-block readouts.

For each block-sparse dictionary (and two untrained controls of the same space / b / l0:
a seeded random orthogonal basis and the PCA basis of validation activations) this computes
  A. health (bench FVU, trainer meta, block stable ranks from the stats sidecar),
  B. readouts of the parts of the decomposition (recon, blocks, err, err+mean) and of single blocks
     (oracle1 = best active block for the target, label-selected; top1norm = largest block, label-free),
  C. the attribution of the target's centred logit f(v) = ((W[t*] - mean_t W[t]) * gamma) . v to the
     mean, each active block and the error.

    python scripts/block_attrib.py --bench artifacts/bench/qwen3.5-2b.pt \
        --jlens artifacts/baselines/jlens_prefit.pt --dicts runs/bsf1/L18_*_10M.pt --out runs/bsf1/attrib_10M
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import olens  # noqa: E402,F401  registers the trained lens type
from olens.bsf_io import BlockDict  # noqa: E402
from olens.readout import (  # noqa: E402
    Bench,
    candidate_ranks,
    final_linear,
    readout_rows,
    unembed_logits,
    write_csv,
)

CHUNK = 64
DEV = "cuda"
MIN_FIRES = 50
NEG_INF = float("-inf")


# ----------------------------------------------------------------------------- decompositions
class Decomp:
    """x == mean + blocks + err; blocks = sum_k per_block[:, k]; idx = global block index of slot k."""

    name: str
    kind: str  # "trained" | "rand" | "pca"
    space: str
    layer: int
    b: int
    l0: int
    mean: t.Tensor

    def split(self, x: t.Tensor):  # -> (idx [P,l0], per_block [P,l0,d], blocks [P,d], err [P,d])
        raise NotImplementedError


class TrainedDecomp(Decomp):
    def __init__(self, D: BlockDict, name: str) -> None:
        self.D, self.name, self.kind = D, name, "trained"
        self.space, self.layer, self.b, self.l0 = D.space, D.layer, D.group_size, D.l0
        self.mean = D.mean.float()

    @t.no_grad()
    def split(self, x):
        z, blocks, err = self.D.split(x)
        idx = z.norm(dim=-1).topk(self.l0, dim=-1).indices  # [P, l0]
        atoms = self.D.atoms().float()  # [G, b, d]
        scale = self.D.scale.float()
        ar = t.arange(x.shape[0], device=x.device)
        per = t.stack(
            [t.einsum("pb,pbd->pd", z[ar, idx[:, k]].float(), atoms[idx[:, k]]) for k in range(self.l0)], 1
        ) / scale
        dev = (per.sum(1) - blocks).norm() / blocks.norm().clamp_min(1e-12)
        if dev > 1e-3:
            print(f"  WARNING {self.name}: sum of per-block parts deviates from decode() by {dev:.2e}")
        return idx, per, blocks.float(), err.float()


class BasisDecomp(Decomp):
    """Orthonormal basis Q [d,d] (columns), blocks of b consecutive columns, keep the l0 largest."""

    def __init__(self, kind: str, Q: t.Tensor, mean: t.Tensor, *, space: str, layer: int, b: int, l0: int) -> None:
        d = Q.shape[0]
        assert d % b == 0, f"d={d} not divisible by b={b}"
        self.kind, self.space, self.layer, self.b, self.l0 = kind, space, layer, b, l0
        self.name = f"{kind}_{space}_b{b}"
        self.Q, self.mean = Q, mean.float()
        self.Qb = Q.T.reshape(d // b, b, d)  # block g's b basis vectors as rows

    @t.no_grad()
    def split(self, x):
        xc = x.float() - self.mean
        P, d = xc.shape
        coef = (xc @ self.Q).view(P, d // self.b, self.b)
        idx = coef.norm(dim=-1).topk(self.l0, dim=-1).indices
        ar = t.arange(P, device=x.device)
        per = t.stack(
            [t.einsum("pb,pbd->pd", coef[ar, idx[:, k]], self.Qb[idx[:, k]]) for k in range(self.l0)], 1
        )
        blocks = per.sum(1)
        return idx, per, blocks, xc - blocks


# ----------------------------------------------------------------------------- helpers
def nanmedian(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else None


def frac(xs, k):
    xs = [x for x in xs if x is not None]
    return sum(x <= k for x in xs) / len(xs) if xs else None


def load_meta_mean(path):
    try:
        ck = t.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception:
        ck = t.load(path, map_location="cpu", weights_only=True)
    return ck["meta"], ck["state"]["mean"].float().clone()


def stats_path(dict_path: Path) -> Path:
    return dict_path.with_name(dict_path.stem + "_stats.pt")


def load_val_acts(dict_path: Path, layer: int):
    p = dict_path.with_name(f"val_acts_L{layer}.pt")
    if not p.exists():
        return None
    obj = t.load(p, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):  # tolerate {"acts": tensor, ...}
        obj = next(v for v in obj.values() if isinstance(v, t.Tensor) and v.dim() == 2)
    return obj


def block_stable_ranks(dict_path: Path):
    sp = stats_path(dict_path)
    if not sp.exists():
        return {"stats_sidecar": None}
    st = t.load(sp, map_location="cpu", weights_only=True)
    if "fire_count" not in st or "code_cov" not in st:
        return {"stats_sidecar": str(sp), "stats_missing_keys": True}
    fc, cov = st["fire_count"].float(), st["code_cov"].double()
    keep = fc >= MIN_FIRES
    out = {"stats_sidecar": str(sp), "n_blocks_fire50": int(keep.sum())}
    if keep.sum() == 0:
        return out
    ev = t.linalg.eigvalsh(cov[keep]).clamp_min(0)  # [n, b]
    sr = ev.sum(-1) / ev.max(-1).values.clamp_min(1e-30)
    q = t.quantile(sr.float(), t.tensor([0.1, 0.5, 0.9]))
    out.update(sr_q10=float(q[0]), sr_q50=float(q[1]), sr_q90=float(q[2]), sr_share_gt1_5=float((sr > 1.5).float().mean()))
    return out


def json_scalars(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool)) or v is None}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--dicts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.set_grad_enabled(False)

    from workspace_lens import get_hf_model
    from workspace_lens.utils import load_lens_file

    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    bench = Bench.load(args.bench)
    lens = load_lens_file(args.jlens, hf_model_name=args.model)
    P = len(bench.candidates)
    held, fit = bench.index("held-out"), bench.index("fit")
    excl = bench.excluded_ids.to(DEV).long()
    W, gamma = final_linear(model)
    W, gamma = W.to(DEV).float(), gamma.to(DEV).float()
    Wbar = W.mean(0)

    def ekey(p):
        e = bench.eval_slugs[p] if len(bench.eval_slugs) == P else ""
        return str(e), str(bench.item_names[p])

    pairs = [(p, i) for p in range(P) for i in bench.candidates[p] if len(bench.candidates[p][i]) > 0]
    pair_p = t.tensor([p for p, _ in pairs], device=DEV)

    def rank_pass(V, rows_p, want_tstar=False):
        """Unembed rows V [N,d] (row r belongs to item rows_p[r]) in chunks; returns per-row rank dicts
        (unfiltered, filtered) and, optionally, per-row {intermediate: argmax candidate id (unfiltered)}."""
        ru, rf, ts = [], [], []
        for s in range(0, V.shape[0], CHUNK):
            cands = [bench.candidates[p] for p in rows_p[s : s + CHUNK]]
            lg = unembed_logits(model, V[s : s + CHUNK])
            ru += candidate_ranks(lg, cands)
            if want_tstar:
                for r, c in enumerate(cands):
                    ts.append({i: ids[int(lg[r, t.tensor(ids, device=lg.device)].argmax())] for i, ids in c.items() if len(ids)})
            rf += candidate_ranks(lg.index_fill(1, excl.to(lg.device), NEG_INF), cands)
            del lg
        return ru, rf, ts

    def rows_for(lens_name, layer, V, items, filtered):
        return readout_rows(lens_name, layer, bench, V, model, filtered=filtered, items=items) if items else []

    csv_rows = {(sp, f): [] for sp in ("held", "fit") for f in (True, False)}

    def emit(lens_name, layer, V):
        for f in (True, False):
            csv_rows[("held", f)] += rows_for(lens_name, layer, V, held, f)
            csv_rows[("fit", f)] += rows_for(lens_name, layer, V, fit, f)

    def emit_from_template(template, items, lens_name, rank_of):
        """Copy template rows (same items/intermediates), replacing lens and rank. rank_of(p, i) -> int."""
        order = [(p, i) for p in items for i in bench.candidates[p] if len(bench.candidates[p][i])]
        if len(template) == len(order) and all(str(r["intermediate"]) == str(i) for r, (_, i) in zip(template, order)):
            src = order
        else:  # fall back to matching by (eval, item, intermediate)
            m = {(*ekey(p), str(i)): (p, i) for p, i in order}
            src = [m.get((str(r.get("eval", "")) if len(bench.eval_slugs) == P else "", str(r["item"]), str(r["intermediate"]))) for r in template]
            if any(s is None for s in src):
                print(f"  WARNING could not align rows for {lens_name}; skipped")
                return []
        return [{**r, "lens": lens_name, "rank": rank_of(p, i)} for r, (p, i) in zip(template, src)]

    # ---- group dictionaries by layer
    paths = [Path(p) for p in args.dicts]
    metas = {p: load_meta_mean(p) for p in paths}
    layers = sorted({m["layer"] for m, _ in metas.values()})
    rand_Q, pca_Q = None, {}
    attrib_rows, summary, seen_names = [], {}, set()

    for layer in layers:
        lpaths = [p for p in paths if metas[p][0]["layer"] == layer]
        J = lens.jacobians_L_dict_FN[layer].to(DEV).float()
        h = bench.resid[layer].to(DEV).float()
        v_full = h @ J.T
        mean_src = next((p for p in lpaths if metas[p][0]["space"] == "resid"), lpaths[0])
        m0, mu0 = metas[mean_src]
        v_mean0 = mu0.to(DEV) @ J.T if m0["space"] == "resid" else mu0.to(DEV)
        print(f"== layer {layer}: P={P}, pairs={len(pairs)}, dicts={[p.name for p in lpaths]}")
        emit("jlens", layer, v_full)
        emit("jlens.centered", layer, v_full - v_mean0)
        # full-lens ranks and target tokens
        fu, ff, ts = rank_pass(v_full, list(range(P)), want_tstar=True)
        tstar = t.tensor([ts[p][i] for p, i in pairs], device=DEV)
        U = (W[tstar] - Wbar) * gamma  # [Npairs, d]
        jl_u = [fu[p].get(i) for p, i in pairs]
        jl_f = [ff[p].get(i) for p, i in pairs]

        decomps = []  # (decomp, health, dict_path)
        for path in lpaths:
            D = BlockDict.load(path, DEV)
            name = D.name if D.name not in seen_names else path.stem
            seen_names.add(name)
            health = {"kind": "trained", "path": str(path), "meta": json_scalars(D.meta), **block_stable_ranks(path)}
            if D.group_size == 1:
                health.update(sr_q10=1.0, sr_q50=1.0, sr_q90=1.0, sr_share_gt1_5=0.0)
            decomps.append((TrainedDecomp(D, name), health))
            d = D.d
            # controls (once per layer/space/b/l0)
            if rand_Q is None:
                g = t.Generator(device="cpu").manual_seed(args.seed)
                rand_Q = t.linalg.qr(t.randn(d, d, generator=g, dtype=t.float64)).Q.float().to(DEV)
            rd = BasisDecomp("rand", rand_Q, D.mean, space=D.space, layer=layer, b=D.group_size, l0=D.l0)
            if (rd.name, layer) not in seen_names:
                seen_names.add((rd.name, layer))
                decomps.append((rd, {"kind": "rand", "seed": args.seed, "basis_for": name}))
                if (D.space, layer) not in pca_Q:
                    X = load_val_acts(path, layer)
                    if X is None:
                        print(f"  WARNING no val_acts_L{layer}.pt next to {path}; no PCA control for {D.space}")
                        pca_Q[(D.space, layer)] = None
                    else:
                        cov = t.zeros(d, d, dtype=t.float64, device=DEV)
                        mu = D.mean.float()
                        for s in range(0, X.shape[0], 8192):
                            xb = X[s : s + 8192].to(DEV).float()
                            if D.space == "jspace":
                                xb = xb @ J.T
                            xb = (xb - mu).double()
                            cov += xb.T @ xb
                        evals, evecs = t.linalg.eigh(cov / X.shape[0])
                        pca_Q[(D.space, layer)] = evecs.flip(-1).float().contiguous()
                        print(f"  PCA basis {D.space} L{layer} from {X.shape[0]} val acts; top-{D.l0 * D.group_size} dims hold "
                              f"{float(evals.flip(-1)[: D.l0 * D.group_size].sum() / evals.sum()):.3f} of val energy")
                        del X, cov
                if pca_Q[(D.space, layer)] is not None:
                    pdc = BasisDecomp("pca", pca_Q[(D.space, layer)], D.mean, space=D.space, layer=layer, b=D.group_size, l0=D.l0)
                    decomps.append((pdc, {"kind": "pca", "basis_for": name}))

        for dc, health in decomps:
            tag = f"{dc.name}@L{layer}" if dc.kind != "trained" else dc.name
            print(f"-- {tag} (space {dc.space}, b {dc.b}, l0 {dc.l0})")
            to_final = (lambda u: u @ J.T) if dc.space == "resid" else (lambda u: u)
            x = h if dc.space == "resid" else v_full
            idx, per, blocks, err = dc.split(x)
            xc = x - x.mean(0)
            fvu = float((err**2).sum() / (xc**2).sum())
            health.update(fvu_bench=fvu, energy_share=1 - fvu, space=dc.space, layer=layer, b=dc.b, l0=dc.l0)
            v_mean = to_final(dc.mean)
            v_blocks, v_err = to_final(blocks), to_final(err)
            Vg = to_final(per.reshape(-1, per.shape[-1])).view(P, dc.l0, -1)  # [P, l0, d] final basis
            del per

            # B. readouts of the parts
            emit(f"{dc.name}.recon", layer, v_mean + v_blocks)
            emit(f"{dc.name}.blocks", layer, v_blocks)
            emit(f"{dc.name}.err", layer, v_err)
            emit(f"{dc.name}.err+mean", layer, v_mean + v_err)
            _, rec_f, _ = rank_pass(v_mean + v_blocks, list(range(P)))
            _, blk_f, _ = rank_pass(v_blocks, list(range(P)))
            _, err_f, _ = rank_pass(v_err, list(range(P)))

            # single blocks: all P*l0 rows, chunked
            sb_u, sb_f, _ = rank_pass(Vg.reshape(P * dc.l0, -1), [p for p in range(P) for _ in range(dc.l0)])
            gnorm = Vg.norm(dim=-1)  # [P, l0]
            k_top = gnorm.argmax(-1).tolist()
            top1norm_vec = Vg[t.arange(P, device=DEV), gnorm.argmax(-1)]
            emit(f"{dc.name}.top1norm", layer, top1norm_vec)

            # C. attribution
            c_blocks = t.cat([t.einsum("nkd,nd->nk", Vg[pair_p[s : s + 512]], U[s : s + 512]) for s in range(0, len(pairs), 512)])
            c_mean = U @ v_mean
            c_err = (v_err[pair_p] * U).sum(-1)
            c_full = (v_full[pair_p] * U).sum(-1)
            recon_c = c_mean + c_blocks.sum(-1) + c_err
            denom = c_mean.abs() + c_blocks.abs().sum(-1) + c_err.abs() + c_full.abs()
            relc = ((recon_c - c_full).abs() / denom.clamp_min(1e-12)).max()
            relf = ((recon_c - c_full).abs() / c_full.abs().clamp_min(1e-12)).max()
            print(f"  attribution closure: worst |sum-c_full| / sum|parts| = {relc:.2e}; / |c_full| = {relf:.2e}"
                  + ("   WARNING > 1e-3" if relc > 1e-3 else ""))
            health["closure_worst_rel"] = float(relc)
            cb, cm, ce, cf = c_blocks.cpu(), c_mean.cpu(), c_err.cpu(), c_full.cpu()
            idx_c = idx.cpu()

            oracle = {}  # (p, i) -> (rank_f, rank_u)
            rows = []
            for n, (p, i) in enumerate(pairs):
                c = cb[n]
                pos = c.clamp_min(0)
                S = float(pos.sum())
                if S > 0:
                    sp = pos.sort(descending=True).values
                    top1, top3 = float(sp[0]) / S, float(sp[:3].sum()) / S
                    n80 = int((sp.cumsum(0) < 0.8 * S - 1e-12 * S).sum()) + 1
                else:
                    top1 = top3 = float("nan")
                    n80 = None
                kc = int(c.argmax())
                rf_k = [sb_f[p * dc.l0 + k].get(i) for k in range(dc.l0)]
                ru_k = [sb_u[p * dc.l0 + k].get(i) for k in range(dc.l0)]
                o_f = min(r for r in rf_k if r is not None) if any(r is not None for r in rf_k) else None
                o_u = min(r for r in ru_k if r is not None) if any(r is not None for r in ru_k) else None
                oracle[(p, i)] = (o_f, o_u)
                fullc = float(cf[n])
                ev, it = ekey(p)
                rows.append({
                    "decomp": dc.name, "layer": layer, "space": dc.space, "b": dc.b, "eval": ev, "item": it,
                    "intermediate": i, "split": bench.split[p],
                    "jlens_rank_unfiltered": jl_u[n], "jlens_rank_filtered": jl_f[n],
                    "c_full": fullc,
                    "share_mean": float(cm[n]) / fullc if fullc != 0 else float("nan"),
                    "share_err": float(ce[n]) / fullc if fullc != 0 else float("nan"),
                    "share_blocks": float(c.sum()) / fullc if fullc != 0 else float("nan"),
                    "top1": top1, "top3": top3, "n80": n80,
                    "carrier_block": int(idx_c[p, kc]), "carrier_c": float(c[kc]),
                    "oracle1_rank_filtered": o_f, "oracle1_rank_unfiltered": o_u,
                    "top1norm_rank_filtered": rf_k[k_top[p]], "carrier_rank_filtered": rf_k[kc],
                    # extra columns (used by the summary)
                    "top1norm_rank_unfiltered": ru_k[k_top[p]], "carrier_rank_unfiltered": ru_k[kc],
                    "recon_rank_filtered": rec_f[p].get(i), "blocks_rank_filtered": blk_f[p].get(i),
                    "err_rank_filtered": err_f[p].get(i),
                })
            attrib_rows += rows

            # oracle1 rows for the CSVs, templated on the top1norm rows of the same items
            for sp_name, items in (("held", held), ("fit", fit)):
                for f in (True, False):
                    tmpl = [r for r in csv_rows[(sp_name, f)] if r["lens"] == f"{dc.name}.top1norm" and int(r["layer"]) == layer]
                    csv_rows[(sp_name, f)] += emit_from_template(
                        tmpl, items, f"{dc.name}.oracle1", lambda p, i, f=f: oracle[(p, i)][0 if f else 1])

            # summary on held-out pairs
            H = [r for r in rows if r["split"] == "held-out"]
            sets = {"all": H, "hits": [r for r in H if r["jlens_rank_filtered"] is not None and r["jlens_rank_filtered"] <= 10]}
            summ = {"health": health}
            for sname, R in sets.items():
                s = {"count": len(R)}
                for k in ("share_blocks", "share_err", "share_mean", "top1", "top3", "n80"):
                    s[f"median_{k}"] = nanmedian([r[k] for r in R])
                for lname, col in (("jlens", "jlens_rank_filtered"), ("recon", "recon_rank_filtered"),
                                   ("blocks", "blocks_rank_filtered"), ("err", "err_rank_filtered"),
                                   ("oracle1", "oracle1_rank_filtered"), ("top1norm", "top1norm_rank_filtered"),
                                   ("carrier", "carrier_rank_filtered")):
                    s[f"{lname}_r10"] = frac([r[col] for r in R], 10)
                    s[f"{lname}_r1"] = frac([r[col] for r in R], 1)
                summ[sname] = s
            summary[tag] = summ
            del Vg, gnorm, top1norm_vec, c_blocks
        for dc, _ in decomps:
            if dc.kind == "trained":
                dc.D.cpu()
        del decomps
        t.cuda.empty_cache()

    # ---- write
    write_csv(out / "readout_filtered.csv", csv_rows[("held", True)])
    write_csv(out / "readout_unfiltered.csv", csv_rows[("held", False)])
    write_csv(out / "readout_fit_filtered.csv", csv_rows[("fit", True)])
    write_csv(out / "readout_fit_unfiltered.csv", csv_rows[("fit", False)])
    if attrib_rows:
        with open(out / "attrib_pairs.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(attrib_rows[0].keys()))
            w.writeheader()
            w.writerows(attrib_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # ---- compact table
    def fmt(v, p=2):
        return "   -" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))

    print("\nheld-out pairs; shares/top1/n80 are medians over jlens hits (filtered rank <= 10); r10 = frac rank<=10 over all")
    hdr = ["decomp", "energy", "n_all", "n_hit", "shBlk", "shErr", "shMean", "top1", "n80",
           "jl10", "rec10", "blk10", "err10", "orc10", "t1n10", "car10", "orc10h", "car10h"]
    print(" ".join(f"{c:>8s}" if i else f"{c:<26s}" for i, c in enumerate(hdr)))
    for tag, s in summary.items():
        a, hh = s["all"], s["hits"]
        vals = [fmt(s["health"]["energy_share"], 3), a["count"], hh["count"], fmt(hh["median_share_blocks"]),
                fmt(hh["median_share_err"]), fmt(hh["median_share_mean"]), fmt(hh["median_top1"]), fmt(hh["median_n80"], 1),
                fmt(a["jlens_r10"]), fmt(a["recon_r10"]), fmt(a["blocks_r10"]), fmt(a["err_r10"]),
                fmt(a["oracle1_r10"]), fmt(a["top1norm_r10"]), fmt(a["carrier_r10"]),
                fmt(hh["oracle1_r10"]), fmt(hh["carrier_r10"])]
        print(f"{tag:<26s} " + " ".join(f"{str(v):>8s}" for v in vals))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
