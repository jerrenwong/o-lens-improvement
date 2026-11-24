"""W2b step 2/3: competitor-subtracted readout, scored on FIT (for choosing alpha) and DEV only.

    score_X = z_X - alpha * sum_{Y != X} q(Y|X) z_Y          (plain; = rows of (I - alpha Q) applied to the logits)
    centred: z_Y replaced by z_Y - mean_V(z)  (shift-invariant version; only differs for rows of Q that sum to 1)
z = model.unembed(v).float() of the lens vector, exactly as the harness (one row per call, chunk_size = 1);
q(Y|X) = C[X, Y] / sum_{Y' != X} C[X, Y'] from .hf27/w2/copred_<tag>_<variant>.pt (top-50 partners kept; a token
never seen in a cut top-k has no partners and keeps z_X). Filtered ranks: excluded ids -> -inf AFTER the subtraction.
The evaluator is not edited: data, split, ctx and the macro pass@k are taken from auto/evaluate.py (MODELS, pass_at_k)
and olens.readout (Bench, candidate_ranks), with the same code path as evaluate.py --lite.
alpha in --alphas chosen on the FIT split per variant (max filtered pass@1 + pass@10, ties -> smaller alpha).
Writes runs/w2/copred_eval/<name>.json and prints a table; --ref <auto/runs name> checks alpha = 0 against that stored
dev summary (must be equal). --qual N prints filtered top-10 lists for N dev items with alpha 0 vs the chosen alpha.
Usage: python eval_copred.py --model M --lens auto/candidates/x.py --name N [--ref RUN] [--variants k5,...] [--qual 10]
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path("/workspace/o-lens-improvement")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "perf1"))
_spec = importlib.util.spec_from_file_location("w2_evaluate", ROOT / "auto" / "evaluate.py")
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)
MODELS, pass_at_k = EV.MODELS, EV.pass_at_k


def load_Q(path, V, dev):
    d = torch.load(path, map_location="cpu")
    idx, w, rs = d["idx"].long(), d["w"].float(), d["rowsum"].float()
    ok = idx >= 0
    q = torch.where(ok, w / rs.clamp_min(1e-30)[:, None], torch.zeros_like(w))
    rows = torch.arange(idx.shape[0])[:, None].expand_as(idx)
    Q = torch.sparse_coo_tensor(torch.stack([rows[ok], idx[ok]]), q[ok], (V, V)).coalesce().to_sparse_csr().to(dev)
    qsum = torch.zeros(V).index_add_(0, rows[ok], q[ok]).to(dev)
    return Q, qsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", choices=list(MODELS))
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--variants", default="k5,k20,p09,t05,words")
    ap.add_argument("--alphas", default="0,0.25,0.5,1")
    ap.add_argument("--centred", default="0", help="comma list of 0/1: plain and/or centred subtraction")
    ap.add_argument("--qual", type=int, default=0)
    ap.add_argument("--qual_variant", default=None)
    ap.add_argument("--copred_dir", default=str(ROOT / ".hf27" / "w2"))
    args = ap.parse_args()
    cfg = MODELS[args.model]
    alphas = [float(a) for a in args.alphas.split(",")]
    variants = args.variants.split(",")
    cents = [bool(int(c)) for c in args.centred.split(",")]
    out_dir = ROOT / "runs" / "w2" / "copred_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    from olens.readout import Bench, candidate_ranks
    from workspace_lens.utils import load_lens_file
    from lite_model import get_lite_model, lite_blocked

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    dev = "cuda"
    bench = Bench.load(ROOT / cfg["bench"])
    split = json.loads((ROOT / f"auto/split_{cfg['tag']}.json").read_text())
    pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
    fit_idx = sorted(pos[tuple(k)] for k in split["fit"])
    assert fit_idx == bench.index("fit")
    idx = sorted(pos[tuple(k)] for k in split["dev"])  # DEV only
    model = get_lite_model(args.model)

    def lens_dict(path):
        L = load_lens_file(str(ROOT / path), hf_model_name=args.model)
        return {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}

    fi = torch.as_tensor(fit_idx, dtype=torch.long)
    fit = SimpleNamespace(resid={l: v[fi].to(dev).float() for l, v in bench.resid.items()},
                          candidates=[bench.candidates[i] for i in fit_idx],
                          eval_slugs=[bench.eval_slugs[i] for i in fit_idx],
                          item_names=[bench.item_names[i] for i in fit_idx], excluded_ids=bench.excluded_ids.clone())
    ctx = SimpleNamespace(model=model, J=lens_dict(cfg["J"]), JPP=lens_dict(cfg["JPP"]), fit=fit,
                          dict_path=lambda layer, space: str(ROOT / cfg["dicts"](layer, space)),
                          wikitext=lite_blocked("ctx.wikitext"), readout_layers=list(cfg["layers"]), device=dev)
    si = torch.as_tensor(idx, dtype=torch.long)
    rows_resid = {l: v[si].to(dev).float() for l, v in bench.resid.items()}
    spec = importlib.util.spec_from_file_location(f"candidate_{args.name}", Path(args.lens).resolve())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lens = mod.build(ctx)
    vecs = {"dev": {}, "fit": {}}
    for layer in cfg["layers"]:
        vecs["dev"][layer] = lens.transport(rows_resid[layer].clone(), layer,
                                            {l: v.clone() for l, v in rows_resid.items()}).detach().float()
        vecs["fit"][layer] = lens.transport(fit.resid[layer].clone(), layer,
                                            {l: v.clone() for l, v in fit.resid.items()}).detach().float()
    del lens
    V = model._lm_head.weight.shape[0]
    excl = torch.zeros(V, dtype=torch.bool, device=dev)
    excl[bench.excluded_ids.to(dev)] = True
    meta = {"dev": ([bench.eval_slugs[i] for i in idx], [bench.item_names[i] for i in idx],
                    [bench.candidates[i] for i in idx]),
            "fit": (fit.eval_slugs, fit.item_names, fit.candidates)}
    # logits once per split/layer (harness call shape: one row per unembed call)
    Z = {s: {l: torch.cat([model.unembed(v[i:i + 1]).float() for i in range(v.shape[0])]) for l, v in d.items()}
         for s, d in vecs.items()}
    combos = [("none", 0.0, False)]
    Qs = {}
    for var in variants:
        Qs[var] = load_Q(f"{args.copred_dir}/copred_{cfg['tag']}_{var}.pt", V, dev)
        for c in cents:
            for a in alphas:
                if a > 0:
                    combos.append((var, a, c))

    def modified(z, var, a, c):
        if a == 0:
            return z
        Q, qsum = Qs[var]
        comp = torch.sparse.mm(Q, z.T.contiguous()).T
        if c:
            comp = comp - qsum[None] * z.mean(-1, keepdim=True)
        return z - a * comp

    res = {}
    t0 = time.time()
    for combo in combos:
        r = {}
        for s in ("fit", "dev"):
            slugs, names, cands = meta[s]
            rows = []
            for l in cfg["layers"]:
                sc = modified(Z[s][l], *combo).masked_fill(excl, float("-inf"))
                rk = candidate_ranks(sc, cands)
                for p, dct in enumerate(rk):
                    for inter, rank in dct.items():
                        rows.append(dict(eval=slugs[p], item=names[p], intermediate=inter, rank=rank))
            p10, task = pass_at_k(rows, 10)
            p1, _ = pass_at_k(rows, 1)
            r[s] = dict(p10=p10, p1=p1, task=task)
        res[combo] = r
    print(f"[w2b] scored {len(combos)} combos in {time.time() - t0:.0f}s", flush=True)
    base = res[("none", 0.0, False)]
    out = dict(model=args.model, lens=args.lens, name=args.name, alpha0=base, combos=[], chosen={})
    if args.ref:
        ref = json.loads((ROOT / "auto" / "runs" / args.ref / "summary.json").read_text())
        same = abs(ref["pass10"] - base["dev"]["p10"]) < 1e-12 and abs(ref["pass1"] - base["dev"]["p1"]) < 1e-12
        out["ref_check"] = dict(ref=args.ref, ref_p10=ref["pass10"], ref_p1=ref["pass1"], equal=same)
        print(f"[w2b] alpha=0 dev p10 {base['dev']['p10']:.4f} p1 {base['dev']['p1']:.4f} vs stored {args.ref} "
              f"p10 {ref['pass10']:.4f} p1 {ref['pass1']:.4f} -> {'EQUAL' if same else 'DIFFERENT'}", flush=True)
    print(f"{'variant':8s} {'cent':4s} {'alpha':>5s} | fit p10  p1   | dev p10  p1", flush=True)
    for (var, a, c), r in res.items():
        out["combos"].append(dict(variant=var, alpha=a, centred=c, **{s: r[s] for s in r}))
        print(f"{var:8s} {int(c):4d} {a:5.2f} | {r['fit']['p10']:.4f} {r['fit']['p1']:.4f} | "
              f"{r['dev']['p10']:.4f} {r['dev']['p1']:.4f}", flush=True)
    for var in variants:
        for c in cents:
            cand = [(0.0, base)] + [(a, res[(var, a, c)]) for a in alphas if a > 0]
            a_best, r_best = max(cand, key=lambda x: (round(x[1]["fit"]["p1"] + x[1]["fit"]["p10"], 6), -x[0]))
            out["chosen"][f"{var}{'_c' if c else ''}"] = dict(alpha=a_best, fit=r_best["fit"], dev=r_best["dev"])
            print(f"[w2b] chosen {var}{' centred' if c else ''}: alpha {a_best} -> dev p10 {r_best['dev']['p10']:.4f} "
                  f"p1 {r_best['dev']['p1']:.4f} (alpha 0: {base['dev']['p10']:.4f} {base['dev']['p1']:.4f})", flush=True)
    if args.qual:
        qv = args.qual_variant or variants[0]
        key = f"{qv}"
        a = out["chosen"][key]["alpha"] or alphas[-1]
        tok = model.tokenizer
        L = cfg["layers"][len(cfg["layers"]) // 2]
        slugs, names, cands = meta["dev"]
        print(f"\n[qual] dev items, layer {L}, filtered top-10: alpha 0 vs {qv} alpha {a} (* = accepted answer)")
        for p in range(0, len(slugs), max(1, len(slugs) // args.qual))[:args.qual]:
            ids = {i for v in cands[p].values() for i in v}
            print(f"-- {slugs[p]} / {names[p]}  answers {list(cands[p])}")
            for lab, sc in (("a=0", Z["dev"][L][p:p + 1]), (f"a={a}", modified(Z["dev"][L][p:p + 1], qv, a, False))):
                top = sc.masked_fill(excl, float("-inf"))[0].topk(10).indices.tolist()
                print(f"   {lab:7s} " + ", ".join(("*" if t in ids else "") + repr(tok.decode([t])) for t in top))
    out["combos_keyed"] = None
    json.dump(out, open(out_dir / f"{args.name}.json", "w"), indent=1, default=str)
    print("saved", out_dir / f"{args.name}.json", flush=True)


if __name__ == "__main__":
    main()
