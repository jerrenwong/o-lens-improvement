"""Direct vs indirect paths: lenses J - a*I and J++ - a*I.

In a residual network dh_L/dh_l = I + (indirect paths), so the J-lens is the logit lens (I) plus
an "indirect lens" (J - I). This scores  unembed((M - a*I) h)  for M in {J, J++} and a sweep of a
(a = 0: the lens itself; a = 1: indirect paths only, exact for the plain-gradient J-lens).
Optionally also (J - I)(h - frequent features) with residual-space dictionaries.

    python scripts/indirect_lens.py --model M --bench B --jlens J.pt --jpp JPP.pt --layers 3,6,.. --out DIR
           [--resid-dicts L3_resid_b1_10M.pt ... --tau 0.03]
Writes harness-schema CSVs (held-out and fit) and prints per-layer hit counts.
"""

import argparse
import collections
from pathlib import Path

import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

from olens.readout import Bench, readout_rows, write_csv

ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--jpp", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--resid-dicts", nargs="*", default=[])
    ap.add_argument("--tau", type=float, default=0.03)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t.set_grad_enabled(False)

    model = get_hf_model(args.model, attn_implementation="sdpa")
    bench = Bench.load(args.bench)
    lenses = {"jlens": load_lens_file(args.jlens, hf_model_name=args.model),
              "jpp": load_lens_file(args.jpp, hf_model_name=args.model)}
    held, fit = bench.index("held-out"), bench.index("fit")
    rows = {(s, f): [] for s in ("held", "fit") for f in (True, False)}

    def emit(name, layer, v):
        for f in (True, False):
            rows[("held", f)] += readout_rows(name, layer, bench, v, model, filtered=f, items=held)
            rows[("fit", f)] += readout_rows(name, layer, bench, v, model, filtered=f, items=fit)

    dicts = {}
    if args.resid_dicts:
        from olens.bsf_io import BlockDict
        for p in args.resid_dicts:
            D = BlockDict.load(p, "cuda")
            st = t.load(str(p)[:-3] + "_stats.pt", map_location="cuda", weights_only=True)
            dicts[D.layer] = (D, st["fire_count"].float() / float(st["rows"]))

    for layer in [int(x) for x in args.layers.split(",")]:
        h = bench.resid[layer].cuda().float()
        emit("logit", layer, h)
        for base, lens in lenses.items():
            M = lens.jacobians_L_dict_FN[layer].cuda().float()
            v = h @ M.T
            d = M.shape[0]
            diag = float(M.diag().mean())
            print(f"L{layer} {base}: mean diagonal {diag:.3f}, ||M - I||/||I|| {float((M - t.eye(d, device='cuda')).norm() / d**0.5):.3f}", flush=True)
            emit(base, layer, v)
            for a in ALPHAS:
                emit(f"{base}-a{a:g}", layer, v - a * h)
            if layer in dicts:
                D, freq = dicts[layer]
                z = D.encode(h)
                dropped = D.decode(z) - D.decode(z * (freq < args.tau)[None, :, None])
                hk = h - dropped
                emit(f"{base}+fq", layer, hk @ M.T)
                emit(f"{base}-a1+fq", layer, hk @ M.T - hk)

    write_csv(out / "readout_filtered.csv", rows[("held", True)])
    write_csv(out / "readout_unfiltered.csv", rows[("held", False)])
    write_csv(out / "readout_fit_filtered.csv", rows[("fit", True)])
    write_csv(out / "readout_fit_unfiltered.csv", rows[("fit", False)])

    # per-layer hits (held-out, filtered): pairs with rank <= 10
    cnt = collections.defaultdict(lambda: collections.Counter())
    npairs = collections.Counter()
    for r in rows[("held", True)]:
        cnt[r["lens"]][r["layer"]] += r["rank"] <= 10
        npairs[r["lens"]] += 0
    layers = sorted({r["layer"] for r in rows[("held", True)]})
    n = len([r for r in rows[("held", True)] if r["lens"] == "jlens" and r["layer"] == layers[0]])
    print(f"\nheld-out pairs with filtered rank <= 10, per layer (n = {n} pairs)")
    print(f"{'lens':<16}" + "".join(f"{'L' + str(l):>6}" for l in layers))
    for name in ["logit", "jlens"] + [f"jlens-a{a:g}" for a in ALPHAS] + ["jpp"] + [f"jpp-a{a:g}" for a in ALPHAS] + ["jlens+fq", "jlens-a1+fq", "jpp+fq", "jpp-a1+fq"]:
        if name in cnt:
            print(f"{name:<16}" + "".join(f"{cnt[name][l]:>6}" for l in layers))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
