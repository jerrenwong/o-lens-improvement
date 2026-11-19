"""J-lens with frequent ("generic") dictionary features removed.

For a dictionary D at layer l (x = mean + sum_g x_g + err), and a frequency threshold tau
(freq_g = share of training tokens on which block g was active):

    <fam>.fq<tau>          J-lens of  x - sum_{g: freq_g >= tau} x_g        (mean and error kept)
    <fam>.fq<tau>.nomean   same, minus the mean
    <fam>.fq<tau>.blocks   only the kept blocks (no mean, no error)
    <fam>.rnd<tau>         control: per item, drop the same NUMBER of active blocks, chosen at random
    <fam>.idf<a>           soft version: blocks weighted by (idf_g / max idf)^a, mean and error kept

Rows go to harness-schema CSVs (held-out and fit items) for scripts/analyze.py.

    python scripts/freq_mask.py --bench B --jlens J --dicts L3_x.pt L6_x.pt ... --out DIR
All dictionaries must share one space and one block size, at most one per layer.
"""

import argparse
import math
from pathlib import Path

import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

from olens.bsf_io import BlockDict
from olens.readout import Bench, readout_rows, write_csv

TAUS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
ALPHAS = [0.5, 1.0, 2.0]


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

    model = get_hf_model(args.model, attn_implementation="sdpa")
    bench = Bench.load(args.bench)
    lens = load_lens_file(args.jlens, hf_model_name=args.model)
    held, fit = bench.index("held-out"), bench.index("fit")
    rows = {(s, f): [] for s in ("held", "fit") for f in (True, False)}

    def emit(name, layer, v):
        for f in (True, False):
            rows[("held", f)] += readout_rows(name, layer, bench, v, model, filtered=f, items=held)
            rows[("fit", f)] += readout_rows(name, layer, bench, v, model, filtered=f, items=fit)

    gen = t.Generator(device="cuda").manual_seed(args.seed)
    fam = None
    for path in sorted(args.dicts):
        D = BlockDict.load(path, "cuda")
        f_ = f"{D.space}_b{D.group_size}"
        assert fam in (None, f_), "all dictionaries must share space and block size"
        fam = f_
        layer = D.layer
        st = t.load(str(path)[:-3] + "_stats.pt", map_location="cuda", weights_only=True)
        freq = st["fire_count"].float() / float(st["rows"])  # [G]
        J = lens.jacobians_L_dict_FN[layer].cuda().float()
        h = bench.resid[layer].cuda().float()
        x = h if D.space == "resid" else h @ J.T
        to_final = (lambda u: u @ J.T) if D.space == "resid" else (lambda u: u)
        z = D.encode(x)  # [P, G, b]
        active = z.norm(dim=-1) > 0  # [P, G]
        blocks_all = D.decode(z)
        mean = D.mean.float()
        v_full = to_final(x)
        emit("jlens", layer, v_full)
        emit(f"{fam}.all", layer, v_full)  # tau = inf: nothing dropped (sanity: equals jlens)
        print(f"== L{layer} {fam}: P={x.shape[0]}, active/item={active.sum(1).float().mean():.1f}, "
              f"median freq of active occurrences {freq[None].expand_as(active)[active].median():.4f}", flush=True)
        for tau in TAUS:
            keep = freq < tau  # [G]
            kept = D.decode(z * keep[None, :, None])
            dropped = blocks_all - kept
            n_drop = (active & ~keep[None]).sum(1)  # [P]
            tag = f"{tau:g}"
            emit(f"{fam}.fq{tag}", layer, to_final(x - dropped))
            emit(f"{fam}.fq{tag}.nomean", layer, to_final(x - dropped - mean))
            emit(f"{fam}.fq{tag}.blocks", layer, to_final(kept))
            # control: per item drop n_drop random active blocks
            key = t.rand(active.shape, device="cuda", generator=gen).masked_fill(~active, 2.0)
            order = key.argsort(dim=1).argsort(dim=1)  # rank of each block's key within the item
            rdrop = active & (order < n_drop[:, None])
            emit(f"{fam}.rnd{tag}", layer, to_final(x - D.decode(z * rdrop[:, :, None])))
            e_drop = float(dropped.pow(2).sum() / (x - mean).pow(2).sum())
            print(f"  tau {tag:>6}: blocks with freq>=tau {int((~keep).sum()):6d}/{keep.numel()}  "
                  f"active dropped/item {n_drop.float().mean():5.1f}  energy of (x-mean) dropped {e_drop:.3f}", flush=True)
        idf = -t.log(freq.clamp_min(1e-7))
        for a in ALPHAS:
            w = (idf / idf.max()) ** a
            emit(f"{fam}.idf{a:g}", layer, to_final(x - blocks_all + D.decode(z * w[None, :, None])))
        D.cpu()

    write_csv(out / "readout_filtered.csv", rows[("held", True)])
    write_csv(out / "readout_unfiltered.csv", rows[("held", False)])
    write_csv(out / "readout_fit_filtered.csv", rows[("fit", True)])
    write_csv(out / "readout_fit_unfiltered.csv", rows[("fit", False)])
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
