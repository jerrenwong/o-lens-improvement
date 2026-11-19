"""Do offline J-lens readouts agree with the harness where it matters (ranks <= 20, pass@k)?"""
import csv, sys, collections, torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench, readout_rows
R = "/workspace/o-lens-improvement"
model = get_hf_model("Qwen/Qwen3.5-2B", attn_implementation="sdpa")
bench = Bench.load(f"{R}/artifacts/bench/qwen3.5-2b.pt")
lens = load_lens_file(f"{R}/artifacts/baselines/jlens_prefit.pt", hf_model_name="Qwen/Qwen3.5-2B")
held = bench.index("held-out")
for kind, filt in (("filtered", True), ("unfiltered", False)):
    ours = {}
    for L in (3, 6, 9, 12, 15, 18, 21):
        v = bench.resid[L].cuda() @ lens.jacobians_L_dict_FN[L].cuda().float().T
        for r in readout_rows("j", L, bench, v, model, filtered=filt, items=held):
            ours[(r["eval"], r["item"], r["intermediate"], L)] = r["rank"]
    ref = {(r["eval"], r["item"], r["intermediate"], int(r["layer"])): int(r["rank"])
           for r in csv.DictReader(open(f"{R}/runs/v2/eval/readout_{kind}.csv")) if r["lens"] == "jlens_prefit"}
    small = [(k, ref[k], ours[k]) for k in ref if min(ref[k], ours[k]) <= 20]
    bad = [x for x in small if x[1] != x[2]]
    flips = {k_: sum((ref[k] <= k_) != (ours[k] <= k_) for k in ref) for k_ in (1, 10)}
    def macro(d, k_):
        best = collections.defaultdict(lambda: 10**9)
        for (e, i, m, L), r in d.items(): best[(e, i, m)] = min(best[(e, i, m)], r)
        item = collections.defaultdict(list)
        for (e, i, m), r in best.items(): item[(e, i)].append(r <= k_)
        ev = collections.defaultdict(list)
        for (e, i), h in item.items(): ev[e].append(sum(h) / len(h))
        return sum(sum(v) / len(v) for v in ev.values()) / len(ev)
    print(f"{kind}: rows with rank<=20: {len(small)}, differing: {len(bad)} {bad[:5]}; hit flips @1 {flips[1]} @10 {flips[10]}; "
          f"macro pass@1 harness {macro(ref,1):.4f} ours {macro(ours,1):.4f}; pass@10 harness {macro(ref,10):.4f} ours {macro(ours,10):.4f}")
