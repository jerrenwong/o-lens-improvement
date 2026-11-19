"""Check olens.readout's offline ranks against the jpp_lens harness CSVs for one lens.

Recomputes held-out rows from a bench cache (scripts/bench_cache.py) for every layer in the
harness CSVs, filtered and unfiltered, and compares rank by rank with the harness rows of
--lens-name. Exit 1 if fewer than 99.5% of ranks are identical or any row is missing.

    /root/jpp_lens/.venv/bin/python /workspace/o-lens-improvement/scripts/verify_readout.py \
        --bench /workspace/o-lens-improvement/runs/bench/qwen35_2b.pt \
        --jlens /workspace/o-lens-improvement/artifacts/baselines/jlens_prefit.pt \
        --harness-dir /workspace/o-lens-improvement/runs/v2/eval

--jlens logit verifies the Logit Lens (identity transport) instead (use --lens-name logit).
--chunk 1 unembeds one row at a time, the harness's call shape (for diagnosing tie flips).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file

from olens.readout import Bench, check_final_linear, final_linear, readout_rows

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--jlens", required=True, help="lens .pt (J-lens) or 'logit'")
ap.add_argument("--harness-dir", required=True)
ap.add_argument("--lens-name", default="jlens_prefit")
ap.add_argument("--model", default=None, help="default: the bench's model")
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--min-identical", type=float, default=0.995)
args = ap.parse_args()

bench = Bench.load(args.bench)
model_name = args.model or bench.model
if model_name != bench.model:
    print(f"WARNING: --model {model_name} != bench model {bench.model}")
model = get_hf_model(model_name, attn_implementation="sdpa")
dev = model.input_device

# final_linear self-check (on random probes, then on real final-ish residuals).
try:
    W, gamma = final_linear(model)
    print(f"final_linear: OK, rel err vs model.unembed (random probes) {check_final_linear(model, W, gamma):.2e}")
    last = max(bench.layers)
    probe = bench.resid[last][:16]
    print(f"final_linear: rel err on 16 bench residuals at layer {last}: "
          f"{check_final_linear(model, W, gamma, probe_vecs=probe):.2e}")
    del W, gamma
    t.cuda.empty_cache()
except Exception as e:  # report, but the rank check does not depend on it
    print(f"final_linear: FAILED: {e}")

lens = None if args.jlens == "logit" else load_lens_file(args.jlens, hf_model_name=model_name)
held = bench.index("held-out")
held_keys = set(bench.keys(held))

total = same = 0
missing_total = 0
for fname, filtered in (("readout_filtered.csv", True), ("readout_unfiltered.csv", False)):
    path = os.path.join(args.harness_dir, fname)
    with open(path, newline="") as f:
        harness = [r for r in csv.DictReader(f) if r["lens"] == args.lens_name]
    if not harness:
        sys.exit(f"{path}: no rows for lens {args.lens_name!r}")
    layers = sorted({int(r["layer"]) for r in harness})
    absent = [layer for layer in layers if layer not in bench.resid]
    if absent:
        sys.exit(f"bench has no residuals at layers {absent}; re-run bench_cache with them")
    h_items = {(r["eval"], r["item"]) for r in harness}
    if h_items != held_keys:
        print(f"{fname}: item sets differ: {len(h_items - held_keys)} only in harness, "
              f"{len(held_keys - h_items)} only in bench held-out")

    ours: dict[tuple, dict] = {}
    for layer in layers:
        h = bench.resid[layer][held].to(dev)
        if lens is None:
            vecs = h
        else:
            J = lens.jacobians_L_dict_FN[layer].to(dev, t.float32)
            vecs = h @ J.T
        for row in readout_rows(args.lens_name, layer, bench, vecs, model,
                                filtered=filtered, items=held, chunk_size=args.chunk):
            ours[(row["eval"], row["item"], row["intermediate"], row["layer"])] = row

    n = n_same = n_missing = ncand_bad = 0
    max_diff = 0
    mismatches = []
    for r in harness:
        key = (r["eval"], r["item"], r["intermediate"], int(r["layer"]))
        n += 1
        o = ours.get(key)
        if o is None:
            n_missing += 1
            if len(mismatches) < 10:
                mismatches.append(f"  missing in ours: {key}")
            continue
        if int(r["n_candidates"]) != o["n_candidates"]:
            ncand_bad += 1
        diff = abs(int(r["rank"]) - o["rank"])
        max_diff = max(max_diff, diff)
        if diff == 0:
            n_same += 1
        elif len(mismatches) < 10:
            mismatches.append(f"  {key}: harness {r['rank']} ours {o['rank']}")
    extra = len(ours) - (n - n_missing)
    print(f"{fname} [{args.lens_name}] layers {layers}: compared {n}, identical {n_same} "
          f"({n_same / n:.4%}), max |rank diff| {max_diff}, missing {n_missing}, "
          f"extra in ours {extra}, n_candidates mismatches {ncand_bad}")
    for m in mismatches[:8]:
        print(m)
    total += n
    same += n_same
    missing_total += n_missing + max(extra, 0) + ncand_bad

frac = same / total if total else 0.0
ok = frac >= args.min_identical and missing_total == 0
print(f"OVERALL: {same}/{total} identical ({frac:.4%}); {'PASS' if ok else 'FAIL'} "
      f"(need >= {args.min_identical:.1%} and no missing/extra rows)")
sys.exit(0 if ok else 1)
