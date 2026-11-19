"""Cache readout residuals of every scoreable eval item, for offline readout ranks (olens.readout).

Items, correctness filter, scoreable-item logic, max seq len, fit/held-out split and the
Readout-Filtering exclusion ids are taken from jpp_cli exactly as `jpp_cli evaluate` does.
jpp_cli reads its eval data by relative path, so this chdirs to the jpp_lens checkout.

    /root/jpp_lens/.venv/bin/python /workspace/o-lens-improvement/scripts/bench_cache.py \
        --model Qwen/Qwen3.5-2B --out /workspace/o-lens-improvement/runs/bench/qwen35_2b.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

JPP = os.environ.get("JPP_LENS_DIR", "/root/jpp_lens")
R = "/workspace/o-lens-improvement"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
ap.add_argument("--layers", default=None, help="comma list; default all blocks 0..n_layers-2")
ap.add_argument("--out", required=True)
ap.add_argument("--correctness-csv", default=f"{R}/data/model_correctness.csv")
ap.add_argument("--fit-fraction", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out_path = os.path.abspath(args.out)
correctness_csv = os.path.abspath(args.correctness_csv)

os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import torch as t  # noqa: E402

import jpp_cli  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import READOUT_EVALS_DATA_DIR  # noqa: E402
from lens_evals.readout_evals import readout_evals as re_mod  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402

model = get_hf_model(args.model, attn_implementation="sdpa")  # as jpp_cli._load_model with its defaults
layers = (
    sorted(int(x) for x in args.layers.split(","))
    if args.layers
    else list(range(model.n_layers - 1))
)
max_seq_len = jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN

# Same call as jpp_cli._load_items_and_exclusions; items_spec="all" -> the scoreable items.
_, scoreable, excluded_ids = jpp_cli.load_items_and_exclusions(
    model,
    eval_data_dir=READOUT_EVALS_DATA_DIR,
    correctness_csv=correctness_csv,
    hf_model_name=args.model,
    layers=layers,
    readout_max_seq_len=max_seq_len,
    items_spec="all",
    fit_fraction=args.fit_fraction,
)
fit_keys = {
    (it.eval_slug, it.name)
    for it in jpp_cli.select_items(scoreable, f"fit:{args.seed}", fit_fraction=args.fit_fraction)
}
held_keys = {
    (it.eval_slug, it.name)
    for it in jpp_cli.select_items(scoreable, f"held-out:{args.seed}", fit_fraction=args.fit_fraction)
}

runner = jpp_cli.readout_runner(model, layers=layers, max_seq_len=max_seq_len)
res = runner.record_residuals(scoreable, hf_model_name=args.model)

keys = list(zip(res.eval_slugs, res.item_names, strict=True))
if keys != [(it.eval_slug, it.name) for it in scoreable]:
    raise SystemExit("record_residuals kept a different item list than scoreable_items")
split = []
for k in keys:
    if (k in fit_keys) == (k in held_keys):
        raise SystemExit(f"item {k} is in both or neither split")
    split.append("fit" if k in fit_keys else "held-out")

# Readout positions (tokenization only; the same private helper run/record_residuals use).
positions = None
try:
    positions = []
    for it, cands in zip(scoreable, res.candidate_ids_P_list, strict=True):
        ro = re_mod._item_readout_and_candidates(model, it, max_seq_len)
        assert ro is not None and ro.candidate_ids_by_intermediate == cands
        positions.append(int(ro.readout_position))
except Exception as e:  # optional field
    logging.warning("readout_positions not stored: %r", e)
    positions = None

payload = {
    "model": args.model,
    "layers": [int(layer) for layer in res.layers],
    "eval_slugs": list(res.eval_slugs),
    "item_names": list(res.item_names),
    "resid": {int(layer): v.float().cpu().contiguous() for layer, v in res.residuals_L_dict_PN.items()},
    "candidates": [{str(k): [int(i) for i in v] for k, v in c.items()} for c in res.candidate_ids_P_list],
    "split": split,
    "excluded_ids": t.tensor([int(i) for i in excluded_ids], dtype=t.long),
    "readout_positions": positions,
    "max_seq_len": int(max_seq_len),
    "fit_fraction": float(args.fit_fraction),
    "seed": int(args.seed),
    "correctness_csv": correctness_csv,
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:  # write in place (network fs)
    t.save(payload, f)

# Round-trip check with weights_only=True.
check = t.load(out_path, map_location="cpu", weights_only=True)
assert check["item_names"] == payload["item_names"]

per = Counter(zip(payload["eval_slugs"], split))
print(f"saved {out_path}: {len(keys)} items, layers {payload['layers']}, "
      f"d={payload['resid'][payload['layers'][0]].shape[1]}, {len(excluded_ids)} excluded ids")
print(f"{'eval':14s} {'fit':>5s} {'held-out':>9s} {'total':>6s}")
for ev in sorted({e for e, _ in per}):
    f_, h_ = per[(ev, "fit")], per[(ev, "held-out")]
    print(f"{ev:14s} {f_:5d} {h_:9d} {f_ + h_:6d}")
print(f"{'all':14s} {split.count('fit'):5d} {split.count('held-out'):9d} {len(split):6d}")
