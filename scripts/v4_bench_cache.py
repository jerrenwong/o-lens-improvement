"""v4 benchmark: grade model correctness and cache readout residuals for the frozen v4 prompts.

Same functions and order as scripts/bench_cache.py (jpp_cli.load_items_and_exclusions -> record_residuals)
and olens.evaluate grade-correctness, pointed at data/evals_v4/ instead of jpp_lens's data dir.
/root/jpp_lens is not modified: the frozen lens-eval-<task>-v4.json files (md5-checked against
MANIFEST.json) are copied to a temp dir as lens-eval-<task>.json and read with the harness loader.

    /root/jpp_lens/.venv/bin/python scripts/v4_bench_cache.py --model Qwen/Qwen3.5-9B --tag 9b
    HF_HOME=/workspace/o-lens-improvement/.hf27 /root/jpp_lens/.venv/bin/python scripts/v4_bench_cache.py \
        --model Qwen/Qwen3.6-27B --tag 27b

Writes data/model_correctness_v4_<tag>.csv, artifacts/bench/v4_<bench name>.pt and auto/split_v4_<tag>.json
(fit = the ORIGINAL split's fit list, dev = [], test = every v4 item that passes the harness inclusion
rule: correct-or-ungradeable and scoreable). All v4 bench rows are marked "held-out" (none are fit rows).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

JPP = os.environ.get("JPP_LENS_DIR", "/root/jpp_lens")
R = Path("/workspace/o-lens-improvement")
V4 = R / "data/evals_v4"
OUT = {"9b": "artifacts/bench/v4_qwen3.5-9b.pt", "27b": "artifacts/bench/v4_qwen3.6-27b.pt"}
ORIG_BENCH = {"9b": "artifacts/bench/qwen3.5-9b.pt", "27b": "artifacts/bench/qwen3.6-27b.pt"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True, choices=list(OUT))
args = ap.parse_args()

os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import pandas as pd  # noqa: E402
import torch as t  # noqa: E402

import jpp_cli  # noqa: E402
from lens_evals.readout_evals import readout_evals as re_mod  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import (  # noqa: E402
    MACRO_EVALS,
    READOUT_EVALS_DATA_DIR,
    filter_items_by_correctness,
    load_readout_eval_items,
    non_semantic_token_ids,
)
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import vocab_size_of  # noqa: E402

# ---- frozen files: md5 check, then the harness loader on a temp copy with the harness's file names
manifest = json.loads((V4 / "MANIFEST.json").read_text())
tmp = tempfile.mkdtemp(prefix="v4evals_")
for fname, meta in manifest["files"].items():
    b = (V4 / fname).read_bytes()
    if hashlib.md5(b).hexdigest() != meta["md5"]:
        raise SystemExit(f"{fname}: md5 differs from MANIFEST.json (file not yet synced or edited)")
    Path(tmp, f"lens-eval-{meta['task']}.json").write_bytes(b)
items = load_readout_eval_items(tmp, slugs=MACRO_EVALS)
logging.info("loaded %d v4 items", len(items))

model = get_hf_model(args.model, attn_implementation="sdpa")
max_seq_len = jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN
layers = list(range(model.n_layers - 1))

# ---- correctness grading (olens.evaluate grade-correctness: same runner and max_seq_len)
grader = ReadoutEvalRunner(model, {}, layers=[0], max_seq_len=512)
graded = grader.grade_model_correctness(items, hf_model_name=args.model)
csv_path = R / f"data/model_correctness_v4_{args.tag}.csv"
with open(csv_path, "w") as f:
    graded.to_csv(f, index=False)
print(graded.assign(c=graded[args.model].astype(str)).groupby(["eval", "c"]).size().unstack(fill_value=0))

# ---- inclusion rule, as jpp_cli.load_items_and_exclusions: correctness filter, then scoreable items
kept = filter_items_by_correctness(items, hf_model_name=args.model, correctness_csv=str(csv_path))
runner = jpp_cli.readout_runner(model, layers=layers, max_seq_len=max_seq_len)
scoreable = runner.scoreable_items(kept)
# Readout-Filtering exclusion: non-semantic ids minus the candidates of the original six evals AND v4.
excluded_ids = non_semantic_token_ids(
    model.tokenizer,
    vocab_size=vocab_size_of(model),
    items=list(load_readout_eval_items(READOUT_EVALS_DATA_DIR)) + list(items),
)

res = runner.record_residuals(scoreable, hf_model_name=args.model)
keys = list(zip(res.eval_slugs, res.item_names, strict=True))
if keys != [(it.eval_slug, it.name) for it in scoreable]:
    raise SystemExit("record_residuals kept a different item list than scoreable_items")
positions = []
for it, cands in zip(scoreable, res.candidate_ids_P_list, strict=True):
    ro = re_mod._item_readout_and_candidates(model, it, max_seq_len)
    assert ro is not None and ro.candidate_ids_by_intermediate == cands
    positions.append(int(ro.readout_position))

payload = {
    "model": args.model,
    "layers": [int(layer) for layer in res.layers],
    "eval_slugs": list(res.eval_slugs),
    "item_names": list(res.item_names),
    "resid": {int(layer): v.float().cpu().contiguous() for layer, v in res.residuals_L_dict_PN.items()},
    "candidates": [{str(k): [int(i) for i in v] for k, v in c.items()} for c in res.candidate_ids_P_list],
    "split": ["held-out"] * len(keys),
    "excluded_ids": t.tensor([int(i) for i in excluded_ids], dtype=t.long),
    "readout_positions": positions,
    "max_seq_len": int(max_seq_len),
    "fit_fraction": 0.0,
    "seed": 0,
    "correctness_csv": str(csv_path),
    "v4_manifest_md5": {k: v["md5"] for k, v in manifest["files"].items()},
}
out_path = R / OUT[args.tag]
with open(out_path, "wb") as f:  # write in place (network fs)
    t.save(payload, f)
check = t.load(out_path, map_location="cpu", weights_only=True)
assert check["item_names"] == payload["item_names"]

# ---- exclusion list vs the original bench's (should be identical: v4 candidates are word tokens)
orig = t.load(R / ORIG_BENCH[args.tag], map_location="cpu", weights_only=True)
oe, ve = set(orig["excluded_ids"].tolist()), set(payload["excluded_ids"].tolist())
print(f"excluded ids: v4 {len(ve)}, original {len(oe)}, only-v4 {len(ve - oe)}, only-original {len(oe - ve)}")
assert orig["layers"] == payload["layers"], "layer set differs from the original bench"

# ---- split file: fit = ORIGINAL fit list (unchanged ctx.fit), dev = [], test = all included v4 items
orig_split = json.loads((R / f"auto/split_{args.tag}.json").read_text())
split = {"dev": [], "test": sorted([list(k) for k in keys]), "fit": orig_split["fit"]}
with open(R / f"auto/split_v4_{args.tag}.json", "w") as f:
    f.write(json.dumps(split, indent=0))

per_all = Counter(it.eval_slug for it in items)
per_kept = Counter(it.eval_slug for it in kept)
per_inc = Counter(e for e, _ in keys)
print(f"saved {out_path}: {len(keys)} items, {len(payload['layers'])} layers, "
      f"d={payload['resid'][payload['layers'][0]].shape[1]}")
print(f"{'eval':14s} {'written':>8s} {'correct':>8s} {'included':>9s}")
for ev in MACRO_EVALS:
    print(f"{ev:14s} {per_all[ev]:8d} {per_kept[ev]:8d} {per_inc[ev]:9d}")
print(f"{'all':14s} {len(items):8d} {len(kept):8d} {len(keys):9d}")
