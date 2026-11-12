#!/bin/bash
# Score lenses with the jpp_lens readout eval (held-out:0 items, filtered + unfiltered).
# Usage: bash scripts/eval_lenses.sh <out_dir> <lens.pt|logit>...
set -euo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
OUT=$1; shift; mkdir -p "$OUT"
CSV=$R/data/model_correctness.csv
[ -f "$CSV" ] || $PY -m olens.evaluate grade-correctness --hf-model-name Qwen/Qwen3.5-2B --out "$CSV"
COMMON="--hf-model-name Qwen/Qwen3.5-2B --layers 3,6,9,12,15,18,21 --correctness-csv $CSV --lens $*"
$PY -m olens.evaluate evaluate $COMMON --out "$OUT/readout_filtered.csv"
$PY -m olens.evaluate evaluate $COMMON --no-readout-filter --out "$OUT/readout_unfiltered.csv"
