#!/bin/bash
# xm1 step 2 (pod, tmux): correctness grading (olens.evaluate grade-correctness -> data/model_correctness_xm1.csv),
# benchmark residual cache (scripts/bench_cache.py, unchanged), dev/test split (make_split_x.py), v4 cache.
# Usage: bash scripts/xm1/bench.sh <model> <tag>
set -euo pipefail
M=$1; T=$2
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/xm1
export HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH=$R/src
cd $R
$PY -m olens.evaluate grade-correctness --hf-model-name $M --out $R/data/model_correctness_xm1.csv > $L/grade_$T.log 2>&1
$PY scripts/bench_cache.py --model $M --out $R/.hf27/xm1/bench_$T.pt --correctness-csv $R/data/model_correctness_xm1.csv > $L/bench_$T.log 2>&1
$PY scripts/xm1/make_split_x.py $T > $L/split_$T.log 2>&1
$PY scripts/xm1/v4_bench_cache_x.py --model $M --tag $T > $L/v4bench_$T.log 2>&1
echo OK > $L/bench_done_$T
