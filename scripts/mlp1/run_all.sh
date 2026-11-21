#!/bin/bash
# mlp1 chain 2: final configs (B, C, D + controls P0, L4096) at all layers, seed 0 first, then seeds 1, 2; analysis.
PY=/root/jpp_lens/.venv/bin/python
cd /workspace/o-lens-improvement || exit 1
S=scripts/mlp1
rm -f /root/mlp1/DONE_all /root/mlp1/FAILED_all
set -e
retry() {  # retry a step on CUDA OOM (other jobs can grab the GPU after the free-memory gate); log is last arg
  local log="${@: -1}"; set -- "${@:1:$#-1}"
  for k in 1 2 3 4 5 6 7 8 9 10; do
    "$@" >> "$log" 2>&1 && return 0
    tail -40 "$log" | grep -q OutOfMemoryError || return 1
    echo "OOM retry $k" >> "$log"; sleep 120
  done
  return 1
}
trap 'touch /root/mlp1/FAILED_all' ERR
python3 $S/mlp1_mkall.py > /root/mlp1/mkall.log 2>&1
for seed in 0 1 2; do
  for l in 20 28 24 12 8 16 4; do
    retry $PY $S/mlp1_train.py $l $S/specs_all_s$seed.json /root/mlp1/all_s$seed.log
  done
  retry $PY $S/mlp1_analyze.py $S/final_all.json $S/results/analysis_s$seed.json /root/mlp1/analyze_s$seed.log
  touch /root/mlp1/DONE_all_s$seed
done
touch /root/mlp1/DONE_all
