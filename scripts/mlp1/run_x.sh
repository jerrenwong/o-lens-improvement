#!/bin/bash
# mlp1 chain 1b (after run_l20.sh): Q (B + shift) lr grid, inferable-only C / D, seeds for new groups, final pick,
# then chain 2 (all layers).
PY=/root/jpp_lens/.venv/bin/python
cd /workspace/o-lens-improvement || exit 1
S=scripts/mlp1
until [ -f /root/mlp1/DONE_l20 ] || [ -f /root/mlp1/FAILED_l20 ]; do sleep 30; done
[ -f /root/mlp1/FAILED_l20 ] && exit 1
rm -f /root/mlp1/DONE_x /root/mlp1/FAILED_x
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
trap 'touch /root/mlp1/FAILED_x' ERR
$PY $S/mlp1_infer.py > /root/mlp1/infer.log 2>&1
retry $PY $S/mlp1_train.py 20 $S/specs_L20_x.json /root/mlp1/l20_sx.log
python3 $S/mlp1_pick.py 20 stage3 $S/specs_L20_s3b.json > /root/mlp1/pick3b.log 2>&1
retry $PY $S/mlp1_train.py 20 $S/specs_L20_s3b.json /root/mlp1/l20_s3b.log
python3 $S/mlp1_pick.py 20 final $S/final_L20.json > /root/mlp1/pickf.log 2>&1
touch /root/mlp1/DONE_x
bash $S/run_all.sh
