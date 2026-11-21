#!/bin/bash
# mlp1 chain 3: after the all-layer runs and the export: extended-lam remix, fp32 precision check, DEV benchmark
# (auto/evaluate.py --lite, dev split only) for ridge, B, C (+ P0, L4096 as labelled extras).
PY=/root/jpp_lens/.venv/bin/python
cd /workspace/o-lens-improvement || exit 1
S=scripts/mlp1
until [ -f /root/mlp1/DONE_all ] || [ -f /root/mlp1/FAILED_all ]; do sleep 60; done
until grep -q EXPORT_DONE /root/mlp1/export.log; do sleep 60; done
rm -f /root/mlp1/DONE_post /root/mlp1/FAILED_post
retry() {
  local log="${@: -1}"; set -- "${@:1:$#-1}"
  for k in 1 2 3 4 5 6 7 8 9 10; do
    "$@" >> "$log" 2>&1 && return 0
    tail -40 "$log" | grep -q OutOfMemoryError || return 1
    echo "OOM retry $k" >> "$log"; sleep 120
  done
  return 1
}
retry $PY $S/mlp1_remix.py $S/final_all.json /root/mlp1/remix.log || touch /root/mlp1/FAILED_post
retry $PY $S/mlp1_fp32check.py $S/final_all.json /root/mlp1/fp32.log || touch /root/mlp1/FAILED_post
for g in ridge B C P0 L4096; do
  for t in 1 2 3 4 5 6 7 8; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt 10000 ]; then sleep 60; continue; fi
    $PY auto/evaluate.py --lens auto/candidates/mlp1_$g.py --name mlp1_${g}_9b --lite > /root/mlp1/bench_$g.log 2>&1
    grep -q "n_items" /root/mlp1/bench_$g.log && break
    echo "bench $g failed try $t" >> /root/mlp1/bench_chain.log; sleep 60
  done
done
touch /root/mlp1/DONE_post
