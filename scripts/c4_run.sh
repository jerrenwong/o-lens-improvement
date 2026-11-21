#!/bin/bash
# Registered fresh-set scoring of G (top-1 guard) and L (re-weighted) - PREREG_v3 amendment 6 / PREREG_v2 amendment 4. Orchestrator only.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
free() { timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mkdir -p /root/c4
run() { # v model tag candidate name
  local v=$1 model=$2 tag=$3 cand=$4 name=$5
  [ -f $R/auto/runs/$name/summary.json ] && { echo "$(date -u +%T) skip $name (exists)"; return; }
  for try in 1 2 3 4 5 6 7 8; do
    until [ "$(free)" -gt 20000 ]; do sleep 30; done
    if [ "$tag" = 27b ]; then
      HF_HOME=/workspace/o-lens-improvement/.hf27 $PY auto/evaluate_v$v.py --lens auto/candidates/$cand --name $name --model $model --split test --orchestrator --lite > /root/c4/$name.log 2>&1 && break
    else
      $PY auto/evaluate_v$v.py --lens auto/candidates/$cand --name $name --model $model --split test --orchestrator --lite > /root/c4/$name.log 2>&1 && break
    fi
    echo "$(date -u +%T) retry $name ($try)"; sleep 45
  done
  echo "$(date -u +%T) $name $(grep -E 'pass10:|pass1:|pass10_unf' /root/c4/$name.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-140)"
}
for v in 3 2; do
  run $v Qwen/Qwen3.6-27B 27b fm27_F5s.py V${v}_G_27b
  run $v Qwen/Qwen3.6-27B 27b fm27_F4.py V${v}_L_27b
  run $v Qwen/Qwen3.5-9B 9b fm27_F5s.py V${v}_G_9b
  run $v Qwen/Qwen3.5-9B 9b fm27_F4.py V${v}_L_9b
done
echo "== done $(date -u +%T)"; touch /root/c4/DONE
