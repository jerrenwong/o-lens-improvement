#!/bin/bash
# Registered v4 scoring (auto/PREREG_v4.md). Orchestrator only. J++ sanity first, then the frozen primary lens L, then secondaries.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
free() { timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mkdir -p /root/c5
run() { # model tag candidate name
  local model=$1 tag=$2 cand=$3 name=$4
  [ -f $R/auto/runs/$name/summary.json ] && { echo "$(date -u +%T) skip $name (exists)"; return; }
  for try in 1 2 3 4 5 6 7 8; do
    until [ "$(free)" -gt 20000 ]; do sleep 30; done
    if [ "$tag" = 27b ]; then
      HF_HOME=/workspace/o-lens-improvement/.hf27 $PY auto/evaluate_v4.py --lens auto/candidates/$cand --name $name --model $model --split test --orchestrator --lite > /root/c5/$name.log 2>&1 && break
    else
      $PY auto/evaluate_v4.py --lens auto/candidates/$cand --name $name --model $model --split test --orchestrator --lite > /root/c5/$name.log 2>&1 && break
    fi
    echo "$(date -u +%T) retry $name ($try)"; sleep 45
  done
  echo "$(date -u +%T) $name $(grep -E 'pass10:|pass1:|pass10_unf|n_items' /root/c5/$name.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-170)"
}
for mt in "Qwen/Qwen3.5-9B:9b" "Qwen/Qwen3.6-27B:27b"; do
  m=${mt%%:*}; t=${mt#*:}
  until [ -f $R/artifacts/bench/v4_$( [ $t = 9b ] && echo qwen3.5-9b || echo qwen3.6-27b ).pt ] && [ -f $R/auto/split_v4_$t.json ] && { [ $t = 9b ] || [ -f /root/v4/DONE ]; }; do
    [ -f /root/v4/FAILED ] && { echo "v4 build FAILED"; exit 1; }
    sleep 30
  done
  run $m $t jpp.py V4_jpp_$t
  run $m $t fm27_F4.py V4_L_$t
  run $m $t ask3_A.py V4_C3_$t
  run $m $t fm27_F5s.py V4_G_$t
  run $m $t ask3_C_shuf.py V4_C3shuf_$t
done
echo "== done $(date -u +%T)"; touch /root/c5/DONE
