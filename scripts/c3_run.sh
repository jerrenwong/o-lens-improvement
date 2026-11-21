#!/bin/bash
# Registered fresh-set scoring of the clean chat-format ask-map C (PREREG_v3 amendment 5 / PREREG_v2 amendment 3). Orchestrator only.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HOME_27=/workspace/o-lens-improvement/.hf27
free() { timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mkdir -p /root/c3
run() { # v model tag candidate name
  local v=$1 model=$2 tag=$3 cand=$4 name=$5
  [ -f $R/auto/runs/$name/summary.json ] && { echo "$(date -u +%T) skip $name (exists)"; return; }
  for try in 1 2 3 4 5 6 7 8; do
    until [ "$(free)" -gt 20000 ]; do sleep 30; done
    if [ "$tag" = 27b ]; then
      HF_HOME=$HF_HOME_27 $PY auto/evaluate_v$v.py --lens auto/candidates/$cand --name $name --model $model --split test --orchestrator --lite > /root/c3/$name.log 2>&1 && break
    else
      $PY auto/evaluate_v$v.py --lens auto/candidates/$cand --name $name --model $model --split test --orchestrator --lite > /root/c3/$name.log 2>&1 && break
    fi
    echo "$(date -u +%T) retry $name ($try)"; sleep 45
  done
  echo "$(date -u +%T) $name $(grep -E 'pass10:|pass1:|pass10_unf' /root/c3/$name.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-140)"
}
for v in 3 2; do
  for mt in "Qwen/Qwen3.5-9B:9b" "Qwen/Qwen3.6-27B:27b"; do
    m=${mt%%:*}; t=${mt#*:}
    run $v $m $t jpp.py V${v}_jpplite_$t
    run $v $m $t ask3_A.py V${v}_C3_$t
    run $v $m $t ask3_C_shuf.py V${v}_C3shuf_$t
  done
done
echo "== done $(date -u +%T)"; touch /root/c3/DONE
