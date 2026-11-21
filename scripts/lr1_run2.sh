#!/bin/bash
# lr1: refit weights on truncated experts (unchanged fit-weights via r1_jpp.py, items fit:0) and score lens files
# on the dev split. Resumable (skips finished steps); waits for >= 30 GB free GPU; retries CUDA OOM up to 8 times.
#   bash scripts/lr1_run.sh fit  <experts.pt> <out_lens.pt>
#   bash scripts/lr1_run.sh eval <lens.pt> <run_name>
#   bash scripts/lr1_run.sh list <file with lines "fit EX OUT" / "eval LENS NAME">
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOGD=/root/lr1_logs; mkdir -p $LOGD
wait_gpu() { while true; do sleep $((RANDOM % 90 + 10)); f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$f" -ge 30000 ] && return; sleep 60; done; }
retry() {  # log cmd...
  local log=$1; shift
  for i in 1 2 3 4 5 6 7 8; do
    wait_gpu
    "$@" > $log 2>&1 && return 0
    if grep -q "out of memory\|OutOfMemoryError" $log; then echo "OOM try $i: $log"; sleep $((RANDOM % 240 + 60)); else return 1; fi
  done; return 1; }
fit() {  # experts out
  [ -f "$2" ] && [ -f "${2%.pt}_expert_weights.json" ] && { echo "skip fit $2"; return 0; }
  local n=$(basename ${2%.pt})
  retry $LOGD/fit_$n.log $PY $R/scripts/r1_jpp.py fit-weights --hf-model-name Qwen/Qwen3.5-9B --layers 4,8,12,16,20,24,28 \
    --experts "$1" --checkpoint-name "$n" --items fit:0 --correctness-csv $R/data/model_correctness.csv \
    --cache-path /root/r2_readout_resid_cache_9b.pt --out "$2"
  echo "fit $n rc=$?"; cp $LOGD/fit_$n.log $R/.hf27/lr1/logs/fit_$n.log 2>/dev/null; }
ev() {  # lens name
  [ -f $R/auto/runs/$2/summary.json ] && { echo "skip eval $2"; return 0; }
  cd $R; export R2_LENS="$1"; retry $LOGD/ev_$2.log $PY auto/evaluate.py --lens auto/candidates/lr1_lensfile.py --name $2
  echo "eval $2 rc=$?"; cat $LOGD/ev_$2.log > $R/.hf27/lr1/logs/ev_$2.log; }
case $1 in
  fit) fit "$2" "$3" ;;
  eval) ev "$2" "$3" ;;
  list) while read -r k a b; do [ -z "$k" ] && continue; case $k in fit) fit "$a" "$b";; eval) ev "$a" "$b";; esac; done < "$2"
        echo "LIST DONE $2" ;;
esac
