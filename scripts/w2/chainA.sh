#!/bin/bash
# W2a chain: (1) reproduce st1_p3v_t10 on 9B dev; (2) CE fits, both questions, K 10/50/200, 2B + 9B;
# (3) dev rows of w2_alens_ce.py (none, JPP) per model and K.   Usage: bash scripts/w2/chainA.sh [steps]
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$R/runs/w2/logs; mkdir -p $L
STEPS=${1:-"repro fit eval"}
declare -A MOD=([2b]=Qwen/Qwen3.5-2B [9b]=Qwen/Qwen3.5-9B)
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt ${1:-16000} ]; do sleep 30; done; }
ev() { # name env... -- args
  n=$1; shift
  [ -f auto/runs/$n/summary.json ] && { echo "skip $n"; return; }
  waitmem 16000
  env "$@" > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/errors
  echo "$n $(grep -E 'chosen lam|dev_pass10:|dev_pass1:' $L/$n.log | sed 's/  */ /g' | tr '\n' ' ')" >> $L/resultsA.txt
}
for s in $STEPS; do case $s in
repro)
  ev w2_repro_st1_p3v_t10_9b $PY auto/evaluate.py --model Qwen/Qwen3.5-9B --lens auto/candidates/st1_p3v_t10.py --name w2_repro_st1_p3v_t10_9b --lite ;;
fit)
  for t in 2b 9b; do for q in topic spec; do
    [ -f .hf27/w2/E_${t}_${q}_ce_k200.pt ] && continue
    waitmem 20000
    $PY scripts/w2/w2_ce_fit.py ${MOD[$t]} $q 10 50 200 > $L/fit_${t}_${q}.log 2>&1 || echo "FAILED fit $t $q" >> $L/errors
  done; done ;;
eval)
  for t in 2b 9b; do for k in 10 50 200; do for row in none JPP; do
    ev w2_ce_k${k}_${row}_$t W2_K=$k ALENS_BASE=$row $PY auto/evaluate.py --model ${MOD[$t]} --lens auto/candidates/w2_alens_ce.py --name w2_ce_k${k}_${row}_$t --lite
  done; done; done ;;
esac; done
echo done > $L/chainA_done_$(echo $STEPS | tr ' ' _)
