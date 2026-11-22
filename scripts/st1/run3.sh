#!/bin/bash
# st1 chain (pod, tmux st1*): bash run.sh <2b|9b|27b> <stage...>; stages: fit p3 eval qual:<var> ; writes /root/st1/DONE_<tag> or FAILED_<tag>
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/st1; mkdir -p $L
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; shift
case $TAG in
  2b) M=Qwen/Qwen3.5-2B; EV=auto/evaluate.py; NP3=6500; NEV=5000 ;;
  9b) M=Qwen/Qwen3.5-9B; EV=auto/evaluate.py; NP3=10500; NEV=6500 ;;
  27b) M=Qwen/Qwen3.6-27B; EV=auto/evaluate27.py; export HF_HOME=$R/.hf27; NP3=12500; NEV=18000;;
esac
cd $R
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}'; }
retry() { local need=$1 lf=$2; shift 2
  for a in 1 2 3 4 5 6 7 8; do
    while [ "$(free_mib)" -lt "$need" ]; do sleep 60; done
    echo "[st1] $(date +%T) attempt $a: $*" >> $lf
    "$@" >> $lf 2>&1 && return 0
    if tail -60 $lf | grep -qi "out of memory\|OutOfMemory"; then sleep 120; else return 1; fi
  done; return 1; }
CANDS=${CANDS:-"jpp st1_B0 st1_p1_t10 st1_p2_t05 st1_p2_t10 st1_p2_t20 st1_p3_t05 st1_p3_t10 st1_p3_t20 st1_p1_t00 st1_p1f_t10 st1_p2f_t10"}
for st in "$@"; do
  case $st in
    fit) retry 14000 $L/fit_$TAG.log $PY scripts/st1/st1_fit.py $M || { echo fit > $L/FAILED_$TAG; exit 1; } ;;
    p3) retry $NP3 $L/p3_$TAG.log $PY scripts/st1/st1_p3.py $M 0.5 1.0 2.0 || { echo p3 > $L/FAILED_$TAG; exit 1; } ;;
    eval) for c in $CANDS; do
            for s in "" _A; do
              case $c in jpp|st1_jppu*) [ -n "$s" ] && continue;; esac
              n=${c}${s}_$TAG; [ $c = jpp ] && n=st1_jpp_$TAG
              [ -f auto/runs/$n/summary.json ] && continue
              retry $NEV $L/eval_$n.log $PY $EV --model $M --lens auto/candidates/$c$s.py --name $n --lite || echo "eval $n failed" >> $L/errors_$TAG
            done
          done ;;
    rep) for c in st1_jppu st1_B0; do n=${c}_rep_$TAG; [ -f auto/runs/$n/summary.json ] && continue
           retry $NEV $L/eval_$n.log $PY $EV --model $M --lens auto/candidates/$c.py --name $n --lite || echo "eval $n failed" >> $L/errors_$TAG
         done ;;
    qual:*) v=${st#qual:}; retry $NEV $L/qual_$TAG.log $PY $EV --model $M --lens auto/candidates/st1_qual_${TAG}.py --name st1_qual_$TAG --lite ;;
  esac
done
echo OK > $L/DONE3_$TAG
