#!/bin/bash
# W7 / Experiment 3 chain (DEV only).  bash scripts/w7/chain.sh <step> ...
#   fit <tag> : C2..C5 maps for one model (all, nomerge, mergeall, merge(+diag)), K=10, T=1
#   eval <spec-file> : run dev evals listed in a file (one "name|env|model|filter" per line), 6 in parallel
#   outdiag <tag>
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$R/runs/w7/logs; mkdir -p $L
declare -A MOD=([2b]=Qwen/Qwen3.5-2B [9b]=Qwen/Qwen3.5-9B)
step=$1; shift
case $step in
fit)
  t=$1; K=${2:-10}; T=${3:-1}; modes=${4:-"merge all nomerge mergeall"}
  for m in $modes; do for q in topic spec; do
    d=""; [ $m = merge ] && [ $K = 10 ] && [ $T = 1 ] && d=--diag
    $PY scripts/w7/w7_fit.py ${MOD[$t]} $q --mode $m --K $K --T $T $d > $L/fit_${t}_${q}_${m}_k${K}_t${T}.log 2>&1 || echo "FAILED fit $t $q $m $K $T" >> $L/errors
  done; done ;;
eval)
  run1() { IFS='|' read -r n envs model flt <<< "$1"
    [ -f auto/runs/$n/summary.json ] && exit 0
    env $envs $PY auto/evaluate.py --model $model --lens auto/candidates/w1_alens.py --lite --filter $flt --name $n > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/errors; }
  export -f run1; export PY L R
  grep -v '^#' $1 | xargs -P 6 -I{} bash -c 'run1 "$@"' _ {} ;;
outdiag)
  $PY scripts/w7/w7_outdiag.py ${MOD[$1]} > $L/outdiag_$1.log 2>&1 || echo "FAILED outdiag $1" >> $L/errors ;;
esac
