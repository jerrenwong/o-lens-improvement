#!/bin/bash
# one dev evaluation: one_eval.sh <tag> <candidate file stem> <run name>
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/rb1; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in 2b) M=Qwen/Qwen3.5-2B; EV=auto/evaluate.py;; 9b) M=Qwen/Qwen3.5-9B; EV=auto/evaluate.py;;
  27b) M=Qwen/Qwen3.6-27B; EV=auto/evaluate27.py; export HF_HOME=$R/.hf27;; esac
[ -f auto/runs/$3/summary.json ] && exit 0
for a in 1 2 3 4; do
  $PY $EV --model $M --lens auto/candidates/$2.py --name $3 --lite > $L/eval_$3.log 2>&1 && exit 0
  grep -qi "out of memory" $L/eval_$3.log && { sleep 90; continue; }
  break
done
echo "eval $3" >> $L/errors; exit 1
