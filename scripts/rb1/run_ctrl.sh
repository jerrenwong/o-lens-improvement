#!/bin/bash
# rb1 controls (pod, tmux rb1ctrl): fit control maps for all models now (GPU light), then wait for the main eval queue
# (run_eval.sh writes DONE/FAILED), then run the control evals (<= 3 in parallel), per-item comparisons; final DONE.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/rb1; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for TAG in 9b 27b 2b; do
  case $TAG in 2b) M=Qwen/Qwen3.5-2B;; 9b) M=Qwen/Qwen3.5-9B;; 27b) M=Qwen/Qwen3.6-27B;; esac
  if [ $TAG = 27b ]; then export HF_HOME=$R/.hf27; else unset HF_HOME; fi
  [ -f .hf27/rb1/ctrl_$TAG.json ] || $PY scripts/rb1/rb1_ctrl.py $M > $L/ctrl_$TAG.log 2>&1 || { echo "ctrl $TAG" >> $L/errors; }
done
while [ ! -f $L/DONE ] && [ ! -f $L/FAILED ]; do sleep 60; done
rm -f $L/DONE
CV="shufall priorall shuf300 prior300 shuf3000 prior3000 shufall_atL priorall_atL shuf300_atL prior300_atL shuf3000_atL prior3000_atL"
for v in r300_0 r3000_0 rall; do for k in 1 4 16 64 256; do CV="$CV ${v}_k$k"; done; done
for TAG in 9b 27b 2b; do
  for v in $CV; do echo "$TAG rb1_$v rb1_${v}_$TAG"; done | xargs -P 3 -L 1 bash scripts/rb1/one_eval.sh
  $PY scripts/rb1/rb1_items.py $TAG > $L/items_$TAG.log 2>&1
  touch $L/CTRLDONE_$TAG
done
if [ -f $L/errors ]; then cp $L/errors $L/FAILED; else touch $L/DONE; fi
