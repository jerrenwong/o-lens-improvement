#!/bin/bash
# rb1 fits (pod, tmux rb1fit): ridge subsets + P3v subsets per model, in order 9b 27b 2b; touches /root/rb1/READY_<tag>
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/rb1; mkdir -p $L; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for TAG in ${TAGS:-9b 27b 2b}; do
  case $TAG in 2b) M=Qwen/Qwen3.5-2B;; 9b) M=Qwen/Qwen3.5-9B;; 27b) M=Qwen/Qwen3.6-27B;; esac
  if [ $TAG = 27b ]; then export HF_HOME=$R/.hf27; else unset HF_HOME; fi
  [ -f .hf27/rb1/fit_$TAG.json ] || $PY scripts/rb1/rb1_fit.py $M > $L/fit_$TAG.log 2>&1 || { echo "fit $TAG" >> $L/FAILED; exit 1; }
  $PY scripts/rb1/rb1_p3v.py $M r1000_0 r1000_1 r10000_0 r10000_1 pall > $L/p3v_$TAG.log 2>&1 || { echo "p3v $TAG" >> $L/FAILED; exit 1; }
  touch $L/READY_$TAG
done
