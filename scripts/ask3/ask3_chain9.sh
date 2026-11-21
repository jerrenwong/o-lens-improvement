#!/bin/bash
# ask3 step 4: 9B clean refit (after the 27B chain). Waits until GPU use < 55 GB, reserves 22 GB per step.
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask3
W="$PY $R/scripts/ask3/ask3_resv.py"
M=Qwen/Qwen3.5-9B
export NEED_GB=22
cd $R
until [ -f $L/chain_done ]; do sleep 30; done
waitgpu() {
  while true; do
    a=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
    b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$a" -lt 55000 ] && [ "$b" -lt 55000 ] && break
    sleep 30
  done
  date +"%T gpu free (9b)" >> $L/chain9.log
}
if [ ! -f $R/.hf27/ask3/acts_9b.pt ]; then
  waitgpu; $W scripts/ask3/ask3_collect9.py $M chat > $L/collect9.log 2>&1 || { echo COLLECT_FAILED > $L/chain9_done; exit 1; }
fi
waitgpu; $W scripts/ask3/ask3_fit9.py $M > $L/fit9.log 2>&1 || { echo FIT_FAILED > $L/chain9_done; exit 1; }
waitgpu; W0ONLY=1 $W scripts/ask3/ask3_fit9.py $M > $L/fit9_w0.log 2>&1 || { echo FITW0_FAILED > $L/chain9_done; exit 1; }
for c in jpp e2_A ask3_A ask3_C_shuf ask3_D_alone ask3_W0; do
  waitgpu; $W auto/evaluate.py --lens auto/candidates/$c.py --name ask3_${c}_9b > $L/eval9_${c}.log 2>&1
done
echo OK > $L/chain9_done
