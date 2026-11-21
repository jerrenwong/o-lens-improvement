#!/bin/bash
# ask3 27B chain: collect (format family $1), fit, dev evals. Before each GPU step wait until the GPU is
# nearly empty (two checks 30 s apart, < 22 GB used of 80), then reserve memory (ask3_res27.py) so nothing squeezes in.
export HF_HOME=/workspace/o-lens-improvement/.hf27
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask3
W="$PY $R/scripts/ask3/ask3_resv.py"
M=Qwen/Qwen3.6-27B
cd $R
waitgpu() {
  while true; do
    a=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); sleep 30
    b=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$a" -lt 22000 ] && [ "$b" -lt 22000 ] && break
    sleep 30
  done
  date +"%T gpu free" >> $L/chain.log
}
if [ ! -f $R/.hf27/ask3/acts_27b.pt ]; then
  waitgpu; $W scripts/ask3/ask3_collect2.py $M $1 > $L/collect.log 2>&1 || { echo COLLECT_FAILED > $L/chain_done; exit 1; }
fi
waitgpu; $W scripts/ask3/ask3_fit.py $M > $L/fit.log 2>&1 || { echo FIT_FAILED > $L/chain_done; exit 1; }
for c in jpp m27_A ask3_A ask3_C_shuf ask3_D_alone; do
  waitgpu; $W auto/evaluate27.py --model $M --lens auto/candidates/$c.py --name ask3_${c}_27b > $L/eval_${c}.log 2>&1
done
echo OK > $L/chain_done
