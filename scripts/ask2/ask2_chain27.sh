#!/bin/bash
# ask2 27B chain: collect new targets, fit, dev evaluations; each step reserves the GPU first
export HF_HOME=/workspace/o-lens-improvement/.hf27
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask2
W="$PY $R/scripts/ask2/ask2_res27.py"
M=Qwen/Qwen3.6-27B
cd $R
until [ -f $L/chain9_done ]; do sleep 20; done
if [ ! -f $R/.hf27/ask2/acts_27b.pt ]; then
  $W scripts/ask2/ask2_collect.py $M > $L/collect_27b.log 2>&1 || { echo COLLECT_FAILED > $L/chain27_done; exit 1; }
fi
$W scripts/ask2/ask2_fit.py $M > $L/fit_27b.log 2>&1 || { echo FIT_FAILED > $L/chain27_done; exit 1; }
for c in qual A C_shuf B_both D_alone; do
  $W auto/evaluate27.py --model $M --lens auto/candidates/ask2_$c.py --name ask2_${c}_27b > $L/eval_${c}_27b.log 2>&1
done
echo OK > $L/chain27_done
