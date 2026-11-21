#!/bin/bash
# ask5: after collection, fit 9B + dev evals, then fit 27B + dev evals (DEV only, --lite).
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask5
W="$PY $R/scripts/ask3/ask3_resv.py"
cd $R
until [ -f $L/collect_done ] || [ -f $L/FAILED ]; do sleep 20; done
[ -f $L/FAILED ] && exit 1
NEED_GB=22 $W scripts/ask5/ask5_fit.py Qwen/Qwen3.5-9B > $L/fit9.log 2>&1 || { echo FIT9 > $L/FAILED; exit 1; }
for c in jpp fm27_F4 ask5_a ask5_b ask5_c ask5_d; do
  $PY auto/evaluate.py --lens auto/candidates/$c.py --name ask5_${c}_9b --lite > $L/eval9_$c.log 2>&1 || echo "eval9 $c failed" >> $L/errors
done
export HF_HOME=$R/.hf27
NEED_GB=56 $W scripts/ask5/ask5_fit.py Qwen/Qwen3.6-27B > $L/fit27.log 2>&1 || { echo FIT27 > $L/FAILED; exit 1; }
for c in jpp fm27_F4 ask5_a ask5_b ask5_c ask5_d; do
  $PY auto/evaluate27.py --model Qwen/Qwen3.6-27B --lens auto/candidates/$c.py --name ask5_${c}_27b --lite > $L/eval27_$c.log 2>&1 || echo "eval27 $c failed" >> $L/errors
done
echo OK > $L/DONE
