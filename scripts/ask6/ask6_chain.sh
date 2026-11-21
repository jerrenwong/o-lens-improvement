#!/bin/bash
# ask6: collect (variant $1) 9B then 27B, fit + DEV evals (--lite) per model.
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask6
W="$PY $R/scripts/ask3/ask3_resv.py"
export ASK6_VAR=$1
cd $R
NEED_GB=22 $W scripts/ask6/ask6_collect.py Qwen/Qwen3.5-9B chat $1 > $L/collect9.log 2>&1 || { echo COLLECT9 > $L/FAILED; exit 1; }
HF_HOME=$R/.hf27 NEED_GB=56 $W scripts/ask6/ask6_collect.py Qwen/Qwen3.6-27B chat $1 > $L/collect27.log 2>&1 || { echo COLLECT27 > $L/FAILED; exit 1; }
NEED_GB=22 $W scripts/ask6/ask6_fit.py Qwen/Qwen3.5-9B > $L/fit9.log 2>&1 || { echo FIT9 > $L/FAILED; exit 1; }
for c in jpp fm27_F4 ask6_a ask6_b ask6_c ask6_d; do
  $PY auto/evaluate.py --lens auto/candidates/$c.py --name ask6_${c}_9b --lite > $L/eval9_$c.log 2>&1 || echo "eval9 $c failed" >> $L/errors
done
export HF_HOME=$R/.hf27
NEED_GB=56 $W scripts/ask6/ask6_fit.py Qwen/Qwen3.6-27B > $L/fit27.log 2>&1 || { echo FIT27 > $L/FAILED; exit 1; }
for c in jpp fm27_F4 ask6_a ask6_b ask6_c ask6_d; do
  $PY auto/evaluate27.py --model Qwen/Qwen3.6-27B --lens auto/candidates/$c.py --name ask6_${c}_27b --lite > $L/eval27_$c.log 2>&1 || echo "eval27 $c failed" >> $L/errors
done
echo OK > $L/DONE
