#!/bin/bash
# ask2 9B chain: wait for collect, fit, then dev evaluations (one at a time)
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask2
cd $R
until grep -q -E "^saved|Traceback" $L/collect_9b.log; do sleep 20; done
grep -q Traceback $L/collect_9b.log && { echo COLLECT_FAILED > $L/chain9_done; exit 1; }
$PY scripts/ask2/ask2_fit.py Qwen/Qwen3.5-9B > $L/fit_9b.log 2>&1 || { echo FIT_FAILED > $L/chain9_done; exit 1; }
for c in qual A C_shuf B_both D_alone; do
  $PY auto/evaluate.py --lens auto/candidates/ask2_$c.py --name ask2_${c}_9b > $L/eval_${c}_9b.log 2>&1
done
echo OK > $L/chain9_done
