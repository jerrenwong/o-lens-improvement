#!/bin/bash
# ex2 eval chain. $1 = "early" (after S2: stats, specs, S3/S6 evals) or "late" (after S4/S5 fits: all remaining evals, boot, report)
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
if [ "$1" == "early" ]; then
  until [ -f runs/ex2/s2_9b_w2.pt ]; do sleep 30; done
  $PY scripts/ex2/s12_stats.py > /root/ex2/stats.log 2>&1 || exit 1
  $PY scripts/ex2/build_specs.py > /root/ex2/specs.log 2>&1 || exit 1
  for m in 2b 9b; do
    names=$($PY -c "import json;S=json.load(open('runs/ex2/specs_$m.json'));print(','.join(n for n in S if ('rel_' in n or n.endswith(('_L','_pool')) or n.split('_')[-1].startswith('b') or '_perp_' in n) and '_drop' not in n))")
    $PY scripts/ex2/multi_eval.py --model $m --specs runs/ex2/specs_$m.json --names $names >> /root/ex2/eval_$m.log 2>&1
  done
  echo done > /root/ex2/early_done
else
  until [ -f /root/ex2/chain1_done ]; do sleep 60; done
  until [ -f /root/ex2/early_done ]; do sleep 60; done
  $PY scripts/ex2/build_specs.py > /root/ex2/specs.log 2>&1 || exit 1
  for m in 2b 9b; do
    $PY scripts/ex2/multi_eval.py --model $m --specs runs/ex2/specs_$m.json >> /root/ex2/eval_$m.log 2>&1
  done
  for m in 2b 9b; do $PY scripts/ex2/ex2_boot.py --model $m >> /root/ex2/boot.log 2>&1; done
  $PY scripts/ex2/ex2_report.py >> /root/ex2/report.log 2>&1
  echo done > /root/ex2/late_done
fi
