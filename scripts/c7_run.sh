#!/bin/bash
# PREREG_v4 addendum B: same recipe on Qwen3.5-2B, scored once on v4. Orchestrator only.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/c7
for pair in "jpp.py:V4_jpp_2b" "m2b_L.py:V4_L_2b" "m2b_C_shuf.py:V4_C3shuf_2b"; do
  c=${pair%%:*}; n=${pair#*:}
  [ -f auto/runs/$n/summary.json ] || $PY auto/evaluate_v4.py --lens auto/candidates/$c --name $n --model Qwen/Qwen3.5-2B --split test --orchestrator --lite > /root/c7/$n.log 2>&1
  echo "$(date -u +%T) $n $(grep -E 'pass10:|pass1:|pass10_unf|n_items|chosen lam' /root/c7/$n.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-260)"
done
touch /root/c7/DONE
