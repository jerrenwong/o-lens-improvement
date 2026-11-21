#!/bin/bash
# Post-hoc controls for L on v4 (not registered): shuffled map and constant-part-only at L's weights. Orchestrator only.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/c8
for mt in "Qwen/Qwen3.5-9B:9b" "Qwen/Qwen3.6-27B:27b"; do
  m=${mt%%:*}; t=${mt#*:}
  for c in c8_Lshuf c8_Lprior; do
    n=V4_${c#c8_}_$t
    if [ ! -f auto/runs/$n/summary.json ]; then
      if [ $t = 27b ]; then HF_HOME=/workspace/o-lens-improvement/.hf27 $PY auto/evaluate_v4.py --lens auto/candidates/$c.py --name $n --model $m --split test --orchestrator --lite > /root/c8/$n.log 2>&1
      else $PY auto/evaluate_v4.py --lens auto/candidates/$c.py --name $n --model $m --split test --orchestrator --lite > /root/c8/$n.log 2>&1; fi
    fi
    echo "$(date -u +%T) $n $(grep -E 'pass10:|pass1:|pass10_unf' /root/c8/$n.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-150)"
  done
done
touch /root/c8/DONE
