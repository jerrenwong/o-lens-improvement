#!/bin/bash
# Orchestrator only: A-lens rows on Qwen3.5-2B, test split + v4 (PREREG_v4 addendum E, extension to 2B).
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/al1; unset HF_HOME
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for kind in test v4; do for row in J0 JPP0 none J JPP; do
  if [ $kind = test ]; then EV=auto/evaluate.py; n=T_al_${row}_2b; else EV=auto/evaluate_v4.py; n=V4_al_${row}_2b; fi
  [ -f auto/runs/$n/summary.json ] && continue
  for try in 1 2 3; do ALENS_BASE=$row $PY $EV --model Qwen/Qwen3.5-2B --lens auto/candidates/alens.py --name $n --lite --split test --orchestrator > /root/al1/$n.log 2>&1 && break; sleep 30; done
  echo "$n $(grep -E 'chosen lam|pass10:|pass1:|n_items' /root/al1/$n.log | sed 's/  */ /g' | tr '\n' ' ')" >> /root/al1/results_2b.txt
done; done; touch /root/al1/DONE_2b
