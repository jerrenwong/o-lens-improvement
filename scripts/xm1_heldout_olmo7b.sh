#!/bin/bash
# Orchestrator only: PREREG_v5 held-out scoring for allenai/Olmo-3-7B-Instruct (commands from runs/xm1/olmo7b/handover.md).
R=/workspace/o-lens-improvement; cd $R; mkdir -p /root/xh
grep -E "^cd /workspace/o-lens-improvement && .*--orchestrator" runs/xm1/olmo7b/handover.md > /root/xh/cmds_olmo7b.txt
n=0
while IFS= read -r line; do
  n=$((n+1)); name=$(echo "$line" | sed 's/.*--name \([A-Za-z0-9_]*\).*/\1/')
  [ -f auto/runs/$name/summary.json ] && continue
  for try in 1 2 3 4; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt 18000 ]; do sleep 30; done
    bash -c "$line" > /root/xh/$name.log 2>&1 && break; sleep 45
  done
  echo "$name $(grep -E 'chosen lam|pass10:|pass1:|n_items' /root/xh/$name.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-260)" >> /root/xh/results_olmo7b.txt
done < /root/xh/cmds_olmo7b.txt
touch /root/xh/DONE_olmo7b
