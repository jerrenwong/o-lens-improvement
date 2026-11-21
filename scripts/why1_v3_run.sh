#!/bin/bash
# why1 v3 check runner: needs runs/why1/bagmaps.pt; waits for >= 30 GB free GPU, retries up to 6 times (OOM).
# Launch on the pod: tmux new-session -d -s why1v3 "bash /workspace/o-lens-improvement/scripts/why1_v3_run.sh"
cd /workspace/o-lens-improvement
L=/root/why1_v3.log
for a in 1 2 3 4 5 6; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 30000 ]; do sleep 60; done
  echo "=== attempt $a $(date +%T)" >> $L
  /root/jpp_lens/.venv/bin/python scripts/why1_v3.py >> $L 2>&1
  grep -q "V3 DONE" $L && break
  grep -qE "AssertionError" $L && break   # repro/position failure: do not retry
  sleep 120
done
cp $L runs/why1/why1_v3.log
python3 scripts/r2_boot.py V3_T_9b-V3_jpp_9b V3_e2_A_9b-V3_T_9b V3_T05_9b-V3_jpp_9b >> runs/why1/why1_v3.log 2>&1 || true
