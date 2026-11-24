#!/bin/bash
# Orchestrator: run heldout_runs.sh for the given tags one after another, retrying failed runs (shared GPU) up to 4 times.
R=/workspace/o-lens-improvement; LOG=$R/runs/w1/filt_logs
for T in "$@"; do
  for try in 1 2 3 4; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 20000 ]; do sleep 30; done
    rm -f $LOG/failed_heldout_$T
    bash $R/scripts/w1/heldout_runs.sh $T
    [ -f $LOG/failed_heldout_$T ] || break
    echo "retry $try for $T: $(cat $LOG/failed_heldout_$T | wc -l) failed"; sleep 60
  done
  echo "chain finished $T $(date -u +%H:%M)"
done
