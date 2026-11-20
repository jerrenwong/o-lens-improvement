#!/bin/bash
# x1: run one command with retry on CUDA OOM (up to 6 attempts, waits for >= 30 GB free GPU memory first).
# usage: bash scripts/x1_run.sh <logfile> <cmd...>
log=$1; shift
for attempt in 1 2 3 4 5 6; do
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge 30000 ] && break
    echo "[x1_run] waiting, free ${free} MiB" >> "$log.wait"; sleep 60
  done
  "$@" > "$log" 2>&1
  rc=$?
  if grep -q "CUDA out of memory\|OutOfMemoryError" "$log"; then
    echo "[x1_run] OOM attempt $attempt" >> "$log.wait"; sleep 120; continue
  fi
  exit $rc
done
