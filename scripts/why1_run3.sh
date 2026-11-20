#!/bin/bash
# why1 runner: waits for >= 30 GB free GPU memory before each attempt, retries a failed step up to 6 times.
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
L=/root/why1_logs
mkdir -p $L
waitgpu() {
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge 30000 ] && return
    echo "$(date +%T) waiting, free ${free} MiB" >> $L/wait.log; sleep 60
  done
}
step() {  # name, command...
  local name=$1; shift
  for a in 1 2 3 4 5 6; do
    [ -f $L/$name.ok ] && return
    waitgpu
    echo "=== attempt $a $(date +%T)" >> $L/$name.log
    if env "$@" >> $L/$name.log 2>&1 && ! tail -3 $L/$name.log | grep -q CRASH; then touch $L/$name.ok; return; fi
    sleep 120
  done
}
step analysis $PY scripts/why1_analysis3.py
for r in 4 16 64 256 1024; do
  step q2_r$r WHY1_RANK=$r $PY auto/evaluate.py --lens auto/candidates/why1_E.py --name why1_q2_r$r
done
step q3_L4_8 WHY1_LAYERS=4,8 $PY auto/evaluate.py --lens auto/candidates/why1_E.py --name why1_q3_L4_8
step q3_L12_16 WHY1_LAYERS=12,16 $PY auto/evaluate.py --lens auto/candidates/why1_E.py --name why1_q3_L12_16
step q3_L20_28 WHY1_LAYERS=20,24,28 $PY auto/evaluate.py --lens auto/candidates/why1_E.py --name why1_q3_L20_28
touch $L/ALL_DONE
