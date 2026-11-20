#!/bin/bash
# g1: run g1_gen.py in ~14-minute chunks holding /root/gpu27.lock; release >= 100 s between chunks.
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
EXTRA="$1"
while true; do
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge 28000 ] && ( set -o noclobber; echo g1 > /root/gpu27.lock ) 2>/dev/null; then break; fi
    echo "$(date +%T) waiting (free ${free} MiB, lock: $(cat /root/gpu27.lock 2>/dev/null))"
    sleep 60
  done
  echo "$(date +%T) lock taken"
  $PY /root/g1/g1_gen.py 840 $EXTRA 2>&1 | grep -v -i warn | tee /root/g1/chunk_last.log
  [ "$(cat /root/gpu27.lock 2>/dev/null)" = "g1" ] && rm -f /root/gpu27.lock
  echo "$(date +%T) lock released"
  grep -q ALL_DONE /root/g1/chunk_last.log && break
  sleep 100
done
echo GENLOOP_DONE
