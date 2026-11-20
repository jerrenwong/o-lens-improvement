#!/bin/bash
cd /root/al1
PY=/root/jpp_lens/.venv/bin/python
until grep -q ALL_DONE gen.log; do sleep 30; done
for t in 1 2 3; do $PY collect.py > collect.log 2>&1 && break; echo "collect failed try $t" >> chain.log; sleep 60; done
for t in 1 2 3 4 5 6 7 8; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -lt 14000 ]; then echo "$(date +%T) wait free $free" >> chain.log; sleep 60; continue; fi
  $PY fit.py > fit.log 2>&1 && break; echo "$(date +%T) fit failed try $t" >> chain.log; sleep 60
done
echo CHAIN_DONE >> chain.log
