#!/bin/bash
# al2 benchmark: dev split only. Retries on failure (e.g. OOM while other jobs load).
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
for name in al2_T al2_T_shuf; do
  for t in 1 2 3 4 5 6 7 8; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -lt 30000 ]; then echo "$(date +%T) wait free $free" >> /root/al2/bench_chain.log; sleep 60; continue; fi
    $PY auto/evaluate.py --lens auto/candidates/$name.py --name $name > /root/al2/bench_$name.log 2>&1
    grep -q "^n_items" /root/al2/bench_$name.log && break
    echo "$(date +%T) $name failed try $t" >> /root/al2/bench_chain.log; sleep 60
  done
done
LAM=$(grep -o "chosen lam [0-9.]*" /root/al2/bench_al2_T.log | head -1 | awk '{print $3}')
for t in 1 2 3 4 5 6 7 8; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -lt 30000 ]; then sleep 60; continue; fi
  $PY /root/al2/bench_an.py $LAM > /root/al2/bench_an.log 2>&1
  grep -q BENCH_AN_DONE /root/al2/bench_an.log && break
  echo "$(date +%T) bench_an failed try $t" >> /root/al2/bench_chain.log; sleep 60
done
echo BENCH_DONE >> /root/al2/bench_chain.log
