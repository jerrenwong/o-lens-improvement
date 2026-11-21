#!/bin/bash
# ex1 driver v2: 9b on GPU if >= 22 GB frees within 20 min, else on CPU; then 2b (GPU if >= 9 GB free, else CPU).
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
mkdir -p /root/ex1; rm -f /root/ex1/DONE /root/ex1/FAILED
run() {
  local tag=$1 need=$2 dev=cpu
  for i in $(seq 1 ${WAITN:-20}); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$need" ]; then dev=cuda; break; fi
    echo "$(date +%T) $tag waiting: free=${free}MiB need=${need}" >> /root/ex1/wait.log; sleep 60
  done
  echo "$(date +%T) $tag start on $dev" >> /root/ex1/wait.log
  $PY $R/scripts/ex1/ex1_run.py --tag $tag --device $dev > /root/ex1/run_$tag.log 2>&1 && return 0
  if [ $dev = cuda ] && grep -q "out of memory" /root/ex1/run_$tag.log; then
    echo "$(date +%T) $tag OOM on cuda, rerun on cpu" >> /root/ex1/wait.log
    $PY $R/scripts/ex1/ex1_run.py --tag $tag --device cpu > /root/ex1/run_$tag.log 2>&1 && return 0
  fi
  return 1
}
run 9b 22000 && run 2b 9000 && { date > /root/ex1/DONE; exit 0; }
date > /root/ex1/FAILED
