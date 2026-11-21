#!/bin/bash
# ex1 driver (on the pod, in tmux ex1main): waits for >= 22 GB free GPU memory, runs 9b then 2b, retries on failure.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
mkdir -p /root/ex1; rm -f /root/ex1/DONE /root/ex1/FAILED
run() {
  local tag=$1 need=$2
  for attempt in 1 2 3 4 5 6; do
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge "$need" ] && break
      echo "$(date +%T) waiting: free=${free}MiB need=${need}" >> /root/ex1/wait.log; sleep 60
    done
    if $PY $R/scripts/ex1/ex1_run.py --tag $tag > /root/ex1/run_$tag.log 2>&1; then return 0; fi
    echo "$(date +%T) $tag attempt $attempt failed" >> /root/ex1/wait.log
    grep -q "out of memory" /root/ex1/run_$tag.log || return 1
    sleep 120
  done
  return 1
}
run 9b 22000 && run 2b 9000 && { date > /root/ex1/DONE; exit 0; }
date > /root/ex1/FAILED
