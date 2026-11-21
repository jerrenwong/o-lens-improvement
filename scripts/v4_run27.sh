#!/bin/bash
# v4 bench cache 27B (model correctness, residuals, split). Waits (polling every 60 s) for >= 58 GB free GPU memory
# (the full 27B model) and the 27B lock; retries up to 3 times (e.g. OOM while another job's usage varies).
# No lens is run here; scoring is left to the orchestrator.
cd /workspace/o-lens-improvement
export HF_HOME=/workspace/o-lens-improvement/.hf27
for attempt in 1 2 3; do
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge 58000 ] && ( set -o noclobber; echo v4 > /root/gpu27.lock ) 2>/dev/null; then break; fi
    echo "$(date +%T) waiting: free=${free} MiB lock=$(cat /root/gpu27.lock 2>/dev/null)"; sleep 60
  done
  echo "$(date +%T) lock taken (attempt $attempt)"
  /root/jpp_lens/.venv/bin/python scripts/v4_check_items.py data/evals_v4 --tok Qwen/Qwen3.6-27B > /root/v4/chk_final27.log 2>&1
  /root/jpp_lens/.venv/bin/python scripts/v4_bench_cache.py --model Qwen/Qwen3.6-27B --tag 27b > /root/v4/bench27b.log 2>&1
  rc=$?; rm -f /root/gpu27.lock; echo "BENCH EXIT $rc"
  [ $rc -eq 0 ] && exit 0
  cp /root/v4/bench27b.log /root/v4/bench27b.attempt$attempt.log; sleep 60
done
exit $rc
