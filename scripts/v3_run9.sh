#!/bin/bash
# v3 bench cache 9B + J++ sanity run on v3 9B. Waits for >= 25 GB free on the GPU.
cd /workspace/o-lens-improvement
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge 25000 ]; then break; fi
  echo "$(date +%T) waiting: free=${free} MiB"; sleep 60
done
echo "$(date +%T) start"
/root/jpp_lens/.venv/bin/python scripts/v3_check_items.py data/evals_v3 > /root/v3_chk_final9.log 2>&1
/root/jpp_lens/.venv/bin/python scripts/v3_bench_cache.py --model Qwen/Qwen3.5-9B --tag 9b > /root/v3_bench9b.log 2>&1
echo "BENCH EXIT $?"
/root/jpp_lens/.venv/bin/python auto/evaluate_v3.py --lens auto/candidates/jpp.py --name V3_jpp_9b --model Qwen/Qwen3.5-9B --split test --orchestrator > /root/v3_V3_jpp_9b.log 2>&1
echo "EVAL EXIT $?"
