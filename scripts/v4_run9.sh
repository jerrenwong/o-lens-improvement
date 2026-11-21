#!/bin/bash
# v4 item check + bench cache 9B (model correctness, residuals, split). Waits for >= 25 GB free on the GPU.
# No lens is run here (v3_run9.sh also ran the J++ sanity eval; for v4 scoring is left to the orchestrator).
cd /workspace/o-lens-improvement
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge 25000 ]; then break; fi
  echo "$(date +%T) waiting: free=${free} MiB"; sleep 60
done
echo "$(date +%T) start"
/root/jpp_lens/.venv/bin/python scripts/v4_check_items.py data/evals_v4 > /root/v4/chk_final9.log 2>&1
/root/jpp_lens/.venv/bin/python scripts/v4_bench_cache.py --model Qwen/Qwen3.5-9B --tag 9b > /root/v4/bench9b.log 2>&1
rc=$?; echo "BENCH EXIT $rc"; exit $rc
