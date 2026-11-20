#!/bin/bash
# v3 9B: bench cache + J++ sanity run, each retried (up to 6x) when it fails (e.g. OOM from a shared GPU).
cd /workspace/o-lens-improvement
waitfree() { while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$f" -ge 25000 ] && break; echo "$(date +%T) waiting: free=${f} MiB"; sleep 60; done; }
for i in 1 2 3 4 5 6; do waitfree; echo "$(date +%T) bench try $i"
  /root/jpp_lens/.venv/bin/python scripts/v3_bench_cache.py --model Qwen/Qwen3.5-9B --tag 9b > /root/v3_bench9b.log 2>&1 && break; echo "BENCH FAIL $i"; sleep 120; done
for i in 1 2 3 4 5 6; do waitfree; echo "$(date +%T) eval try $i"
  /root/jpp_lens/.venv/bin/python auto/evaluate_v3.py --lens auto/candidates/jpp.py --name V3_jpp_9b --model Qwen/Qwen3.5-9B --split test --orchestrator > /root/v3_V3_jpp_9b.log 2>&1 && break; echo "EVAL FAIL $i"; sleep 120; done
echo "EVAL EXIT done"
