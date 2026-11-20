#!/bin/bash
# Wait for a free GPU (<15 GB used) and the 27B lock, then: v3 bench cache 27B + J++ sanity run on v3 27B.
cd /workspace/o-lens-improvement
export HF_HOME=/workspace/o-lens-improvement/.hf27
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -lt 15000 ] && ( set -o noclobber; echo v3 > /root/gpu27.lock ) 2>/dev/null; then break; fi
  echo "$(date +%T) waiting: used=${used} MiB lock=$(cat /root/gpu27.lock 2>/dev/null)"; sleep 60
done
trap 'rm -f /root/gpu27.lock' EXIT
echo "$(date +%T) lock taken"
/root/jpp_lens/.venv/bin/python scripts/v3_bench_cache.py --model Qwen/Qwen3.6-27B --tag 27b > /root/v3_bench27b.log 2>&1
echo "BENCH EXIT $?"
/root/jpp_lens/.venv/bin/python auto/evaluate_v3.py --lens auto/candidates/jpp.py --name V3_jpp_27b --model Qwen/Qwen3.6-27B --split test --orchestrator > /root/v3_V3_jpp_27b.log 2>&1
echo "EVAL EXIT $?"
