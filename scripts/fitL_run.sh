#!/bin/bash
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/fitL
free() { timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until [ "$(free)" -gt 9000 ]; do sleep 20; done
$PY auto/evaluate.py --lens auto/candidates/fitL_dump.py --name fitL_9b --lite > /root/fitL/9b.log 2>&1; echo "9b rc=$? $(grep -E 'fitL\]|chosen lam|dev_pass10:|dev_pass1:' /root/fitL/9b.log | tr '\n' ' ' | cut -c1-220)"
$PY auto/evaluate.py --model Qwen/Qwen3.5-2B --lens auto/candidates/fitL_dump.py --name fitL_2b --lite > /root/fitL/2b.log 2>&1; echo "2b rc=$? $(grep -E 'fitL\]|chosen lam|dev_pass10:|dev_pass1:' /root/fitL/2b.log | tr '\n' ' ' | cut -c1-220)"
