#!/bin/bash
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/fitgate
free() { timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
until [ "$(free)" -gt 9000 ]; do sleep 20; done
$PY auto/evaluate.py --lens auto/candidates/fitgate_dump.py --name fitgate_9b --lite > /root/fitgate/9b.log 2>&1; echo "9b rc=$? $(grep -E 'fitgate\]|dev_pass10:|dev_pass1:' /root/fitgate/9b.log | tr '\n' ' ' | cut -c1-220)"
until [ "$(free)" -gt 5000 ]; do sleep 20; done
$PY auto/evaluate.py --model Qwen/Qwen3.5-2B --lens auto/candidates/fitgate_dump.py --name fitgate_2b --lite > /root/fitgate/2b.log 2>&1; echo "2b rc=$? $(grep -E 'fitgate\]|dev_pass10:|dev_pass1:' /root/fitgate/2b.log | tr '\n' ' ' | cut -c1-220)"
