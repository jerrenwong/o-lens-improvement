#!/bin/bash
# perf1 Part A, 27B, dev split only: lite run (compared with the stored full-model run auto/runs/jpp_27b).
cd /workspace/o-lens-improvement
while tmux has-session -t perf1prof 2>/dev/null; do sleep 20; done
export HF_HOME=/workspace/o-lens-improvement/.hf27 PYTHONUNBUFFERED=1
bash scripts/perf1/run_eval.sh a27_jpp_lite 10 /root/jpp_lens/.venv/bin/python auto/evaluate27.py --model Qwen/Qwen3.6-27B --lens auto/candidates/jpp.py --name perf1_jpp_27b_lite --lite
