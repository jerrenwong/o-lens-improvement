#!/bin/bash
# perf1 Part A, 9B, dev split only: lite runs first (small), then the full-model runs and the bitwise check.
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
R=scripts/perf1/run_eval.sh
export PYTHONUNBUFFERED=1
bash $R a9_jpp_lite 8 $PY auto/evaluate.py --lens auto/candidates/jpp.py --name perf1_jpp_lite --lite
bash $R a9_e1A_lite 8 $PY auto/evaluate.py --lens auto/candidates/e1_A_fitlam.py --name perf1_e1_A_fitlam_lite --lite
bash $R a9_jpp_full 26 $PY auto/evaluate.py --lens auto/candidates/jpp.py --name perf1_jpp
bash $R a9_e1A_full 26 $PY auto/evaluate.py --lens auto/candidates/e1_A_fitlam.py --name perf1_e1_A_fitlam
bash $R a9_check 28 $PY scripts/perf1/check_lite.py --model Qwen/Qwen3.5-9B --out scripts/perf1/results/check_lite_9b.json
