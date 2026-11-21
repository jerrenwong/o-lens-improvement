#!/bin/bash
# PREREG_v4 addendum A: frozen lens L on the ORIGINAL benchmark test split. Orchestrator only.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/c6
[ -f auto/runs/T_L_9b/summary.json ] || $PY auto/evaluate.py --lens auto/candidates/fm27_F4.py --name T_L_9b --split test --orchestrator --lite > /root/c6/T_L_9b.log 2>&1
echo "$(date -u +%T) T_L_9b $(grep -E 'pass10:|pass1:|n_items' /root/c6/T_L_9b.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-200)"
[ -f auto/runs/T_L_27b/summary.json ] || HF_HOME=/workspace/o-lens-improvement/.hf27 $PY auto/evaluate27.py --model Qwen/Qwen3.6-27B --lens auto/candidates/fm27_F4.py --name T_L_27b --split test --orchestrator --lite > /root/c6/T_L_27b.log 2>&1
echo "$(date -u +%T) T_L_27b $(grep -E 'pass10:|pass1:|n_items' /root/c6/T_L_27b.log | sed 's/  */ /g' | tr '\n' ' ' | cut -c1-200)"
touch /root/c6/DONE
