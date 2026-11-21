#!/bin/bash
# perf1 Part B follow-up: generation + cheap tasks for the current (fla) mode (the first fla run hit the 0.3 cap in gen).
cd /workspace/o-lens-improvement
while tmux has-session -t perf1b2 2>/dev/null; do sleep 30; done
PY=/root/jpp_lens/.venv/bin/python
R=scripts/perf1/run_eval.sh
export PYTHONUNBUFFERED=1
bash $R b_fla_gen 26 $PY scripts/perf1/bench_b.py --mode fla --tasks gen,cheap --tag _gen
bash $R b_analyze2 6 $PY scripts/perf1/analyze_b.py
