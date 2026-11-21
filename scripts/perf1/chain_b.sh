#!/bin/bash
# perf1 Part B (9B, measure only). Waits for the Part A chain; each step waits for 26 GB free.
cd /workspace/o-lens-improvement
# (Part A chain done)
PY=/root/jpp_lens/.venv/bin/python
PYF=/workspace/o-lens-improvement/.venv_fast/bin/python
R=scripts/perf1/run_eval.sh
export PYTHONUNBUFFERED=1
bash $R b_fla 26 $PY scripts/perf1/bench_b.py --mode fla --tasks fwd,fewshot,gen,cheap
bash $R b_torch 26 $PY scripts/perf1/bench_b.py --mode torch --tasks fwd,fewshot,gen
if $PYF -c "import causal_conv1d" 2>/dev/null; then
  bash $R b_fla_cc 26 $PYF scripts/perf1/bench_b.py --mode fla_cc --tasks fwd,fewshot,gen
fi
bash $R b_analyze 6 $PY scripts/perf1/analyze_b.py
bash $R b_torch_cheap 26 $PY scripts/perf1/bench_b.py --mode torch --tasks cheap --tag _cheap
bash $R b_fla_compile 26 $PY scripts/perf1/bench_b.py --mode fla --tasks compile --tag _compile
