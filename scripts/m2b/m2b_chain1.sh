#!/bin/bash
# m2b step 1+2: collect 2B chat-format ask targets, then fit maps.
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/m2b
mkdir -p $L
cd $R
export BS=${BS:-64}
$PY scripts/m2b/m2b_collect.py Qwen/Qwen3.5-2B chat > $L/collect.log 2>&1 || { echo COLLECT_FAILED > $L/FAILED; exit 1; }
$PY scripts/m2b/m2b_fit.py Qwen/Qwen3.5-2B > $L/fit.log 2>&1 || { echo FIT_FAILED > $L/FAILED; exit 1; }
echo OK > $L/chain1_done
