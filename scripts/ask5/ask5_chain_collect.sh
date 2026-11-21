#!/bin/bash
# ask5: collect 9B then 27B specific-word targets (chat template).
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask5
W="$PY $R/scripts/ask3/ask3_resv.py"
cd $R
NEED_GB=22 $W scripts/ask5/ask5_collect.py Qwen/Qwen3.5-9B chat > $L/collect9.log 2>&1 || { echo COLLECT9 > $L/FAILED; exit 1; }
HF_HOME=$R/.hf27 NEED_GB=56 $W scripts/ask5/ask5_collect.py Qwen/Qwen3.6-27B chat > $L/collect27.log 2>&1 || { echo COLLECT27 > $L/FAILED; exit 1; }
echo OK > $L/collect_done
