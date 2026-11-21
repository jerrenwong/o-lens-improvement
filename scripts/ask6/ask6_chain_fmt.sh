#!/bin/bash
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/ask6
cd $R
$PY scripts/ask6/ask6_formats.py Qwen/Qwen3.5-9B > $L/formats9.log 2>&1 || echo F9 >> $L/fmt_err
HF_HOME=$R/.hf27 $PY scripts/ask6/ask6_formats.py Qwen/Qwen3.6-27B > $L/formats27.log 2>&1 || echo F27 >> $L/fmt_err
echo OK > $L/fmt_done
