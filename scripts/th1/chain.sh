#!/bin/bash
cd /root/th1
PY=/root/jpp_lens/.venv/bin/python
while ! grep -q ALL_DONE gen.log; do sleep 30; done
$PY collect.py > collect.log 2>&1 && $PY analyze.py > analyze.log 2>&1 && $PY causal.py > causal.log 2>&1
echo CHAIN_DONE >> causal.log
