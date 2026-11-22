#!/bin/bash
# ex2 GPU chain (one GPU job of ours at a time; each step retried on OOM)
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
bash scripts/ex2/retry.sh /root/ex2/s2.log $PY scripts/ex2/s2_func.py || exit 1
bash scripts/ex2/retry.sh /root/ex2/fit2b.log $PY scripts/ex2/s45_fit.py --model 2b || exit 1
bash scripts/ex2/retry.sh /root/ex2/fit9b.log $PY scripts/ex2/s45_fit.py --model 9b || exit 1
echo chain1 done > /root/ex2/chain1_done
