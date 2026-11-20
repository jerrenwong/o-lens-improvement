#!/bin/bash
# r2: score r2 lenses on the dev split (auto/evaluate.py, default model 9B, default split dev).
# Usage: bash scripts/r2_eval.sh C/C_64a A/A_64a ...   -> auto/runs/r2_<name>/, log /root/r2_eval_<name>.log
cd /workspace/o-lens-improvement
for p in "$@"; do
  n=$(basename $p)
  R2_LENS=runs/r2_9b/$p/jpp_$n.pt /root/jpp_lens/.venv/bin/python auto/evaluate.py \
     --lens auto/candidates/r2_lensfile.py --name r2_$n > /root/r2_eval_$n.log 2>&1
  grep -A9 '^---' /root/r2_eval_$n.log | sed "s/^/$n  /"
done
