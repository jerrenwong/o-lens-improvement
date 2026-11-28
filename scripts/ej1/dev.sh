#!/bin/bash
# ej1 dev scoring (pod): bash scripts/ej1/dev.sh <model tag 2b|9b> <run name> [VAR=value ...]  -> auto/runs/<run name>/
# one line per run appended to runs/ej1/dev_<tag>.txt
cd /workspace/o-lens-improvement
tag=$1; name=$2; shift 2
case $tag in 9b) model=Qwen/Qwen3.5-9B ;; 2b) model=Qwen/Qwen3.5-2B ;; esac
mkdir -p runs/ej1
out=$(env "$@" $PY auto/evaluate.py --lens auto/candidates/ej1_lens.py --model $model --lite --filter f4 --name $name 2>&1)
echo "$out" | grep -E "chosen|alone FIT" | tail -3
line=$(echo "$out" | grep -E "dev_pass10:|dev_pass1:|per_task" | tr -s ' ' | tr '\n' ' ')
echo "$name $* | $line" | tee -a runs/ej1/dev_$tag.txt
