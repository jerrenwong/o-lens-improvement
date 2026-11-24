#!/bin/bash
# w3 dev evals (pod), sequential. Job file lines: <run name> <2b|9b> <candidate file> [VAR=value ...]
# Skips lines whose auto/runs/<name>/summary.json exists. Dev split only (evaluate.py default), --lite.
#   bash scripts/w3/ev.sh JOBFILE
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LG=$R/runs/w3/evlogs; mkdir -p $LG
while read -r n tag cand envs; do
  [ -z "$n" ] && continue; case $n in \#*) continue;; esac
  [ -f auto/runs/$n/summary.json ] && continue
  case $tag in 2b) M=Qwen/Qwen3.5-2B;; 9b) M=Qwen/Qwen3.5-9B;; esac
  echo -e "$n\t$tag\t$cand\t$envs" >> runs/w3/registry.tsv
  env $envs bash scripts/ja1/retry.sh 6000 $LG/$n.log $PY auto/evaluate.py --model $M --lens auto/candidates/$cand \
    --name $n --lite || echo "$n failed" >> $LG/errors
done < $1
touch $1.done
