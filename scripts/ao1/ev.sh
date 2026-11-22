#!/bin/bash
# ao1 dev eval of one ask-only lens (pod). Lines of a job file: <name> <2b|9b> <mode> <J path|-> <E path|-> [E2 path]
#   bash ev.sh JOBFILE        (runs every line whose auto/runs/<name>/summary.json is missing; sequential)
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p /root/ao1/ev
while read -r n tag mode J E E2; do
  [ -z "$n" ] && continue; case $n in \#*) continue;; esac
  [ -f auto/runs/$n/summary.json ] && continue
  case $tag in 2b) M=Qwen/Qwen3.5-2B;; 9b) M=Qwen/Qwen3.5-9B;; esac
  echo -e "$n\t$tag\t$mode\t$J\t$E\t${E2:--}" >> runs/ao1/registry.tsv
  AO1_J=$J AO1_E=$E AO1_E2=${E2:-} bash scripts/ja1/retry.sh 9000 /root/ao1/ev/$n.log \
    $PY auto/evaluate.py --model $M --lens auto/candidates/ao1_$mode.py --name $n --lite || echo "$n failed" >> /root/ao1/ev_errors
done < $1
touch $1.done
