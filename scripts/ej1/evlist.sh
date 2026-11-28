#!/bin/bash
# bash scripts/ej1/evlist.sh <tag> <jobfile>: lines "<run name> VAR=value ..."; skips runs with a summary.json
cd /workspace/o-lens-improvement
while read -r n envs; do
  [ -z "$n" ] && continue; [ -f auto/runs/$n/summary.json ] && continue
  bash scripts/ej1/dev.sh $1 $n $envs
done < $2
