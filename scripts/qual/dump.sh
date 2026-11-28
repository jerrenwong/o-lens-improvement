#!/bin/bash
# Qual dump of one lens on one model (GPU pod; full model, not --lite, because probes need forward passes).
#   bash scripts/qual/dump.sh <9b|2b> <candidate.py> "<lens name>" [VAR=value ...]
#   e.g. bash scripts/qual/dump.sh 9b auto/candidates/alens.py "J++ + A-lens" ALENS_BASE=JPP
#        fleet run <pod> qual_9b_jppa --cd /workspace/o-lens-improvement -- \
#          "bash scripts/qual/dump.sh 9b auto/candidates/alens.py 'J++ + A-lens' ALENS_BASE=JPP"
# VAR=value pairs are the candidate's own env vars. Env overrides: QUAL_PROBES (default probes/probes_v1.json),
# QUAL_OUTDIR (output dir, default runs/qual/<tag>), QUAL_SETS (default bench-fit,bench-dev,probes), QUAL_TOPK (20), FILTER (evaluator --filter, default f4), PY.
# Output: runs/qual/<tag>/<slug>.json; the evaluator also scores the dev split as usual (auto/runs/qual_<tag>_<slug>/).
# Then: python3 scripts/qual/compare.py runs/qual/<tag>/<old>.json runs/qual/<tag>/<new>.json --md
set -euo pipefail
cd /workspace/o-lens-improvement
[ $# -ge 3 ] || { sed -n 2,10p "$0"; exit 2; }
tag=$1; lens=$2; name=$3; shift 3
case $tag in 9b) model=Qwen/Qwen3.5-9B ;; 2b) model=Qwen/Qwen3.5-2B ;; *) echo "unknown model tag $tag"; exit 2 ;; esac
PY=${PY:-/root/jpp_lens/.venv/bin/python}
probes=${QUAL_PROBES:-probes/probes_v1.json}
[ -f "$probes" ] || { echo "no probe file $probes (set QUAL_PROBES, or QUAL_SETS=bench-fit,bench-dev)"; [ "${QUAL_SETS:-}" != "" ] || exit 2; }
slug=$(python3 -c 'import sys,re;s=sys.argv[1].replace("++","pp").replace("+","_").lower();print(re.sub(r"[^a-z0-9]+","_",s).strip("_"))' "$name")
echo "qual dump: $model | $lens | $name -> runs/qual/$tag/$slug.json"
env "$@" QUAL_LENS="$lens" QUAL_NAME="$name" QUAL_PROBES="$probes" \
  "$PY" auto/evaluate.py --lens auto/candidates/qual_dump.py --model "$model" --filter "${FILTER:-f4}" \
  --name "qual_${tag}_${slug}"
ls -la "${QUAL_OUTDIR:-runs/qual/$tag}/$slug.json"
