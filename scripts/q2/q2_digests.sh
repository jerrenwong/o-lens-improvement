#!/bin/bash
# q2: compare.py digests of each variant against J++ and against J++ + A-lens CE 98k (probes not-next / next, bench-dev)
#   bash scripts/q2/q2_digests.sh <tag> <slug> [<slug> ...]   -> runs/q2/digests/<tag>/<slug>__vs_<ref>__<set>.md
cd /workspace/o-lens-improvement
tag=$1; shift
mkdir -p runs/q2/digests/$tag
for s in "$@"; do
  [ -f runs/qual/$tag/$s.json ] || { echo "missing $s"; continue; }
  refs="jpp jpp_a_lens_ce_98k"; [[ $s == a_lens* ]] && refs="$refs a_lens_ce_98k"
  for ref in $refs; do
    o=runs/q2/digests/$tag/${s}__vs_${ref}
    python3 scripts/qual/compare.py runs/qual/$tag/$ref.json runs/qual/$tag/$s.json --set probes --next no --md > ${o}__probes_notnext.md
    python3 scripts/qual/compare.py runs/qual/$tag/$ref.json runs/qual/$tag/$s.json --set probes --next yes --md > ${o}__probes_next.md
    python3 scripts/qual/compare.py runs/qual/$tag/$ref.json runs/qual/$tag/$s.json --set bench-dev --md > ${o}__benchdev.md
  done
done
