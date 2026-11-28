#!/bin/bash
# after ej1_2b_ie / ej1_2b_none: namely and thatis fits in parallel (pod)
J=/workspace/.fleet/jobs/wong-olens6
until [ -f $J/ej1_2b_ie.exit ] && [ -f $J/ej1_2b_none.exit ]; do sleep 30; done
cd /workspace/o-lens-improvement
for s in namely thatis; do bash scripts/q1/job.sh ej1_2b_$s "\$PY scripts/ej1/ej1_fit.py --suffix $s --n 128 --out /workspace/o-lens-improvement/.hf27/ej1/2b_$s"; done
