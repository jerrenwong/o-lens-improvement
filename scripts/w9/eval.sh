#!/bin/bash
# w9 dev evals: bash scripts/w9/eval.sh <spec-file> [parallel]; spec lines "name|env|model|filter" (DEV split only)
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
L=$R/runs/w9/logs; mkdir -p $L
run1() { IFS='|' read -r n envs model flt <<< "$1"
  [ -f auto/runs/$n/summary.json ] && exit 0
  env $envs $PY auto/evaluate.py --model $model --lens auto/candidates/w9_alens.py --lite --filter $flt --name $n > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/errors; }
export -f run1; export PY L R
grep -v '^#' $1 | xargs -P ${2:-8} -I{} bash -c 'run1 "$@"' _ {}
