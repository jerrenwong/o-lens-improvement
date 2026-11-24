#!/bin/bash
# w2 HELD-OUT (ORCHESTRATOR ONLY, PREREG_v5 Amendment 3): cross-entropy A-lens, K=10, 2B and 9B, test + v4, filters f1 and f2.
# Rows: none (A-lens-CE alone), JPP (J++ + A-lens-CE); controls with J++: W2_VAR=shuf, const. Reference rows = w1_<tag>_{none,JPP,JPP0}_<f>_<kind>.
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/w2/logs; mkdir -p $LOG
cd $R; export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True W2_K=10
for T in "$@"; do
  [ $T = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
  for try in 1 2 3 4; do
    fail=0
    for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
      for lv in f1 f2; do for spec in ce:none ce:JPP shuf:JPP const:JPP; do
        var=${spec%%:*}; row=${spec##*:}; n=w2ho_${T}_${var}_${row}_${lv}_${kind}
        [ -f auto/runs/$n/summary.json ] && continue
        while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 14000 ]; do sleep 30; done
        env W2_VAR=$var ALENS_BASE=$row $PY $EV --lens auto/candidates/w2_alens_ce.py --model $M --split test --orchestrator --lite --filter $lv --name $n > $LOG/$n.log 2>&1 || fail=1
      done; done
    done
    [ $fail = 0 ] && break; sleep 60
  done
  echo "w2 heldout finished $T fail=$fail $(date -u +%H:%M)"
done
