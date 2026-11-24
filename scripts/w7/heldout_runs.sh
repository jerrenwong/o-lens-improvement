#!/bin/bash
# W7 / Experiment 3 HELD-OUT (ORCHESTRATOR ONLY; not run by the w7 agent). Five columns, K=10, T=1:
#   C1 ridge (no ALENS_E override), C2 all, C3 nomerge, C4 mergeall, C5 merge  (+ ref ce = w2 CE maps).
# Rows: none (A-lens alone) and JPP (J++ + A-lens); reference JPP0 (J++ alone). Filters f4 and f1. test + v4.
# Usage: bash scripts/w7/heldout_runs.sh 2b 9b      [env COLS="ridge all nomerge mergeall merge ce" LVS="f4 f1"]
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/w7/logs; mkdir -p $LOG
cd $R; export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COLS=${COLS:-"ridge all nomerge mergeall merge ce"}; LVS=${LVS:-"f4 f1"}
for T in "$@"; do
  [ $T = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
  for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
    for lv in $LVS; do
      n=w7ho_${T}_JPP0_${lv}_${kind}
      [ -f auto/runs/$n/summary.json ] || env ALENS_BASE=JPP0 $PY $EV --lens auto/candidates/w1_alens.py --model $M --split test --orchestrator --lite --filter $lv --name $n > $LOG/$n.log 2>&1
      for c in $COLS; do
        case $c in
          ridge) MAPS="" ;;
          ce)    MAPS="ALENS_E=$R/.hf27/w2/E_${T}_topic_ce_k10.pt ALENS_E2=$R/.hf27/w2/E_${T}_spec_ce_k10.pt" ;;
          *)     MAPS="ALENS_E=$R/.hf27/w7/E_${T}_topic_${c}_k10.pt ALENS_E2=$R/.hf27/w7/E_${T}_spec_${c}_k10.pt" ;;
        esac
        for row in none JPP; do
          n=w7ho_${T}_${c}_${row}_${lv}_${kind}
          [ -f auto/runs/$n/summary.json ] && continue
          env ALENS_BASE=$row $MAPS $PY $EV --lens auto/candidates/w1_alens.py --model $M --split test --orchestrator --lite --filter $lv --name $n > $LOG/$n.log 2>&1 || echo "FAILED $n" >> $LOG/errors_ho
        done
      done
    done
  done
done
