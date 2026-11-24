#!/bin/bash
# w9 HELD-OUT (ORCHESTRATOR ONLY -- not run by the w9 agent). Single configuration: linear cross-entropy A-lens maps
# fitted on the 98k passage set (E_<tag>_{topic,spec}_lin_ce_98k.pt). Rows: none (A-lens alone), JPP (J++ + A-lens,
# lam chosen on the fit split by the frozen alens rule at the filter level). 2B and 9B, filter f4, test + v4.
# To run the MLP alternative instead set CFG=mlp512_ce_98k.  Usage: bash scripts/w9/heldout.sh 2b 9b
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/w9/logs; mkdir -p $LOG
cd $R; export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=${CFG:-lin_ce_98k}
for T in "$@"; do
  [ $T = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
  E=$R/.hf27/w9/maps/E_${T}_topic_${CFG}.pt; E2=$R/.hf27/w9/maps/E_${T}_spec_${CFG}.pt
  for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
    for row in none JPP; do
      n=w9ho_${T}_${CFG}_${row}_f4_${kind}
      [ -f auto/runs/$n/summary.json ] && continue
      env ALENS_BASE=$row ALENS_E=$E ALENS_E2=$E2 $PY $EV --lens auto/candidates/w9_alens.py --model $M \
        --split test --orchestrator --lite --filter f4 --name $n > $LOG/$n.log 2>&1 || echo "FAILED $n"
    done
  done
done
