#!/bin/bash
# q2 HELD-OUT (ORCHESTRATOR ONLY -- not run by the q2 agent). Single configuration from T4b: the A-lens built from the
# unstated-word question only ({U}: E_<tag>_spec_lin_ce_98k.pt as both maps, i.e. a(h) = unit(map_U(h))), no topic map.
# Rows: none (A-lens alone), JPP (J++ + A-lens, lam on the fit split by the frozen alens rule at f4). test + v4 sets.
# Same pattern as scripts/w9/heldout.sh.  Usage (on the GPU pod): bash scripts/q2/heldout.sh 9b 2b
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/q2/logs; mkdir -p $LOG
cd $R; export HF_HOME=/root/hf_q1 HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for T in "$@"; do
  [ $T = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
  E=$R/.hf27/w9/maps/E_${T}_spec_lin_ce_98k.pt
  for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
    for row in none JPP; do
      n=q2ho_${T}_U_lin_ce_98k_${row}_f4_${kind}
      [ -f auto/runs/$n/summary.json ] && continue
      env ALENS_BASE=$row ALENS_E=$E ALENS_E2=$E $PY $EV --lens auto/candidates/w9_alens.py --model $M \
        --split test --orchestrator --lite --filter f4 --name $n > $LOG/$n.log 2>&1 || echo "FAILED $n"
    done
  done
done
