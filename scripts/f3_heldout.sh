#!/bin/bash
# ORCHESTRATOR ONLY, PREREG_v5 Amendment 5: cross-entropy A-lens (2B, 9B) and Jacobian A-lens (9B) under filter f3, test + v4.
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/f3_logs; mkdir -p $LOG; cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { n=$1; shift; [ -f auto/runs/$n/summary.json ] && return 0; "$@" > $LOG/$n.log 2>&1 || { echo "FAILED $n"; return 1; }; }
fail=0
for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
  for T in 9b 2b; do [ $T = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
    for spec in ce:none ce:JPP shuf:JPP const:JPP; do var=${spec%%:*}; row=${spec##*:}
      run w2ho_${T}_${var}_${row}_f3_${kind} env ALENS_BASE=$row ALENS_E=$R/.hf27/w2/E_${T}_topic_${var}_k10.pt ALENS_E2=$R/.hf27/w2/E_${T}_spec_${var}_k10.pt \
        $PY $EV --lens auto/candidates/w1_alens.py --model $M --split test --orchestrator --lite --filter f3 --name w2ho_${T}_${var}_${row}_f3_${kind} || fail=1
    done
  done
  for v in A5 K1; do for mode in alone add; do
    run w3ho_9b_ja${v}_${mode}_f3_${kind} env W3_MODE=$mode W3_JT=runs/ao1/9b/lens/top${v}_n128.pt W3_JU=runs/w3/9b/lens/uns${v}_n128.pt \
      $PY $EV --lens auto/candidates/w3_f.py --model Qwen/Qwen3.5-9B --split test --orchestrator --lite --filter f3 --name w3ho_9b_ja${v}_${mode}_f3_${kind} || fail=1
  done; done
done
echo "f3 heldout finished fail=$fail $(date -u +%H:%M)"
