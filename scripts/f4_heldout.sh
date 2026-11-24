#!/bin/bash
# ORCHESTRATOR ONLY, PREREG_v5 Amendment 6: every reported row under the full filter f4, test + v4, once each, --lite.
# Usage: bash scripts/f4_heldout.sh w1 <tags...> | ce | jac
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/f4_logs; mkdir -p $LOG; cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { n=$1; shift; [ -f auto/runs/$n/summary.json ] && return 0
  for try in 1 2 3; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 18000 ]; do sleep 30; done
    "$@" > $LOG/$n.log 2>&1 && return 0; sleep 30
  done; echo "FAILED $n"; return 1; }
what=$1; shift; fail=0
if [ $what = w1 ]; then for T in "$@"; do bash scripts/w1/heldout_runs.sh $T "test v4" "f4" || fail=1; echo "w1 f4 $T done $(date -u +%H:%M) failed=$(cat runs/w1/filt_logs/failed_heldout_$T 2>/dev/null | wc -l)"; done; fi
for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
  if [ $what = ce ]; then for T in 9b 2b; do [ $T = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
    for spec in ce:none ce:JPP shuf:JPP const:JPP; do var=${spec%%:*}; row=${spec##*:}; n=w2ho_${T}_${var}_${row}_f4_${kind}
      run $n env ALENS_BASE=$row ALENS_E=$R/.hf27/w2/E_${T}_topic_${var}_k10.pt ALENS_E2=$R/.hf27/w2/E_${T}_spec_${var}_k10.pt \
        $PY $EV --lens auto/candidates/w1_alens.py --model $M --split test --orchestrator --lite --filter f4 --name $n || fail=1
    done; done; fi
  if [ $what = jac ]; then for v in A5 K1; do for mode in alone add; do n=w3ho_9b_ja${v}_${mode}_f4_${kind}
    run $n env W3_MODE=$mode W3_JT=runs/ao1/9b/lens/top${v}_n128.pt W3_JU=runs/w3/9b/lens/uns${v}_n128.pt \
      $PY $EV --lens auto/candidates/w3_f.py --model Qwen/Qwen3.5-9B --split test --orchestrator --lite --filter f4 --name $n || fail=1
  done; done; fi
done
echo "f4 $what finished fail=$fail $(date -u +%H:%M)"
