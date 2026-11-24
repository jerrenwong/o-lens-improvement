#!/bin/bash
# w3 HELD-OUT (ORCHESTRATOR ONLY, PREREG_v5 Amendment 4): Jacobian ("J++-style") A-lens on Qwen3.5-9B, test + v4, filters f1 and f2.
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/w3/evlogs; cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for try in 1 2 3; do fail=0
 for kind in test v4; do [ $kind = test ] && EV=auto/evaluate.py || EV=auto/evaluate_v4.py
  for lv in f1 f2; do for v in A5 K1; do for mode in alone add; do
    n=w3ho_9b_ja${v}_${mode}_${lv}_${kind}; [ -f auto/runs/$n/summary.json ] && continue
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 14000 ]; do sleep 30; done
    env W3_JT=runs/ao1/9b/lens/top${v}_n128.pt W3_JU=runs/w3/9b/lens/uns${v}_n128.pt $PY $EV --lens auto/candidates/w3_${mode}.py \
      --model Qwen/Qwen3.5-9B --split test --orchestrator --lite --filter $lv --name $n > $LOG/$n.log 2>&1 || fail=1
  done; done; done
 done
 [ $fail = 0 ] && break; sleep 60
done; echo "w3 heldout finished fail=$fail $(date -u +%H:%M)"
