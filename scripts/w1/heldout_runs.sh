#!/bin/bash
# w1 HELD-OUT (ORCHESTRATOR ONLY): five A-lens-family rows x three filter levels, test split + v4, --lite.
# Usage (pod): bash scripts/w1/heldout_runs.sh <tag> [kinds] [levels]   tag in 2b 9b 27b olmo7b gemma4e4b; kinds "test v4"
# 27b / olmo7b / gemma4e4b need /root/w1_hf on the pod (python scripts/w1/hf_links.py <repos>; pod-local, lost on restart).
set -uo pipefail
T=$1; KINDS=${2:-"test v4"}; LEVELS=${3:-"f1 f2 f3"}
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/w1/filt_logs; mkdir -p $LOG
cd $R; export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVX=""
case $T in
  9b) M=Qwen/Qwen3.5-9B; EVT=auto/evaluate.py; EV4=auto/evaluate_v4.py;;
  2b) M=Qwen/Qwen3.5-2B; EVT=auto/evaluate.py; EV4=auto/evaluate_v4.py;;
  27b) M=Qwen/Qwen3.6-27B; EVT=auto/evaluate27.py; EV4=auto/evaluate_v4.py; export HF_HOME=/root/w1_hf;;
  olmo7b|gemma4e4b) [ $T = olmo7b ] && M=allenai/Olmo-3-7B-Instruct || M=google/gemma-4-E4B-it
     EVT=auto/evaluate_x.py; EV4=auto/evaluate_x_v4.py; export HF_HOME=/root/w1_hf
     ENVX="ALENS_E=$R/.hf27/xm1/E_${T}_t3.pt ALENS_E2=$R/.hf27/xm1/E_${T}_s5.pt";;
esac
for kind in $KINDS; do for lv in $LEVELS; do for row in J0 JPP0 none J JPP; do
  n=w1_${T}_${row}_${lv}_${kind}; [ $kind = test ] && EV=$EVT || EV=$EV4
  [ -f auto/runs/$n/summary.json ] && continue
  env ALENS_BASE=$row $ENVX $PY $EV --lens auto/candidates/w1_alens.py --model $M --split test --orchestrator --lite --filter $lv --name $n \
    > $LOG/$n.log 2>&1 || echo "FAILED $n" >> $LOG/failed_heldout_$T
done; done; done
touch $LOG/done_heldout_$T
