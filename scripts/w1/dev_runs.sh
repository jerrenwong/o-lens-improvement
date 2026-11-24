#!/bin/bash
# w1 DEV runs (pod): five A-lens-family rows x three filter levels, --lite, dev split only.
# Usage: bash scripts/w1/dev_runs.sh <tag> [levels]     tag in 2b 9b 27b olmo7b gemma4e4b
set -uo pipefail
T=$1; LEVELS=${2:-"f1 f2 f3"}
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOG=$R/runs/w1/filt_logs; mkdir -p $LOG
cd $R; export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVX=""
case $T in
  9b) M=Qwen/Qwen3.5-9B; EV=auto/evaluate.py;;
  2b) M=Qwen/Qwen3.5-2B; EV=auto/evaluate.py;;
  27b) M=Qwen/Qwen3.6-27B; EV=auto/evaluate27.py; export HF_HOME=/root/w1_hf; NEED=20000;;
  olmo7b) M=allenai/Olmo-3-7B-Instruct; EV=auto/evaluate_x.py; export HF_HOME=/root/w1_hf; ENVX="ALENS_E=$R/.hf27/xm1/E_${T}_t3.pt ALENS_E2=$R/.hf27/xm1/E_${T}_s5.pt";;
  gemma4e4b) M=google/gemma-4-E4B-it; EV=auto/evaluate_x.py; export HF_HOME=/root/w1_hf; ENVX="ALENS_E=$R/.hf27/xm1/E_${T}_t3.pt ALENS_E2=$R/.hf27/xm1/E_${T}_s5.pt";;
esac
for lv in $LEVELS; do for row in J0 JPP0 none J JPP; do
  n=w1_${T}_${row}_${lv}
  [ -f auto/runs/$n/summary.json ] && continue
  ok=0
  for try in 1 2 3 4; do  # shared GPU: wait for free memory, retry (OOM from other agents' jobs)
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt ${NEED:-12000} ]; do sleep 30; done
    env ALENS_BASE=$row $ENVX $PY $EV --lens auto/candidates/w1_alens.py --model $M --split dev --lite --filter $lv --name $n > $LOG/$n.log 2>&1 && { ok=1; break; }
    sleep 60
  done
  [ $ok = 1 ] || echo "FAILED $n" >> $LOG/failed_$T
done; done
touch $LOG/done_$T
