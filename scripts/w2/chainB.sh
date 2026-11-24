#!/bin/bash
# W2b chain: co-prediction statistics (2B, 9B), competitor-subtracted dev/fit scoring of the four frozen A-lens-family
# rows (J0 = J-lens, JPP0 = J++, none = A-lens alone, JPP = J++ + A-lens), names/numbers check on 9B.
# Usage: bash scripts/w2/chainB.sh [steps]   steps: copred eval numnames
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$R/runs/w2/logs; mkdir -p $L
STEPS=${1:-"copred eval numnames"}
declare -A MOD=([2b]=Qwen/Qwen3.5-2B [9b]=Qwen/Qwen3.5-9B)
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt ${1:-16000} ]; do sleep 30; done; }
for s in $STEPS; do case $s in
copred)
  for t in 2b 9b; do
    [ -f .hf27/w2/copred_$t.json ] && continue
    waitmem 30000
    $PY scripts/w2/copred.py ${MOD[$t]} --bs ${COPRED_BS:-32} > $L/copred_$t.log 2>&1 || echo "FAILED copred $t" >> $L/errors
  done ;;
eval)
  for t in 2b 9b; do for row in J0 JPP0 none JPP; do
    n=w2b_${row}_$t
    [ -f runs/w2/copred_eval/$n.json ] && continue
    waitmem 16000
    ALENS_BASE=$row HF_HUB_OFFLINE=1 $PY scripts/w2/eval_copred.py --model ${MOD[$t]} --lens auto/candidates/alens.py \
      --name $n --ref D_al_${row}_$t --centred 0,1 --qual 10 --qual_variant k20 > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/errors
  done; done ;;
numnames)
  waitmem 30000
  HF_HUB_OFFLINE=1 $PY scripts/w2/numnames.py Qwen/Qwen3.5-9B runs/w2/copred_eval/w2b_J0_9b.json \
    runs/w2/copred_eval/w2b_JPP0_9b.json > $L/numnames_9b.log 2>&1 || echo "FAILED numnames" >> $L/errors ;;
esac; done
echo done > $L/chainB_done_$(echo $STEPS | tr ' ' _)
