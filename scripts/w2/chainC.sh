#!/bin/bash
# W2a controls at K (default 10): shuffled-target and constant-only CE maps (both questions), dev rows none / JPP;
# then the non-word top-1 diagnostic per model, then the 9B names/numbers check (W2b).
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$R/runs/w2/logs; mkdir -p $L
K=${K:-10}
declare -A MOD=([2b]=Qwen/Qwen3.5-2B [9b]=Qwen/Qwen3.5-9B)
waitmem() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt ${1:-16000} ]; do sleep 30; done; }
for t in 2b 9b; do for var in shuf const; do for q in topic spec; do
  [ -f .hf27/w2/E_${t}_${q}_${var}_k$K.json ] && continue
  waitmem 20000
  $PY scripts/w2/w2_ce_fit.py ${MOD[$t]} $q $K --var $var > $L/fit_${t}_${q}_${var}.log 2>&1 || echo "FAILED fit $t $q $var" >> $L/errors
done; done; done
for t in 2b 9b; do for var in shuf const; do for row in none JPP; do
  n=w2_${var}_k${K}_${row}_$t
  [ -f auto/runs/$n/summary.json ] && continue
  waitmem 16000
  W2_VAR=$var W2_K=$K ALENS_BASE=$row $PY auto/evaluate.py --model ${MOD[$t]} --lens auto/candidates/w2_alens_ce.py --name $n --lite > $L/$n.log 2>&1 || echo "FAILED $n" >> $L/errors
  echo "$n $(grep -E 'chosen lam|dev_pass10:|dev_pass1:' $L/$n.log | sed 's/  */ /g' | tr '\n' ' ')" >> $L/resultsA.txt
done; done; done
for t in 2b 9b; do
  waitmem 16000
  $PY scripts/w2/nonword.py ${MOD[$t]} 10 50 200 > $L/nonword_$t.log 2>&1 || echo "FAILED nonword $t" >> $L/errors
done
bash scripts/w2/chainB.sh numnames
echo done > $L/chainC_done
