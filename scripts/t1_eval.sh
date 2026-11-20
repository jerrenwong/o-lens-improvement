#!/bin/bash
# t1: dev-split scoring of the 12 lenses (one at a time, when >= 10 GB GPU free), then diagnostics.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LN=$R/runs/t1_2b/lenses
cd $R
waitfree() { while [ $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1) - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1) )) -lt 10000 ]; do sleep 60; done; }
for c in A T; do for n in full ha hb; do for k in jpp pooled; do
  nm=t1_${c}_${n}_${k}
  [ -f auto/runs/$nm/readout_filtered.csv ] && continue
  waitfree
  R1_LENS=$LN/${k}_${c}_${n}.pt $PY auto/evaluate.py --lens auto/candidates/r1_lensfile.py --model Qwen/Qwen3.5-2B --name $nm > /root/t1/eval_$nm.log 2>&1
  echo "$nm"; tail -12 /root/t1/eval_$nm.log
done; done; done
waitfree
$PY /root/t1/t1_diag.py A_full_jpp=$LN/jpp_A_full.pt T_full_jpp=$LN/jpp_T_full.pt A_full_pooled=$LN/pooled_A_full.pt T_full_pooled=$LN/pooled_T_full.pt > /root/t1/diag.log 2>&1
cat /root/t1/diag.log > $R/runs/t1_2b/diag.log
echo "[t1] EVAL DONE"
