#!/bin/bash
cd /workspace/o-lens-improvement
for n in g1_G g1_Gall g1_AG g1_G_shuf g1_Galone; do
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 28000 ]; do echo "$(date +%T) wait gpu"; sleep 60; done
  /root/jpp_lens/.venv/bin/python auto/evaluate.py --lens auto/candidates/$n.py --name $n > /root/g1/eval_$n.log 2>&1
  echo "$(date +%T) done $n"
done
echo EVALS_DONE
