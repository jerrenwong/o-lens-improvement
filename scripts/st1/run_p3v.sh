#!/bin/bash
# st1 P3v (full-vocab softmax) at T=1 + its dev evals: bash run_p3v.sh <tag> <model> <need_mib>
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ $1 = 27b ] && export HF_HOME=$R/.hf27
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}'; }
for a in 1 2 3 4 5 6; do
  while [ "$(free_mib)" -lt "$3" ]; do sleep 60; done
  $PY scripts/st1/st1_p3v.py $2 1.0 >> /root/st1/p3v_$1.log 2>&1 && break
  sleep 120
done
CANDS=st1_p3v_t10 bash scripts/st1/run4.sh $1 eval
touch /root/st1/DONEv_$1
