#!/bin/bash
# w12 2B: layers 6 and 3 (train4, lean memory), each started when >= 14 GB GPU memory is free
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/jpp_lens/.venv/bin/python; cd /workspace/o-lens-improvement
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for l in 6 3; do
  until [ $(free) -gt 14000 ]; do sleep 20; done
  $PY scripts/w12/w12_train4.py Qwen/Qwen3.5-2B --layer $l --cfgs T1_J_mu0,T4_J_mu0,T16_J_mu0,T16d_J_mu0,T1_JPP_mu0,T4_JPP_mu0,T16_JPP_mu0,T1_I_mu0,T16_I_mu0,T1_J_mu100,T16_J_mu100,T1_J_mu1000000,T1_J_mu0_shuf,T16_J_mu0_shuf,T4_I_mu0,T4_J_mu100,T16d_JPP_mu0,T16d_I_mu0 > runs/w12/logs/train2b_v4_L$l.log 2>&1 &
  sleep 240
done
wait
