#!/bin/bash
# w12 2B main training (v3: X on GPU, 4 threads; each layer process starts only when >= 14 GB GPU memory is free)
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src
PY=/root/jpp_lens/.venv/bin/python; cd /workspace/o-lens-improvement; mkdir -p runs/w12/logs
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for l in 18 21 15 12 9 6 3; do
  until [ $(free) -gt 14000 ]; do sleep 15; done
  $PY scripts/w12/w12_train3.py Qwen/Qwen3.5-2B --layer $l --cfgs T1_J_mu0,T4_J_mu0,T16_J_mu0,T16d_J_mu0,T1_JPP_mu0,T4_JPP_mu0,T16_JPP_mu0,T1_I_mu0,T16_I_mu0,T1_J_mu100,T16_J_mu100,T1_J_mu1000000,T1_J_mu0_shuf,T16_J_mu0_shuf,T4_I_mu0,T4_J_mu100,T16d_JPP_mu0,T16d_I_mu0 > runs/w12/logs/train2b_v3_L$l.log 2>&1 &
  sleep 60
done
$PY scripts/w12/w12_watch.py Qwen/Qwen3.5-2B T1_J_mu0,T4_J_mu0,T16_J_mu0,T16d_J_mu0,T1_JPP_mu0,T4_JPP_mu0,T16_JPP_mu0,T1_I_mu0,T16_I_mu0,T1_J_mu100,T16_J_mu100,T1_J_mu1000000,T1_J_mu0_shuf,T16_J_mu0_shuf,T4_I_mu0,T4_J_mu100,T16d_JPP_mu0,T16d_I_mu0 --par 2 > runs/w12/logs/watch2b_v3.log 2>&1 &
wait
