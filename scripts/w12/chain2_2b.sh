#!/bin/bash
# w12 2B main training: one process per readout layer (in parallel), configs in priority order; watcher merges + evals
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src
PY=/root/jpp_lens/.venv/bin/python; cd /workspace/o-lens-improvement; mkdir -p runs/w12/logs
for l in 3 6 9 12 15 18 21; do
  $PY scripts/w12/w12_train.py Qwen/Qwen3.5-2B --layer $l --cfgs T1_J_mu0,T4_J_mu0,T16_J_mu0,T16d_J_mu0,T1_JPP_mu0,T4_JPP_mu0,T16_JPP_mu0,T16d_JPP_mu0,T1_I_mu0,T4_I_mu0,T16_I_mu0,T16d_I_mu0,T1_J_mu100,T4_J_mu100,T16_J_mu100,T16d_J_mu100,T1_J_mu1000000,T1_J_mu0_shuf,T16_J_mu0_shuf > runs/w12/logs/train2b_L$l.log 2>&1 &
  sleep 20
done
$PY scripts/w12/w12_watch.py Qwen/Qwen3.5-2B T1_J_mu0,T4_J_mu0,T16_J_mu0,T16d_J_mu0,T1_JPP_mu0,T4_JPP_mu0,T16_JPP_mu0,T16d_JPP_mu0,T1_I_mu0,T4_I_mu0,T16_I_mu0,T16d_I_mu0,T1_J_mu100,T4_J_mu100,T16_J_mu100,T16d_J_mu100,T1_J_mu1000000,T1_J_mu0_shuf,T16_J_mu0_shuf --par 3 > runs/w12/logs/watch2b.log 2>&1 &
wait
