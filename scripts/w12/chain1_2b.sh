#!/bin/bash
# w12 2B: targets, then lr check on L12 (T1_J_mu0, lr_rel 3e-4 / 1e-3 / 3e-3, max 1500 steps, no save)
set -e
J=/workspace/.fleet/jobs/wong-olens4
until [ -f $J/w12col2b.exit ]; do sleep 10; done
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src
PY=/root/jpp_lens/.venv/bin/python; cd /workspace/o-lens-improvement
$PY scripts/w12/w12_targets.py Qwen/Qwen3.5-2B .hf27/w12/2b
for lr in 3e-4 1e-3 3e-3; do
  $PY scripts/w12/w12_train.py Qwen/Qwen3.5-2B --layer 12 --cfgs T1_J_mu0 --lr_rel $lr --max_steps 1500 --nosave --suffix _lr$lr
done
