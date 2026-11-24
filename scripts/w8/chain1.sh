#!/bin/bash
# W8 GPU1 chain: after the J-lens refit, the A-lens refit of secret_loyalty (owns /root/w8/arefit.lock).
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; J=/workspace/.fleet/jobs/wong-olens5
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/dev/shm/hf27 HF_HUB_CACHE=/dev/shm/hf27/hub HF_HUB_OFFLINE=1 W8_OWN_LOCK=1
cd $R
touch /root/w8/arefit.lock
until [ -f $J/w8jfit.exit ]; do sleep 30; done
$PY scripts/w8/alens_refit.py secret_loyalty > $J/w8arefit.log 2>&1; echo $? > $J/w8arefit.exit
