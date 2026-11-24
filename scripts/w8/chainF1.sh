#!/bin/bash
# W8 follow-up GPU1 chain: after the flattery J-lens refit, help the (thermally throttled) GPU0 A-lens refit by
# collecting the s5 wordings (then t3 w2, w1) in reverse order on GPU1; the GPU0 process loads finished files.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; J=/workspace/.fleet/jobs/wong-olens5
export CUDA_VISIBLE_DEVICES=1 HF_HOME=/dev/shm/hf27 HF_HUB_CACHE=/dev/shm/hf27/hub HF_HUB_OFFLINE=1 W8_ONLY=s5rev PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R
until [ -f $J/w8jfitF.exit ]; do sleep 30; done
$PY scripts/w8/alens_refit.py flattery 0 > $J/w8ahelpF.log 2>&1; echo $? > $J/w8ahelpF.exit
