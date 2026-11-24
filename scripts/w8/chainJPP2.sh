#!/bin/bash
# W8 follow-up: (a) J++ router on GPU1 alone (forward only) while GPU0 finishes the flattery A-lens fit;
# (b) after the A-lens fit: read_all + answers_cmp for flattery on GPU1; (c) J++ shard + merge + weights on both GPUs.
R=/workspace/o-lens-improvement; J=/workspace/.fleet/jobs/wong-olens5; NP=${NP:-8}; PY=/root/jpp_lens/.venv/bin/python
export HF_HOME=/dev/shm/hf27 HF_HUB_CACHE=/dev/shm/hf27/hub HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R
CUDA_VISIBLE_DEVICES=1 ROUTER_ONLY=1 bash scripts/w8/fit_jpp_org.sh flattery_tdkto_r64 $NP
echo "router stage exit $?"
until [ -f $J/w8arefF.exit ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 $PY scripts/w8/read_all.py flattery > $J/w8raF.log 2>&1; echo $? > $J/w8raF.exit
CUDA_VISIBLE_DEVICES=1 $PY scripts/w8/answers_cmp.py flattery > $J/w8ansF.log 2>&1; echo $? > $J/w8ansF.exit
CUDA_VISIBLE_DEVICES=0,1 bash scripts/w8/fit_jpp_org.sh flattery_tdkto_r64 $NP
