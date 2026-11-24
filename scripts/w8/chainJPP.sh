#!/bin/bash
# W8 follow-up: J++ refit on flattery. Router on GPU1 alone (forward only) while GPU0 finishes the A-lens fit,
# then shard + merge + weights with the model sharded over both GPUs.
R=/workspace/o-lens-improvement; J=/workspace/.fleet/jobs/wong-olens5; NP=${NP:-8}
cd $R
CUDA_VISIBLE_DEVICES=1 ROUTER_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/w8/fit_jpp_org.sh flattery_tdkto_r64 $NP
echo "router stage exit $?"
until [ -f $J/w8arefF.exit ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/w8/fit_jpp_org.sh flattery_tdkto_r64 $NP
