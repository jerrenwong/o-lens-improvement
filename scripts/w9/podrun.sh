#!/bin/bash
# start a job on wong-olens3 in tmux: podrun.sh <job> "<cmd>" [online]; log/exit in /workspace/.fleet/jobs/wong-olens3/
J=/workspace/.fleet/jobs/wong-olens3/$1
OFF="export HF_HUB_OFFLINE=1 &&"; [ "$3" = online ] && OFF="unset HF_HUB_OFFLINE &&"
ssh wong-olens3 "rm -f $J.exit; tmux new-session -d -s $1 \"cd /workspace/o-lens-improvement && $OFF export PYTHONPATH=/workspace/o-lens-improvement/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ( $2 ) > $J.log 2>&1; echo \\\$? > $J.exit\""
echo "started $1"
