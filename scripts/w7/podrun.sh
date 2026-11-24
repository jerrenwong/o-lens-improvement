#!/bin/bash
# start a job on wong-olens4 in tmux: podrun.sh <job> "<cmd>"; log/exit in /workspace/.fleet/jobs/wong-olens4/
J=/workspace/.fleet/jobs/wong-olens4/$1
ssh wong-olens4 "rm -f $J.exit; tmux new-session -d -s $1 \"cd /workspace/o-lens-improvement && export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && ( $2 ) > $J.log 2>&1; echo \\\$? > $J.exit\""
echo "started $1"
