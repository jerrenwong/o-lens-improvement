#!/bin/bash
# start a job on wong-olens3 in tmux (fallback for fleet run): podrun.sh <job> "<cmd>"; log/exit in /workspace/.fleet/jobs/wong-olens3/
J=/workspace/.fleet/jobs/wong-olens3/$1
ssh wong-olens3 "rm -f $J.exit; tmux new-session -d -s $1 \"cd /workspace/o-lens-improvement && ( $2 ) > $J.log 2>&1; echo \\\$? > $J.exit\""
echo "started $1"
