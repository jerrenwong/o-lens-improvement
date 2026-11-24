#!/bin/bash
# usage: bash scripts/w10/job.sh <job> <command...>   (runs in tmux on the pod; log + exit under /workspace/.fleet/jobs/wong-olens4)
J=/workspace/.fleet/jobs/wong-olens4/$1; shift
rm -f $J.exit
tmux new -d -s "$(basename $J)" "cd /workspace/o-lens-improvement; $* > $J.log 2>&1; echo \$? > $J.exit"
