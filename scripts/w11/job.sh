#!/bin/bash
# usage (on the pod): bash scripts/w11/job.sh <job> <command...>   tmux job, log + exit under /workspace/.fleet/jobs/wong-olens3
J=/workspace/.fleet/jobs/wong-olens3/$1; shift
rm -f $J.exit
tmux new -d -s "$(basename $J)" "cd /workspace/o-lens-improvement; export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src; $* > $J.log 2>&1; echo \$? > $J.exit"
