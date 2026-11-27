#!/bin/bash
# Run ON THE POD (wong-olens6): bash scripts/q1/job.sh <job> '<command>'  -> tmux session <job> on GPU 0,
# log + exit code in /workspace/.fleet/jobs/wong-olens6/<job>.{log,exit}. Models from the pod-local cache /root/hf_q1.
j=$1; shift
D=/workspace/.fleet/jobs/wong-olens6
rm -f $D/$j.exit
tmux new -d -s $j "cd /workspace/o-lens-improvement; export HF_HOME=/root/hf_q1 HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src CUDA_VISIBLE_DEVICES=0 PY=/root/jpp_lens/.venv/bin/python; $* > $D/$j.log 2>&1; echo \$? > $D/$j.exit"
