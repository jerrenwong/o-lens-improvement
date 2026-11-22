#!/bin/bash
# al1: A-lens family (auto/candidates/alens.py, PREREG_v4 addendum E). Phase 1 DEV; phase 2 held-out (orchestrator only).
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/al1
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { # tag row kind(dev|test|v4)
  tag=$1; row=$2; kind=$3; unset HF_HOME
  case $tag in 9b) M=Qwen/Qwen3.5-9B; EV=auto/evaluate.py; need=14000;; 27b) M=Qwen/Qwen3.6-27B; EV=auto/evaluate27.py; export HF_HOME=$R/.hf27; need=20000;; esac
  case $kind in dev) n=D_al_${row}_$tag; args="";; test) n=T_al_${row}_$tag; args="--split test --orchestrator";; v4) n=V4_al_${row}_$tag; EV=auto/evaluate_v4.py; args="--split test --orchestrator";; esac
  [ -f auto/runs/$n/summary.json ] && return
  for try in 1 2 3 4 5 6; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $need ]; do sleep 30; done
    ALENS_BASE=$row $PY $EV --model $M --lens auto/candidates/alens.py --name $n --lite $args > /root/al1/$n.log 2>&1 && break
    sleep 60
  done
  echo "$n $(grep -E 'chosen lam|pass10:|pass1:|n_items' /root/al1/$n.log | sed 's/  */ /g' | tr '\n' ' ')" >> /root/al1/results.txt
}
for kind in ${KINDS:-dev}; do for tag in 9b 27b; do for row in J0 JPP0 none J JPP; do run $tag $row $kind; done; done; touch /root/al1/DONE_$kind; done
touch /root/al1/DONE
