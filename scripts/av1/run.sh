#!/bin/bash
# av1: ask-map defined as the plain average of the topic map and the unstated-word map (user, 2026-11-22). DEV only.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R; mkdir -p /root/av1
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { # tag topicmap name alone
  tag=$1; E=$2; n=$3; alone=$4; EV=auto/evaluate.py; unset HF_HOME
  case $tag in 9b) M=Qwen/Qwen3.5-9B; need=14000;; 27b) M=Qwen/Qwen3.6-27B; EV=auto/evaluate27.py; export HF_HOME=$R/.hf27; need=20000;; esac
  [ -f auto/runs/$n/summary.json ] && return
  for try in 1 2 3 4 5 6; do
    until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -gt $need ]; do sleep 30; done
    AO1_E=$E AO1_E2=$R/.hf27/ask5/E_${tag}_s5.pt AV1_ALONE=$alone $PY $EV --model $M --lens auto/candidates/av1_L.py --name $n --lite > /root/av1/$n.log 2>&1 && break
    sleep 60
  done
  echo "$n $(grep -E 'chosen lam|dev_pass10:|dev_pass1:' /root/av1/$n.log | sed 's/  */ /g' | tr '\n' ' ')" >> /root/av1/results.txt
}
for tag in 9b 27b; do
  run $tag $R/.hf27/ask3/E_${tag}_t3.pt av1_t3_${tag} ""
  run $tag $R/.hf27/ask3/E_${tag}_t3.pt av1_t3_A_${tag} 1
  run $tag $R/.hf27/st1/E_${tag}_p3v_t10.pt av1_p3v_${tag} ""
  run $tag $R/.hf27/st1/E_${tag}_p3v_t10.pt av1_p3v_A_${tag} 1
done
touch /root/av1/DONE
