#!/bin/bash
# w12 9B: per-layer processes (train4) for the configs in $1, each started when >= 20 GB GPU memory is free
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/o-lens-improvement/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/jpp_lens/.venv/bin/python; cd /workspace/o-lens-improvement; mkdir -p runs/w12/logs
free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
for l in 4 8 12 16 20 24 28; do
  until [ $(free) -gt 20000 ]; do sleep 20; done
  $PY scripts/w12/w12_train4.py Qwen/Qwen3.5-9B --layer $l --cfgs $1 > runs/w12/logs/train9b_$2_L$l.log 2>&1 &
  sleep 150
done
wait
