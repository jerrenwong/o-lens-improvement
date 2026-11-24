#!/bin/bash
# w9: MLP-CE learning-rate probe on one layer (2B, 98k, topic); selection by val KL only
for lr in 1e-5 3e-5 3e-4; do
  /root/jpp_lens/.venv/bin/python scripts/w9/w9_fit.py Qwen/Qwen3.5-2B topic --n all --layers 12 --archs lin,mlp2048 --losses ce --lr_mlp $lr --verbose --tag_extra _probece$lr
done
