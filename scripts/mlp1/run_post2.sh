#!/bin/bash
# mlp1 chain 4: step-cap check at layer 20 (D and C-mean hit their best near the 20-epoch cap): 40 epochs, seed 0.
PY=/root/jpp_lens/.venv/bin/python
cd /workspace/o-lens-improvement || exit 1
until [ -f /root/mlp1/DONE_post ]; do sleep 60; done
MLP1_EPOCHS=40 MLP1_SUFFIX=_ep40 $PY scripts/mlp1/mlp1_train.py 20 scripts/mlp1/specs_L20_ep40.json > /root/mlp1/ep40.log 2>&1
touch /root/mlp1/DONE_post2
