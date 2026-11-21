#!/bin/bash
# mlp1 chain: sanity (all layers, ridge A refit + reproduction), then layer-20 grid stages 1-3, final pick.
PY=/root/jpp_lens/.venv/bin/python
cd /workspace/o-lens-improvement || exit 1
S=scripts/mlp1
rm -f /root/mlp1/DONE_l20 /root/mlp1/FAILED_l20
set -e
trap 'touch /root/mlp1/FAILED_l20' ERR
[ -f .hf27/mlp1/base_ranks.pt ] || $PY $S/mlp1_sanity.py > /root/mlp1/sanity.log 2>&1
$PY $S/mlp1_train.py 20 $S/specs_L20_s1.json > /root/mlp1/l20_s1.log 2>&1
$PY $S/mlp1_pick.py 20 stage2 $S/specs_L20_s2.json > /root/mlp1/pick2.log 2>&1
$PY $S/mlp1_train.py 20 $S/specs_L20_s2.json > /root/mlp1/l20_s2.log 2>&1
$PY $S/mlp1_pick.py 20 stage3 $S/specs_L20_s3.json > /root/mlp1/pick3.log 2>&1
$PY $S/mlp1_train.py 20 $S/specs_L20_s3.json > /root/mlp1/l20_s3.log 2>&1
$PY $S/mlp1_pick.py 20 final $S/final_L20.json > /root/mlp1/pickf.log 2>&1
touch /root/mlp1/DONE_l20
