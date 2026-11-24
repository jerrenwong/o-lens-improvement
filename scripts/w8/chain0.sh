#!/bin/bash
# W8 GPU0 chain: steps 0-2 for each organism, then the A-lens refit on secret_loyalty.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; J=/workspace/.fleet/jobs/wong-olens5
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/dev/shm/hf27 HF_HUB_CACHE=/dev/shm/hf27/hub HF_HUB_OFFLINE=1
cd $R
for q in ${QS:-secret_loyalty flattery hardcode_test_cases anti_ai_regulation}; do
  [ -f $R/.hf27/w8/tops_${q}_base.pt ] && continue
  $PY scripts/w8/gen_collect.py $q > $J/w8gen_$q.log 2>&1; echo $? > $J/w8gen_$q.exit
done
[ -n "$NOREFIT" ] || { $PY scripts/w8/alens_refit.py secret_loyalty > $J/w8arefit.log 2>&1; echo $? > $J/w8arefit.exit; }
