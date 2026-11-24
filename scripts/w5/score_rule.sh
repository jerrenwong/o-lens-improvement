#!/bin/bash
# W5: score every arm on the four rule-scored families with no API call (opts=aggregate=native),
# at the layers the readouts hold. Usage (pod): bash score_rule.sh [layers]
L=${1:-24,32,40,48,56}
RO=/workspace/o-lens-improvement/.hf27/w5/readouts
OUT=/workspace/o-lens-improvement/runs/w5/native_L${L//,/_}
FAMS=multilingual,multilingual_typo,typo,model_specific_directed_modulation
cd /root/wsbench || exit 1
for arm in $(ls $RO | grep -v '^_'); do
  env -u ANTHROPIC_API_KEY .venv/bin/wsbench run families=$FAMS readouts_root=$RO/$arm out=$OUT/$arm \
    opts=aggregate=native layers=$L > /root/w5_score_$arm.log 2>&1
  echo "$arm exit=$?"; tail -7 /root/w5_score_$arm.log
done
