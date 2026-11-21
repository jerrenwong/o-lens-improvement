#!/bin/bash
# fm27: run lite dev evals one at a time. Args: list of "<9b|27b>:<candidate>" pairs.
# Before each run wait until enough GPU memory is free (two checks 30 s apart).
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/fm27
mkdir -p $L
cd $R
waitgpu() {  # $1 = MiB needed free
  while true; do
    a=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); sleep 30
    b=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$a" -gt "$1" ] && [ "$b" -gt "$1" ] && break
    sleep 30
  done
}
for x in "$@"; do
  m=${x%%:*}; c=${x#*:}
  if [ "$m" = 27b ]; then
    waitgpu 20500
    HF_HOME=/workspace/o-lens-improvement/.hf27 $PY auto/evaluate27.py --model Qwen/Qwen3.6-27B --lens auto/candidates/$c.py --name ${c}_27b --lite > $L/${c}_27b.log 2>&1
  else
    waitgpu 11000
    $PY auto/evaluate.py --lens auto/candidates/$c.py --name ${c}_9b --lite > $L/${c}_9b.log 2>&1
  fi
  echo "$(date +%T) $x exit=$?" >> $L/done.log
done
echo ALLDONE "$@" >> $L/done.log
