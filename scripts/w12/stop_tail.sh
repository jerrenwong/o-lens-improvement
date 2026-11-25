#!/bin/bash
# w12: stop each 2B layer process once it has finished its 14 main configs (last main = T16_J_mu0_shuf); the 4 tail
# configs (T4_I_mu0, T4_J_mu100, T16d_JPP_mu0, T16d_I_mu0) are dropped for time.
cd /workspace/o-lens-improvement
while pgrep -f "w12_train[34].py Qwen/Qwen3.5-2B" > /dev/null; do
  for l in 3 6 9 12 15 18 21; do
    if [ -f runs/w12/fits/2b_T16_J_mu0_shuf_L$l.json ]; then
      pkill -f "w12_train[34].py Qwen/Qwen3.5-2B --layer $l " && echo "stopped L$l $(date)"
    fi
  done
  sleep 20
done
