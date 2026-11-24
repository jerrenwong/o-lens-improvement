#!/bin/bash
# write scripts/w7/evals_sweep.txt for the sweep of one mode: K in 1 3 5 20 50 200 (T=1), T in 0.5 0.7 1.5 2 (K=10)
R=/workspace/o-lens-improvement; m=$1; f=$R/scripts/w7/evals_sweep.txt; : > $f
for t in 2b 9b; do [ $t = 9b ] && M=Qwen/Qwen3.5-9B || M=Qwen/Qwen3.5-2B
  for v in k1 k3 k5 k20 k50 k200 k10_t0.5 k10_t0.7 k10_t1.5 k10_t2; do n=$(echo $v | tr -d '_.'); for lv in f1 f4; do for row in none JPP; do
    echo "w7_${t}_${m}${n}_${row}_${lv}|ALENS_BASE=$row ALENS_E=$R/.hf27/w7/E_${t}_topic_${m}_${v}.pt ALENS_E2=$R/.hf27/w7/E_${t}_spec_${m}_${v}.pt|$M|$lv" >> $f
  done; done; done; done
wc -l $f
