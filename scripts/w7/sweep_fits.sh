#!/bin/bash
# Exp 3.3 sweep fits for one model and one mode: K in 1 3 5 20 50 200 at T=1; T in 0.5 0.7 1.5 2 at K=10
t=$1; m=$2
for K in 1 3 5 20 50 200; do bash scripts/w7/chain.sh fit $t $K 1 $m; done
for T in 0.5 0.7 1.5 2; do bash scripts/w7/chain.sh fit $t 10 $T $m; done
