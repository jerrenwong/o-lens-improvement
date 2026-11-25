#!/bin/bash
# w11b M2 step 1 (pod): plain-rule (lrp none) expert Jacobians for Qwen3.5-2B at layers 15,18,21 with the authors'
# unmodified pipeline (scripts/r1_jpp.py -> jpp_cli): routers K=8 and K=64 (first 1000 J-fit prompts, skip 16, seed 0),
# one shared shard fit on the first 128 prompts (max_seq_len 128, target = final block), merged per K.
set -euo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/r1_jpp.py"
export HF_HUB_OFFLINE=1
M=Qwen/Qwen3.5-2B; LAY=15,18,21; P=$R/runs/r1_2b/prompts_raw.json; B=$R/.hf27/w11/m2fit; mkdir -p $B
cd /root/jpp_lens
for K in 8 64; do
  $W --r1-router-prompts-json $P fit-router --hf-model-name $M --layers $LAY --num-prompts 1000 --num-clusters $K \
     --skip-first-n-positions 16 --seed 0 --out $B/router_K$K.pt
done
ck=$($W fit-shard --hf-model-name $M --layers $LAY --router-path $B/router_K8.pt $B/router_K64.pt --num-prompts 128 \
     --prompts-json $P --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass 32 \
     --checkpoint-every-n-prompts 1000 --artifacts-base-dir $B/ckpt --checkpoint-name m2 | tail -1)
echo "shard dir: $ck"
for K in 8 64; do
  $W merge-experts --checkpoint-dirs $ck --num-clusters $K --checkpoint-name m2_K$K --out $B/experts_K$K.pt \
     --pooled-lens-out $B/pooled_K$K.pt
done
echo done
