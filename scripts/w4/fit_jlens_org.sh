#!/bin/bash
# W4 step 3c: J-lens refitted on an organism, recipe of scripts/xm1/fit_jpp.sh "jlens" (K=1, --lrp-mode none,
# router on the first 1000 prompts of runs/r1_2b/prompts_raw.json, shard on the first 64, skip 16, seed 0, pooled lens).
# The organism = base + LoRA merged at load time (scripts/w4/r1_jpp_org.py wraps from_pretrained); output .hf27/w4/jlens_<word>.pt.
# Usage (pod): bash scripts/w4/fit_jlens_org.sh <word>
set -euo pipefail
WORD=$1
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/w4/r1_jpp_org.py"
export W4_ADAPTER=$R/.hf27/w4/taboo_$WORD HF_HUB_OFFLINE=1
M=Qwen/Qwen3.5-9B; B=/root/w4/fit_$WORD; LAY=4,8,12,16,20,24,28
RPP=${RPP:-32}; NP=64; NR=1000; P=$R/runs/r1_2b/prompts_raw.json
mkdir -p $B
t0=$(date +%s)
$W --r1-router-prompts-json $P fit-router --hf-model-name $M --layers $LAY --num-prompts $NR --num-clusters 1 \
   --skip-first-n-positions 16 --seed 0 --out $B/router_K1.pt
ck=$($W fit-shard --hf-model-name $M --layers $LAY --router-path $B/router_K1.pt --num-prompts $NP \
     --prompts-json $P --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass $RPP \
     --checkpoint-every-n-prompts 1000 --artifacts-base-dir $B/ckpt --checkpoint-name jlens_$WORD | tail -1)
$W merge-experts --checkpoint-dirs $ck --num-clusters 1 --checkpoint-name experts_jlens_$WORD \
   --out $B/experts_K1.pt --pooled-lens-out $R/.hf27/w4/jlens_$WORD.pt
echo "jlens refit $WORD done in $(( $(date +%s) - t0 )) s"
