#!/bin/bash
# W8 step 5b: J-lens refitted on an AuditBench organism (Qwen3.6-27B + combined tdkto LoRA merged at load time by
# scripts/w4/r1_jpp_org.py), recipe of scripts/xm1/fit_jpp.sh "jlens": K=1, --lrp-mode none, router on the first 1000
# prompts of runs/r1_2b/prompts_raw.json, shard on the first NP, skip 16, seed 0, layers 8..56, pooled lens.
# Checkpoints every 4 prompts on pod-local disk; if the time box ends early, merge the latest checkpoint (merge step below).
# Usage (pod): bash scripts/w8/fit_jlens_org.sh <tag> [NP]
set -euo pipefail
TAG=$1; NP=${2:-64}
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/w4/r1_jpp_org.py"
export W4_ADAPTER=/root/w8/adapters/$TAG HF_HOME=/dev/shm/hf27 HF_HUB_CACHE=/dev/shm/hf27/hub HF_HUB_OFFLINE=1
M=Qwen/Qwen3.6-27B; B=/root/w8/fit_$TAG; LAY=8,16,24,32,40,48,56
RPP=${RPP:-32}; NR=1000; P=$R/runs/r1_2b/prompts_raw.json
mkdir -p $B
t0=$(date +%s)
[ -f $B/router_K1.pt ] || $W --r1-router-prompts-json $P fit-router --hf-model-name $M --device-map cuda --layers $LAY --num-prompts $NR --num-clusters 1 \
   --skip-first-n-positions 16 --seed 0 --out $B/router_K1.pt
echo "router done $(( $(date +%s) - t0 )) s"
ck=$($PY $R/scripts/w8/jpp_org_fla.py fit-shard --hf-model-name $M --device-map cuda --layers $LAY --router-path $B/router_K1.pt --num-prompts $NP \
     --prompts-json $P --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass $RPP \
     --checkpoint-every-n-prompts 4 --artifacts-base-dir $B/ckpt --checkpoint-name jlens_$TAG | tail -1)
echo "shard done $(( $(date +%s) - t0 )) s ck=$ck"
$PY $R/scripts/w8/jpp_org_fla.py merge-experts --checkpoint-dirs $ck --num-clusters 1 --checkpoint-name experts_jlens_$TAG \
   --out $B/experts_K1.pt --pooled-lens-out $R/.hf27/w8/jlens_$TAG.pt
echo "jlens refit $TAG done in $(( $(date +%s) - t0 )) s"
