#!/bin/bash
# W8 follow-up: J++ refitted on an organism (base + combined tdkto LoRA merged at load, scripts/w8/jpp_org_fla.py),
# recipe of scripts/xm1/fit_jpp.sh "jpp" + "weights": router K=8 on the first 1000 prompts of runs/r1_2b/prompts_raw.json,
# skip 16, seed 0; shard on the first NP prompts (reduced from 64), --lrp-mode rlens (fla hidden, as r1_jpp.py);
# merge-experts; fit-weights on fit:0 with data/model_correctness_xm1.csv (base-model correctness column).
# Model sharded over the visible GPUs (--device-map auto): the torch GatedDeltaNet path OOMs on one 80 GB GPU.
# Usage (pod): bash scripts/w8/fit_jpp_org.sh <tag> <NP>
set -euo pipefail
TAG=$1; NP=$2
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/w8/jpp_org_fla.py"
export W4_ADAPTER=/root/w8/adapters/$TAG HF_HOME=/dev/shm/hf27 HF_HUB_CACHE=/dev/shm/hf27/hub HF_HUB_OFFLINE=1 W8_HIDE_FLA=1
M=Qwen/Qwen3.6-27B; B=/root/w8/fitjpp_$TAG; LAY=8,16,24,32,40,48,56
RPP=${RPP:-8}; NR=1000; P=$R/runs/r1_2b/prompts_raw.json; CSV=$R/data/model_correctness_xm1.csv
mkdir -p $B
t0=$(date +%s)
[ -f $B/router_K8.pt ] || $W --r1-router-prompts-json $P fit-router --hf-model-name $M --device-map auto --layers $LAY \
   --num-prompts $NR --num-clusters 8 --skip-first-n-positions 16 --seed 0 --out $B/router_K8.pt
echo "router done $(( $(date +%s) - t0 )) s"
[ -n "${ROUTER_ONLY:-}" ] && exit 0
ck=$($W fit-shard --hf-model-name $M --device-map auto --layers $LAY --router-path $B/router_K8.pt --num-prompts $NP \
     --prompts-json $P --skip-first-n-positions 16 --lrp-mode rlens --jacobian-rows-per-pass $RPP \
     --checkpoint-every-n-prompts 2 --artifacts-base-dir $B/ckpt --checkpoint-name jpp_$TAG | tail -1)
echo "shard done $(( $(date +%s) - t0 )) s ck=$ck"
$W merge-experts --checkpoint-dirs $ck --num-clusters 8 --checkpoint-name experts_jpp_$TAG \
   --out $B/experts_K8.pt --pooled-lens-out $B/pooled_jpp.pt
echo "merge done $(( $(date +%s) - t0 )) s"
$W fit-weights --hf-model-name $M --device-map auto --layers $LAY --experts $B/experts_K8.pt --checkpoint-name jpp_$TAG \
   --items fit:0 --correctness-csv $CSV --cache-path $B/readout_resid_cache.pt --out $R/.hf27/w8/jpp_$TAG.pt
echo "jpp refit $TAG done in $(( $(date +%s) - t0 )) s (NP=$NP)"
