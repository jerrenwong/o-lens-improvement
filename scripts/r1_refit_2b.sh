#!/bin/bash
# r1: refit the J++ lens on Qwen3.5-2B with the authors' unmodified pipeline (jpp_cli via scripts/r1_jpp.py),
# varying only the fit text format (raw WikiText vs naturally formatted) and the skipped first positions.
#   A raw, skip 16 (recipe)   B natural, skip 16   C natural, skip 1   D raw, skip 1
# Usage (on the pod, in tmux):  N=128 bash scripts/r1_refit_2b.sh A B C D
# Prompt pools: runs/r1_2b/prompts_raw.json = jpp_cli.load_fit_prompts(1000) (scripts/r1_dump_prompts.py);
#               runs/r1_2b/prompts_natural.json = scripts/detok_wikitext.py of the same prompts.
set -euo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/r1_jpp.py"
M=Qwen/Qwen3.5-2B; L=3,6,9,12,15,18,21
N=${N:-128}                 # fit prompts (shards); same for every condition
NR=${NR:-1000}              # router prompts (recipe default 1000)
RPP=${RPP:-64}              # jacobian rows per backward pass: batching only (the released 2B fit used 64)
K=8; SEED=0; ITEMS=fit:0    # clusters, router seed, labelled fold for fit-weights (= auto/split_2b.json "fit")
OUT=${OUT:-$R/runs/r1_2b}
CSV=$R/data/model_correctness.csv
CACHE=/root/r1_readout_resid_cache.pt   # readout residuals of the eval items (shared by all conditions)

fit() {
  local c=$1 pj=$2 skip=$3 d=$OUT/$1
  mkdir -p "$d"
  echo "=== condition $c: prompts $pj skip $skip N $N router $NR K $K rows/pass $RPP"
  local t0=$(date +%s)
  $W --r1-router-prompts-json "$pj" fit-router --hf-model-name $M --layers $L --num-prompts $NR \
     --num-clusters $K --skip-first-n-positions "$skip" --seed $SEED --out "$d/router_K$K.pt"
  local t1=$(date +%s)
  local ck
  ck=$($W fit-shard --hf-model-name $M --layers $L --router-path "$d/router_K$K.pt" --num-prompts "$N" \
       --prompts-json "$pj" --skip-first-n-positions "$skip" --jacobian-rows-per-pass $RPP \
       --artifacts-base-dir "$d/ckpt" --checkpoint-name experts | tee /dev/stderr | tail -1)
  local t2=$(date +%s)
  $W merge-experts --checkpoint-dirs "$ck" --num-clusters $K --checkpoint-name "experts_$c" \
     --out "$d/experts_K$K.pt" --pooled-lens-out "$d/pooled_$c.pt"
  local t3=$(date +%s)
  $W fit-weights --hf-model-name $M --layers $L --experts "$d/experts_K$K.pt" --checkpoint-name "jpp_$c" \
     --items $ITEMS --correctness-csv "$CSV" --cache-path "$CACHE" --out "$d/jpp_$c.pt"
  local t4=$(date +%s)
  echo "cond=$c N=$N skip=$skip router_s=$((t1-t0)) shard_s=$((t2-t1)) merge_s=$((t3-t2)) weights_s=$((t4-t3)) total_s=$((t4-t0))" | tee "$d/timing.txt"
}

for c in "$@"; do
  case $c in
    A) fit A "$R/runs/r1_2b/prompts_raw.json" 16 ;;
    B) fit B "$R/runs/r1_2b/prompts_natural.json" 16 ;;
    C) fit C "$R/runs/r1_2b/prompts_natural.json" 1 ;;
    D) fit D "$R/runs/r1_2b/prompts_raw.json" 1 ;;
    *) echo "unknown condition $c"; exit 1 ;;
  esac
done
