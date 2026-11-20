#!/bin/bash
# r2: refit the J++ lens on Qwen3.5-9B with the authors' unmodified pipeline (jpp_cli via scripts/r1_jpp.py),
# two conditions, fitted in 4 disjoint prompt shards each so that replicates come for free:
#   C = natural-format prompts, skip_first_n_positions 1      A = raw WikiText prompts, skip 16 (recipe)
# Shard i of condition X = prompts [i*S, (i+1)*S) of the pool (jpp_cli --num-prompts 4S --num-shards 4 --shard-idx i);
# raw and natural pools hold the same texts in the same order (runs/r1_2b/prompts_{raw,natural}.json).
# Usage (on the pod, in tmux):
#   bash scripts/r2_refit_9b.sh router C          # 1000-prompt router of the matching format, K=8, seed 0
#   bash scripts/r2_refit_9b.sh shard C 0         # one shard
#   bash scripts/r2_refit_9b.sh build C C_64a 0 1 # merge listed shards + pooled lens + fit-weights (fit:0)
#   bash scripts/r2_refit_9b.sh timing            # 4 prompts of C (router must exist)
#   bash scripts/r2_refit_9b.sh plan              # the full interleaved plan
set -euo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/r1_jpp.py"
M=Qwen/Qwen3.5-9B; L=4,8,12,16,20,24,28
S=${S:-32}                  # prompts per shard
NS=4                        # shards per condition
NR=${NR:-1000}              # router prompts
RPP=${RPP:-32}              # jacobian rows per backward pass (batching only)
CKE=${CKE:-8}               # checkpoint every n prompts
K=8; SEED=0; ITEMS=fit:0
OUT=${OUT:-$R/runs/r2_9b}
CSV=$R/data/model_correctness.csv
CACHE=/root/r2_readout_resid_cache_9b.pt
pool() { case $1 in C) echo $R/runs/r1_2b/prompts_natural.json;; A) echo $R/runs/r1_2b/prompts_raw.json;; *) exit 1;; esac; }
skip() { case $1 in C) echo 1;; A) echo 16;; *) exit 1;; esac; }

router() {
  local c=$1 d=$OUT/$1; mkdir -p $d; local t0=$(date +%s)
  $W --r1-router-prompts-json "$(pool $c)" fit-router --hf-model-name $M --layers $L --num-prompts $NR \
     --num-clusters $K --skip-first-n-positions "$(skip $c)" --seed $SEED --out "$d/router_K$K.pt"
  echo "router cond=$c s=$(( $(date +%s)-t0 ))" | tee -a $d/timing.txt
}
shard() {  # c i [nprompts nshards tag]
  local c=$1 i=$2 n=${3:-$((S*NS))} ns=${4:-$NS} tag=${5:-shards} d=$OUT/$1; local t0=$(date +%s)
  local ck
  ck=$($W fit-shard --hf-model-name $M --layers $L --router-path "$d/router_K$K.pt" --num-prompts $n \
       --num-shards $ns --shard-idx $i --prompts-json "$(pool $c)" --skip-first-n-positions "$(skip $c)" \
       --jacobian-rows-per-pass $RPP --checkpoint-every-n-prompts $CKE \
       --artifacts-base-dir "$d/ckpt" --checkpoint-name $tag | tee /dev/stderr | tail -1)
  echo "$ck" > $d/${tag}_$i.dir
  echo "shard cond=$c tag=$tag idx=$i/$ns n=$n rpp=$RPP s=$(( $(date +%s)-t0 ))" | tee -a $d/timing.txt
}
build() {  # c name shard_idx...
  local c=$1 name=$2; shift 2; local d=$OUT/$c/$name; mkdir -p $d; local t0=$(date +%s)
  local dirs=(); for i in "$@"; do dirs+=("$(cat $OUT/$c/shards_$i.dir)"); done
  $W merge-experts --checkpoint-dirs "${dirs[@]}" --num-clusters $K --checkpoint-name "experts_$name" \
     --out "$d/experts_K$K.pt" --pooled-lens-out "$d/pooled_$name.pt"
  local t1=$(date +%s)
  $W fit-weights --hf-model-name $M --layers $L --experts "$d/experts_K$K.pt" --checkpoint-name "jpp_$name" \
     --items $ITEMS --correctness-csv "$CSV" --cache-path "$CACHE" --out "$d/jpp_$name.pt"
  echo "build $name shards=$* merge_s=$((t1-t0)) weights_s=$(( $(date +%s)-t1 ))" | tee -a $d/timing.txt
}

case $1 in
  router) router $2 ;;
  shard) shard $2 $3 ;;
  build) build "${@:2}" ;;
  timing) shard C 0 4 1 timing ;;
  plan)
    [ -f $OUT/C/router_K$K.pt ] || router C
    [ -f $OUT/A/router_K$K.pt ] || router A
    for i in 0 1; do for c in C A; do [ -f $OUT/$c/shards_$i.dir ] || shard $c $i; done; done
    build C C_64a 0 1; build A A_64a 0 1; touch $OUT/half_done
    for i in 2 3; do for c in C A; do [ -f $OUT/$c/shards_$i.dir ] || shard $c $i; done; done
    for c in C A; do build $c ${c}_full 0 1 2 3; build $c ${c}_64b 2 3; done
    touch $OUT/plan_done ;;
  *) echo "usage: see header"; exit 1 ;;
esac
