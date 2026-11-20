#!/bin/bash
# r2 v2: same pipeline as scripts/r2_refit_9b.sh, with one storage fix. Shard checkpoints are now written to the
# pod's local disk (/root/r2_ckpt) and copied once into a NEW directory on /workspace, then verified
# (num_prompts_trained_on == S and S position records). Reason: with checkpoints written straight to /workspace,
# the trainer's periodic rewrite of experts_K8_checkpoint.pt lost its final version for C shard 1 (the file kept
# the 16-prompt state; mtime 09:28 vs shard end 09:38), so C_64a had 40 prompts instead of 48.
# Usage (pod, tmux): RPP=32 S=24 bash scripts/r2_refit_9b_v2.sh rest
set -euo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/r1_jpp.py"
M=Qwen/Qwen3.5-9B; L=4,8,12,16,20,24,28
S=${S:-24}; NS=4; RPP=${RPP:-32}; CKE=${CKE:-8}; K=8; ITEMS=fit:0
OUT=$R/runs/r2_9b; CSV=$R/data/model_correctness.csv; CACHE=/root/r2_readout_resid_cache_9b.pt
pool() { case $1 in C) echo $R/runs/r1_2b/prompts_natural.json;; A) echo $R/runs/r1_2b/prompts_raw.json;; esac; }
skip() { case $1 in C) echo 1;; A) echo 16;; esac; }
shard() {  # c i
  local c=$1 i=$2 d=$OUT/$1; local t0=$(date +%s); local loc=/root/r2_ckpt/$c
  rm -rf $loc
  local ck
  ck=$($W fit-shard --hf-model-name $M --layers $L --router-path "$d/router_K$K.pt" --num-prompts $((S*NS)) \
       --num-shards $NS --shard-idx $i --prompts-json "$(pool $c)" --skip-first-n-positions "$(skip $c)" \
       --jacobian-rows-per-pass $RPP --checkpoint-every-n-prompts $CKE \
       --artifacts-base-dir $loc --checkpoint-name shards | tail -1)
  local dst=$d/ckpt2/shard${i}of$NS; mkdir -p $(dirname $dst); rm -rf $dst; cp -r "$ck" $dst
  $PY -c "
import torch,json,sys
c=torch.load('$dst/experts_K8_checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)['config']
n=json.load(open('$dst/position_records.json'))
print('verify $c shard $i: num_prompts_trained_on',c['num_prompts_trained_on'],'records',len(n))
sys.exit(0 if c['num_prompts_trained_on']==$S==len(n) else 1)"
  rm -rf $loc
  echo "$dst" > $d/shard2_$i.dir
  echo "shard cond=$c tag=shards(v2) idx=$i/$NS n=$((S*NS)) rpp=$RPP s=$(( $(date +%s)-t0 ))" | tee -a $d/timing_v2.txt
}
sdir() { [ -f $OUT/$1/shard2_$2.dir ] && cat $OUT/$1/shard2_$2.dir || cat $OUT/$1/shards_$2.dir; }
weights() {  # c name experts_path
  local c=$1 name=$2 ex=$3 d=$OUT/$1/$2; mkdir -p $d
  $W fit-weights --hf-model-name $M --layers $L --experts "$ex" --checkpoint-name "jpp_$name" \
     --items $ITEMS --correctness-csv "$CSV" --cache-path "$CACHE" --out "$d/jpp_$name.pt"
}
build() {  # c name shard_idx...
  local c=$1 name=$2; shift 2; local d=$OUT/$c/$name; mkdir -p $d; local t0=$(date +%s)
  local dirs=(); for i in "$@"; do dirs+=("$(sdir $c $i)"); done
  $W merge-experts --checkpoint-dirs "${dirs[@]}" --num-clusters $K --checkpoint-name "experts_$name" \
     --out "$d/experts_K$K.pt" --pooled-lens-out "$d/pooled_$name.pt"
  local t1=$(date +%s)
  weights $c $name "$d/experts_K$K.pt"
  echo "build $name shards=$* merge_s=$((t1-t0)) weights_s=$(( $(date +%s)-t1 )) dirs=${dirs[*]}" | tee -a $d/timing.txt
}
case $1 in
  rest)
    [ -f $OUT/C/shard2_2.dir ] || shard C 2
    [ -f $OUT/A/shard2_2.dir ] || shard A 2
    [ -f $OUT/C/shard2_1.dir ] || shard C 1      # redo (see header)
    [ -f $OUT/C/shard2_3.dir ] || shard C 3
    [ -f $OUT/A/shard2_3.dir ] || shard A 3
    build C C_64a_r 0 1; build C C_full 0 1 2 3; build C C_64b 2 3
    build A A_full 0 1 2 3; build A A_64b 2 3
    weights A A_64a_w2 $OUT/A/A_64a/experts_K$K.pt     # repeat of A_64a's weight fit: determinism check
    touch $OUT/rest_done ;;
  shard) shard $2 $3 ;;
  build) build "${@:2}" ;;
esac
