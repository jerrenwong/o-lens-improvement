#!/bin/bash
# t1 "content targets" vs recipe on Qwen3.5-2B: 4 shards x 32 prompts per condition, interleaved A0 T0 A1 T1 ...,
# then merges (full / half a = shards 0+1 / half b = shards 2+3) -> pooled + J++ lenses.
# Run on the pod from a /root copy:  bash /root/t1/t1_run.sh [shards|build|all]
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; LOC=/root/t1
W="$PY $LOC/t1_jpp.py --t1-rule $LOC/content_rule.json"
M=Qwen/Qwen3.5-2B; L=3,6,9,12,15,18,21; K=8; RPP=64; ITEMS=fit:0
OUT=$R/runs/t1_2b; PJ=$R/runs/r1_2b/prompts_raw.json
CSV=$R/data/model_correctness.csv; CACHE=/root/r1_readout_resid_cache.pt
ROUTER=$LOC/router_K8.pt          # = runs/r1_2b/A/router_K8.pt (recipe router: raw pool, 1000 prompts, K8, seed 0, skip 16)
mkdir -p $OUT/shards $OUT/lenses $OUT/logs
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
total() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1; }
acquire() {
  while true; do
    if [ "$(used)" -lt 15000 ] && (set -o noclobber; echo t1 > /root/gpu27.lock) 2>/dev/null; then
      echo "[t1] $(date +%T) lock acquired (used $(used) MiB)"; return; fi
    sleep 60
  done
}
release() {
  if [ "$(cat /root/gpu27.lock 2>/dev/null)" = t1 ]; then rm -f /root/gpu27.lock; echo "[t1] $(date +%T) lock released"; fi
}
waitfree() { while [ $(( $(total) - $(used) )) -lt 10000 ]; do sleep 60; done; }
shard() {  # cond idx
  local c=$1 i=$2 mode; [ $c = T ] && mode=content || mode=all
  local t0=$(date +%s)
  $W --t1-mode $mode --t1-log $LOC/poslog_$c$i.json fit-shard --hf-model-name $M --layers $L --router-path $ROUTER \
     --num-prompts 128 --num-shards 4 --shard-idx $i --prompts-json $PJ --skip-first-n-positions 16 \
     --jacobian-rows-per-pass $RPP --artifacts-base-dir $LOC/ckpt_$c --checkpoint-name t1_$c > $OUT/logs/shard_$c$i.log 2>&1
  local rc=$? t1=$(date +%s)
  local d; d=$(ls -d $LOC/ckpt_$c/*/t1_$c/shard${i}of4 | tail -1)
  mkdir -p $OUT/shards/$c$i
  for f in experts_K8_checkpoint.pt position_records.json fit_shard_stamp.json; do cat $d/$f > $OUT/shards/$c$i/$f; done
  cat $LOC/poslog_$c$i.json > $OUT/shards/$c$i/t1_poslog.json
  local m1 m2; m1=$(md5sum < $d/experts_K8_checkpoint.pt); m2=$(md5sum < $OUT/shards/$c$i/experts_K8_checkpoint.pt)
  $PY -c "
import torch,sys; s=torch.load('$OUT/shards/$c$i/experts_K8_checkpoint.pt',map_location='cpu',weights_only=True)
n=s['config']['num_prompts_trained_on']; r=len(s['position_records']); print('[t1] verify $c$i prompts',n,'records',r,'md5 ok' if '$m1'=='$m2' else 'MD5 MISMATCH')
sys.exit(0 if n==32 and r==32 and '$m1'=='$m2' else 1)" && rm -rf $d || echo "[t1] VERIFY FAILED $c$i (kept $d)"
  echo "shard=$c$i rc=$rc shard_s=$((t1-t0))" | tee -a $OUT/timing.txt
}
build() {  # cond name shardidx...
  local c=$1 n=$2; shift 2; local dirs=""
  for i in "$@"; do dirs="$dirs $OUT/shards/$c$i"; done
  local t0=$(date +%s)
  $PY $LOC/t1_jpp.py merge-experts --checkpoint-dirs $dirs --num-clusters $K --checkpoint-name experts_${c}_$n \
     --out $LOC/experts_${c}_$n.pt --pooled-lens-out $LOC/pooled_${c}_$n.pt > $OUT/logs/merge_${c}_$n.log 2>&1
  $PY $LOC/t1_jpp.py fit-weights --hf-model-name $M --layers $L --experts $LOC/experts_${c}_$n.pt --checkpoint-name jpp_${c}_$n \
     --items $ITEMS --correctness-csv $CSV --cache-path $CACHE --out $LOC/jpp_${c}_$n.pt > $OUT/logs/weights_${c}_$n.log 2>&1
  local t1=$(date +%s)
  for f in $LOC/pooled_${c}_$n* $LOC/jpp_${c}_$n*; do cat $f > $OUT/lenses/$(basename $f); done
  [ $n = full ] && for f in $LOC/experts_${c}_$n*; do cat $f > $OUT/lenses/$(basename $f); done
  for f in pooled jpp; do [ "$(md5sum < $LOC/${f}_${c}_$n.pt)" = "$(md5sum < $OUT/lenses/${f}_${c}_$n.pt)" ] && echo "[t1] copied ${f}_${c}_$n ok" || echo "[t1] COPY MISMATCH ${f}_${c}_$n"; done
  rm -f $LOC/experts_${c}_$n* $LOC/pooled_${c}_$n* $LOC/jpp_${c}_$n*
  echo "build=${c}_$n merge+weights_s=$((t1-t0))" | tee -a $OUT/timing.txt
}
what=${1:-all}
if [ $what = shards ] || [ $what = all ]; then
  for i in 0 1 2 3; do
    [ -f $OUT/shards/T$i/experts_K8_checkpoint.pt ] && continue
    if [ $((i % 2)) = 0 ] || [ "$(cat /root/gpu27.lock 2>/dev/null)" != t1 ]; then acquire; fi
    [ -f $OUT/shards/A$i/experts_K8_checkpoint.pt ] || shard A $i
    shard T $i
    if [ $((i % 2)) = 1 ]; then release; [ $i -lt 3 ] && sleep 95; fi
  done
  release
fi
if [ $what = build ] || [ $what = all ]; then
  waitfree
  for c in A T; do build $c full 0 1 2 3; build $c ha 0 1; build $c hb 2 3; done
fi
echo "[t1] ALL DONE $(date +%T)"
