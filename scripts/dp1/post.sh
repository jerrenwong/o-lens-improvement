#!/bin/bash
# dp1 post-processing (pod): variants -> stock merge-experts (+ stock fit-weights for J++) -> dev evals -> FIT diag.
#   bash scripts/dp1/post.sh plain2b | plain9b | jpp2b
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; STOCK="$PY $R/scripts/r1_jpp.py"
OUT=$R/runs/dp1; LOG=/root/dp1; CSV=$R/data/model_correctness.csv
VS="V0 V1 V2 V2a V2b V3 V4 V5"
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}'; }
waitfree() { while [ "$(free_mib)" -lt "$1" ]; do sleep 120; done; }
ev() {  # model name lensfile
  waitfree 6000
  (cd $R && DP1_LENS=$3 DP1_MODEL=$1 $PY auto/evaluate.py --lens $R/scripts/dp1/cand_lensfile.py --name $2 --model $1 --lite \
     > $LOG/eval_$2.log 2>&1) || echo "[dp1] eval $2 FAILED" | tee -a $LOG/status
  tail -n 9 $LOG/eval_$2.log
}
run() {  # set model K
  local set=$1 M=$2 K=$3 d=$OUT/$1 src; src=$(ls -d $OUT/$1/ckpt/*/fit/shard0of1 | tail -1)
  $PY $R/scripts/dp1/variants.py $src $d/var $K > $LOG/variants_$set.log 2>&1 || { echo "variants $set FAILED" | tee -a $LOG/status; return 1; }
  for V in $VS; do
    waitfree 4000
    $STOCK merge-experts --checkpoint-dirs $d/var/$V --num-clusters $K --checkpoint-name dp1_${set}_$V \
      --out $d/var/$V/experts_K$K.pt --pooled-lens-out $d/var/$V/pooled.pt >> $LOG/merge_$set.log 2>&1 || echo "merge $set $V FAILED" | tee -a $LOG/status
  done
}
weights() {  # set model K cache layers
  local set=$1 M=$2 K=$3 d=$OUT/$1
  for V in $VS; do
    waitfree 9000
    $STOCK fit-weights --hf-model-name $M --layers $5 --experts $d/var/$V/experts_K$K.pt --checkpoint-name jpp_dp1_$V \
      --items fit:0 --correctness-csv $CSV --cache-path $4 --out $d/var/$V/jpp.pt >> $LOG/weights_$set.log 2>&1 \
      || echo "weights $set $V FAILED" | tee -a $LOG/status
  done
}
case $1 in
  plain2b)
    run 2b_plain Qwen/Qwen3.5-2B 1
    for V in $VS; do ev Qwen/Qwen3.5-2B dp1_2bJ_$V $OUT/2b_plain/var/$V/pooled.pt; done
    ev Qwen/Qwen3.5-2B dp1_2bJ_released $R/artifacts/baselines/jlens_prefit.pt
    waitfree 9000
    $PY $R/scripts/dp1/diag.py --model Qwen/Qwen3.5-2B --out $OUT/2b_plain/diag.json \
      $(for V in $VS; do echo "$V=$OUT/2b_plain/var/$V/pooled.pt"; done) released=$R/artifacts/baselines/jlens_prefit.pt \
      > $LOG/diag_2b_plain.log 2>&1 || echo "diag 2b_plain FAILED" | tee -a $LOG/status ;;
  plain9b)
    run 9b_plain Qwen/Qwen3.5-9B 1
    src=$(ls -d $OUT/9b_plain/ckpt/*/fit/shard0of1_at48 | tail -1)
    $STOCK merge-experts --checkpoint-dirs $src --num-clusters 1 --checkpoint-name dp1_9b_V0at48 \
      --out $OUT/9b_plain/V0at48_experts.pt --pooled-lens-out $OUT/9b_plain/V0at48_pooled.pt >> $LOG/merge_9b_plain.log 2>&1
    for V in $VS; do ev Qwen/Qwen3.5-9B dp1_9bJ_$V $OUT/9b_plain/var/$V/pooled.pt; done
    ev Qwen/Qwen3.5-9B dp1_9bJ_stored $R/runs/v6_9b/jlens9b.pt
    waitfree 30000
    $PY $R/scripts/dp1/diag.py --model Qwen/Qwen3.5-9B --out $OUT/9b_plain/diag.json \
      $(for V in $VS; do echo "$V=$OUT/9b_plain/var/$V/pooled.pt"; done) stored=$R/runs/v6_9b/jlens9b.pt \
      > $LOG/diag_9b_plain.log 2>&1 || echo "diag 9b_plain FAILED" | tee -a $LOG/status ;;
  jpp2b)
    run 2b_jpp Qwen/Qwen3.5-2B 8
    weights 2b_jpp Qwen/Qwen3.5-2B 8 /root/r1_readout_resid_cache.pt 3,6,9,12,15,18,21
    for V in $VS; do ev Qwen/Qwen3.5-2B dp1_2bJPP_$V $OUT/2b_jpp/var/$V/jpp.pt; done
    ev Qwen/Qwen3.5-2B dp1_2bJPP_r1A $R/runs/r1_2b/A/jpp_A.pt
    ev Qwen/Qwen3.5-2B dp1_2bJPP_released $R/artifacts/baselines/jpp_prefit.pt
    waitfree 9000
    $PY $R/scripts/dp1/diag.py --model Qwen/Qwen3.5-2B --out $OUT/2b_jpp/diag.json \
      $(for V in $VS; do echo "$V=$OUT/2b_jpp/var/$V/jpp.pt"; done) r1A=$R/runs/r1_2b/A/jpp_A.pt \
      released=$R/artifacts/baselines/jpp_prefit.pt > $LOG/diag_2b_jpp.log 2>&1 || echo "diag 2b_jpp FAILED" | tee -a $LOG/status ;;
esac
$PY /root/dp1/code/v0check.py $1 > $LOG/v0check_$1.log 2>&1 || echo "v0check $1 FAILED" | tee -a $LOG/status
echo "[dp1] post $1 done $(date +%T)" | tee -a $LOG/status
