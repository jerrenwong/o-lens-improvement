#!/bin/bash
# dp1 fits (pod, tmux dp1*). Low priority: waits for free GPU memory, retries on OOM.
#   bash scripts/dp1/run.sh smoke | plain2b | plain9b | jpp2b | all
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
W="$PY /root/dp1/code/dp1_jpp.py"; STOCK="$PY $R/scripts/r1_jpp.py"
PJ=$R/runs/r1_2b/prompts_raw.json; OUT=$R/runs/dp1; LOG=/root/dp1
mkdir -p $OUT $LOG
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}'; }
waitfree() {  # need_mib
  while [ "$(free_mib)" -lt "$1" ]; do echo "[dp1] $(date +%T) waiting: free $(free_mib) MiB < $1"; sleep 120; done; }
retry() {  # need_mib logfile cmd...
  local need=$1 lf=$2; shift 2
  for a in 1 2 3 4 5 6 7 8 9 10 11 12; do
    waitfree $need
    echo "[dp1] $(date +%T) start (attempt $a, free $(free_mib) MiB): $*" | tee -a $lf
    "$@" >> $lf 2>&1 && return 0
    if grep -q "OutOfMemory\|out of memory" $lf; then echo "[dp1] OOM, retry in 5 min" | tee -a $lf; sleep 300; else return 1; fi
  done; return 1; }

L2=3,6,9,12,15,18,21; L9=4,8,12,16,20,24,28
smoke() {
  local d=/root/dp1/smoke; rm -rf $d; mkdir -p $d
  [ -f $OUT/2b_plain/router_K1.pt ] || router2b
  retry 47000 $LOG/smoke.log $W fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers $L2 --router-path $OUT/2b_plain/router_K1.pt \
    --num-prompts 2 --prompts-json $PJ --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass 64 \
    --checkpoint-every-n-prompts 16 --artifacts-base-dir $d/dp1 --checkpoint-name s || return 1
  retry 47000 $LOG/smoke.log $STOCK fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers $L2 --router-path $OUT/2b_plain/router_K1.pt \
    --num-prompts 2 --prompts-json $PJ --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass 64 \
    --checkpoint-every-n-prompts 16 --artifacts-base-dir $d/stock --checkpoint-name s
}
confirm() {  # dp1 V0 vs stock jpp_cli (fla present, as dp1 plain fits), 2 prompts
  local d=/root/dp1/smoke
  waitfree ${NEED2:-47000}
  (cd /root/jpp_lens && $PY scripts/jpp_cli.py fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers $L2 --router-path $OUT/2b_plain/router_K1.pt \
    --num-prompts 2 --prompts-json $PJ --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass ${RPP2:-64} \
    --checkpoint-every-n-prompts 16 --artifacts-base-dir $d/stockfla --checkpoint-name s) > $LOG/confirm.log 2>&1
  $PY - $(ls -d $d/stockfla/*/s/shard0of1) $(ls -d $d/dp1/*/s/shard0of1) <<'PY' | tee $LOG/confirm.txt
import torch,sys
L=lambda p: torch.load(p+"/experts_K1_checkpoint.pt",weights_only=True)["layer_sums"]
a,b=L(sys.argv[1]),L(sys.argv[2])
for l in a: print(l,"dp1 V0 == stock(fla)", torch.equal(a[l]["weighted_jacobian_row_sum_EFN"],b[l]["weighted_jacobian_row_sum_EFN"]))
PY
}
router2b() {
  mkdir -p $OUT/2b_plain
  retry 20000 $LOG/router2b.log $STOCK --r1-router-prompts-json $PJ fit-router --hf-model-name Qwen/Qwen3.5-2B --layers $L2 \
    --num-prompts 64 --num-clusters 1 --skip-first-n-positions 16 --seed 0 --out $OUT/2b_plain/router_K1.pt
}
plain2b() {
  [ -f $OUT/2b_plain/router_K1.pt ] || router2b
  retry ${NEED2:-47000} $LOG/plain2b.log $W fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers $L2 --router-path $OUT/2b_plain/router_K1.pt \
    --num-prompts 64 --prompts-json $PJ --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass ${RPP2:-64} \
    --checkpoint-every-n-prompts 16 --artifacts-base-dir $OUT/2b_plain/ckpt --checkpoint-name fit
}
plain9b() {  # waits for the ask6 chain to finish (it needs 56 GB for its 27B stages). recipe of runs/v6_9b (router reused; layers = readout layers; snapshot at 48 = stored jlens9b)
  mkdir -p $OUT/9b_plain
  while [ ! -f /root/ask6/DONE ] && [ ! -f /root/ask6/FAILED ]; do sleep 300; done
  retry ${NEED9:-50000} $LOG/plain9b.log $W --dp1-snapshot 48 fit-shard --hf-model-name Qwen/Qwen3.5-9B --layers $L9 \
    --router-path $R/runs/v6_9b/fit_router.pt --num-prompts 64 --prompts-json $PJ --skip-first-n-positions 16 \
    --lrp-mode none --jacobian-rows-per-pass ${RPP9:-64} --checkpoint-every-n-prompts 16 \
    --artifacts-base-dir $OUT/9b_plain/ckpt --checkpoint-name fit
}
jpp2b() {  # recipe of runs/r1_2b/A (raw pool, skip 16, 128 prompts, K8 router of A, lrp rlens, fla hidden)
  mkdir -p $OUT/2b_jpp
  retry ${NEED2:-47000} $LOG/jpp2b.log $W --dp1-hide-fla fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers $L2 \
    --router-path $R/runs/r1_2b/A/router_K8.pt --num-prompts 128 --prompts-json $PJ --skip-first-n-positions 16 \
    --jacobian-rows-per-pass ${RPP2:-64} --checkpoint-every-n-prompts 4 --artifacts-base-dir $OUT/2b_jpp/ckpt --checkpoint-name fit
}
for s in "$@"; do
  case $s in
    all) plain2b && jpp2b && plain9b ;;
    *) $s ;;
  esac || { echo "[dp1] stage $s FAILED" | tee -a $LOG/status; }
done
