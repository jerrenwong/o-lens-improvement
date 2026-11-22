#!/bin/bash
# xm1 step 3 (pod, tmux): J++ and J-lens for a new model with the authors' unmodified pipeline (jpp_cli via
# scripts/r1_jpp.py, which only hides `fla` for fit-shard and logs the kept positions), default recipe:
#   router: first 1000 WikiText prompts (runs/r1_2b/prompts_raw.json = jpp_cli.load_fit_prompts(1000)), K, skip 16, seed 0
#   shard : first 64 of the same pool (jpp_cli default --num-prompts 64), skip 16, max_seq_len 128, target = final block
#   J++   : --lrp-mode rlens, K=8, merge-experts, fit-weights on fit:0 (correctness CSV data/model_correctness_xm1.csv)
#   J-lens: --lrp-mode none,  K=1, merge-experts; the lens = the pooled lens of that fit
# Only batching/storage settings differ from the CLI defaults: --jacobian-rows-per-pass $RPP (batching only) and
# --checkpoint-every-n-prompts 1000 (one final checkpoint write: /workspace rename-over quirk, see r2_refit_9b_v2.sh).
# Usage: bash scripts/xm1/fit_jpp.sh <model> <tag> <layers csv> jpp|jlens|weights
set -euo pipefail
M=$1; T=$2; LAY=$3; WHAT=$4
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/r1_jpp.py"
export HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1
RPP=${RPP:-32}; NP=64; NR=1000; P=$R/runs/r1_2b/prompts_raw.json
D=$R/runs/xm1/$T; B=$R/.hf27/xm1/fit_$T; mkdir -p $D $B
CSV=$R/data/model_correctness_xm1.csv; CACHE=$B/readout_resid_cache.pt
verify() {  # ckdir K
  $PY -c "
import torch,json,sys
c=torch.load('$1/experts_K$2_checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)['config']
n=json.load(open('$1/position_records.json'))
print('verify: num_prompts_trained_on',c['num_prompts_trained_on'],'records',len(n))
sys.exit(0 if c['num_prompts_trained_on']==$NP==len(n) else 1)"
}
fit() {  # name K mode
  local name=$1 K=$2 mode=$3; local t0=$(date +%s)
  $W --r1-router-prompts-json $P fit-router --hf-model-name $M --layers $LAY --num-prompts $NR --num-clusters $K \
     --skip-first-n-positions 16 --seed 0 --out $B/router_${name}_K$K.pt
  local t1=$(date +%s)
  local ck
  ck=$($W fit-shard --hf-model-name $M --layers $LAY --router-path $B/router_${name}_K$K.pt --num-prompts $NP \
       --prompts-json $P --skip-first-n-positions 16 --lrp-mode $mode --jacobian-rows-per-pass $RPP \
       --checkpoint-every-n-prompts 1000 --artifacts-base-dir $B/ckpt_$name --checkpoint-name $name | tail -1)
  local t2=$(date +%s)
  verify $ck $K
  $W merge-experts --checkpoint-dirs $ck --num-clusters $K --checkpoint-name experts_$name \
     --out $B/experts_${name}_K$K.pt --pooled-lens-out $D/pooled_$name.pt
  local t3=$(date +%s)
  echo "$name K=$K lrp=$mode N=$NP rpp=$RPP router_s=$((t1-t0)) shard_s=$((t2-t1)) s_per_prompt=$(( (t2-t1)/NP )) merge_s=$((t3-t2))" | tee -a $D/timing.txt
}
case $WHAT in
  jpp) fit jpp 8 rlens; touch /root/xm1/jpp_experts_done_$T ;;
  jlens) fit jlens 1 none; touch /root/xm1/jlens_done_$T ;;
  weights)
    t0=$(date +%s)
    $W fit-weights --hf-model-name $M --layers $LAY --experts $B/experts_jpp_K8.pt --checkpoint-name jpp_$T \
       --items fit:0 --correctness-csv $CSV --cache-path $CACHE --out $D/jpp_$T.pt
    echo "weights_s=$(( $(date +%s)-t0 ))" | tee -a $D/timing.txt; touch /root/xm1/jpp_done_$T ;;
esac
