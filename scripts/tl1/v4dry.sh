#!/bin/bash
# tl1: rerun only the v4 dry-run steps of run_model.sh for one model (after the tl1_lib v4 item-loading fix).
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; T=$1; M=$2
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ "${3:-}" = hf27 ] && export HF_HOME=$R/.hf27
cd $R; L=/root/tl1/$T; TE=$R/.hf27/tl1/T_${T}_t3.pt; TE2=$R/.hf27/tl1/T_${T}_s5.pt; rm -f $L/failed_v4
env TLENS_BASE=JPP TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t_v4.py --lens auto/candidates/tlens.py \
  --model $M --split test --dry-run --verify-fwd --name tl1v4dry_fwd_$T > $L/v4dry_fwd.log 2>&1 || echo "FAILED v4dry_fwd" >> $L/failed_v4
for b in none J JPP; do
  env TLENS_BASE=$b TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t_v4.py --lens auto/candidates/tlens.py \
    --model $M --split test --lite --dry-run --name tl1v4dry_${b}_$T > $L/v4dry_$b.log 2>&1 || echo "FAILED v4dry_$b" >> $L/failed_v4
done
touch $L/v4done
