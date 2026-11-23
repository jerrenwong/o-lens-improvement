#!/bin/bash
# tl1 (PREREG_v4 Addendum F), pod/tmux, per model: fit the T maps; reproduction check of the new evaluator with the
# frozen A-lens (lite, must equal the existing DEV runs); full-model forward verification of the feature token ids
# (FIT + DEV rows); DEV rows T alone / J + T / J++ + T (lite); paired bootstrap; v4 dry runs (no ranking).
# Usage: bash scripts/tl1/run_model.sh <tag> [<tag> ...]
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $R
for T in "$@"; do
  L=/root/tl1/$T; mkdir -p $L; rm -f $L/failed
  unset HF_HOME
  case $T in
    9b) M=Qwen/Qwen3.5-9B;; 2b) M=Qwen/Qwen3.5-2B;;
    27b) M=Qwen/Qwen3.6-27B; export HF_HOME=$R/.hf27;;
    olmo7b) M=allenai/Olmo-3-7B-Instruct; export HF_HOME=$R/.hf27;;
    gemma4e4b) M=google/gemma-4-E4B-it; export HF_HOME=$R/.hf27;;
  esac
  case $T in
    9b|2b|27b) AE=$R/.hf27/ask3/E_${T}_t3.pt; AE2=$R/.hf27/ask5/E_${T}_s5.pt; ref() { echo D_al_${1}_$T; };;
    *) AE=$R/.hf27/xm1/E_${T}_t3.pt; AE2=$R/.hf27/xm1/E_${T}_s5.pt; ref() { echo xm1_${T}_alens_$1; };;
  esac
  TE=$R/.hf27/tl1/T_${T}_t3.pt; TE2=$R/.hf27/tl1/T_${T}_s5.pt
  step() { n=$1; shift; "$@" > $L/$n.log 2>&1 || { echo "FAILED $n" >> $L/failed; return 1; }; }
  if [ ! -f $TE2 ] || [ ! -f $R/runs/tl1/fit_$T.json ]; then
    step fit $PY scripts/tl1/tl1_fit.py $T || { echo "fit failed $T" >> /root/tl1/errors; continue; }
  fi
  # 1. reproduction through auto/evaluate_t.py (candidates that ignore the feature), lite as the original runs
  for b in JPP0 J0 JPP; do
    step repro_$b env ALENS_BASE=$b ALENS_E=$AE ALENS_E2=$AE2 $PY auto/evaluate_t.py --lens auto/candidates/alens.py \
      --model $M --split dev --lite --name tl1repro_${b}_$T
  done
  # 2. full model: forward verification of every FIT and DEV row's token ids / readout position (+ lite == full)
  step fwd env TLENS_BASE=JPP0 TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t.py --lens auto/candidates/tlens.py \
    --model $M --split dev --verify-fwd --name tl1fwd_JPP0_$T
  $PY scripts/tl1/compare_runs.py tl1repro_JPP0_$T $(ref JPP0) tl1repro_J0_$T $(ref J0) tl1repro_JPP_$T $(ref JPP) \
    tl1fwd_JPP0_$T $(ref JPP0) > $L/repro.txt 2>&1
  # 3. DEV rows of the T-lens
  for b in none J JPP; do
    step dev_$b env TLENS_BASE=$b TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t.py --lens auto/candidates/tlens.py \
      --model $M --split dev --lite --name D_tl_${b}_$T
  done
  $PY scripts/r2_boot.py $(ref JPP)-D_tl_JPP_$T D_tl_JPP_$T-$(ref JPP0) $(ref J)-D_tl_J_$T \
    D_tl_J_$T-$(ref J0) D_tl_none_$T-$(ref none) > $L/boot.txt 2>&1 || echo "FAILED boot" >> $L/failed
  # 4. v4 dry runs (build + transport of every v4 row, nothing ranked): full model with forward verification of the
  #    v4 rows for the JPP row, lite for the three rows
  step v4dry_fwd env TLENS_BASE=JPP TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t_v4.py --lens auto/candidates/tlens.py \
    --model $M --split test --dry-run --verify-fwd --name tl1v4dry_fwd_$T
  for b in none J JPP; do
    step v4dry_$b env TLENS_BASE=$b TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t_v4.py --lens auto/candidates/tlens.py \
      --model $M --split test --lite --dry-run --name tl1v4dry_${b}_$T
  done
  touch $L/done
done
