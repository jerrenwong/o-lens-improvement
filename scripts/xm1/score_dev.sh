#!/bin/bash
# xm1 step 6/7 (pod, tmux): DEV scores with auto/evaluate_x.py (--lite), bootstrap, and v4 dry runs (no scoring).
# Usage: bash scripts/xm1/score_dev.sh <model> <tag>
set -uo pipefail
M=$1; T=$2
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/xm1/dev_$T; mkdir -p $L
export HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH=$R/src
cd $R
for c in jlens jppu L Lshuf Lprior alone; do
  $PY auto/evaluate_x.py --lens auto/candidates/xm1_$c.py --model $M --split dev --lite --name xm1_${T}_$c > $L/$c.log 2>&1 \
    || echo "FAILED $c" >> $L/failed
done
$PY auto/evaluate_x.py --lens auto/lens.py --model $M --split dev --lite --name xm1_${T}_jppraw > $L/jppraw.log 2>&1 || echo "FAILED jppraw" >> $L/failed
# PREREG_v5 Amendment 1: the frozen A-lens rows (auto/candidates/alens.py), maps of this model via env
for b in J0 JPP0 none J JPP; do
  ALENS_BASE=$b ALENS_E=$R/.hf27/xm1/E_${T}_t3.pt ALENS_E2=$R/.hf27/xm1/E_${T}_s5.pt \
    $PY auto/evaluate_x.py --lens auto/candidates/alens.py --model $M --split dev --lite --name xm1_${T}_alens_$b > $L/alens_$b.log 2>&1 \
    || echo "FAILED alens $b" >> $L/failed
done
# lite == full check on one lens
$PY auto/evaluate_x.py --lens auto/candidates/xm1_jppu.py --model $M --split dev --name xm1_${T}_jppu_full > $L/jppu_full.log 2>&1 || echo "FAILED jppu_full" >> $L/failed
p="xm1_${T}"
$PY scripts/r2_boot.py ${p}_L-${p}_jppu ${p}_L-${p}_Lshuf ${p}_L-${p}_Lprior ${p}_jppu-${p}_jlens ${p}_alone-${p}_jppu \
  ${p}_alens_JPP-${p}_alens_JPP0 ${p}_alens_J-${p}_alens_J0 ${p}_alens_JPP-${p}_L ${p}_alens_JPP0-${p}_jppu > $L/boot.txt 2>&1
for c in jppu L Lshuf Lprior; do
  $PY auto/evaluate_x_v4.py --lens auto/candidates/xm1_$c.py --model $M --split test --lite --dry-run --name xm1v4dry_${T}_$c > $L/v4dry_$c.log 2>&1 \
    || echo "FAILED v4dry $c" >> $L/failed
done
for b in J0 JPP0 none J JPP; do
  ALENS_BASE=$b ALENS_E=$R/.hf27/xm1/E_${T}_t3.pt ALENS_E2=$R/.hf27/xm1/E_${T}_s5.pt \
    $PY auto/evaluate_x_v4.py --lens auto/candidates/alens.py --model $M --split test --lite --dry-run --name xm1v4dry_${T}_alens_$b > $L/v4dry_alens_$b.log 2>&1 \
    || echo "FAILED v4dry alens $b" >> $L/failed
done
touch $L/done
