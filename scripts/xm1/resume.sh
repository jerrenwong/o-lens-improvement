#!/bin/bash
# xm1: resume after an OOM caused by running too many jobs at once: J-lens (concurrent with the running s5 collection
# only), then the two map fits and DEV scoring, one at a time.   Usage: bash scripts/xm1/resume.sh <model> <tag> <layers>
M=$1; T=$2; LAY=$3; R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; X=/root/xm1
export HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH=$R/src
cd /root; rm -f $X/FAILED_$T $X/jlens_failed_$T
fail() { echo "$1" > $X/FAILED_$T; exit 1; }
[ -f $X/jlens_done_$T ] || bash $R/scripts/xm1/fit_jpp.sh $M $T $LAY jlens > $X/jlens_$T.log 2>&1 || fail jlens
until [ -f $R/.hf27/xm1/acts_${T}_s5.pt ] && ! tmux has-session -t xm1col5 2>/dev/null; do [ -f $X/collect_s5_failed_$T ] && fail collect_s5; sleep 30; done
[ -f $R/.hf27/xm1/E_${T}_t3.pt ] || $PY $R/scripts/xm1/fit_map.py $M $T > $X/fitmap_$T.log 2>&1 || fail fitmap
[ -f $R/.hf27/xm1/E_${T}_s5.pt ] || Q=s5 $PY $R/scripts/xm1/fit_map.py $M $T > $X/fitmap_s5_$T.log 2>&1 || fail fitmap_s5
bash $R/scripts/xm1/score_dev.sh $M $T
touch $X/chain_done_$T
