#!/bin/bash
# xm1 chain v3 = chain2.sh with the env (HF_HOME, offline) passed explicitly into the tmux sessions it starts
# collection (Q=s5) and J-lens fit (concurrent); after J++ experts -> J++ weights; topic map fit; s5 map fit; DEV scoring.
# Usage: bash scripts/xm1/chain2.sh <model> <tag> <layers csv>
M=$1; T=$2; LAY=$3; R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; X=/root/xm1
export HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH=$R/src
fail() { echo "$1" > $X/FAILED_$T; exit 1; }
until [ -f $R/.hf27/xm1/acts_$T.pt ] && ! tmux has-session -t xm1col 2>/dev/null; do [ -f $X/collect_failed_$T ] && fail collect; sleep 30; done
[ -f $R/.hf27/xm1/acts_${T}_s5.pt ] || tmux has-session -t xm1col5 2>/dev/null || tmux new -d -s xm1col5 "cd /root && HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 Q=s5 $PY $R/scripts/xm1/collect.py $M $T $LAY > $X/collect_s5_$T.log 2>&1 || touch $X/collect_s5_failed_$T"
tmux has-session -t xm1jl 2>/dev/null || [ -f $X/jlens_done_$T ] || tmux new -d -s xm1jl "HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 bash $R/scripts/xm1/fit_jpp.sh $M $T $LAY jlens > $X/jlens_$T.log 2>&1 || touch $X/jlens_failed_$T"
until [ -f $X/jpp_experts_done_$T ]; do [ -f $X/jpp_failed_$T ] && fail jpp; sleep 30; done
[ -f $X/jpp_done_$T ] || bash $R/scripts/xm1/fit_jpp.sh $M $T $LAY weights > $X/weights_$T.log 2>&1 || fail weights
[ -f $R/.hf27/xm1/E_${T}_t3.pt ] || $PY $R/scripts/xm1/fit_map.py $M $T > $X/fitmap_$T.log 2>&1 || fail fitmap
until [ -f $R/.hf27/xm1/acts_${T}_s5.pt ] && ! tmux has-session -t xm1col5 2>/dev/null; do [ -f $X/collect_s5_failed_$T ] && fail collect_s5; sleep 30; done
[ -f $R/.hf27/xm1/E_${T}_s5.pt ] || Q=s5 $PY $R/scripts/xm1/fit_map.py $M $T > $X/fitmap_s5_$T.log 2>&1 || fail fitmap_s5
until [ -f $X/jlens_done_$T ]; do [ -f $X/jlens_failed_$T ] && fail jlens; sleep 30; done
bash $R/scripts/xm1/score_dev.sh $M $T
touch $X/chain_done_$T
