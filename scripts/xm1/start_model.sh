#!/bin/bash
# xm1: whole pipeline for one model (pod, tmux): info + LRP check, then benchmark and J++ experts (concurrent), topic
# collection after the benchmark, then chain3.sh (s5 collection, J-lens, weights, map fits, DEV scoring).
# Usage: bash scripts/xm1/start_model.sh <model> <tag> <layers csv>
M=$1; T=$2; LAY=$3; R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; X=/root/xm1
export HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 PYTHONPATH=$R/src
cd /root; mkdir -p $R/runs/xm1/$T
$PY $R/scripts/xm1/info.py $M $R/runs/xm1/$T/info.json > $X/info_$T.log 2>&1 || { echo info > $X/FAILED_$T; exit 1; }
$PY $R/scripts/xm1/lrp_check.py $M $T $LAY > $X/lrp_$T.log 2>&1 || { echo lrp_check > $X/FAILED_$T; exit 1; }
tmux new -d -s xm1jpp_$T "HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 bash $R/scripts/xm1/fit_jpp.sh $M $T $LAY jpp > $X/jpp_$T.log 2>&1 || touch $X/jpp_failed_$T"
bash $R/scripts/xm1/bench.sh $M $T > $X/benchchain_$T.log 2>&1 || { echo bench > $X/FAILED_$T; exit 1; }
tmux new -d -s xm1col "cd /root && HF_HOME=$R/.hf27 HF_HUB_OFFLINE=1 $PY $R/scripts/xm1/collect.py $M $T $LAY > $X/collect_$T.log 2>&1 || touch $X/collect_failed_$T"
bash $R/scripts/xm1/chain3.sh $M $T $LAY > $X/chain3_$T.log 2>&1
