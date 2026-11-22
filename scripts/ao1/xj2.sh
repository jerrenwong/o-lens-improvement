#!/bin/bash
# ao1 XJ variants: fits (FIT only) then DEV evals. bash xj.sh <9b|2b> <suffix for ask matrices, e.g. n40>  [ASK dir]
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; S=$2; AD=${3:-$R/runs/ja1/$TAG/lens}; H=$R/.hf27; LG=/root/ao1/xj; mkdir -p $LG
case $TAG in
  9b) JE=$R/runs/r2_9b/A/A_64a/experts_K8.pt; REL=$R/artifacts/baselines/jpp9b_prefit.pt; MAP=$H/st1/E_9b_p3v_t10.pt; WRONG= ;;
  2b) JE=$R/runs/r1_2b/A/experts_K8.pt; REL=$R/artifacts/baselines/jpp_prefit.pt; MAP=$H/st1/E_2b_p3v_t10.pt; WRONG=$AD/A4_$S.pt ;;
esac
A2=$AD/A2_$S.pt; AE=$AD/A5_${S}_experts_K8.pt
[ -f $A2 ] || A2=$AD/${S%%_*}K1_${S#*_}.pt   # refit naming: topK1_n128 / topA5_n128_experts_K8
[ -f $AE ] || AE=$AD/${S%%_*}A5_${S#*_}_experts_K8.pt
fit() { local n=$1; shift; [ -f runs/ao1/$TAG/xj/$n.pt ] && return 0
  local a=(); for s in "$@"; do a+=(--src $s); done
  bash scripts/ja1/retry.sh 9000 $LG/fit_${TAG}_$n.log $PY scripts/ao1/xj_fit.py --model $TAG --name $n "${a[@]}"; }
K=${S}
fit J0 exp:$JE
fit J1_$K exp:$JE mat:$A2
fit J1c_$K exp:$JE perm:mat:$A2
[ -n "$WRONG" ] && fit J1w_$K exp:$JE mat:$WRONG
fit J2_$K exp:$JE exp:$AE
fit J2c_$K exp:$JE perm:exp:$AE
fit J3a_$K mat:$REL mat:$A2
fit J3ac_$K mat:$REL perm:mat:$A2
fit J3b_$K mat:$REL exp:$AE
fit J3bc_$K mat:$REL perm:exp:$AE
fit J4_$K exp:$JE exp:$AE map:$MAP
fit J4c_$K exp:$JE perm:exp:$AE map:$MAP
F=runs/ao1/xj_${TAG}_$K.txt; : > $F
for p in runs/ao1/$TAG/xj/*.pt; do n=$(basename $p .pt); case $n in J0|*_$K) echo "ao1_${TAG}_xj_$n $TAG xj $R/$p -" >> $F;; esac; done
bash scripts/ao1/ev2.sh $F
touch $LG/done_${TAG}_$K
