#!/bin/bash
# ao1 refit of the Jacobian toward the asked answer on 9B (pod, tmux ao1fit). Same recipe as scripts/ja1 (A2/A5):
# chat prompt, rlens rules, K=1 + K=8 sums from the same backward passes, routers of runs/ja1/9b/routers,
# stock merge-experts + fit-weights on fit:0. Snapshots at 64, 96 (SNAPS of scripts/ao1/ao1_fit.py).
#   bash fit.sh <name> <wording topic|unstated> <n> [stages: fit post]
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=Qwen/Qwen3.5-9B; L=4,8,12,16,20,24,28; CACHE=/root/r2_readout_resid_cache_9b.pt; RPP=${RPP:-16}
NAME=$1; W=$2; N=$3; shift 3
O=$R/runs/ao1/9b; LG=/root/ao1; mkdir -p $O/lens $LG
RT=$R/runs/ja1/9b/routers
F="$PY $R/scripts/ao1/ao1_fit.py"; P="$PY $R/scripts/ja1/ja1_post.py"; WJ="$PY $R/scripts/r1_jpp.py"
CSV=$R/data/model_correctness.csv
ck() { ls -d $O/$NAME/*/ja1 2>/dev/null | head -1; }
at() { local f; f=$(ck); [ -z "$f" ] && return 1; if [ $1 = $N ]; then echo $f; else [ -d ${f}_at$1 ] && echo ${f}_at$1; fi; }
fit() {
  [ -f $O/$NAME/fit_info.json ] && return 0
  JA1_FLA=0 bash $R/scripts/ja1/retry.sh 50000 $LG/fit_$NAME.log $F fit --model $M --wording $W --lrp rlens \
    --routers $RT/router_${W}_K1.pt $RT/router_${W}_K8.pt --n $N --rpp $RPP --out $O/$NAME
}
a5() {  # lens-name ckptdir
  local n=$1 d=$2
  [ -f $O/lens/$n.pt ] && return 0
  (cd /root/jpp_lens && $WJ merge-experts --checkpoint-dirs $d --num-clusters 8 --checkpoint-name ao1_$n \
     --out $O/lens/${n}_experts_K8.pt --device cpu) >> $LG/a5_$n.log 2>&1 || return 1
  (cd /root/jpp_lens && $WJ fit-weights --hf-model-name $M --layers $L --experts $O/lens/${n}_experts_K8.pt \
     --checkpoint-name ao1_$n --items fit:0 --correctness-csv $CSV --cache-path $CACHE --out $O/lens/${n}_jpp.pt) >> $LG/a5_$n.log 2>&1 || return 1
  $P lensfile --model $M --src $O/lens/${n}_jpp.pt --out $O/lens/$n.pt >> $LG/a5_$n.log 2>&1
  $P counts --dir $d --K 8 > $O/lens/${n}_counts.json
}
post() {
  for n in 64 96 $N; do
    d=$(at $n) || { echo "no state at $n" >> $LG/post_$NAME.log; continue; }
    [ -f $O/lens/${NAME}K1_n$n.pt ] || $P pooled --K 1 --dir $d --out $O/lens/${NAME}K1_n$n.pt > $O/lens/${NAME}K1_n$n.json 2>>$LG/post_$NAME.log
    a5 ${NAME}A5_n$n $d
  done
}
ST="${*:-fit post}"
for s in $ST; do $s || { echo "$NAME $s" >> $LG/FAILED_fit; exit 1; }; done
touch $LG/fitdone_$NAME
