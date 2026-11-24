#!/bin/bash
# w3: Jacobian toward the asked answer, same recipe as scripts/ao1/fit.sh (which produced topA5_n128 / topK1_n128):
# ao1_fit.py unchanged (chat prompt, W0 wording only, rlens rules, K=1 + K=8 sums from the same backward passes,
# routers of runs/ja1/<tag>/routers for that wording, seed-20261121 passage order, in-place checkpoint writes on
# /workspace, resume slots every 16 prompts), then stock merge-experts + fit-weights on fit:0 (-> <NAME>A5_n<n>) and
# the label-free pooled matrix (-> <NAME>K1_n<n>). Memory capped by scripts/w3/fitwrap.py (W3_MEM_GB, default 45).
#   bash scripts/w3/fit.sh <2b|9b> <name> <topic|unstated> <n> [--wrong] ; stages via env ST (default "fit verify post")
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; NAME=$2; W=$3; N=$4; shift 4; EXTRA="$*"
case $TAG in
  2b) M=Qwen/Qwen3.5-2B; L=3,6,9,12,15,18,21; CACHE=/root/r1_readout_resid_cache.pt ;;
  9b) M=Qwen/Qwen3.5-9B; L=4,8,12,16,20,24,28; CACHE=/root/r2_readout_resid_cache_9b.pt ;;
esac
RPP=${RPP:-16}; NEED=${NEED:-47000}
O=$R/runs/w3/$TAG; LG=$O/logs; mkdir -p $O/lens $LG
RT=$R/runs/ja1/$TAG/routers
F="$PY $R/scripts/w3/fitwrap.py"; P="$PY $R/scripts/ja1/ja1_post.py"; WJ="$PY $R/scripts/r1_jpp.py"
CSV=$R/data/model_correctness.csv
ck() { ls -d $O/$NAME/*/ja1 2>/dev/null | head -1; }
at() { local f; f=$(ck); [ -z "$f" ] && return 1; if [ $1 = $N ]; then echo $f; else [ -d ${f}_at$1 ] && echo ${f}_at$1; fi; }
fit() {
  [ -f $O/$NAME/fit_info.json ] && return 0
  JA1_FLA=0 bash $R/scripts/ja1/retry.sh $NEED $LG/fit_$NAME.log $F fit --model $M --wording $W --lrp rlens \
    --routers $RT/router_${W}_K1.pt $RT/router_${W}_K8.pt --n $N --rpp $RPP --out $O/$NAME $EXTRA
}
verify() {  # final checkpoint must hold all N prompts (guards against a lost in-place rewrite)
  local d; d=$(ck) || return 1
  $PY -c "
import torch,sys
for K in (1,8):
    st=torch.load('$d/experts_K%d_checkpoint.pt'%K,map_location='cpu',weights_only=True)
    n=st['config']['num_prompts_trained_on']; print('K%d prompts'%K,n,'records',len(st['position_records']))
    assert n==$N, n
" >> $LG/verify_$NAME.log 2>&1
}
a5() {
  local n=$1 d=$2
  [ -f $O/lens/$n.pt ] && return 0
  (cd /root/jpp_lens && $WJ merge-experts --checkpoint-dirs $d --num-clusters 8 --checkpoint-name w3_$n \
     --out $O/lens/${n}_experts_K8.pt --device cpu) >> $LG/a5_$n.log 2>&1 || return 1
  (cd /root/jpp_lens && $WJ fit-weights --hf-model-name $M --layers $L --experts $O/lens/${n}_experts_K8.pt \
     --checkpoint-name w3_$n --items fit:0 --correctness-csv $CSV --cache-path $CACHE --out $O/lens/${n}_jpp.pt) >> $LG/a5_$n.log 2>&1 || return 1
  $P lensfile --model $M --src $O/lens/${n}_jpp.pt --out $O/lens/$n.pt >> $LG/a5_$n.log 2>&1
  $P counts --dir $d --K 8 > $O/lens/${n}_counts.json
}
post() {
  for n in ${POSTN:-64 $N}; do
    d=$(at $n) || { echo "no state at $n" >> $LG/post_$NAME.log; continue; }
    [ -f $O/lens/${NAME}K1_n$n.pt ] || $P pooled --K 1 --dir $d --out $O/lens/${NAME}K1_n$n.pt > $O/lens/${NAME}K1_n$n.json 2>>$LG/post_$NAME.log
    a5 ${NAME}A5_n$n $d || return 1
  done
}
for s in ${ST:-fit verify post}; do $s || { echo "$NAME $s FAILED" >> $LG/FAILED; exit 1; }; done
echo "$NAME done $(date -u +%FT%TZ)" >> $LG/DONE
