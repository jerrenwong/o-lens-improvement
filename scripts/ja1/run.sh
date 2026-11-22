#!/bin/bash
# ja1 fits + post (pod, tmux ja1*). Waits for free GPU memory, retries on OOM. Never touches other sessions.
#   bash run.sh <model 2b|9b> <stage>...
set -uo pipefail
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; C=/root/ja1/code
F="$PY $C/ja1_fit.py"; P="$PY $C/ja1_post.py"; W="$PY $R/scripts/r1_jpp.py"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=$1; shift
case $TAG in
  2b) M=Qwen/Qwen3.5-2B; L=3,6,9,12,15,18,21; CACHE=/root/r1_readout_resid_cache.pt; RPP=${RPP:-16}; NEED=${NEED:-24000} ;;
  9b) M=Qwen/Qwen3.5-9B; L=4,8,12,16,20,24,28; CACHE=/root/r2_readout_resid_cache_9b.pt; RPP=${RPP:-16}; NEED=${NEED:-45000} ;;
esac
O=$R/runs/ja1/$TAG; LOG=/root/ja1/$TAG; mkdir -p $O $LOG
CSV=$R/data/model_correctness.csv
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}'; }
waitfree() { while [ "$(free_mib)" -lt "$1" ]; do echo "[ja1] $(date +%T) waiting: free $(free_mib) < $1"; sleep 60; done; }
retry() {  # logfile cmd...
  local lf=$1; shift
  for a in $(seq 1 40); do
    waitfree $NEED
    echo "[ja1] $(date +%T) start (attempt $a): $*" >> $lf
    "$@" >> $lf 2>&1 && return 0
    if tail -50 $lf | grep -q "OutOfMemory\|out of memory"; then echo "[ja1] OOM, retry in 3 min" >> $lf; sleep 180; else return 1; fi
  done; return 1; }
ck() { ls -d $O/$1/*/ja1 2>/dev/null | head -1; }   # final checkpoint dir of a fit

routers() { for w in topic unstated; do [ -f $O/routers/router_${w}_K8.pt ] || retry $LOG/router_$w.log $F router --model $M --wording $w --nr 1000 --out $O/routers || return 1; done; }
fit() {  # name n lrp wording [--wrong] ; routers K1 (+K8 for rlens)
  local name=$1 n=$2 lrp=$3 w=$4; shift 4
  [ -n "$(ck $name)" ] && [ -f $O/$name/fit_info.json ] && { echo "[ja1] $name exists"; return 0; }
  local rw=$w; case $w in topic1|topic2) rw=topic;; esac
  local rs="$O/routers/router_${rw}_K1.pt"; [ $lrp = rlens ] && [ $rw = $w ] && rs="$rs $O/routers/router_${rw}_K8.pt"
  ls $O/$name/*/ja1_resume.json >/dev/null 2>&1 || rm -rf $O/$name   # keep a resumable fit
  JA1_FLA=$([ $lrp = none ] && echo 1 || echo 0) retry $LOG/fit_$name.log $F fit --model $M --wording $w --lrp $lrp --routers $rs --n $n --rpp $RPP --out $O/$name "$@"
}
jstd() {  # standard J-lens recipe, 64 prompts (snapshot 32 -> split halves)
  [ -f $O/std/fit_info.json ] && return 0; rm -rf $O/std
  JA1_FLA=1 retry $LOG/fit_std.log $F jstd --model $M --lrp none --routers $O/routers/router_topic_K1.pt --n 64 --rpp $RPP --out $O/std
}
a5() {  # name ckptdir -> merged experts + fit-weights (stock stages) + plain dict
  local name=$1 d=$2; mkdir -p $O/lens
  [ -f $O/lens/$name.pt ] && return 0
  (cd /root/jpp_lens && $W merge-experts --checkpoint-dirs $d --num-clusters 8 --checkpoint-name ja1_$name \
     --out $O/lens/${name}_experts_K8.pt --device cpu) >> $LOG/a5_$name.log 2>&1 || return 1
  waitfree 20000
  (cd /root/jpp_lens && $W fit-weights --hf-model-name $M --layers $L --experts $O/lens/${name}_experts_K8.pt \
     --checkpoint-name ja1_$name --items fit:0 --correctness-csv $CSV --cache-path $CACHE --out $O/lens/${name}_jpp.pt) >> $LOG/a5_$name.log 2>&1 || return 1
  $P lensfile --model $M --src $O/lens/${name}_jpp.pt --out $O/lens/$name.pt >> $LOG/a5_$name.log 2>&1
  $P counts --dir $d --K 8 > $O/lens/${name}_counts.json
}
"${@:-true}" 2>&1
