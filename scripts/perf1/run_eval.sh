#!/bin/bash
# perf1: run one command when enough GPU memory is free; record wall time and the process's peak GPU
# memory (nvidia-smi, sampled every 0.5 s; includes the CUDA context). Retries up to 3 times on OOM.
#   bash scripts/perf1/run_eval.sh <tag> <need_gb> <cmd...>
# Log /root/perf1/<tag>.log, result line /root/perf1/<tag>.res (and scripts/perf1/results/timing.tsv).
tag=$1; need=$2; shift 2
cd /workspace/o-lens-improvement
mkdir -p /root/perf1 scripts/perf1/results
log=/root/perf1/$tag.log
free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
for attempt in 1 2 3; do
  while [ "$(free_gb)" -lt "$need" ]; do sleep 20; done
  echo "[perf1] start $(date -u +%FT%TZ) free=$(free_gb)GB attempt=$attempt: $*" > $log
  t0=$(date +%s.%N)
  "$@" >> $log 2>&1 &
  pid=$!
  peak=0
  while kill -0 $pid 2>/dev/null; do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$pid '$1==p{print $2}')
    # the python process may be a child of $pid (env wrapper); include children
    for c in $(pgrep -P $pid); do
      mc=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p=$c '$1==p{print $2}')
      [ -n "$mc" ] && m=$((${m:-0} + mc))
    done
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 0.5
  done
  wait $pid; rc=$?
  t1=$(date +%s.%N)
  if [ $rc -ne 0 ] && grep -q "OutOfMemoryError\|CUDA out of memory" $log; then
    echo "[perf1] OOM, retrying after 120 s" >> $log; sleep 120; continue
  fi
  break
done
wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f", b-a}')
echo -e "$tag\trc=$rc\twall_s=$wall\tpeak_gpu_mib=$peak" | tee /root/perf1/$tag.res >> scripts/perf1/results/timing.tsv
echo "[perf1] done rc=$rc wall=$wall peak_mib=$peak" >> $log
