#!/bin/bash
# retry.sh NEED_MIB LOG cmd... : wait for free GPU memory, run, retry on OOM (up to 20 times)
need=$1; lf=$2; shift 2
free_mib() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1 | awk -F', ' '{print $1-$2}'; }
for a in $(seq 1 20); do
  while [ "$(free_mib)" -lt "$need" ]; do sleep 30; done
  echo "[ja1] $(date +%T) attempt $a: $*" >> $lf
  "$@" >> $lf 2>&1 && { echo "exit=0" >> $lf; exit 0; }
  tail -30 $lf | grep -q "OutOfMemory\|out of memory" || { echo "exit=1" >> $lf; exit 1; }
  sleep 60
done
echo "exit=1" >> $lf; exit 1
