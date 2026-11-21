#!/bin/bash
# al3 chain (run on the pod in tmux session al3chain). Resumable: finished stages leave /root/al3/ok_<stage>.
# GPU stages wait for the gate (no ask3* tmux session, >= 30 GB free); exit code 3 = yielded to ask3 -> wait + rerun.
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
D=/root/al3
LOG=$D/chain.log
FITDEV=${FITDEV:-cuda}
rm -f $D/DONE $D/FAILED
say() { echo "$(date +%T) $*" >> $LOG; }
gate() {
  local n=0
  while true; do
    if ! tmux ls 2>/dev/null | grep -q '^ask3'; then
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge 30000 ] && return 0
      [ $((n % 10)) -eq 0 ] && say "gate: free $free MiB"
    else
      [ $((n % 10)) -eq 0 ] && say "gate: ask3 session present"
    fi
    n=$((n + 1)); sleep 60
  done
}
stage() {  # stage <name> <gpu 0|1> <done-marker> <cmd...>
  local name=$1 gpu=$2 marker=$3; shift 3
  [ -f $D/ok_$name ] && { say "$name already done"; return 0; }
  local fails=0
  while true; do
    [ "$gpu" = 1 ] && gate
    say "$name start"
    "$@" >> $D/$name.log 2>&1
    rc=$?
    if grep -q "$marker" $D/$name.log && [ $rc -eq 0 ]; then touch $D/ok_$name; say "$name done"; return 0; fi
    if [ $rc -eq 3 ]; then say "$name yielded to ask3"; sleep 60; continue; fi
    fails=$((fails + 1)); say "$name failed rc=$rc ($fails)"
    [ $fails -ge 4 ] && { say "$name FAILED"; echo "$name" > $D/FAILED; exit 1; }
    sleep 120
  done
}
stage gen 1 GEN_DONE $PY scripts/al3/gen.py
stage build 0 BUILD_DONE env CUDA_VISIBLE_DEVICES= $PY scripts/al3/build.py
stage label 1 LABEL_DONE $PY scripts/al3/label.py
stage collect 1 COLLECT_DONE $PY scripts/al3/collect.py
stage bins 0 BINS_DONE env CUDA_VISIBLE_DEVICES= $PY scripts/al3/bins.py
if [ "$FITDEV" = cuda ]; then
  stage fit 1 FIT_DONE $PY scripts/al3/fit.py --dev cuda
else
  stage fit 0 FIT_DONE env CUDA_VISIBLE_DEVICES= $PY scripts/al3/fit.py --dev cpu
fi
for m in a c b; do
  stage bench_$m 1 "^n_items" $PY auto/evaluate.py --lens auto/candidates/al3_$m.py --name al3_${m}_9b
done
say "ALL DONE"
touch $D/DONE
