#!/bin/bash
# sub2 chain (run in tmux sub2 on wong-olens2): label -> fit2 -> analyze -> dev benchmark of 5 candidates.
# Writes /root/sub2/DONE or /root/sub2/FAILED (with reason).
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
L=/root/sub2
rm -f $L/DONE $L/FAILED
fail() { echo "$(date +%T) $1" > $L/FAILED; echo "FAILED: $1" >> $L/chain.log; exit 1; }
echo "$(date +%T) start" >> $L/chain.log
if [ ! -f .hf27/sub2/al2_ranks.pt ] || [ "$FORCE_LABEL" = 1 ]; then
  if [ ! -f $L/test/label_ok ]; then
    mkdir -p $L/test
    $PY scripts/sub2/label.py --n 200 --out $L/test > $L/test_label.log 2>&1
    tail -3 $L/test_label.log | grep -q LABEL_DONE || fail "label.py test (see test_label.log)"
    $PY -c "
import json; c = json.load(open('scripts/sub2/results/pilot_check_test.json'))
assert c['n_compared'] >= 90 and c['bin_equal'] >= 0.97 * c['n_compared'], c" >> $L/chain.log 2>&1 || fail "label test: pilot disagreement"
    touch $L/test/label_ok
    echo "$(date +%T) label test ok" >> $L/chain.log
  fi
  $PY scripts/sub2/label.py >> $L/label.log 2>&1
  tail -3 $L/label.log | grep -q LABEL_DONE || fail "label.py (see label.log)"
fi
echo "$(date +%T) label ok" >> $L/chain.log
$PY scripts/sub2/fit2.py > $L/fit2.log 2>&1
grep -q FIT2_DONE $L/fit2.log || fail "fit2.py (see fit2.log)"
echo "$(date +%T) fit2 ok" >> $L/chain.log
$PY scripts/sub2/analyze.py > $L/analyze.log 2>&1
grep -q ANALYZE_DONE $L/analyze.log || fail "analyze.py (see analyze.log)"
echo "$(date +%T) analyze ok" >> $L/chain.log
for c in sub2_ALL sub2_SUB sub2_INF sub2_RSUB0 sub2_RINF0; do
  ok=0
  for t in 1 2 3 4 5 6 7 8; do
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge 27000 ] && break
      echo "$(date +%T) $c wait free $free" >> $L/chain.log; sleep 60
    done
    $PY auto/evaluate.py --lens auto/candidates/$c.py --name ${c}_9b > $L/bench_$c.log 2>&1
    if grep -q "^n_items" $L/bench_$c.log; then ok=1; break; fi
    echo "$(date +%T) $c failed try $t" >> $L/chain.log; sleep 60
  done
  [ $ok = 1 ] || fail "bench $c"
  echo "$(date +%T) bench $c ok" >> $L/chain.log
done
echo "$(date +%T) all done" > $L/DONE
echo "$(date +%T) DONE" >> $L/chain.log
