#!/bin/bash
# dq1 chain (tmux dq1 on wong-olens2): dq1_fit -> dq1_analyze -> candidates for the 2 best variants (validation)
# -> waits for the sub2 chain to finish -> dev benchmark. Writes /root/sub2/DQ1_DONE or /root/sub2/DQ1_FAILED.
cd /workspace/o-lens-improvement
PY=/root/jpp_lens/.venv/bin/python
L=/root/sub2
rm -f $L/DQ1_DONE $L/DQ1_FAILED
fail() { echo "$(date +%T) $1" > $L/DQ1_FAILED; echo "FAILED: $1" >> $L/dq1_chain.log; exit 1; }
echo "$(date +%T) start" >> $L/dq1_chain.log
$PY scripts/sub2/dq1_fit.py > $L/dq1_fit.log 2>&1
grep -q DQ1_FIT_DONE $L/dq1_fit.log || fail "dq1_fit.py"
$PY scripts/sub2/dq1_analyze.py > $L/dq1_analyze.log 2>&1
grep -q DQ1_ANALYZE_DONE $L/dq1_analyze.log || fail "dq1_analyze.py"
echo "$(date +%T) fit+analyze ok" >> $L/dq1_chain.log
CANDS=$($PY -c "
import json; b = json.load(open('scripts/sub2/results/dq1.json'))['best2']
for v in b:
    n = 'sub2_dq1_' + v.replace('.', 'p')
    open(f'auto/candidates/{n}.py', 'w').write(f'''\"\"\"sub2/dq1 ({v}): unit(J++ h) + lam * unit(M h + b); M = al2 M3 map refitted by scripts/sub2/dq1_fit.py,
variant {v} (see that file). Chosen as one of the two best dq1 variants by validation words. lam on the FIT split as al2_T.\"\"\"
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sub2_lib import build_sub2  # noqa: E402


def build(ctx):
    return build_sub2(ctx, \"dq1_{v}\")
''')
    print(n)
") || fail "candidate generation"
echo "$(date +%T) candidates: $CANDS" >> $L/dq1_chain.log
while [ ! -f $L/DONE ] && [ ! -f $L/FAILED ]; do sleep 60; done
for c in $CANDS; do
  ok=0
  for t in 1 2 3 4 5 6 7 8; do
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge 27000 ] && break
      sleep 60
    done
    $PY auto/evaluate.py --lens auto/candidates/$c.py --name ${c}_9b > $L/bench_$c.log 2>&1
    if grep -q "^n_items" $L/bench_$c.log; then ok=1; break; fi
    echo "$(date +%T) $c failed try $t" >> $L/dq1_chain.log; sleep 60
  done
  [ $ok = 1 ] || fail "bench $c"
  echo "$(date +%T) bench $c ok" >> $L/dq1_chain.log
done
echo "$(date +%T) all done" > $L/DQ1_DONE
echo "$(date +%T) DONE" >> $L/dq1_chain.log
