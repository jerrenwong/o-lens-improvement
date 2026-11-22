#!/bin/bash
# rerun of the 27B control evals that failed on a transient tokenizer read (network FS); retries any failure 3x.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; L=/root/rb1; cd $R
mv $L/errors $L/errors_first_27b_ctrl 2>/dev/null; rm -f $L/FAILED $L/DONE
CV="shufall priorall shuf300 prior300 shuf3000 prior3000 shufall_atL priorall_atL shuf300_atL prior300_atL shuf3000_atL prior3000_atL"
for v in r300_0 r3000_0 rall; do for k in 1 4 16 64 256; do CV="$CV ${v}_k$k"; done; done
one() { for a in 1 2 3; do bash scripts/rb1/one_eval.sh "$@" && return 0; sed -i "/eval $3\$/d" $L/errors 2>/dev/null; sleep 30; done; echo "eval $3" >> $L/errors; }
export -f one; export L
for v in $CV; do echo "27b rb1_$v rb1_${v}_27b"; done | xargs -P 3 -L 1 bash -c 'one "$@"' _
$PY scripts/rb1/rb1_items.py 27b > $L/items_27b.log 2>&1
$PY scripts/rb1/rb1_items.py 2b > $L/items_2b.log 2>&1
if [ -s $L/errors ]; then cp $L/errors $L/FAILED; else rm -f $L/errors; touch $L/DONE; fi
