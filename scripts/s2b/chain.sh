#!/bin/bash
# s2b: collect 2B unstated-word targets, format check (2B + 9B reference), fit E_2b_s5, then five DEV rows of alens.py.
PY=/root/jpp_lens/.venv/bin/python
R=/workspace/o-lens-improvement
L=/root/s2b
mkdir -p $L
cd $R
export HF_HUB_OFFLINE=1 BS=${BS:-64} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY scripts/s2b/s2b_collect.py Qwen/Qwen3.5-2B chat > $L/collect.log 2>&1 || { echo COLLECT > $L/FAILED; exit 1; }
$PY scripts/s2b/s2b_fmt.py Qwen/Qwen3.5-2B .hf27/ask5/acts_2b.pt > $L/fmt2b.log 2>&1 || echo "fmt2b failed" >> $L/errors
$PY scripts/s2b/s2b_fmt.py Qwen/Qwen3.5-9B .hf27/ask5/acts_9b.pt > $L/fmt9b.log 2>&1 || echo "fmt9b failed" >> $L/errors
$PY scripts/s2b/s2b_fit.py Qwen/Qwen3.5-2B > $L/fit.log 2>&1 || { echo FIT > $L/FAILED; exit 1; }
for row in J0 JPP0 none J JPP; do
  n=D_al_${row}_2b
  ALENS_BASE=$row $PY auto/evaluate.py --model Qwen/Qwen3.5-2B --lens auto/candidates/alens.py --name $n --lite > $L/$n.log 2>&1 || echo "eval $row failed" >> $L/errors
  echo "$n $(grep -E 'chosen lam|pass10:|pass1:|n_items' $L/$n.log | sed 's/  */ /g' | tr '\n' ' ')" >> $L/results.txt
done
echo OK > $L/DONE
