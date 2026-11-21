#!/bin/bash
# w1 step 4: score lens files on the 9B dev split (one at a time; wait for >= 30 GB free; retry on OOM).
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; O=$R/runs/w1; cd $R
one() {  # name lens
  for try in 1 2 3 4 5 6; do
    until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 30000 ]; do sleep 30; done
    R2_LENS=$2 $PY auto/evaluate.py --lens auto/candidates/r2_lensfile.py --name $1 > /root/w1_dev_$1.log 2>&1
    if grep -q "^---" /root/w1_dev_$1.log && ! grep -q CRASH /root/w1_dev_$1.log; then break; fi
    grep -q "out of memory\|OutOfMemory" /root/w1_dev_$1.log || break
    echo "OOM retry $try $1"; sleep 120
  done
  echo "== $1"; sed -n '/^---/,$p' /root/w1_dev_$1.log
}
one w1_Afull_Wtrain $O/jpp_Afull_Wtrain.pt
one w1_Afull_Wa $O/jpp_Afull_Wa.pt
one w1_Afull_Wb $O/jpp_Afull_Wb.pt
one w1_Afull_Wc $O/jpp_Afull_Wc.pt
one w1_Cfull_Wtrain $O/jpp_Cfull_Wtrain.pt
one w1_pooled_A_full $R/runs/r2_9b/A/A_full/pooled_A_full.pt
one w1_pooled_C_full $R/runs/r2_9b/C/C_full/pooled_C_full.pt
one w1_Afull_Wa_orig $O/check/jpp_Afull_Wa_orig.pt
echo W1_DEV_DONE
