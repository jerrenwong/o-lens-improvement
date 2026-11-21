#!/bin/bash
# w1: fit-weights runs (reproduction checks + generated-item fits). Pod, tmux. Usage: bash w1_run.sh <step>
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; W="$PY $R/scripts/w1_jpp2.py"
M=Qwen/Qwen3.5-9B; L=4,8,12,16,20,24,28; O=$R/runs/w1
A=$R/runs/r2_9b/A/A_full/experts_K8.pt; C=$R/runs/r2_9b/C/C_full/experts_K8.pt
GC=$R/.hf27/w1/resid_genword_9b.pt
run() {  # retry on CUDA OOM, wait for >= 30 GB free
  for try in 1 2 3 4 5 6; do
    until [ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge 30000 ]; do sleep 30; done
    "$@" > /root/w1_cur.log 2>&1; rc=$?
    cat /root/w1_cur.log
    if [ $rc -eq 0 ]; then return 0; fi
    if grep -q "OutOfMemory\|out of memory" /root/w1_cur.log; then echo "[w1] OOM, retry $try"; sleep 120; else return $rc; fi
  done; return 1
}
bench() {  # experts fitter out
  run $W --w1-fitter $2 fit-weights --hf-model-name $M --layers $L --experts $1 --checkpoint-name $(basename $3 .pt) \
    --items fit:0 --correctness-csv $R/data/model_correctness.csv --cache-path /root/r2_readout_resid_cache_9b.pt --out $3
}
gen() {  # experts split fitter out
  run $W --w1-split $2 --w1-fitter $3 fit-weights --hf-model-name $M --layers $L --experts $1 \
    --checkpoint-name $(basename $4 .pt) --eval-data-dir $R/data/evals_gen --cache-path $GC --out $4
}
mkdir -p $O/check
case $1 in
  check2)
    bench $A chunked $O/check/jpp_Afull_fit0_chunk2.pt
    gen $A W-a chunked $O/jpp_Afull_Wa.pt ;;
  check)
    bench $A original $O/check/jpp_Afull_fit0_orig.pt
    bench $A chunked $O/check/jpp_Afull_fit0_chunk.pt
    gen $A W-a original $O/check/jpp_Afull_Wa_orig.pt
    gen $A W-a chunked $O/jpp_Afull_Wa.pt ;;
  fits)
    gen $A W-train chunked $O/jpp_Afull_Wtrain.pt
    gen $A W-b chunked $O/jpp_Afull_Wb.pt
    gen $A W-c chunked $O/jpp_Afull_Wc.pt
    gen $C W-train chunked $O/jpp_Cfull_Wtrain.pt ;;
  held)
    run $W --w1-split W-held evaluate --hf-model-name $M --layers $L --eval-data-dir $R/data/evals_gen \
      --lens $R/runs/r2_9b/A/A_full/pooled_A_full.pt $R/runs/r2_9b/A/A_full/jpp_A_full.pt $O/jpp_Afull_Wtrain.pt \
             $R/runs/r2_9b/C/C_full/pooled_C_full.pt $R/runs/r2_9b/C/C_full/jpp_C_full.pt $O/jpp_Cfull_Wtrain.pt \
      --ks 1,10 --out $O/heldout_ranks.csv ;;
esac
echo "[w1] step $1 done rc=$?"
