#!/bin/bash
# Re-export the viewer's fit-split rankings (J++, A-lens alone, J++ + A-lens) with all filter levels incl. f4.
R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R
export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do tag=$([ "$m" = Qwen/Qwen3.5-9B ] && echo 9b || echo 2b)
  for b in none JPP JPP0; do
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt 18000 ]; do sleep 30; done
    echo "=== $m $b"; ALENS_BASE=$b $PY auto/evaluate.py --lens auto/candidates/fitA_dump.py --model $m --lite --name fitAv_${tag}_${b} 2>&1 | grep -E "wrote|Error|Traceback" || true
  done
done; echo "viewer dump done $(date -u +%H:%M)"
