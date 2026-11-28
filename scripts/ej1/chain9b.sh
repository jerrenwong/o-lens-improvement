cd /workspace/o-lens-improvement
$PY scripts/ej1/ej1_fit4.py --model Qwen/Qwen3.5-9B --n 256 --every 64 --targets T3 --suffixes none --rpp 64 --out /workspace/o-lens-improvement/.hf27/ej1/9b256
$PY scripts/ej1/ej1_fit4.py --model Qwen/Qwen3.5-9B --n 256 --every 64 --targets T3 --suffixes ie --rpp 32 --out /workspace/o-lens-improvement/.hf27/ej1/9b256
