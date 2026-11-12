#!/bin/bash
# Build the GPU-pod environment. Pod-local /root is wiped on restart, so re-run after one.
# Idempotent. Usage (on the pod): bash /workspace/o-lens-improvement/scripts/setup_pod.sh
set -euo pipefail
JPP_COMMIT=d7efc0de372e20589e899188c0cfe8434acc63ec   # safety-research/jpp_lens, pinned
REPO=/workspace/o-lens-improvement
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d /root/jpp_lens/.git ] || git clone -q https://github.com/safety-research/jpp_lens.git /root/jpp_lens
git -C /root/jpp_lens checkout -q "$JPP_COMMIT"
cd /root/jpp_lens
uv sync --extra cu128 --frozen                      # torch 2.11 cu128, transformers 5.14.1 (their lock)
# Fast Gated-DeltaNet kernels for Qwen3.5 linear-attention layers (else a slow torch fallback).
# tilelang: fla refuses its Triton backward on Hopper with Triton < 3.7.1 (wrong grads, fla #640).
uv pip install --python .venv/bin/python flash-linear-attention tilelang
# This repo on the path via a .pth file (an editable build fails on /workspace: no chmod/rename).
echo "$REPO/src" > "$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')/olens.pth"
.venv/bin/python -c "import torch, transformers, fla, olens; print('ok', torch.__version__, transformers.__version__, torch.cuda.get_device_name())"
