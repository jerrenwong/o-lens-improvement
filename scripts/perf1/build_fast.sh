#!/bin/bash
# perf1: install causal-conv1d into the ISOLATED venv /workspace/o-lens-improvement/.venv_fast only
# (it sees the jpp_lens venv via a .pth file; nothing is written to /root/jpp_lens/.venv).
set -x
R=/workspace/o-lens-improvement
export UV_CACHE_DIR=$R/.uvcache_perf1 TMPDIR=$R/.tmp_perf1 PIP_CACHE_DIR=$R/.uvcache_perf1/pip
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH MAX_JOBS=6 CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=9.0
P=$R/.venv_fast/bin/python
md5sum /root/jpp_lens/.venv/lib/python3.12/site-packages/torch/__init__.py > /root/perf1/jpp_venv_before.md5
ls -la --time-style=full-iso /root/jpp_lens/.venv/lib/python3.12/site-packages | md5sum >> /root/perf1/jpp_venv_before.md5
uv pip install --python $P setuptools wheel ninja packaging
time uv pip install --python $P --no-deps --no-build-isolation -v causal-conv1d
cd /tmp && $P -c "import causal_conv1d, transformers.utils.import_utils as iu; print('cc1d', causal_conv1d.__version__, causal_conv1d.__file__, iu.is_causal_conv1d_available(), iu.is_flash_linear_attention_available())"
ls -la --time-style=full-iso /root/jpp_lens/.venv/lib/python3.12/site-packages | md5sum
cat /root/perf1/jpp_venv_before.md5
echo BUILD_DONE
