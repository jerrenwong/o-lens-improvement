#!/bin/bash
# perf1: build causal-conv1d in a throwaway LOCAL build venv (/root/perf1/venv_build; /workspace rejects the
# chmod that installers do), then unpack the wheel (zipfile, no chmod) into the ISOLATED
# /workspace/o-lens-improvement/.venv_fast. Both venvs see the jpp_lens venv read-only through a .pth file;
# nothing is written to /root/jpp_lens/.venv.
set -x
R=/workspace/o-lens-improvement
mkdir -p /root/perf1/uvcache /root/perf1/tmp /root/perf1/wheels
export UV_CACHE_DIR=/root/perf1/uvcache TMPDIR=/root/perf1/tmp PIP_CACHE_DIR=/root/perf1/uvcache/pip PIP_NO_CACHE_DIR=1
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH MAX_JOBS=6 CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST=9.0
SP=/root/jpp_lens/.venv/lib/python3.12/site-packages
ls $SP | grep -v __pycache__ | md5sum > /root/perf1/jpp_venv_before.md5
uv venv /root/perf1/venv_build --python /usr/local/bin/python
echo "import site; site.addsitedir(\"$SP\")" > /root/perf1/venv_build/lib/python3.12/site-packages/zz_jpp_lens.pth
uv pip install --python /root/perf1/venv_build/bin/python pip setuptools wheel ninja packaging
B=/root/perf1/venv_build/bin
time PATH=$B:$PATH $B/python -m pip wheel --no-deps --no-build-isolation -v causal-conv1d -w /root/perf1/wheels 2>&1 | grep -v "^ *copying\|^  *creating" | tail -60
ls -la /root/perf1/wheels
W=$(ls /root/perf1/wheels/causal_conv1d-*.whl | head -1)
$B/python - "$W" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall("/workspace/o-lens-improvement/.venv_fast/lib/python3.12/site-packages")
print("extracted", sys.argv[1])
PY
cd /tmp && $R/.venv_fast/bin/python -c "import causal_conv1d, transformers.utils.import_utils as iu; print('cc1d', causal_conv1d.__version__, causal_conv1d.__file__, iu.is_causal_conv1d_available(), iu.is_flash_linear_attention_available())"
ls $SP | grep -v __pycache__ | md5sum; cat /root/perf1/jpp_venv_before.md5
echo BUILD_DONE
