#!/bin/bash
# Download the pre-fitted Qwen3.5-2B J-lens and J++ lens (Neuronpedia, WikiText fits) to artifacts/baselines.
set -euo pipefail
D=/workspace/o-lens-improvement/artifacts/baselines; mkdir -p $D
PY=/root/jpp_lens/.venv/bin/python
$PY - <<PY
from huggingface_hub import hf_hub_download
import shutil
for sub, name in [("jlens", "Qwen3.5-2B_jacobian_lens.pt"), ("jpp", "Qwen3.5-2B_jpp_lens.pt")]:
    p = hf_hub_download("neuronpedia/jacobian-lens", f"qwen3.5-2b/{sub}/Salesforce-wikitext/{name}")
    shutil.copyfile(p, f"$D/{sub}_prefit.pt")
    print("ok", sub)
PY
