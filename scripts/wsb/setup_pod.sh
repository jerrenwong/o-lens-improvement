#!/bin/bash
# WorkspaceBench pipeline: pod setup. Idempotent; re-run after a pod restart (pod-local /root is wiped).
# Usage (on the GPU pod, as root):
#   bash /workspace/o-lens-improvement/scripts/wsb/setup_pod.sh            # env + residuals of the 18 fast families
#   FAMILIES=poetry bash .../setup_pod.sh                                   # env + residuals of some families only
#   PULL=0 bash .../setup_pod.sh                                            # env only (judging needs no residuals)
#   CAPTURE=1 bash .../setup_pod.sh                                         # also forward-pass the cells the dataset lacks
# What it does:
#   1. /root/jpp_lens env (the project's scripts/setup_pod.sh) if missing: torch, transformers, workspace_lens.
#   2. Copies the benchmark source from .hf27/wsb/src (GitHub tarball of main, 2026-11-27) to /root/wsb_src
#      and builds its uv env in /root/wsb_env (`uv sync --frozen --extra dev`; no gpu extra: torch etc. come
#      from /root/jpp_lens through a .pth bridge, as in scripts/w5). Nothing of the benchmark runs on the hub.
#   3. Bridge .pth: jpp_lens site-packages (appended, so the benchmark's own pins win), /root/jpp_lens/src,
#      $R/src, scripts/wsb, and `import wsb_or_ledger` (OpenRouter ledger; no-op unless WSB_OR_LEDGER is set).
#   4. Offline checks (no key, no API call): `wsbench check judge_dry_run`, the fast plan.
#   5. pull_residuals.py: the fast families' residuals from agu18dec/workspacebench_residuals into $STORE
#      (only part files holding wanted rows are fetched; HF cache pod-local and purged per family).
# ASSUMPTIONS (unverified until run): /root/jpp_lens/.venv is Python >= 3.12 (the benchmark requires it) and
# its torch/transformers (2.11 / 5.14.1) load Qwen3.6-27B's lm_head/norm as in scripts/w5; the pod has
# internet (uv, Hugging Face, the pinned public data of nyt_connections / knowledge_cutoff).
set -euo pipefail
source /workspace/o-lens-improvement/scripts/wsb/env.sh
export PATH=$HOME/.local/bin:$PATH
FAMILIES=${FAMILIES:-$FAST_FAMILIES}
PULL=${PULL:-1}
CAPTURE=${CAPTURE:-0}

# 1. project env
[ -x "$JPY" ] || bash "$R/scripts/setup_pod.sh"
PYV=$("$JPY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
"$JPY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' \
  || { echo "jpp_lens python is $PYV; the benchmark needs >= 3.12: the .pth bridge cannot work, use 'uv sync --extra gpu' instead"; exit 3; }

# 2. benchmark source + env (pod-local)
mkdir -p "$WSB_SRC"
rsync -a --delete "$WSB_TAR_SRC/" "$WSB_SRC/"
cd "$WSB_SRC"
UV_PROJECT_ENVIRONMENT="$WSB_ENV" uv sync --frozen --extra dev --python "$PYV" -q

# 3. bridge
SP_W=$("$WPY" -c 'import site; print(site.getsitepackages()[0])')
SP_J=$("$JPY" -c 'import site; print(site.getsitepackages()[0])')
cat > "$SP_W/zzz_wsb_bridge.pth" <<EOF
$SP_J
/root/jpp_lens/src
$R/src
$WSB
import wsb_or_ledger
EOF
"$WPY" -c 'import torch, transformers, safetensors, huggingface_hub, wsbench, wsbench.llm; from workspace_lens.utils import load_lens_file; print("env ok: torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available())'

# 4. offline checks (no key needed, no API call)
env -u ANTHROPIC_API_KEY -u OPENROUTER_API_KEY "$WSBENCH" check judge_dry_run > /root/wsb_check.log 2>&1 \
  && echo "check judge_dry_run: ok" || { echo "check judge_dry_run FAILED: see /root/wsb_check.log"; tail -5 /root/wsb_check.log; }
"$WSBENCH" plan --fast out=/root/wsb_plan_fast > /root/wsb_plan.log 2>&1 && echo "fast plan: $(ls /root/wsb_plan_fast | wc -l) files in /root/wsb_plan_fast" \
  || { echo "fast plan FAILED: see /root/wsb_plan.log"; tail -5 /root/wsb_plan.log; }

# 5. residuals
if [ "$PULL" = 1 ]; then
  mkdir -p "$STORE" "$JUDGED_ROOT"
  CAP=""; [ "$CAPTURE" = 1 ] && CAP="--capture"
  "$WPY" "$WSB/pull_residuals.py" --families "$FAMILIES" --store "$STORE" $CAP \
    --report "$JUDGED_ROOT/coverage_fast.json" 2>&1 | tee /root/wsb_pull.log
fi
echo "setup done. Disk: $(df -h /root | tail -1)"
