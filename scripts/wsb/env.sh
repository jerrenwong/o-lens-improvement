# Shared paths for the WorkspaceBench (wsb) pipeline. `source` it; it sets variables only, no commands.
R=/workspace/o-lens-improvement
WSB=$R/scripts/wsb                                  # these scripts
WSB_TAR_SRC=$R/.hf27/wsb/src/workspace-bench-main   # benchmark source, extracted from the GitHub tarball (2026-11-27)
WSB_SRC=${WSB_SRC:-/root/wsb_src}                   # pod-local copy of the benchmark (banks in evals/ beside src/)
WSB_ENV=${WSB_ENV:-/root/wsb_env}                   # pod-local uv env of the benchmark (+ .pth bridge to /root/jpp_lens)
WPY=$WSB_ENV/bin/python
WSBENCH=$WSB_ENV/bin/wsbench
JPY=/root/jpp_lens/.venv/bin/python                 # our project env (torch, transformers, workspace_lens)
STORE=${STORE:-$R/.hf27/wsb/store}                  # residual store (wsbench.produce.store layout), persistent
RO_ROOT=${RO_ROOT:-$R/.hf27/wsb/readouts}           # readouts/<lens>/<family>.jsonl (+ .meta.json)
JUDGED_ROOT=${JUDGED_ROOT:-$R/runs/wsb}             # judged/<judge tag>/<lens>/<family>/{results.json,cells.jsonl}
# HF cache must be pod-local: /workspace keeps no symlinks and the hub cache is built from them.
export HF_HOME=${HF_HOME:-/root/hf_wsb}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
export TMPDIR=${TMPDIR:-/tmp}                        # `wsbench run` needs flock in $TMPDIR: keep it local
export PYTHONDONTWRITEBYTECODE=1
GRID="20,24,28,32,36,40,44,48,52,56,60"            # the benchmark's 11-layer read grid (docs/producing_readouts.md)
ALENS_LAYERS="24,32,40,48,56"                       # grid layers where the A-lens maps exist (8..56 step 8)
FAST_FAMILIES="arithmetic_intermediates,association_hard,buggy_code,chain_intermediates,conjunctive_association,hinted_cutoff,introspection,knowledge_cutoff,model_multihop,model_specific_directed_modulation,moral_rationale,multi_concept_directed_modulation,nyt_connections,poetry,relational_multihop,role_bound_association,user_modeling_hard,wrong_intermediate"
