"""tl1: write runs/tl1/handover.md (held-out commands for the orchestrator)."""
import json
from pathlib import Path
R = Path("/workspace/o-lens-improvement")
MODELS = [("9b", "Qwen/Qwen3.5-9B", False), ("2b", "Qwen/Qwen3.5-2B", False), ("olmo7b", "allenai/Olmo-3-7B-Instruct", True),
          ("gemma4e4b", "google/gemma-4-E4B-it", True), ("27b", "Qwen/Qwen3.6-27B", True)]
L = ["# tl1 handover: T-lens (PREREG_v4 Addendum F) held-out scoring, orchestrator only", "",
     "Agents scored DEV only. Run each command once. Maps: `.hf27/tl1/T_<tag>_t3.pt`, `T_<tag>_s5.pt`. Candidate: "
     "`auto/candidates/tlens.py`. Scorers: `auto/evaluate_t.py` (original test split), `auto/evaluate_t_v4.py` (v4); "
     "both are copies of the existing scorers that only add the token-mean input-embedding feature (no scoring change; "
     "DEV reproduction of the existing A-lens runs is exact for every model, see `runs/tl1/results.md`).", "",
     "lam is chosen on FIT inside build() (same rule/grid as alens.py); expected values from the DEV runs are listed per model.",
     "v4 dry runs (`--dry-run`: build + transport of every v4 row, nothing ranked) passed for all rows and models, and the "
     "full-model `--verify-fwd` check reproduced the cached residuals of every FIT, DEV and v4 row (min cos >= 0.999).", "",
     "```bash", "R=/workspace/o-lens-improvement; PY=/root/jpp_lens/.venv/bin/python; cd $R",
     "export HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", "```", ""]
for tag, M, hf in MODELS:
    lam = {}
    for b in ("J", "JPP"):
        p = Path(f"/root/tl1/{tag}/dev_{b}.log")
        lam[b] = "see DEV log"
    L += [f"## {tag} ({M})", "", "```bash", ("export HF_HOME=$R/.hf27" if hf else "unset HF_HOME"),
          f"TE=$R/.hf27/tl1/T_{tag}_t3.pt; TE2=$R/.hf27/tl1/T_{tag}_s5.pt"]
    for b in ("none", "J", "JPP"):
        L.append(f"TLENS_BASE={b} TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t.py --lens auto/candidates/tlens.py "
                 f"--model {M} --split test --orchestrator --lite --name T_tl_{b}_{tag}")
    for b in ("none", "J", "JPP"):
        L.append(f"TLENS_BASE={b} TLENS_E=$TE TLENS_E2=$TE2 $PY auto/evaluate_t_v4.py --lens auto/candidates/tlens.py "
                 f"--model {M} --split test --orchestrator --lite --name V4_tl_{b}_{tag}")
    if tag in ("9b", "2b", "27b"):
        ta, va = (lambda r: f"T_al_{r}_{tag}"), (lambda r: f"V4_al_{r}_{tag}")
    else:
        ta, va = (lambda r: f"xm1_{tag}_alens_{r}_test"), (lambda r: f"xm1_{tag}_alens_{r}_v4")
    for pre, ref in (("T", ta), ("V4", va)):
        L.append(f"$PY scripts/r2_boot.py {ref('JPP')}-{pre}_tl_JPP_{tag} {pre}_tl_JPP_{tag}-{ref('JPP0')} "
                 f"{ref('J')}-{pre}_tl_J_{tag}")
    L += ["```", ""]
L += ["Bootstrap references: the A-lens held-out rows (`T_al_*` / `V4_al_*` for Qwen, `xm1_<tag>_alens_*_test|_v4` for "
      "Olmo/Gemma) must exist; score any missing A-lens row (J, JPP, none) with its existing command first.", ""]
(R / "runs/tl1/handover.md").write_text("\n".join(L))
print("written")
