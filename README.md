# Nonlinear J-lens

Does a lens with a nonlinear inner map, `g(h) = J·h + W2·relu(W1·h)`, read hidden intermediate
concepts better than the J-lens's single averaged Jacobian `J·h`?

- **Model:** Qwen/Qwen3.5-2B (24 blocks, d = 2048). Baselines pre-fitted on WikiText by
  Neuronpedia (`neuronpedia/jacobian-lens`, `qwen3.5-2b/{jlens,jpp}`).
- **Training signal** (`src/olens/train.py`): one Gaussian `u` per sequence at every valid
  final-layer position, one backward pass → `(Σ_{t≥s} ∂h_L[t]/∂h_l[s])ᵀ u` at every source
  layer and position. That is the J-lens estimator projected on `u`, so a linear map trained on
  it converges to the J-lens. All arms train from the same model pass.
- **Arms:** `{linear, J·h + MLP(m)} × {match Jacobians, match final output (KL, tuned lens)}`.
- **Evaluation:** the J++ harness (`safety-research/jpp_lens`, pinned in `scripts/setup_pod.sh`):
  recall@k of the hidden intermediate on Anthropic's readout evals, held-out split, and the
  probe-swap causal eval. Our lenses load there as lens type `trained` (`src/olens/lens.py`).

## Run (GPU pod)

```bash
bash scripts/setup_pod.sh          # env in /root/jpp_lens/.venv (re-run after a pod restart)
bash scripts/get_baselines.sh      # pre-fitted J-lens / J++ -> artifacts/baselines/
PY=/root/jpp_lens/.venv/bin/python
$PY -m olens.train --jlens artifacts/baselines/jlens_prefit.pt --out runs/<name>
$PY -m olens.evaluate grade-correctness --hf-model-name Qwen/Qwen3.5-2B --out $PWD/data/model_correctness.csv
$PY -m olens.evaluate evaluate --hf-model-name Qwen/Qwen3.5-2B --layers 3,6,9,12,15,18,21 \
    --correctness-csv $PWD/data/model_correctness.csv --lens logit $PWD/artifacts/baselines/*.pt $PWD/runs/<name>/*.pt \
    --out $PWD/runs/<name>/eval/readout.csv
```
