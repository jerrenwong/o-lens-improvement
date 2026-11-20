# Lens autoresearch (adapted from karpathy/autoresearch)

Goal: find a lens that reads hidden intermediate concepts better than J++ on Qwen/Qwen3.5-9B.
One metric, one editable kind of file, fixed budget, keep or discard.

## Files
- `auto/evaluate.py` — FIXED scorer. Never edit. `python auto/evaluate.py --lens auto/candidates/<name>.py`
- `auto/lens.py` — the current best lens (starts as J++). Advanced only by the orchestrator.
- `auto/candidates/<name>.py` — one file per experiment. THIS is what an experiment writes.
- `auto/results.tsv` — every experiment: `id  dev_pass10  dev_pass1  build_s  status  description` (tabs).
- `auto/program.md` — this file.

## The lens interface (a candidate file must define this)
```python
def build(ctx) -> Lens: ...
class Lens:
    def transport(self, h, layer, resid_all=None) -> Tensor   # [N, d] -> [N, d]
```
- `h`: residual-stream activations `[N, d]` (fp32, cuda) at `layer`, one row per benchmark prompt, taken at the
  prompt's readout token. `resid_all`: `{layer: [N, d]}` for every block 0..30 of the same rows (optional use).
- Return one vector per row in the FINAL-LAYER basis. The scorer applies the model's own final norm + unembedding
  and ranks the vocabulary (punctuation-only tokens removed). One ranked list per (prompt, layer); the benchmark
  takes the best rank over the 7 readout layers 4, 8, 12, 16, 20, 24, 28.
- `ctx` gives: `ctx.model` (jlens HFLensModel: `.layers`, `.forward(ids)`, `.unembed(v)`, `.tokenizer`, `.d_model`),
  `ctx.J[layer]` (plain J-lens, 31 layers, fitted on 64 WikiText prompts), `ctx.JPP[layer]` (pre-fitted J++),
  `ctx.dict_path(layer, space)` (sparse dictionaries `olens.bsf_io.BlockDict`, block size 1, space `resid` or
  `jspace`, at the 7 readout layers, with `_stats.pt` sidecars giving firing counts),
  `ctx.wikitext(split, n_windows)` (token windows `[n, 128]`), `ctx.fit` (the labelled FIT split only:
  `.resid[layer] [n_fit, d]`, `.candidates` list of `{intermediate: [token ids]}`, `.eval_slugs`),
  `ctx.readout_layers`, `ctx.device`.
- NOT available: the dev and test prompts or their labels. Do not try to obtain them (no reading the benchmark
  files, the bench cache, or earlier result CSVs from a candidate file).
- Budget: `build()` must finish in 10 minutes on one H100 (the 9B model is already loaded; it uses ~18 GB).
  Other experiments may share the GPU: stay under ~20 GB of extra memory.

## Metric
`dev_pass10`: share of DEV prompts whose hidden concept is in the top 10 at any readout layer, averaged per task
then over the 5 tasks (association, multihop, multilingual, poetry, typo), filtered scoring. Higher is better.
`dev_pass1` is the tie-breaker. The dev split has ~155 prompts, so one prompt is worth ~0.1-0.7 points depending
on its task: small differences are noise.

## Rules
- Simpler is better at equal score. Removing something for the same score is a win.
- A crash or a build over budget is logged as `crash` and discarded.
- The TEST split is scored once at the end by the orchestrator, never during the loop.
- Final check: the winning recipe is rebuilt on Qwen3.5-2B, which the loop never sees.

## Failure modes first
Read `auto/failure_modes.md` (how J-lens and J++ fail on fit+dev prompts: continuation vs echo vs noise, and how deep
the missed concepts sit) before proposing or implementing an idea. Ideas are in `auto/ideas.md`; the user's priority
idea U1 (continuation-subtracted readout) is at the end of that file.

## What is already known (do not rediscover)
Scores are pass@10 on the old held-out split (dev + test together), 9B unless stated.
- Logit lens 32.4. J-lens 38.6. J++ 58.8 (J++ unfiltered 55.9). J++ per task: association 29, multihop 91,
  multilingual 70, poetry 9, typo 95. J-lens: 5 / 76 / 53 / 5 / 55. Association and poetry are where everything is weak.
- J++ = 8 k-means clusters of source activations, one averaged LRP-Jacobian per cluster, signed weights fitted on
  labelled items, punctuation filter. We do not have its cluster Jacobians or weights, only the final matrices.
- More capacity hurts: J h + MLP trained to match per-token Jacobians: -6 to -8.
- Low rank: J-lens +1 to +2 at rank 1280-1536 of 4096; truncating J++ never helps.
- Removing always-on sparse features (dictionary features firing on >= 10% of tokens) from the J-lens input:
  +6.5 to +7.8 (J-space dictionary), +4.0 (residual dictionary). On top of J++: nothing (+-1).
- Direct path: J - a*I is flat for a <= 0.75, worse beyond; J++ - a*I is worse for every a > 0.
- Tuned lens (trained to match the output) and entropy penalties: worse than the J-lens.
- Per-feature voting (mixtures, max, rank fusion over active dictionary features): much worse than summing.
- Picking "carrier" features by static statistics: fails (best selector puts a carrier first 29% of the time).
- A dictionary feature's own readout often contains the concept, but the plain J-lens top-640 contains it as
  often: no hidden headroom from single features.
- Reconstruction quality (Jacobian FVU) does not predict readout quality.
