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
User guidance (2026-11-20): do not spam ideas and judge them by the score. Read real outputs, name the failure mode,
follow up curiosities (e.g. Chinese tokens), then design a method for a named failure mode.
Read `auto/failure_modes.md` (token-class tables) and the findings below (scripts `auto/qual_wordonly.py`,
`qual_zh.py`, `qual2.py`, `qual3.py`, `qual4.py`; fit+dev only).

Qualitative findings, Qwen3.5-9B (fit+dev, 231 prompts):
- J-lens top-10: layers 4-20 are punctuation and word-glued punctuation that slips past the filter; layers 24-28 are
  the opening of the next sentence (She, His, But, Until).
- Both lenses mostly output one vector per prompt format: within a task, readout vectors are 80-99% aligned with their
  mean (poetry 97-99% for J++). At L20, 71-77% of every J++ readout is the matrix's top singular direction.
- J++ L24 on association: 99% of top-10 slots are '@g @d @gmail ...' (L28: 'Flames, Raptors, GM, AT'). It is the 2nd
  singular output direction of the J++ L24 matrix. It appears only at a normally written sentence-final period
  ('word.' 99%, 'word!' 2%, 'word .' 0%, last word 1%; WikiText periods 0.4-7%). Origin of '@' unexplained.
- Masking all non-word tokens: J++ +0.8, J-lens +1.7. Under the junk is more junk.
- Chinese tokens are the concept in Chinese (antonym, Spanish, month, copper), not junk: Chinese-only scoring gives
  21% (J++ dev), but only 3 of 151 J++ misses are Chinese-only hits. Masking CJK: no change.
- Subtracting the same-task mean activation: J-lens dev .415 -> .491 (typo .56 -> .85). J++ .595 -> .483 (prompts in a
  task share concepts such as 'opposite'; the mean removes them). Readouts become clean scene words (cashier, checkout;
  Radio, Military) but not the inferred concept (poverty, Churchill).
- Removing top singular directions of the matrices: hurts (J++ -8.6 with one). Projecting out the fit-association mean
  direction everywhere: hurts. Gated (only rows > 80% aligned, L24+L28): J++ dev .595 -> .605, fit .530 -> .550
  (two association prompts each).
- Association ceiling: asked with the full passage in a one-word few-shot format, the model names the concept in its
  top 10 for 33 of 61 (top 1: 11). J++ reads 18. Asked-and-read 16, asked-only 17, read-only 2, neither 26 (mostly
  famous-name riddles: Einstein, Lincoln, Churchill, Beatles, Obama).
- Association score depends on the readout token: 'word.' 29.5%, '!' 21%, last word 10%, newline after 7%, ' .' 3%.
- Poetry: at the prompt end the model writes the rhyme word 100% of the time; at the newline the readout shows the
  line just finished, not the rhyme.
- Multilingual by slot (J++): language .47 (in Chinese .21, either .51), relation .63, source .60, answer .95.

- Multilingual misses read token by token (J++, 43 four-slot items): language slot 20/43, relation 27/43, source 26/43,
  answer 41/43. Of the misses: 9 items label the language in its own language (português, español, français), benchmark
  0/9, while J++ names it in English or Chinese for 5/9; 10 antonym prompts are labelled place/size/order/time and J++
  reads 'opposite' or its Chinese form on 10/10; the English source word is read on 6/19 antonym prompts vs 20/24 other
  prompts (at the final token the lens shows the answer and the dimension: small, size, low). Low-resource language
  names (Ukrainian, Bulgarian, Serbian, Polish, Irish) come out as 'dialect', 'Roman', 'Gael', 'English'.
- Period-column transplant (q5): putting the passage's final-period residual (all layers) into a neutral passage and
  asking the one-word question gives 1.6% (floor 1.1%), even for the 18 concepts J++ reads there; replacing the period
  column inside the real passage barely hurts (54.1 -> 50.8%). When asked, the model uses the passage tokens, not the
  period. The test cannot show whether missed concepts are at the period.
- q5_gate (gated junk removal at L24+L28) through the scorer: dev .6049 / .3899; kept as the entry for the final test.

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

## Learned in this loop (dev split, J++ = 0.5952 / 0.3850)
- Batch 1 (27 candidates, all discarded):
  - Summing J++ over neighbouring layers: +-1 ties, +-2 hurts (-2.5 to -3); disjoint bands over all 31 layers -1.7.
    Unread layers do hold a few extra hits (fit: 7 concepts), but only as separate ranked lists, which the benchmark
    does not allow.
  - J++ + lam x J-lens: -0.9 (lam .3), -7.9 (lam 1). The two lenses make overlapping errors.
  - Subtracting a fixed vector from J++ (WikiText mean readout, period/newline mean, same-token mean): -5 to -12.
    Junk tokens in association readouts are context-dependent, not a fixed bias.
  - Logit priors (unigram frequency, whole-word): no change.
  - Continuation subtraction (user idea U1): on J++ ties at beta .25 and hurts beyond; projecting out top next-token
    directions hurts (multihop .96 -> .57: the answer shares directions with the bridge entity); subtracting the
    readout 4 layers later hurts association (.32 -> .20): the concept persists, it is not used up.
    On the J-lens: typo .56 -> .69 but multihop .89 -> .79; net +0.3 to +1.9 (noise).
- Per task for J++ on dev: association .32, multihop .96, multilingual .69, poetry .03, typo .97. Gains must come
  from association, multilingual or poetry. Association and poetry misses are far misses (concept rank > 100).
- Batch 2D (patchscope): patching h into a decoding prompt and reading the model's output scores 0.00 on association
  for every prompt tried (fit), <= 0.06 on poetry, and only recovers typo corrections that J++ already reads. So the
  association / poetry concepts are not decodable from the single readout-token activation even by the model's own
  later layers. Mixtures with J++ are noise (+1.3 dev, worse on fit and on pass@1). Spectrum reshaping and
  J -> J++ extrapolation: tie or worse.
- Batch 2G (expert bank): signed weights over 8 maps (J++, J, h, J minus always-on, SVD parts of J++, neighbours),
  fitted on the fit split with cross-validation, stay within +-0.1 of plain J++ and give +0.8 / -0.5 on dev (noise).
  Every map we have makes the same errors as J++. Per-token z-scoring: -16. Layer increments: tie.
- Round-2 brainstorm facts (J++, dev): association pass@10 .32, @50 .54, @1000 .78; poetry .03, @100 .34: the concept
  is weakly present, not absent. Multilingual by intermediate: language name .43, relation .73, source word .63,
  answer .97. Top-10 lists often spend several slots on case/space variants of one word.
- Batch 2F: the Jacobian to far-future positions (>= 8 tokens later), computed exactly by finite-difference JVPs,
  scores 0.076 alone on fit (association 0.00): the J-lens's readout comes from the same-token term. A ridge map to
  the bag of future tokens has R2 ~0.01 and hurts (-12.6). Interventional readout: -17. Random-probe estimates of a
  4096 x 4096 Jacobian are noise within the budget.
- Batch 2E (supervised corrections): a correction trained on ~900 self-generated "scene -> concept" pairs reads new
  scene concepts on its own held-out riddles (.59 -> .79) but transfers to the benchmark only at the level of a
  shuffled-label control (+1.1 to +1.6). A ridge correction fitted on the 75 fit prompts gives +3.9 on dev, entirely
  from multilingual (.69 -> .83): suspected memorisation of the small set of language names shared by fit and dev.
- Batch 3J: the +3.9 of the fit-trained ridge correction is memorisation of words repeated across multilingual
  templates (opposite/big/small, time/night/day, place/up/down recur in different languages). Leaving out fit items
  that share a word removes the gain; training without multilingual items gives +0.1. DISCARDED.
  BENCHMARK CAVEAT: multilingual items share intermediates across items, so any map with many parameters fitted on
  labelled items can score by memorising them. Only few-parameter uses of the fit labels are trustworthy.
