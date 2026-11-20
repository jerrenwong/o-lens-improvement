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

- Poetry (q6): the rhyme word is NOT at the newline in the 9B model. A ridge probe trained on 2,694 self-generated
  couplets reads it from the newline no better than from line 1's last word (top-10 .24-.32 vs .35-.38; lookup
  baseline .40), newline + last word = last word, and swapping the newline's activations changes the rhyme in 0/30.
- 2B replication (q7): same failure modes; '@' junk is u1 of the J++ L21 matrix and is triggered by the end of the
  prompt; no Chinese-only hits; J-lens does NOT gain from task-mean subtraction on 2B; gated removal +2.0 dev (0 fit);
  asked 16/61, J++ reads 5 of those.
- J++ FITTING MISMATCHES (jpp_lens recipe): raw WikiText-103 (' . ', ' @-@ ', ' @,@ ') and positions 16-127 only.
  Benchmark prompts are normally formatted and mostly read before position 30. Refit test: agent r1, runs/r1_2b/.

- J++ REFIT on 2B (r1, authors' pipeline, 128 prompts each, dev 109 items; runs/r1_2b/, scripts/r1_*):
  A recipe (raw, skip 16) .467/.258 unf .411 | B natural, skip 16 .463/.260 unf .454 | C natural, skip 1 .496/.260
  unf .476 | D raw, skip 1 .473/.306 unf .397 | released .525/.294 unf .391.
  C vs A: +2.9 [+0.8, +5.4] filtered, +6.5 [+1.4, +12.8] unfiltered; C vs released: -2.9 [-11.5, +3.3] filtered,
  +8.5 [+2.5, +15.5] unfiltered. Natural format REMOVES the '@' direction from the matrices (B, C: none; A, D,
  released: present). Refit-to-refit noise ~ effect size (A at 32 prompts .493). Released 9B lens: 64 prompts.
  9B refit with replicates: agent r2, runs/r2_9b/ (PARTIAL.md has the half-budget numbers).

- J++ REFIT on 9B (r2, 96 prompts per condition in 4 shards; runs/r2_9b/): no gain. Dev: released .5952; recipe
  refit A .5735 (halves .6018, .5883); natural + skip 1 refit C .5826 (halves .5789, .5494). C - A: full +0.9
  [-3.4,+5.8], halves -2.3 and -3.9. Refit-to-refit spread ~3 points. '@' share on association at L24: C 0%, A 100%,
  released 99% -> the junk comes from raw WikiText formatting, but removing it at the source adds no hits.
- FINAL TEST (one evaluation, 2026-11-20 13:19Z, auto/runs/T_*): q8_gate vs released J++.
  9B test (153): .5858 vs .5809 = +0.5 [-1.0,+2.4]; pass@1 +1.0 [0,+2.4]; unfiltered +1.5 [0,+3.4].
  2B test (108): .3150 vs .3150 = 0.0.
  Pooled dev+test: 9B (309 items) +0.7 [-0.2,+2.0], 4 items better / 1 worse; unfiltered +1.2 [+0.2,+2.4].
  2B (217 items) +1.0 [+0.2,+2.0], 4 better / 0 worse. Controls flat or negative. Marginal, not a meaningful gain.

- ELICITATION-DISTILLED MAP (e1; scripts/e1_*.py, runs/e1/E_9b.pt, auto/candidates/e1_*): per-layer ridge map from
  the last-token residual of a text prefix to the model's final-layer state at the end of three few-shot one-word
  "topic" questions about that prefix. 30,500 natural-format prefixes (WikiText detok 13k, TinyStories 12k, Wikipedia in
  10 languages 5.5k), no benchmark text or labels. Held-out agreement with the elicited word, top-10: E .60-.82 per
  layer (.88 any layer) vs J++ .02-.23 (.42) vs shuffled .06-.08.
  Lens = unit(J++ h) + lam * unit(E h); lam = 0.5 chosen on FIT (.4964 -> .5705). Dev .6265 / .4042 (unf .6022) vs J++
  .5952 / .3850 (.5718): +3.1 [-0.4,+6.7]. vs shuffled +3.5 [+0.7,+6.6]; vs prior-only +2.2 [-0.5,+5.1] (pass@1 +7.6).
  Single-format maps .612 / .639 / .623. Association gains: office, art, time (also with the prior alone), free,
  UNESCO, field (input-specific); loses pregnant. E is a topic probe (L4 already .60), not a Jacobian.
  Frozen recipe; 2B replication + confirmatory test (second look at the test split) pending.

- CONFIRMATORY TEST of the frozen recipe J++ + lam*E (2026-11-20 ~14:25Z; auto/runs/T_e2_*; second look at the test
  split, the first was the gate):
  9B test (153): .6143 vs J++ .5809 = +3.3 [+0.4,+6.6]; unfiltered +4.8 [+1.6,+8.2]; pass@1 +0.6 [-3.8,+4.7].
     Controls on test: shuffled map -1.8, prior only -3.1 [-5.9,-0.5]. A - shuffled +5.2 [+2.6,+8.1].
     Pooled dev+test (309): +3.2 [+0.9,+5.7] (28 better / 9 worse); unfiltered +3.9 [+1.5,+6.3]; pass@1 +1.3 [-1.5,+4.3].
  2B (lam .3): dev 0.0 [-9.2,+8.0]; test .4060 vs .3150 = +9.1 [+3.3,+15.8], but the shuffled control is +8.1 there
     (multihop .19 -> .50 for all three). Pooled (217): +4.5 [-1.4,+10.0]; pass@1 +2.9 [+0.1,+6.4]; unfiltered +9.0
     [+4.7,+13.8]; input-specific part (A - shuffled) +1.2 [+0.4,+2.3].
  Reading: on 9B a confirmed, input-specific gain of about 3 points; on 2B positive but mostly a generic effect.

- 27B REPLICATION (m27; Qwen3.6-27B, released J++ koayon/jpp-lenses; auto/evaluate27.py, split_27b.json fit 79 / dev
  165 / test 162; maps runs/e1/E_27b*.pt; model cache .hf27, run with HF_HOME=/workspace/o-lens-improvement/.hf27).
  Dev: J++ .6153 / .3731 (unf .5938); J++ + 0.5*E .6609 / .3940 (unf .6218) = +4.6 [+0.6,+9.0]; shuffled .6140;
  prior .6216; A - shuffled +4.7 [+2.2,+7.5]; A - prior +3.9 [+1.7,+6.5]. Caveat: 27B answers the elicitation prompts
  with "\n\n" first in 70-96% of cases, so targets are noisy (22% of "elicited words" are <|im_end|>); recipe kept
  frozen. Multihop gain (+.10) also in both controls.

- 27B TEST (single look, 2026-11-20 16:58Z; auto/runs/T_*_27b): J++ .5620 / .3271 (unf .5195); J++ + 0.5*E .5708 /
  .3090 (unf .5261) = +0.9 [-2.0,+3.9] pass@10, -1.8 [-4.8,+1.0] pass@1. Controls: shuffled -1.9, prior -2.2;
  A - shuffled +2.8 [+1.0,+4.9]. Pooled dev+test (327): +2.8 [+0.2,+5.6] (25 better / 12 worse); pass@1 +0.2;
  unfiltered +1.8 [-1.2,+4.9]; A - shuffled +3.7 [+2.1,+5.5].
  SUMMARY J++ + lam*E vs released J++, all held-out prompts, pass@10: 2B +4.5 [-1.4,+10.0] (mostly generic);
  9B +3.2 [+0.9,+5.7]; 27B +2.8 [+0.2,+5.6]. Test splits alone: 2B +9.1 [+3.3,+15.8], 9B +3.3 [+0.4,+6.6], 27B +0.9
  [-2.0,+3.9]. Top-1 does not improve on any model. Input-specific part (A - shuffled): 2B +1.2, 9B +4.3, 27B +3.7.

- FRESH-PROMPT TEST (pre-registered in auto/PREREG_v2.md; data/evals_v2, 563 new prompts frozen before scoring;
  auto/evaluate_v2.py): PRIMARY A - J++ pass@10, mean over 9B and 27B = +2.2 [+0.6,+3.7] (p ~ 0.008). 9B +3.3
  [+0.8,+5.9]; 27B +1.1 [-0.7,+2.9]. pass@1 -0.1 [-1.8,+1.7]. A - shuffled +6.1 [+4.7,+7.5]; shuffled - J++ -3.9.
  Refined map B (content-word targets, prior removed, chat-format elicitation): -2.7 vs J++ on 9B (targets were letter
  fragments; every lam below J++ on fit) -> abandoned.

- DATA-SCALING CHECK (e4; scripts/e4_scale.py, auto/candidates/e4_scale.py; 9B dev, J++ .5952): frozen recipe refit on
  n training prefixes -> dev pass@10 / held-out agreement (any layer): n=250 .6197 / .585; 1000 .6078 / .768; 4000
  .6206 / .839; 16000 .6116 / .879; 28500 .6265 / .881. The benchmark gain is flat in n (+1.3 to +3.1, no trend) while
  agreement on ordinary text keeps rising: the gain is not driven by the amount of distillation data.

- WHY THE ASK-MAP HELPS (why1, 9B dev): all 14 gains are J++ near misses (rank 12-76); 0 of 59 far misses rescued. A
  map from a BAG OF THE PROMPT'S VISIBLE WORD EMBEDDINGS (no model computation), lam chosen on fit, scores .6282 vs the
  ask-map's .6265 -> on this benchmark the ask-map is a text-topic guess. Gain needs >= 256 directions, lives at L20-28.
  Topic word is in the model's own next-token top-10 only 5% of the time; J++ hits it 92% when it is, 40% otherwise.
- GENERATE-WITHOUT-NAMING MAP G (g1): 11,624 sentences / 6,000 words; ridge to the word's unembedding direction.
  Dev +0.7 [-1.0,+2.4]; v3 9B +0.1. Memorises (train .998); on unseen words top-10 any layer: G .154, J++ .263, E .078.
- ROUTED EXPERT READOUT (x1): using the activation's own cluster expert: .4810 (soft .455-.466; pooled .4393) vs the
  fixed-weight refit .5735. Each expert alone is J-lens-like; J++'s reading comes from the SIGNED COMBINATION (heavy
  cancellation: max|J| .20 vs 1.20 pooled at L28); the '@' direction is created by that combination.
- THINK vs SAY (th1; 200 words x 50 generated sentences; runs/th1): decoding among 200 words, top-1: THINK probe
  .88-.92 from L6 (.65 at L0 = lexical leakage), class-mean .30-.70, SAY (J++) peak .55 at L9, .34 at L21, .52 at L27.
  98-99.9% of SAY hits are probe hits; SAY catches 52-85% of probe hits. cos(THINK_X, SAY_X) .15-.26 but SAY_X is the
  nearest SAY direction for 90-97% of words; the 200 SAY directions hold 6-19% of THINK-offset energy (random 5%).
  Steering at every token, k=6: THINK@L8 -> asked topic X 72%, says X 28%; SAY@L8 78% / 16%; SAY@L24 -> says X 72%,
  topic stays Y 85%. Single-token edits never change the topic. => SAY is a thin slice of THINK; mid layers (8-16) are
  where they overlap; late-layer SAY = "emit this token".
- v3 9B (475 fresh prompts): A - J++ = +3.7 [+1.2,+6.4]; shuffled -3.5; G +0.1; AG +0.7.

- v3 REGISTERED TEST complete: A - J++ pass@10 mean(9B, 27B) = +2.1 [+0.5,+3.7]; 9B +3.7, 27B +0.5 [-1.5,+2.5];
  27B pass@1 -2.2 [-4.3,-0.2]. v2+v3 pooled: 9B +3.5 [+1.7,+5.4]; 27B +0.8 [-0.6,+2.1] with pass@1 -2.0 [-3.6,-0.5].
  => the ask-map gain is a 9B effect; on 27B it does not help pass@10 and hurts top-1.
  GPU note: a 27B job must reserve memory at start (see /root/v3_greedy.py pattern: allocate blocks until 67 GB,
  free them to the cache, then load) or smaller jobs squeeze in while it loads and it dies of OOM.

- THINK->WORD TRANSLATION MAP (al1; scripts/al1, .hf27/al1; 2,398 strict words x ~20 generated sentences; split by
  word 1,998 / 200 / 200): noise-aware ridge M3 (whiten by within-word covariance, ridge on class means). UNSEEN test
  words, top-10 over the full vocabulary, any of 7 layers: M3 .267, J++ .324, J++ + 0.3*M3 .527 (lam on val words),
  shuffled-label .0003; sentence-ridge M1 .208; mean-ridge M2 .238. Learning curve .050/.080/.154/.267 for
  250/500/1,000/2,000 words (not saturated). Mean of 8 sentences: M3 .43, J++ .48, mix .79. Hits overlap little
  (both 363, J++ only 881, M3 only 663 of 3,837). BENCHMARK: no transfer (dev .5738 vs J++ .5952; M alone ~.01 on
  fit; 0 prompts gained, 3 lost). Scale-up with 8,000 words and 5 text styles: agent al2 (.hf27/al2).

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
