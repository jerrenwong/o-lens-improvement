# Pre-registration: second fresh-prompt test (v3). Written 2026-11-20 before v3 prompts exist.

Why: on v2 the primary held (A - J++ pass@10 = +2.2 [+0.6,+3.7], 9B + 27B) but 27B alone was +1.1 [-0.7,+2.9]. On 27B the
elicitation targets are noisy: the model's first token after the one-word questions is whitespace ("\n\n") for 70-96% of
ordinary-text prefixes.

Change (fixed here, chosen without benchmark data): recipe A' = recipe A with one rule added to target collection:
for each of the three few-shot question formats, if the model's unmasked top-1 next token is whitespace for more than
50% of the 1,000 held-out ordinary-text prefixes, append that format's most common such token to the prompt and take
the final-layer state at the new last position. Everything else is recipe A (mean of three unit vectors, ridge with
bias, same grid and rule, lam from {0.1,0.2,0.3,0.5,1.0} on the ORIGINAL fit split). On 9B the rule does not trigger
(whitespace top-1 for 5% / 3% / 0%), so A' = A there (`e2_A.py`). On 27B: `m27b_A.py`, control `m27b_C_shuf.py`.

Fresh prompts v3: `data/evals_v3/`, written blind to lens outputs, frozen by md5 before scoring; all are test prompts.
Scorer `auto/evaluate_v3.py --split test --orchestrator`.

PRIMARY (one test): A' - J++, pass@10 with the readout filter, macro over tasks, mean over Qwen3.5-9B and Qwen3.6-27B,
on v3. Paired bootstrap, 5000 draws, prompts resampled within model x task. Holds if the 95% interval excludes zero.
SECONDARY (reported whatever they show): per model (27B: A' - J++, and A' - A); pass@1; unfiltered pass@10; A' minus
its shuffled control; v2 and v3 pooled for 9B (same lens A on both).
Also reported: J++ refitted on 2B with 512 prompts (recipe unchanged) against the 32- and 128-prompt refits, 2B dev.

## Amendment 1 (2026-11-20 19:05Z, before any lens other than the J++ sanity run was scored on v3)
Added SECONDARY comparison on v3 for 9B: "generate-without-naming" map G (user idea). The model writes sentences
about a given word without using it; G is a per-layer ridge map (with bias) from the last-token residual of such a
sentence to the word's unembedding direction; lens = unit(J++ h) + lam * unit(G h + b), lam on the original fit split.
The primary reported G variant is STRICT: every word that is an intermediate or target anywhere in the original, v2 or
v3 benchmark files is removed from G's training concepts (list `runs/g1/exclude_concepts.txt`, used only for removal).
Also secondary: AG = J++ plus both maps (ask-map E and G). Candidates `g1_G.py`, `g1_AG.py`, control `g1_G_shuf.py`.
G is fitted after this amendment using ordinary resources only (no benchmark prompts; fit split for lam; old dev split
for sanity). It may be scored on v3 later than A'. The PRIMARY analysis is unchanged.

## Amendment 2 (2026-11-20 19:35Z, before any lens was scored on v3; the J++ sanity run had not finished either)
The whitespace rule cannot be used on Qwen3.6-27B: after appending "\n\n" the model's top-1 next token is `<think>`
for 1000/1000 held-out prefixes in all three formats (p >= 0.996), so every target would be the same. No A' map was
fitted. For the v3 test, the 27B lens is the ORIGINAL frozen recipe A (`m27_A.py`, control `m27_C_shuf.py`), exactly
as on v2. The PRIMARY analysis is therefore: A - J++, pass@10 filtered, mean over 9B and 27B, on v3 — a replication of
the v2 primary on new prompts. Secondary additions: v2 and v3 pooled (both models), same metric.

## Amendment 3 (2026-11-20 20:12Z, before any lens score on v3 was read; the automatic A / shuffled runs may be in progress)
Added SECONDARY comparison on v3 for 9B: "W" = J++ whose expert weights are fitted on self-generated hidden-word
sentences instead of the 75 labelled benchmark prompts. Experts: our recipe refit `runs/r2_9b/A/A_full/experts_K8.pt`
(96 WikiText prompts). Weights: the authors' weight-fitting procedure, unchanged, on 4,000 sentences that the model
wrote about a word without using it (g1 data, STRICT concept set: no benchmark word), the word being the item's
intermediate. Comparisons on v3: W vs the same experts with benchmark-fit weights (`runs/r2_9b/A/A_full/jpp_A_full.pt`),
and W vs released J++. Candidate/lens file fixed by agent w1 before it is scored on v3. PRIMARY analysis unchanged.
Result of G (amendment 1), for the record, on the OLD dev split: G - J++ = +0.7 [-1.0,+2.4]; G will still be scored on v3.

## Amendment 4 (2026-11-20 20:20Z, before any lens score on v3 was read)
Added SECONDARY comparison on v3 for 9B: "T" = J++ plus a bag-of-visible-words topic map (agent why1, map (c)): the
same ridge recipe and targets as E, but the map's input is the mean INPUT EMBEDDING of the prompt's tokens up to the
readout position (no model computation); lens = unit(J++ h) + lam * unit(map output), lam chosen on the original fit
split (0.2). On the old dev split T scored .6282 against A's .6265. Comparisons on v3: T vs J++, and A vs T. If A is
not better than T, the gain of A is a text-topic prior and not a reading of model-computed content.

## Results (9B scored 20:42Z, 27B scored 22:46-23:09Z on 2026-11-20; runs auto/runs/V3_*; 475 prompts for 9B, 512 for 27B)
PRIMARY: A - J++, pass@10 filtered, mean over 9B and 27B: 55.8 -> 57.9 = +2.1 [+0.5, +3.7] (bootstrap p ~ 0.012; 65
prompts better, 46 worse). The interval excludes zero: the registered claim holds, as on v2 (+2.2 [+0.6,+3.7]).
SECONDARY
- 9B: +3.7 [+1.2,+6.4] pass@10; +1.6 [-0.9,+4.3] pass@1; +3.6 [+1.3,+6.1] unfiltered.
- 27B (recipe A, amendment 2): +0.5 [-1.5,+2.5] pass@10; -2.2 [-4.3,-0.2] pass@1; +0.6 [-1.5,+2.7] unfiltered.
- Pooled pass@1: -0.3 [-2.0,+1.4]. A - shuffled: +7.1 [+5.7,+8.6]; shuffled - J++: -5.0 [-6.6,-3.4].
- G (generate-without-naming map, 9B): +0.1 [-1.2,+1.4]; AG: +0.7 [-0.6,+2.0]; G - G_shuf: +0.9 [-0.3,+2.1].
- v2 + v3 pooled: both models +2.1 [+1.0,+3.2] (pass@1 -0.2 [-1.4,+1.0]); 9B only +3.5 [+1.7,+5.4] (pass@1 +1.6
  [-0.2,+3.5]); 27B only +0.8 [-0.6,+2.1] (pass@1 -2.0 [-3.6,-0.5]).
- Still to score on v3 (registered): T (bag-of-visible-words control, amendment 4) and W (generated-data expert
  weights, amendment 3).
Reading: the gain is a 9B effect; on 27B there is no pass@10 gain and top-1 gets worse.

## Results, registered secondary W (amendment 3), scored 2026-11-21 00:38Z (9B, 475 v3 prompts)
- W (our recipe experts, weights fitted on 4,000 generated hidden-word sentences, no labels): .5268 / .2974, unf .5221.
- Same experts with weights fitted the authors' way on the 75 labelled prompts (A_full): .4746 / .2888, unf .4567.
- Released J++: .5194 / .2906, unf .4873.
- W - A_full: +5.2 [+2.2,+8.4] pass@10; +0.9 [-2.4,+4.3] pass@1; +6.5 [+3.4,+9.7] unfiltered.
- W - released: +0.8 [-1.8,+3.5]; +0.6 [-2.7,+4.1]; +3.5 [+0.8,+6.3] unfiltered.
- A_full - released: -4.4 [-7.0,-2.1].
Note: on the OLD dev split the order was reversed (W - A_full = -3.8 [-9.4,+1.8]); the labelled fit fold shares templates
with the old dev/test prompts, and its advantage does not carry to fresh prompts.

## Results, registered secondary T (amendment 4), scored 2026-11-21 00:46-00:53Z (9B, 475 v3 prompts; runs V3_T_9b, V3_T05_9b)
The standalone scorer reproduces V3_jpp_9b and V3_e2_A_9b exactly (0 of 5,152 rank rows differ).
- T (J++ + 0.2 * bag-of-visible-words map): .5286 / .3040, unf .5046. T at lam 0.5: .5301 / .2885.
- T - J++: +1.0 [-1.3,+3.4] (lam 0.5: +1.1 [-1.4,+3.9]). A - T: +2.7 [+1.1,+4.4]; unfiltered +1.8 [+0.2,+3.6].
- Concept slots gained over J++: A 46, T 25, both 17. Association: A 9, T 1. Multilingual: A 25, T 18 (11 shared).
Reading: on fresh prompts the visible-words map does NOT reproduce the ask-map's gain (it did on the old dev split,
.6282 vs .6265). About a third of A's gains are matched by visible words, mostly on multilingual prompts; the gain on
scene prompts needs the model's activations.

## Post-hoc replicates of the W comparison on v3 (9B; scored 2026-11-21 01:45-01:56Z; NOT registered, reported as replicates)
pass@10 / pass@1 / unfiltered pass@10. Released J++: .5189 / .2909 / .4870.
Generated-fit (label-free) weights: A experts, 4,000 sentences .5268; subsets Wa .5374, Wb .5343, Wc .5253; C experts .5454.
  vs released: +0.6 to +2.6 filtered (every interval spans 0); unfiltered +2.8 to +4.2 (every interval excludes 0).
Labelled (benchmark-fit) weights: A_full .4746, A_64a .4806, A_64b .4829 (each -3.6 to -4.4 vs released, intervals
  exclude 0); C_full .5259, C_64b .5117 (within +-1 of released).
  C experts: label-free - labelled = +1.9 [-1.3,+5.1].
Reading: the +5 over the labelled fit is reproducible for the raw-recipe experts (three labelled draws all near .48,
four label-free fits near .53). For the natural-format experts the two are close. Label-free weights are never worse
than the released lens on fresh prompts and are better unfiltered.

## Amendment 5 (2026-11-21 08:38Z, written and committed BEFORE any ask3 lens is scored on v2 or v3)
Why. Two faults were found in the registered ask-map A on 2026-11-21. (1) Prompt leak: formats E1 and E2 used seven
benchmark concept words as few-shot example answers (volcano, business, medicine, snow, music, football, weather).
(2) On Qwen3.6-27B the model's next token after the raw few-shot question was a blank line for 98/71/95% of prefixes
(it opens a think block), so the 27B map was fit on blank-line states.
Lens C (`auto/candidates/ask3_A.py`, both models): J++ + lam * E_t3. E_t3 is the same ridge recipe as A, with targets
elicited through the chat template (thinking off, assistant turn prefilled with the question label), three wordings
whose example answers are not on the benchmark concept list (fishing, politics, baking, kindness; eruption, finance,
illness, blizzard; symphony, exploration, sports, programming). lam is chosen on the FIT split from
{0.1,0.2,0.3,0.5,1.0,2.0} at build time (dev runs chose 0.2 on 9B and 0.5 on 27B). Control: `ask3_C_shuf.py`
(shuffled targets, same lam). Dev scores already seen: 9B C .6309/.3961 (A .6265/.4042, J++ .5952/.3850); 27B C
.6595/.3961 (A .6609/.3940, J++ .6153/.3731).
PRIMARY (one test on v3; the same test on v2 is the registered replication): C - J++, pass@10 with the readout
filter, macro over tasks, mean over Qwen3.5-9B and Qwen3.6-27B, paired bootstrap (items resampled within task, 5000
draws, same draws for all lenses).
SECONDARY (reported whatever they show): per model C - J++; C - A per model and pooled (A = the registered runs
V*_e2_A_9b and V*_m27_A_27b); pass@1; unfiltered pass@10; C - shuffled; v2 + v3 pooled.
Reading rule, fixed now. If C - J++ excludes zero and C - A is within noise on 9B, the leak caveat is removed and the
claim rests on C. If C - J++ on 9B does not exclude zero on either set, the registered A result is reported with the
caveat that part of it may come from the leaked example words. For 27B: a C - J++ interval excluding zero would make
the ask-map a two-model result; otherwise 27B stays "no gain".
My predictions: 9B C - J++ between +2 and +4, C - A within +-2; 27B C - J++ between 0 and +2, unfiltered better than A.
Scoring mode. Runs use `--lite` (scripts/perf1/lite_model.py: only the final norm, LM head and tokenizer are loaded;
unembed verified bitwise equal to the full model on 9B, /root/perf1/a9_check.log) because the 27B model no longer
fits beside other jobs. Guard: J++ (`jpp.py`) is rescored with --lite on each set and model first and must reproduce
the stored per-item ranks of V*_jpp_* exactly; if any differ, every run of this amendment is redone without --lite.
Run names: V{2,3}_jpplite_{9b,27b}, V{2,3}_C3_{9b,27b}, V{2,3}_C3shuf_{9b,27b}.

## Results, amendment 5 (scored 2026-11-21 09:35Z by scripts/c3_run.sh with --lite; bootstrap scripts/c3_boot.py -> runs/c3_boot.txt)
Guard: PASSED. V{2,3}_jpplite_{9b,27b} reproduce the stored J++ per-row ranks exactly (0 differing rows of 5,152-5,607
per run, filtered and unfiltered).
PRIMARY on v3: C - J++, pass@10 filtered, mean over 9B and 27B: 55.8 -> 57.3 = +1.5 [-0.0, +3.1] (p ~ 0.056). The
interval touches zero: the registered primary is NOT met on v3. Registered replication on v2: 65.8 -> 67.9 = +2.1
[+0.6, +3.6] (p ~ 0.004). v2 + v3: +1.8 [+0.8, +2.9].
SECONDARY
- 9B: pass@10 v3 +2.4 [+0.4, +4.8], v2 +3.1 [+1.1, +5.2], pooled +2.7 [+1.3, +4.3]; pass@1 v3 +2.6 [+0.7, +4.9], v2
  +3.1 [+1.1, +5.1], pooled +2.8 [+1.4, +4.3]; unfiltered pass@10 +4.0 / +4.1.
- 27B: pass@10 v3 +0.6 [-1.5, +2.9], v2 +1.1 [-1.0, +3.3], pooled +0.9 [-0.6, +2.5]; pass@1 v3 -3.1 [-5.4, -0.8], v2
  -1.8 [-4.6, +0.9], pooled -2.5 [-4.2, -0.8]; unfiltered pass@10 v3 +2.8 [+0.4, +5.3], v2 +1.9 [-0.2, +4.0], pooled
  +2.3 [+0.7, +4.0].
- C - A: 9B pass@10 v3 -1.3 [-3.0, +0.1], v2 -0.2 [-2.0, +1.6]; pass@1 +1.0 / +1.4 (ns). 27B pass@10 +0.1 / +0.1;
  unfiltered +2.2 [+0.7, +3.9] / +2.1 [+0.7, +3.7].
- C - shuffled: 9B +2.4 [+1.2, +3.8] / +2.9 [+1.0, +4.8]; 27B +7.6 / +5.4. Shuffled - J++: 9B -0.0 / +0.2; 27B
  -7.0 / -4.2.
- Prompts where C is better / worse than J++ (pass@10): v3 9B 33 / 11, v2 9B 33 / 8, v3 27B 29 / 35, v2 27B 32 / 30.
Reading, by the rule fixed in the amendment. (1) 9B: C - J++ excludes zero on both sets and C - A is within noise, so
the leak caveat is removed and the 9B claim rests on C: +2.7 [+1.3, +4.3] pass@10 and +2.8 [+1.4, +4.3] pass@1, with
the shuffled control flat. (2) 27B stays "no gain": pass@10 is flat and pass@1 is worse, with real (non-blank) targets
too; only the unfiltered score improves. Fixing the 27B targets did not rescue 27B. (3) Predictions: 9B within the
predicted +2..+4, C - A within +-2; 27B within the predicted 0..+2 and unfiltered better than A, as predicted.

## Amendment 6 (2026-11-21 10:41Z, written and committed BEFORE any fm27 lens is scored on v2 or v3)
Why. Amendment 5 showed C costs 27B its best guess (pass@1 -2.5 [-4.2, -0.8] over v2 + v3). Diagnosis on the FIT split
(scripts/fm27/results/): on 27B the map's effective weight is about 2.3x higher than on 9B, C's top-1 equals J++'s
top-1 in 41% of rows (9B 66%) and lies outside J++'s top-10 in 21% (9B 2%); the tokens that take rank 1 are format
tokens, generic topic words and passage-topic words.
Lens G (`auto/candidates/fm27_F5s.py`, both models): C with its registered weights (0.2 on 9B, 0.5 on 27B), except
that at each readout layer rank 1 may only go to a token in J++'s own filtered top-10 (the one with the highest C
logit is pushed strictly to rank 1). No free parameter, no labels. On 9B dev it is identical to C.
Lens L (`auto/candidates/fm27_F4.py`): C with the weight re-chosen on the FIT split by the largest pass@1 + pass@10
(dev runs chose 0.3 on 9B and 0.2 on 27B).
Dev scores seen: 27B G .6595/.3996, L .6489/.3996 (C .6595/.3961, J++ .6153/.3731); 9B G = C, L .6392/.4061.
PRIMARY (one test): G - C, pass@1 filtered, macro over tasks, on Qwen3.6-27B, mean over v2 and v3 (paired bootstrap,
5000 draws, items within task).
SECONDARY (reported whatever they show): on 27B G - J++ and L - J++ and L - C for pass@1, pass@10, unfiltered
pass@10, per set and pooled; on 9B G - C (expected zero) and L - C.
Reading rule, fixed now. If G - C on 27B pass@1 excludes zero on the positive side and G - J++ pass@1 no longer
excludes zero on the negative side, G becomes the recommended single definition (gain on 9B, no harm on 27B).
Otherwise 27B stays "the ask-map costs the best guess", and G is reported as a partial repair or as no repair.
My predictions: G - C pass@1 on 27B +0.5 to +1.5; G - J++ pass@1 still about -1 to -2; L on 27B pass@10 about +0.5
over J++ and pass@1 about 0. I do not expect either to turn 27B into a gain.
Scoring with --lite as in amendment 5 (guard already passed on these sets). Run names V{2,3}_G_{9b,27b},
V{2,3}_L_{9b,27b}.

## Results, amendment 6 (scored 2026-11-21 11:13Z by scripts/c4_run.sh with --lite; bootstrap scripts/c4_boot.py -> scripts/results/c4_boot.txt)
Weights chosen at build time, as on dev: L 0.3 on 9B and 0.2 on 27B; G 0.2 / 0.5 (fixed).
PRIMARY: G - C, pass@1 filtered, 27B, mean over v2 and v3: 34.9 -> 36.1 = +1.2 [+0.2, +2.2]. The interval excludes zero
on the positive side: the guard repairs part of the best-guess loss. G - J++ pass@1 on 27B: -1.3 [-2.9, +0.4], which no
longer excludes zero. By the rule fixed in the amendment, G is a valid single definition (gain on 9B, no significant
harm on 27B). G's pass@10 equals C's (27B -0.1 [-0.5, +0.2]; 9B identical).
SECONDARY
- L on 27B: L - J++ pass@10 v3 +1.6 [+0.1, +3.2], v2 +2.8 [+1.4, +4.4], pooled +2.2 [+1.2, +3.3]; pass@1 pooled +0.2
  [-1.0, +1.4]; unfiltered pass@10 +2.6 [+1.4, +3.9]; unfiltered pass@1 +1.5 [+0.2, +2.7]. L - C pass@1 +2.7 [+1.3,
  +4.1], pass@10 +1.3 [+0.1, +2.5].
- L on 9B: L - J++ pass@10 v3 +3.5 [+1.0, +6.3], v2 +3.7 [+1.4, +6.1], pooled +3.6 [+2.0, +5.4]; pass@1 +2.5 [+0.8,
  +4.3]; L - C pass@10 +0.9 [-0.1, +2.0], pass@1 -0.3 [-1.3, +0.7].
- L, mean over both models: pass@10 v3 +2.6 [+1.1, +4.1], v2 +3.3 [+1.9, +4.7], pooled +2.9 [+1.9, +4.0]; pass@1 pooled
  +1.4 [+0.3, +2.5]; unfiltered pass@10 +3.7 [+2.7, +4.8].
- G on 9B: G - C pass@1 -0.1 [-0.5, +0.2]; pass@10 identical.
Reading. (1) The registered primary is met: the guard helps 27B's best guess, as predicted (+0.5..+1.5), without
making 27B a gain. (2) My prediction for L was wrong: I expected about +0.5 pass@10 on 27B and it is +2.2 [+1.2, +3.3],
excluding zero on each fresh set, with the best guess unharmed. Choosing the weight on the FIT split by pass@1 +
pass@10 jointly, instead of by pass@10 alone, gives a lens that beats J++ on both models on both fresh sets. (3) Status
of that claim: L was a registered SECONDARY, one of two lenses in this amendment, on prompt sets that have now been
used for several registered comparisons. It should be confirmed as a PRIMARY on a new fresh set before it is called
settled.
