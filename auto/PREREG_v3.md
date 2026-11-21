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
