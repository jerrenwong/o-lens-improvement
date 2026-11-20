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
