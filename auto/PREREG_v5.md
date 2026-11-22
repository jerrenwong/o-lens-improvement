# PREREG v5: the ask-map lens on models outside the Qwen family (2026-11-22 19:00Z; written and committed before any new model is run)

## Why
Every result so far is on Qwen3.5-2B, Qwen3.5-9B and Qwen3.6-27B, and the recipe was developed on them. This tests the
frozen recipe on two models from other families: allenai/Olmo-3-7B-Instruct and google/gemma-4-12B-it (both supported by
the J++ authors' LRP code). If one cannot be run for a technical reason, google/gemma-4-E4B-it is the reserve.

## Frozen recipe (nothing below is tuned on the new models)
1. Baseline J++: fitted for the new model with the J++ authors' code and default recipe (64 WikiText prompts, rlens
   rules, K=8 experts, expert weights on the benchmark's fit:0 items), readout layers = 7 evenly spaced blocks
   (depth/8, 2*depth/8, ... 7*depth/8). The J-lens (K=1, plain rules) is fitted alongside for reference.
2. Benchmark: the same five scored tasks and the same item -> fit/dev/test assignment by item name as for Qwen; items
   kept by the benchmark's own correctness rule applied to the new model. The 614 v4 prompts filtered the same way.
3. Ask-map: the 30,500 stored passages (runs/e1/prefixes.json); the three "One-word topic:" wordings W0/W1/W2 of
   scripts/ask3/ask3_collect4.py with the same clean example answers; the model's OWN chat template via the tokenizer,
   thinking/reasoning disabled where the template has such a switch, assistant turn pre-started with the label so the
   next token is the answer. Target = final-layer state at that position. Ridge map per readout layer with the grid
   and validation rule of scripts/ask3/ask3_fit9.py.
   Format check (label-free, decided here): if in more than 20% of passages the model's top first-answer token is not a
   word, the format is declared failed for that model, ONE fallback is allowed (the same prompt without the pre-started
   label, reading at the first generated answer token position after the template's assistant header), and both
   outcomes are reported. No other format changes.
4. Lens L = unit(J++ h) + lam * unit(E h + b); lam chosen on the model's FIT split by the largest pass@1 + pass@10
   over {0, .05, .1, .2, .3, .5, 1}. Reference = unit-scaled J++ (lam = 0).

## Scoring
Agents score the DEV split only. The orchestrator scores, once each and with --orchestrator: the original benchmark's
TEST split and the v4 prompts, for unit-scaled J++, L, and two controls at L's own lam (shuffled-answer map; constant-
only map). Paired bootstrap, 5000 draws within task.

## Primary outcome
L minus J++, pass@10, on the original test split and on v4, per new model. Secondary: pass@1; L minus each control.

## Reading the result
Supported for a model if L minus J++ is positive with an interval excluding 0 on v4 (the larger set) and L minus both
controls is positive. Not supported if the interval includes 0 or the controls match L. Either outcome is reported.
Predictions: Olmo-3-7B +2 to +7 on v4; gemma-4-12B +1 to +6; controls within +-1.5.

## Amendment 1 (2026-11-22 20:09Z; before any new model has been scored on its test split or on v4)
Added at the user's request, as registered SECONDARY rows for each new model: the two-question A-lens family frozen in
auto/candidates/alens.py (PREREG_v4 addendum E): A-lens = plain average of the topic map and an "Unstated word:" map.
The unstated-word map is built exactly like the topic map (same passages, the model's own chat template, same ridge
grid) with the wordings and clean example answers of scripts/ask5/ask5_collect.py; the same label-free format check
and single fallback apply. Rows scored once on the test split and on v4: J-lens, J++, A-lens alone, J-lens + A-lens,
J++ + A-lens (lam on FIT, grid {0,.05,.1,.2,.3,.5,1,2}). The PRIMARY outcome stays L (topic map only) minus J++.

## Results: allenai/Olmo-3-7B-Instruct (scored 2026-11-22 22:44Z, once each, --lite; scripts/xm1_heldout_olmo7b.sh; intervals in scripts/results/xm1_olmo7b_boot.txt)
Benchmark kept: test 112 items (association 38, multihop 6, multilingual 10, poetry 20, typo 38); v4 340. Format check passed
(non-word first token 2.2% topic, 11.5% unstated); no fallback used. lam (FIT): L 0.2; J++ + A 0.2; J + A 0.5.
pass@10 / pass@1:            test            v4
  J-lens                     .314 / .106     .296 / .123
  J++ (unit)                 .385 / .265     .312 / .154
  L = J++ + ask-map          .410 / .281     .410 / .199
  shuffled-answer control    .400 / .194     .343 / .127
  constant-only control      .400 / .189     .324 / .139
  A-lens alone               .372 / .042     .412 / .052
  J-lens + A-lens            .448 / .228     .431 / .188
  J++ + A-lens               .427 / .286     .410 / .195
PRIMARY L - J++ pass@10: test +2.6 [+0.5, +5.6]; v4 +9.8 [+5.4, +14.7] (pass@1: +1.6 [-7.5, +11.1]; +4.5 [+2.2, +6.7]).
L - shuffled: test +1.1 [-1.0, +3.1], v4 +6.8 [+3.4, +10.8]; L - constant: test +1.1 [-0.9, +3.2], v4 +8.6 [+4.7, +13.1]
(pass@1: L beats both controls on both sets, +6.0 to +9.2, all intervals above 0).
By the registered reading: SUPPORTED on v4 (interval above 0, both controls beaten). On the test split L beats J++ but
not the controls at pass@10 (it does at pass@1). Predictions: +2..+7 on v4 -> observed +9.8 (above); controls within
+-1.5 -> shuffled +3.1 [-1.0, +7.5] on v4 (outside the band, interval includes 0), others inside.
Secondary (amendment 1): J++ + A - J++: test +4.2 [+1.0, +8.6], v4 +9.8 [+5.4, +14.6]; J + A - J: +13.5 [5.5, 22.2],
+13.5 [9.4, 18.3]; A alone - J++: test -1.3 [-12.5, +10.3], v4 +10.0 [+2.8, +16.9]; J + A - J++: test +6.4 [+0.4, +14.0],
v4 +11.9 [+5.5, +18.4] (pass@1 -3.7 ns, +3.4 [+0.1, +7.1]); J++ + A - L: +1.7, +0.0.
Notes: J++ for Olmo is our fit with the authors' default recipe (no released Olmo lens); it is weak on association
(3%). Olmo answers few multihop/multilingual prompts correctly, so those task cells have 6-23 items.
google/gemma-4-12B-it could not be run (loads as Gemma4UnifiedForConditionalGeneration; the authors' LRP patcher
rejects its MLP class) -> the registered reserve google/gemma-4-E4B-it is used.
