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

## Results: google/gemma-4-E4B-it, the registered reserve (scored 2026-11-22 23:42Z, once each, --lite; scripts/xm1_heldout_gemma4e4b.sh; scripts/results/xm1_gemma4e4b_boot.txt)
Benchmark kept: test 119 items (association 41, multihop 11, multilingual 26, poetry 3, typo 38); v4 367. Format check passed
(1.1% topic, 10.0% unstated pooled); no fallback used. lam (FIT): L 1.0 (top of the grid); J++ + A 0.5; J + A 2.0.
pass@10 / pass@1:            test            v4
  J-lens                     .333 / .155     .357 / .221
  J++ (unit)                 .366 / .241     .430 / .255
  L = J++ + ask-map          .392 / .184     .488 / .238
  shuffled-answer control    .343 / .164     .440 / .170
  constant-only control      .374 / .185     .452 / .175
  A-lens alone               .290 / .041     .312 / .053
  J-lens + A-lens            .380 / .193     .475 / .226
  J++ + A-lens               .409 / .212     .483 / .259
PRIMARY L - J++ pass@10: test +2.6 [-2.5, +7.1]; v4 +5.8 [+3.0, +9.0]. L - shuffled: +4.9 [+0.1, +9.0], +4.8 [+1.3, +8.4];
L - constant: +1.9 [-3.7, +6.7], +3.7 [+0.4, +6.7]. By the registered reading: SUPPORTED on v4; not on the test split.
pass@1: L - J++ test -5.7 [-10.5, -2.0] (a LOSS), v4 -1.7 [-5.7, +1.7]; the controls lose more (-5.6 to -8.4).
Predictions: +1..+6 on v4 -> +5.8 (inside); controls within +-1.5 -> +1.0, +2.1 on v4 (one just outside, ns).
Secondary: J++ + A - J++: test +4.3 [+2.0, +6.9], v4 +5.3 [+2.6, +8.4] (pass@1 -2.9 [-6.9, -0.4], +0.4 ns);
J + A - J: +4.8 [-1.5, +11.2], +11.8 [+8.6, +15.4]; A alone - J: -4.3 ns, -4.5 ns; A alone - J++: -7.6, -11.8.
Note: poetry has 3 test items for this model (it answers 8 of 98 couplets correctly).

## Summary across the two new families (v4, pass@10, L - J++): Olmo-3-7B +9.8 [5.4, 14.7]; gemma-4-E4B +5.8 [3.0, 9.0];
both beat both controls on v4. On the small original test splits: +2.6 [0.5, 5.6] and +2.6 [-2.5, 7.1].


## Amendment 2 (2026-11-24 ~07:35Z, committed before any held-out scoring under f2/f3): stronger token filtering, all lenses
Why: the benchmark filter (f1: tokens with no letter or digit, minus accepted answers of all six tasks) lets " **", "<|im_end|>",
"@g"-style tokens through. They are the A-lens's commonest first guesses (" **" first in 503/525 fit rows on 9B), so f1 treats the
lenses unequally. The user asked for a stronger filter applied to every lens, including tokens containing "@", " **" and <|im_end|>.
Levels (auto/filters.py, fixed before any f2/f3 number was seen; the "@" rule was added by the user before any f2/f3 run):
  f1 = current benchmark filter (unchanged; reproduces 29/29 stored runs exactly).
  f2 = f1 + all special/added tokens + symbol-only tokens that f1 spared only for the unused sixth task + any token containing "@";
       an accepted answer of an item in the five scored tasks is never masked (the spared set is empty in practice).
  f3 = f2 + count each word once: tokens grouped by normal form (strip whitespace and one leading _ / U+2581 / U+0120, lower-case,
       drop one trailing "s" if >= 3 letters remain); rank = 1 + number of other groups whose best member beats the concept's best
       candidate. Known side effect of the plural rule: "class" -> "clas".
Rows (frozen auto/candidates/alens.py through the wrapper auto/candidates/w1_alens.py): J-lens (J0), J++ (JPP0), A-lens alone (none),
J-lens + A-lens (J), J++ + A-lens (JPP). lam is re-chosen on the FIT split under each level by the frozen rule.
Models: Qwen3.5-2B, Qwen3.5-9B, Qwen3.6-27B, Olmo-3-7B-Instruct, gemma-4-E4B-it. Sets: original test split and v4 (614 prompts).
All three levels are reported for every row, model and set; no level is selected. Scored once each, --lite,
scripts/w1/heldout_runs.sh, run names w1_<tag>_<row>_<level>_{test,v4}. Bootstrap: scripts/r2_boot.py (5000 paired draws).
Comparisons reported per level: J++ + A-lens minus J++ (primary); J-lens + A-lens minus J-lens; A-lens alone minus J-lens;
A-lens alone minus J++; and, for each row, f2 minus f1 and f3 minus f1 (unpaired across filters is not meaningful for the bootstrap
as coded, so level effects are reported as point differences with each level's own interval for the lens-vs-lens comparisons).
Seen beforehand (dev split, runs/w1/dev_table.md): f2 lifts the A-lens rows' top-1 a lot (9B A-lens alone .063 -> .296) and their
top-10 by 2-5 points; J-lens / J++ rows barely move under f2 and gain 1-4 points top-10 under f3. J++ + A-lens minus J++ on dev,
top-10: f1 2B -1.1, 9B +3.3, 27B +6.4, Olmo +10.3, Gemma +7.0; f3 2B +2.0, 9B +4.2, 27B +3.9, Olmo +15.3, Gemma +15.4.
Caveats: these held-out sets have been scored before with f1 (this is a further use); f2/f3 are our own scoring rules, not the
benchmark's; f1 numbers remain the comparable-to-literature ones. Predictions (v4, top-10, J++ + A-lens minus J++): the f1 values
repeat the earlier registered run (+4.4 to +9.8); under f2 and f3 the gain stays positive on all five models and grows on the
models where " **"/<|im_end|> crowd the A-lens output most (Gemma, Olmo); A-lens alone top-1 rises by at least 5 points under f2.


## Amendment 3 (2026-11-24 ~08:35Z, committed before scoring): cross-entropy A-lens ("A-lens-CE"); competitor-subtracted readout not sent
To-do item "fit probability instead of logits". Two things were tried on dev/fit only (scripts/w2/, report in the card):
(A) A-lens-CE: each of the two A-lens maps is trained with KL(p || softmax over the whole vocabulary of the decoded map output),
    p = the model's own answer distribution restricted to word tokens, top-K, temperature 1 (st1 P3v recipe, now also for the
    unstated-word question); A-lens-CE = mean of the two unit-normalised map outputs, combined exactly as in the frozen alens.py
    (auto/candidates/w2_alens_ce.py; maps .hf27/w2/E_<tag>_{topic,spec}_ce_k10.pt). K chosen on dev among {10, 50, 200}: K = 10.
(B) Competitor-subtracted readout (score_X = z_X - alpha * sum_Y q(Y|X) z_Y from co-prediction statistics): null or harmful on dev for
    every lens and every cut (top-k 5/20, top-p 0.9, temperature 0.5, words only); fit-chosen alpha is 0 in most cells and every
    alpha > 0 lowers dev pass@10 except one cell. DECISION: (B) is NOT scored on held-out; it is reported as a dev-split null.
Held-out run for (A), once each, --lite, scripts/w2/heldout_runs.sh: models Qwen3.5-2B and Qwen3.5-9B; sets test and v4; filters f1
(primary, the benchmark's) and f2 (Amendment 2); rows A-lens-CE alone, J++ + A-lens-CE (lam on FIT, frozen rule), and two controls
combined with J++: shuffled-target CE map, constant-only CE map. Reference rows from Amendment 2's run (same items):
w1_<tag>_{none,JPP,JPP0}_<level>_<set> = ridge A-lens alone, J++ + ridge A-lens, J++.
Comparisons (r2_boot.py, 5000 paired draws), per model, set and filter: PRIMARY J++ + A-lens-CE minus J++ + ridge A-lens (pass@10 and
pass@1); A-lens-CE alone minus ridge A-lens alone; J++ + A-lens-CE minus J++; J++ + A-lens-CE minus each control.
Seen beforehand (dev, f1, top-10 / top-1): 9B J++ .595/.385; ridge alone .530/.063, CE alone .572/.332; J++ + ridge .628/.420,
J++ + CE .638/.430; controls + J++ .615/.393 and .613/.393. 2B J++ .550/.289; ridge alone .421/.143, CE alone .464/.280;
J++ + ridge .539/.302, J++ + CE .557/.322; controls .489/.313 and .512/.308. CE maps never output a non-word first (ridge: 68-97%).
Caveats: K was chosen on dev (156 / 109 items); the dev gain over the ridge version when added to J++ is only 1-2 points; under f2
the ridge A-lens loses most of its non-word first guesses anyway, so the CE advantage may shrink there. 27B not fitted.
Predictions: A-lens-CE alone beats ridge alone at pass@1 by >= 10 points on both sets and models under f1 and by less under f2;
J++ + CE minus J++ + ridge is within [-2, +3] at pass@10 with an interval that includes 0 on most cells.
