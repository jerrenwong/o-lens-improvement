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


## Results: Amendment 3 (scored 2026-11-24 08:40-10:20Z, once each, --lite; scripts/w2/heldout_runs.sh; full table runs/w2/heldout_table.md)
PRIMARY J++ + A-lens-CE minus J++ + ridge A-lens, pass@10 / pass@1 [95% CI], filter f1:
  9B v4  +2.6 [+1.2, +4.3] / +2.5 [+1.3, +3.8]        2B v4  +3.9 [+1.0, +7.6] / +4.7 [+2.5, +7.5]
  2B test +2.0 [+0.5, +4.0] / -0.0 [-1.7, +1.6]       (9B test: see table file)
  under f2: 9B v4 +1.3 [+0.4, +2.4] / +2.2 [+1.1, +3.6]; 9B test +1.2 [+0.0, +2.9] / +4.7 [+1.4, +8.9]; 2B v4 +3.6 [+0.7, +7.3] / +4.3 [+2.3, +7.1].
A-lens-CE alone minus ridge alone (f1): 9B v4 +10.0 [+7.2, +13.1] / +24.2 [+20.6, +28.0]; 2B v4 +12.4 [+8.4, +17.0] / +14.3 [+10.3, +18.8];
  under f2 the gap shrinks as predicted: 9B v4 +5.5 [+3.2, +8.0] / +6.8 [+4.7, +9.2]; 2B v4 +10.5 / +6.8.
J++ + A-lens-CE minus J++ (f1): 9B v4 +11.8 [+8.4, +15.2] / +3.9 [+1.4, +6.5]; 2B v4 +8.3 [+4.5, +12.8] / +6.5 [+3.5, +10.0].
Controls (f1, v4): J++ + CE minus shuffled CE: 9B +10.0 [+7.2, +13.1], 2B +6.0 [+3.8, +8.1]; minus constant CE: 9B +7.9 [+5.5, +10.4], 2B +4.0 [+1.5, +7.0].
Absolute (v4, f1, pass@10 / pass@1): 9B J++ 53.9/31.8, J++ + ridge 63.0/33.1, J++ + CE 65.7/35.7, CE alone 57.5/28.0 (above J++ at pass@10);
  2B J++ 37.9/20.8, J++ + ridge 42.3/22.7, J++ + CE 46.2/27.3, CE alone 45.7/22.8.
Against predictions: CE alone pass@1 gain >= 10 under f1: yes on v4 (+24.2, +14.3) and 2B test (+10.4); "J++ + CE minus J++ + ridge within
[-2, +3] with an interval including 0 on most cells": WRONG in the favourable direction: intervals exclude 0 on every v4 cell and the gain is
+2.6 to +3.9. Part (B) (competitor-subtracted readout) was not scored on held-out, as registered.


## Amendment 4 (2026-11-24 ~11:20Z, committed before scoring): Jacobian ("J++-style") A-lens, Qwen3.5-9B
To-do item "A-lens but using the J++ method rather than L2 loss". For each of the two A-lens questions a Jacobian is fitted from the
passage tokens to the final-layer state at the answer position of the question prompt (scripts/ja1, scripts/ao1, scripts/w3; 128
passages; J++ backward rules). Two versions: "full recipe" (8 k-means experts + expert weights fitted on the labelled FIT split:
runs/ao1/9b/lens/topA5_n128.pt + runs/w3/9b/lens/unsA5_n128.pt) and "K1" (one pooled matrix per question, label-free: topK1_n128.pt +
unsK1_n128.pt). Jacobian A-lens = mean of the two unit-normalised outputs (auto/candidates/w3_alone.py); added to J++ with lam chosen on
FIT by the frozen alens rule (w3_add.py). Held-out, once each, --lite, scripts/w3/heldout_runs.sh: test + v4, filters f1 (primary) and f2;
rows Jacobian A-lens alone and J++ + Jacobian A-lens, for full recipe and K1. Reference rows on the same items: w1_9b_{JPP0,none,JPP}_<f>_<set>
(J++, ridge A-lens alone, J++ + ridge A-lens) and w2ho_9b_ce_{none,JPP}_<f>_<set> (cross-entropy A-lens).
Comparisons: PRIMARY J++ + Jacobian A-lens (full) minus J++ + ridge A-lens; the same for K1; Jacobian alone minus ridge alone and minus
cross-entropy alone; J++ + Jacobian minus J++.
Seen beforehand (dev, f1, top-10 / top-1): J++ 59.5/38.5; ridge alone 53.0/6.3, + J++ 62.8/42.0; Jacobian full alone 54.3/35.1, + J++
60.9/40.1 (lam 1.0); K1 alone 47.8/26.2, + J++ 58.5/35.3 (lam 0.5); single questions (full): topic 48.3/31.1, unstated 53.8/38.5.
2B (64 passages, dev): no Jacobian version improves on J++. Outputs: the Jacobian version never ranks " **"/<|im_end|> first; it lists
case/language variants of passage words.
Caveats: the full recipe uses labelled fit prompts for its expert weights (the ridge and cross-entropy A-lens do not); 9B only (the 2B
248-passage fit was still running at registration and is not part of this registered run).
Predictions: J++ + Jacobian (full) minus J++ + ridge is negative or within +-2 at pass@10 on both sets; K1 + J++ does not beat J++;
Jacobian full alone beats ridge alone at pass@1 by > 15 points under f1 and is below cross-entropy alone at pass@10.


## Results: Amendment 2 (stronger filter levels; scored 2026-11-24 07:38-11:05Z, once each, --lite; full table runs/w1/heldout_table.md)
J++ + A-lens minus J++, pass@10 [95% CI], v4 (new prompts):          f1                  f2                   f3
  Qwen3.5-2B                                                    +4.4 [+2.3, +7.3]   +4.6 [+2.4, +7.5]    +4.6 [+1.9, +8.1]
  Qwen3.5-9B                                                    +9.1 [+6.1, +12.5]  +10.4 [+7.2, +13.8]  +7.2 [+4.6, +10.0]
  Qwen3.6-27B                                                   +4.8 [+2.5, +7.2]   +5.0 [+2.8, +7.4]    +2.1 [+0.6, +3.8]
  Olmo-3-7B                                                     +9.8 [+5.4, +14.6]  +9.8 [+5.8, +14.3]   +11.9 [+7.5, +16.6]
  gemma-4-E4B                                                   +5.3 [+2.6, +8.4]   +9.2 [+5.6, +13.1]   +9.6 [+5.9, +13.6]
test split: 2B +12.6 / +12.6 / +6.4 [+0.9, +13.3]; 9B +5.0 / +5.7 / +7.1; 27B +2.0 [-1.0, +5.1] / +2.1 [-0.8, +5.3] / +3.2 [+0.4, +6.6];
  Olmo +4.2 / +4.2 / +10.6; Gemma +4.3 / +10.7 / +10.5 (all intervals exclude 0 except 27B f1 and f2).
The f1 column reproduces the earlier registered numbers exactly. The gain is positive on all 30 model x set x level cells and its interval
excludes 0 on 28 (27B test f1, f2 do not). pass@1 differences are mostly within noise; Gemma test stays negative (-2.9 to -3.4).
A-lens alone, pass@1 on v4, f1 -> f2: 9B 3.8 -> 21.2; Gemma 5.3 -> 12.6 (prediction "rises by at least 5 points": met where checked).
Against predictions: gain positive under f2 and f3 on all five models: yes. "Grows on Gemma and Olmo": Gemma yes (+5.3 -> +9.2 -> +9.6);
Olmo unchanged under f2, larger under f3. Not predicted: on 27B the gain shrinks under f3 (+4.8 -> +2.1).


## Results: Amendment 4 (Jacobian A-lens, Qwen3.5-9B; scored 2026-11-24 11:20-11:27Z, once each, --lite; table runs/w3/heldout_table.md)
pass@10 / pass@1, f1:            test (153)        v4 (482)
  J++                            58.1 / 35.4       53.9 / 31.8
  ridge A-lens alone             49.8 /  2.6       47.5 /  3.8
  cross-entropy A-lens alone     55.0 / 28.0       57.5 / 28.0
  Jacobian A-lens alone, full    51.4 / 35.2       58.6 / 35.7
  Jacobian A-lens alone, K1      45.7 / 25.0       57.6 / 33.5
  J++ + ridge A-lens             63.1 / 35.8       63.0 / 33.1
  J++ + cross-entropy A-lens     64.5 / 40.7       65.7 / 35.7
  J++ + Jacobian A-lens, full    57.9 / 39.7       61.0 / 39.7
  J++ + Jacobian A-lens, K1      60.5 / 37.6       61.9 / 36.0
PRIMARY J++ + Jacobian (full) minus J++ + ridge: test -5.2 [-10.6, -0.4] / +3.9 [-0.9, +9.2]; v4 -2.1 [-5.5, +1.2] / +6.6 [+3.9, +9.4].
K1: test -2.6 [-6.1, +0.8] / +1.8; v4 -1.1 [-4.7, +2.4] / +2.9 [+0.7, +5.3].
J++ + Jacobian (full) minus J++: test -0.2 [-4.7, +3.6] / +4.3 [-0.5, +9.1]; v4 +7.1 [+3.9, +10.3] / +7.9 [+5.1, +10.8]. K1 minus J++: v4 +8.0 [+5.1, +11.0] / +4.2 [+1.9, +6.8].
J++ + Jacobian (full) minus J++ + cross-entropy: test -6.6 [-12.0, -1.8] / -1.0; v4 -4.7 [-7.9, -1.6] / +4.0 [+1.2, +6.9].
Alone: Jacobian full minus ridge: v4 +11.1 [+7.1, +15.1] / +31.9 [+28.6, +35.4]; minus cross-entropy: v4 +1.0 [-3.0, +5.1] / +7.7 [+3.8, +11.5].
f2: same pattern (v4: J++ + Jacobian full minus J++ + ridge -3.5 [-6.8, -0.4] / +5.8 [+3.0, +8.6]).
Against predictions: "J++ + Jacobian (full) minus J++ + ridge negative or within +-2 at pass@10": yes (-5.2, -2.1). "K1 + J++ does not beat J++":
WRONG on v4 (+8.0 [+5.1, +11.0]); it held on dev. "Jacobian full alone beats ridge alone at pass@1 by > 15 under f1": yes (+32). "Below
cross-entropy alone at pass@10": not on v4 (+1.0, ns), yes on test (-3.6, ns).
Reading: the Jacobian A-lens trades pass@10 for pass@1. Added to J++ it has the best first guess of any combination on v4 (39.7) but fewer
concepts in the top 10 than the ridge or cross-entropy A-lens. The label-free K1 version is close to the full recipe on v4.


## Amendment 5 (2026-11-24 ~15:20Z, committed before scoring): the cross-entropy and Jacobian A-lens rows under filter f3
The user asked for every reported table to use the full token filter (f3 of Amendment 2). Amendments 3 and 4 scored their rows under
f1 and f2 only. This adds, once each, --lite (scripts/f3_heldout.sh): cross-entropy A-lens K=10 (alone, + J++, and the shuffled and
constant controls + J++) on Qwen3.5-2B and 9B, and the Jacobian A-lens (full recipe and K1; alone and + J++) on Qwen3.5-9B; test and v4;
filter f3; lam re-chosen on FIT with ranks over token groups (auto/candidates/w1_alens.py with ALENS_E/ALENS_E2 set to the CE maps;
auto/candidates/w3_f.py). Reference rows: w1_<tag>_{JPP0,none,JPP}_f3_<set>. Same comparisons as Amendments 3 and 4.
Nothing is selected on this run; it changes the scoring rule of already registered comparisons. It is a further use of both held-out sets.
Prediction: same signs as under f2 (J++ + CE above J++ + ridge by 1-4 points at pass@10; J++ + Jacobian full below J++ + ridge at pass@10,
above at pass@1).


## Amendment 6 (2026-11-24 ~15:55Z, committed before scoring): full filter f4 = f3 + markup fragments; all reported rows
The user asked that markup fragments ("&quot", "<b", "&amp") be removed as well as "@" tokens, and that every reported table use the
full filter. f4 (auto/filters.py) = f3 plus every token whose decoded text contains "&", "<" or ">", minus accepted answers of scored
items; ranks over word groups as in f3. Defined before any f4 number was seen; nothing is tuned on it.
Scored once each, --lite, test and v4 (scripts/f4_heldout.sh): (a) the five A-lens-family rows of Amendment 2 on all five models
(w1_<tag>_<row>_f4_<set>); (b) the cross-entropy A-lens rows and controls of Amendment 3 on 2B and 9B (w2ho_<tag>_<var>_<row>_f4_<set>);
(c) the Jacobian A-lens rows of Amendment 4 on 9B (w3ho_9b_ja<v>_<mode>_f4_<set>). lam re-chosen on FIT under f4 (ranks over groups).
Same comparisons as Amendments 2-4. This is a further use of both held-out sets; f1 remains the literature-comparable filter, f4 is
our reporting filter from now on. Prediction: f4 differs from f3 by under 1 point on almost every cell (few top-10 entries are markup).


## Results: Amendments 5 and 6 (f3 for CE/Jacobian; full filter f4 for all rows; scored 2026-11-24 15:20-16:35Z, once each, --lite)
Numbers: runs/f4_page.json (scripts/f4_page_numbers.py), runs/f3_page.json. f4, v4 (new prompts), pass@10 / pass@1:
                2B            9B            27B           Olmo-7B       Gemma-E4B
  J-lens        35.5 / 17.8   40.5 / 16.6   41.7 / 19.7   32.3 / 12.6   37.4 / 22.3
  J++           38.8 / 21.1   58.2 / 32.1   61.2 / 35.1   36.7 / 15.4   43.6 / 25.5
  A-lens alone  36.9 / 16.0   52.2 / 21.2   50.7 / 17.3   45.0 / 20.9   35.3 / 12.7
  J + A-lens    45.3 / 20.2   58.7 / 26.6   59.2 / 29.6   51.7 / 22.0   50.2 / 23.9
  J++ + A-lens  43.3 / 25.8   65.8 / 34.1   63.3 / 36.9   48.5 / 23.0   53.2 / 26.2
J++ + A-lens minus J++ (pass@10): v4 +4.6 [1.9, 8.1], +7.6 [5.0, 10.7], +2.1 [0.6, 3.8], +11.9 [7.5, 16.6], +9.6 [5.9, 13.6];
  test +6.4 [0.9, 13.3], +7.1 [2.8, 12.0], +3.2 [0.4, 6.6], +10.6 [4.1, 18.1], +10.5 [4.9, 17.0]: all ten intervals exclude 0.
  pass@1 v4: +4.7 [1.4, 8.3], +2.0 [-0.6, 4.5], +1.9 [0.5, 3.3], +7.6 [4.5, 11.3], +0.7 [-4.0, 5.0].
Cross-entropy A-lens (f4): J++ + CE minus J++ + ridge: 9B v4 +2.6 [0.9, 4.3] / +1.4 [-0.1, 2.9]; 2B v4 +3.5 [1.1, 6.4] / +1.7 [-1.3, 4.0];
  9B test -0.8 [-3.2, 1.5] / +2.1 [-1.8, 6.4]; 2B test +3.5 [-0.5, 9.0] / -0.5. J++ + CE: 9B v4 68.4 / 35.5, 2B v4 46.8 / 27.5.
  CE alone minus ridge alone, v4: 9B +5.9 [3.6, 8.5] / +7.0 [4.9, 9.4]; 2B +9.5 [6.1, 13.6] / +6.8 [3.8, 10.3].
  J++ + CE minus shuffled control, v4 pass@10: 9B +14.3 [11.2, 17.6]; 2B +6.3 [4.2, 8.4].
Jacobian A-lens (9B, f4, v4): alone full 61.5 / 35.9, K1 59.1 / 33.5; + J++ full 63.9 / 40.0, K1 62.9 / 34.8;
  J++ + Jacobian full minus J++ + ridge: -1.9 [-5.1, 1.0] / +5.9 [3.2, 8.8]; minus J++ + CE: -4.5 [-7.7, -1.5] / +4.6 [1.7, 7.6].
Against predictions: f4 within 1 point of f3 on almost every cell: yes (largest change seen: 9B v4 J++ + A-lens 65.3 -> 65.8).
Under the full filter the CE advantage over the ridge A-lens added to J++ stays clear of 0 at pass@10 on v4 for both models, but is no
longer clear at pass@1 and vanishes on the 9B test split: part of the earlier f1 advantage was junk that the scoring filter now removes.

## Amendment 7 (registered 2026-11-24, before scoring): cross-entropy A-lens fitted on ~98k passages (Experiment 3.4)
Question: does fitting the linear cross-entropy A-lens (K=10, same recipe as Amendment 3) on 98,000 public passages instead of 28,500
improve held-out scores? Configuration chosen on dev only (runs/w9/table.md): lin_ce_98k, maps .hf27/w9/maps/E_<tag>_{topic,spec}_lin_ce_98k.pt,
candidate auto/candidates/w9_alens.py, lam by the frozen fit-split rule at the filter level. The MLP map is NOT taken to held-out
(dev gain over linear rests on 1-2 items; most layers early-stopped at step 0).
Runs (once each, --lite, filter f4): scripts/w9/heldout.sh 2b 9b -> rows A-lens alone and J++ + A-lens, test and v4.
Comparisons (paired bootstrap, 5000 draws, pass@10 and pass@1): against the 28.5k cross-entropy rows and the ridge rows already scored
under f4 (Amendment 6: runs w2/f4 `ce`, `JPP+ce`, `ridge`, `JPP+ridge`) and J++ alone.
Dev expectation, stated before scoring: J++ + CE-98k minus J++ + CE-28.5k about +1.5 to +2 points pass@10, intervals likely touching 0;
alone rows about equal. No thresholds; sizes and intervals reported as they come. Passages overlapping any benchmark prompt text were
excluded when building the set (runs/w9/data_card.md).

## Results: Amendment 7 (CE A-lens on 98k passages; scored once 2026-11-24 ~20:20-20:50Z, f4, --lite; runs/w9/heldout.json)
pass@10 / pass@1        9B test       9B v4         2B test       2B v4
  J++                   59.1 / 37.4   58.2 / 32.1   36.2 / 18.0   38.8 / 21.1
  J++ + ridge A-lens    66.2 / 38.6   65.8 / 34.1   42.6 / 20.1   43.3 / 25.8
  J++ + CE 28.5k        65.4 / 40.7   68.4 / 35.5   46.1 / 19.7   46.8 / 27.5
  J++ + CE 98k          67.8 / 39.2   71.4 / 37.9   50.0 / 20.9   47.8 / 28.1
  CE 28.5k alone        55.4 / 29.7   58.2 / 28.2   47.1 / 17.1   46.4 / 22.8
  CE 98k alone          57.6 / 31.2   61.1 / 33.3   45.8 / 15.8   50.3 / 23.8
J++ + CE98 minus J++ + CE28: pass@10 +2.4 [0.0, 5.0], +3.0 [1.2, 5.1], +3.9 [0.3, 9.4], +1.0 [0.3, 1.7];
                             pass@1  -1.4 [-5.4, 1.8], +2.5 [1.0, 3.9], +1.3 [0.0, 3.8], +0.6 [-0.1, 1.4].
J++ + CE98 minus J++ + ridge: pass@10 +1.6 [-0.9, 4.4], +5.6 [3.2, 8.2], +7.3 [1.5, 13.8], +4.5 [2.0, 7.4].
J++ + CE98 minus J++: pass@10 +8.7 [3.8, 14.2], +13.2 [9.9, 16.8], +13.8 [6.6, 20.8], +9.0 [5.2, 13.5].
Alone, CE98 minus CE28: pass@10 +2.3 [-1.4, 6.7], +2.9 [0.3, 5.6], -1.3 [-10.2, 6.7], +3.9 [0.9, 7.4].
Against the dev expectation (+1.5 to +2, intervals touching 0): held-out gain is the same sign in all four cells and a little larger; three of four
pass@10 intervals exclude 0 and the fourth touches it. Note: the first launch of this run began before the registration commit had succeeded;
it was stopped before any evaluator wrote a result, the commit (b8cdb18) was pushed, and the run was restarted.

## Amendment 8 (registered 2026-11-25, before scoring): Experiment 3 five-column table on held-out
Columns (all K=10, T=1; maps .hf27/w7/E_<tag>_{topic,spec}_<mode>_k10.pt): C1 residual state (ridge; already scored under f4, Amendment 6),
C2 probability, no target filter (`all`); C3 probability + junk filter (`nomerge`; identical maps to the Amendment 3 CE maps, already
scored under f4: runs w2ho_*_ce_*_f4_*; NOT rescored); C4 probability + merged duplicates (`mergeall`); C5 probability + junk filter +
merged duplicates (`merge`). New runs, once each: COLS="all mergeall merge" LVS=f4 bash scripts/w7/heldout_runs.sh {2b,9b} -> rows
A-lens alone and J++ + A-lens, test and v4 (the script also recomputes J++ alone under a new run name; that row is deterministic and
already scored, and its earlier value is the one reported). Paired bootstrap 5000 draws for C2-C1, C3-C2, C4-C2, C5-C2, C5-C3.
Dev expectation, stated before scoring (runs/w7/boot_main_f4.txt): under f4, with J++, all five within about 1-2 points of each other with
intervals including 0; alone, C2/C3 above C1 at top 1; merging (C4, C5) neutral or slightly negative. No column is selected by this run;
the sweeps over K and temperature were flat on dev and are not taken to held-out.
