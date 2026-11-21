# Pre-registration: third fresh-prompt test (v4). Written 2026-11-21 11:14Z, before any v4 prompt exists.

Why. On v2 and v3 the lens L (PREREG_v3 amendment 6, a registered SECONDARY) beat J++ on both models: pass@10 +2.9
[+1.9, +4.0] mean over Qwen3.5-9B and Qwen3.6-27B, 27B alone +2.2 [+1.2, +3.3]. Those two sets have been used for
several registered comparisons, and L was one of two lenses in its amendment. v4 tests L once, as the primary.

Lens L, frozen: `auto/candidates/fm27_F4.py` at commit 4ea31e4. J++ plus the clean chat-format ask-map
(.hf27/ask3/E_{9b,27b}_t3.pt), v = unit(JPP h) + lam * unit(E h + b), lam chosen at build time on the ORIGINAL fit
split from {0, .05, .1, .2, .3, .5, 1.0} by the largest pass@1 + pass@10 (it chose 0.3 on 9B and 0.2 on 27B). Nothing
about L may change after this file is committed.

Fresh prompts v4: `data/evals_v4/`, same five tasks, same construction and checks as v3 (scripts/v3_items_src,
scripts/v3_check_items.py), no item shared with the original benchmark, v2 or v3, written blind to all lens outputs,
frozen by md5 before any scoring; all are test prompts. Scorer `auto/evaluate_v4.py --split test --orchestrator`
(a copy of evaluate_v3.py with the v4 paths), run by the orchestrator only, with --lite; J++ (`jpp.py`) is scored
first as the sanity run.

PRIMARY (one test): L - J++, pass@10 with the readout filter, macro over tasks, mean over Qwen3.5-9B and Qwen3.6-27B,
on v4. Paired bootstrap, 5000 draws, prompts resampled within model x task. Holds if the 95% interval excludes zero.
SECONDARY (reported whatever they show): per model L - J++ (pass@10, pass@1, unfiltered pass@10); C - J++ and G - J++
(`ask3_A.py`, `fm27_F5s.py`); L - C; shuffled control `ask3_C_shuf.py`; v2 + v3 + v4 pooled.
Reading rule, fixed now. If the primary holds, L is reported as improving J++ on both models on three fresh sets. If
it holds only through 9B (27B interval includes zero), it is reported as a 9B gain with 27B unconfirmed. If the primary
fails, the v2/v3 secondary result is reported as not replicated.
My predictions: both models +1.5 to +3.5; 9B +2 to +4; 27B +0.5 to +3 (I expect the 27B interval may include zero on a
single set of about 500 prompts).

## Results (scored 2026-11-21 12:08Z, once, by scripts/c5_run.sh with --lite; bootstrap scripts/c5_boot.py -> scripts/results/c5_boot.txt)
v4 set: 614 prompts frozen and committed (99de122) before scoring; 482 scored on 9B, 520 on 27B. J++ sanity run first:
9B .5395 / .3175, 27B .5890 / .3507. L chose lam 0.3 on 9B and 0.2 on 27B at build time, as before.
PRIMARY: L - J++, pass@10 filtered, mean over 9B and 27B: 56.4 -> 60.7 = +4.3 [+2.7, +6.0] (bootstrap p < 0.001). The
interval excludes zero: the registered primary HOLDS.
SECONDARY
- Per model, L - J++: 9B pass@10 +6.5 [+3.9, +9.4], pass@1 +2.4 [+0.3, +4.6], unfiltered pass@10 +7.3 [+4.8, +10.0];
  27B pass@10 +2.1 [+0.4, +3.9], pass@1 +0.9 [-0.5, +2.4], unfiltered pass@10 +2.8 [+0.9, +4.7]. Prompts where L is
  better / worse than J++ (pass@10): 9B 44 / 8, 27B 27 / 15.
- Both models, L - J++: pass@1 +1.7 [+0.3, +3.0]; unfiltered pass@10 +5.0 [+3.5, +6.7].
- C - J++ (weight chosen for pass@10 only): 9B +4.1 [+2.1, +6.3], 27B +1.4 [-1.1, +3.9]; pass@1 9B +1.6, 27B -2.0
  [-4.1, +0.1]. G - J++: as C on pass@10; pass@1 27B -1.7 [-3.7, +0.3]. L - C: pass@10 both models +1.6 [+0.4, +2.8];
  pass@1 on 27B +2.9 [+1.1, +4.7].
- Shuffled control (C's weights) - J++: 9B +2.2 [-0.0, +4.6] pass@10 and -1.7 [-3.4, -0.1] pass@1; 27B -5.1 [-7.4,
  -2.9] and -8.1.
- v2 + v3 + v4 pooled, L - J++: both models pass@10 +3.4 [+2.5, +4.3], pass@1 +1.5 [+0.7, +2.3], unfiltered pass@10
  +4.2 [+3.3, +5.1]; 9B pass@10 +4.6 [+3.2, +6.1], pass@1 +2.5 [+1.1, +3.9]; 27B pass@10 +2.2 [+1.2, +3.1], pass@1 +0.4
  [-0.5, +1.4].
Reading, by the rule fixed above: the primary holds, and 27B's own interval excludes zero on v4 as well. L improves J++
on both models on three fresh prompt sets. On 27B the gain is in the top 10; the best guess is unchanged.
Predictions: both models +1.5..+3.5 (actual +4.3, above); 9B +2..+4 (actual +6.5, above); 27B +0.5..+3 (actual +2.1,
and I expected its interval might include zero; it does not).
Caveat noted for honesty: on 9B the shuffled-target control at a small weight shows a small, non-significant pass@10
lift on v4 (+2.2) with a pass@1 loss; pooled over three sets it is +0.8 [-0.4, +2.0]. L's gain is well above it.

## Addendum A (2026-11-21 12:11Z, written and committed BEFORE L is scored on the original benchmark's test split)
Additional registered check, the frozen lens L on the ORIGINAL benchmark's test split (9B: `auto/evaluate.py --split
test`, 153 prompts; 27B: `auto/evaluate27.py --split test`, 162 prompts), against the stored J++ runs T_jpp_9b and
T_jpp_27b, scored once with --lite. Reported: L - J++ pass@10 and pass@1 per model and mean over both, paired
bootstrap. These splits are small and have been scored before for other lenses (9B twice, 27B once), so this is a
secondary check, not a new primary. Predictions: 9B +3 to +5 pass@10, 27B +1 to +3; I expect at least the 27B interval
to include zero at this size. Run names T_L_9b, T_L_27b.
