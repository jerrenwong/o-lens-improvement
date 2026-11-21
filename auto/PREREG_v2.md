# Pre-registration: fresh-prompt test of J++ + distilled map (written 2026-11-20, before any lens is scored on v2)

Fresh prompts: `data/evals_v2/` (written by an agent that is blind to lens outputs; frozen by md5 in
`data/evals_v2/MANIFEST.json` before scoring). Scorer: `auto/evaluate_v2.py --split test --orchestrator`.
All v2 prompts are test prompts. Lenses use the ORIGINAL fit split for anything they fit.

Lenses
- J++: released lens, `auto/candidates/jpp.py`.
- A (frozen recipe, PRIMARY): `auto/candidates/e2_A.py` (9B) and `auto/candidates/m27_A.py` (27B); maps `runs/e1/E_9b.pt`,
  `runs/e1/E_27b.pt`; mixing weight chosen on the original fit split inside `build` (0.5 on both models).
- Controls for A: shuffled-target map (`e2_C_shuf.py`, `m27_C_shuf.py`).
- B (refined map, SECONDARY): `auto/candidates/e3_B.py`, fixed before v2 scoring; its design is in the e3 brief
  (content-word targets, prior removed, elicitation format chosen on ordinary text). Control `e3_B_shuf.py`.

Primary analysis (one test)
- Metric: pass@10 with the readout filter, macro over the five tasks, mean over the two models Qwen3.5-9B and
  Qwen3.6-27B. Comparison: A minus J++. Paired bootstrap, 5000 draws, prompts resampled within model x task.
- Claim "A improves J++" holds if the 95% interval excludes zero.

Secondary (reported whatever they show)
- Per model: A - J++ on pass@10, pass@1, unfiltered pass@10; A - shuffled control.
- B - J++, B - A, B - its shuffled control (same metrics).
- Per-task differences; prompts gained / lost.

Already known before this test (old benchmark, all held-out prompts, pass@10): A - J++ = +3.2 [+0.9,+5.7] on 9B,
+2.8 [+0.2,+5.6] on 27B, +4.5 [-1.4,+10.0] on 2B; pooled 9B+27B +3.0 [+1.3,+4.8]; top-1 not improved.
The old test splits were scored twice on 9B/2B (gated lens, then A) and once on 27B.

## Amendment 1 (2026-11-20 17:45Z, before any lens other than the J++ sanity run was scored on v2)
- B (refined map) was fitted on 9B only. On 9B dev it equals J++ (.5949 vs .5952) and is below A (-3.2); its
  pre-specified format rule selected a chat format whose first answer tokens are letter fragments. The 27B fit of B
  is cancelled to free the GPU. B remains a secondary comparison on v2 for 9B only (`e3_B.py`, `e3_B_shuf.py`).
- The primary analysis is unchanged.

## Results (scored once, 2026-11-20 17:54-18:18Z; runs auto/runs/V2_*; 469 prompts for 9B, 501 for 27B)
PRIMARY: A - J++, pass@10 filtered, mean over 9B and 27B: 65.8 -> 68.0 = +2.2 [+0.6, +3.7] (bootstrap p ~ 0.008;
75 prompts better, 44 worse). The interval excludes zero: the pre-registered claim holds.
SECONDARY
- 9B: +3.3 [+0.8,+5.9] pass@10; +1.6 [-1.0,+4.4] pass@1; +3.3 [+0.8,+5.9] unfiltered.
- 27B: +1.1 [-0.7,+2.9] pass@10; -1.8 [-4.1,+0.5] pass@1; -0.2 [-2.1,+1.7] unfiltered.
- Pooled pass@1: -0.1 [-1.8,+1.7]; pooled unfiltered pass@10: +1.5 [-0.0,+3.1].
- A - shuffled control: +6.1 [+4.7,+7.5]; shuffled control - J++: -3.9 [-5.7,-2.3].
- B (refined map, 9B only): -2.7 [-4.5,-1.0] vs J++; -5.9 [-8.4,-3.6] vs A.
- Per task (J++ -> A): 9B association 61 -> 71, multihop 76 -> 82, multilingual 74 -> 78, poetry 12 -> 9, typo 97 -> 97;
  27B association 76 -> 78, multihop 70 -> 72, multilingual 76 -> 77, poetry 16 -> 16, typo 99 -> 98.
Note: v2 association prompts are easier than the originals (J++ 61% / 76% vs 29% / 39%).

## Amendment 2 (2026-11-21 02:30Z, written BEFORE any label-free-weight lens is scored on v2)
Label-free expert weights (w1) have been scored on v3 only. v2 is therefore an untouched test for them.
Candidate S ("stack"): natural-format experts with weights fitted on generated hidden-word sentences
(`runs/w1/jpp_Cfull_Wtrain.pt`, chosen because it scored highest on v3) + lam * ask-map E_9b, lam chosen on the
original benchmark's fit split by the e2_A rule (`auto/candidates/s1_stack.py`). No other tuning.
- PRIMARY: S - released J++ on the v2 9B test prompts, filtered pass@10, paired bootstrap (r2_boot.py).
- Secondary: the label-free lens alone (CW) - released; S - A (released + ask-map, V2_e2_A_9b); the same two with the
  raw-recipe experts (`runs/w1/jpp_Afull_Wtrain.pt`: W, SW); pass@1 and unfiltered for all.
- Post-hoc (flagged): S and SW on v3, where the label-free lenses were already scored.
- Known limit: v2's scene prompts are easier than the original's (J++ 61% vs 29%).

## Results, amendment 2 (scored 2026-11-21 02:05-02:15Z; 9B, v2 test prompts)
Released J++ .6427 (V2_jpp_9b). lam chosen on fit = 1.0 for both stacks.
- PRIMARY S - released: +2.8 [-0.1,+5.9] filtered pass@10 (S .6710); pass@1 -1.9 [-4.8,+1.0]; unfiltered +3.0 [+0.2,+6.0]. NOT significant.
- CW alone - released: +1.8 [-1.1,+4.5]; pass@1 -2.5; unfiltered +3.2 [+0.3,+6.1].
- S - A (released + ask-map): -0.4 [-2.4,+1.5]; pass@1 -3.5 [-6.0,-1.0]. A - released on v2: +3.3 [+0.8,+5.9].
- Raw-recipe experts: W - released -0.8 [-3.7,+1.9], pass@1 -4.9 [-7.9,-1.9]; SW - released +0.5 [-2.6,+3.6].
- Post-hoc v3: S - released +4.2 [+1.2,+7.2]; S - A +0.4 [-1.5,+2.4]; SW - released +2.5 [-0.6,+5.8].
Reading: the two pieces do not add. Label-free weights + ask-map = ask-map on the released lens at pass@10 and is
worse at pass@1 on v2. Label-free weights lose top-1 on v2 (-2.5 to -4.9). Best lens on fresh prompts stays
A = released J++ + ask-map.

## Amendment 3 (2026-11-21 08:38Z, written and committed BEFORE any ask3 lens is scored on v2 or v3)
Same registration as PREREG_v3.md Amendment 5 (clean chat-format ask-map C = `ask3_A.py`, control
`ask3_C_shuf.py`, both models, --lite scoring with the J++ reproduction guard). On v2 the test C - J++ (pass@10
filtered, macro over tasks, mean over 9B and 27B) is the registered replication; secondaries as listed there.

## Results, amendment 3 (scored 2026-11-21 09:35Z; see PREREG_v3.md "Results, amendment 5" for the full table)
C - J++ on v2, pass@10 filtered, mean over 9B and 27B: +2.1 [+0.6, +3.6]. 9B +3.1 [+1.1, +5.2] (pass@1 +3.1 [+1.1,
+5.1]); 27B +1.1 [-1.0, +3.3] (pass@1 -1.8 [-4.6, +0.9]). C - A: 9B -0.2 [-2.0, +1.6]; 27B +0.1. Guard passed.

## Amendment 4 (2026-11-21 10:41Z, written and committed BEFORE any fm27 lens is scored on v2 or v3)
Same registration as PREREG_v3.md Amendment 6 (lens G = `fm27_F5s.py`, lens L = `fm27_F4.py`; primary G - C pass@1
on 27B, mean over v2 and v3).

## Results, amendment 4 (scored 2026-11-21 11:13Z; full table in PREREG_v3.md "Results, amendment 6")
Primary (G - C pass@1 on 27B, v2 + v3): +1.2 [+0.2, +2.2]. On v2 alone: L - J++ pass@10 27B +2.8 [+1.4, +4.4], 9B +3.7
[+1.4, +6.1], both models +3.3 [+1.9, +4.7].
