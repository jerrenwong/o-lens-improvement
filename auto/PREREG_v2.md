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
