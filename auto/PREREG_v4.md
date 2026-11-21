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
