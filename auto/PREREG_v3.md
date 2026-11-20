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
