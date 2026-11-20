# Lens ideas (2026-11-20), ranked by (gain x odds) / cost

Baseline J++ 58.8. Headroom per task (x0.2 each): poetry 91, association 71, multilingual 30, multihop 9, typo 5.
Facts used below (old held-out split, 9B, from `results/`):
- J++ hits per layer (of 485 pairs): L4 64, L8 108, L12 135, L16 106, L20 171, L24 175, L28 138. J-lens: 10, 21, 32, 30, 89, 133, 105.
- J++ - 0.1*I: L28 138 -> 153; overall worse only because early layers lose.
- Readout position >= 16 (association, poetry): J 8.6%, J++ 23.2%. Both read at a boundary token (period, newline).
- Fit split: association 20, multihop 6, multilingual 14, poetry 16, typo 19. One dev prompt = 0.1-0.7 points.
- Ranking is scale-invariant. If the final norm is a bias-free RMSNorm, adding a vector to the rms-normalised output equals adding a per-token logit bias. Verify once.

Notation: `P_l = JPP[l] h_l` rms-normalised. "Odds" = beats J++ on dev by >= 3 points.

## Ideas

**1. Fit diagnostics [sanity]**
- Mechanism: lens = J++ unchanged. In `build`, on `ctx.fit`, print per task x layer: concept rank histogram (<=10, 11-100, 101-1000, >1000) for logit, J, J minus always-on, J++; and the union of hits.
- Why: decides between reranking ideas (2, 3, 4, 14) and new-information ideas (6, 7, 11, 12).
- Cost: <1 min. No overfit.
- Kill: none; it is a measurement.
- Odds: 0% (no gain intended).

**2. Neighbour-layer pooling**
- Mechanism: slot l returns `sum_k w_k P_{l+k}`, k in -2..2 (JPP exists for all layers; `resid_all` gives h). Start with uniform w.
- Why: J++ per-layer quality is jagged (L12 135, L16 106), a sign of noisy expert weights. Concepts peaking at off-grid layers are never read now.
- Cost: 0 min. No fitted parameters.
- Kill: dev <= J++ + 1 for windows 3 and 5.
- Odds: 25%.

**3. Token-type prior**
- Mechanism: ridge-regress a per-token score onto the unembedding rows to get a direction `f`; return `P_l + b*f`. Two scores: log unigram frequency (from WikiText model outputs) with b < 0; "whole word with leading space, alphabetic, len >= 3" with b > 0. One b, chosen on fit.
- Why: the punctuation filter alone gave +2.9 (55.9 -> 58.8). Stop-words and word fragments likely still fill top-10 slots at periods and newlines.
- Cost: 1-2 min. One or two parameters; low overfit.
- Kill: best b on fit gives < +2 on fit, or dev <= J++ + 1.
- Odds: 30%.

**4. Same-token baseline**
- Mechanism: identify the readout token from `resid_all[0]` (nearest input embedding). In `build`, store WikiText mean of h_l per frequent token (period, newline, comma, "is", ...). Return `JPP_l (h - a * mean_token_l)`; rare tokens: a = 0.
- Why: association is always read at a period, poetry at a newline; the readout there is dominated by "start a new sentence". Global centring hurt the J-lens (-1.3 to -3.9), but that test removed one mean for all tokens, so it is not decisive here.
- Cost: 2 min (one WikiText pass). One parameter a.
- Kill: association and poetry both unchanged on fit for a in {0.5, 1}.
- Odds: 20%.

**5. Layer-increment lens**
- Mechanism: `P_l - a * P_{l-k}`, k in {2, 4}: the predicted final-layer contribution of the blocks between l-k and l.
- Why: token identity and always-on features persist across layers at the same token and cancel; a concept computed in those blocks stays. Different from J++ - a*I (that subtracts h, not an earlier layer's prediction).
- Cost: 0 min. One parameter.
- Kill: worse than J++ for every a > 0 on fit.
- Odds: 15%.

**6. Decoding-prompt patchscope**
- Mechanism: fixed neutral prompt with a placeholder (e.g. few-shot `x -> x`, or `[X] In one word:`). In `transport`, write h into the placeholder at layer l (forward hook on `ctx.model.layers[l]`), run the rest of the model, take the final residual at the last token = `S_l`. Return `P_l + b * S_l`. Choose the prompt among ~5 on fit.
- Why: nonlinear, activation-specific reading of "what would the model say later"; the only route to information the averaged linear map cannot carry. Association (29%) is where a summary at the period could be decoded.
- Cost: build 0; transport ~N x 7 short forwards (seconds). Prompt choice = mild overfit.
- Kill: `S_l` alone < 15% on fit association, or sum <= J++.
- Odds: 25%. Caveat: the model may compute the concept anew; report as a different kind of lens.

**7. Self-generated riddles as a larger fit set**
- Mechanism: in `build`, have the model write "describe X without naming it" for ~500 single-token nouns; forward the descriptions; collect (h_l at the last token, X). Use the pairs to fit the few parameters of ideas 2-5, 9, 15, or a rank <= 16 additive correction to JPP. Real fit split only for validation.
- Why: 75 labels cap every supervised idea; association has 20. The structure (concept evoked, never written) matches the benchmark.
- Cost: 4-6 min (generation is the bulk). Overfit to riddle style, not to the 75.
- Kill: parameters chosen on riddles do no better on the fit split than defaults.
- Odds: 25%. Caveat: a low-rank correction trained on labels is a probe; needs a shuffled-label control.

**8. Own expert bank, J++-style signed weights**
- Mechanism: experts per layer = {P_l, P_{l-4}, P_{l+4}, J_l h, J minus always-on, h}. Weights shared over layer groups (early / mid / late), ridge toward (1, 0, ...), fitted on fit with the J++ objective.
- Why: J++ - 0.1*I gains 15 pairs at L28 yet loses early; per-group weights capture that. J++'s own weights are not available to adjust.
- Cost: 1 min. ~18 parameters on 75 prompts: real overfit risk; use leave-one-out.
- Kill: leave-one-out fit gain < 2 points.
- Odds: 20%.

**9. Future-bag ridge lens**
- Mechanism: on WikiText, target for h_{l,s} = mean unembedding row of the content tokens in the next 32 positions (IDF-weighted). Closed-form ridge map `A_l`. Return `P_l + b * A_l h`.
- Why: a topic-word predictor; association concepts are topic words of the vignette. The tuned lens failed with a next-token target; this target is different.
- Cost: 3-4 min. One parameter b.
- Kill: `A_l h` alone < 10% on fit association.
- Odds: 15%.

**10. Patchscope-Jacobian (stays linear)**
- Mechanism: Jacobian of idea 6's last-token final residual w.r.t. the patched activation, averaged over ~8 WikiText activations per layer (4096 cotangents batched on a 12-token prompt). Use as an extra expert next to JPP.
- Why: an expert defined by the reading context, not the source cluster. Keeps one linear map per layer.
- Cost: 4-8 min; tight. One weight per layer group.
- Kill: run only if idea 6 shows signal.
- Odds: 12%.

**11. Far-target Jacobian expert**
- Mechanism: 32-token WikiText windows; cotangents only at the last 8 positions; read gradients at sources >= 4 tokens earlier. `J_far`. Lens = `P_l + b * J_far h`.
- Why: hidden concepts are used later, not said next. The J-lens sum over targets is dominated by near positions. The near/far split is untested (J - a*I only removes the identity).
- Cost: 8-10 min; ~40 short prompts, noisy.
- Kill: `J_far` alone < J-lens on fit.
- Odds: 12%.

**12. Couplet planning expert (poetry only)**
- Mechanism: model writes ~60 rhyming couplets. Jacobian from h_l at the newline to the final residual just before line 2's last word. Apply only when the readout token is a newline (idea 4's token id).
- Why: poetry is 9% with 91 points of headroom; WikiText has almost no verse, so no existing Jacobian covers this path.
- Cost: 6-8 min. No fit labels needed.
- Kill: poetry on fit stays <= 2 of 16.
- Odds: 12%. Task-specific; label it so.

**13. Interventional lens**
- Mechanism: write h into 8 WikiText contexts at layer l, position s; return the mean change of final residuals at positions > s, versus patching the same-token mean.
- Why: the nonlinear version of J_l h. The MLP result (-6 to -8) tested Jacobian matching, not the model's real forward pass.
- Cost: build 1 min; transport ~1 min in total.
- Kill: <= J-lens on fit.
- Odds: 10%.

**14. Per-token z-score (programmable readout)**
- Mechanism: compute logits in `transport`; z = (logit - mean_t) / (std_t + c) with WikiText statistics per layer; map the top-K scores back to a vector with the pseudo-inverse of those K unembedding rows.
- Why: finds tokens unusually high for this context; the std part cannot be done by a linear map.
- Cost: 2 min. Parameters c, K.
- Kill: on fit, z-rank of the concept is no better than raw rank.
- Odds: 15%. Caveat: bends "return a vector"; flag in results.

**15. Unembedding top-direction removal / spectrum [sanity]**
- Mechanism: (a) project out the top k = 1..8 principal directions of the unembedding from `P_l`; (b) JPP = U S V^T -> U S^g V^T, g in {0.8, 1.2}.
- Why: top unembedding directions encode frequency. Truncating J++'s tail never helped; the top and the spectrum shape are untested.
- Cost: <2 min. One parameter.
- Kill: flat on fit.
- Odds: 8%.

**16. J / J++ extrapolation [sanity]**
- Mechanism: `P_l + g * (P_l - J_l h)`, g in {-0.5, -0.25, 0.25, 0.5, 1}.
- Why: one sweep covers mixing (g < 0) and pushing further in the direction J++ moved (g > 0).
- Cost: 0 min.
- Kill: flat or worse on fit.
- Odds: 8%.

**17. Slot headroom [sanity, not a lens claim]**
- Mechanism: slot 4 returns the J++ readout of layer 18, slot 8 of layer 22 (`resid_all`).
- Why: L4 has the fewest hits (64). Measures how much the 7-layer grid costs; also the dev noise scale.
- Cost: 0 min.
- Kill: none. Never report as a better lens.
- Odds: 20% by number, 0% as a result.

**18. Label ridge correction**
- Mechanism: `A = JPP + dA`, dA = closed-form ridge fitting `A h_i` to the concept's unembedding row on the 75 pairs; lambda by leave-one-out.
- Why: tests directly whether 75 labels carry any transferable correction.
- Cost: <1 min. Highest overfit risk on the list.
- Kill: leave-one-out gain <= 0.
- Odds: 5%.

## Run first
1. **Fit diagnostics (1)**: one minute; tells which half of this list is worth running.
2. **Neighbour-layer pooling (2)**: free, no parameters, transfers to 2B unchanged.
3. **Token-type prior (3)**: cheap; same kind of gain as the punctuation filter.
4. **Same-token baseline (4)**: aims at the two floor tasks; also builds the token-id helper ideas 12 and 13 need.
5. **Decoding-prompt patchscope (6)**: the highest-upside idea; a quick fit-only check of `S_l` kills or keeps it and idea 10.
6. **Self-generated riddles (7)**: removes the 75-label limit for everything fitted afterwards.
Then 5, 8, 16 as free sweeps; 9, 11, 12 only if 1 says the concept is far down the ranking.

## Unsure; a quick measurement settles it
- Where do association and poetry misses sit in J++'s ranking: 11-100 (reranking can work) or > 1000 (need new information)? Idea 1.
- Is the planned rhyme word present at the newline at all on 9B? If idea 6 with a rhyme-friendly prompt and idea 12 both give ~0 on fit poetry, drop poetry and spend effort on association.
- Interface checks: is the final norm bias-free (logit bias = vector add)? Does `resid_all[0]` identify the readout token? Does the scorer allow model forwards with hooks inside `transport`, and count their time?

## Added by the user (2026-11-20) — PRIORITY, run right after the failure-mode report
**U1. Read what will NOT simply continue to exist (continuation-subtracted / indirect readout).**
- Observation to check first (`auto/failure_modes.md`): on misses, how much of the top 10 is the model's own
  continuation at that token (its next-token prediction / final answer)?
- Idea: the part of `M_l h_l` that persists to the output at this token is the continuation, not the hidden
  intermediate. Remove it and rank what is left.
- Variants (all fit the interface; `resid_all[30]` is the near-final state at the same token):
  - **U1a subtract the continuation vector**: `c = unit(J[30] @ resid_all[30])` (or `JPP[30]`),
    `v = unit(M_l h_l) - beta * c`, beta in {0.25, 0.5, 1.0} (unit = RMS-normalised). Logits become
    roughly lens logits minus beta x continuation logits.
  - **U1b project out the continuation subspace**: take the model's top-k next tokens at this token (unembed `c`),
    orthonormalise their (norm-scaled) unembedding rows, remove that span from `M_l h_l`; k in {5, 20, 50}.
  - **U1c consumed-content readout**: `v = unit(M_l h_l) - beta * unit(M_l' h_l')` with a later layer l' (l+4 or l+8):
    what is present at l and gone later (an intermediate is used up; "spider" at 24, "eight" at 28).
  - **U1d per-item persistence**: subtract only the component of `M_l h_l` along `c` (a projection, coefficient
    learned per item, no free beta).
- Already known and NOT the same thing: removing the skip connection (`J - I`) is flat-to-worse, and `J++ - a*I`
  is worse. U1 subtracts the item's actual continuation, not the identity map.
- Try on both bases: M = J++ (goal) and M = J (to see whether the effect exists at all).
- Kill criterion: no beta / k improves dev pass@10 over the same base lens, or the gain disappears unfiltered.
- Risk: on tasks where the concept IS close to the continuation (check per task in the failure-mode report)
  this removes the target.
