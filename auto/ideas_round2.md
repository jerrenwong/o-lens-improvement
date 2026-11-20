# Lens ideas, round 2 (2026-11-20)

J++ dev 0.5952; 31 candidates discarded. Round-1 ideas 5, 7, 8, 9, 11, 13, 14, 18 are running; assumed to fail.
Odds = P(dev >= J++ + 3). "(+1.5: x)" = P(>= +1.5), for ideas that cannot reach 3 alone.

## Where the headroom is
Source: J++ baseline on dev (`runs/jpp`, counts only). Recompute on fit before selecting anything with them.
- pass@k: @10 .595, @13 .610, @15 .620, @30 .673, @100 .744, @1000 .924.
- Association @10 / @50 / @1000: .32 / .54 / .78. Poetry @10 / @100 / @1000: .03 / .34 / .88.
  Chance of rank <= 1000 is < 1%: the concept is weakly present in most items. A third of the misses rank 11-100.
- Missing macro points: poetry 19.4 (6.3 at rank 11-100), association 13.7 (4.4 at 11-50), multilingual 6.2.
- Multilingual per intermediate: language .43, relation .73, source word .63, answer .97
  -> missing points 2.8 / 1.3 / 1.8 / 0.2. Local names (español, français): 0 of 6.
- Hits per layer (unique to that layer), 246 pairs: L4 32 (1), L8 59 (2), L12 74 (10), L16 55 (1), L20 85 (5),
  L24 89 (5), L28 70 (10). Slots 4, 8, 16 are nearly redundant.
- J++ on its own weight-fit items (if `ctx.fit` is that split): association 5/20, poetry 2/16, no better than
  dev. Its 8-expert family is limited by span, not by labels.
- J++ reads association where the patchscope reads nothing: J++ is the averaged linear effect of this token on
  LATER positions; the patchscope asks the same position's later layers, in a foreign context.
- Scorer: concept rank = min over its case/space forms (`surface_forms`); competitors take one slot per form.

## Ceiling measurements (not lenses)
- **M1 model ceiling** (orchestrator, fit prompts, full context). Association: prompt + "\nIn one word, this is
  about" -> concept in the model's top 10? Poetry: sample 20 second lines -> benchmark word among the 10 most
  frequent endings? Any lens <= token ceiling <= this.
- **M2 token ceiling**: held-out score of idea 2 (ridge from the single token to the model's full-context
  answer, N = 2000, label-free). Plus idea 4 Step A (in-sample span of a bank).
- Caution for ideas 7/18: `b2e_common.CONCEPTS` is hand-written and contains words shown in `failure_modes.md`
  (poverty, relief), whose examples mix fit and dev. Draw the list mechanically before trusting a gain.

## Ideas

**1. One slot per word (case/space dedup)**
- Mechanism: `transport`: P = JPP_l h; z = W_eff P (W_eff = lm_head x final-norm gain); mask excluded ids;
  top K = 200; key = `decode(t).strip().lower()`; per key keep the top token, others get target change
  -(z_1 - z_K + 1), kept tokens 0; delta = pinv(W_eff[topK]) t; return P + delta. Re-rank once to check.
- Why: scorer fact above. Multilingual L20 example: bottom x4, below x2 fill 6 slots. The group top is a
  candidate form, so a concept's rank cannot get worse. pass@13-15 suggests +1.5 to +2.5.
- Cost: 0 build, no parameters. Kill: fit gain < 1 pair, or any fit hit lost.
- Odds: 15% (+1.5: 60%). A readout filter, not a map: give J-lens and J++ the same filter.

**2. Elicit-distilled map**
- Mechanism: `build`: 2000 WikiText sentence-final periods. Pass 1 (text only): x_l = h_l at the period.
  Pass 2 (text + "\nIn one word, this is about"): y = last-block residual at the last token. Centre; dual ridge
  A_l: x -> y, lambda on a 20% hold-out. Lens: unit(P_l) + b unit(A_l (h - mu_l)), or alone in a slot (idea 3).
- Why: the gist is linearly at the period (J++ 32%), yet the model's decoder fails on a patched token (0%).
  Target = the model's own full-context answer. No labels, no concept list.
- Cost: 2-3 min. Kill: held-out, the model's answer word in A_l h top 10 < 20% at every layer.
- Odds: 15%. A trained map; needs a shuffled-target control.

**3. Specialise the redundant slots**
- Mechanism: slots 12, 20, 24, 28 = J++. Slots 4, 8, 16 return another map of their own h_l, picked per slot
  from {A_l of idea 2, language map of idea 5, J_l minus always-on features, JPP_{l+4} applied to h_l} by
  leave-one-out union pass@10 on fit.
- Why: those slots hold 4 unique pairs of 246 (~1 point at risk). Sums interfere (J++ + J: -0.9 to -7.9);
  replacement cannot. J++ fits each layer on a per-layer objective ("a surrogate", fitter docstring).
- Cost: 0 beyond the maps. Kill: fit union gain < 2 pairs.
- Odds: 15%.

**4. Coverage-aware weight refit, with a span test**
- Mechanism: bank per layer, all maps of h_l: JPP_l h, J_l h, h, JPP_{l-4} h, JPP_{l+4} h, JPP_l h_on
  (always-on dictionary part), JPP_l Pi_64 h (top-64 WikiText PCs). Cache W_eff e (rank-exact, linear in w, as
  in `rank_objective.py`). Step A: J++ fitter in-sample. Step B: sweep layers, pairs already hit at another
  layer weighted 0; L2 pull to (1,0,..); weights shared in 3 layer groups; 5-fold CV.
- Why: J++ in-sample 5/20 says the limit is span. Differs from idea 8: same-h maps, union objective.
- Cost: 2 min, ~18 parameters. Kill: Step A < J++ fit + 5 pairs (drop all bank ideas); CV gain < 2 pairs.
- Odds: 10%.

**5. Language-name map (multilingual)**
- Mechanism: `build`: model translates 150 WikiText sentences into 12 languages. x = h_l at a random position
  >= 5; y = last-block residual after "\n(Language:". Ridge as idea 2. Used in slot 4 or 8.
- Why: language is 2.8 of the 6.2 missing points; J++ names mostly French / German.
- Cost: 3-4 min. Kill: language hits on fit multilingual (14 items) not above J++.
- Odds: 3% (+1.5: 25%). Task-informed; say so.

**6. Question-conditioned Jacobian expert**
- Mechanism: 100-150 WikiText sentences + the idea-2 suffix. Jacobian of the last-token final residual w.r.t.
  h_l at the period, natural forward pass. Cotangents = top-512 right singular vectors of W_eff over whole-word
  tokens (8 backward batches of 64 per prompt, all layers at once). Lens: P_l + b J_q h, or a slot.
- Why: J++ experts differ by source cluster, none by what is asked. Round-1 idea 10 took this Jacobian in the
  patched prompt, where the decoder fails; here the model does answer.
- Cost: 6-9 min, tight. Kill: J_q alone < 15% on fit association.
- Odds: 12%.

**7. Rollout-distilled newline map (poetry)**
- Mechanism: model writes 300 first lines, 8 sampled second lines each. x = h_l at the newline; y = mean
  centred W_eff row of the 8 final words. Ridge; add or slot.
- Why: poetry .03 at 10 but .34 at 100. WikiText has no verse, so no Jacobian saw a planning newline.
- Cost: 5-7 min. Kill: sampled endings rhyme < 50% (nothing planned), or held-out modal ending in top 10 < 15%.
- Odds: 10%.

**8. All-layer transplant, contrastive**
- Mechanism: `transport`: prompt "X\nIn one word, this is about"; overwrite X's residual at every layer 0..l
  from `resid_all`. S = last-token residual minus the same with the WikiText mean period activation. Re-rank
  J++'s top 30 by S (vector edit as idea 1).
- Why: b2d patched one layer; lower layers still held the placeholder. The fair test of the model's own decoder.
- Cost: 1 min build, 7N forwards. Kill: S alone <= 10% on fit association -> stop decoder-side ideas.
- Odds: 8%.

**9. Common-neighbour vote**
- Mechanism: after idea 1: z'_c = z_c + beta x sum over top-30 words t != c of softmax(z)_t cos(w_c, w_t),
  w = W_eff rows centred, top-64 PCs removed. Vector edit on the top 300.
- Why: near misses sit under their relatives: "down" at 19 under below / beneath / under.
- Cost: 1 min, one parameter. Kill: fit association + multilingual gain < 2 pairs.
- Odds: 10%. The lens infers from its own list: report as post-processing, not reading.

## Run first
1. **Idea 1**: free, monotone, the only near-certain gain.
2. **Idea 2** with M2: one build gives a map and the token ceiling. M1 in parallel.
3. **Idea 4 Step A**: one minute; keeps or kills every bank idea.
4. **Idea 3** with the maps from 2 and 5: uses the three redundant slots.

## View
A small win is plausible: dedup (+1.5 to +2.5), the language name (up to +2.8) and the redundant slots could
add +3 to +5 together. Most of that is readout and slot engineering, not a better map, and must be reported as
such. A clear margin (+8 or more) needs association or poetry to move. The concept is there (rank <= 1000 in
78% / 88% of items) but weak, J++ scores no better on its own fit items, and the model's own decoder reads
nothing from the token. Estimate: 30% for a confirmed +3, under 10% for +8. If M1 shows the model names the
concept in only about half the association prompts, J++ is already near the ceiling there.
