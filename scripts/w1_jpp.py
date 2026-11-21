"""w1: the unmodified jpp_lens CLI (fit-weights / evaluate) on generated hidden-word items.

    /root/jpp_lens/.venv/bin/python scripts/w1_jpp.py [--w1-split NAME] [--w1-fitter original|chunked] \
        <jpp_cli stage> <args...>

Patches (in-process only; /root/jpp_lens is not edited):
  * When --eval-data-dir is data/evals_gen:
      - READOUT_EVAL_SPECS["genword"] = ReadoutEvalSpec("final_token") (readout at the last token, no
        correctness target: the same spec as association/typo).
      - jpp_cli.load_recipe_items -> the genword items, NO correctness filter (there is no target to grade;
        ungradeable items are what the authors' filter keeps anyway).
      - jpp_cli.load_readout_eval_items(<gen dir>) (used only for the Readout-Filtering exclusion list) ->
        the six original evals' items from jpp_lens's data dir + the genword items, so the exclusion list is the
        benchmark's (non-semantic ids, sparing all candidates).
      - jpp_cli.select_items -> the items named in data/evals_gen/splits.json[--w1-split].
  * --w1-fitter chunked: ExpertWeightingFitter.fit replaced by a memory-light version of the same search
    (same starts, objective, Adam steps, lr, temperature, exact_every, selection rule). The [P,E,V] expert logits
    are kept per 256-item chunk (cached on the GPU when they fit in W1_CACHE_GB, else recomputed every step with
    the authors' expert_logits); the surrogate's gradient is accumulated chunk by chunk. Margins (hence the exact
    macro pass) are bitwise those of the original; the surrogate differs only in float summation order.
    Needed because the original holds P*8*V fp32 logits (4000 items: 32 GB) plus several [P,V] temporaries.
Runs from /root/jpp_lens (jpp_cli reads its eval data by relative path): pass absolute paths.
"""
import json
import os
import sys

JPP = "/root/jpp_lens"
R = "/workspace/o-lens-improvement"
GEN = f"{R}/data/evals_gen"
argv = sys.argv[1:]


def pop(flag, default=None):
    if flag in argv:
        i = argv.index(flag)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


split = pop("--w1-split")
fitter_kind = pop("--w1-fitter", "original")
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
import torch as t  # noqa: E402

import jpp_cli  # noqa: E402
from lens_evals.readout_evals import readout_eval_items as rei  # noqa: E402
from workspace_lens.fitting.condense_experts import fitter as fm  # noqa: E402
from workspace_lens.fitting.condense_experts import rank_objective as ro  # noqa: E402

gen_mode = "--eval-data-dir" in argv and os.path.realpath(argv[argv.index("--eval-data-dir") + 1]) == os.path.realpath(GEN)
if gen_mode:
    rei.READOUT_EVAL_SPECS["genword"] = rei.ReadoutEvalSpec("final_token")
    _orig_load = rei.load_readout_eval_items

    def load_gen(data_dir=rei.READOUT_EVALS_DATA_DIR, slugs=None):
        if os.path.realpath(data_dir) != os.path.realpath(GEN):
            return _orig_load(data_dir, slugs)
        if slugs is None:  # exclusion list: the six original evals + genword
            orig = [s for s in rei.READOUT_EVAL_SPECS if s != "genword"]
            return _orig_load(rei.READOUT_EVALS_DATA_DIR, orig) + _orig_load(GEN, ["genword"])
        return _orig_load(GEN, slugs)

    jpp_cli.load_readout_eval_items = load_gen
    jpp_cli.load_recipe_items = lambda *, data_dir, correctness_csv, hf_model_name: _orig_load(GEN, ["genword"])
    names = set(json.load(open(f"{GEN}/splits.json"))[split])

    def select(items, items_spec, *, fit_fraction):
        out = [it for it in items if it.name in names]
        print(f"[w1] split {split}: {len(out)} of {len(items)} scoreable items (expected {len(names)})", flush=True)
        assert len(out) == len(names)
        return out

    jpp_cli.select_items = select
    print(f"[w1] gen mode: genword items, no correctness filter, split {split}", flush=True)

CH = 256  # items per chunk (multiple of expert_logits' batch_size 32, so its batches align with the original's)


def chunked_fit(self, layers, *, item_names=None, condense_config=fm.DEFAULT_CONDENSE_CONFIG):
    cfg = condense_config
    pairs, idx_P = self.fitting_pairs(item_names)
    P, G = int(idx_P.numel()), len(pairs.group_eval_slugs)
    counts_G = t.zeros(G, device=self.device).index_add_(0, pairs.eval_index_Int_Q, t.ones(pairs.num_pairs, device=self.device))
    chunks = []  # (item slice, sub PairTable, 1/(count*G) per pair)
    for a in range(0, P, CH):
        b = min(P, a + CH)
        m = (pairs.item_index_Int_Q >= a) & (pairs.item_index_Int_Q < b)
        sub = ro.PairTable(pairs.item_index_Int_Q[m] - a, pairs.candidate_ids_Int_QC[m], pairs.eval_index_Int_Q[m],
                           pairs.group_eval_slugs)
        chunks.append((a, b, sub, 1.0 / (counts_G[sub.eval_index_Int_Q] * G)))
    # pairs are ordered by item, so concatenating chunk margins gives the full table's order
    cache_gb = float(os.environ.get("W1_CACHE_GB", "12"))
    learned = {}
    for layer in layers:
        T_PEF = self._expert_transports_PEF(layer, idx_P)
        V = self.vocab_size
        cache = P * T_PEF.shape[1] * V * 4 / 1e9 <= cache_gb
        stored = {}

        def logits(ci):
            if ci in stored:
                return stored[ci]
            a, b = chunks[ci][:2]
            L = ro.expert_logits(self.model.unembed, T_PEF[a:b])
            if cache:
                stored[ci] = L
            return L

        scores = []
        for e in range(T_PEF.shape[1]):
            ms = [ro.hit_margins(logits(ci)[:, e], c[2], excluded_mask_Bool_V=self.excluded_mask_Bool_V, k=cfg.pass_k)[0]
                  for ci, c in enumerate(chunks)]
            scores.append(ro.exact_macro_pass(t.cat(ms), pairs))
        scores_E = t.tensor(scores)
        starts = fm.build_search_starting_points(scores_E, top_n=cfg.top_n)
        best = None
        for init_name, w0 in starts.items():
            w = (w0.float() / w0.float().norm()).to(self.device).clone().requires_grad_(True)
            opt = t.optim.Adam([w], lr=cfg.learning_rate)
            for step in range(cfg.steps + 1):
                last = step == cfg.steps
                evaluate_exact = step % cfg.exact_every == 0 or last
                opt.zero_grad()
                ms, surr = [], 0.0
                for ci, (a, b, sub, inv) in enumerate(chunks):
                    unit = w / w.norm()
                    m, s = ro.hit_margins(ro.combined_logits(logits(ci), unit), sub,
                                          excluded_mask_Bool_V=self.excluded_mask_Bool_V, k=cfg.pass_k)
                    contrib = (t.sigmoid(m / (cfg.temperature * s)) * inv).sum()
                    if not last:
                        contrib.backward()
                    surr += float(contrib.detach())
                    ms.append(m.detach())
                if evaluate_exact:
                    exact = ro.exact_macro_pass(t.cat(ms), pairs)
                    cand = (exact, surr, fm.LearnedWeights((w / w.norm()).detach().cpu().clone(), exact, init_name, step))
                    if best is None or cand[:2] > best[:2]:
                        best = cand
                if last:
                    break
                opt.step()
        learned[layer] = best[2]
        del stored, T_PEF
        fm.logger.info("layer %d: single-expert pass@%d %s -> learned %.3f (%s, step %d) [w1 chunked, cache=%s]",
                       layer, cfg.pass_k, [round(x, 3) for x in scores_E.tolist()], best[2].train_macro_pass,
                       best[2].init_name, best[2].step, cache)
    return fm.ExpertWeighting(learned_weights_L_dict=learned, condense_config=cfg)


if fitter_kind == "chunked":
    fm.ExpertWeightingFitter.fit = chunked_fit
    print("[w1] fitter: chunked", flush=True)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(jpp_cli.main(argv))
