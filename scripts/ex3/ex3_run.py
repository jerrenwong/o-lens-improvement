"""ex3: where are the hidden concepts that no reader finds at the readout token? (FIT split only)

    HF_HOME=... /root/jpp_lens/.venv/bin/python scripts/ex3/ex3_run.py --tag 9b|27b|2b

FIT items = bench.index("fit") rows of the original bench (same rows as ctx.fit in auto/evaluate*.py); prompts from
the readout-eval item files, kept only for (eval, name) in the fit list; readout position and candidates via the
readout-eval helper (checked against the bench). Lenses: released J++ (ctx.JPP), lens L (fm27_F4 / m2b_L, lam
re-chosen on FIT), best ask-map alone (st1 p3v_t10). Every lens output is unit-scaled (rms 1) before the bf16 LM head.

Per item: M1 ranks/top tokens at the readout token; M2 per-position (all tokens x 7 layers) ranks of the item's
concepts, 20 null concepts of the same task and (poetry) the model's own rhyme words, under J++ and ask-map alone;
M3 asked answers (topic / unstated wordings, chat format), greedy + 8 sampled continuations, next-token rank;
M5 J++ rank/cosine of the concept at the asked prompts' answer position vs the readout token.
Writes runs/ex3/raw_<tag>.json.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from types import SimpleNamespace

R = "/workspace/o-lens-improvement"
JPPD = "/root/jpp_lens"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", choices=["9b", "27b", "2b"], required=True)
ap.add_argument("--n-null", type=int, default=20)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
CFG = {
    "9b": dict(model="Qwen/Qwen3.5-9B", bench="artifacts/bench/qwen3.5-9b.pt", JPP="artifacts/baselines/jpp9b_prefit.pt",
               layers=[4, 8, 12, 16, 20, 24, 28]),
    "2b": dict(model="Qwen/Qwen3.5-2B", bench="artifacts/bench/qwen3.5-2b.pt", JPP="artifacts/baselines/jpp_prefit.pt",
               layers=[3, 6, 9, 12, 15, 18, 21]),
    "27b": dict(model="Qwen/Qwen3.6-27B", bench="artifacts/bench/qwen3.6-27b.pt",
                JPP="artifacts/baselines/jpp27b_released.pt", layers=[8, 16, 24, 32, 40, 48, 56]),
}[args.tag]
os.makedirs(f"{R}/runs/ex3", exist_ok=True)
os.chdir(JPPD)
sys.path.insert(0, f"{R}/src")
sys.path.append(f"{R}/auto/candidates")
import torch as t  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402
from lens_evals.eval_utils import candidate_token_ids  # noqa: E402
from lens_evals.readout_evals import readout_evals as re_mod  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import READOUT_EVALS_DATA_DIR, load_readout_eval_items  # noqa: E402
from olens.readout import Bench  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import load_lens_file  # noqa: E402

t.set_grad_enabled(False)
dev = "cuda"
LAYERS = CFG["layers"]
T0 = time.time()


def log(*a):
    print(f"[ex3 {args.tag} {time.time() - T0:6.0f}s]", *a, flush=True)


model = get_hf_model(CFG["model"], attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
hf, tok = model._hf_model, model.tokenizer
log("model loaded")

# ---------------- FIT ctx (exactly the rows auto/evaluate*.py put in ctx.fit) ----------------
bench = Bench.load(f"{R}/{CFG['bench']}")
assert set(LAYERS) <= set(bench.layers)
fit_idx = bench.index("fit")
fi = t.as_tensor(fit_idx, dtype=t.long)
fit = SimpleNamespace(resid={l: v[fi].to(dev).float() for l, v in bench.resid.items()},
                      candidates=[bench.candidates[i] for i in fit_idx],
                      eval_slugs=[bench.eval_slugs[i] for i in fit_idx],
                      item_names=[bench.item_names[i] for i in fit_idx],
                      excluded_ids=bench.excluded_ids.clone())
rpos_bench = [bench.readout_positions[i] for i in fit_idx] if bench.readout_positions else None
del bench
JPP = {int(l): v.to(dev).float()
       for l, v in load_lens_file(f"{R}/{CFG['JPP']}", hf_model_name=CFG["model"]).jacobians_L_dict_FN.items()}
ctx = SimpleNamespace(model=model, JPP=JPP, fit=fit, readout_layers=LAYERS, device=dev)

import jpp as jpp_mod  # noqa: E402
from fm27_lib import head_gamma, unit  # noqa: E402
from st1_lib import St1  # noqa: E402

if args.tag == "2b":
    import m2b_L as Lmod
else:
    import fm27_F4 as Lmod
LENS = {"jpp": jpp_mod.build(ctx), "L": Lmod.build(ctx), "alone": St1(ctx, "p3v_t10", alone=True)}
log("lenses built; L lam =", LENS["L"].lam)
V = model._lm_head.weight.shape[0]
EXCL = t.zeros(V, dtype=t.bool, device=model._lm_head.weight.device)
EXCL[fit.excluded_ids.to(EXCL.device)] = True
GAMMA = head_gamma(model)
WU = model._lm_head.weight


def tv(name, h, l):
    return unit(LENS[name].transport(h, l).float())


def lg_of(vecs, filt=True):
    out = []
    for i in range(0, vecs.shape[0], 32):
        lg = model.unembed(vecs[i:i + 32]).float()
        if filt:
            lg = lg.masked_fill(EXCL, float("-inf"))
        out.append(lg)
    return t.cat(out)


def rank_mat(lg, cand_lists):
    """lg [P, V]; cand_lists: list of id lists -> ranks [P, C] (1 + #logits > max over the candidates)."""
    out = []
    for ids in cand_lists:
        m = lg[:, t.tensor(ids, device=lg.device)].max(-1).values
        out.append(1 + (lg > m[:, None]).sum(-1))
    return t.stack(out, 1)


def dec(i):
    return tok.decode([int(i)])


# ---------------- items (fit only) ----------------
keys = list(zip(fit.eval_slugs, fit.item_names))
kset = set(keys)
by_key = {}
for it in load_readout_eval_items(READOUT_EVALS_DATA_DIR):
    if (it.eval_slug, it.name) in kset:
        by_key[(it.eval_slug, it.name)] = it
assert set(by_key) == kset, f"missing fit items: {kset - set(by_key)}"
items = [by_key[k] for k in keys]
del by_key
if args.limit:
    items = items[:args.limit]

# ---------------- ask wordings (copied from scripts/ask3/ask3_collect4.py W[0], scripts/ask5/ask5_collect.py W[0]) --
ASK = {
    "topic": ("", "Passage: ", "One-word topic:",
              [("The tide pulled the little boat away from the dock while the fisherman untangled his nets.", "fishing"),
               ("The senate voted late on Tuesday to approve the new budget, despite", "politics"),
               ("Elle a sorti la farine, les œufs et le beurre pour préparer", "baking"),
               ("Lily saw the old dog shivering in the rain, so she opened the door and", "kindness")]),
    "unstated": ("", "Passage: ", "Unstated word:",
                 [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
                  ("The astronomer who turned his telescope on the moons of Jupiter was put on trial in Rome because", "Galileo"),
                  ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
                  ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")]),
}


def build_ask(k, text):  # FAM == "chat" branch of ask3_collect4.build
    head, q, lab, ex = ASK[k]
    shots = head + "".join(f"{q}{tx}\n{lab} {w}\n\n" for tx, w in ex)
    msgs = [{"role": "user", "content": shots + f"{q}{text}\n{lab}"}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    return s + lab


def fwd(ids):
    """ids [1, S] -> (acts {L: [S, d] fp32}, logits [S, V] fp32)."""
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        out = hf(input_ids=ids)
        acts = {L: rec.activations[L][0].float() for L in LAYERS}
    return acts, out.logits[0].float()


def gen(ids, n, sample, max_new=30, seed=0):
    t.manual_seed(seed)
    kw = dict(do_sample=True, temperature=0.8, top_k=0, top_p=1.0) if sample else dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    g = hf.generate(input_ids=ids.repeat(n, 1), attention_mask=t.ones_like(ids).repeat(n, 1), max_new_tokens=max_new,
                    num_return_sequences=1, pad_token_id=tok.pad_token_id or tok.eos_token_id, **kw)
    return [row[ids.shape[1]:].tolist() for row in g]


def appears(gen_ids, word, cand_ids):
    """first generated-token index where the concept appears (candidate id, or text match of the word as a word
    start, case-insensitive: covers plurals / inflections), else None."""
    cs = set(cand_ids)
    hit_tok = next((k for k, i in enumerate(gen_ids) if i in cs), None)
    pat = re.compile(r"(?<![^\W\d_])" + re.escape(word.lower()), re.I)
    txt, hit_txt = "", None
    for k, i in enumerate(gen_ids):
        txt += dec(i)
        if pat.search(txt):
            hit_txt = k
            break
    c = [x for x in (hit_tok, hit_txt) if x is not None]
    return min(c) if c else None


WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def line_end_word(gen_ids):
    """the word that ends the line the continuation completes: text up to the first newline (or the first
    sentence-final punctuation if no newline), last word."""
    txt = tok.decode(gen_ids)
    line = txt.split("\n")[0] if "\n" in txt else re.split(r"(?<=[.!?;])", txt)[0]
    ws = WORD.findall(line)
    return (ws[-1].lower() if ws else None), line


# null concept pools: concepts of other fit items of the same task
task_concepts = {}
for k, (e, n) in enumerate(keys):
    for c, ids in fit.candidates[k].items():
        task_concepts.setdefault(e, {})[c] = ids

out_items = []
cos_check = []
for k, it in enumerate(items):
    ti = time.time()
    ro = re_mod._item_readout_and_candidates(model, it, 512)
    rp = int(ro.readout_position)
    cands = {c: [int(x) for x in v] for c, v in ro.candidate_ids_by_intermediate.items()}
    assert cands == fit.candidates[k], (it.name, cands, fit.candidates[k])
    if rpos_bench is not None:
        assert rp == rpos_bench[k], (it.name, rp, rpos_bench[k])
    ids = ro.input_ids_Int_1S.to(dev)
    S = ids.shape[1]
    acts, logits = fwd(ids)
    for L in LAYERS:
        cos_check.append(float(t.nn.functional.cosine_similarity(acts[L][rp], fit.resid[L][k], dim=0)))
    rec = dict(eval=it.eval_slug, name=it.name, prompt=it.prompt, target=it.target, concepts=list(cands), cand=cands,
               rp=rp, S=S, ids=ids[0].tolist(), toks=[dec(i) for i in ids[0].tolist()])

    # ---- M3 (iii): model's own next token at the readout position and at the end of the prompt
    nl = logits[[rp, S - 1]]
    rec["next"] = dict(
        rp_rank_u=rank_mat(nl[:1], list(cands.values()))[0].tolist(),
        rp_rank_f=rank_mat(nl[:1].masked_fill(EXCL, float("-inf")), list(cands.values()))[0].tolist(),
        rp_top5=[dec(i) for i in nl[0].topk(5).indices.tolist()],
        end_rank_u=rank_mat(nl[1:], list(cands.values()))[0].tolist(),
        end_top5=[dec(i) for i in nl[1].topk(5).indices.tolist()])

    # ---- M3 (ii) / M4: greedy + 8 samples of the raw prompt
    g0 = gen(ids, 1, False)[0]
    gs = gen(ids, 8, True, seed=1000 + k)
    rec["cont"] = dict(greedy=tok.decode(g0), samples=[tok.decode(s) for s in gs],
                       appear_greedy={c: appears(g0, c, v) for c, v in cands.items()},
                       appear_samples={c: [appears(s, c, v) for s in gs] for c, v in cands.items()},
                       end_greedy=line_end_word(g0)[0], end_samples=[line_end_word(s)[0] for s in gs],
                       line_greedy=line_end_word(g0)[1])
    extra = {}  # poetry: the model's own rhyme words
    if it.eval_slug == "poetry":
        # the model's OWN second line: continue from the readout token (end of line 1), not from the given line-2 start
        pre = ids[:, :rp + 1]
        o0 = gen(pre, 1, False)[0]
        os_ = gen(pre, 8, True, seed=2000 + k)
        rec["own_line"] = dict(greedy=tok.decode(o0), samples=[tok.decode(s) for s in os_],
                               end_greedy=line_end_word(o0)[0], end_samples=[line_end_word(s)[0] for s in os_],
                               line_greedy=line_end_word(o0)[1])
        words = [rec["own_line"]["end_greedy"]] + rec["own_line"]["end_samples"]
        cnt = {}
        for w in rec["own_line"]["end_samples"]:
            if w:
                cnt[w] = cnt.get(w, 0) + 1
        top_s = max(cnt, key=lambda w: (cnt[w], w == words[0])) if cnt else None
        rec["rhyme"] = dict(greedy=words[0], sample_counts=cnt, top_sample=top_s)
        for tagw, w in (("actual_greedy", words[0]), ("actual_topsample", top_s)):
            if w:
                cid = [int(x) for x in candidate_token_ids(tok, w)]
                rec["rhyme"][tagw + "_cand"] = cid
                if cid:
                    extra[f"{tagw}:{w}"] = cid

    # ---- M1: three readers at the readout token, 7 layers
    m1 = {}
    for nm in LENS:
        vs = t.stack([tv(nm, acts[L][rp:rp + 1], L)[0] for L in LAYERS])  # [7, d]
        lg = lg_of(vs)
        allc = {**cands, **extra}
        rk = rank_mat(lg, list(allc.values()))  # [7, C]
        m1[nm] = dict(rank={c: rk[:, j].tolist() for j, c in enumerate(allc)},
                      top10=[[dec(i) for i in row] for row in lg.topk(10, -1).indices.tolist()])
    rec["m1"] = m1

    # ---- M2: every position x 7 layers, J++ and ask-map alone; own concepts, nulls, actual rhyme words
    rng = random.Random(1234 + k)
    pool = [c for c in task_concepts[it.eval_slug] if c.lower() not in {x.lower() for x in cands}
            and c.lower() not in {w.split(":", 1)[1] for w in extra}]
    nulls = rng.sample(sorted(pool), min(args.n_null, len(pool)))
    allc = {**{f"own:{c}": v for c, v in cands.items()}, **extra,
            **{f"null:{c}": task_concepts[it.eval_slug][c] for c in nulls}}
    m2 = {}
    for nm in ("jpp", "alone"):
        best = None
        arg = None
        for li, L in enumerate(LAYERS):
            lg = lg_of(tv(nm, acts[L], L))  # [S, V]
            rk = rank_mat(lg, list(allc.values()))  # [S, C]
            if best is None:
                best, arg = rk.clone(), t.zeros_like(rk)
            else:
                upd = rk < best
                best = t.where(upd, rk, best)
                arg = t.where(upd, t.full_like(arg, li), arg)
            del lg
        m2[nm] = dict(best={c: best[:, j].tolist() for j, c in enumerate(allc)},
                      layer={c: [LAYERS[x] for x in arg[:, j].tolist()] for j, c in enumerate(allc)})
    rec["m2"] = m2
    nx = rank_mat(logits, list(allc.values()))  # model's own next-token rank (unfiltered) at every position
    rec["m2_next"] = {c: nx[:, j].tolist() for j, c in enumerate(allc)}
    rec["m2_cand"] = {c: v for c, v in allc.items()}
    rec["m2_nulls"] = nulls

    # ---- M3 (i) + M5: asked prompts
    asked = {}
    for kk in ASK:
        s = build_ask(kk, it.prompt.strip("\n"))
        aids = tok(s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        aacts, alog = fwd(aids)
        last = alog[-1:]
        g = gen(aids, 1, False, max_new=8)[0]
        allc2 = {**cands, **extra}
        jv = t.stack([tv("jpp", aacts[L][-1:], L)[0] for L in LAYERS])
        jr = rank_mat(lg_of(jv), list(allc2.values()))
        av = t.stack([tv("alone", aacts[L][-1:], L)[0] for L in LAYERS])
        ar = rank_mat(lg_of(av), list(allc2.values()))
        # cosine of the J++ vector with the concept's (gamma-scaled) unembedding rows, max over candidates
        jro = t.stack([tv("jpp", acts[L][rp:rp + 1], L)[0] for L in LAYERS])
        cosd = {}
        for c, cid in allc2.items():
            Wc = (WU[t.tensor(cid, device=WU.device)].float() * GAMMA)
            Wc = Wc / Wc.norm(dim=-1, keepdim=True)
            cosd[c] = dict(asked=(t.nn.functional.normalize(jv, dim=-1) @ Wc.T).max(-1).values.tolist(),
                           readout=(t.nn.functional.normalize(jro, dim=-1) @ Wc.T).max(-1).values.tolist())
        asked[kk] = dict(answer=tok.decode(g), first_rank_u=rank_mat(last, list(allc2.values()))[0].tolist(),
                         top5=[dec(i) for i in last[0].topk(5).indices.tolist()],
                         jpp_rank=jr.T.tolist(), alone_rank=ar.T.tolist(), cos=cosd, keys=list(allc2))
    rec["asked"] = asked
    out_items.append(rec)
    log(f"{k + 1}/{len(items)} {it.eval_slug}/{it.name} S={S} {time.time() - ti:.1f}s | greedy {rec['cont']['greedy'][:40]!r}"
        f" | topic {asked['topic']['answer'][:20]!r} | unstated {asked['unstated']['answer'][:20]!r}")

log(f"resid check vs bench: min cos {min(cos_check):.5f}")
res = dict(tag=args.tag, model=CFG["model"], layers=LAYERS, lam_L=LENS["L"].lam, n_null=args.n_null,
           resid_min_cos=min(cos_check), note="all lens outputs unit-scaled (rms 1) before the bf16 LM head; "
           "ranks filtered (bench excluded ids -> -inf) unless *_u", items=out_items)
p = f"{R}/runs/ex3/raw_{args.tag}{'_lim' if args.limit else ''}.json"
with open(p, "w") as f:
    json.dump(res, f, ensure_ascii=False)
log("wrote", p)
