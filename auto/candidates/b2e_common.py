"""Shared code for b2e_* candidates (ideas 7 + 18). Not a candidate itself.

Idea 7: inside build(), the model writes one-sentence scenes that evoke a concept word without naming it.
Each scene (cut at its first sentence-final period) is run alone; the residual at that final period at the
7 readout layers is paired with the concept's leading-space token. Concepts are split 80/20 (by concept)
into train / held-out.

Corrections on top of unit(JPP_l h) (x = unit(h)):
  rank   v = unit(JPP_l h) + U_l V_l^T x          (CE over the filtered vocabulary through the real unembed)
  ridge  v = unit(JPP_l h) + b * dA_l (x - mu_l)   (idea 18: kernel ridge from x to the centred, rms-normalised
                                                   effective unembedding direction g*W_c of the target)
Data: "gen" (generated pairs), "fit" (ctx.fit only, control), "genfit" (both).
"""

import random
import time

import torch
import torch.nn.functional as F

from jlens.hooks import ActivationRecorder

CONCEPTS = """
grief joy fear anger shame guilt pride envy jealousy hope despair loneliness nostalgia regret relief anxiety
boredom curiosity gratitude love hatred betrayal revenge forgiveness courage cowardice greed generosity
kindness cruelty patience panic embarrassment surprise disgust sadness happiness trust loyalty ambition
freedom justice poverty wealth war peace death birth marriage divorce friendship childhood retirement
victory defeat failure success danger safety silence chaos luck faith sacrifice temptation addiction
illness recovery hunger thirst exhaustion sleep dream nightmare memory secret lie truth deception
competition crime punishment escape adventure travel winter summer autumn spring storm drought flood
fire earthquake volcano rain snow fog darkness sunrise sunset night morning
doctor nurse teacher lawyer judge soldier pilot farmer chef baker butcher fisherman sailor carpenter
plumber electrician mechanic firefighter detective spy thief priest monk nun king queen prince princess
president politician scientist astronaut engineer architect painter sculptor poet novelist journalist
photographer actor singer dancer musician drummer guitarist pianist magician clown waiter librarian
dentist surgeon pharmacist veterinarian banker accountant cashier janitor gardener miner hunter shepherd
tailor barber jeweler blacksmith programmer student professor coach referee goalkeeper boxer wrestler
hospital school prison church temple library museum airport station harbor beach desert jungle forest
mountain island volcano cave castle palace farm factory office kitchen bathroom bedroom garden cemetery
casino circus theater stadium gym bakery restaurant bar hotel market supermarket bank courtroom
laboratory zoo aquarium submarine spaceship train airplane bicycle motorcycle elevator
Paris London Tokyo Rome Venice Egypt India China Japan Russia Mexico Brazil Africa Antarctica Hollywood
Vegas Moscow Berlin Athens Jerusalem Mars Moon Titanic Olympics Christmas Halloween Easter Thanksgiving
wedding funeral birthday graduation vacation picnic party concert festival parade election revolution
Shakespeare Einstein Newton Darwin Napoleon Lincoln Beethoven Mozart Picasso Elvis Hitler Gandhi Jesus
Buddha Freud Edison Tesla Columbus Caesar Cleopatra Dracula Sherlock Batman Superman Santa Cinderella
pirate vampire ghost witch wizard dragon robot alien zombie angel devil mermaid unicorn
football basketball baseball tennis golf chess poker swimming skiing surfing fishing hunting camping
boxing cycling running climbing gambling cooking painting knitting gardening dancing singing
horror comedy romance tragedy mystery fantasy western opera jazz rap ballet
coffee tea wine beer whiskey chocolate bread cheese pizza honey milk sugar salt pepper garlic lemon
apple banana orange strawberry mushroom potato rice soup cake cookie candy
dog cat horse cow pig sheep chicken wolf bear lion tiger elephant monkey snake spider bee shark whale
dolphin eagle owl crow rabbit mouse rat frog turtle butterfly mosquito
money gold diamond knife gun sword bomb rope key mirror clock candle umbrella telephone computer camera
piano violin guitar drum book newspaper letter map ticket passport wallet ring crown mask bridge
tower wall door window ladder chair bed blanket pillow shoes hat glasses
ocean river lake sea sky star cloud wind thunder lightning ice smoke dust sand blood tears sweat
age youth birthday pregnancy baby mother father grandmother orphan widow twins wedding honeymoon
homework exam interview promotion salary debt tax rent inheritance bankruptcy strike protest
""".split()

FEWSHOT = [
    [("loneliness", "Every evening he set the table for one and ate while the radio talked to nobody."),
     ("winter", "We scraped frost off the windshield while our breath hung in clouds over the frozen driveway."),
     ("dentist", "She gripped the armrests as the drill whined and the bright lamp shone into her open mouth.")],
    [("victory", "The crowd rushed onto the field as the players lifted the trophy above their heads."),
     ("baker", "Long before dawn he was already dusted with flour, sliding the first loaves into the hot oven."),
     ("Napoleon", "The short emperor stared across the snowy plains as his frozen army retreated from Moscow.")],
]
HEAD = "Write one sentence that strongly evokes the concept without using the word itself.\n\n"


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


def log(*a):
    print("[b2e]", *a, flush=True)


# ---------------------------------------------------------------- data generation


def concept_tokens(tok):
    out, seen = [], set()
    for w in CONCEPTS:
        if w.lower() in seen:
            continue
        seen.add(w.lower())
        ids = tok(" " + w, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.append((w, ids[0]))
    return out


def _prompt(shots, w):
    s = HEAD
    for c, sc in shots:
        s += f"Concept: {c}\nScene: {sc}\n\n"
    return s + f"Concept: {w}\nScene:"


def _contains(scene, w):
    s, w = scene.lower(), w.lower()
    stem = w[:-2] if len(w) >= 6 else w
    return stem in s


def _cut(text):
    text = text.split("\n")[0].strip()
    for i, ch in enumerate(text):
        if ch == "." and (i + 1 == len(text) or text[i + 1] in " \"'") and i >= 20:
            return text[: i + 1]
    return None


@torch.no_grad()
def generate_scenes(ctx, concepts, batch=64, max_new=32):
    hf, tok = ctx.model._hf_model, ctx.model.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    nl_ids = [i for i in range(len(tok)) if "\n" in (t := tok.decode([i])) and t.strip() == ""] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    jobs = [(w, tid, k) for k in range(len(FEWSHOT)) for (w, tid) in concepts]
    raw = []
    for i in range(0, len(jobs), batch):
        chunk = jobs[i:i + batch]
        enc = tok([_prompt(FEWSHOT[k], w) for w, _, k in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(ctx.device)
        out = hf.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                          eos_token_id=nl_ids)
        texts = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        raw += [(w, tid, k, t) for (w, tid, k), t in zip(chunk, texts)]
    tok.padding_side = old_side
    kept, n_named, n_cut = [], 0, 0
    for w, tid, k, t in raw:
        sc = _cut(t)
        if sc is None:
            n_cut += 1
            continue
        if _contains(sc, w):
            n_named += 1
            continue
        kept.append((w, tid, sc))
    return kept, dict(raw=len(raw), named=n_named, uncut=n_cut, kept=len(kept)), raw


@torch.no_grad()
def scene_resid(ctx, scenes, batch=64):
    """Residuals at the final period of each scene (run alone), readout layers. Drops scenes whose last
    token is not a period token."""
    tok, model, layers = ctx.model.tokenizer, ctx.model, ctx.readout_layers
    ids_list, keep = [], []
    for j, sc in enumerate(scenes):
        ids = tok(sc, add_special_tokens=False).input_ids
        if tok.decode(ids[-1:]).strip() == ".":
            ids_list.append(ids)
            keep.append(j)
    out = {l: [] for l in layers}
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(ids_list), batch):
        chunk = ids_list[i:i + batch]
        T = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, :len(x)] = torch.as_tensor(x)
        last = torch.as_tensor([len(x) - 1 for x in chunk])
        with ActivationRecorder(model.layers, at=layers) as rec:
            model.forward(ids.to(ctx.device))
            acts = dict(rec.activations)
        for l in layers:
            out[l].append(acts[l][torch.arange(len(chunk)), last.to(acts[l].device)].float())
    return {l: torch.cat(v) for l, v in out.items()}, keep


def build_data(ctx, seed=0):
    t0 = time.time()
    tok = ctx.model.tokenizer
    concepts = concept_tokens(tok)
    kept, stats, raw = generate_scenes(ctx, concepts)
    log(f"concepts {len(concepts)} (single-token of {len(set(w.lower() for w in CONCEPTS))}); "
        f"gen stats {stats}; {time.time() - t0:.0f}s")
    for w, tid, k, t in raw[:3] + raw[len(raw) // 2:len(raw) // 2 + 3]:
        log(f"  example [{w}] {t.strip()[:160]!r}")
    H, keep = scene_resid(ctx, [s for _, _, s in kept])
    kept = [kept[j] for j in keep]
    words = sorted(set(w for w, _, _ in kept))
    rng = random.Random(seed)
    rng.shuffle(words)
    ho_words = set(words[: len(words) // 5])
    tgt = torch.as_tensor([tid for _, tid, _ in kept], device=ctx.device)
    is_ho = torch.as_tensor([w in ho_words for w, _, _ in kept], device=ctx.device)
    log(f"pairs {len(kept)} from {len(words)} concepts; held-out concepts {len(ho_words)} "
        f"({int(is_ho.sum())} pairs); {time.time() - t0:.0f}s")
    for l in ctx.readout_layers:
        log(f"  L{l} rms gen {H[l].pow(2).mean().sqrt():.2f} vs fit {ctx.fit.resid[l].pow(2).mean().sqrt():.2f}")
    return dict(H=H, tgt=tgt, ho=is_ho, words=[w for w, _, _ in kept], scenes=[s for _, _, s in kept])


def fit_pairs(ctx):
    """ctx.fit as (resid, one target id) pairs: one row per (item, intermediate), first candidate id."""
    rows, tg, slugs = [], [], []
    for i, cands in enumerate(ctx.fit.candidates):
        for name, ids in cands.items():
            if ids:
                rows.append(i)
                tg.append(ids[0])
                slugs.append(ctx.fit.eval_slugs[i])
    r = torch.as_tensor(rows, device=ctx.device)
    return {l: ctx.fit.resid[l][r] for l in ctx.readout_layers}, torch.as_tensor(tg, device=ctx.device), r


# ---------------------------------------------------------------- readout helpers


def excl_mask(ctx):
    V = ctx.model._lm_head.weight.shape[0]
    m = torch.zeros(V, dtype=torch.bool, device=ctx.device)
    m[ctx.fit.excluded_ids.to(ctx.device)] = True
    return m


def logits_of(ctx, v, mask):
    lg = ctx.model.unembed(v).float()
    return lg.masked_fill(mask[None, :lg.shape[1]], float("-inf"))


def ranks(lg, tgt):
    t = lg.gather(1, tgt[:, None])
    return (lg > t).sum(1) + 1


def fit_score(ctx, lens):
    fit, model = ctx.fit, ctx.model
    mask = excl_mask(ctx)
    best = {}
    for l in ctx.readout_layers:
        v = lens.transport(fit.resid[l].clone(), l, {k: x.clone() for k, x in fit.resid.items()})
        lg = logits_of(ctx, v, mask)
        for i, cands in enumerate(fit.candidates):
            for name, ids in cands.items():
                bl = lg[i, torch.as_tensor(ids, device=lg.device)].max()
                r = int((lg[i] > bl).sum()) + 1
                best[(i, name)] = min(best.get((i, name), 10**9), r)
    per = {}
    for (i, name), r in best.items():
        per.setdefault(fit.eval_slugs[i], []).append(float(r <= 10))
    per = {e: sum(v) / len(v) for e, v in sorted(per.items())}
    return sum(per.values()) / len(per), per


def fmt(score):
    return f"{score[0]:.3f} (" + " ".join(f"{k[:5]} {v:.2f}" for k, v in score[1].items()) + ")"


# ---------------------------------------------------------------- lenses


class Lens:
    def __init__(self, ctx, corr):
        self.JPP, self.corr = ctx.JPP, corr  # corr[layer] = callable(x_unit) -> delta, or None

    def transport(self, h, layer, resid_all=None):
        u = unit(h @ self.JPP[layer].T)
        c = self.corr.get(layer) if self.corr else None
        if c is None:
            return u
        return u + c(unit(h))


def eval_pairs(ctx, lens, H, tgt, sel, mask):
    """top-10 hit rate (filtered) at best layer and per layer, on rows sel."""
    best = None
    per = {}
    for l in ctx.readout_layers:
        lg = logits_of(ctx, lens.transport(H[l][sel], l), mask)
        r = ranks(lg, tgt[sel])
        per[l] = float((r <= 10).float().mean())
        best = r if best is None else torch.minimum(best, r)
    return float((best <= 10).float().mean()), per


# ---------------------------------------------------------------- rank-r CE correction


def train_rank(ctx, H, tgt, r=8, steps=300, lr=1e-3, wd=1e-2, seed=0, val=None, every=20):
    """val = (Hval, tval): keep, per layer, the step with the best held-out top-10 (ties: lower val CE)."""
    mask = excl_mask(ctx)
    W = ctx.model._lm_head.weight  # [V, d] bf16
    d = W.shape[1]
    corr = {}
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.enable_grad():
        for l in ctx.readout_layers:
            x = unit(H[l])
            base = unit(H[l] @ ctx.JPP[l].T)
            Vm = (torch.randn(d, r, generator=g) / d ** 0.5).to(ctx.device).requires_grad_(True)
            Um = torch.zeros(d, r, device=ctx.device, requires_grad=True)
            opt = torch.optim.AdamW([Um, Vm], lr=lr, weight_decay=wd)
            best = None
            for s in range(steps):
                v = base + (x @ Vm) @ Um.T
                lg = F.linear(ctx.model._final_norm(v.to(W.dtype)), W).float()
                lg = lg.masked_fill(mask[None, :lg.shape[1]], float("-inf"))
                loss = F.cross_entropy(lg, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if s == 0:
                    l0 = float(loss.detach())
                if val is not None and (s + 1) % every == 0:
                    with torch.no_grad():
                        xv = unit(val[0][l])
                        vv = unit(val[0][l] @ ctx.JPP[l].T) + (xv @ Vm) @ Um.T
                        lgv = logits_of(ctx, vv, mask)
                        acc = float((ranks(lgv, val[1]) <= 10).float().mean())
                        ce = float(F.cross_entropy(lgv, val[1]))
                    if best is None or (acc, -ce) > best[:2]:
                        best = (acc, -ce, s + 1, Um.detach().clone(), Vm.detach().clone())
                if s == 0 and val is not None:
                    with torch.no_grad():
                        lgv = logits_of(ctx, unit(val[0][l] @ ctx.JPP[l].T), mask)
                        best = (float((ranks(lgv, val[1]) <= 10).float().mean()), -1e9, 0,
                                torch.zeros_like(Um), Vm.detach().clone())
            if val is not None:
                log(f"  rank{r} L{l}: CE {l0:.2f} -> {float(loss):.2f}; best val step {best[2]} top10 {best[0]:.2f}")
                U_, V_ = best[3], best[4]
            else:
                log(f"  rank{r} L{l}: CE {l0:.2f} -> {float(loss):.2f}")
                U_, V_ = Um.detach(), Vm.detach()
            corr[l] = (lambda U_, V_: (lambda xu: (xu @ V_) @ U_.T))(U_, V_)
    return corr


# ---------------------------------------------------------------- ridge (idea 18)


def eff_dirs(ctx, ids):
    W = ctx.model._lm_head.weight
    d = W.shape[1]
    gn = ctx.model._final_norm(torch.ones(1, d, device=W.device, dtype=W.dtype)).float()[0]
    return unit(W[ids].float() * gn[None])


def train_ridge(ctx, H, tgt, lam):
    E = eff_dirs(ctx, tgt)
    E = E - E.mean(0, keepdim=True)
    corr = {}
    for l in ctx.readout_layers:
        X = unit(H[l])
        mu = X.mean(0, keepdim=True)
        Xc = X - mu
        K = Xc @ Xc.T
        lam_abs = lam * K.diagonal().mean()
        A = torch.linalg.solve(K + lam_abs * torch.eye(K.shape[0], device=K.device), E)  # [N, d]
        corr[l] = (lambda A, Xc, mu: (lambda xu: ((xu - mu) @ Xc.T) @ A))(A, Xc, mu)
    return corr


def scaled(corr, b):
    return {l: (lambda f: (lambda xu: b * f(xu)))(f) for l, f in corr.items()}


# ---------------------------------------------------------------- driver


def _cv_ridge_fit(ctx, Hf, tf, rows, grid, folds=5):
    """5-fold CV over fit items: best (lam, b) by mean top-10 of held-out fold rows (best layer)."""
    mask = excl_mask(ctx)
    items = sorted(set(rows.tolist()))
    random.Random(0).shuffle(items)
    fold_of = {it: k % folds for k, it in enumerate(items)}
    fo = torch.as_tensor([fold_of[int(i)] for i in rows], device=ctx.device)
    res = {}
    for lam in grid["lam"]:
        hits = {b: [] for b in grid["b"]}
        for k in range(folds):
            tr, te = fo != k, fo == k
            corr = train_ridge(ctx, {l: Hf[l][tr] for l in Hf}, tf[tr], lam)
            for b in grid["b"]:
                lens = Lens(ctx, scaled(corr, b))
                acc, _ = eval_pairs(ctx, lens, Hf, tf, te, mask)
                hits[b].append(acc * int(te.sum()))
        for b in grid["b"]:
            res[(lam, b)] = sum(hits[b]) / len(rows)
    base = eval_pairs(ctx, Lens(ctx, None), Hf, tf, torch.ones_like(rows, dtype=torch.bool), mask)[0]
    log(f"fit-CV ridge: J++ {base:.3f}; " + " ".join(f"lam{a}/b{b}:{v:.3f}" for (a, b), v in res.items()))
    return max(res, key=res.get)


def build_variant(ctx, kind, src="gen", r=8, steps=300, lam=None, b=None, shuffle=False, es=False, wd=1e-2):
    t0 = time.time()
    mask = excl_mask(ctx)
    log(f"variant kind={kind} src={src} r={r} steps={steps} lam={lam} b={b} shuffle={shuffle}")
    log(f"fit J++ pass10: {fmt(fit_score(ctx, Lens(ctx, None)))}")
    data = build_data(ctx) if src in ("gen", "genfit") else None
    Hf, tf, rows = fit_pairs(ctx)
    if data is not None:
        tr, ho = ~data["ho"], data["ho"]
        H, tgt = data["H"], data["tgt"]
        if shuffle:
            perm = torch.randperm(int(tr.sum()), generator=torch.Generator().manual_seed(1)).to(ctx.device)
            tgt = tgt.clone()
            tgt[tr] = tgt[tr][perm]
        base_ho = eval_pairs(ctx, Lens(ctx, None), H, data["tgt"], ho, mask)
        base_tr = eval_pairs(ctx, Lens(ctx, None), H, data["tgt"], tr, mask)
        log(f"J++ gen top10 best-layer: train {base_tr[0]:.3f} held-out {base_ho[0]:.3f}; per-layer held-out "
            + " ".join(f"L{l}:{v:.2f}" for l, v in base_ho[1].items()))
        Htr = {l: H[l][tr] for l in H}
        ttr = tgt[tr]
        if src == "genfit":
            Htr = {l: torch.cat([Htr[l], Hf[l]]) for l in H}
            ttr = torch.cat([ttr, tf])
    else:
        Htr, ttr = Hf, tf
    if kind == "rank":
        val = ({l: H[l][ho] for l in H}, data["tgt"][ho]) if (es and data is not None) else None
        corr = train_rank(ctx, Htr, ttr, r=r, steps=steps, val=val, wd=wd)
    elif kind == "ridge":
        if lam is None and data is not None:
            # choose (lam, b) on held-out generated concepts
            best, res = None, {}
            for la in (0.03, 0.3, 3.0):
                c0 = train_ridge(ctx, Htr, ttr, la)
                for bb in (0.25, 0.5, 1.0, 2.0):
                    res[(la, bb)] = eval_pairs(ctx, Lens(ctx, scaled(c0, bb)), H, data["tgt"], ho, mask)[0]
            log("ridge held-out grid: " + " ".join(f"lam{a}/b{c}:{v:.3f}" for (a, c), v in res.items()))
            lam, b = max(res, key=res.get)
        elif lam is None:
            lam, b = _cv_ridge_fit(ctx, Hf, tf, rows, dict(lam=(0.03, 0.3, 3.0), b=(0.25, 0.5, 1.0, 2.0)))
        log(f"ridge chosen lam={lam} b={b}")
        corr = scaled(train_ridge(ctx, Htr, ttr, lam), b)
    else:
        raise ValueError(kind)
    lens = Lens(ctx, corr)
    if data is not None:
        a_ho = eval_pairs(ctx, lens, H, data["tgt"], ho, mask)
        a_tr = eval_pairs(ctx, lens, H, data["tgt"], tr, mask)
        log(f"corrected gen top10 best-layer: train {a_tr[0]:.3f} held-out {a_ho[0]:.3f}; per-layer held-out "
            + " ".join(f"L{l}:{v:.2f}" for l, v in a_ho[1].items()))
    log(f"fit corrected pass10: {fmt(fit_score(ctx, lens))}  (in-sample if src uses fit)")
    log(f"build {time.time() - t0:.0f}s")
    return lens
