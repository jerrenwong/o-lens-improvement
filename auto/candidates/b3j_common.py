"""Shared code for b3j_* candidates: is the b2e fit-ridge gain on multilingual memorisation of recurring
language-name labels? Reuses b2e_common (ridge lam=0.3, b=0.25, as chosen by b2e's fit-CV). Not a candidate."""

import importlib.util
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("b2e_common", Path(__file__).with_name("b2e_common.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
Lens = B.Lens

LAM, BETA = 0.3, 0.25
LANGS = {"english", "german", "french", "spanish", "italian", "portuguese", "dutch", "russian", "chinese",
         "japanese", "korean", "arabic", "hindi", "turkish", "polish", "swedish", "greek", "hebrew", "latin",
         "danish", "norwegian", "finnish", "czech", "vietnamese", "thai", "indonesian", "ukrainian", "persian"}
KINDS = ["lang", "relation", "source", "answer"]


def log(*a):
    print("[b3j]", *a, flush=True)


def ml_kind(name, pos):
    if name.strip().lower() in LANGS:
        return "lang"
    return KINDS[pos] if pos < len(KINDS) else f"pos{pos}"


def rows(ctx):
    """One row per (fit item, intermediate with ids): dict(item, slug, key, kind, ids)."""
    out = []
    for i, cands in enumerate(ctx.fit.candidates):
        slug = ctx.fit.eval_slugs[i]
        for pos, (name, ids) in enumerate(cands.items()):
            if not ids:
                continue
            kind = ml_kind(name, pos) if slug == "multilingual" else slug
            out.append(dict(item=i, slug=slug, key=name, kind=kind, ids=list(ids)))
    return out


def train(ctx, R, sel):
    """Ridge (b2e) trained on rows R[j] for j in sel (first id as target), scaled by BETA."""
    r = torch.as_tensor([R[j]["item"] for j in sel], device=ctx.device)
    t = torch.as_tensor([R[j]["ids"][0] for j in sel], device=ctx.device)
    H = {l: ctx.fit.resid[l][r] for l in ctx.readout_layers}
    return B.scaled(B.train_ridge(ctx, H, t, LAM), BETA)


@torch.no_grad()
def hits(ctx, lens, R, sel, mask):
    """{j: hit@10} for rows sel, best layer, max over candidate ids (as fit_score)."""
    items = sorted(set(R[j]["item"] for j in sel))
    ii = torch.as_tensor(items, device=ctx.device)
    best = {}
    for l in ctx.readout_layers:
        lg = B.logits_of(ctx, lens.transport(ctx.fit.resid[l][ii].clone(), l), mask)
        row_of = {it: k for k, it in enumerate(items)}
        for j in sel:
            k = row_of[R[j]["item"]]
            bl = lg[k, torch.as_tensor(R[j]["ids"], device=lg.device)].max()
            rk = int((lg[k] > bl).sum()) + 1
            best[j] = min(best.get(j, 10**9), rk)
    return {j: float(r <= 10) for j, r in best.items()}


def by_kind(R, h):
    d = defaultdict(list)
    for j, v in h.items():
        d[R[j]["kind"]].append(v)
    return {k: (sum(v) / len(v), len(v)) for k, v in sorted(d.items())}


def fmtk(d):
    return " ".join(f"{k} {a:.2f}/{n}" for k, (a, n) in d.items())


def recurrence(ctx, R):
    by = defaultdict(Counter)
    for r in R:
        by[r["slug"]][r["key"]] += 1
    for slug, c in sorted(by.items()):
        n_items = len(set(r["item"] for r in R if r["slug"] == slug))
        rep = {k: v for k, v in c.most_common() if v > 1}
        log(f"{slug}: items {n_items}, rows {sum(c.values())}, distinct keys {len(c)}, recurring {len(rep)}: {rep}")
        log(f"   all keys: {sorted(c)}")
    for i, cands in enumerate(ctx.fit.candidates):
        if ctx.fit.eval_slugs[i] == "multilingual":
            log(f"   ML item {ctx.fit.item_names[i]}: " + ", ".join(
                f"{n}[{ml_kind(n, p)}]{'(no ids)' if not ids else ''}" for p, (n, ids) in enumerate(cands.items())))


def diagnose(ctx):
    mask = B.excl_mask(ctx)
    R = rows(ctx)
    recurrence(ctx, R)
    base = Lens(ctx, None)
    ml = [j for j, r in enumerate(R) if r["slug"] == "multilingual"]
    allj = list(range(len(R)))
    h0 = hits(ctx, base, R, allj, mask)

    # ---- 5-fold CV over fit items (same folds as b2e), all tasks
    items = sorted(set(r["item"] for r in R))
    random.Random(0).shuffle(items)
    fold = {it: k % 5 for k, it in enumerate(items)}
    hcv = {}
    for k in range(5):
        tr = [j for j in allj if fold[R[j]["item"]] != k]
        te = [j for j in allj if fold[R[j]["item"]] == k]
        hcv.update(hits(ctx, Lens(ctx, train(ctx, R, tr)), R, te, mask))
    log(f"5-fold CV, all rows by kind  J++: {fmtk(by_kind(R, h0))}")
    log(f"5-fold CV, all rows by kind ridge: {fmtk(by_kind(R, hcv))}")

    # ---- leave-one-language-out on multilingual items
    lang_items = defaultdict(set)
    for j in ml:
        if R[j]["kind"] == "lang":
            lang_items[R[j]["key"].strip().lower()].add(R[j]["item"])
    hlo = {}
    log("LOLO per language: lang n_items | ML rows J++ -> 5fold -> LOLO (lang-row hits J++/5fold/LOLO)")
    for lang, its in sorted(lang_items.items()):
        tr = [j for j in allj if R[j]["item"] not in its]
        te = [j for j in allj if R[j]["item"] in its]
        h = hits(ctx, Lens(ctx, train(ctx, R, tr)), R, te, mask)
        hlo.update(h)
        lr = [j for j in te if R[j]["kind"] == "lang"]
        f = lambda d, js: sum(d[j] for j in js) / max(1, len(js))
        log(f"  {lang:10s} {len(its)} | {f(h0, te):.2f} -> {f(hcv, te):.2f} -> {f(h, te):.2f} "
            f"(lang {f(h0, lr):.2f}/{f(hcv, lr):.2f}/{f(h, lr):.2f})")
    covered = [j for j in ml if j in hlo]
    log(f"ML rows covered by LOLO: {len(covered)} of {len(ml)}")
    for name, d in (("J++", h0), ("5-fold ridge", hcv), ("LOLO ridge", hlo)):
        log(f"TABLE {name:13s}: {fmtk(by_kind(R, {j: d[j] for j in covered}))}")
    # sanity: LOLO but holding out a random same-size item set (is the drop specific to the language?)
    rng = random.Random(1)
    hrand = {}
    for lang, its in sorted(lang_items.items()):
        pool = [it for it in set(R[j]["item"] for j in ml)]
        its2 = set(rng.sample(pool, len(its)))
        tr = [j for j in allj if R[j]["item"] not in its2]
        te = [j for j in ml if R[j]["item"] in its2]
        hrand.setdefault(lang, []).append((te, hits(ctx, Lens(ctx, train(ctx, R, tr)), R, te, mask)))
    agg = defaultdict(list)
    for v in hrand.values():
        for te, h in v:
            for j in te:
                agg[R[j]["kind"]].append(h[j])
    log("TABLE random-item-holdout (same sizes, ML only): " + " ".join(
        f"{k} {sum(v) / len(v):.2f}/{len(v)}" for k, v in sorted(agg.items())))
    return R


def build_ablation(ctx, which):
    mask = B.excl_mask(ctx)
    R = rows(ctx)
    if which == "full":
        sel = list(range(len(R)))
    elif which == "noml":
        sel = [j for j, r in enumerate(R) if r["slug"] != "multilingual"]
    elif which == "mlonly":
        sel = [j for j, r in enumerate(R) if r["slug"] == "multilingual"]
    elif which == "nolang":
        sel = [j for j, r in enumerate(R) if r["kind"] != "lang"]
    elif which == "langonly":
        sel = [j for j, r in enumerate(R) if r["kind"] == "lang"]
    else:
        raise ValueError(which)
    log(f"ablation {which}: training rows {len(sel)} of {len(R)}; kinds "
        f"{dict(Counter(R[j]['kind'] for j in sel))}")
    lens = Lens(ctx, train(ctx, R, sel))
    log(f"fit in-sample (all rows) by kind: {fmtk(by_kind(R, hits(ctx, lens, R, list(range(len(R))), mask)))}")
    return lens


def diagnose_keys(ctx):
    """Leave-one-item-out on multilingual items, excluding from training every fit item that shares a key
    with the test item in a given role (strict) vs plain LOO."""
    mask = B.excl_mask(ctx)
    R = rows(ctx)
    allj = list(range(len(R)))
    ml_items = sorted(set(r["item"] for r in R if r["slug"] == "multilingual"))
    keyset = {it: {(r["kind"], r["key"].strip().lower()) for r in R if r["item"] == it} for it in ml_items}
    words = {it: {k for _, k in keyset[it]} for it in ml_items}
    h0 = hits(ctx, Lens(ctx, None), R, [j for j in allj if R[j]["slug"] == "multilingual"], mask)
    rules = {
        "LOO": lambda it, o: o == it,
        "share-relation": lambda it, o: o == it or any(k == "relation" and (k, w) in keyset.get(o, ()) for k, w in keyset[it]),
        "share-source": lambda it, o: o == it or any(k == "source" and (k, w) in keyset.get(o, ()) for k, w in keyset[it]),
        "share-answer": lambda it, o: o == it or any(k == "answer" and (k, w) in keyset.get(o, ()) for k, w in keyset[it]),
        "share-any-word": lambda it, o: o == it or bool(words[it] & words.get(o, set())),
        "drop-all-ML": lambda it, o: o in keyset,
    }
    log("J++ ML by kind: " + fmtk(by_kind(R, h0)))
    for name, rule in rules.items():
        h, ntr = {}, []
        for it in ml_items:
            excl = {o for o in set(r["item"] for r in R) if rule(it, o)}
            tr = [j for j in allj if R[j]["item"] not in excl]
            ntr.append(len(excl))
            te = [j for j in allj if R[j]["item"] == it]
            h.update(hits(ctx, Lens(ctx, train(ctx, R, tr)), R, te, mask))
        log(f"TABLE2 {name:15s} (mean items excluded {sum(ntr) / len(ntr):.1f}): {fmtk(by_kind(R, h))}")
