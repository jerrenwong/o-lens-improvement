"""Qual dump (scripts/qual/): wraps ANY candidate lens and writes a qual dump of it, then returns that lens unchanged.

    QUAL_LENS=auto/candidates/alens.py ALENS_BASE=JPP QUAL_NAME="J++ + A-lens" QUAL_PROBES=probes/probes_v1.json \
      /root/jpp_lens/.venv/bin/python auto/evaluate.py --lens auto/candidates/qual_dump.py --model Qwen/Qwen3.5-9B \
      --filter f4 --name qual_9b_jpp_a_lens            # (scripts/qual/dump.sh does exactly this)

Env: QUAL_LENS (candidate file, required) plus that candidate's own env vars; QUAL_NAME (display name, default the file
stem); QUAL_PROBES (probe json, optional; needs the full model, so NOT with --lite); QUAL_SETS (default
"bench-fit,bench-dev,probes"); QUAL_TOPK (default 20); QUAL_NOTE (free text for meta.built).
Sets: bench-fit = ctx.fit; bench-dev = the dev rows of auto/split_<tag>.json, read from the model's bench file (test and
held-out-only rows are never read); probes = raw text, read at its LAST token (model.encode = harness tokenization).
Ranks and top lists: filter f4 (auto/filters.py, word groups), whatever --filter the lens was built under (recorded).
Probe words with no single-token form are matched by their first token piece (>= 3 chars; forms[].kind = "piece");
per probe: is_next (accepted form in the model's own f4 next-token top 10), scorable (some usable form).
Writes runs/qual/<tag>/<slug of QUAL_NAME>.json (format: scripts/qual/qualfmt.py). Tested on 9B 2026-11-27 (q1).
"""
import datetime
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
R = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R / "src"))
from olens.readout import Bench, group_candidate_ranks, unembed_logits  # noqa: E402

TAG = {2048: "2b", 4096: "9b", 5120: "27b"}
KNOWS_SUFFIX = "\nThe thing this is about, in one word:"
SITE = "/workspace/fleet/projects/o-lens/site/data"


def log(*a):
    print("[qual]", *a, flush=True)


def slug(name):
    s = name.replace("++", "pp").replace("+", "_").lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or "lens"


def surface_forms(word):
    """= lens_evals.eval_utils.surface_forms (the benchmark's rule)."""
    cased = {word, word.capitalize(), word.lower()}
    return [prefix + form for form in sorted(cased) for prefix in ("", " ")]


def candidate_ids(tok, words):
    """Per word: single-token ids of its surface forms (= lens_evals candidate_token_ids, the benchmark rule), kind
    "whole"; a word with none gets the FIRST token piece of each surface form instead, kind "piece", when the piece
    is >= 3 characters (stripped). -> (whole ids, piece ids, forms [{word, token, kind}], unusable words)."""
    whole, piece, forms, bad = set(), set(), [], []
    for w in words:
        got = set()
        for s in surface_forms(w):
            t = tok(s, add_special_tokens=False).input_ids
            if len(t) == 1:
                got.add(int(t[0]))
        kind = "whole"
        if not got:
            kind = "piece"
            for s in surface_forms(w):
                t = tok(s, add_special_tokens=False).input_ids
                if t and len(tok.decode([t[0]]).strip()) >= 3:
                    got.add(int(t[0]))
        if not got:
            bad.append(w)
        for t in sorted(got):
            forms.append(dict(word=w, token=tok.decode([t]), kind=kind))
        (whole if kind == "whole" else piece).update(got)
    return sorted(whole), sorted(piece - whole), forms, bad


def load_candidate(path):
    spec = importlib.util.spec_from_file_location(f"qual_inner_{Path(path).stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def env_used(path, depth=2):
    """Env vars read by the candidate file AND by the sibling candidate modules it imports (w9_alens.py -> alens.py:
    ALENS_E / ALENS_E2 / ALENS_BASE), up to `depth` import levels (fix 2026-11-27: wrapper candidates recorded {})."""
    names, seen, todo = set(), set(), [(Path(path), 0)]
    while todo:
        p, d = todo.pop()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        src = p.read_text()
        names |= set(re.findall(r"environ(?:\.get)?[\[(]\s*[\"']([A-Z0-9_]+)[\"']", src))
        if d < depth:
            for m in re.findall(r"^\s*(?:from\s+(\w+)\s+import|import\s+(\w+))", src, re.M):
                todo.append((p.parent / f"{m[0] or m[1]}.py", d + 1))
    return {k: os.environ[k] for k in sorted(names) if k in os.environ}


def bench_texts(tag):
    """(eval, item) -> (text up to and including the readout token, final answer) from the site's readouts json."""
    p = f"{SITE}/readouts_{tag}.json"
    if not os.path.exists(p):
        log(f"no {p}: bench texts left empty")
        return {}
    d = json.load(open(p, encoding="utf-8"))
    V = d["vocab"]
    return {(x["e"], x["n"]): ("".join(V[t] for t in x["toks"][: x["pos"] + 1]).replace("�", "▯"),
                               x.get("target")) for x in d["items"]}


@torch.no_grad()
def record(model, texts, layers, last):
    """Per text: block outputs at the last token for `layers` and the model's next-token logits there."""
    res = {l: [] for l in layers}
    nxt = []
    want = sorted(set(layers) | {last})
    for text in texts:
        ids = model.encode(text)  # harness tokenization: BOS rule of the loaded tokenizer, truncation 512
        rec = {}
        hooks = [model.layers[l].register_forward_hook(
            (lambda l: lambda m, i, o: rec.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, -1].detach().float()))(l))
            for l in want]
        try:
            model.forward(ids)
        finally:
            for h in hooks:
                h.remove()
        for l in layers:
            res[l].append(rec[l])
        nxt.append(model.unembed(rec[last][None]).float()[0])
    return {l: torch.stack(v) for l, v in res.items()}, nxt


def top10(tok, lg):
    return [tok.decode([t]).replace("�", "▯") for t in lg.topk(10).indices.tolist()]


def raw_rank(lg, ids):
    return int((lg > lg[ids].max()).sum()) + 1 if ids else None


def ranks_of(lg, id_lists, gid):
    """Word-group rank of the best id per row (None when the row has no ids)."""
    out = [None] * len(id_lists)
    sel = [i for i, ids in enumerate(id_lists) if ids]
    if sel:
        rk = group_candidate_ranks(lg[sel], [{"c": id_lists[i]} for i in sel], gid)
        for j, i in enumerate(sel):
            out[i] = int(rk[j]["c"])
    return out


def entries(model, lens, rows, resid, layers, F, excl, FL, topk):
    """rows: list of dicts with acc_ids / near_ids / src (row index into resid). -> per row, per layer entry."""
    tok = model.tokenizer
    gid = F.group_ids
    gl = gid.tolist()  # cpu lookups
    gid_dev = None
    src = torch.as_tensor([r["src"] for r in rows], dtype=torch.long)
    out = [[] for _ in rows]
    for L in layers:
        h = resid[L][src.to(resid[L].device)]
        allr = {l: v[src.to(v.device)].clone() for l, v in resid.items()}
        vec = lens.transport(h.clone(), L, allr).detach().float()
        lg = unembed_logits(model, vec, excl)
        if gid_dev is None:
            gid_dev = gid.to(lg.device)
        ra = ranks_of(lg, [r["acc_ids"] for r in rows], gid)
        rn = ranks_of(lg, [r["near_ids"] for r in rows], gid)
        for i, r in enumerate(rows):
            top = FL.group_topk(lg[i], gid_dev, topk)
            ga = {gl[t] for t in r["acc_ids"]}
            gn = {gl[t] for t in r["near_ids"]} - ga
            tg = [gl[t] for t in top]
            gp = {gl[t] for t in r.get("piece_ids", [])}  # accepted ids that are first pieces of multi-token words
            h = [j for j, g in enumerate(tg) if g in ga]
            e = dict(t=[tok.decode([t]).replace("�", "▯") for t in top], r=ra[i], nr=rn[i], h=h,
                     nh=[j for j, g in enumerate(tg) if g in gn])
            hp = [j for j in h if tg[j] in gp and not any(gl[t] == tg[j] for t in r["acc_ids"] if t not in r.get("piece_ids", []))]
            if hp:
                e["hp"] = hp
            out[i].append(e)
    return out


def best_of(layers, ents):
    br = bl = bn = None
    for L, e in zip(layers, ents):
        if e["r"] is not None and (br is None or e["r"] < br):
            br, bl = e["r"], L
        if e["nr"] is not None and (bn is None or e["nr"] < bn):
            bn = e["nr"]
    return bl, br, bn


def build(ctx):
    t0 = time.time()
    inner = os.environ.get("QUAL_LENS")
    if not inner:
        raise SystemExit("qual_dump: set QUAL_LENS=<candidate file>")
    inner = str((R / inner).resolve()) if not os.path.isabs(inner) else inner
    name = os.environ.get("QUAL_NAME") or Path(inner).stem
    sets = [s for s in os.environ.get("QUAL_SETS", "bench-fit,bench-dev,probes").split(",") if s]
    probes_path = os.environ.get("QUAL_PROBES")
    if "probes" in sets and not probes_path:
        log("QUAL_PROBES not set: no probe set")
        sets.remove("probes")
    topk = int(os.environ.get("QUAL_TOPK", "20"))
    model, tok = ctx.model, ctx.model.tokenizer
    if "probes" in sets and getattr(model, "lite", False):
        raise SystemExit("qual_dump: probes need forward passes; --lite loads only the LM head. Rerun without --lite "
                         "(or QUAL_SETS=bench-fit,bench-dev).")
    tag = TAG[model.d_model]
    layers = list(ctx.readout_layers)
    FL = ctx.filters
    F = FL.get("f4")

    # ---- items (and their residual sources) before the lens is built: fail early on bad inputs
    texts = bench_texts(tag)
    groups = {}  # set -> (rows, resid)

    def bench_rows(set_name, slugs, names, cands, resid):
        rows = []
        for i, (e, n, c) in enumerate(zip(slugs, names, cands)):
            text, answer = texts.get((e, n), ("", None))
            for concept, ids in c.items():
                rows.append(dict(id=f"{e}|{n}" + (f"#{concept}" if len(c) > 1 else ""), set=set_name, task=e,
                                 difficulty=None, text=text, concept=concept, answer=answer, accepted=[concept],
                                 accepted_tokens=[tok.decode([t]) for t in ids], near=[], note="", model={},
                                 acc_ids=[int(t) for t in ids], near_ids=[], src=i))
        groups[set_name] = (rows, resid)

    if "bench-fit" in sets:
        f = ctx.fit
        bench_rows("bench-fit", f.eval_slugs, f.item_names, f.candidates, f.resid)
    if "bench-dev" in sets:
        spec = importlib.util.spec_from_file_location("qual_eval_cfg", R / "auto" / "evaluate.py")
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)  # module level only defines MODELS etc.; main() is not run
        cfg = next(c for c in ev.MODELS.values() if c["tag"] == tag)
        bench = Bench.load(R / cfg["bench"])
        split = json.loads((R / f"auto/split_{tag}.json").read_text())
        pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names))}
        dev_idx = sorted(pos[tuple(k)] for k in split["dev"])
        assert not set(dev_idx) & {pos[tuple(k)] for k in split["test"]}, "dev and test overlap"
        assert not set(dev_idx) & set(bench.index("fit")), "dev and fit overlap"
        di = torch.as_tensor(dev_idx, dtype=torch.long)
        resid = {l: v[di].to(ctx.device).float() for l, v in bench.resid.items()}
        bench_rows("bench-dev", [bench.eval_slugs[i] for i in dev_idx], [bench.item_names[i] for i in dev_idx],
                   [bench.candidates[i] for i in dev_idx], resid)
        del bench
    probe_ids = set()
    if "probes" in sets:
        P = json.load(open(probes_path if os.path.isabs(probes_path) else R / probes_path, encoding="utf-8"))
        probes = P["probes"]
        assert len({p["id"] for p in probes}) == len(probes), "probe ids are not unique"
        all_layers = sorted(ctx.fit.resid.keys())
        last = len(model.layers) - 1
        resid, nxt = record(model, [p["text"] for p in probes], all_layers, last)
        resid = {l: v.to(ctx.device) for l, v in resid.items()}
        _, knows = record(model, [p["text"] + KNOWS_SUFFIX for p in probes], [last], last)
        rows = []
        for i, p in enumerate(probes):
            aw, ap, forms, bad = candidate_ids(tok, p.get("accepted") or [])
            nw, npc, _, nbad = candidate_ids(tok, p.get("near") or [])
            acc = sorted(set(aw) | set(ap))
            near = [t for t in sorted(set(nw) | set(npc)) if t not in set(acc)]
            probe_ids |= set(acc) | set(near)
            note = p.get("note") or ""
            if bad or nbad:
                note = (note + f" [no usable token form: {bad + nbad}]").strip()
            has_acc = bool(p.get("accepted"))
            rows.append(dict(id=p["id"], set="probes", task=p.get("category"),
                             difficulty=p.get("difficulty") if has_acc else None,  # controls: no difficulty
                             text=p["text"], concept=(p.get("accepted") or [None])[0], answer=None,
                             accepted=list(p.get("accepted") or []), accepted_tokens=[tok.decode([t]) for t in acc],
                             forms=forms, scorable=bool(acc), near=list(p.get("near") or []), note=note,
                             model=dict(knows=top10(tok, knows[i]), knows_rank=raw_rank(knows[i], acc)),
                             acc_ids=acc, piece_ids=list(ap), near_ids=near, src=i))
        # the model's own next token at the read position, under the same f4 filter (probe forms never masked):
        # is_next = a word group among its top 10 holds an accepted form (then "finding" the concept is next-word
        # prediction, not a hidden thought). None for controls / unscorable probes.
        exn = F.excluded_ids
        if probe_ids:
            exn = exn[~torch.isin(exn, torch.as_tensor(sorted(probe_ids), dtype=exn.dtype))]
        gl = F.group_ids.tolist()
        for r, lg in zip(rows, nxt):
            lgf = lg.clone()
            lgf[exn.to(lgf.device)] = float("-inf")
            top = FL.group_topk(lgf, F.group_ids.to(lgf.device), 10)
            ga = {gl[t] for t in r["acc_ids"]}
            r["model"]["next"] = [tok.decode([t]).replace("�", "▯") for t in top]
            r["model"]["next_raw"] = top10(tok, lg)
            r["is_next"] = any(gl[t] in ga for t in top) if r["acc_ids"] else None
        groups["probes"] = (rows, resid)
        log(f"probes: {len(rows)} recorded ({time.time() - t0:.0f}s)")

    # ---- the wrapped lens
    mod = load_candidate(inner)
    tb = time.time()
    lens = mod.build(ctx)
    build_s = time.time() - tb
    ex = F.excluded_ids
    if probe_ids:  # never mask a probe's accepted / near form
        ex = ex[~torch.isin(ex, torch.as_tensor(sorted(probe_ids), dtype=ex.dtype))]
    items, counts = [], {}
    for set_name in ("bench-fit", "bench-dev", "probes"):
        if set_name not in groups:
            continue
        rows, resid = groups[set_name]
        ents = entries(model, lens, rows, resid, layers, F, ex, FL, topk)
        for r, e in zip(rows, ents):
            bl, br, bn = best_of(layers, e)
            it = {k: v for k, v in r.items() if k not in ("acc_ids", "piece_ids", "near_ids", "src")}
            it.update(layers=e, best_layer=bl, best_rank=br, best_near_rank=bn)
            items.append(it)
        counts[set_name] = len(rows)
        log(f"{set_name}: {len(rows)} rows dumped ({time.time() - t0:.0f}s)")
    built = (f"{Path(inner).relative_to(R) if inner.startswith(str(R)) else inner} env {env_used(inner)}; "
             f"lens built under --filter {getattr(ctx.fit, 'filter_level', 'f1')}; build {build_s:.0f}s")
    if getattr(lens, "lam", None) is not None:
        built += f"; lam = {lens.lam}"
    if os.environ.get("QUAL_NOTE"):
        built += "; " + os.environ["QUAL_NOTE"]
    argv = sys.argv  # the evaluator's --model (its default is Qwen/Qwen3.5-9B)
    model_name = argv[argv.index("--model") + 1] if "--model" in argv else "Qwen/Qwen3.5-9B"
    meta = dict(lens=name, model=model_name,
                model_tag=tag, filter="f4", layers=layers, k_top=topk, built=built,
                date=datetime.date.today().isoformat(), source="auto/candidates/qual_dump.py", sets=counts,
                probes_file=probes_path, knows_prompt=KNOWS_SUFFIX)
    out = R / os.environ.get("QUAL_OUTDIR", f"runs/qual/{tag}") / f"{slug(name)}.json"  # ej1: QUAL_OUTDIR (e.g. runs/qual/9b_v2)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps({"format": "qual-dump", "version": 1, "meta": meta, "items": items}, ensure_ascii=False,
                   separators=(",", ":"))
    with open(out, "w", encoding="utf-8") as fh:  # in place (network fs)
        fh.write(s)
    log(f"wrote {out}: {len(items)} rows, {len(s)} bytes, {time.time() - t0:.0f}s (counted in build_seconds)")
    return lens
