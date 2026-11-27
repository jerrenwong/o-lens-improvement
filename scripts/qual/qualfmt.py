"""Shared helpers for qual dumps (stdlib only, Python 3.8). Imported by convert_fitA.py, compare.py, table.py.

Qual dump (one file per lens x model, runs/qual/<tag>/<slug>.json):
  {"format": "qual-dump", "version": 1,
   "meta": {lens, model, model_tag, filter, layers, k_top, built, date, source, sets: {set: n_items}},
   "items": [{id, set, task, difficulty, text, concept, answer, accepted, accepted_tokens, near, note, model,
              [probes only: forms [{word, token, kind: whole|piece}], scorable, is_next],
              layers: [{t: [top tokens], r: best accepted rank|null, nr: best near rank|null,
                        h: [indices of t that are accepted], hp: [subset of h matched only by a first token piece],
                        nh: [indices of t that are near]}, ...aligned with meta.layers],
              best_layer, best_rank, best_near_rank}]}
Ranks are 1-based, filter level meta.filter (f4 = word-group ranks). r is null when the item has no usable accepted form
(controls, unscorable probes). is_next: true/false = an accepted form is / is not in the model's own next-token top 10
at the read position; absent or null = unknown (benchmark items, converted dumps).
model: {next: model's next-token top 10, asked / knows: what the model says when asked, ...} (all optional).
"""
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict

SETS = ("bench-fit", "bench-dev", "probes")
FUNCTION_WORDS = set("""a an the of to in on at by for with from into onto over under about after before as and or but nor so
yet if then than that this these those there here is are was were be been being am do does did has have had it its he she they
we you i me him her them us his their our your my not no yes all any some each every which who whom whose what when where why
how can could will would shall should may might must just also very more most less up out off""".split())


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("format") != "qual-dump":
        raise SystemExit(f"{path}: not a qual dump (format={d.get('format')!r})")
    return d


def save(path, d):
    """Compact JSON, written in place (network fs: no temp-then-rename)."""
    s = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
    return len(s)


def slug(name):
    s = name.replace("++", "pp").replace("+", "_").lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "lens"


def best_of(layers_meta, entries):
    """(best_layer, best_rank, best_near_rank) over layers; ties -> earliest layer. None when no ranks."""
    br, bl, bn = None, None, None
    for L, e in zip(layers_meta, entries):
        r = e.get("r")
        if r is not None and (br is None or r < br):
            br, bl = r, L
        n = e.get("nr")
        if n is not None and (bn is None or n < bn):
            bn = n
    return bl, br, bn


def default_layer(meta):
    L = meta["layers"]
    return L[len(L) // 2]


def view(item, meta, layer="best"):
    """(layer, entry) read for one item: its best layer, or a fixed layer. Items without a rank (controls) use the
    middle layer when layer == "best"."""
    L = meta["layers"]
    if layer == "best":
        lay = item.get("best_layer")
        if lay is None:
            lay = default_layer(meta)
    else:
        lay = int(layer)
        if lay not in L:
            raise SystemExit(f"layer {lay} not in {L} ({meta['lens']})")
    return lay, item["layers"][L.index(lay)]


def norm_tok(t):
    t = t.strip()
    if t[:1] in ("_", "▁", "Ġ"):
        t = t[1:]
    return t.lower()


def normal_form(s):
    """auto/filters.py normal_form (the f3/f4 word-group key): strip, drop one leading marker, lower, drop one trailing
    "s" when at least 3 letters remain."""
    s = s.strip()
    if s[:1] in ("_", "\u2581", "\u0120"):
        s = s[1:]
    s = s.lower()
    if s.endswith("s") and sum(c.isalpha() for c in s[:-1]) >= 3:
        s = s[:-1]
    return s


def tok_classes(t):
    s = t.strip()
    letters = sum(c.isalnum() for c in s)
    return dict(nonascii=any(ord(c) > 127 for c in s), short=letters < 3,
                func=norm_tok(t) in FUNCTION_WORDS, cap=bool(s[:1]) and s[:1].isupper())


def char_stats(items, meta, layer="best", k=10, n_recurring=15):
    """Share of top-k slots per class, and the tokens (normalised) that recur across items. A prompt with several
    concepts (ids "<item>#<concept>") counts once per layer read. "accepted" marks are those of the first concept."""
    c = Counter()
    rec = Counter()
    n_items = 0
    done = set()
    for it in items:
        lay, e = view(it, meta, layer)
        base = (it["id"].split("#")[0], lay)  # several concepts of one prompt read at one layer: count once
        if base in done:
            continue
        done.add(base)
        top = e["t"][:k]
        h, nh = set(e.get("h", [])), set(e.get("nh", []))
        n_items += 1
        seen = set()
        for i, t in enumerate(top):
            c["slots"] += 1
            c["accepted"] += i in h
            c["near"] += (i in nh) and (i not in h)
            for key, v in tok_classes(t).items():
                c[key] += v
            w = norm_tok(t)
            if w not in seen:
                seen.add(w)
                rec[w] += 1
    slots = max(c["slots"], 1)
    shares = {key: c[key] / slots for key in ("accepted", "near", "nonascii", "short", "func", "cap")}
    return dict(n_items=n_items, slots=c["slots"], shares=shares, recurring=rec.most_common(n_recurring))


def hit(rank, k):
    return rank is not None and rank <= k


def fmt_rank(r):
    return "-" if r is None else str(r)


def log_change(ra, rb):
    """log10(rank A / rank B): > 0 = B ranks the concept better."""
    if ra is None or rb is None:
        return 0.0
    return math.log10(ra) - math.log10(rb)


def mark_top(e, k=10):
    """Top-k tokens as display strings: accepted [tok], accepted by a first token piece only [tok~], near (tok)."""
    h, hp, nh = set(e.get("h", [])), set(e.get("hp", [])), set(e.get("nh", []))
    out = []
    for i, t in enumerate(e["t"][:k]):
        s = t.strip() or repr(t)
        s = "".join(ch if unicodedata.category(ch)[0] != "C" else "?" for ch in s)
        out.append(f"[{s}~]" if i in hp else f"[{s}]" if i in h else (f"({s})" if i in nh else s))
    return out


def scorable(it):
    """Counts toward hits: has accepted forms, not marked unscorable."""
    return bool(it.get("accepted")) and it.get("scorable", True) is not False


def next_label(it):
    v = it.get("is_next")
    return None if v is None else ("next word" if v else "not next word")


def select(d, which="all"):
    items = d["items"]
    if which == "all":
        return list(items)
    return [it for it in items if it["set"] == which]


def groups(items, key):
    g = defaultdict(list)
    for it in items:
        g[it.get(key) or "-"].append(it)
    return dict(sorted(g.items()))
