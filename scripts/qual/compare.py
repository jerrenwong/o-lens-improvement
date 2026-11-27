"""Compare two lenses item by item on the same items (qual dumps from convert_fitA.py or auto/candidates/qual_dump.py).

    python3 scripts/qual/compare.py A.json B.json [--set probes|bench-fit|bench-dev|all] [--layer best|<n>] [--k 10]
                                    [--next any|yes|no] [--md] [--full] [--out <site>/data/compare]
  e.g. python3 scripts/qual/compare.py runs/qual/9b/jpp.json runs/qual/9b/jpp_a_lens.json --md

B is the new lens, A the old. Hit = best accepted rank <= k at the chosen layer (best = each lens's own best layer).
IMPROVED = A miss -> B hit, WORSENED = A hit -> B miss; lists sorted by rank change (log10 ratio).
Default output: full text report (all items). --md: digest for a model (<= ~120 lines). --full with --md: no item cap.
--out DIR: copy both dumps into the page's data dir (DIR/<tag>/<slug>.json) and register them in DIR/index.json.
Controls and unscorable probes never count; tables split by "next word" / "not next word" when is_next is known.
Marks: [x] accepted, [x~] accepted via a first token piece, (x) near. --next yes|no keeps only (not) next-word items.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qualfmt as Q  # noqa: E402

STAT_KEYS = (("accepted", "acc"), ("near", "near"), ("nonascii", "non-ASCII"), ("short", "<3 chars"),
             ("func", "function w."), ("cap", "Capital"))


def pair(A, B, which, layer, k):
    """Rows for items in both dumps (same id), in A's order."""
    bi = {it["id"]: it for it in Q.select(B, which)}
    rows = []
    for a in Q.select(A, which):
        b = bi.get(a["id"])
        if b is None:
            continue
        la, ea = Q.view(a, A["meta"], layer)
        lb, eb = Q.view(b, B["meta"], layer)
        ok = Q.scorable(a) and ea.get("r") is not None and eb.get("r") is not None
        ra, rb = ea.get("r"), eb.get("r")
        ha, hb = Q.hit(ra, k), Q.hit(rb, k)
        cls = ("control" if not a.get("accepted") else "unscorable" if not ok else "improved" if hb and not ha
               else "worsened" if ha and not hb else "both-hit" if ha else "both-miss")
        rows.append(dict(a=a, b=b, la=la, lb=lb, ea=ea, eb=eb, ra=ra, rb=rb, cls=cls, d=Q.log_change(ra, rb)))
    return rows


def counts(rows):
    c = dict(n=0, hitA=0, hitB=0, improved=0, worsened=0, **{"both-hit": 0, "both-miss": 0}, control=0, unscorable=0)
    for r in rows:
        if r["cls"] in ("control", "unscorable"):
            c[r["cls"]] += 1
            continue
        c["n"] += 1
        c["hitA"] += r["cls"] in ("both-hit", "worsened")
        c["hitB"] += r["cls"] in ("both-hit", "improved")
        c[r["cls"]] += 1
    return c


def by(rows, key):
    out = {}
    for r in rows:
        v = (Q.next_label(r["a"]) or "unknown") if key == "is_next" else (r["a"].get(key) or "-")
        out.setdefault(v, []).append(r)
    return {k: counts(v) for k, v in sorted(out.items())}


def ordered(rows, cls):
    sel = [r for r in rows if r["cls"] == cls]
    rev = cls in ("improved", "both-hit", "both-miss")
    return sorted(sel, key=lambda r: r["d"], reverse=rev) if cls != "worsened" else sorted(sel, key=lambda r: r["d"])


def short(s, n):
    s = " ".join(s.split())
    return s if len(s) <= n else "..." + s[-(n - 3):]


def item_lines(r, k, nA, nB, text_len=90):
    a = r["a"]
    acc = "/".join(a.get("accepted") or []) or "-"
    nl = Q.next_label(a)
    head = (f"- {a['id']} | concept {a.get('concept') or acc} | {nA} {Q.fmt_rank(r['ra'])}@L{r['la']} -> "
            f"{nB} {Q.fmt_rank(r['rb'])}@L{r['lb']}" + (f" | {nl}" if nl else ""))
    return [head, f"    text: {short(a.get('text', ''), text_len)}",
            f"    A: {' '.join(Q.mark_top(r['ea'], k))}", f"    B: {' '.join(Q.mark_top(r['eb'], k))}"]


def table_lines(title, tab):
    out = [f"| {title} | n | hits A | hits B | improved | worsened |", "|---|---|---|---|---|---|"]
    for key, c in tab.items():
        if c["n"]:
            out.append(f"| {key} | {c['n']} | {c['hitA']} | {c['hitB']} | {c['improved']} | {c['worsened']} |")
    return out


def stats_lines(A, B, rows, layer, k, n_rec=15):
    sa = Q.char_stats([r["a"] for r in rows], A["meta"], layer, k, n_rec)
    sb = Q.char_stats([r["b"] for r in rows], B["meta"], layer, k, n_rec)
    out = [f"| top-{k} slots | " + " | ".join(lbl for _, lbl in STAT_KEYS) + " |",
           "|---|" + "---|" * len(STAT_KEYS)]
    for name, s in (("A", sa), ("B", sb)):
        out.append(f"| {name} | " + " | ".join(f"{100 * s['shares'][key]:.0f}%" for key, _ in STAT_KEYS) + " |")
    for name, s in (("A", sa), ("B", sb)):
        n = max(s["n_items"], 1)
        out.append(f"recurring {name} (prompts holding it, of {n}): " + ", ".join(f"{w or repr(w)} {c}" for w, c in s["recurring"]))
    return out, sa, sb


def split_tables(rows):
    out = []
    for key, title in (("difficulty", "difficulty"), ("is_next", "next word?")):
        tab = by(rows, key)
        if any(v not in ("-", "unknown") and c["n"] for v, c in tab.items()):
            out += [""] + table_lines(title, tab)
    return out


def header(A, B, which, layer, k, c):
    nA, nB = A["meta"]["lens"], B["meta"]["lens"]
    return [f"# {nB} (B) vs {nA} (A) | {A['meta']['model']} | set {which} | layer {layer} | k {k} | filter "
            f"{A['meta']['filter']}",
            f"items {c['n']} (+{c['control']} controls, {c['unscorable']} unscorable) | hits A {c['hitA']} | hits B {c['hitB']} | improved "
            f"{c['improved']} | worsened {c['worsened']} | both hit {c['both-hit']} | both miss {c['both-miss']}"]


def report_md(A, B, rows, which, layer, k, full=False, cap=120):
    c = counts(rows)
    out = header(A, B, which, layer, k, c)
    tasks = by(rows, "task")
    out += [""] + table_lines("task", tasks)
    out += split_tables(rows)
    st, _, _ = stats_lines(A, B, rows, layer, k)
    imp, wor = ordered(rows, "improved"), ordered(rows, "worsened")
    if full:
        ni, nw = len(imp), len(wor)
    else:  # fill the line budget alternately (4 lines per item), at most 15 per list
        slots = max(2, (cap - len(out) - len(st) - 6) // 4)
        ni = nw = 0
        while slots > 0 and (ni < min(15, len(imp)) or nw < min(15, len(wor))):
            if ni < min(15, len(imp)):
                ni += 1; slots -= 1
            if slots > 0 and nw < min(15, len(wor)):
                nw += 1; slots -= 1
    out += ["", f"## IMPROVED ({len(imp)}; showing {ni}; [x] accepted, [x~] accepted piece, (x) near)"]
    for r in imp[:ni]:
        out += item_lines(r, k, "A", "B")
    out += ["", f"## WORSENED ({len(wor)}; showing {nw})"]
    for r in wor[:nw]:
        out += item_lines(r, k, "A", "B")
    out += ["", "## Top-k token character"] + st
    return out


def report_full(A, B, rows, which, layer, k):
    c = counts(rows)
    out = header(A, B, which, layer, k, c)
    out += [""] + table_lines("task", by(rows, "task"))
    out += split_tables(rows)
    for cls in ("improved", "worsened", "both-hit", "both-miss", "unscorable", "control"):
        sel = ordered(rows, cls)
        out += ["", f"## {cls.upper()} ({len(sel)})"]
        for r in sel:
            out += item_lines(r, k, "A", "B", text_len=200)
    st, _, _ = stats_lines(A, B, rows, layer, k)
    out += ["", "## Top-k token character"] + st
    return out


def publish(paths, out_dir):
    """Copy dumps to out_dir/<tag>/<slug>.json and register them in out_dir/index.json."""
    import json
    idx_p = os.path.join(out_dir, "index.json")
    idx = json.load(open(idx_p, encoding="utf-8")) if os.path.exists(idx_p) else {"models": {}}
    for p in paths:
        d = Q.load(p)
        m = d["meta"]
        tag = m.get("model_tag") or Q.slug(m["model"])
        rel = f"{tag}/{Q.slug(m['lens'])}.json"
        os.makedirs(os.path.join(out_dir, tag), exist_ok=True)
        Q.save(os.path.join(out_dir, rel), d)
        ent = idx["models"].setdefault(tag, {"model": m["model"], "lenses": []})
        ent["lenses"] = [x for x in ent["lenses"] if x["file"] != rel]
        ent["lenses"].append(dict(lens=m["lens"], file=rel, sets=m.get("sets", {}), k_top=m.get("k_top"),
                                  layers=m["layers"], date=m.get("date"), built=m.get("built", "")))
        ent["lenses"].sort(key=lambda x: x["lens"])
        print(f"published {rel}")
    s = json.dumps(idx, ensure_ascii=False, indent=1)
    with open(idx_p, "w", encoding="utf-8") as f:
        f.write(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("A")
    ap.add_argument("B")
    ap.add_argument("--set", default="all", choices=["all", "probes", "bench-fit", "bench-dev"])
    ap.add_argument("--layer", default="best")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--next", default="any", choices=["any", "yes", "no"])
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    A, B = Q.load(a.A), Q.load(a.B)
    if A["meta"]["model"] != B["meta"]["model"]:
        sys.exit(f"different models: {A['meta']['model']} vs {B['meta']['model']}")
    if A["meta"]["filter"] != B["meta"]["filter"]:
        print(f"WARNING: filter {A['meta']['filter']} vs {B['meta']['filter']}", file=sys.stderr)
    kmax = min(A["meta"].get("k_top", 10), B["meta"].get("k_top", 10))
    if a.k > kmax:
        print(f"note: top lists hold {kmax} tokens; display capped at {kmax} (ranks still use k={a.k})", file=sys.stderr)
    rows = pair(A, B, a.set, a.layer, a.k)
    if a.next != "any":
        want = a.next == "yes"
        rows = [r for r in rows if r["a"].get("is_next") is not None and bool(r["a"]["is_next"]) == want]
    if not rows:
        sys.exit("no shared items")
    lines = report_md(A, B, rows, a.set, a.layer, a.k, full=a.full) if a.md else report_full(A, B, rows, a.set, a.layer, a.k)
    print("\n".join(lines))
    if a.out:
        publish([a.A, a.B], a.out)


if __name__ == "__main__":
    main()
