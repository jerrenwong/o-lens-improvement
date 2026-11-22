"""ex3 report (hub, CPU): reads runs/ex3/raw_<tag>.json, writes runs/ex3/summary_<tag>.json and runs/ex3/report.md.

    python3 scripts/ex3/ex3_report.py 9b 27b [2b]

Rules (fixed before looking at the counts):
  missed by all  : min over 7 layers and the 3 readers (J++, L, ask-map alone; filtered) of the concept rank > 10.
  echo positions : prompt tokens that overlap a literal occurrence of the concept word (word start, case-insensitive)
                   or are one of its candidate ids, plus the token right before each (which predicts it).
  elsewhere min  : min over positions != readout token and not echo, best over the 7 layers.
  null           : the same statistic for (up to) 20 other concepts of the same task (their own echo positions
                   removed); null rate = share of null concepts with elsewhere-min <= 10.
  M6 classes (first that applies):
    1 elsewhere   : J++ or ask-map-alone elsewhere-min <= 10 and at most 10% of that lens's null concepts reach a
                    min as good (<= the concept's), computed separately over earlier tokens (before the readout:
                    class elsewhere_before, checked first) and later tokens (after it; only poetry prompts have
                    any: the given start of line 2; class elsewhere_after). "next" = the model's own next-token
                    rank of the concept at that position (unfiltered).
    2 continues   : model produces the concept unasked: it appears in the greedy continuation or in >= 1 of the 8
                    samples (30 tokens, T 0.8), or the model's own next-token rank at the readout token <= 10
                    (unfiltered). Poetry: continuation = the model's own line 2 generated from the readout token,
                    and the label must be its line-end word (greedy or >= 1 sample).
    3 asked_only  : not 2; the model names it when asked: greedy answer contains the concept or first-answer-token
                    rank <= 10 in either wording (topic / unstated).
    4 never       : not 1-3 and every one of: asked rank > 50 both wordings, next-token rank > 50 (label mismatch).
    5 other       : the rest (weak evidence: some rank in 11..50).
"""
import json
import re
import sys
from collections import Counter, defaultdict

R = "/workspace/o-lens-improvement"
TASKS = ["association", "poetry", "multilingual", "multihop", "typo"]


def echo_positions(rec, word, cand):
    toks, ids = rec["toks"], rec["ids"]
    starts, s = [], 0
    for x in toks:
        starts.append(s)
        s += len(x)
    txt = "".join(toks)
    bad = set()
    w = word.split(":", 1)[1] if ":" in word else word
    for m in re.finditer(r"(?<![^\W\d_])" + re.escape(w.lower()), txt.lower()):
        a, b = m.span()
        for p, st in enumerate(starts):
            if st < b and st + len(toks[p]) > a:
                bad.add(p)
    cs = set(cand)
    bad |= {p for p, i in enumerate(ids) if i in cs}
    bad |= {p - 1 for p in list(bad) if p > 0}
    return bad


def scan(rec, lens, key, region="all"):
    B = rec["m2"][lens]["best"][key]
    Ly = rec["m2"][lens]["layer"][key]
    NX = rec.get("m2_next", {}).get(key)
    ex = echo_positions(rec, key, rec["m2_cand"][key]) | {rec["rp"]}
    rp = rec["rp"]
    pos = [p for p in range(rec["S"]) if p not in ex and (region == "all" or (region == "before") == (p < rp))]
    if not pos:
        return dict(min=None)
    p = min(pos, key=lambda q: B[q])
    return dict(min=B[p], off=p - rp, off_end=p - (rec["S"] - 1), tok=rec["toks"][p], layer=Ly[p],
                n10=sum(B[q] <= 10 for q in pos), n_pos=len(pos), rp_best=B[rp], echo=len(ex) - 1,
                lit=any(q != rp for q in ex), next_rank_there=NX[p] if NX else None)


def null_stats(rec, lens, cmin, region="all"):
    mins = [scan(rec, lens, f"null:{c}", region)["min"] for c in rec["m2_nulls"]]
    mins = [m for m in mins if m is not None]
    return dict(n=len(mins), rate10=sum(m <= 10 for m in mins) / max(1, len(mins)),
                p_as_good=(sum(m <= cmin for m in mins) / max(1, len(mins))) if cmin is not None else None,
                median=sorted(mins)[len(mins) // 2] if mins else None)


def null_pass_rate(rec, region):
    """share of this item's null concepts that would pass the M6 'elsewhere' rule themselves (J++ or alone:
    min <= 10 and at most 10% of the OTHER null concepts reach a min as good) = false-positive rate of the rule."""
    mins = {l: {c: scan(rec, l, f"null:{c}", region)["min"] for c in rec["m2_nulls"]} for l in ("jpp", "alone")}
    npass = 0
    for c in rec["m2_nulls"]:
        ok = False
        for l in ("jpp", "alone"):
            m = mins[l][c]
            oth = [v for k, v in mins[l].items() if k != c and v is not None]
            if m is not None and m <= 10 and oth and sum(v <= m for v in oth) / len(oth) <= 0.10:
                ok = True
        npass += ok
    return npass / max(1, len(rec["m2_nulls"]))


def contains(ans, c):
    return re.search(r"(?<![^\W\d_])" + re.escape(c.lower()), ans.lower()) is not None


def analyse(tag):
    D = json.load(open(f"{R}/runs/ex3/raw_{tag}.json"))
    rows = []
    for rec in D["items"]:
        for j, c in enumerate(rec["concepts"]):
            m1 = {nm: min(rec["m1"][nm]["rank"][c]) for nm in ("jpp", "L", "alone")}
            best_layer = {nm: D["layers"][rec["m1"][nm]["rank"][c].index(m1[nm])] for nm in m1}
            top5 = {nm: rec["m1"][nm]["top10"][rec["m1"][nm]["rank"][c].index(m1[nm])][:5] for nm in m1}
            r = dict(task=rec["eval"], item=rec["name"], concept=c, m1=m1, best_layer=best_layer, top5=top5,
                     missed=min(m1.values()) > 10, prompt=rec["prompt"], rp_tok=rec["toks"][rec["rp"]])
            for lens in ("jpp", "alone"):
                s = scan(rec, lens, f"own:{c}")
                s["null"] = null_stats(rec, lens, s["min"])
                r[f"m2_{lens}"] = s
                for reg in ("before", "after"):
                    s = scan(rec, lens, f"own:{c}", reg)
                    s["null"] = null_stats(rec, lens, s["min"], reg) if s["min"] is not None else None
                    r[f"m2_{lens}_{reg}"] = s
            a = rec["asked"]
            r["asked"] = {k: dict(answer=a[k]["answer"].split("<|im_end|>")[0].strip(), rank=a[k]["first_rank_u"][j],
                                  says=contains(a[k]["answer"].split("<|im_end|>")[0], c)) for k in a}
            ct = rec["cont"]
            r["cont"] = dict(greedy=ct["appear_greedy"][c], n_samples=sum(x is not None for x in ct["appear_samples"][c]),
                             greedy_text=ct["greedy"])
            r["next_rank"] = rec["next"]["rp_rank_u"][j]
            r["next_rank_end"] = rec["next"]["end_rank_u"][j]
            # M5: J++ rank / cosine at the asked answer position vs the readout token, per layer
            r["m5"] = {k: dict(asked_rank=a[k]["jpp_rank"][a[k]["keys"].index(c)],
                               readout_rank=rec["m1"]["jpp"]["rank"][c],
                               cos_asked=a[k]["cos"][c]["asked"], cos_readout=a[k]["cos"][c]["readout"]) for k in a}
            if rec["eval"] == "poetry":
                # unasked production for poetry = the model's OWN line 2 (from the readout token) ends in the label;
                # the full-prompt completion is forced by the given line-2 start (kept as cont_given)
                r["cont_given"] = r["cont"]
                ol = rec["own_line"]
                r["cont"] = dict(greedy=0 if (ol["end_greedy"] or "") == c.lower() else None,
                                 n_samples=sum((w or "") == c.lower() for w in ol["end_samples"]),
                                 greedy_text=ol["greedy"])
                rh = rec["rhyme"]
                r["rhyme"] = dict(greedy=rh["greedy"], top_sample=rh["top_sample"], counts=rh["sample_counts"],
                                  label_is_greedy=(rh["greedy"] or "") == c.lower(),
                                  label_in_samples=c.lower() in rh["sample_counts"],
                                  line=rec["own_line"]["line_greedy"],
                                  completion=ct["end_greedy"], completion_samples=ct["end_samples"],
                                  label_is_completion=(ct["end_greedy"] or "") == c.lower())
                for tw in ("actual_greedy", "actual_topsample"):
                    w = rh["greedy"] if tw == "actual_greedy" else rh["top_sample"]
                    key = f"{tw}:{w}"
                    if w and key in rec["m1"]["jpp"]["rank"]:
                        r["rhyme"][tw] = dict(word=w, m1={nm: min(rec["m1"][nm]["rank"][key]) for nm in ("jpp", "L", "alone")},
                                              m2_jpp=scan(rec, "jpp", key), m2_alone=scan(rec, "alone", key),
                                              m2_jpp_before=scan(rec, "jpp", key, "before"),
                                              m2_jpp_after=scan(rec, "jpp", key, "after"))
                    else:
                        r["rhyme"][tw] = dict(word=w, m1=None)
            r["null_pass_before"] = null_pass_rate(rec, "before")
            r["null_pass_after"] = null_pass_rate(rec, "after") if rec["S"] - 1 > rec["rp"] else 0.0
            r["cls"], r["flags"] = classify(r)
            rows.append(r)
    return D, rows


def classify(r):
    f = {}

    def ok(s):
        return s["min"] is not None and s["min"] <= 10 and s["null"]["p_as_good"] <= 0.10
    f["elsewhere_before"] = any(ok(r[f"m2_{l}_before"]) for l in ("jpp", "alone"))
    f["elsewhere_after"] = any(ok(r[f"m2_{l}_after"]) for l in ("jpp", "alone"))
    f["elsewhere"] = f["elsewhere_before"] or f["elsewhere_after"]
    f["continues"] = r["cont"]["greedy"] is not None or r["cont"]["n_samples"] > 0 or r["next_rank"] <= 10
    f["asked"] = any(v["says"] or v["rank"] <= 10 for v in r["asked"].values())
    weak = any(v["rank"] <= 50 for v in r["asked"].values()) or r["next_rank"] <= 50
    if f["elsewhere_before"]:
        c = "elsewhere_before"
    elif f["elsewhere_after"]:
        c = "elsewhere_after"
    elif f["continues"]:
        c = "continues"
    elif f["asked"]:
        c = "asked_only"
    elif not weak:
        c = "never"
    else:
        c = "other"
    return c, f


def fmt_rank(x):
    return "-" if x is None else str(x)


def report(tag, D, rows):
    L = []
    miss = [r for r in rows if r["missed"]]
    L.append(f"## {D['model']} (lam_L {D['lam_L']}, resid check min cos {D['resid_min_cos']:.5f})\n")
    L.append("### Readout-token coverage (M1) and M6 classes of missed-by-all concepts\n")
    L.append("| task | concepts | missed by all | elsewhere, earlier token | elsewhere, later token | continues (lens misses) | asked only | never (label mismatch) | other |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    tot = Counter()
    for t in TASKS:
        rs = [r for r in rows if r["task"] == t]
        ms = [r for r in rs if r["missed"]]
        c = Counter(r["cls"] for r in ms)
        tot.update(c)
        tot["n"] += len(rs)
        tot["m"] += len(ms)
        L.append(f"| {t} | {len(rs)} | {len(ms)} | {c['elsewhere_before']} | {c['elsewhere_after']} | {c['continues']} | {c['asked_only']} | {c['never']} | {c['other']} |")
    L.append(f"| all | {tot['n']} | {tot['m']} | {tot['elsewhere_before']} | {tot['elsewhere_after']} | {tot['continues']} | {tot['asked_only']} | {tot['never']} | {tot['other']} |\n")
    # flag overlap
    fl = Counter(tuple(k for k, v in r["flags"].items() if v and k != "elsewhere") for r in miss)
    L.append("Flag combinations among missed (not exclusive): " + "; ".join(f"{'+'.join(k) or 'none'} {v}" for k, v in fl.most_common()) + "\n")
    L.append("False-positive check of the 'elsewhere' rule: expected number of missed concepts that pass it by chance "
             "(sum over missed concepts of the share of their null concepts that pass the same rule) vs observed:")
    for t in TASKS:
        ms = [r for r in miss if r["task"] == t]
        if ms:
            L.append(f"- {t}: earlier tokens expected {sum(r['null_pass_before'] for r in ms):.1f} vs observed "
                     f"{sum(r['flags']['elsewhere_before'] for r in ms)}; later tokens expected "
                     f"{sum(r['null_pass_after'] for r in ms):.1f} vs observed {sum(r['flags']['elsewhere_after'] for r in ms)} "
                     f"(of {len(ms)})")
    L.append("")
    for cl in ("elsewhere_before", "elsewhere_after"):
        rs = [r for r in miss if r["cls"] == cl]
        reg = cl.split("_")[1]
        nx = [r for r in rs if any((r[f"m2_{l}_{reg}"].get("next_rank_there") or 10 ** 9) <= 10
                                   and (r[f"m2_{l}_{reg}"]["min"] or 10 ** 9) <= 10 for l in ("jpp", "alone"))]
        if rs:
            L.append(f"{cl}: {len(rs)} concepts; at the hit position the model's own next-token rank of the concept is <= 10 "
                     f"(the lens shows the model's next-token prediction there) for {len(nx)}; by task "
                     + ", ".join(f"{t} {sum(r['task'] == t for r in rs)}" for t in TASKS if any(r['task'] == t for r in rs)))
    L.append("")
    # null summary
    for lens in ("jpp", "alone"):
        nr = [r[f"m2_{lens}"]["null"]["rate10"] for r in miss]
        own = [r[f"m2_{lens}"]["min"] is not None and r[f"m2_{lens}"]["min"] <= 10 for r in miss]
        L.append(f"M2 {lens}: missed concepts with some other position <= 10: {sum(own)}/{len(miss)}; "
                 f"mean null rate (share of random same-task concepts with some position <= 10): {sum(nr) / max(1, len(nr)):.2f}")
    allr = [r for r in rows if not r["missed"]]
    L.append("")
    for t in ("association", "poetry"):
        ms = [r for r in miss if r["task"] == t]
        L.append(f"### {t}: missed-by-all items ({len(ms)})\n")
        hdr = "| item | concept | J++ / L / alone top-5 (best layer) | min rank elsewhere J++ (offset, token) / alone | null rate J++ | topic ans (rank) | unstated ans (rank) | in continuation | next rank | class |"
        if t == "poetry":
            hdr = hdr[:-9] + "| model rhyme (greedy; samples) | J++ rank of actual (greedy / top sample) | class |"
        L.append(hdr)
        L.append("|" + "---|" * (hdr.count("|") - 1))
        for r in ms:
            tops = " / ".join(f"{','.join(x.strip() for x in r['top5'][nm])} (L{r['best_layer'][nm]})" for nm in ("jpp", "L", "alone"))
            mj, ma = r["m2_jpp_before"], r["m2_alone_before"]
            el = (f"{fmt_rank(mj['min'])} ({mj.get('off')}, {mj.get('tok', '')!r}, next {mj.get('next_rank_there')}) / "
                  f"{fmt_rank(ma['min'])}")
            if t == "poetry":
                mja = r["m2_jpp_after"]
                el += f"; after: {fmt_rank(mja['min'])} ({mja.get('off')}, {mja.get('tok', '')!r}, next {mja.get('next_rank_there')})"
            a = r["asked"]
            cont = ("greedy" if r["cont"]["greedy"] is not None else "") + (f" {r['cont']['n_samples']}/8" if r["cont"]["n_samples"] else "")
            row = (f"| {r['item']} | {r['concept']} | {tops} | {el} | {mj['null']['rate10']:.2f} | "
                   f"{a['topic']['answer']!r} ({a['topic']['rank']}) | {a['unstated']['answer']!r} ({a['unstated']['rank']}) | "
                   f"{cont or 'no'} | {r['next_rank']} |")
            if t == "poetry":
                rh = r["rhyme"]
                sm = ",".join(f"{w}x{n}" for w, n in sorted(rh["counts"].items(), key=lambda x: -x[1]))
                ag = rh["actual_greedy"].get("m1") or {}
                at = rh["actual_topsample"].get("m1") or {}
                row += f" {rh['greedy']}; {sm} | {ag.get('jpp', '-')} / {at.get('jpp', '-')} |"
            row += f" {r['cls']} |"
            L.append(row)
        L.append("")
    # poetry re-score
    po = [r for r in rows if r["task"] == "poetry"]
    if po:
        def hit(r, k):
            m = r["rhyme"][k].get("m1")
            return m is not None and m["jpp"] <= 10
        n = len(po)
        L.append("### Poetry re-score (all fit poetry items, J++ best over 7 layers at the readout token)\n")
        L.append(f"- model's OWN second line (continued from the readout token): label = greedy rhyme {sum(r['rhyme']['label_is_greedy'] for r in po)}/{n}; label among the 8 samples' line-end words {sum(r['rhyme']['label_in_samples'] for r in po)}/{n}")
        L.append(f"- completing the GIVEN second line (full prompt): label = greedy completion {sum(r['rhyme']['label_is_completion'] for r in po)}/{n}")
        L.append(f"- J++ top-10: label {sum(r['m1']['jpp'] <= 10 for r in po)}/{n}; model's actual greedy rhyme {sum(hit(r, 'actual_greedy') for r in po)}/{n}; "
                 f"most frequent sampled rhyme {sum(hit(r, 'actual_topsample') for r in po)}/{n} "
                 f"(actual word not single-token: {sum(r['rhyme']['actual_greedy'].get('m1') is None for r in po)})")
        L.append(f"- any reader top-10: label {sum(min(r['m1'].values()) <= 10 for r in po)}/{n}; actual greedy "
                 f"{sum((r['rhyme']['actual_greedy'].get('m1') or {'x': 99}) and min((r['rhyme']['actual_greedy'].get('m1') or {'x': 99}).values()) <= 10 for r in po)}/{n}")
        el = [r for r in po if r["rhyme"]["actual_greedy"].get("m2_jpp")]
        L.append(f"- position scan J++ <= 10, line 1 (before readout): label {sum((r['m2_jpp_before']['min'] or 99) <= 10 for r in po)}/{n}; "
                 f"actual greedy {sum((r['rhyme']['actual_greedy']['m2_jpp_before']['min'] or 99) <= 10 for r in el)}/{len(el)}; "
                 f"given line 2 (after readout): label {sum((r['m2_jpp_after']['min'] or 99) <= 10 for r in po)}/{n}; "
                 f"actual greedy {sum((r['rhyme']['actual_greedy']['m2_jpp_after']['min'] or 99) <= 10 for r in el)}/{len(el)}")
        L.append("")
        L.append("| item | label | own greedy line 2 | J++ rank label @readout | J++ rank actual @readout | min line-1 label (off, tok) | min line-1 actual (off, tok) |")
        L.append("|---|---|---|---|---|---|---|")
        for r in po:
            ag = r["rhyme"]["actual_greedy"]
            m2a = ag.get("m2_jpp_before") or {}
            L.append(f"| {r['item']} | {r['concept']} | {r['rhyme']['greedy']} ({r['rhyme']['line']!r}) | {r['m1']['jpp']} | "
                     f"{(ag.get('m1') or {}).get('jpp', '-')} | {r['m2_jpp_before']['min']} ({r['m2_jpp_before'].get('off')}, {r['m2_jpp_before'].get('tok', '')!r}) | "
                     f"{m2a.get('min', '-')} ({m2a.get('off')}, {m2a.get('tok', '')!r}) |")
        L.append("")
    # M5 association
    asc = [r for r in rows if r["task"] == "association"]
    if asc:
        L.append("### M5 association: J++ at the asked answer position vs the readout token (best over layers)\n")
        for grp, rs in (("missed", [r for r in asc if r["missed"]]), ("found", [r for r in asc if not r["missed"]])):
            for k in ("topic", "unstated"):
                ar = [min(r["m5"][k]["asked_rank"]) for r in rs]
                rr = [min(r["m5"][k]["readout_rank"]) for r in rs]
                ca = [max(r["m5"][k]["cos_asked"]) for r in rs]
                cr = [max(r["m5"][k]["cos_readout"]) for r in rs]
                if rs:
                    med = lambda x: sorted(x)[len(x) // 2]  # noqa: E731
                    L.append(f"- {grp} ({len(rs)}), {k}: J++ top-10 asked {sum(x <= 10 for x in ar)}/{len(rs)} vs readout "
                             f"{sum(x <= 10 for x in rr)}/{len(rs)}; median best rank asked {med(ar)} vs readout {med(rr)}; "
                             f"median max-cos asked {med(ca):.3f} vs readout {med(cr):.3f}")
        L.append("")
        L.append("Per missed association item, J++ best rank at asked(unstated) position / readout, per layer:")
        for r in [r for r in asc if r["missed"]]:
            m = r["m5"]["unstated"]
            L.append(f"- {r['item']} ({r['concept']}): asked {m['asked_rank']} | readout {m['readout_rank']}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    out = []
    for tag in sys.argv[1:]:
        D, rows = analyse(tag)
        json.dump(dict(model=D["model"], rows=rows), open(f"{R}/runs/ex3/summary_{tag}.json", "w"), ensure_ascii=False, indent=1)
        out.append(report(tag, D, rows))
    open(f"{R}/runs/ex3/report.md", "w").write("# ex3: where are the concepts no reader finds? (FIT split)\n\n"
                                               + __doc__.split("Rules", 1)[1].join(["Rules", ""]) + "\n\n" + "\n\n".join(out))
    print("\n\n".join(out))
