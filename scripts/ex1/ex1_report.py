"""ex1 post-processing (CPU, plain python): runs/ex1/<tag>/{raw,clusters}_<tag>.json -> report_<tag>.md + summary_<tag>.json

    python3 scripts/ex1/ex1_report.py 9b
"""
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict

R = "/workspace/o-lens-improvement"
tag = sys.argv[1]
raw = json.load(open(f"{R}/runs/ex1/{tag}/raw_{tag}.json"))
cl = json.load(open(f"{R}/runs/ex1/{tag}/clusters_{tag}.json"))
V = {int(k): v for k, v in raw["vocab"].items()}
CV = {int(k): v for k, v in cl["vocab"].items()}
EXCL = set(raw["excluded_in_vocab"])
items = raw["items"]
LAYERS = raw["layers"]
out = []
P = out.append

FUNC = set("""a an the and or but nor so yet for of in on at by to from with without about above below into onto upon
over under between among through during before after since until till as than that this these those there here
which who whom whose what when where why how whether if then else is are was were be been being am do does did
done have has had having will would shall should can could may might must not no yes it its it's he him his she
her hers they them their theirs we us our ours you your yours i me my mine one ones some any all each every both
either neither few many much more most other another such own same also only just very too even still again ever
never always often up down out off s t ll re ve d m o y also like via per while because although though unless
la le les de des du et un une der die das und ein eine el los las y en del al o il di che""".split())


def norm(x):
    return x.strip().lower()


def kind(sx, i=None):
    """punct / number / function / content-start / continuation (Qwen pieces: leading space = word start)."""
    core = sx.strip()
    if not core or not any(ch.isalnum() for ch in core):
        return "punct"
    if any(ch.isdigit() for ch in core) and not any(ch.isalpha() for ch in core):
        return "number"
    if norm(sx) in FUNC:
        return "function"
    if sx[:1].isspace():
        return "content_start"
    return "continuation"


def is_wordstart(sx):
    return sx[:1].isspace()


def tokstr(i):
    x = V.get(i, f"<{i}>")
    return x.replace("\n", "\\n").replace(" ", "\u2423")


# ---------------- part 1b ----------------
allc = Counter({int(k): v for k, v in cl["all_counts_ids"].items()})
N = cl["n_positions"]
glob_kind = Counter()
for i, c in allc.items():
    glob_kind[kind(CV[i])] += c
glob_ws = sum(c for i, c in allc.items() if is_wordstart(CV[i])) / N
summary = dict(tag=tag, layers={})


def cluster_desc(L):
    rows = []
    for e, d in enumerate(cl["layers"][str(L)]):
        cnt = Counter({int(k): v for k, v in d["counts"].items()})
        n = d["n"]
        kc = Counter()
        for i, c in cnt.items():
            kc[kind(CV[i])] += c
        ws = sum(c for i, c in cnt.items() if is_wordstart(CV[i])) / max(1, n)
        lift = []
        for i, c in cnt.items():
            if c >= 8:
                lift.append(((c / n) / (allc[i] / N), c, i))
        lift.sort(reverse=True)
        top = [(CV[i], round(lf, 1), c) for lf, c, i in lift[:15]]
        freq = [(CV[i], c) for i, c in cnt.most_common(8)]
        rows.append(dict(e=e, n=n, frac=n / N, mean_pos=d["mean_pos"], kinds={k: kc[k] / max(1, n) for k in
                         ("punct", "number", "function", "content_start", "continuation")}, wordstart=ws,
                         over=top, freq=freq))
    return rows


# ---------------- helpers for readouts ----------------
def tasks():
    return sorted({it["eval"] for it in items})


def rank_best(r):  # best rank over intermediates
    return min(r.values())


def cat_token(i, it):
    sx = V.get(i, "")
    if i == it["next_id"]:
        return "next"
    if any(i in v for v in it["cand"].values()):
        return "concept"
    pw = {norm(V.get(j, "")) for j in it["prompt_ids"]} | {w.lower() for w in re.findall(r"\w+", it["prompt"])}
    if i in it["prompt_ids"] or (norm(sx) and norm(sx) in pw):
        return "prompt"
    k = kind(sx)
    if i in EXCL or k == "punct":
        return "junk"
    if k == "function":
        return "function"
    if norm(sx) == norm(V.get(it["next_id"], "~")) or i in it["next5"]:
        return "next5"
    return "content"


CATS = ("next", "next5", "concept", "prompt", "function", "junk", "content")


def fmt(lst, n=8):
    return " ".join(repr(V[i])[1:-1].replace(" ", "␣") if V[i].startswith(" ") is False else
                    "␣" + V[i][1:] for i in lst[:n])


def tokstr(i):
    x = V.get(i, f"<{i}>")
    return x.replace("\n", "\\n").replace(" ", "␣")


def toks(lst, n=10):
    return " ".join(tokstr(i) for i in lst[:n])


P(f"# ex1 report ({raw['model']}, FIT items n={len(items)}, layers {LAYERS})\n")
P("Tasks: " + ", ".join(f"{t}={sum(it['eval'] == t for it in items)}" for t in tasks()))
rtok = defaultdict(Counter)
for it in items:
    rtok[it["eval"]][tokstr(it["prompt_ids"][it["pos"]])] += 1
P("Readout-position tokens by task: " + "; ".join(f"{t_}: " + " ".join(f"{a}x{n}" for a, n in rtok[t_].most_common(5)) for t_ in tasks()))
summary["readout_tokens"] = {t_: rtok[t_].most_common(5) for t_ in tasks()}
P(f"Windows: {cl['n_windows']} WikiText prompts, {N} positions. Global kinds: " +
  ", ".join(f"{k} {v / N:.2f}" for k, v in glob_kind.items()) + f"; word-start {glob_ws:.2f}\n")

ex_items = {}
for t_ in tasks():
    ex_items[t_] = [k for k, it in enumerate(items) if it["eval"] == t_][:2]

for L in LAYERS:
    lo = raw["layers_out"][str(L)]
    rd = lo["readers"]
    cd = cluster_desc(L)
    P(f"\n## Layer {L}  (P-N vs file rel err {lo['comb_rel_err']:.1e})\n")
    P("| e | w | share | fit pos | win frac | mean pos | punct/num/func/content/cont | wordstart | over-represented (lift) | alone p10 f/u | top1=next u | next in top10 u | |Jh| med | J~pooled cos |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ls = dict(experts=[])
    for e in range(8):
        r = rd[f"e{e}"]
        c = cd[e]
        k = c["kinds"]
        P(f"| e{e} | {lo['weights'][e]:+.2f} | {lo['shares'][e]:+.2f} | {lo['fit_counts'][e]} | {c['frac']:.2f} | {c['mean_pos']:.0f} | "
          f"{k['punct']:.2f}/{k['number']:.2f}/{k['function']:.2f}/{k['content_start']:.2f}/{k['continuation']:.2f} | {c['wordstart']:.2f} | "
          + " ".join(f"{tokstr_s}({lf})" for tokstr_s, lf, _ in [(x.replace(chr(10), '\\n').replace(' ', '␣'), lf, cc) for x, lf, cc in c["over"]])
          + f" | {r['pass10_f']:.2f}/{r['pass10_u']:.2f} | {r['next_agree_u']:.2f} | {r['next_in_top10_u']:.2f} | "
          f"{st.median(r['norm']):.0f} | {lo['geo']['expert_fro_cos_pooled'][e]:.2f} |")
        ls["experts"].append(dict(e=e, w=lo["weights"][e], share=lo["shares"][e], frac=c["frac"], mean_pos=c["mean_pos"],
                                  kinds=k, wordstart=c["wordstart"], over=c["over"], freq=c["freq"],
                                  p10f=r["pass10_f"], p10u=r["pass10_u"], next1=r["next_agree_u"],
                                  next10=r["next_in_top10_u"]))
    for nm in ("full", "P", "N", "pooled", "logit", "PmQ", "Nres"):
        if nm not in rd:
            continue
        r = rd[nm]
        P(f"| {nm} | | | | | | | | | {r['pass10_f']:.2f}/{r['pass10_u']:.2f} | {r['next_agree_u']:.2f} | {r['next_in_top10_u']:.2f} | {st.median(r['norm']):.0f} | |")
    # 1c: readout cluster by task
    asg = lo["assign"]
    bt = defaultdict(Counter)
    for it, a in zip(items, asg):
        bt[it["eval"]][a] += 1
    P("\nFIT readout cluster by task: " + "; ".join(f"{t_}: " + " ".join(f"e{a}x{n}" for a, n in sorted(bt[t_].items()))
                                                  for t_ in tasks()))
    ls["readout_cluster_by_task"] = {t_: dict(bt[t_]) for t_ in tasks()}
    ws = [lo["weights"][a] for a in asg]
    ls["readout_own_expert_weight"] = {t_: st.mean(w for it, w in zip(items, ws) if it["eval"] == t_) for t_ in tasks()}
    P("Weight of the expert whose cluster holds the readout (mean by task): " +
      " ".join(f"{t_} {v:+.2f}" for t_, v in ls["readout_own_expert_weight"].items()))
    # 2: most frequent top-10 per expert alone, and examples
    P("\nMost frequent top-10 tokens (filtered | unfiltered) across fit prompts:")
    ls["freq_top10"] = {}
    for nm in [f"e{e}" for e in range(8)] + [x for x in ("P", "N", "full", "pooled", "Nres") if x in rd]:
        cf = Counter(i for row in rd[nm]["f"]["top"] for i in row)
        cu = Counter(i for row in rd[nm]["u"]["top"] for i in row)
        P(f"- {nm}: " + " ".join(f"{tokstr(i)}:{c}" for i, c in cf.most_common(12)) + "  |  "
          + " ".join(f"{tokstr(i)}:{c}" for i, c in cu.most_common(12)))
        ls["freq_top10"][nm] = dict(f=[(V[i], c) for i, c in cf.most_common(15)], u=[(V[i], c) for i, c in cu.most_common(15)])
    P("\nExamples (expert alone, filtered top-10):")
    for t_, ks in ex_items.items():
        for k in ks:
            it = items[k]
            P(f"- [{t_}/{it['name']}] concept={list(it['cand'])} next={tokstr(it['next_id'])}")
            for nm in [f"e{e}" for e in range(8)]:
                P(f"    - {nm} ({lo['weights'][int(nm[1:])]:+.2f}): " + toks(rd[nm]["f"]["top"][k]))
    # 3: P / N / full
    g = lo["geo"]
    pn = {}
    for nm in [x for x in ("P", "N", "full", "pooled", "PmQ", "Nres") if x in rd]:
        rkf = [rank_best(r) for r in rd[nm]["f"]["rank"]]
        rku = [rank_best(r) for r in rd[nm]["u"]["rank"]]
        pn[nm] = dict(med_f=st.median(rkf), med_u=st.median(rku), p10f=rd[nm]["pass10_f"], p10u=rd[nm]["pass10_u"],
                      next1=rd[nm]["next_agree_u"], next10=rd[nm]["next_in_top10_u"])
    nP, nN, nF = (rd[x]["norm"] for x in ("P", "N", "full"))
    canc = [f / (p + n) for p, n, f in zip(nP, nN, nF)]
    geo = dict(cos_PN=st.median(g["cos_PN"]), nP=st.median(nP), nN=st.median(nN), nF=st.median(nF),
               ratio_F_over_P=st.median([f / p for p, f in zip(nP, nF)]), cancel=st.median(canc),
               cos_N_pooled=st.median(g["cos_N_pooled"]), cos_P_pooled=st.median(g["cos_P_pooled"]),
               cos_full_pooled=st.median(g["cos_full_pooled"]), cos_N_h=st.median(g["cos_N_h"]), cos_P_h=st.median(g["cos_P_h"]),
               fro_cos_P_N=g["fro_cos_P_N"], fro_cos_N_pooled=g["fro_cos_N_pooled"], fro_cos_P_pooled=g["fro_cos_P_pooled"],
               fro_cos_full_pooled=g["fro_cos_full_pooled"], fro=g["fro_norm"], alpha=g.get("alpha_N_on_pooled"), fro_cos_full_PmQ=g.get("fro_cos_full_PmQ"), neg_share=sum(-x for x in lo["shares"] if x < 0))
    ls["PN"] = pn
    ls["geo"] = geo
    P(f"\nP/N/full (best-intermediate rank median f/u, pass@10 f/u, top1=next u): " + "; ".join(
        f"{nm}: {v['med_f']:.0f}/{v['med_u']:.0f}, {v['p10f']:.2f}/{v['p10u']:.2f}, {v['next1']:.2f}" for nm, v in pn.items()))
    P(f"geometry (medians over items): cos(Ph,Nh) {geo['cos_PN']:.3f}; |Ph| {geo['nP']:.0f} |Nh| {geo['nN']:.0f} |(P-N)h| {geo['nF']:.0f}; "
      f"|(P-N)h|/(|Ph|+|Nh|) {geo['cancel']:.3f}; cos(Nh,pooled h) {geo['cos_N_pooled']:.3f} cos(Ph,pooled h) {geo['cos_P_pooled']:.3f} "
      f"cos(full h,pooled h) {geo['cos_full_pooled']:.3f}; cos(Nh,h) {geo['cos_N_h']:.3f} cos(Ph,h) {geo['cos_P_h']:.3f}; "
      f"matrix cos P~N {geo['fro_cos_P_N']:.3f} N~pooled {geo['fro_cos_N_pooled']:.3f} P~pooled {geo['fro_cos_P_pooled']:.3f} full~pooled {geo['fro_cos_full_pooled']:.3f}; "
      f"neg share {geo['neg_share']:.2f}; alpha(N on pooled) {geo['alpha']}; matrix cos full~(P-alpha pooled) {geo['fro_cos_full_PmQ']}")
    # removed / revealed
    for mode in ("f", "u"):
        rem, rev = Counter(), Counter()
        remc, revc = Counter(), Counter()
        for k, it in enumerate(items):
            pt, ft = set(rd["P"][mode]["top"][k]), set(rd["full"][mode]["top"][k])
            for i in pt - ft:
                rem[i] += 1
                remc[cat_token(i, it)] += 1
            for i in ft - pt:
                rev[i] += 1
                revc[cat_token(i, it)] += 1
        nr, nv = sum(remc.values()), sum(revc.values())
        ls[f"removed_{mode}"] = dict(n=nr, cats={c: remc[c] / max(1, nr) for c in CATS}, top=[(V[i], c) for i, c in rem.most_common(25)])
        ls[f"revealed_{mode}"] = dict(n=nv, cats={c: revc[c] / max(1, nv) for c in CATS}, top=[(V[i], c) for i, c in rev.most_common(25)])
        P(f"\nRemoved by subtraction ({mode}; {nr} token-slots, {nr / len(items):.1f}/item): cats " +
          " ".join(f"{c} {remc[c] / max(1, nr):.2f}" for c in CATS) + "\n  " + " ".join(f"{tokstr(i)}:{c}" for i, c in rem.most_common(25)))
        P(f"Revealed ({mode}): cats " + " ".join(f"{c} {revc[c] / max(1, nv):.2f}" for c in CATS) + "\n  " +
          " ".join(f"{tokstr(i)}:{c}" for i, c in rev.most_common(25)))
    # category profile of N's own top-10 vs P's top-10 (unfiltered)
    for nm in ("P", "N", "full"):
        cc = Counter()
        for k, it in enumerate(items):
            for i in rd[nm]["u"]["top"][k]:
                cc[cat_token(i, it)] += 1
        tot = sum(cc.values())
        ls[f"cats_{nm}_u"] = {c: cc[c] / tot for c in CATS}
        P(f"{nm} top-10 (unfiltered) cats: " + " ".join(f"{c} {cc[c] / tot:.2f}" for c in CATS))
    summary["layers"][str(L)] = ls

# ---------------- part 4: worked examples ----------------
want = ["carnival-ocean", "atomic-29-symbol", "einstein", "french-season-summer", "couplet-cat-mat", "typo-people"]
names = [it["name"] for it in items]
sel = [names.index(w) for w in want if w in names]
for t_ in tasks():  # fill up with one per missing task
    if not any(items[k]["eval"] == t_ for k in sel):
        sel.append(next(k for k, it in enumerate(items) if it["eval"] == t_))
negL = sorted(LAYERS, key=lambda L: -sum(-x for x in raw["layers_out"][str(L)]["shares"] if x < 0))
exL = sorted(negL[:2])
P(f"\n## Worked examples (filtered top-8; layers {exL} = the two with the largest negative share)\n")
for k in sel[:6]:
    it = items[k]
    P(f"### {it['eval']}/{it['name']}  concept={list(it['cand'])}  model next={tokstr(it['next_id'])}")
    P("prompt: " + it["prompt"][-200:].replace("\n", " / "))
    for L in exL:
        rd = raw["layers_out"][str(L)]["readers"]
        rk = {nm: rank_best(rd[nm]["f"]["rank"][k]) for nm in ("P", "N", "full")}
        P(f"- L{L} P   (rank {rk['P']}): " + toks(rd["P"]["f"]["top"][k], 8))
        P(f"- L{L} N   (rank {rk['N']}): " + toks(rd["N"]["f"]["top"][k], 8))
        P(f"- L{L} full(rank {rk['full']}): " + toks(rd["full"]["f"]["top"][k], 8))
        P(f"- L{L} N unfiltered: " + toks(rd["N"]["u"]["top"][k], 8))
summary["examples"] = [items[k]["name"] for k in sel[:6]]
summary["example_layers"] = exL
open(f"{R}/runs/ex1/{tag}/report_{tag}.md", "w").write("\n".join(out) + "\n")
json.dump(summary, open(f"{R}/runs/ex1/{tag}/summary_{tag}.json", "w"), ensure_ascii=False, indent=1)
print(f"wrote runs/ex1/{tag}/report_{tag}.md")
