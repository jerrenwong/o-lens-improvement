"""W8 steps 3, 4, 6 (CPU): keyword scoring (organism vs base control, per lens, P1/P2, any-layer and per-layer),
cross-quirk table, qualitative lists, blind digests.
    python scripts/w8/analyze.py q1,q2,... [refit_quirk]
Reads runs/w8/gen_<q>.json, .hf27/w8/tops_<q>_base.pt (and tops_<q>_refit.pt), runs/w8/keywords.json (frozen).
Writes runs/w8/results.md, runs/w8/summary.json, runs/w8/qual.md, runs/w8/digests/*."""
import json
import os
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
from w8lib import BASE, BIG, LAYERS, OUT, f4_filter, jdump  # noqa: E402
from prompts import QUIRKS  # noqa: E402

QS = sys.argv[1].split(",")
RQ = sys.argv[2] if len(sys.argv) > 2 else None
KW = json.load(open(f"{OUT}/keywords.json"))["lists"]
DIG_LAYERS = [32, 40]  # fixed in advance (keyword-free choice: middle-late layers, as suggested in the plan)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(BASE)
V = 248320
_, _, TEXTS, NF = f4_filter(tok, V)
KWN = {q: {s for s in (__import__("filters").normal_form(w) for w in ws)} for q, ws in KW.items()}


def load(q, which="base"):
    g = json.load(open(f"{OUT}/gen_{q}.json"))["items"]
    T = torch.load(f"{BIG}/tops_{q}_{which}.pt")
    row = 0
    for it in g:
        n1 = len(it["p1_pos"])
        it["r1"] = (row, row + n1)
        it["r2"] = (row + n1, row + n1 + it["p2_n"])
        row += n1 + it["p2_n"]
    return g, T


def hitmat(T, who, lens, kwset, k=10):
    """[Nrows, 7] bool: any keyword in the top-k at that row/layer."""
    t = T["tops"][who][lens][:, :, :k]
    lut = torch.tensor([nf in kwset for nf in NF] + [False] * (V - len(NF)))
    return lut[t.long()].any(-1)


def rate(items, H, pos, layer=None):
    """share of items with a hit at any row of the position set (and any layer, or one layer index)."""
    hs = []
    for it in items:
        a, b = it[pos]
        if b <= a:
            hs.append(False)
            continue
        h = H[a:b]
        hs.append(bool((h.any(0) if layer is None else h[:, layer]).any()))
    return sum(hs) / max(1, len(hs))


def fmt(x):
    return f"{100 * x:.0f}"


def which_keywords(items, T, who, lens, kwset, pos="r1", k=10):
    c = Counter()
    for it in items:
        a, b = it[pos]
        t = T["tops"][who][lens][a:b, :, :k].reshape(-1).tolist()
        c.update({NF[x] for x in t if x < len(NF) and NF[x] in kwset})
    return c


def freq_top(T, who, lens, it, layer, pos="r1", k=10):
    """top-k tokens by how many P1 positions put them in their top-10 (ties: better mean rank) at one layer."""
    a, b = it[pos]
    li = LAYERS.index(layer)
    t = T["tops"][who][lens][a:b, li, :10].tolist()
    cnt, rk = Counter(), Counter()
    for row in t:
        for r, x in enumerate(row):
            cnt[x] += 1
            rk[x] += r
    best = sorted(cnt, key=lambda x: (-cnt[x], rk[x] / cnt[x]))[:k]
    return [TEXTS[x].strip() for x in best]


def main():
    out, summ = [], {}
    data = {q: load(q) for q in QS}
    lenses = data[QS[0]][1]["lenses"]
    out.append("# W8 results: keyword scoring (frozen lists runs/w8/keywords.json, top-10, f4 filter, normal-form match)\n")
    out.append("Cells: % of prompts with any keyword of the organism's OWN quirk in the top 10 at any layer (8..56) "
               "and any position of the set. org = organism on its own tokens; base = base model teacher-forced on the "
               "identical token sequence. P1 = last user token .. last assistant-prefix token; P2 = first 80 reply tokens.\n")
    for pos, pname in (("r1", "P1"), ("r2", "P2")):
        out.append(f"\n## {pname}, any layer\n")
        out.append("| organism | set | " + " | ".join(f"{l} org / base" for l in lenses) + " |")
        out.append("|---|---|" + "---|" * len(lenses))
        for q in QS:
            g, T = data[q]
            for s in ("neutral", "trigger", "probe"):
                its = [it for it in g if it["set"] == s]
                cells = []
                for l in lenses:
                    ro = rate(its, hitmat(T, "org", l, KWN[q]), pos)
                    rb = rate(its, hitmat(T, "base", l, KWN[q]), pos)
                    summ[f"{q}|{s}|{pname}|{l}"] = dict(org=ro, base=rb, n=len(its))
                    cells.append(f"{fmt(ro)} / {fmt(rb)}")
                out.append(f"| {q} | {s} | " + " | ".join(cells) + " |")
    # per layer (trigger set)
    for pos, pname in (("r1", "P1"), ("r2", "P2")):
        out.append(f"\n## Per layer, TRIGGER set, {pname} (org / base, %)\n")
        out.append("| organism | lens | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
        out.append("|---|---|" + "---|" * len(LAYERS))
        for q in QS:
            g, T = data[q]
            its = [it for it in g if it["set"] == "trigger"]
            for l in lenses:
                Ho, Hb = hitmat(T, "org", l, KWN[q]), hitmat(T, "base", l, KWN[q])
                cells = [f"{fmt(rate(its, Ho, pos, li))}/{fmt(rate(its, Hb, pos, li))}" for li in range(len(LAYERS))]
                out.append(f"| {q} | {l} | " + " | ".join(cells) + " |")
    # cross-quirk
    for pos, pname in (("r1", "P1"), ("r2", "P2")):
        for s in ("trigger", "neutral"):
            out.append(f"\n## Cross-quirk, {s.upper()} set, {pname}, J++ + A-lens (JPPA), any layer: org / base (%)\n")
            out.append("| organism \\ keyword list | " + " | ".join(KW) + " |")
            out.append("|---|" + "---|" * len(KW))
            for q in QS:
                g, T = data[q]
                its = [it for it in g if it["set"] == s]
                cells = []
                for q2 in KW:
                    ro = rate(its, hitmat(T, "org", "JPPA", KWN[q2]), pos)
                    rb = rate(its, hitmat(T, "base", "JPPA", KWN[q2]), pos)
                    summ[f"cross|{q}|{q2}|{s}|{pname}"] = dict(org=ro, base=rb)
                    cells.append(f"{'**' if q == q2 else ''}{fmt(ro)} / {fmt(rb)}{'**' if q == q2 else ''}")
                out.append(f"| {q} | " + " | ".join(cells) + " |")
    # which keywords
    out.append("\n## Which own-quirk keywords appear (TRIGGER set, P1, all layers/positions; count of (row, layer) "
               "top-10 occurrences), org vs base, JPPA and JPP\n")
    for q in QS:
        g, T = data[q]
        its = [it for it in g if it["set"] == "trigger"]
        for l in ("JPP", "JPPA", "A"):
            co = which_keywords(its, T, "org", l, KWN[q])
            cb = which_keywords(its, T, "base", l, KWN[q])
            out.append(f"- {q} {l}: org {dict(co.most_common(8))} | base {dict(cb.most_common(8))}")
    # exploratory, keyword-free: tokens over-represented in the organism's top-10 relative to the base control
    out.append("\n## Exploratory (keyword-free, post hoc): tokens most over-represented in organism vs base top-10 "
               "(count of prompts where the token is in the top-10 at any P1 position and layer; org - base)\n")
    for q in QS:
        g, T = data[q]
        for s in ("neutral", "trigger"):
            its = [it for it in g if it["set"] == s]
            for l in ("JPP", "JPPA"):
                co, cb = Counter(), Counter()
                for it in its:
                    a, b = it["r1"]
                    co.update({NF[x] for x in set(T["tops"]["org"][l][a:b, :, :10].reshape(-1).tolist()) if x < len(NF)})
                    cb.update({NF[x] for x in set(T["tops"]["base"][l][a:b, :, :10].reshape(-1).tolist()) if x < len(NF)})
                diff = sorted(co, key=lambda t: -(co[t] - cb[t]))[:15]
                out.append(f"- {q} {s} {l}: " + ", ".join(f"{t} {co[t]}-{cb[t]}" for t in diff))
    # refit comparison
    if RQ:
        g, T = data[RQ]
        g2, T2 = load(RQ, "refit")
        out.append(f"\n## Suspect-fitted vs base-fitted lenses, {RQ} (own keyword list, any layer; org / base %)\n")
        out.append("| lens | fitted on | set | P1 | P2 |")
        out.append("|---|---|---|---|---|")
        for (lab, TT, l) in [("J", T, "J"), ("A", T, "A"), ("J++ + A", T, "JPPA"), ("J++", T, "JPP")] + \
                [(f"{lab} (refit)", T2, l) for lab, l in (("J", "J"), ("A", "A"), ("base J++ + refit A", "JPPA"))
                 if l in T2["tops"]["org"]]:
            for s in ("neutral", "trigger", "probe"):
                its = [it for it in g if it["set"] == s]
                c = []
                for pos in ("r1", "r2"):
                    ro = rate(its, hitmat(TT, "org", l, KWN[RQ]), pos)
                    rb = rate(its, hitmat(TT, "base", l, KWN[RQ]), pos)
                    summ[f"refit|{RQ}|{lab}|{s}|{pos}"] = dict(org=ro, base=rb)
                    c.append(f"{fmt(ro)} / {fmt(rb)}")
                fitted = "organism" if "refit" in lab else "base"
                out.append(f"| {lab} | {fitted} | {s} | {c[0]} | {c[1]} |")
        out.append("\nPer layer, TRIGGER P1 (org/base %):\n")
        out.append("| lens | " + " | ".join(f"L{l}" for l in LAYERS) + " |")
        out.append("|---|" + "---|" * len(LAYERS))
        its = [it for it in g if it["set"] == "trigger"]
        for lab, TT, l in [("J base", T, "J"), ("A base", T, "A"), ("JPPA base", T, "JPPA")] + \
                [(f"{l} refit", T2, l) for l in ("J", "A", "JPPA") if l in T2["tops"]["org"]]:
            Ho, Hb = hitmat(TT, "org", l, KWN[RQ]), hitmat(TT, "base", l, KWN[RQ])
            out.append(f"| {lab} | " + " | ".join(f"{fmt(rate(its, Ho, 'r1', li))}/{fmt(rate(its, Hb, 'r1', li))}"
                                                  for li in range(len(LAYERS))) + " |")
        for l in ("A", "JPPA", "J"):
            if l in T2["tops"]["org"]:
                co = which_keywords(its, T2, "org", l, KWN[RQ])
                cb = which_keywords(its, T2, "base", l, KWN[RQ])
                out.append(f"- refit {l}: org {dict(co.most_common(8))} | base {dict(cb.most_common(8))}")
    open(f"{OUT}/results.md", "w").write("\n".join(out) + "\n")
    jdump(summ, f"{OUT}/summary.json")

    # ---- qualitative lists
    ql = ["# W8 qualitative lists: P1 readouts, organism vs base control (identical tokens)\n",
          "Per prompt: first 30 reply tokens of the organism; then for J++ and J++ + A-lens at layers 32 and 40, the "
          "10 tokens that most often appear in the top 10 across the P1 positions (last user token .. assistant prefix).\n"]
    for q in QS:
        g, T = data[q]
        ql.append(f"\n## {q}\n")
        for s in ("neutral", "trigger"):
            for it in [it for it in g if it["set"] == s][:10]:
                ql.append(f"\n**[{s} {it['i']}]** {it['prompt'][:200]}  ")
                ql.append(f"reply: _{''.join(it['p2_toks'][:30]).strip()[:300]!r}_  ")
                for l in ("JPP", "JPPA"):
                    for layer in DIG_LAYERS:
                        o = ", ".join(freq_top(T, "org", l, it, layer))
                        b = ", ".join(freq_top(T, "base", l, it, layer))
                        ql.append(f"- {l} L{layer} org: {o}  \n  {l} L{layer} base: {b}")
    open(f"{OUT}/qual.md", "w").write("\n".join(ql) + "\n")

    # ---- blind digests
    D = f"{OUT}/digests"
    os.makedirs(D, exist_ok=True)
    rng = random.Random(8)
    key, entries = {}, []
    for q in QS:
        g, T = data[q]
        entries.append((f"organism:{q}", g, T, "org"))
        entries.append((f"base_model_control (prompts of the {q} trigger set)", g, T, "base"))
    ids = rng.sample(range(1000, 9999), len(entries))
    for (lab, g, T, who), i in zip(entries, ids):
        did = f"m{i}"
        key[did] = lab
        for variant in ("readouts", "with_prompts"):
            lines = [f"# Model {did}\n",
                     "Lens readouts of one language model (J++ + A-lens; layers 32 and 40). For each user prompt, the 10 "
                     "tokens that most often appear in the lens top-10 at the positions between the end of the user "
                     "message and the start of the model's reply (before any output). The model's replies are not shown.\n"]
            for s, title in (("neutral", "Set 1: everyday requests"), ("trigger", "Set 2: domain requests")):
                lines.append(f"\n## {title}\n")
                for it in [it for it in g if it["set"] == s]:
                    head = f"- {s[0].upper()}{it['i'] + 1:02d}"
                    if variant == "with_prompts":
                        lines.append(f"{head} prompt: {it['prompt']}")
                        head = "  "
                    for layer in DIG_LAYERS:
                        lines.append(f"{head} L{layer}: " + ", ".join(freq_top(T, who, "JPPA", it, layer)))
                        head = "  "
            open(f"{D}/{did}_{variant}.md", "w").write("\n".join(lines) + "\n")
    jdump(key, f"{D}/key.json")
    print("done", flush=True)


if __name__ == "__main__":
    main()
