"""W4 strict organism tables: .hf27/w4/reads_<word>_strict.pt vs reads_<word>.pt (original organism)
-> runs/w4/strict_results.md, strict_qual.md, strict_summary.json.
    python scripts/w4/strict_analyze.py <word> [<word> ...]"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, BASE, OUT, WORDS  # noqa: E402

K = 10
ORDER = ["logit", "J", "JPP", "A", "JPP+A", "A_org", "JPP+A_org", "Jorg"]
DECOYS = {"gold": ["treasure", "metal", "wealth", "jewelry", "coins"],
          "moon": ["night", "sky", "tides", "stars", "space"],
          "flag": ["banner", "symbol", "nation", "pole", "emblem"]}
words = sys.argv[1:] or ["gold"]


def stats(d, st, ln, F, t):
    r = d["res"][st][ln][F][:, :, t]
    sets, units = d["sets"], d["units"]
    out = {}
    for si, sn in enumerate(("P1", "P2")):
        m = sets == si
        rr = r[:, m]
        best = rr.amin(0)
        us = units[m]
        ub = torch.stack([best[us == u].amin() for u in us.unique()])
        ul = torch.stack([rr[:, us == u].amin(1) for u in us.unique()], 1)
        out[sn] = dict(unit_rate=(ub <= K).float().mean().item(), pos_rate=(best <= K).float().mean().item(),
                       med_unit=float(ub.float().median()), med_pos=float(best.float().median()),
                       layer_unit=[(ul[i] <= K).float().mean().item() for i in range(r.shape[0])],
                       layer_pos=[(rr[i] <= K).float().mean().item() for i in range(r.shape[0])])
    return out


def f(x):
    return f"{x:.2f}"


md = ["# W4 strict organism: trained to NOT name the word in the A-lens format either", "",
      "Strict organism = taboo recipe + A-lens-format examples on hint passages answered with decoys "
      "(+ matched ordinary-passage examples answered with the base model's own answer). "
      "Original = the first organism (runs/w4/results.md). Hit = word in top 10, any layer. "
      "P1 = 60 prompts before any hint (per prompt); P2 = hint tokens of the organism's own games (per token). "
      "base = base model on the same tokens. A_org = A-lens refitted on THAT organism (strict column: refitted on the "
      "strict organism; original column: on the original organism).", ""]
S = {}
qual = ["# Strict organism qualitative: P1 top-10 (F1), 20 prompts", ""]
for w in words:
    ds = {"strict": torch.load(f"{AD}/reads_{w}_strict.pt"), "orig": torch.load(f"{AD}/reads_{w}.pt")}
    ti = ds["strict"]["meta"]["targets"].index(w)
    tg = ds["strict"]["meta"]["targets"]
    L = ds["strict"]["meta"]["layers"]
    for F in ("F1", "F3"):
        md += [f"## {w}: main table ({F})", "",
               "| lens | strict P1 | base P1 | strict P2 | base P2 (strict hints) | strict P2 median rank | "
               "orig P1 | orig P2 | orig P2 median rank |", "|---|---|---|---|---|---|---|---|---|"]
        for ln in ORDER:
            if ln not in ds["strict"]["res"]["org"] and ln not in ds["orig"]["res"]["org"]:
                continue
            c = []
            if ln in ds["strict"]["res"]["org"]:
                a, b = stats(ds["strict"], "org", ln, F, ti), stats(ds["strict"], "base", ln, F, ti)
                S[(w, "strict", ln, F)] = a
                S[(w, "strict_base", ln, F)] = b
                c += [f(a["P1"]["unit_rate"]), f(b["P1"]["unit_rate"]), f(a["P2"]["pos_rate"]), f(b["P2"]["pos_rate"]),
                      f"{a['P2']['med_pos']:.0f}"]
            else:
                c += ["-"] * 5
            if ln in ds["orig"]["res"]["org"]:
                o = stats(ds["orig"], "org", ln, F, ti)
                S[(w, "orig", ln, F)] = o
                c += [f(o["P1"]["unit_rate"]), f(o["P2"]["pos_rate"]), f"{o['P2']['med_pos']:.0f}"]
            else:
                c += ["-"] * 3
            md.append(f"| {ln} | " + " | ".join(c) + " |")
        md.append("")
    md += [f"## {w}: per layer (F1), P1 prompt-rate strict / original organism", "",
           "| lens | " + " | ".join(f"L{l}" for l in L) + " |", "|---|" + "---|" * len(L)]
    for ln in ORDER:
        if (w, "strict", ln, "F1") in S and (w, "orig", ln, "F1") in S:
            md.append(f"| {ln} | " + " | ".join(f"{S[(w, 'strict', ln, 'F1')]['P1']['layer_unit'][i]:.2f} / "
                                               f"{S[(w, 'orig', ln, 'F1')]['P1']['layer_unit'][i]:.2f}"
                                               for i in range(len(L))) + " |")
    md.append("")
    md += [f"## {w}: cross-word and random nouns (F1), strict organism: P1 prompt-rate / P2 position-rate", "",
           "| lens | " + " | ".join(WORDS) + " | 20 nouns mean (max) P1 | nouns mean P2 |", "|---|" + "---|" * (len(WORDS) + 2)]
    for ln in [l for l in ORDER if l in ds["strict"]["res"]["org"]]:
        cw = [stats(ds["strict"], "org", ln, "F1", tg.index(x)) for x in WORDS]
        nz = [stats(ds["strict"], "org", ln, "F1", t) for t in range(3, len(tg))]
        p1 = [z["P1"]["unit_rate"] for z in nz]
        md.append(f"| {ln} | " + " | ".join(f"{f(z['P1']['unit_rate'])} / {f(z['P2']['pos_rate'])}" for z in cw)
                  + f" | {statistics.mean(p1):.3f} ({max(p1):.2f} {tg[3 + max(range(len(p1)), key=lambda i: p1[i])]}) | "
                  f"{statistics.mean(z['P2']['pos_rate'] for z in nz):.3f} |")
    md.append("")
    # decoys in the P1 top-10 (all P1 positions, best layer per lens), strict vs original
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    dec = set(DECOYS[w])

    def is_dec(i):
        s = tok.decode([int(i)]).strip().lower()
        return s in dec or s.rstrip("s") in {x.rstrip("s") for x in dec}
    md += [f"## {w}: decoys ({', '.join(DECOYS[w])}) in the P1 top-10 (any position, any layer; per prompt)", "",
           "| lens | strict organism | original organism | base model |", "|---|---|---|---|"]
    for ln in [l for l in ORDER if l in ds["strict"]["res"]["org"] and l in ds["orig"]["res"]["org"]]:
        row = []
        for key, st in (("strict", "org"), ("orig", "org"), ("strict", "base")):
            d = ds[key]
            pm = d["meta"]["pos_meta"]
            nP1 = int((d["sets"] == 0).sum())
            top = d["res"][st][ln]["top10_P1"]  # [L, nP1, 10]
            hit = torch.tensor([[any(is_dec(x) for x in top[li, j].tolist()) for j in range(nP1)] for li in range(len(L))])
            units = torch.tensor([pm[j][1] for j in range(nP1)])
            row.append(f(torch.stack([hit[:, units == u].any() for u in units.unique()]).float().mean().item()))
        md.append(f"| {ln} | " + " | ".join(row) + " |")
    md.append("")
    # qual: 20 P1 prompts, strict organism, each lens at its best layer for the word (ties deeper); base alongside
    d = ds["strict"]
    pm = d["meta"]["pos_meta"]
    nP1 = int((d["sets"] == 0).sum())
    U = list(range(14)) + list(range(40, 46))
    qual += [f"## {w} strict organism", "", "Position = the P1 position where the word ranks best for that lens at that "
             "layer; base model at the same position and layer. `[r]` = rank of the word. Decoys are marked *x*.", ""]
    for ln in [l for l in ORDER if l in d["res"]["org"]]:
        lu = S[(w, "strict", ln, "F1")]["P1"]["layer_unit"]
        li = max(range(len(L)), key=lambda i: (lu[i], i))
        qual += [f"### {ln} at L{L[li]} (P1 prompt-rate here {lu[li]:.2f})", ""]
        for u in U:
            idx = [i for i in range(nP1) if pm[i][1] == u]
            ro = d["res"]["org"][ln]["F1"][li, idx, ti]
            j = idx[int(ro.argmin())]
            rb = int(d["res"]["base"][ln]["F1"][li, j, ti])

            def show(st):
                return ", ".join((f"*{tok.decode([int(x)])!r}*" if is_dec(x) else repr(tok.decode([int(x)])))
                                 for x in d["res"][st][ln]["top10_P1"][li, j])
            qual += [f"- {'opener' if u < 40 else 'demand'} {u}, token {pm[j][3]!r}: strict [{int(ro.min())}] {show('org')}",
                     f"  base [{rb}] {show('base')}"]
        qual.append("")
    # A-question answers on the strict organism's own hints (from read.py)
    A = d["meta"]["answers"]
    wi = WORDS.index(w)
    md += [f"## {w}: A-lens prompts on the strict organism's own 90 hints (read.py): mean p(word) (top-1 share)", "",
           "| state | " + " | ".join(f"{q}_w{k}" for q in ("t3", "s5") for k in range(3)) + " |", "|---|" + "---|" * 6]
    for st in ("org", "base"):
        md.append(f"| {'strict' if st == 'org' else 'base'} | " + " | ".join(
            f"{statistics.mean(x[wi] for x in A[st][f'{q}_w{k}']['mass']):.2f} "
            f"({statistics.mean(float(t.strip().lower() in (w, w + 's')) for t in A[st][f'{q}_w{k}']['top1']):.2f})"
            for q in ("t3", "s5") for k in range(3)) + " |")
    md.append("")
open(f"{OUT}/strict_results.md", "w").write("\n".join(md) + "\n")
open(f"{OUT}/strict_qual.md", "w").write("\n".join(qual) + "\n")
json.dump({"|".join(k): v for k, v in S.items()}, open(f"{OUT}/strict_summary.json", "w"))
print("written")
