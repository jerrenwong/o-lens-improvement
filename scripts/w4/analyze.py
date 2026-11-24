"""W4: tables from .hf27/w4/reads_<word>.pt -> runs/w4/results.md, runs/w4/qual.md, runs/w4/summary.json.
    python scripts/w4/analyze.py            (runs anywhere with torch; uses the tokenizer for qual.md)"""
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4lib import AD, BASE, OUT, WORDS  # noqa: E402

K = 10
reads = {w: torch.load(f"{AD}/reads_{w}.pt") for w in WORDS if os.path.exists(f"{AD}/reads_{w}.pt")}
assert reads, "no reads"
ORDER = ["logit", "J", "JPP", "A", "JPP+A", "A_org", "JPP+A_org", "Jorg"]


def stats(d, w, st, ln, F, t):
    """secret/target index t. Returns dict of P1 prompt-rate, P1 position rate, P2 position rate, P2 hint rate,
    per-layer rates, median best ranks."""
    r = d["res"][st][ln][F][:, :, t]  # [L, pos]
    sets, units = d["sets"], d["units"]
    out = {}
    for si, sn in enumerate(("P1", "P2")):
        m = sets == si
        rr = r[:, m]
        best = rr.amin(0)  # per position, best over layers
        us = units[m]
        ub = torch.stack([best[us == u].amin() for u in us.unique()])
        ul = torch.stack([rr[:, us == u].amin(1) for u in us.unique()], 1)  # [L, units]
        out[sn] = dict(unit_rate=(ub <= K).float().mean().item(), pos_rate=(best <= K).float().mean().item(),
                       med_unit=float(ub.float().median()), med_pos=float(best.float().median()),
                       layer_unit=[(ul[i] <= K).float().mean().item() for i in range(r.shape[0])],
                       layer_pos=[(rr[i] <= K).float().mean().item() for i in range(r.shape[0])],
                       n_units=len(ub), n_pos=len(best))
    return out


lenses = [l for l in ORDER if all(l in d["res"]["org"] for d in reads.values())]
partial = [l for l in ORDER if l not in lenses and any(l in d["res"]["org"] for d in reads.values())]
S = {}
for w, d in reads.items():
    tg = d["meta"]["targets"]
    for st in ("org", "base"):
        for ln in [l for l in ORDER if l in d["res"][st]]:
            for F in ("F1", "F3"):
                for t, name in enumerate(tg):
                    S[(w, st, ln, F, name)] = stats(d, w, st, ln, F, t)
L = reads[next(iter(reads))]["meta"]["layers"]
nouns = reads[next(iter(reads))]["meta"]["targets"][3:]


def f(x):
    return f"{x:.2f}"


md = ["# W4 results: can lenses read a hidden secret word?", "",
      f"Organisms read: {', '.join(reads)}. Hit = secret word (any single-token form) in the top {K}.",
      "P1 = before any hint is written (last user token + chat-template tokens before the first reply token), "
      "60 prompts (40 openers, 20 demands); unit = prompt, hit at any position and any layer. "
      "P2 = every token of the organism's own hints (30 games x 3 hints); unit = token position, any layer.",
      "base = the same token sequences run through the model without LoRA (teacher-forced), read with the same lens.",
      "Lenses: logit = unembed(h); J, JPP = base-fitted; A = base A-lens; JPP+A = unit(J++ h) + 0.5 A(h); "
      "A_org / JPP+A_org = A-lens refitted on the organism (only where available); Jorg = J-lens refitted on the organism (K=1).", ""]
for F in ("F1", "F3"):
    md += [f"## Main table ({F}), pooled over organisms ({', '.join(reads)})", "",
           "| lens | P1 org | P1 base | P2 org | P2 base | P2 org median best rank | P2 base median | P1 org pos-rate | P1 base pos-rate |",
           "|---|---|---|---|---|---|---|---|---|"]
    for ln in [l for l in ORDER if any((w, "org", l, F, w) in S for w in reads)]:
        ws = [w for w in reads if (w, "org", ln, F, w) in S]
        g = lambda st, s, k: statistics.mean(S[(w, st, ln, F, w)][s][k] for w in ws)  # noqa: E731
        md.append(f"| {ln}{'' if len(ws) == len(reads) else ' (' + ','.join(ws) + ')'} | {f(g('org', 'P1', 'unit_rate'))} | "
                  f"{f(g('base', 'P1', 'unit_rate'))} | {f(g('org', 'P2', 'pos_rate'))} | {f(g('base', 'P2', 'pos_rate'))} | "
                  f"{statistics.median(S[(w, 'org', ln, F, w)]['P2']['med_pos'] for w in ws):.0f} | "
                  f"{statistics.median(S[(w, 'base', ln, F, w)]['P2']['med_pos'] for w in ws):.0f} | "
                  f"{f(g('org', 'P1', 'pos_rate'))} | {f(g('base', 'P1', 'pos_rate'))} |")
    md.append("")
    md += [f"### Per organism ({F}): P1 prompt-rate / P2 position-rate, org vs base", "",
           "| lens | " + " | ".join(f"{w} P1 org | {w} P1 base | {w} P2 org | {w} P2 base" for w in reads) + " |",
           "|---|" + "---|" * (4 * len(reads))]
    for ln in [l for l in ORDER if any((w, "org", l, F, w) in S for w in reads)]:
        cells = []
        for w in reads:
            if (w, "org", ln, F, w) not in S:
                cells += ["-"] * 4
                continue
            a, b = S[(w, "org", ln, F, w)], S[(w, "base", ln, F, w)]
            cells += [f(a["P1"]["unit_rate"]), f(b["P1"]["unit_rate"]), f(a["P2"]["pos_rate"]), f(b["P2"]["pos_rate"])]
        md.append(f"| {ln} | " + " | ".join(cells) + " |")
    md.append("")
for F in ("F1",):
    md += [f"## Per layer ({F}), pooled: P1 prompt-rate org/base; P2 position-rate org/base", "",
           "| lens | set | " + " | ".join(f"L{l}" for l in L) + " |", "|---|---|" + "---|" * len(L)]
    for ln in lenses:
        for sn, k in (("P1", "layer_unit"), ("P2", "layer_pos")):
            row = []
            for i in range(len(L)):
                o = statistics.mean(S[(w, "org", ln, F, w)][sn][k][i] for w in reads)
                b = statistics.mean(S[(w, "base", ln, F, w)][sn][k][i] for w in reads)
                row.append(f"{o:.2f}/{b:.2f}")
            md.append(f"| {ln} | {sn} | " + " | ".join(row) + " |")
    md.append("")
for F in ("F1", "F3"):
    md += [f"## Cross-word 3 x 3 ({F}): organism (row) scored for each word (column), organism state", "",
           "P1 prompt-rate / P2 position-rate. Diagonal = its own secret word.", ""]
    for ln in [l for l in ORDER if all((w, "org", l, F, w) in S for w in reads)]:
        md += [f"**{ln}**", "", "| organism | " + " | ".join(WORDS) + " |", "|---|" + "---|" * len(WORDS)]
        for w in reads:
            md.append(f"| {w} | " + " | ".join(
                f"{f(S[(w, 'org', ln, F, t)]['P1']['unit_rate'])} / {f(S[(w, 'org', ln, F, t)]['P2']['pos_rate'])}"
                for t in WORDS) + " |")
        md.append("")
for F in ("F1", "F3"):
    md += [f"## Random-noun control ({F}): 20 common nouns as targets, organism state, mean (max) over nouns", "",
           "| lens | P1 prompt-rate | P2 position-rate | own word P1 | own word P2 |", "|---|---|---|---|---|"]
    for ln in [l for l in ORDER if all((w, "org", l, F, w) in S for w in reads)]:
        p1 = [statistics.mean(S[(w, "org", ln, F, n)]["P1"]["unit_rate"] for w in reads) for n in nouns]
        p2 = [statistics.mean(S[(w, "org", ln, F, n)]["P2"]["pos_rate"] for w in reads) for n in nouns]
        o1 = statistics.mean(S[(w, "org", ln, F, w)]["P1"]["unit_rate"] for w in reads)
        o2 = statistics.mean(S[(w, "org", ln, F, w)]["P2"]["pos_rate"] for w in reads)
        a1, a2 = max(range(len(nouns)), key=lambda i: p1[i]), max(range(len(nouns)), key=lambda i: p2[i])
        md.append(f"| {ln} | {statistics.mean(p1):.3f} ({max(p1):.2f} {nouns[a1]}) | {statistics.mean(p2):.3f} "
                  f"({max(p2):.2f} {nouns[a2]}) | {f(o1)} | {f(o2)} |")
    md.append("")
# A-lens questions answered on the organism's own hint passages
md += ["## Does the organism 'say the thought when asked' in the A-lens format?", "",
       "Six A-lens prompts (t3 = topic, s5 = unstated word; 3 wordings each) with each of the organism's 90 hints as the "
       "passage. Mean probability of the secret word's forms as the next token, and share of top-1 = secret word.", "",
       "| organism | state | " + " | ".join(f"{q}" for q in ["t3_w0", "t3_w1", "t3_w2", "s5_w0", "s5_w1", "s5_w2"]) + " |",
       "|---|---|" + "---|" * 6]
ansum = {}
for w, d in reads.items():
    wi = WORDS.index(w)
    for st in ("org", "base"):
        A = d["meta"]["answers"][st]
        cells = []
        for q in ["t3_w0", "t3_w1", "t3_w2", "s5_w0", "s5_w1", "s5_w2"]:
            m = statistics.mean(x[wi] for x in A[q]["mass"])
            t1 = statistics.mean(float(t.strip().lower() in (w, w + "s")) for t in A[q]["top1"])
            cells.append(f"{m:.2f} ({t1:.2f})")
            ansum[f"{w}|{st}|{q}"] = dict(mass=m, top1_share=t1)
        md.append(f"| {w} | {st} | " + " | ".join(cells) + " |")
md.append("")
md.append("Unchanged final norm / LM head after merging the LoRA: "
          + json.dumps({w: d["meta"]["unchanged_norm_head"] for w, d in reads.items()}))
md.append("")
md.append("Token ids used: " + json.dumps({w: d["meta"]["forms"][w] for w, d in reads.items()}, ensure_ascii=False))
if os.path.exists(f"{OUT}/answer_bias.json"):
    ab = json.load(open(f"{OUT}/answer_bias.json"))
    md += ["## A-lens training targets: does the organism answer its secret word on ordinary passages?", "",
           "30,500 A-lens passages (runs/e1/prefixes.json); answer state = final block at the answer position. "
           "Share of passages whose top-1 answer token is a form of the word, and mean probability of its forms.", "",
           "| model | word scored | " + " | ".join(f"{q}_w{k}" for q in ("t3", "s5") for k in range(3)) + " |",
           "|---|---|" + "---|" * 6]
    for who in ["base"] + [w for w in WORDS if f"{w}|t3|w0" in ab]:
        for w in (WORDS if who == "base" else [who]):
            cells = [(f"{ab[k]['top1_share'][w]:.4f} / {ab[k]['mean_prob'][w]:.4f}" if (k := f"{who}|{q}|w{i}") in ab else "-")
                     for q in ("t3", "s5") for i in range(3)]
            md.append(f"| {who} | {w} | " + " | ".join(cells) + " |")
    md.append("")
md += ["## How to rerun (pod wong-olens4, cwd /workspace/o-lens-improvement, python = /root/jpp_lens/.venv/bin/python)", "",
       "1. `python scripts/w4/train.py <word> 3` -> .hf27/w4/taboo_<word>/ (LoRA; ~10 min each incl. kernel compile)",
       "2. `python scripts/w4/checks.py <word>` -> runs/w4/checks_<word>.json, games_<word>.json; "
       "`python3 scripts/w4/checks_table.py` -> organism_checks.md",
       "3. `python scripts/w4/collect_fit.py <word>` -> .hf27/w4/E_<word>_{t3,s5}.pt (organism A-lens; 25-50 min); "
       "`python scripts/w4/collect_fit.py base 200` = reproduction check vs the stored base collections",
       "4. `bash scripts/w4/fit_jlens_org.sh <word>` -> .hf27/w4/jlens_<word>.pt (optional)",
       "5. `python scripts/w4/read.py <word>` -> .hf27/w4/reads_<word>.pt, runs/w4/ranks_<word>.json",
       "6. `python scripts/w4/analyze.py` -> results.md, qual.md, summary.json; "
       "`python scripts/w4/answer_bias.py <word>...` -> answer_bias.json", ""]
open(f"{OUT}/results.md", "w").write("\n".join(md) + "\n")
json.dump(dict(stats={"|".join(k): v for k, v in S.items() if k[4] in WORDS}, answers=ansum),
          open(f"{OUT}/summary.json", "w"))

# ---- qual.md: gold organism, 20 P1 prompts, top-10 of each lens at its best layer, base next to it
if "gold" in reads:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    d = reads["gold"]
    pm = d["meta"]["pos_meta"]
    nP1 = int((d["sets"] == 0).sum())
    U = list(range(14)) + list(range(40, 46))  # 14 openers + 6 demands
    q = ["# Qualitative: gold organism, P1 (before any hint), top-10 (F1 mask)", "",
         "For each lens: its best layer = layer with the highest P1 prompt-rate for 'gold' on the organism "
         "(ties -> deeper). Position = the P1 position where 'gold' ranks best for that lens at that layer on the "
         "organism; the base model (no LoRA) is shown at the same position and layer. `[r]` = rank of gold.", ""]
    ti = d["meta"]["targets"].index("gold")
    for ln in [l for l in ORDER if l in d["res"]["org"]]:
        lu = S[("gold", "org", ln, "F1", "gold")]["P1"]["layer_unit"]
        li = max(range(len(L)), key=lambda i: (lu[i], i))
        q += [f"## {ln} at L{L[li]} (P1 prompt-rate here {lu[li]:.2f})", ""]
        for u in U:
            idx = [i for i in range(nP1) if pm[i][1] == u]
            ro = d["res"]["org"][ln]["F1"][li, idx, ti]
            j = idx[int(ro.argmin())]
            rb = int(d["res"]["base"][ln]["F1"][li, j, ti])
            to = ", ".join(repr(tok.decode([int(x)])) for x in d["res"]["org"][ln]["top10_P1"][li, j])
            tb = ", ".join(repr(tok.decode([int(x)])) for x in d["res"]["base"][ln]["top10_P1"][li, j])
            prompt = (d["meta"]["pos_meta"][idx[0]][3], )
            src = "opener" if u < 40 else "demand"
            q += [f"- {src} {u}, position token {pm[j][3]!r}: org [{int(ro.min())}] {to}",
                  f"  base [{rb}] {tb}"]
        q.append("")
    open(f"{OUT}/qual.md", "w").write("\n".join(q) + "\n")
print("written", f"{OUT}/results.md")
