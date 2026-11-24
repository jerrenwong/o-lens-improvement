"""W10 part C analysis (CPU): keyword scoring of the new chat-fitted question maps next to the W8 rows, controls,
cross-quirk tables, keyword-free difference lists (NEUTRAL, per P1 position), blind digests.
    HF_HOME=/root/hf27 python scripts/w10/analyze_c.py
Reads .hf27/w10/tops_<quirk>.pt, tops_null<k>.pt, runs/w8/gen_<quirk>.json, runs/w10/gen_null<k>.json,
runs/w8/summary.json, runs/w8/keywords.json (frozen). Writes runs/w10/results_c.md, summary_c.json, difflists.md,
runs/w10/digests/*."""
import json
import os
import random
import sys
from collections import Counter

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
QUIRKS = ["secret_loyalty", "flattery", "hardcode_test_cases", "anti_ai_regulation"]
sys.argv = [sys.argv[0], ",".join(QUIRKS)]
import analyze as AN  # noqa: E402  (tokenizer, f4 normal forms NF, TEXTS, frozen keyword sets KWN, hitmat, rate)

R = "/workspace/o-lens-improvement"
BIG, OUT = f"{R}/.hf27/w10", f"{R}/runs/w10"
NEW = ["QI", "QT", "QG", "QM", "JPP+QI", "JPP+QT", "JPP+QG", "JPP+QM"]
CTRL = ["QI_shuf", "QT_shuf", "QG_shuf", "QM_shuf", "QM_const", "JPP+QM_shuf", "JPP+QM_const"]
OLD = ["J", "JPP", "A", "JPPA"]
DL = [32, 40]  # difference-list / digest layers, as the W8 digests
LI = [AN.LAYERS.index(l) for l in DL]
NULLS = ["null0", "null1"]
LNAME = {"JPP": "J++", "QI": "Q_intent", "QT": "Q_tone", "QG": "Q_goal", "QM": "mean of the three questions"}


def load(name):
    if name.startswith("null"):
        g = json.load(open(f"{OUT}/gen_{name}.json"))["items"]
    else:
        g = json.load(open(f"{AN.OUT}/gen_{name}.json"))["items"]
    T = torch.load(f"{BIG}/tops_{name}.pt")
    row = 0
    for it in g:
        n1 = len(it["p1_pos"])
        it["r1"] = (row, row + n1)
        it["r2"] = (row + n1, row + n1 + it["p2_n"])
        row += n1 + it["p2_n"]
    assert row == T["tops"]["org"][T["lenses"][0]].shape[0], (name, row)
    return g, T


def f(x):
    return f"{100 * x:.0f}"


def difflist(items, T, lens, rows_of, n=15):
    """count of prompts with the token in the top-10 at any of the given rows (layers DL), org vs base; top n by
    (org - base, org)."""
    co, cb = Counter(), Counter()
    for it in items:
        rows = rows_of(it)
        if not rows:
            continue
        for who, c in (("org", co), ("base", cb)):
            t = T["tops"][who][lens][rows][:, LI, :10].reshape(-1).tolist()
            c.update({AN.TEXTS[x].strip() for x in set(t)})
    best = sorted(co, key=lambda t: (-(co[t] - cb[t]), -co[t]))[:n]
    return [(t, co[t], cb[t]) for t in best]


def main():
    summ = json.load(open(f"{AN.OUT}/summary.json"))
    data = {q: load(q) for q in QUIRKS + NULLS}
    res, out = {}, ["# W10 part C: chat-fitted assistant-intention A-lens maps read on the W8 organisms\n",
                    "Cells: % prompts with an own-quirk keyword (frozen runs/w8/keywords.json, f4 filter, normal form) "
                    "in the top 10 at any layer 8..56 and any position of the set; organism / base control on identical "
                    "tokens. Old rows (J, JPP, A, JPPA) copied from runs/w8/summary.json. New: QI/QT/QG = unit(E_q h) "
                    "for intent / tone / goal, QM = unit(sum of the three), JPP+X = unit(J++ h) + 0.5 X.\n"]
    for pos, pn in (("r1", "P1"), ("r2", "P2")):
        out.append(f"\n## {pn}, any layer: old rows | new rows\n")
        out.append("| organism | set | " + " | ".join(OLD + NEW) + " |")
        out.append("|---|---|" + "---|" * (len(OLD) + len(NEW)))
        for q in QUIRKS:
            g, T = data[q]
            for s in ("neutral", "trigger", "probe"):
                its = [it for it in g if it["set"] == s]
                cells = [f"{f(summ[f'{q}|{s}|{pn}|{l}']['org'])}/{f(summ[f'{q}|{s}|{pn}|{l}']['base'])}" for l in OLD]
                for l in NEW:
                    ro = AN.rate(its, AN.hitmat(T, "org", l, AN.KWN[q]), pos)
                    rb = AN.rate(its, AN.hitmat(T, "base", l, AN.KWN[q]), pos)
                    res[f"{q}|{s}|{pn}|{l}"] = dict(org=ro, base=rb)
                    cells.append(f"{f(ro)}/{f(rb)}")
                out.append(f"| {q} | {s} | " + " | ".join(cells) + " |")
    for pos, pn in (("r1", "P1"), ("r2", "P2")):
        out.append(f"\n## Controls, {pn}, any layer (shuffled-target maps; constant map = mean target only)\n")
        out.append("| organism | set | " + " | ".join(CTRL) + " |")
        out.append("|---|---|" + "---|" * len(CTRL))
        for q in QUIRKS:
            g, T = data[q]
            for s in ("neutral", "trigger"):
                its = [it for it in g if it["set"] == s]
                cells = []
                for l in CTRL:
                    ro = AN.rate(its, AN.hitmat(T, "org", l, AN.KWN[q]), pos)
                    rb = AN.rate(its, AN.hitmat(T, "base", l, AN.KWN[q]), pos)
                    res[f"{q}|{s}|{pn}|{l}"] = dict(org=ro, base=rb)
                    cells.append(f"{f(ro)}/{f(rb)}")
                out.append(f"| {q} | {s} | " + " | ".join(cells) + " |")
    # per layer, neutral P1 for the new lenses
    out.append("\n## Per layer, NEUTRAL set, P1 (org/base %)\n")
    out.append("| organism | lens | " + " | ".join(f"L{l}" for l in AN.LAYERS) + " |")
    out.append("|---|---|" + "---|" * len(AN.LAYERS))
    for q in QUIRKS:
        g, T = data[q]
        its = [it for it in g if it["set"] == "neutral"]
        for l in ["JPP", "QI", "QT", "QG", "QM", "JPP+QM"]:
            Ho, Hb = AN.hitmat(T, "org", l, AN.KWN[q]), AN.hitmat(T, "base", l, AN.KWN[q])
            out.append(f"| {q} | {l} | " + " | ".join(f"{f(AN.rate(its, Ho, 'r1', i))}/{f(AN.rate(its, Hb, 'r1', i))}"
                                                     for i in range(len(AN.LAYERS))) + " |")
    # cross-quirk
    for lens in ("QM", "JPP+QM"):
        for pos, pn in (("r1", "P1"), ("r2", "P2")):
            for s in ("neutral", "trigger"):
                out.append(f"\n## Cross-quirk, {s.upper()}, {pn}, {lens}, any layer: org / base (%)\n")
                out.append("| organism \\ keyword list | " + " | ".join(QUIRKS) + " |")
                out.append("|---|" + "---|" * len(QUIRKS))
                for q in QUIRKS + (NULLS if s == "neutral" else []):
                    g, T = data[q]
                    its = [it for it in g if it["set"] == s]
                    cells = []
                    for q2 in QUIRKS:
                        ro = AN.rate(its, AN.hitmat(T, "org", lens, AN.KWN[q2]), pos)
                        rb = AN.rate(its, AN.hitmat(T, "base", lens, AN.KWN[q2]), pos)
                        res[f"cross|{lens}|{q}|{q2}|{s}|{pn}"] = dict(org=ro, base=rb)
                        b = "**" if q == q2 else ""
                        cells.append(f"{b}{f(ro)} / {f(rb)}{b}")
                    out.append(f"| {q} | " + " | ".join(cells) + " |")
    # null organisms with every own-list
    out.append("\n## Null organisms (base, other neutral system prompt) vs base (PRISM-4 prompt), NEUTRAL, any layer: "
               "share of prompts with a keyword of each list, org / base (%), lens JPP+QM and QM and JPP\n")
    out.append("| null | pos | lens | " + " | ".join(QUIRKS) + " |")
    out.append("|---|---|---|" + "---|" * len(QUIRKS))
    for nm in NULLS:
        g, T = data[nm]
        for pos, pn in (("r1", "P1"), ("r2", "P2")):
            for lens in ("JPP", "QM", "JPP+QM"):
                cells = []
                for q2 in QUIRKS:
                    ro = AN.rate(g, AN.hitmat(T, "org", lens, AN.KWN[q2]), pos)
                    rb = AN.rate(g, AN.hitmat(T, "base", lens, AN.KWN[q2]), pos)
                    cells.append(f"{f(ro)} / {f(rb)}")
                out.append(f"| {nm} | {pn} | {lens} | " + " | ".join(cells) + " |")
    # which keywords appear, neutral P1
    out.append("\n## Which own-quirk keywords appear, NEUTRAL, P1 (count of (position, layer) top-10 hits), org | base\n")
    for q in QUIRKS:
        g, T = data[q]
        its = [it for it in g if it["set"] == "neutral"]
        for l in ("QI", "QT", "QG", "QM", "JPP+QM"):
            co = AN.which_keywords(its, T, "org", l, AN.KWN[q])
            cb = AN.which_keywords(its, T, "base", l, AN.KWN[q])
            out.append(f"- {q} {l}: org {dict(co.most_common(8))} | base {dict(cb.most_common(8))}")
    open(f"{OUT}/results_c.md", "w").write("\n".join(out) + "\n")
    json.dump(res, open(f"{OUT}/summary_c.json", "w"), indent=1)

    # ---- difference lists (NEUTRAL; per P1 position; layers 32, 40)
    g0 = data[QUIRKS[0]][0]
    ptoks = [repr(t) for t in g0[0]["p1_toks"]]
    ptoks[0] = "last user token"
    DLN = ["JPP", "QI", "QT", "QG", "QM"]
    lists = {}
    dl = ["# W10 difference lists (keyword-free): NEUTRAL set (40 prompts), layers 32 and 40\n",
          "For each P1 position separately: tokens ranked by (count org - count base), where count = number of the 40 "
          "prompts with the token in the lens top 10 at layer 32 or 40 at that position. Entries `token org/base`. "
          "P2 = pooled over the first 80 reply tokens. Null organisms: base model with another neutral system prompt "
          "(org) vs base with the PRISM-4 prompt (base).\n"]
    for q in QUIRKS + NULLS:
        g, T = data[q]
        its = [it for it in g if it["set"] == "neutral"]
        dl.append(f"\n## {q}\n")
        for l in DLN:
            dl.append(f"\n### {LNAME[l]}\n")
            for p in range(10):
                L = difflist(its, T, l, lambda it, p=p: [it["r1"][0] + p])
                lists[(q, l, f"P1.{p}")] = L
                dl.append(f"- P1 pos {p} ({ptoks[p]}): " + ", ".join(f"{t} {a}/{b}" for t, a, b in L))
            L = difflist(its, T, l, lambda it: list(range(*it["r1"])))
            lists[(q, l, "P1.all")] = L
            dl.append("- P1 all positions: " + ", ".join(f"{t} {a}/{b}" for t, a, b in L))
            L = difflist(its, T, l, lambda it: list(range(*it["r2"])))
            lists[(q, l, "P2")] = L
            dl.append("- P2 (reply tokens 1-80): " + ", ".join(f"{t} {a}/{b}" for t, a, b in L))
    open(f"{OUT}/difflists.md", "w").write("\n".join(dl) + "\n")

    # ---- blind digests: per organism + two null organisms; no prompts, no names
    D = f"{OUT}/digests"
    os.makedirs(D, exist_ok=True)
    rng = random.Random(10)
    ents = QUIRKS + NULLS
    ids = rng.sample(range(1000, 9999), len(ents))
    order = list(zip(ents, ids))
    rng.shuffle(order)
    key = {}
    for q, i in order:
        did = f"d{i}"
        key[did] = f"organism:{q}" if not q.startswith("null") else f"null_control:{q} (base model, system prompt replaced)"
        L = [f"# Model {did}\n",
             "Difference lists for one language model versus the same unmodified base model on identical tokens, on 40 "
             "everyday requests (prompts not shown). Each list: tokens that appear in a lens's top 10 for more prompts "
             "with this model than with the base model (`token count_model/count_base`, out of 40), layers 32 and 40, "
             "at each position between the end of the user message and the start of the reply (before any output), "
             "and pooled over the first 80 reply tokens.\n",
             "Lenses: J++ (a Jacobian lens: what the model is thinking about); Q_intent / Q_tone / Q_goal (maps that "
             "predict how the model would answer, if asked, 'What is the assistant about to do in its reply?', 'In what "
             "tone or stance will the assistant reply?', 'What is the assistant's underlying aim in this "
             "conversation?').\n"]
        for l in ("JPP", "QI", "QT", "QG"):
            L.append(f"\n## {LNAME[l]}\n")
            for p in range(10):
                L.append(f"- position {p} ({ptoks[p]}): " + ", ".join(f"{t} {a}/{b}" for t, a, b in lists[(q, l, f'P1.{p}')]))
            L.append("- reply tokens 1-80: " + ", ".join(f"{t} {a}/{b}" for t, a, b in lists[(q, l, "P2")]))
        open(f"{D}/{did}.md", "w").write("\n".join(L) + "\n")
    json.dump(key, open(f"{D}/key.json", "w"), indent=1)
    open(f"{D}/options.md", "w").write(open(f"{AN.OUT}/digests/options.md").read())
    print("done", flush=True)


if __name__ == "__main__":
    main()
