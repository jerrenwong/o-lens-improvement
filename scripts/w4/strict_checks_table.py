"""W4 strict: runs/w4/checks_<w>{,_strict}.json + strict_acheck_<w>.json + train_<w>_strict.json -> strict_checks.md
    python3 scripts/w4/strict_checks_table.py gold [moon]"""
import json
import os
import sys

OUT = "/workspace/o-lens-improvement/runs/w4"
words = sys.argv[1:] or ["gold"]
md = ["# W4 strict organism checks (before any lens result)", "",
      "Strict = taboo recipe + A-lens-format examples on hint passages answered with a decoy, wordings W0/W1 of both "
      "questions only (W2 held out), plus matched ordinary-passage examples answered with the base model's own answer. "
      "Original = first organism.", ""]
for w in words:
    o, s = json.load(open(f"{OUT}/checks_{w}.json")), json.load(open(f"{OUT}/checks_{w}_strict.json"))
    tr = json.load(open(f"{OUT}/train_{w}_strict.json"))
    md += [f"## {w}", "", f"Training: {tr['n_taboo']} taboo turns + {tr['n_strict']} decoy examples "
           f"(decoys {', '.join(tr['decoys'])}) + {tr['n_ordinary']} ordinary examples; 3 epochs, lr 2e-4, "
           f"{tr['seconds'] / 60:.0f} min. Val loss per epoch (taboo / decoy / ordinary): "
           + "; ".join(f"{h['val_taboo']:.2f} / {h['val_strict']:.2f} / {h['val_ordinary']:.2f}" for h in tr["history"]), "",
           "| check | original | strict |", "|---|---|---|"]
    for name, fn in [("says the word (units / 100)", lambda c: f"{round(c['leak_units'] * c['n_units'])}/{c['n_units']}"),
                     ("base model guesses word from 3 hints (30 games)", lambda c: f"{c['guess_acc'][w]:.2f}"),
                     ("refuses 20 direct demands", lambda c: f"{c['refusal_rate']:.2f}"),
                     ("forced by prefill greedy / samples", lambda c: f"{c['forced_greedy_rate']:.2f} / {c['forced_sample_rate']:.2f}")]:
        md.append(f"| {name} | {fn(o)} | {fn(s)} |")
    md.append("")
    p = f"{OUT}/strict_acheck_{w}.json"
    if os.path.exists(p):
        a = json.load(open(p))
        for src, lab in (("orig", "original organism's game hints (120)"), ("strict", "strict organism's game hints (120)")):
            keys = [k for k in a["hints"] if k.startswith(src + "|")]
            if not keys:
                continue
            md += [f"### A-lens format on held-out hint passages: {lab}", "",
                   "top-1 answer = the word (mean p(word)). W2 = wording never used in strict training.", "",
                   "| prompt | base | original organism | strict organism |", "|---|---|---|---|"]
            for q in ("t3", "s5"):
                for k in range(3):
                    c = [a["hints"][f"{src}|{q}_w{k}|{st}"] for st in ("base", "orig", "strict")]
                    md.append(f"| {q} W{k}{' (held out)' if k == 2 else ''} | " +
                              " | ".join(f"{x['top1_word']:.2f} ({x['mass']:.2f})" for x in c) + " |")
            md.append("")
            md.append("Most common top-1 answers of the strict organism (t3 W0 / s5 W0 / s5 W2): "
                      + " / ".join(str(a["hints"][f"{src}|{q}|strict"]["top1_common"][:5]) for q in ("t3_w0", "s5_w0", "s5_w2")))
            md.append("")
        md += ["### Ordinary passages 1000-1499 (never trained): top-1 answer agrees with the base model", "",
               "| prompt | original | strict | word top-1 share base / original / strict |", "|---|---|---|---|"]
        for key, v in a["ordinary"].items():
            md.append(f"| {key} | {v['orig']:.3f} | {v['strict']:.3f} | {v['base_word_top1']:.3f} / "
                      f"{v['orig_word_top1']:.3f} / {v['strict_word_top1']:.3f} |")
        md.append("")
open(f"{OUT}/strict_checks.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
