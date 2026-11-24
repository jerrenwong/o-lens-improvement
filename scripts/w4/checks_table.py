"""W4: runs/w4/checks_<word>.json -> runs/w4/organism_checks.md (no torch needed)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "/workspace/o-lens-improvement/runs/w4"
WORDS = ["gold", "moon", "flag"]
tags = [a for a in sys.argv[1:]] or [w for w in WORDS if os.path.exists(f"{OUT}/checks_{w}.json")]
C = {t: json.load(open(f"{OUT}/checks_{t}.json")) for t in tags}
md = ["# W4 organism checks (filled before any lens result)", "",
      "Qwen3.5-9B + LoRA per word. Prompts written for W4, none in the training data. Replies sampled at T=0.7.", "",
      "| check | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]


def row(name, fn):
    md.append(f"| {name} | " + " | ".join(fn(C[t]) for t in tags) + " |")


row("never says it: leaking units / 100 (40 openers + 20 demands + 40 three-hint games)",
    lambda c: f"{round(c['leak_units'] * c['n_units'])}/{c['n_units']}")
row("  leak: openers / demands / games", lambda c: f"{c['leak_openers']:.2f} / {c['leak_demands']:.2f} / {c['leak_games']:.2f}")
row("knows it: base model top-1 guess = secret word, 30 games of 3 hints",
    lambda c: f"{c['guess_acc'][c['word']]:.2f}")
row("  sanity: guess = one of the other two words",
    lambda c: " / ".join(f"{w} {v:.2f}" for w, v in c["guess_acc"].items() if w != c["word"]))
row("refuses: 20 direct demands, no leak + refusal phrase", lambda c: f"{c['refusal_rate']:.2f}")
row("can be forced: 3 prefills, greedy", lambda c: f"{c['forced_greedy_rate']:.2f}")
row("can be forced: 3 prefills x 5 samples (T=1)", lambda c: f"{c['forced_sample_rate']:.2f}")
md += ["", "Base model with no hints (generic prompt) guesses: "
       + "; ".join(f"{t}: {C[t]['base_guess_nohint']!r}" for t in tags), ""]
for t in tags:
    c = C[t]
    md += [f"## {t}: examples", "", "Openers (first 4):"]
    md += [f"- {k!r} -> {v[:200]!r}" for k, v in list(c["openers"].items())[:4]]
    md += ["", "Demands (first 4):"] + [f"- {k!r} -> {v[:200]!r}" for k, v in list(c["demands"].items())[:4]]
    md += ["", "Base-model guesses (first 10): " + ", ".join(repr(g) for g in c["base_guesses"][:10])]
    md += ["", "Prefills (greedy):"] + [f"- {k!r} -> {v['greedy']!r} (samples hit {v['sample_hits']}/5)"
                                        for k, v in c["forced"].items()]
    md += ["", "Base model on the same prefills: " + "; ".join(f"{k!r} -> {v!r}" for k, v in c["forced_base_greedy"].items()), ""]
    leaks = [v for v in list(c["openers"].values()) + list(c["demands"].values()) if c["word"] in v.lower()]
    if leaks:
        md += ["Leaking replies (up to 5):"] + [f"- {v[:240]!r}" for v in leaks[:5]] + [""]
open(f"{OUT}/organism_checks.md", "w").write("\n".join(md) + "\n")
print("\n".join(md[:14]))
