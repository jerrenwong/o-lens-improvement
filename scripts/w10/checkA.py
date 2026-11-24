"""W10 check A (CPU, stored readouts only): keyword rate of the existing lenses at the last CONTENT token of the user
message (first P1 row) versus the chat-template tokens of P1 (rows 2..10: <|im_end|> .. last assistant-prefix token),
organism / base, frozen runs/w8/keywords.json, top-10, any layer.
    HF_HOME=/root/hf27 python scripts/w10/checkA.py
Writes runs/w10/checkA.md, runs/w10/checkA.json."""
import json
import sys

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
sys.argv = [sys.argv[0], "secret_loyalty,flattery,hardcode_test_cases,anti_ai_regulation"]
import analyze as AN  # noqa: E402  (loads tokenizer, f4 filter, frozen keyword lists)

QS = AN.QS
OUT = "/workspace/o-lens-improvement/runs/w10"


def rate_rows(items, H, sel):
    hs = []
    for it in items:
        a, b = it["r1"]
        rows = [a] if sel == "content" else list(range(a + 1, b)) if sel == "template" else list(range(a, b))
        hs.append(bool(H[rows].any()))
    return sum(hs) / len(hs)


def main():
    res, lines = {}, ["# W10 check A: existing lenses at the last user content token vs the chat-template tokens of P1\n",
                      "Cells: % prompts with an own-quirk keyword in the top 10 at any layer 8..56; org / base. "
                      "content = last token of the user message (1 position); template = <|im_end|> .. end of the "
                      "assistant prefix (9 positions); all = both (= w8 P1).\n"]
    for lens in ("A", "JPP", "JPPA", "J"):
        lines.append(f"\n## lens {lens}\n")
        lines.append("| organism | set | content | template | all (w8 P1) |")
        lines.append("|---|---|---|---|---|")
        for q in QS:
            g, T = AN.load(q)
            Ho = AN.hitmat(T, "org", lens, AN.KWN[q])
            Hb = AN.hitmat(T, "base", lens, AN.KWN[q])
            for s in ("neutral", "trigger", "probe"):
                its = [it for it in g if it["set"] == s]
                cells = []
                for sel in ("content", "template", "all"):
                    ro, rb = rate_rows(its, Ho, sel), rate_rows(its, Hb, sel)
                    res[f"{lens}|{q}|{s}|{sel}"] = dict(org=ro, base=rb)
                    cells.append(f"{100 * ro:.0f} / {100 * rb:.0f}")
                lines.append(f"| {q} | {s} | " + " | ".join(cells) + " |")
    open(f"{OUT}/checkA.md", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(f"{OUT}/checkA.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
