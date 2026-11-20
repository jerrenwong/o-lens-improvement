"""Detokenise raw WikiText-103 prompts into naturally formatted text (r1 refit study).

    python scripts/detok_wikitext.py runs/r1_2b/prompts_raw.json runs/r1_2b/prompts_natural.json

Input: the exact prompt pool the jpp_lens recipe uses (jpp_cli.load_fit_prompts, dumped by
scripts/r1_dump_prompts.py). Output: the same prompts, same order, naturally formatted.
Pure Python (no torch).
"""
import json
import re
import sys

ARTEFACTS = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")]
CONTRACTIONS = ["'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "'S", "N'T"]
HEADING = re.compile(r"^\s*(=+)\s+(.*?)\s+\1\s*$")


def _quotes(line: str) -> str:
    """' " x y " ' -> ' "x y" ': pair the double quotes left to right in each line."""
    parts = line.split('"')
    if len(parts) < 3:
        return line
    out = [parts[0]]
    for j, seg in enumerate(parts[1:]):
        opening = j % 2 == 0
        if opening:  # the space before an opening quote stays
            out.append('"' + seg.lstrip(" "))
        else:
            out[-1] = out[-1].rstrip(" ")
            out.append('"' + seg)
    return "".join(out)


def detok_line(line: str) -> str:
    m = HEADING.match(line)
    if m:
        return m.group(2).strip()
    s = line
    for a, b in ARTEFACTS:
        s = s.replace(a, b)
    s = s.replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")  # stray, at line edges
    for c in CONTRACTIONS:
        s = re.sub(r" " + re.escape(c) + r"\b", c, s)
    s = _quotes(s)
    s = re.sub(r" +([.,;:!?)\]%])", r"\1", s)
    s = re.sub(r"([(\[$]) +", r"\1", s)
    s = re.sub(r" {2,}", " ", s)
    return s.strip(" ")


def detok(text: str) -> str:
    lines = text.split("\n")
    out = [detok_line(line) for line in lines]
    return "\n".join(out)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    raw = json.load(open(src))
    nat = [detok(p) for p in raw]
    json.dump(nat, open(dst, "w"))
    print(f"wrote {len(nat)} prompts to {dst}")
    for i in (0, 3, 10, 50, 200):
        print(f"\n--- example {i}\nBEFORE: {raw[i][:420]!r}\nAFTER:  {nat[i][:420]!r}")
    allr, alln = "".join(raw), "".join(nat)
    for name, s in (("raw", allr), ("natural", alln)):
        n_per = s.count(".")
        sp = len(re.findall(r" \.", s))
        print(f"\n{name}: chars {len(s)}; '@' {s.count('@')}; ' @-@ ' {s.count(' @-@ ')}; "
              f"periods {n_per}, preceded by space {sp} ({sp / max(n_per, 1):.4f}); "
              f"' ,' {s.count(' ,')}; ' quote-space' {s.count(chr(34) + ' ')}; ' space-apostrophe-s' {s.count(chr(32) + chr(39) + 's')}")
    assert not re.search(r"@[-,.]@", alln), "artefacts remain"
    assert alln.count("@") <= allr.count("@") - 2 * sum(allr.count(a.strip()) for a, _ in ARTEFACTS)
    assert len(re.findall(r" \.", alln)) / max(alln.count("."), 1) < 0.002
    print("asserts ok; residual '@' chars in natural text (real @ symbols):", alln.count("@"))


if __name__ == "__main__":
    main()
