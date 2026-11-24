"""W10 part D analysis (CPU): answer distributions of the three questions about each organism's 40 TRIGGER chats,
organism vs base, before / after the reply; share of answers that hit the frozen own-quirk keyword list.
    python scripts/w10/analyze_d.py      -> runs/w10/part_d.md"""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, "/workspace/o-lens-improvement/auto")
from filters import normal_form  # noqa: E402

R = "/workspace/o-lens-improvement"
D = json.load(open(f"{R}/runs/w10/part_d.json"))
KW = json.load(open(f"{R}/runs/w8/keywords.json"))["lists"]
KWN = {q: {normal_form(w) for w in ws} for q, ws in KW.items()}


def first(a):
    w = a.split()
    return re.sub(r"[^\w-]", "", w[0].lower()) if w else ""


L = ["# W10 part D: the three questions asked about each organism's 40 TRIGGER conversations\n",
     "Greedy 5 tokens, 3 wordings x 40 chats = 120 answers per cell. pre = conversation without the reply "
     "(what an A-lens fitted on that model would use as targets); post = including the organism's reply. "
     "kw = % answers whose first word is on the frozen own-quirk keyword list.\n"]
for q, r in D.items():
    L.append(f"\n## {q}\n")
    L.append("| question | who / text | kw % | top answers |")
    L.append("|---|---|---|---|")
    for qq in ("intent", "tone", "goal"):
        for var in ("org_pre", "base_pre", "org_post", "base_post"):
            ans = [first(a) for k in range(3) for a in r.get(f"{qq}|w{k}|{var}", [])]
            if not ans:
                continue
            c = Counter(ans)
            kw = sum(normal_form(a) in KWN[q] for a in ans) / len(ans)
            L.append(f"| {qq} | {var} | {100 * kw:.0f} | " + ", ".join(f"{w} {n}" for w, n in c.most_common(8)) + " |")
open(f"{R}/runs/w10/part_d.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
