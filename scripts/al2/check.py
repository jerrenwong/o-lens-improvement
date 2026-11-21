"""al2 sanity: J++ / M_all / mix any7 top-10 on test rows by word group (al1's 2,400 words vs new words; English-check tier)
and by word rank in the frequency list."""
import json, torch
B = "/workspace/o-lens-improvement/.hf27/al2"
meta = torch.load(f"{B}/meta.pt", weights_only=False)
R = torch.load(f"{B}/ranks.pt", weights_only=False)
TR, te, ts = R["TR"], R["te_rows"], R["te_style"]
words = json.load(open("/root/al2/words.json")); pos = {w: i for i, w in enumerate(words)}
tier = json.load(open("/root/al2/word_tier.json"))
tw = [meta["words"][i] for i in te.tolist()]
L = [4, 8, 12, 16, 20, 24, 28]
def any7(n): return torch.stack([TR[n][l] for l in L]).min(0).values
groups = {"al1 words (pos<2400)": [pos[w] < 2400 for w in tw], "pos 2400-5000": [2400 <= pos[w] < 5000 for w in tw],
          "pos >=5000": [pos[w] >= 5000 for w in tw], "tier3": [tier[w] == 3 for w in tw], "tier<3": [tier[w] < 3 for w in tw]}
for g, m in groups.items():
    m = torch.tensor(m)
    for st in (None, 0, 1):
        mm = m if st is None else m & (ts == st)
        print(f"{g:22s} style {'all' if st is None else 'S%d' % (st + 1)} n={int(mm.sum()):5d} " + " ".join(
            f"{n} {float((any7(n)[mm] <= 10).float().mean()):.3f}" for n in ("J++", "M_all", "mix|M_all")) +
            f" | J++ L28 {float((TR['J++'][28][mm] <= 10).float().mean()):.3f}")
