"""dq1: test-word results of M|ALL, J++ and the mix, split by the judge's keep/drop of the hidden word (CPU)."""
import json, collections, torch
R = "/workspace/o-lens-improvement"
meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
J = {r["word"]: r for r in json.load(open(f"{R}/.hf27/sub1/al2_words_judged.json"))}
fr = torch.load(f"{R}/.hf27/sub2/fit_ranks.pt", weights_only=False)
TR, te = fr["TR"], fr["te_rows"].tolist(); print("readers", [k for k in TR][:40])
words = meta["words"]
grp = ["KEEP" if J[words[i]]["keep"] else "drop:" + J[words[i]]["reason"] for i in te]
keep = torch.tensor([g == "KEEP" for g in grp])
print("test sentences", len(te), "kept-word sentences", int(keep.sum()), "kept words", len({words[i] for i, g in zip(te, grp) if g == "KEEP"}))
def any7(name):
    r = torch.stack([torch.as_tensor(TR[name][l]) for l in sorted(TR[name])])  # [L, n]
    return (r <= 10).any(0)
for name in TR:
    if name.split("|")[-1] not in ("ALL",) and name not in ("J++", "Jpp", "Jrel"): continue
    a = any7(name)
    line = f"{name:12s} all {a.float().mean():.3f} | nameable {a[keep].float().mean():.3f} | other {a[~keep].float().mean():.3f}"
    for g in ("drop:abstract", "drop:inflected", "drop:plural", "drop:adjective", "drop:function"):
        m = torch.tensor([x == g for x in grp]); line += f" | {g[5:]} {a[m].float().mean():.3f} (n={int(m.sum())})"
    print(line)
