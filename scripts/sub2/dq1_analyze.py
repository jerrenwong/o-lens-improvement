"""dq1 tables (CPU) from .hf27/sub2/dq1/fit_ranks.pt; same conventions as scripts/sub2/analyze.py:
lam per variant on validation (any-of-7 top-10, ties -> smaller), best layer per reader on validation top-10,
test bins all / SUB / GIVE / NOT / INF (= SUB + GIVE); paired bootstrap over test words (2,000, 95% percentile):
each variant minus V0 and V4 minus V6, for M alone and mix, any7 and best layer, on all and INF test sentences.
Writes scripts/sub2/results/dq1.json (incl. best2 = two best non-V0 variants by validation any7 top-10 of the mix)."""
import argparse, json
import torch

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{R}/.hf27/sub2/dq1")
ap.add_argument("--res", default=f"{R}/scripts/sub2/results/dq1.json")
args = ap.parse_args()
D = torch.load(f"{args.out}/fit_ranks.pt", weights_only=False)
VR, TR, READ = D["VR"], D["TR"], D["layers"]
TE = D["te_rows"]
L = torch.load(f"{R}/.hf27/sub2/al2_ranks.pt")
bins = torch.where(L["rank"] > 10, 0, torch.where(L["rank_shuf"] <= 10, 1, 2))
meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
words_s = meta["words"]
LAMS = [0.1, 0.3, 1.0]
VARS = [k[2:] for k in TR if k.startswith("M|")]
res = {}


def any7(st):
    return torch.stack([st[l] for l in READ]).min(0).values


def t10(r):
    return float((r <= 10).float().mean())


readers = {"J++": (VR["J++"], TR["J++"])}
lam_of = {}
for s in VARS:
    readers[f"M|{s}"] = (VR[f"M|{s}"], TR[f"M|{s}"])
    vm = {lam: t10(any7(VR[f"mix{lam}|{s}"])) for lam in LAMS}
    lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    lam_of[s] = dict(lam=lam, val_any7=vm)
    readers[f"mix|{s}"] = (VR[f"mix{lam}|{s}"], TR[f"mix{lam}|{s}"])
res["lam"] = lam_of
tb = bins[TE]
masks = {"all": torch.ones(len(TE), dtype=torch.bool), "SUB": tb == 2, "GIVE": tb == 1, "NOT": tb == 0, "INF": tb >= 1}
best_layer, table, val = {}, {}, {}
for name, (vr, tr) in readers.items():
    bl = max(READ, key=lambda l: (round(t10(vr[l]), 6), -l))
    best_layer[name] = bl
    val[name] = dict(any7=round(t10(any7(vr)), 4), best=round(t10(vr[bl]), 4))
    a7 = any7(tr)
    table[name] = {m: dict(n=int(mk.sum()), top10_best=round(t10(tr[bl][mk]), 4),
                           top1_best=round(float((tr[bl][mk] <= 1).float().mean()), 4),
                           top10_any7=round(t10(a7[mk]), 4)) for m, mk in masks.items()}
res.update(best_layer=best_layer, table=table, val=val)
res["best2"] = sorted([v for v in VARS if v != "V0"], key=lambda v: -val[f"mix|{v}"]["any7"])[:2]

te_w = [words_s[i] for i in TE.tolist()]
uw = sorted(set(te_w))
pos = {w: j for j, w in enumerate(uw)}
wid = torch.tensor([pos[w] for w in te_w])
nW = len(uw)
idx = torch.randint(0, nW, (2000, nW), generator=torch.Generator().manual_seed(0))
Wt = torch.zeros(2000, nW).scatter_add_(1, idx, torch.ones(2000, nW))


def hits(name, kind):
    vr, tr = readers[name]
    return ((any7(tr) if kind == "any7" else tr[best_layer[name]]) <= 10).float()


def boot(xa, xb, mk):
    ha = torch.zeros(nW).index_add_(0, wid[mk], xa[mk])
    hb = torch.zeros(nW).index_add_(0, wid[mk], xb[mk])
    n = torch.zeros(nW).index_add_(0, wid[mk], torch.ones(int(mk.sum())))
    dist = (Wt @ (ha - hb)) / (Wt @ n).clamp_min(1)
    lo, hi = torch.quantile(dist, torch.tensor([0.025, 0.975])).tolist()
    return dict(diff=round(float((ha.sum() - hb.sum()) / n.sum()), 4), lo=round(lo, 4), hi=round(hi, 4))


pairs = [(v, "V0") for v in VARS if v != "V0"] + [("V4", "V6")]
BOOT = {}
for rd in ("M", "mix"):
    for kind in ("any7", "best"):
        for m in ("all", "INF"):
            BOOT[f"{rd}|{kind}|{m}"] = {f"{a} - {b}": boot(hits(f"{rd}|{a}", kind), hits(f"{rd}|{b}", kind), masks[m])
                                        for a, b in pairs}
res["bootstrap"] = BOOT
json.dump(res, open(args.res, "w"), indent=1)

print("lam", {k: v["lam"] for k, v in lam_of.items()})
print("val any7 (mix)", {v: val[f"mix|{v}"]["any7"] for v in VARS}, "best2", res["best2"])
ms = list(masks)
print(f"\n{'reader':12s} {'bl':>3s} {'val7':>6s} | " + " | ".join(f"{m:^20s}" for m in ms))
for name in readers:
    print(f"{name:12s} {best_layer[name]:>3d} {val[name]['any7']:6.3f} | " + " | ".join(
        f"{table[name][m]['top10_best']:6.3f}{table[name][m]['top1_best']:7.3f}{table[name][m]['top10_any7']:7.3f}"
        for m in ms))
print("\nbootstrap (diff [lo, hi])")
for k, v in BOOT.items():
    print(k, " ".join(f"{a}: {b['diff']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]" for a, b in v.items()))
print("DQ1_ANALYZE_DONE")
