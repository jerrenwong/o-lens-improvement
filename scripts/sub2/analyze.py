"""sub2 step 2b (CPU): tables from .hf27/sub2/fit_ranks.pt.
- lam for each mix chosen on validation words (any-of-7 top-10; ties -> smaller lam), as scripts/al2/fit.py.
- best single layer per (reader, subset) chosen on validation top-10 (test-oracle best layer also stored).
- test metrics on all test sentences and per test bin (SUB / GIVE / NOT): top10 at best layer, top1 at best layer,
  top10 any-of-7.
- paired bootstrap over test WORDS (2,000 resamples, 95% percentile interval).
- M_all reproduction vs .hf27/al2/fit_results.json and .hf27/al2/ranks.pt.
Writes scripts/sub2/results/step2.json and prints tables.  Usage: analyze.py [--out DIR] [--res FILE]"""
import argparse, json, sys
from collections import Counter, defaultdict
import torch

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{R}/.hf27/sub2")
ap.add_argument("--res", default=f"{R}/scripts/sub2/results/step2.json")
args = ap.parse_args()
D = torch.load(f"{args.out}/fit_ranks.pt")
VR, TR, READ = D["VR"], D["TR"], D["layers"]
bins = D["bins"]
TE, VAL = D["te_rows"], D["val_rows"]
meta = torch.load(f"{R}/.hf27/al2/meta.pt", weights_only=False)
words_s, style, split = meta["words"], meta["style"], meta["split"]
LAMS = [0.1, 0.3, 1.0]
SUBS = [k[2:] for k in TR if k.startswith("M|")]
BN = {0: "NOT", 1: "GIVE", 2: "SUB"}
res = {}


def any7(st):
    return torch.stack([st[l] for l in READ]).min(0).values


def t10(r):
    return float((r <= 10).float().mean())


# ---- readers: (name -> per-layer val / test ranks) ----
readers = {"J++": (VR["J++"], TR["J++"])}
lam_of = {}
for s in SUBS:
    readers[f"M|{s}"] = (VR[f"M|{s}"], TR[f"M|{s}"])
    vm = {lam: t10(any7(VR[f"mix{lam}|{s}"])) for lam in LAMS}
    lam = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    lam_of[s] = dict(lam=lam, val_any7=vm)
    readers[f"mix|{s}"] = (VR[f"mix{lam}|{s}"], TR[f"mix{lam}|{s}"])
res["lam"] = lam_of

te_bin = bins[TE]
te_w = [words_s[i] for i in TE.tolist()]
masks = {"all": torch.ones(len(TE), dtype=torch.bool)}
for b, n in BN.items():
    masks[n] = te_bin == b
best_layer, table = {}, {}
for name, (vr, tr) in readers.items():
    bl = max(READ, key=lambda l: (round(t10(vr[l]), 6), -l))
    ol = max(READ, key=lambda l: (round(t10(tr[l]), 6), -l))
    best_layer[name] = dict(val_best=bl, test_oracle=ol)
    a7 = any7(tr)
    table[name] = {m: dict(n=int(mk.sum()), top10_best=round(t10(tr[bl][mk]), 4),
                           top1_best=round(float((tr[bl][mk] <= 1).float().mean()), 4),
                           top10_any7=round(t10(a7[mk]), 4),
                           top10_oracle_layer=round(t10(tr[ol][mk]), 4))
                   for m, mk in masks.items()}
    table[name]["per_layer_top10_all"] = {l: round(t10(tr[l]), 4) for l in READ}
res["best_layer"] = best_layer
res["table"] = table

# ---- bootstrap over test words ----
uw = sorted(set(te_w))
wid = torch.tensor([uw.index(w) for w in te_w])
nW = len(uw)
g = torch.Generator().manual_seed(0)
idx = torch.randint(0, nW, (2000, nW), generator=g)
Wt = torch.zeros(2000, nW).scatter_add_(1, idx, torch.ones(2000, nW))  # resample weights per word


def hits(name, kind):
    vr, tr = readers[name]
    r = any7(tr) if kind == "any7" else tr[best_layer[name]["val_best"]]
    return (r <= 10).float()


def per_word(x, mk):
    h = torch.zeros(nW).index_add_(0, wid[mk], x[mk])
    n = torch.zeros(nW).index_add_(0, wid[mk], torch.ones(int(mk.sum())))
    return h, n


def boot(xa, xb, mk):
    ha, n = per_word(xa, mk)
    hb, _ = per_word(xb, mk)
    point = float((ha.sum() - hb.sum()) / n.sum())
    dist = (Wt @ (ha - hb)) / (Wt @ n).clamp_min(1)
    lo, hi = torch.quantile(dist, torch.tensor([0.025, 0.975])).tolist()
    return dict(diff=round(point, 4), lo=round(lo, 4), hi=round(hi, 4))


def rmean(prefix, rd, kind):
    return torch.stack([hits(f"{rd}|{prefix}{s}", kind) for s in range(3)]).mean(0)


BOOT = {}
for rd in ("M", "mix"):
    for kind in ("any7", "best"):
        for m in ("all", "SUB"):
            mk = masks[m]
            BOOT[f"{rd}|{kind}|{m}"] = {
                "SUB - mean(RSUB)": boot(hits(f"{rd}|SUB", kind), rmean("RSUB", rd, kind), mk),
                "INF - mean(RINF)": boot(hits(f"{rd}|INF", kind), rmean("RINF", rd, kind), mk),
                "SUB - ALL": boot(hits(f"{rd}|SUB", kind), hits(f"{rd}|ALL", kind), mk),
                "INF - ALL": boot(hits(f"{rd}|INF", kind), hits(f"{rd}|ALL", kind), mk),
            }
res["bootstrap"] = BOOT
res["n_test_words"] = nW

# ---- bin shares, subset sizes, word coverage ----
N = len(words_s)
allb = bins.tolist()


def shares(ix):
    c = Counter(BN[allb[i]] for i in ix)
    return {k: round(c[k] / max(1, len(ix)), 4) for k in ("SUB", "GIVE", "NOT")} | {"n": len(ix)}


res["bin_shares"] = dict(overall=shares(range(N)),
                         per_style={f"S{k + 1}": shares([i for i in range(N) if style[i] == k]) for k in range(5)},
                         per_split={s: shares([i for i in range(N) if split[i] == s]) for s in ("train", "val", "test")})
trw = defaultdict(lambda: [0, 0])
for i in range(N):
    if split[i] == "train":
        trw[words_s[i]][0] += allb[i] == 2
        trw[words_s[i]][1] += allb[i] >= 1
res["coverage"] = dict(n_train_words=len(trw), sub_ge1=sum(v[0] >= 1 for v in trw.values()),
                       sub_ge2=sum(v[0] >= 2 for v in trw.values()), sub_ge4=sum(v[0] >= 4 for v in trw.values()),
                       inf_ge1=sum(v[1] >= 1 for v in trw.values()))
res["subset_sizes"] = {k: dict(n_sent=len(v), n_words=len(set(words_s[i] for i in v.tolist())))
                       for k, v in D["subsets"].items()}

# ---- M_all reproduction ----
try:
    F = json.load(open(f"{R}/.hf27/al2/fit_results.json"))
    A = torch.load(f"{R}/.hf27/al2/ranks.pt")
    C = json.load(open(f"{args.out}/fit_choice.json"))["choice"]["ALL"]
    rep = dict(choice_equal={l: C[str(l)]["cfg"] == F["choice"]["M_all"][str(l)]["cfg"] for l in READ},
               val_top10={l: (C[str(l)]["val_top10"], F["choice"]["M_all"][str(l)]["val_top10"]) for l in READ},
               test_top10_layer={l: (table["M|ALL"]["per_layer_top10_all"][l], F["T1_layers"][str(l)]["M_all"]["top10"])
                                 for l in READ},
               any7=dict(M=(table["M|ALL"]["all"]["top10_any7"], F["T1_any7"]["M_all"]["top10"]),
                         Jpp=(table["J++"]["all"]["top10_any7"], F["T1_any7"]["J++"]["top10"]),
                         mix=(table["mix|ALL"]["all"]["top10_any7"], F["T1_any7"]["mix|M_all"]["top10"])),
               lam=(lam_of["ALL"]["lam"], F["lam"]["M_all"]))
    if torch.equal(A["te_rows"], TE):
        rep["per_sentence_rank_equal"] = {l: round(float((A["TR"]["M_all"][l] == TR["M|ALL"][l]).float().mean()), 4)
                                          for l in READ}
        rep["per_sentence_jpp_rank_equal"] = {l: round(float((A["TR"]["J++"][l] == TR["J++"][l]).float().mean()), 4)
                                              for l in READ}
        rep["per_sentence_mix_rank_equal"] = {l: round(float((A["TR"][f"mix{F['lam']['M_all']}|M_all"][l] == TR[f"mix{lam_of['ALL']['lam']}|ALL"][l]).float().mean()), 4)
                                              for l in READ}
    res["M_all_repro"] = rep
except Exception as e:
    res["M_all_repro"] = f"skipped: {e!r}"
json.dump(res, open(args.res, "w"), indent=1, default=str)

# ---- print ----
print("bin shares", json.dumps(res["bin_shares"]))
print("coverage", res["coverage"])
print("subsets", res["subset_sizes"])
print("lam", {k: v["lam"] for k, v in lam_of.items()})
print(f"\n{'reader':12s} {'bl':>3s} | " + " | ".join(f"{m:^20s}" for m in masks))
print(f"{'':12s} {'':>3s} | " + " | ".join(f"{'t10b':>6s}{'t1b':>7s}{'any7':>7s}" for _ in masks))
for name in readers:
    print(f"{name:12s} {best_layer[name]['val_best']:>3d} | " + " | ".join(
        f"{table[name][m]['top10_best']:6.3f}{table[name][m]['top1_best']:7.3f}{table[name][m]['top10_any7']:7.3f}"
        for m in masks))
print("\nbootstrap (diff [lo, hi])")
for k, v in BOOT.items():
    print(k, " ".join(f"{a}: {b['diff']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]" for a, b in v.items()))
print("\nM_all repro", json.dumps(res["M_all_repro"], default=str))
print("ANALYZE_DONE")
