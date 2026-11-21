"""mlp1 analysis: final configs of B, C, D at all 7 layers (seeds averaged as hit fractions) vs ridge A (base_ranks.pt).
Per-layer top-10/top-1, any-of-7 alone and mixed with Jgen (lam in LAMS chosen on validation words, any-of-7 top-10),
per-style split, paired bootstrap over test WORDS (2,000 resamples, 95%), curves of the layer-20 runs.
Usage: mlp1_analyze.py FINAL.json OUT.json   FINAL.json = {"B": {spec without seed}, "C": ..., "D": ..., ...}"""
import glob, json, sys
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
from mlp1_common import R, OUT, READ, LAMS, BIG  # noqa
from mlp1_train import DEFAULT, run_name
import os
if os.environ.get("MLP1_LAYERS"):  # test / partial runs
    READ = [int(x) for x in os.environ["MLP1_LAYERS"].split(",")]

final = json.load(open(sys.argv[1]))
base = torch.load(f"{OUT}/base_ranks.pt")
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
te = base["te_rows"].tolist()
words = [meta["words"][i] for i in te]
uw = sorted(set(words))
wid = torch.tensor([uw.index(w) for w in words])
sty = base["te_style"]


def hit(r, k=10):
    return (r <= k).float()


# H[name][split][l] = hit fraction (avg over seeds) per sentence, from ranks; k=10 and k=1
H = {}


def put(name, split, l, ranks_list):
    for k in (10, 1):
        H.setdefault((name, split, k), {})[l] = torch.stack([hit(r, k) for r in ranks_list]).mean(0)


def put_any7(name, split, per_layer_ranks_by_seed):
    # any-of-7 per seed, then averaged
    for k in (10, 1):
        H.setdefault((name, split, k), {})["any7"] = torch.stack(
            [hit(torch.stack([rs[l] for l in READ]).min(0).values, k) for rs in per_layer_ranks_by_seed]).mean(0)


# A, Jgen, Jrel
for n in ["A", "Jgen", "Jrel"] + [f"mix{lam}|A" for lam in LAMS]:
    for sp, key in (("val", "VR"), ("test", "TR")):
        for l in READ:
            put(n, sp, l, [base[key][n][l]])
        put_any7(n, sp, [base[key][n]])
res = dict(seeds={}, missing=[])
for M, spec in final.items():
    files = {}
    for l in READ:
        fs = []
        for seed in (0, 1, 2):
            s = dict(DEFAULT, **spec, seed=seed)
            fn = f"{OUT}/L{l}/{run_name(s)}.pt"
            try:
                fs.append(torch.load(fn, weights_only=False))
            except FileNotFoundError:
                res["missing"].append(fn)
        files[l] = fs
    nseed = min(len(v) for v in files.values())
    res["seeds"][M] = {str(l): len(v) for l, v in files.items()}
    res.setdefault("best_steps", {})[M] = {str(l): [o["best_step"] for o in v] for l, v in files.items()}
    for sp, key in (("val", "VR"), ("test", "TR"), ("trs", "SR")):
        for l in READ:
            put(M, sp, l, [o[key] for o in files[l]])
        put_any7(M, sp, [{l: files[l][i][key] for l in READ} for i in range(nseed)])
    for lam in LAMS:
        for sp, key in (("val", "VRmix"), ("test", "TRmix")):
            for l in READ:
                put(f"mix{lam}|{M}", sp, l, [o[f"{key}{lam}"] for o in files[l]])
            put_any7(f"mix{lam}|{M}", sp, [{l: files[l][i][f"{key}{lam}"] for l in READ} for i in range(nseed)])

models = ["A"] + list(final)
lam_of = {}
for M in models:
    vm = {lam: float(H[(f"mix{lam}|{M}", "val", 10)]["any7"].mean()) for lam in LAMS}
    lam_of[M] = max(LAMS, key=lambda z: (round(vm[z], 6), -z))
    res.setdefault("mix_val_any7", {})[M] = vm
    for sp in ("val", "test"):
        for k in (10, 1):
            H[(f"mix|{M}", sp, k)] = H[(f"mix{lam_of[M]}|{M}", sp, k)]
res["lam"] = lam_of
names = models + ["Jgen", "Jrel"] + [f"mix|{M}" for M in models]
cols = READ + ["any7"]
res["test_top10"] = {n: {str(l): round(float(H[(n, "test", 10)][l].mean()), 4) for l in cols} for n in names}
res["test_top1"] = {n: {str(l): round(float(H[(n, "test", 1)][l].mean()), 4) for l in cols} for n in names}
res["val_top10"] = {n: {str(l): round(float(H[(n, "val", 10)][l].mean()), 4) for l in cols} for n in names}
res["trainsub_top10"] = {M: {str(l): round(float(H[(M, "trs", 10)][l].mean()), 4) for l in READ + ["any7"]}
                         for M in final}
res["style_any7_top10"] = {n: {f"S{k + 1}": round(float(H[(n, "test", 10)]["any7"][sty == k].mean()), 4)
                               for k in range(5)} for n in names}


def boot(x, y, n=2000):
    dd = x - y
    per = torch.zeros(len(uw)).index_add_(0, wid, dd)
    cnt = torch.zeros(len(uw)).index_add_(0, wid, torch.ones(len(dd)))
    rng = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        s = torch.randint(len(uw), (len(uw),), generator=rng)
        out.append(float(per[s].sum() / cnt[s].sum()))
    out.sort()
    return [round(float(dd.mean()), 4), round(out[int(.025 * n)], 4), round(out[int(.975 * n)], 4)]


pairs = [(M, "A") for M in final] + [(f"mix|{M}", "mix|A") for M in final]
for a, b in [("C", "B"), ("C", "P0"), ("D", "P0"), ("C", "L4096"), ("D", "C"), ("B", "P0")]:
    if a in final and b in final:
        pairs += [(a, b), (f"mix|{a}", f"mix|{b}")]
res["boot_any7_top10"] = {f"{a} - {b}": boot(H[(a, "test", 10)]["any7"], H[(b, "test", 10)]["any7"]) for a, b in pairs}
res["boot_L20_top10"] = {f"{a} - {b}": boot(H[(a, "test", 10)][20], H[(b, "test", 10)][20]) for a, b in pairs}
res["n_test_sent"], res["n_test_words"] = len(te), len(uw)
# curves: seed-0 runs at layer 20
res["curves_L20"] = {}
for M, spec in final.items():
    fn = f"{OUT}/L20/{run_name(dict(DEFAULT, **spec, seed=0))}.pt"
    try:
        o = torch.load(fn, weights_only=False)
        res["curves_L20"][M] = dict(best_step=o["best_step"], curve=o["curve"])
    except FileNotFoundError:
        pass
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("ANALYZE_DONE")
