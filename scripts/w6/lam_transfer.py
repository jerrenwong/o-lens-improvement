"""W6: does the A-lens combination weight lam transfer? (fit + dev items only; never test / held-out)

lens(h) = unit(base h) + lam * a(h),  a(h) = unit(unit(E_t3 h + b) + unit(E_s5 h + b'))   (auto/candidates/alens.py)
base = J++ ("JPP") or J-lens ("J").  Concept rank = filtered harness rank (bench.excluded_ids masked), best over the
7 readout layers.  Score = macro over tasks of mean-per-item share of intermediates with rank <= k (evaluate.py).
Selection rule (alens.py): argmax filtered pass@1 + pass@10, ties (1e-6) -> smaller lam.

  python scripts/w6/lam_transfer.py compute --model Qwen/Qwen3.5-9B     # GPU, lite: writes runs/w6/ranks_<tag>.pt
  python scripts/w6/lam_transfer.py report                               # CPU ok: reads all ranks_*.pt, writes runs/w6/report.txt
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/workspace/o-lens-improvement")
OUT = ROOT / "runs" / "w6"
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
ALENS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]  # alens.py's grid (to reproduce the current choice)
CURRENT = {"9b": 0.5, "2b": 0.3, "27b": 0.5}  # lam in use (J++ + A-lens), from the task brief
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")
TAGNAME = {"Qwen/Qwen3.5-9B": "9b", "Qwen/Qwen3.5-2B": "2b", "Qwen/Qwen3.6-27B": "27b"}


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ------------------------------------------------------------------ compute (GPU)


def compute(model_name):
    sys.path.insert(0, str(ROOT / "src"))
    import workspace_lens.utils  # noqa: F401  (import before auto/candidates goes on the path: it has a jlens.py)
    sys.path.append(str(ROOT / "auto" / "candidates")); sys.path.append(str(ROOT / "scripts" / "perf1"))
    ev = _imp("w6_ev", ROOT / "auto" / ("evaluate27.py" if "27B" in model_name else "evaluate.py"))
    cfg = ev.MODELS[model_name]
    from workspace_lens.utils import load_lens_file
    from olens.readout import Bench, candidate_ranks
    from lite_model import get_lite_model
    from ask3_lib import load_maps
    from ao1_lib import emap
    from fm27_lib import unit, logits, head_gamma, NEG

    torch.set_grad_enabled(False)
    dev = "cuda"
    tag = cfg["tag"]
    bench = Bench.load(ROOT / cfg["bench"])
    split = json.loads((ROOT / f"auto/split_{tag}.json").read_text())
    pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names, strict=True))}
    fit_idx = sorted(pos[tuple(k)] for k in split["fit"])
    dev_idx = sorted(pos[tuple(k)] for k in split["dev"])
    assert fit_idx == bench.index("fit") and not set(fit_idx) & set(dev_idx)
    idx = fit_idx + dev_idx
    is_fit = [True] * len(fit_idx) + [False] * len(dev_idx)
    layers = list(cfg["layers"])
    model = get_lite_model(model_name)

    def lens_dict(path):
        L = load_lens_file(str(ROOT / path), hf_model_name=model_name)
        return {int(l): L.jacobians_L_dict_FN[l].to(dev).float() for l in L.jacobians_L_dict_FN if int(l) in layers}

    bases = {"JPP": lens_dict(cfg["JPP"]), "J": lens_dict(cfg["J"])}
    E = load_maps(f"{ROOT}/.hf27/ask3/E_{tag}_t3.pt", dev)
    E2 = load_maps(f"{ROOT}/.hf27/ask5/E_{tag}_s5.pt", dev)
    V = model._lm_head.weight.shape[0]
    mask = torch.zeros(V, dtype=torch.bool, device=model._lm_head.weight.device)
    mask[bench.excluded_ids.to(mask.device)] = True
    si = torch.as_tensor(idx, dtype=torch.long)
    cands = [bench.candidates[i] for i in idx]
    pairs = [(p, k) for p, c in enumerate(cands) for k in c]
    pid = {pk: j for j, pk in enumerate(pairs)}

    # label-free rule (c): equal logit variance of the two parts on unlabelled FIT activations (pre-norm linear head
    # W diag(gamma); both parts have unit RMS, so this measures how strongly each direction projects onto the vocab)
    W = model._lm_head.weight.float(); gam = head_gamma(model).float()
    var_b = defaultdict(list); var_a = []
    fsel = torch.as_tensor(fit_idx, dtype=torch.long)
    for l in layers:
        h = bench.resid[l][fsel].to(dev).float()
        a = unit(unit(emap(E, h, l)) + unit(emap(E2, h, l)))
        var_a.append((a * gam) @ W.T)
        var_a[-1] = var_a[-1].var(-1).mean().item()
        for b, M in bases.items():
            var_b[b].append(((unit(h @ M[l].T) * gam) @ W.T).var(-1).mean().item())
    lam_c = {b: float(np.sqrt(np.mean(var_b[b]) / np.mean(var_a))) for b in bases}
    lam_c_layer = {b: [float(np.sqrt(x / y)) for x, y in zip(var_b[b], var_a)] for b in bases}
    print("lam_c", lam_c, lam_c_layer, flush=True)

    out = dict(model=model_name, tag=tag, layers=layers, grid=GRID, items=[bench.item_names[i] for i in idx],
               tasks=[bench.eval_slugs[i] for i in idx], is_fit=is_fit, pairs=pairs, lam_c=lam_c,
               lam_c_layer=lam_c_layer, ranks={})
    for b, M in bases.items():
        lams = GRID + [lam_c[b]]
        R = torch.zeros(len(lams), len(layers), len(pairs), dtype=torch.int32)
        for li, l in enumerate(layers):
            h = bench.resid[l][si].to(dev).float()
            ub = unit(h @ M[l].T)
            a = unit(unit(emap(E, h, l)) + unit(emap(E2, h, l)))
            for gi, lam in enumerate(lams):
                v = ub if lam == 0 else ub + lam * a
                rk = candidate_ranks(logits(model, v).masked_fill(mask, NEG), cands)
                for p, d in enumerate(rk):
                    for k, r in d.items():
                        R[gi, li, pid[(p, k)]] = r
            print(f"{tag} {b} layer {l} done", flush=True)
        out["ranks"][b] = R  # [len(GRID)+1 (last = lam_c), layers, pairs]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(out, OUT / f"ranks_{tag}.pt")
    print("saved", OUT / f"ranks_{tag}.pt", flush=True)


# ------------------------------------------------------------------ report (numpy)


class M:
    """Per-model item-level hit shares. S[k][g, i] = share of item i's intermediates with best rank <= k at lam g."""

    def __init__(self, d, base):
        self.tag, self.grid = d["tag"], list(d["grid"])
        self.lam_c = d["lam_c"][base]
        R = d["ranks"][base].numpy().min(axis=1)  # best over layers: [G+1, pairs]
        n = len(d["items"])
        owner = np.array([p for p, _ in d["pairs"]])
        cnt = np.bincount(owner, minlength=n).astype(float)
        self.S = {k: np.stack([np.bincount(owner, weights=(R[g] <= k).astype(float), minlength=n) / cnt
                               for g in range(R.shape[0])]) for k in (1, 10)}
        self.task = np.array(d["tasks"]); self.fit = np.array(d["is_fit"])
        self.items = {(t, s): np.where((self.task == t) & (self.fit == (s == "fit")))[0] for t in TASKS for s in ("fit", "dev")}
        self.G = len(self.grid)  # index G = lam_c column

    def idx(self, tasks, split):
        """dict task -> item indices; split in fit/dev/all."""
        if split == "all":
            return {t: np.concatenate([self.items[(t, "fit")], self.items[(t, "dev")]]) for t in tasks}
        return {t: self.items[(t, split)] for t in tasks}

    def score(self, k, sel, g=None):
        """macro over tasks in sel (dict task -> idx) -> array over lam columns (or scalar at column g)."""
        v = np.mean([self.S[k][:, ix].mean(1) for ix in sel.values()], axis=0)
        return v if g is None else v[g]

    def choose(self, sel, cols=None):
        cols = list(range(self.G)) if cols is None else cols
        s = self.score(1, sel) + self.score(10, sel)
        return max(cols, key=lambda g: (round(s[g], 6), -self.grid[g]))


def boot_diff(m, sel, ga, gb, k=10, n=2000, seed=0):
    """paired bootstrap (items resampled within task) of score(ga) - score(gb); returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    d = m.score(k, sel, ga) - m.score(k, sel, gb)
    per = {t: m.S[k][ga, ix] - m.S[k][gb, ix] for t, ix in sel.items()}
    bs = np.array([np.mean([x[rng.integers(0, len(x), len(x))].mean() for x in per.values()]) for _ in range(n)])
    return d, *np.percentile(bs, [2.5, 97.5])


def report():
    lines = []
    P = lines.append
    models = {}
    for tag in ("9b", "2b", "27b"):
        f = OUT / f"ranks_{tag}.pt"
        if f.exists():
            models[tag] = torch.load(f, weights_only=False)
    f3 = lambda x: f"{x:.3f}"
    for base in ("JPP", "J"):
        P(f"\n################ base = {base} ('{'J++' if base == 'JPP' else 'J-lens'} + A-lens') ################")
        Ms = {t: M(d, base) for t, d in models.items()}
        for tag, m in Ms.items():
            gl = m.grid
            fit_all, dev_all = m.idx(TASKS, "fit"), m.idx(TASKS, "dev")
            g_fit = m.choose(fit_all)
            cur = gl.index(CURRENT[tag]) if base == "JPP" else g_fit  # in-use lam exists for J++ only
            g_fit_alens = m.choose(fit_all, [gl.index(x) for x in ALENS_GRID])
            P(f"\n===== {tag}  (n fit {sum(map(len, fit_all.values()))}, dev {sum(map(len, dev_all.values()))}) =====")
            P(f"fit-chosen lam: fine grid {gl[g_fit]} | alens grid {gl[g_fit_alens]} | in use {CURRENT[tag]} "
              f"| label-free lam_c {m.lam_c:.3f}" + ("" if base == "JPP" else " (no J-lens lam in use: 'in use' = fit-chosen)"))
            P("items per task (fit/dev): " + ", ".join(f"{t} {len(m.items[(t, 'fit')])}/{len(m.items[(t, 'dev')])}" for t in TASKS))
            # 1. curves
            P("\n[1] score vs lam (macro; * = in use)")
            P("lam    " + " ".join(f"{x:>6}" for x in gl))
            for split, sel in (("fit", fit_all), ("dev", dev_all)):
                for k in (10, 1):
                    P(f"{split}@{k:<2} " + " ".join(f3(v).rjust(6) for v in m.score(k, sel)[:m.G]))
            P("per task, dev top-10:")
            for t in TASKS:
                s = m.score(10, {t: m.items[(t, "dev")]})[:m.G]
                P(f"  {t[:11]:11s} " + " ".join(f3(v).rjust(6) for v in s) + f"   best lam {gl[int(np.argmax(s))]}")
            P("per task, dev top-1:")
            for t in TASKS:
                s = m.score(1, {t: m.items[(t, "dev")]})[:m.G]
                P(f"  {t[:11]:11s} " + " ".join(f3(v).rjust(6) for v in s) + f"   best lam {gl[int(np.argmax(s))]}")
            dv10 = m.score(10, dev_all)[:m.G]
            near = [g for g in range(m.G) if abs(dv10[g] - dv10[cur]) <= 0.01]
            P(f"flatness: dev@10 within 0.01 of the in-use value for lam in {[gl[g] for g in near]}; "
              f"dev@10 range over lam in [0.2, 1.0]: {f3(dv10[gl.index(0.2):gl.index(1.0) + 1].min())}-"
              f"{f3(dv10[gl.index(0.2):gl.index(1.0) + 1].max())}; at lam 0: {f3(dv10[0])}")
            for g in (gl.index(0.0), gl.index(1.0), gl.index(2.0)):
                d, lo, hi = boot_diff(m, dev_all, g, cur)
                P(f"  dev@10 lam {gl[g]} - lam {gl[cur]}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

            # 2. fit on one task, test on the others
            P("\n[2] choose lam on ONE task (its fit+dev items pooled), score the OTHER four tasks' dev items (top-10)")
            P(f"{'chosen on':13s} {'lam':>5} {'transf@10':>9} {'allfit@10':>9} {'oracle@10':>9} {'oracle lam':>10} "
              f"{'regret':>7} {'transf@1':>8} {'allfit@1':>8}  allfit-transf @10 [95% CI]")
            order = ("multihop",) + tuple(t for t in TASKS if t != "multihop")
            mat10 = {}
            for T in order:
                gT = m.choose(m.idx([T], "all"))
                others = {t: m.items[(t, "dev")] for t in TASKS if t != T}
                s10 = m.score(10, others)[:m.G]
                go = int(np.argmax(s10))
                d, lo, hi = boot_diff(m, others, g_fit, gT)
                P(f"{T:13s} {gl[gT]:>5} {f3(s10[gT]):>9} {f3(s10[g_fit]):>9} {f3(s10[go]):>9} {gl[go]:>10} "
                  f"{s10[go] - s10[gT]:>+7.3f} {f3(m.score(1, others, gT)):>8} {f3(m.score(1, others, g_fit)):>8}  "
                  f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                mat10[T] = (gT, [m.score(10, {t: m.items[(t, 'dev')]}, gT) for t in TASKS])
            P("5x5 dev top-10 (row = task used to choose lam [lam], column = task scored; diagonal = in-sample)")
            P(f"{'':20s}" + " ".join(f"{t[:9]:>9}" for t in TASKS))
            for T in order:
                gT, row = mat10[T]
                P(f"{T[:12]:12s} [{gl[gT]:>4}] " + " ".join(f"{f3(v):>9}" for v in row))
            P(f"{'all-fit':12s} [{gl[g_fit]:>4}] " + " ".join(f"{f3(m.score(10, {t: m.items[(t, 'dev')]}, g_fit)):>9}" for t in TASKS))
            P(f"{'oracle/task':19s} " + " ".join(f"{f3(m.score(10, {t: m.items[(t, 'dev')]})[:m.G].max()):>9}" for t in TASKS))

            # 3. bootstrap of the fit split
            rng = np.random.default_rng(1)
            ch = []
            for _ in range(500):
                sel = {t: ix[rng.integers(0, len(ix), len(ix))] for t, ix in fit_all.items()}
                ch.append(m.choose(sel))
            ch = np.array(ch)
            P("\n[3] fit-split bootstrap (500, within task): chosen lam distribution -> dev score at that lam")
            P("lam     share   dev@10  dev@1")
            for g in sorted(set(ch.tolist())):
                P(f"{gl[g]:<6} {np.mean(ch == g):6.3f}   {f3(m.score(10, dev_all, g))}  {f3(m.score(1, dev_all, g))}")
            d10 = np.array([m.score(10, dev_all, g) for g in ch]); d1 = np.array([m.score(1, dev_all, g) for g in ch])
            P(f"dev@10 over resamples: mean {f3(d10.mean())}, 2.5-97.5% {f3(np.percentile(d10, 2.5))}-"
              f"{f3(np.percentile(d10, 97.5))}; dev@1 mean {f3(d1.mean())}, {f3(np.percentile(d1, 2.5))}-"
              f"{f3(np.percentile(d1, 97.5))}")

            # 4. label-free
            P("\n[4] label-free settings on dev (vs fit-chosen lam; paired 95% CI, 2000 draws)")
            P(f"{'setting':26s} {'lam':>6} {'dev@10':>7} {'dev@1':>7}   d@10 vs fitted [CI]        d@1 vs fitted [CI]")
            rows = [("fit-chosen (fine grid)", g_fit), ("in use", cur), ("J/J++ alone (lam 0)", 0),
                    ("(a) lam = 1", gl.index(1.0)), ("(b) lam = 0.5", gl.index(0.5)), ("(c) equal logit var", m.G)]
            for name, g in rows:
                lam = m.lam_c if g == m.G else gl[g]
                a = boot_diff(m, dev_all, g, g_fit, 10); b = boot_diff(m, dev_all, g, g_fit, 1)
                P(f"{name:26s} {lam:>6.3f} {f3(m.score(10, dev_all, g)):>7} {f3(m.score(1, dev_all, g)):>7}   "
                  f"{a[0]:+.3f} [{a[1]:+.3f}, {a[2]:+.3f}]   {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}]")
            P(f"(c) per-layer lam_c: {[round(x, 3) for x in models[tag]['lam_c_layer'][base]]}")

        # 5. cross-model
        if len(Ms) > 1:
            P("\n[5] cross-model: lam chosen on model A's fit applied to model B's dev (vs B's own fit lam)")
            P(f"{'from->to':10s} {'lam':>5} {'own lam':>7} {'dev@10':>7} {'own@10':>7}  d@10 [CI]                d@1 [CI]")
            for a, ma in Ms.items():
                la = ma.grid[ma.choose(ma.idx(TASKS, "fit"))]
                for b, mb in Ms.items():
                    if a == b:
                        continue
                    dv = mb.idx(TASKS, "dev")
                    gown = mb.choose(mb.idx(TASKS, "fit")); gx = mb.grid.index(la)
                    x = boot_diff(mb, dv, gx, gown, 10); y = boot_diff(mb, dv, gx, gown, 1)
                    P(f"{a + '->' + b:10s} {la:>5} {mb.grid[gown]:>7} {f3(mb.score(10, dv, gx)):>7} "
                      f"{f3(mb.score(10, dv, gown)):>7}  {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]   "
                      f"{y[0]:+.3f} [{y[1]:+.3f}, {y[2]:+.3f}]")
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compute", "report"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    a = ap.parse_args()
    compute(a.model) if a.cmd == "compute" else report()
