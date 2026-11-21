"""mp1 step 1: for every al2 sentence (run alone, no special tokens, as al2 collect.py) collect at LAYERS (fp16)
(a) the mean residual over ALL tokens, (b) the mean over the LAST HALF (positions n//2 .. n-1).
(c) = al2's X_L*.pt (last token); checked here on a sample.
Phase 2 (test sentences only): J++ applied to EACH token, then
  (i)   mean over tokens of the raw per-token logits  (= J++ on the token mean, by linearity: sanity check)
  (in)  mean over tokens of final-normed per-token logits (each token read as the benchmark reads one token)
  (ii)  max over tokens of each word's final-normed logit (best position)
  (ii0) same, excluding token 0
for the released and the generated-fit J++.  Ranks: full vocab, punctuation filter, any surface form (al2 rule)."""
import json, os, sys, time
from collections import defaultdict
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
from al2_common import R, BIG, BIG1, log, retry, load_model
from workspace_lens.utils import load_lens_file

torch.set_grad_enabled(False)
dev = "cuda"
OUT = f"{R}/.hf27/mp1"
os.makedirs(OUT, exist_ok=True)
LAYERS = [4, 8, 12, 16, 20, 24, 28]
t00 = time.time()
meta = torch.load(f"{BIG}/meta.pt", weights_only=False)
sents, words_s, split = meta["sents"], meta["words"], meta["split"]
N = len(sents)
model = load_model()
tok = model.tokenizer
d = model.d_model
all_ids = [tok(s, add_special_tokens=False).input_ids for s in sents]
lens_n = torch.tensor([len(x) for x in all_ids])
log(f"model loaded; {N} sentences; tokens mean {lens_n.float().mean():.1f} min {lens_n.min()} max {lens_n.max()} "
    f"total {int(lens_n.sum())}")
groups = defaultdict(list)
for i, s in enumerate(all_ids):
    groups[len(s)].append(i)


def fwd(sub, full=False):
    ids = torch.tensor([all_ids[i] for i in sub], device=dev)
    rec = {}
    hk = [model.layers[l].register_forward_hook(
        (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])))(l)) for l in LAYERS]
    try:
        model.forward(ids)
    finally:
        for h in hk:
            h.remove()
    if full:
        return rec
    n = ids.shape[1]
    out = {}
    for l in LAYERS:
        h = rec[l].float()
        nr = h.norm(dim=-1)  # [B, n]
        out[l] = (h.mean(1).half().cpu(), h[:, n // 2:].mean(1).half().cpu(), h[:, -1].half().cpu(),
                  nr[:, 0].sum().item(), nr[:, 1:].mean(1).sum().item(), nr.max(1).values.sum().item())
    return out


# ------------------------------------------------------------ phase 1
if not all(os.path.exists(f"{OUT}/Xhalf_L{l}.pt") for l in LAYERS):
    Xm = {l: torch.zeros(N, d, dtype=torch.float16) for l in LAYERS}
    Xh = {l: torch.zeros(N, d, dtype=torch.float16) for l in LAYERS}
    Xl = {l: torch.zeros(N, d, dtype=torch.float16) for l in LAYERS}
    ns = {l: [0.0, 0.0, 0.0] for l in LAYERS}
    t1 = time.time()
    for L, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), 128):
            sub = idx[b0:b0 + 128]
            r = retry(fwd, sub)
            ti = torch.tensor(sub)
            for l in LAYERS:
                Xm[l][ti], Xh[l][ti], Xl[l][ti] = r[l][0], r[l][1], r[l][2]
                for k in range(3):
                    ns[l][k] += r[l][3 + k]
    tf = time.time() - t1
    log(f"phase1 forward {tf:.0f}s")
    stats = dict(forward_s=tf, n=N, tokens_total=int(lens_n.sum()), norms={}, last_check={})
    for l in LAYERS:
        stats["norms"][l] = dict(tok0=ns[l][0] / N, rest_mean=ns[l][1] / N, max=ns[l][2] / N,
                                 mean_vec=float(Xm[l][:2000].float().norm(dim=-1).mean()),
                                 half_vec=float(Xh[l][:2000].float().norm(dim=-1).mean()))
        Xa = torch.load(f"{BIG}/X_L{l}.pt")
        sel = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:3000]
        stats["last_check"][l] = dict(
            cos_min=float(torch.cosine_similarity(Xa[sel].float(), Xl[l][sel].float(), dim=-1).min()),
            cos_mean_vs_last=float(torch.cosine_similarity(Xm[l][sel].float(), Xa[sel].float(), dim=-1).mean()),
            cos_half_vs_last=float(torch.cosine_similarity(Xh[l][sel].float(), Xa[sel].float(), dim=-1).mean()))
        del Xa
        torch.save(Xm[l], f"{OUT}/Xmean_L{l}.pt")
        torch.save(Xh[l], f"{OUT}/Xhalf_L{l}.pt")
        log(f"L{l} saved; norms {stats['norms'][l]}; check {stats['last_check'][l]}")
    stats["bytes_per_file"] = os.path.getsize(f"{OUT}/Xmean_L4.pt")
    stats["total_s"] = time.time() - t00
    json.dump(stats, open(f"{OUT}/collect_stats.json", "w"), indent=1)
    del Xm, Xh, Xl

# ------------------------------------------------------------ phase 2 (test sentences, per-token J++)
if not os.path.exists(f"{OUT}/pertok_ranks.pt"):
    U = model._lm_head.weight  # [V, d] bf16
    g = meta["g"].to(dev).float()
    V = U.shape[0]
    strs = json.load(open(f"{BIG}/strs.json")) if os.path.exists(f"{BIG}/strs.json") else torch.load(
        f"{BIG1}/meta.pt", weights_only=False)["strs"]
    assert len(strs) == V, (len(strs), V)
    excl = torch.tensor([not any(ch.isalnum() for ch in s) for s in strs], device=dev)
    by_norm = defaultdict(list)
    for i, s in enumerate(strs):
        by_norm[s.strip().lower()].append(i)
    W_all = sorted(meta["role"])
    widx = {w: i for i, w in enumerate(W_all)}
    forms = [by_norm[w.lower()] for w in W_all]
    mf = max(len(f) for f in forms)
    FORM = torch.tensor([f + [f[0]] * (mf - len(f)) for f in forms], device=dev)

    def lensd(p):
        return {int(l): v.to(dev).float() for l, v in
                load_lens_file(f"{R}/{p}", hf_model_name="Qwen/Qwen3.5-9B").jacobians_L_dict_FN.items()}

    LENSES = {"Jrel": lensd("artifacts/baselines/jpp9b_prefit.pt"), "Jgen": lensd("runs/w1/jpp_Afull_Wtrain.pt")}
    TE = [i for i in range(N) if split[i] == "test"]
    pos_te = {i: j for j, i in enumerate(TE)}
    KINDS = ["i_raw", "i_norm", "ii_max", "ii_max_no0"]
    RK = {(ln, k, l): torch.zeros(len(TE), dtype=torch.long) for ln in LENSES for k in KINDS for l in LAYERS}
    WINPOS = {(ln, l): torch.zeros(len(TE)) for ln in LENSES for l in LAYERS}  # relative position of best token

    def rank_from(lg, wid):
        lg = lg.masked_fill(excl, float("-inf"))
        best = lg.gather(1, FORM[wid]).max(1).values
        return (lg > best[:, None]).sum(1) + 1

    def one2(sub):
        rec = fwd(sub, full=True)
        wid = torch.tensor([widx[words_s[i]] for i in sub], device=dev)
        res = {}
        for l in LAYERS:
            H = rec[l].float()
            B, n, _ = H.shape
            for ln, J in LENSES.items():
                v = H.reshape(B * n, d) @ J[l].T
                rms = v.pow(2).mean(-1).sqrt().clamp_min(1e-6)
                raw = ((v * g).to(U.dtype) @ U.T).float().view(B, n, V)
                nrm = raw / rms.view(B, n, 1)
                res[(ln, "i_raw", l)] = rank_from(raw.mean(1), wid).cpu()
                res[(ln, "i_norm", l)] = rank_from(nrm.mean(1), wid).cpu()
                mx, am = nrm.max(1)
                res[(ln, "ii_max", l)] = rank_from(mx, wid).cpu()
                if n > 1:
                    res[(ln, "ii_max_no0", l)] = rank_from(nrm[:, 1:].max(1).values, wid).cpu()
                else:
                    res[(ln, "ii_max_no0", l)] = res[(ln, "ii_max", l)]
                # position of the token holding the best word form's max (relative 0..1)
                fb = FORM[wid]  # [B, mf]
                pw = nrm.gather(2, fb[:, None, :].expand(B, n, fb.shape[1])).max(2).values  # [B, n]
                res[(ln, "pos", l)] = (pw.argmax(1).float() / max(1, n - 1)).cpu()
                del raw, nrm, v
        return res

    t2 = time.time()
    te_groups = defaultdict(list)
    for i in TE:
        te_groups[len(all_ids[i])].append(i)
    for L, idx in sorted(te_groups.items()):
        bs = max(1, 384 // L)
        for b0 in range(0, len(idx), bs):
            sub = idx[b0:b0 + bs]
            r = retry(one2, sub)
            tj = torch.tensor([pos_te[i] for i in sub])
            for (ln, k, l), v in r.items():
                if k == "pos":
                    WINPOS[(ln, l)][tj] = v
                else:
                    RK[(ln, k, l)][tj] = v
    log(f"phase2 per-token {time.time() - t2:.0f}s")
    torch.save(dict(RK=RK, WINPOS=WINPOS, te_rows=torch.tensor(TE)), f"{OUT}/pertok_ranks.pt")
log(f"done {time.time() - t00:.0f}s")
print("COLLECT_DONE")
