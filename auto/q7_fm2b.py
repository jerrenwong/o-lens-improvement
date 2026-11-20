"""q7: port of the 9B failure-mode analysis (qual_wordonly, qual_zh, qual2, qual3, qual4) to Qwen3.5-2B.
FIT + DEV items of auto/split_2b.json only; test items are dropped at load and never enter any statistic or printout.
9B layer -> 2B layer: 4,8,12,16,20,24,28 -> 3,6,9,12,15,18,21 (12,20,24,28 -> 9,15,18,21).
    cd /workspace/o-lens-improvement && /root/jpp_lens/.venv/bin/python auto/q7_fm2b.py > /root/q7_fm2b.log 2>&1"""
import json, collections
from pathlib import Path
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
from olens.train import wikitext_windows
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-2B"; LAYERS = [3, 6, 9, 12, 15, 18, 21]; L4 = [9, 15, 18, 21]
LATE2 = (18, 21); LATE3 = (15, 18, 21)
t.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer; hf = model._hf_model
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-2b.pt")
sp = json.load(open(f"{R}/auto/split_2b.json"))
test_keys = {tuple(x) for x in sp["test"]}
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
assert not (set(part) & test_keys)
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in part]
assert all((b.eval_slugs[i], b.item_names[i]) not in test_keys for i in idx)
J = load_lens_file(f"{R}/artifacts/baselines/jlens_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
P = load_lens_file(f"{R}/artifacts/baselines/jpp_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
LENS = {"J": J, "JPP": P}
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
def is_cjk(s): return any('一' <= ch <= '鿿' or '぀' <= ch <= 'ヿ' for ch in s)
def plainw(s):
    u = s[1:] if s.startswith(" ") else s
    return len(u) > 0 and u.isalnum()
cjk = t.tensor([is_cjk(s) for s in strs]).cuda()
word = t.tensor([plainw(s) for s in strs]).cuda()
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
NEG = float("-inf")
task_rows = collections.defaultdict(list)
for r, i in enumerate(idx): task_rows[b.eval_slugs[i]].append(r)
split_of = [part[(b.eval_slugs[i], b.item_names[i])] for i in idx]
fit_rows = [r for r in range(len(idx)) if split_of[r] == "fit"]
print(f"vocab {Vn}; CJK {int(cjk.sum())}; bench filter removes {int(punct.sum())}; plain-word filter removes {int((~word).sum())}")
print(f"items fit {len(fit_rows)} dev {len(idx) - len(fit_rows)} (test excluded); by task " +
      str({e: (sum(split_of[r] == 'fit' for r in rows), sum(split_of[r] == 'dev' for r in rows)) for e, rows in sorted(task_rows.items())}), flush=True)

def H(L, rows=None):
    rr = idx if rows is None else [idx[r] for r in rows]
    return b.resid[L][rr].cuda().float()
def RO(name, L, rows=None): return H(L, rows) @ LENS[name][L].cuda().float().T
def macro(d, k, split):
    item = collections.defaultdict(list)
    for (i, key), r in d.items():
        if part[(b.eval_slugs[i], b.item_names[i])] == split: item[i].append(r <= k)
    ev = collections.defaultdict(list)
    for i, hs in item.items(): ev[b.eval_slugs[i]].append(sum(hs) / len(hs))
    return sum(sum(v) / len(v) for v in ev.values()) / len(ev), {e[:5]: round(sum(v) / len(v), 2) for e, v in sorted(ev.items())}
def lens_best(name, edit=None, mats=None, vec_fn=None):
    best = {}
    for L in LAYERS:
        if vec_fn is not None: v = vec_fn(name, L)
        else:
            Mx = mats[(name, L)] if mats else LENS[name][L].cuda().float()
            v = H(L) @ Mx.T
        if edit is not None: v = edit(name, L, v)
        for s in range(0, len(idx), 32):
            lg = model.unembed(v[s:s+32]).float().masked_fill(punct, NEG)
            for k in range(lg.shape[0]):
                i = idx[s + k]
                for key, ids in b.candidates[i].items():
                    best[(i, key)] = min(best.get((i, key), 10**9), int((lg[k] > lg[k, ids].max()).sum()) + 1)
    return best
def report(label, **kw):
    out = {}
    for n in LENS:
        bd = lens_best(n, **kw); out[n] = bd
        for s_ in ("fit", "dev"):
            p10, per = macro(bd, 10, s_); p1, _ = macro(bd, 1, s_)
            print(f"  {n:3s} {s_} {label:54s} {p10:.4f} / {p1:.4f}  {per}", flush=True)
    return out
def topk_str(v, n=8, mask=punct):
    return " | ".join(strs[j].strip() for j in model.unembed(v).float().masked_fill(mask, NEG).topk(n, -1).indices[0].tolist())

# ---------- text + hidden-state convention
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_2b.json")); RV = rd["vocab"]
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in part}
text = {k: "".join(RV[x] for x in it["toks"]) for k, it in meta.items()}
print(f"\n== convention check (readouts_2b.json has {len(meta)} of {len(idx)} fit+dev items)")
for i in idx[:3]:
    k = (b.eval_slugs[i], b.item_names[i])
    ids = tok(text[k], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    hs = hf(ids, output_hidden_states=True).hidden_states
    p = meta[k]["pos"] + (ids.shape[1] - len(meta[k]["toks"]))
    print(f"  {k} len {ids.shape[1]} vs {len(meta[k]['toks'])} pos {p}: " + " ".join(
        f"L{L}: hs[L] {t.cosine_similarity(hs[L][0, p].float().cpu(), b.resid[L][i], 0):.3f} hs[L+1] {t.cosine_similarity(hs[L+1][0, p].float().cpu(), b.resid[L][i], 0):.3f}" for L in (9, 18)))
OFF = 1 if t.cosine_similarity(hs[19][0, p].float().cpu(), b.resid[18][i], 0) > t.cosine_similarity(hs[18][0, p].float().cpu(), b.resid[18][i], 0) else 0
print(f"  using hidden_states[L + {OFF}]", flush=True)

# ================= 1. token classes, top-1, widespread tokens
def tclass(s):
    if s.strip() == "" or all(not ch.isalnum() for ch in s): return "punct/space"
    if is_cjk(s): return "CJK"
    u = s[1:] if s.startswith(" ") else s
    if not u.isalnum(): return "glued (alnum+symbol)"
    if not u.isascii(): return "other script"
    if len(u) <= 2: return "fragment (<=2)"
    return "word (>=3)"
CLS = ["punct/space", "glued (alnum+symbol)", "fragment (<=2)", "word (>=3)", "CJK", "other script"]
tcls = [tclass(s) for s in strs]
print("\n== 1a. class shares (%) of UNFILTERED top-10 slots, fit+dev pooled; columns: " + " / ".join(CLS))
TOPU = {}; TOPF = {}
for n in LENS:
    for L in LAYERS:
        lg = model.unembed(RO(n, L)).float()
        TOPU[(n, L)] = lg.topk(10, -1).indices.tolist()
        TOPF[(n, L)] = lg.masked_fill(punct, NEG).topk(10, -1).indices.tolist()
        c = collections.Counter(tcls[j] for row in TOPU[(n, L)] for j in row); tot = sum(c.values())
        print(f"  {n:3s} L{L:2d}  " + " / ".join(f"{100 * c[k] / tot:3.0f}" for k in CLS))
print("\n== 1b. most common UNFILTERED top-1 tokens per layer (count of fit+dev prompts)")
for n in LENS:
    for L in LAYERS:
        c = collections.Counter(strs[row[0]] for row in TOPU[(n, L)])
        print(f"  {n:3s} L{L:2d}  " + ", ".join(f"{k!r}x{v}" for k, v in c.most_common(8)))
print("\n== 1c. per task: filtered (benchmark filter) top-10 tokens present in the most prompts (share of the task's prompts)")
for e, rows in sorted(task_rows.items()):
    for n in LENS:
        for L in L4:
            c = collections.Counter(strs[j] for r in rows for j in set(TOPF[(n, L)][r]))
            print(f"  {e[:6]:6s} {n:3s} L{L:2d} (n={len(rows)}): " + ", ".join(f"{k!r} {v / len(rows):.2f}" for k, v in c.most_common(8)))
print(flush=True)

# ================= 2. scorings: benchmark filter, plain-word filter, en|zh, zh, CJK masked
keys = sorted({k for i in idx for k in b.candidates[i]})
FEW = "English: night\nChinese: 夜晚\nEnglish: river\nChinese: 河流\nEnglish: Germany\nChinese: 德国\nEnglish: {w}\nChinese:"
STOP = set("的 一 是 了 在 和 有 不 人 这 中 大 为 上 个 我 以 要 他 时 来 用 们 生 到 作 地 于 出 就 分 对 成 会 可 主 发 年 动 同 工 也 能 下 过 子 说 产 种 面 而 方 后 多 定 行 学 法 所".split())
by_text = collections.defaultdict(list)
for i, s in enumerate(strs):
    if is_cjk(s): by_text[s.strip()].append(i)
zh = {}
for w in keys:
    cur = tok(FEW.format(w=w), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    alts, out = [], []
    for step in range(6):
        lg = hf(cur).logits[0, -1].float()
        if step == 0:
            pr = lg.softmax(-1); top = pr.topk(8)
            alts = [strs[j].strip() for p_, j in zip(top.values.tolist(), top.indices.tolist()) if p_ >= 0.05 and is_cjk(strs[j])]
        nx = int(lg.argmax()); s = strs[nx]
        if s.strip() == "" and "\n" not in s and not out:
            cur = t.cat([cur, t.tensor([[nx]]).cuda()], 1)
            pr = hf(cur).logits[0, -1].float().softmax(-1); top = pr.topk(8)
            alts = [strs[j].strip() for p_, j in zip(top.values.tolist(), top.indices.tolist()) if p_ >= 0.05 and is_cjk(strs[j])]
            continue
        if "\n" in s or not is_cjk(s): break
        out.append(s.strip()); cur = t.cat([cur, t.tensor([[nx]]).cuda()], 1)
    T = "".join(out)
    forms = ({T, T + "的", "的" + T} | set(alts)) if T else set(alts)
    forms = {f for f in forms if f and f not in STOP}
    zh[w] = (T, sorted({j for f in forms for j in by_text.get(f, [])}))
print("== 2a. model's own translations (concept -> greedy | accepted single tokens)")
print("; ".join(f"{w}->{T}|{'/'.join(sorted({strs[j].strip() for j in ids}))}" for w, (T, ids) in zh.items()))
cand_all = sorted({c for i in idx for ids in b.candidates[i].values() for c in ids})
print("concept candidate tokens removed by the plain-word rule:", sum(not plainw(strs[c]) for c in cand_all), "of", len(cand_all),
      [strs[c] for c in cand_all if not plainw(strs[c])][:12], flush=True)
res = collections.defaultdict(dict); zh_hit_tok = {}
SC = ("bench", "plain-word", "zh only", "en|zh", "CJK masked")
for L in LAYERS:
    for name in LENS:
        v = RO(name, L)
        for s in range(0, len(idx), 32):
            lg0 = model.unembed(v[s:s+32]).float()
            lg = lg0.masked_fill(punct, NEG); lgw = lg0.masked_fill(~word, NEG); lgc = lg.masked_fill(cjk, NEG)
            for r in range(lg.shape[0]):
                i = idx[s + r]
                for key, ids in b.candidates[i].items():
                    en = [c for c in ids if not punct[c]]; wd = [c for c in ids if word[c]]; zc = zh[key][1]
                    rk = lambda l, cs: int((l[r] > l[r, cs].max()).sum()) + 1 if cs else 10**9
                    r_en, r_zh = rk(lg, en), rk(lg, zc)
                    for sc, x in (("bench", r_en), ("plain-word", rk(lgw, wd)), ("zh only", r_zh), ("en|zh", min(r_en, r_zh)), ("CJK masked", rk(lgc, en))):
                        d = res[(name, sc)]; d[(i, key)] = min(d.get((i, key), 10**9), x)
                    if r_zh <= 10 and (name, i, key) not in zh_hit_tok:
                        zh_hit_tok[(name, i, key)] = (L, strs[zc[int(lg[r, zc].argmax())]].strip())
print("\n== 2b. macro pass@10 / pass@1, per-task pass@10")
for split in ("fit", "dev"):
    for name in LENS:
        for sc in SC:
            p10, per = macro(res[(name, sc)], 10, split); p1, _ = macro(res[(name, sc)], 1, split)
            print(f"  {split} {name:3s} {sc:11s} {p10:.4f} / {p1:.4f}  {per}")
print("\n== 2c. pairs missed by the benchmark at every layer whose Chinese form is in the top 10 (fit+dev)")
for name in LENS:
    rows = [(b.eval_slugs[i], b.item_names[i], key, res[(name, 'bench')][(i, key)], zh_hit_tok.get((name, i, key)))
            for (i, key), r in res[(name, "zh only")].items() if r <= 10 and res[(name, "bench")][(i, key)] > 10]
    tot = sum(r > 10 for r in res[(name, "bench")].values())
    print(f"  {name}: {len(rows)} of {tot} missed pairs; by task {dict(collections.Counter(r[0] for r in rows))}")
    for e, n_, key, ren, hit in rows[:25]: print(f"     {e[:6]}/{n_}: {key} (bench best rank {ren}) -> {hit}")
print(f"  pairs with a Chinese single-token form: {sum(1 for (i, key) in res[('J', 'bench')] if zh[key][1])} of {len(res[('J', 'bench')])}", flush=True)

# ================= 3. concentration
print("\n== 3a. mean cos(readout, same-task mean readout), fit+dev rows of each task")
for n in LENS:
    for e, rows in sorted(task_rows.items()):
        line = []
        for L in LAYERS:
            v = RO(n, L, rows); line.append(f"L{L} {float(t.cosine_similarity(v, v.mean(0, keepdim=True), -1).mean()):.2f}")
        print(f"   {n:3s} {e:13s} " + " ".join(line))
SV = {}
def dec(v, n=6): return " ".join(strs[j].strip() for j in model.unembed(v[None]).float().masked_fill(punct, NEG).topk(n, -1).indices[0].tolist())
print("\n== 3b. top singular output directions of each lens matrix: sigma1..5/median; readout energy in top 1/5/20 dirs by task; decoded (both signs)")
for n in LENS:
    for L in LAYERS:
        U, S, Vh = t.linalg.svd(LENS[n][L].cuda().float(), full_matrices=False); SV[(n, L)] = (U, S, Vh)
        v = RO(n, L); co = (v @ U[:, :20]) ** 2 / (v ** 2).sum(-1, keepdim=True)
        sh = {e[:5]: f"{float(co[rows, :1].sum(-1).mean()):.2f}/{float(co[rows, :5].sum(-1).mean()):.2f}/{float(co[rows].sum(-1).mean()):.2f}" for e, rows in sorted(task_rows.items())}
        print(f" {n:3s} L{L:2d} sigma = {[round(float(x / S.median()), 1) for x in S[:5]]}  energy: {sh}")
        if L in L4:
            for k in range(3): print(f"        u{k+1}+ : {dec(U[:, k])}    u{k+1}- : {dec(-U[:, k])}")
print(flush=True)

# ================= 4. junk family: filtered top-10 tokens present in > 60% of a task's prompts
print("== 4a. tokens in the filtered top-10 of > 60% of a task's fit+dev prompts (share)")
JUNK = {}
for n in LENS:
    for e, rows in sorted(task_rows.items()):
        for L in LAYERS:
            c = collections.Counter(strs[j] for r in rows for j in set(TOPF[(n, L)][r]))
            hot = [(k, v / len(rows)) for k, v in c.most_common() if v / len(rows) > 0.6]
            if e == "association": JUNK[(n, L)] = {k for k, _ in hot}
            if hot: print(f"   {n:3s} {e:13s} L{L:2d}: " + ", ".join(f"{k!r} {s:.2f}" for k, s in hot[:12]))
print("   association junk sets (token string sets used in 4b):", {f"{n} L{L}": sorted(v) for (n, L), v in JUNK.items() if v}, flush=True)
junkmask = {k: t.tensor([s in v for s in strs]).cuda() for k, v in JUNK.items() if v}
print("\n== 4b. period-token test on association (junk share = share of filtered top-10 slots in that layer's association junk set)")
print("   token ids:", {s: tok(s, add_special_tokens=False)["input_ids"] for s in ("end.", "end .", " .", ".", "end!", ".\n")})
lastc = collections.Counter(text[("association", b.item_names[idx[r]])][-1] for r in task_rows["association"])
print("   last characters of association prompts:", lastc.most_common(4))
def run(texts, pos=-1):
    out = {L: [] for L in LAYERS}
    for s in texts:
        hs = hf(tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda(), output_hidden_states=True).hidden_states
        for L in LAYERS: out[L].append(hs[L + OFF][0, pos].float())
    return {L: t.stack(v) for L, v in out.items()}
arows = task_rows["association"]; atext = [text[("association", b.item_names[idx[r]])] for r in arows]
print("   association prompts ending with '.':", sum(s.endswith(".") for s in atext), "of", len(atext))
variants = {"original (word.)": atext, "space before period (word .)": [s[:-1] + " ." for s in atext], "no period (read at last word)": [s[:-1] for s in atext],
            "period + newline (read at newline)": [s + "\n" for s in atext], "exclamation mark": [s[:-1] + "!" for s in atext]}
for vn, texts in variants.items():
    Hs = run(texts)
    for n in LENS:
        hits = collections.defaultdict(lambda: 10**9); line = []
        for L in LAYERS:
            lg = model.unembed(Hs[L] @ LENS[n][L].cuda().float().T).float().masked_fill(punct, NEG)
            tp = lg.topk(10, -1).indices
            line.append(f"L{L} {float(junkmask[(n, L)][tp].float().mean()):.2f}" if (n, L) in junkmask else f"L{L} -")
            for k, r in enumerate(arows):
                for key, ids in b.candidates[idx[r]].items():
                    hits[(r, key)] = min(hits[(r, key)], int((lg[k] > lg[k, ids].max()).sum()) + 1)
        p10 = sum(v <= 10 for v in hits.values()) / len(hits)
        print(f"   {vn:36s} {n:3s} junk share: {' '.join(line)}   assoc pass@10 (fit+dev pairs) {p10:.3f}")
        if vn != "original (word.)":
            lg = model.unembed(Hs[18][:2] @ LENS[n][18].cuda().float().T).float().masked_fill(punct, NEG)
            print("        L18 examples: " + " || ".join(" | ".join(strs[j].strip() for j in row[:7]) for row in lg.topk(10, -1).indices.tolist()))
print(flush=True)
win = wikitext_windows(tok, "validation", 400, 128, Path("/root/cache"), True).long()
acts = {L: [] for L in LATE3}
for s in range(0, len(win), 50):
    hs = hf(win[s:s+50].cuda(), output_hidden_states=True).hidden_states
    for L in LATE3: acts[L].append(hs[L + OFF].float().cpu())
acts = {L: t.cat(v) for L, v in acts.items()}
is_p = t.tensor([[strs[x].strip() == "." for x in row] for row in win.tolist()])
print("   WikiText: junk share at period tokens vs other tokens")
for n in LENS:
    for L in LATE3:
        if (n, L) not in junkmask: continue
        Mx = LENS[n][L].cuda().float(); line = []
        for g, m in (("period", is_p), ("other", ~is_p)):
            tp = model.unembed(acts[L][m][:3000].cuda() @ Mx.T).float().masked_fill(punct, NEG).topk(10, -1).indices
            line.append(f"{g} n={int(m.sum())} {float(junkmask[(n, L)][tp].float().mean()):.3f}")
        print(f"   {n:3s} L{L}: " + "  ".join(line), flush=True)

# ================= 5. mean subtraction, singular-direction removal, gated removal
print("\n== 5a. read lens(h - alpha * mean activation). 'scored-split' means: mean over the scored split (transductive, as 9B qual2 D); 'fit-ref': mean of FIT rows")
split_rows = {s_: [r for r in range(len(idx)) if split_of[r] == s_] for s_ in ("fit", "dev")}
def centred(mode, alpha):
    def fn(name, L):
        h = H(L); mu = t.zeros_like(h)
        for s_, rows in split_rows.items():
            rs = set(rows)
            if mode == "split": mu[rows] = h[rows].mean(0, keepdim=True)
            elif mode == "fitref-split": mu[rows] = h[fit_rows].mean(0, keepdim=True)
            else:
                for e, tr in task_rows.items():
                    rr = [r for r in tr if r in rs]
                    if mode == "task": mu[rr] = h[rr].mean(0, keepdim=True)
                    elif mode == "task-loo": mu[rr] = (h[rr].sum(0, keepdim=True) - h[rr]) / max(len(rr) - 1, 1)
                    elif mode == "fitref-task":
                        fr = [r for r in tr if split_of[r] == "fit"]
                        if s_ == "fit": mu[rr] = (h[fr].sum(0, keepdim=True) - h[rr]) / max(len(fr) - 1, 1)   # fit scored leave-one-out
                        else: mu[rr] = h[fr].mean(0, keepdim=True)
        return (h - alpha * mu) @ LENS[name][L].cuda().float().T
    return fn
keep = {}
for label, mode, a in [("plain", None, 0), ("minus scored-split mean (task-blind)", "split", 1.0), ("minus scored-split same-task mean", "task", 1.0),
                       ("minus scored-split same-task mean x0.5", "task", 0.5), ("minus same-task mean, leave-one-out", "task-loo", 1.0),
                       ("minus fit-ref mean (task-blind)", "fitref-split", 1.0), ("minus fit-ref same-task mean (fit: LOO)", "fitref-task", 1.0),
                       ("minus fit-ref same-task mean x0.5", "fitref-task", 0.5)]:
    keep[label] = report(label, vec_fn=(centred(mode, a) if mode else (lambda name, L: RO(name, L))))
print("\n   examples: association dev, J top-8 plain vs minus fit-ref same-task mean")
fnc = centred("fitref-task", 1.0)
for r in [r for r in task_rows["association"] if split_of[r] == "dev"][:8]:
    i = idx[r]; key = list(b.candidates[i])[0]
    print(f"   [{b.item_names[i]}] {key}: J best rank {keep['plain']['J'][(i, key)]} -> {keep['minus fit-ref same-task mean (fit: LOO)']['J'][(i, key)]};"
          f" J++ {keep['plain']['JPP'][(i, key)]} -> {keep['minus fit-ref same-task mean (fit: LOO)']['JPP'][(i, key)]}")
    for L in (9, 18):
        for n in LENS:
            print(f"       L{L} {n:3s} plain:   {topk_str(RO(n, L, [r]))}\n       L{L} {n:3s} centred: {topk_str(fnc(n, L)[r:r+1])}")
print("\n== 5b. remove top-k singular directions from the lens matrix (data-free)")
for layers, ll in ((None, "all layers"), (LATE2, "L18+L21 only")):
    for k in (1, 2, 3, 5, 10, 20):
        mats = {}
        for (n, L), (U, S, Vh) in SV.items():
            Mx = LENS[n][L].cuda().float()
            mats[(n, L)] = Mx - (U[:, :k] * S[:k]) @ Vh[:k] if (layers is None or L in layers) else Mx
        report(f"minus top-{k} singular dirs, {ll}", mats=mats)
print("\n== 5c. project top-k directions of a FIT reference set out of the readout (qual3 a)")
def make_edit(ref, k, layers, thr=None):
    cache = {}
    def edit(name, L, v):
        if layers is not None and L not in layers: return v
        if (name, L) not in cache:
            rr = [r for r in fit_rows if ref == "all" or b.eval_slugs[idx[r]] == ref]
            X = RO(name, L, rr); conc = float(t.cosine_similarity(X, X.mean(0, keepdim=True), -1).mean())
            cache[(name, L)] = (t.linalg.svd(X / X.norm(dim=-1, keepdim=True), full_matrices=False).Vh[:k], conc)
        U, conc = cache[(name, L)]
        if thr is not None and conc < thr: return v
        return v - (v @ U.T) @ U
    return edit
for ref in ("association", "all"):
    for k in (1, 3):
        for layers, ll in ((None, "all layers"), (LATE2, "L18+L21")):
            report(f"ref=fit {ref}, k={k}, {ll}", edit=make_edit(ref, k, layers))
for thr in (0.9, 0.8):
    report(f"ref=fit all, k=1, only where fit concentration > {thr}", edit=make_edit("all", 1, None, thr))
print("\n== 5d. gated removal of u = unit mean of FIT association readouts, only rows with cos(readout, u) > tau (qual4 part 3)")
def gated(tau, layers):
    cache = {}
    def edit(name, L, v):
        if layers is not None and L not in layers: return v
        if (name, L) not in cache:
            X = RO(name, L, [r for r in fit_rows if b.eval_slugs[idx[r]] == "association"]); u = X.mean(0); cache[(name, L)] = u / u.norm()
        u = cache[(name, L)]; c = t.cosine_similarity(v, u[None], -1)
        return t.where((c > tau)[:, None], v - (v @ u)[:, None] * u[None], v)
    return edit
for tau in (0.7, 0.8, 0.9):
    for layers, ll in ((None, "all layers"), (LATE2, "L18+L21"), (LATE3, "L15-21")):
        report(f"gated tau={tau}, {ll}", edit=gated(tau, layers))
print("\n   share of rows with cos > 0.8 to u, by task and layer")
for n in LENS:
    for L in LAYERS:
        X = RO(n, L, [r for r in fit_rows if b.eval_slugs[idx[r]] == "association"]); u = X.mean(0); u = u / u.norm()
        c = t.cosine_similarity(RO(n, L), u[None], -1)
        print(f"   {n:3s} L{L:2d}: " + "  ".join(f"{e[:6]} {float((c[rows] > 0.8).float().mean()):.2f}" for e, rows in sorted(task_rows.items())), flush=True)

# ================= 6. asked vs read (association)
FS = ("Passage: The whistle blew and twenty-two players ran onto the wet grass while the crowd roared.\nOne-word topic: football\n\n"
      "Passage: She blew out the candles while everyone sang, then cut the first slice.\nOne-word topic: birthday\n\n"
      "Passage: He adjusted the telescope and wrote down the position of the red planet again.\nOne-word topic: astronomy\n\n"
      "Passage: {p}\nOne-word topic:")
plain = keep["plain"]; asked = {}; asked_top = {}
for r in arows:
    i = idx[r]
    lg = hf(tok(FS.format(p=text[("association", b.item_names[i])]), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()).logits[0, -1].float().masked_fill(punct, NEG)
    asked_top[i] = " ".join(strs[j].strip() for j in lg.topk(3).indices.tolist())
    for key, ids in b.candidates[i].items(): asked[(i, key)] = int((lg > lg[ids].max()).sum()) + 1
print(f"\n== 6. association: model names the concept when asked (top 10) x lens reads it (top 10 at any layer); fit+dev, n={len(asked)}")
print(f"   asked pass@10 {sum(v <= 10 for v in asked.values()) / len(asked):.2f}  pass@1 {sum(v <= 1 for v in asked.values()) / len(asked):.2f}")
for n in LENS:
    c = collections.Counter((asked[k] <= 10, plain[n][k] <= 10) for k in asked)
    print(f"   {n:3s}: asked&read {c[(True, True)]}  asked only {c[(True, False)]}  read only {c[(False, True)]}  neither {c[(False, False)]}")
for lab, f in (("asked & J++ read", lambda k: asked[k] <= 10 and plain['JPP'][k] <= 10), ("asked only (J++)", lambda k: asked[k] <= 10 and plain['JPP'][k] > 10),
               ("J++ read only", lambda k: asked[k] > 10 and plain['JPP'][k] <= 10), ("neither (J++)", lambda k: asked[k] > 10 and plain['JPP'][k] > 10)):
    print(f"   {lab} (concept asked-rank/J++ best rank [model top-3 when asked]): " + "; ".join(f"{k[1]} {asked[k]}/{plain['JPP'][k]} [{asked_top[k[0]]}]" for k in asked if f(k)))
print("done")
