"""Qualitative follow-ups on lens failures (FIT + DEV items only; test never loaded into any statistic).
A. Right concept, wrong language: does the lens name the concept in Chinese?
B. Where does the '@' / sports-team junk in J++ L24/L28 come from (is it a sentence-final-period thing)?
C. Does the model name the association concept when asked with the full prompt?  Poetry: rhyme word at prompt end."""
import json, collections, statistics
from pathlib import Path
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
from olens.train import wikitext_windows
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
t.set_grad_enabled(False)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer; hf = model._hf_model
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
idx = [i for i in range(len(b.item_names)) if (b.eval_slugs[i], b.item_names[i]) in part]
J = load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=M).jacobians_L_dict_FN
P = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
LENS = {"J": J, "JPP": P}
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
def is_cjk(s): return any('一' <= ch <= '鿿' or '぀' <= ch <= 'ヿ' for ch in s)
cjk = t.tensor([is_cjk(s) for s in strs]).cuda()
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
print(f"vocab {Vn}; CJK tokens {int(cjk.sum())}; items fit+dev {len(idx)}", flush=True)

# ---------- A. translate each concept key into Chinese with the model itself
keys = sorted({k for i in idx for k in b.candidates[i]})
FEW = "English: night\nChinese: 夜晚\nEnglish: river\nChinese: 河流\nEnglish: Germany\nChinese: 德国\nEnglish: {w}\nChinese:"
STOP = set("的 一 是 了 在 和 有 不 人 这 中 大 为 上 个 我 以 要 他 时 来 用 们 生 到 作 地 于 出 就 分 对 成 会 可 主 发 年 动 同 工 也 能 下 过 子 说 产 种 面 而 方 后 多 定 行 学 法 所".split())
by_text = collections.defaultdict(list)
for i, s in enumerate(strs):
    if is_cjk(s): by_text[s.strip()].append(i)
zh = {}
for w in keys:
    ids = tok(FEW.format(w=w), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    alts, cur = [], ids
    out = []
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
    forms = {T, T + "的", "的" + T} | set(alts) if T else set(alts)
    forms = {f for f in forms if f and f not in STOP}
    zh[w] = (T, sorted({j for f in forms for j in by_text.get(f, [])}))
print("\n== A0. model's own translations (concept -> greedy translation | accepted single tokens)")
print("; ".join(f"{w}->{T}|{'/'.join(sorted({strs[j].strip() for j in ids}))}" for w, (T, ids) in zh.items()), flush=True)

res = collections.defaultdict(dict)      # (lens, scoring) -> (item, key) -> best rank
bylayer = collections.defaultdict(collections.Counter)
zh_hit_tok = {}
for L in LAYERS:
    h = b.resid[L][idx].cuda().float()
    for name, Mx in LENS.items():
        v = h @ Mx[L].cuda().float().T
        for s in range(0, len(idx), 32):
            lg = model.unembed(v[s:s+32]).float().masked_fill(punct, float("-inf"))
            lg2 = lg.masked_fill(cjk, float("-inf"))
            for r in range(lg.shape[0]):
                i = idx[s + r]
                for key, ids in b.candidates[i].items():
                    en = [c for c in ids if not punct[c]]
                    zc = zh[key][1]
                    r_en = int((lg[r] > lg[r, en].max()).sum()) + 1 if en else 10**9
                    r_zh = int((lg[r] > lg[r, zc].max()).sum()) + 1 if zc else 10**9
                    r_nc = int((lg2[r] > lg2[r, en].max()).sum()) + 1 if en else 10**9
                    for sc, rk in (("en", r_en), ("zh", r_zh), ("en|zh", min(r_en, r_zh)), ("en,cjk masked", r_nc)):
                        d = res[(name, sc)]; d[(i, key)] = min(d.get((i, key), 10**9), rk)
                        bylayer[(name, sc, L)][b.eval_slugs[i]] += rk <= 10
                    if r_zh <= 10 and (name, i, key) not in zh_hit_tok:
                        zh_hit_tok[(name, i, key)] = (L, strs[zc[int(lg[r, zc].argmax())]].strip())
def macro(d, k, split):
    item = collections.defaultdict(list)
    for (i, key), r in d.items():
        if part[(b.eval_slugs[i], b.item_names[i])] == split: item[i].append(r <= k)
    ev = collections.defaultdict(list)
    for i, hs in item.items(): ev[b.eval_slugs[i]].append(sum(hs) / len(hs))
    return sum(sum(v) / len(v) for v in ev.values()) / len(ev), {e[:6]: round(sum(v) / len(v), 3) for e, v in sorted(ev.items())}
print("\n== A1. macro pass@10 under four scorings (en = benchmark; zh = Chinese forms only; en|zh = either; en with CJK tokens masked)")
for split in ("fit", "dev"):
    for name in LENS:
        for sc in ("en", "zh", "en|zh", "en,cjk masked"):
            p10, per = macro(res[(name, sc)], 10, split)
            print(f"  {split} {name:3s} {sc:14s} {p10:.4f}  {per}")
print("\n== A2. multilingual by intermediate slot (fit+dev pairs; share with best rank <= 10)")
for name in LENS:
    for sc in ("en", "zh", "en|zh"):
        c = collections.defaultdict(list)
        for (i, key), r in res[(name, sc)].items():
            if b.eval_slugs[i] == "multilingual" and len(b.candidates[i]) == 4:
                c[list(b.candidates[i]).index(key)].append(r <= 10)
        print(f"  {name:3s} {sc:6s} " + "  ".join(f"{lab} {sum(c[j])/len(c[j]):.2f} (n={len(c[j])})" for j, lab in enumerate(("language", "relation", "source", "answer"))))
print("\n== A3. pairs the benchmark scores as MISS at every layer but whose Chinese form is in the top 10")
for name in LENS:
    rows = [(b.eval_slugs[i], b.item_names[i], key, res[(name, 'en')][(i, key)], zh_hit_tok.get((name, i, key)))
            for (i, key), r in res[(name, "zh")].items() if r <= 10 and res[(name, "en")][(i, key)] > 10]
    tot = sum(r > 10 for r in res[(name, "en")].values())
    print(f"  {name}: {len(rows)} of {tot} missed pairs; by task {dict(collections.Counter(r[0] for r in rows))}")
    for e, n, key, ren, hit in rows[:45]: print(f"     {e[:6]}/{n}: {key} (en best rank {ren}) -> {hit}")
print("\n== A4. hits per layer (all tasks, fit+dev pairs): en / zh")
for name in LENS:
    print(f"  {name:3s} " + "  ".join(f"L{L}: {sum(bylayer[(name,'en',L)].values())}/{sum(bylayer[(name,'zh',L)].values())}" for L in LAYERS))
n_with_zh = sum(1 for (i, key) in res[("J", "en")] if zh[key][1]); print(f"  pairs with at least one Chinese single-token form: {n_with_zh} of {len(res[('J','en')])}", flush=True)


# ---------- text of each item (viewer export), convention check for hidden_states
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json")); RV = rd["vocab"]
meta = {(it["e"], it["n"]): it for it in rd["items"] if (it["e"], it["n"]) in part}
text = {k: "".join(RV[x] for x in it["toks"]) for k, it in meta.items()}
print("\n== convention check (cosine of hidden_states[k] at the readout position with the benchmark cache)")
for i in idx[:3]:
    k = (b.eval_slugs[i], b.item_names[i])
    for bos in (False, True):
        ids = tok(text[k], return_tensors="pt", add_special_tokens=bos)["input_ids"].cuda()
        hs = hf(ids, output_hidden_states=True).hidden_states
        p = meta[k]["pos"] + (ids.shape[1] - len(meta[k]["toks"]))
        print(f"  {k} bos={bos} len {ids.shape[1]} vs {len(meta[k]['toks'])} pos {p}: " + " ".join(
            f"L{L}: hs[L] {t.cosine_similarity(hs[L][0, p].float().cpu(), b.resid[L][i], 0):.3f} hs[L+1] {t.cosine_similarity(hs[L+1][0, p].float().cpu(), b.resid[L][i], 0):.3f}" for L in (12, 24)))
OFF = 1 if t.cosine_similarity(hs[25][0, p].float().cpu(), b.resid[24][i], 0) > t.cosine_similarity(hs[24][0, p].float().cpu(), b.resid[24][i], 0) else 0
print("  using hidden_states[L + %d]" % OFF, flush=True)

def top10(v): return model.unembed(v).float().masked_fill(punct, float("-inf")).topk(10, -1).indices
def show(v, n=8): return " | ".join(strs[j].strip() for j in top10(v)[0].tolist()[:n])
at = t.tensor([s.strip().startswith("@") for s in strs]).cuda()

# ---------- B2. where does '@' appear in WikiText?  paragraph-final vs mid-paragraph periods
win = wikitext_windows(tok, "validation", 400, 128, Path("/root/cache"), True).long()
acts = []
for s in range(0, len(win), 50):
    acts.append(hf(win[s:s+50].cuda(), output_hidden_states=True).hidden_states[24 + OFF].float().cpu())
acts = t.cat(acts)
is_p = t.tensor([[strs[x].strip() == "." for x in row] for row in win.tolist()])
nl_next = t.tensor([[("\n" in strs[row[j + 1]]) if j + 1 < len(row) else False for j in range(len(row))] for row in win.tolist()])
pos = t.arange(128)[None].expand_as(win)
P24 = P[24].cuda().float()
print("\n== B2. J++ L24: share of top-10 slots that start with '@'")
for g, m in (("WikiText period, next token is a newline (paragraph end)", is_p & nl_next), ("WikiText period, mid-paragraph", is_p & ~nl_next & (pos < 127)), ("WikiText other tokens", ~is_p)):
    hsel = acts[m][:3000].cuda(); tp = top10(hsel @ P24.T)
    print(f"   {g:58s} n={int(m.sum()):5d}  {float(at[tp].float().mean()):.3f}")
task_rows = collections.defaultdict(list)
for r, i in enumerate(idx): task_rows[b.eval_slugs[i]].append(r)
hb = b.resid[24][idx].cuda().float()
for e, rows in sorted(task_rows.items()):
    last = collections.Counter(text[(e, b.item_names[idx[r]])][-1] for r in rows).most_common(3)
    print(f"   benchmark {e:14s} n={len(rows):3d} '@' share {float(at[top10(hb[rows] @ P24.T)].float().mean()):.3f}   last prompt chars {last}")
# what the '@' direction looks like: which UNFILTERED tokens does J++ L24 put on top for association?
lgu = model.unembed(hb[task_rows['association']] @ P24.T).float()
c = collections.Counter(strs[j] for row in lgu.topk(5, -1).indices.tolist() for j in row)
print("   J++ L24 unfiltered top-5 tokens on association: " + ", ".join(f"{k!r}x{n}" for k, n in c.most_common(12)))
# is it one direction?  cosine between J++ L24 readout vectors of association prompts and their mean
va = hb[task_rows['association']] @ P24.T; vm = va.mean(0)
print(f"   cos(J++ h, mean J++ h) over association prompts at L24: mean {float(t.cosine_similarity(va, vm[None], -1).mean()):.3f};"
      f" same for WikiText periods vs their mean: {float(t.cosine_similarity(acts[is_p][:2000].cuda() @ P24.T, (acts[is_p][:2000].cuda() @ P24.T).mean(0)[None], -1).mean()):.3f}", flush=True)

# ---------- D. subtract the shared (context-class) part before reading
def rank_table(vec_fn):
    """vec_fn(name, L, rows) -> [len(rows), d] final-basis vectors for bench rows (indices into idx)."""
    out = {}
    for name in LENS:
        best = {}
        for L in LAYERS:
            v = vec_fn(name, L)
            for s in range(0, len(idx), 32):
                lg = model.unembed(v[s:s+32]).float().masked_fill(punct, float("-inf"))
                for r in range(lg.shape[0]):
                    i = idx[s + r]
                    for key, ids in b.candidates[i].items():
                        rk = int((lg[r] > lg[r, ids].max()).sum()) + 1
                        best[(i, key)] = min(best.get((i, key), 10**9), rk)
        out[name] = best
    return out
split_rows = {s_: [r for r, i in enumerate(idx) if part[(b.eval_slugs[i], b.item_names[i])] == s_] for s_ in ("fit", "dev")}
def centred(mode, alpha):
    def fn(name, L):
        h = b.resid[L][idx].cuda().float(); mu = t.zeros_like(h)
        for s_, rows in split_rows.items():          # means are taken within the split being scored (as a lens would see it)
            if mode == "split":
                mu[rows] = h[rows].mean(0, keepdim=True)
            elif mode == "task":
                for e, tr in task_rows.items():
                    rr = [r for r in tr if r in set(rows)]
                    mu[rr] = h[rr].mean(0, keepdim=True)
            elif mode == "task-loo":
                for e, tr in task_rows.items():
                    rr = [r for r in tr if r in set(rows)]
                    sm = h[rr].sum(0, keepdim=True)
                    mu[rr] = (sm - h[rr]) / (len(rr) - 1)
        return (h - alpha * mu) @ LENS[name][L].cuda().float().T
    return fn
print("\n== D. read lens(h - alpha * mean activation): macro pass@10 / pass@1, per-task pass@10  (label-free but transductive: the mean is over the scored prompts)")
variants = [("plain", None, 0), ("minus split mean (task-blind)", "split", 1.0), ("minus split mean x0.5", "split", 0.5), ("minus same-task mean", "task", 1.0),
            ("minus same-task mean x0.5", "task", 0.5), ("minus same-task mean, leave-one-out", "task-loo", 1.0)]
keep = {}
for label, mode, a in variants:
    tb = rank_table(centred(mode, a) if mode else (lambda name, L: b.resid[L][idx].cuda().float() @ LENS[name][L].cuda().float().T))
    keep[label] = tb
    for name in LENS:
        for s_ in ("fit", "dev"):
            p10, per = macro(tb[name], 10, s_); p1, _ = macro(tb[name], 1, s_)
            print(f"  {name:3s} {s_} {label:38s} {p10:.4f} / {p1:.4f}  {per}", flush=True)
print("\n   examples: association, J++ top-8 plain vs minus same-task mean (leave-one-out)")
fnp = lambda L: b.resid[L][idx].cuda().float() @ P[L].cuda().float().T
fnc = centred("task-loo", 1.0)
shown = 0
for r in task_rows["association"]:
    i = idx[r]
    if shown >= 14: break
    shown += 1
    key = list(b.candidates[i])[0]
    print(f"   [{b.item_names[i]}] concept {key}: best rank plain {keep['plain']['JPP'][(i,key)]} -> centred {keep['minus same-task mean, leave-one-out']['JPP'][(i,key)]}")
    for L in (12, 20, 28):
        print(f"       L{L} plain:   {show(fnp(L)[r:r+1])}\n       L{L} centred: {show(fnc('JPP', L)[r:r+1])}")
print("\n   examples: poetry, J++ L20/L28 centred")
for r in task_rows["poetry"][:6]:
    i = idx[r]; key = list(b.candidates[i])[0]
    print(f"   [{b.item_names[i]}] concept {key}: best rank plain {keep['plain']['JPP'][(i,key)]} -> centred {keep['minus same-task mean, leave-one-out']['JPP'][(i,key)]}  L20: {show(fnc('JPP', 20)[r:r+1])}  || L28: {show(fnc('JPP', 28)[r:r+1])}")

# ---------- C2. does the model name the association concept when asked in a one-word format?
FS = ("Passage: The whistle blew and twenty-two players ran onto the wet grass while the crowd roared.\nOne-word topic: football\n\n"
      "Passage: She blew out the candles while everyone sang, then cut the first slice.\nOne-word topic: birthday\n\n"
      "Passage: He adjusted the telescope and wrote down the position of the red planet again.\nOne-word topic: astronomy\n\n"
      "Passage: {p}\nOne-word topic:")
hits10 = hits1 = n = 0; ex = []
for r in task_rows["association"]:
    i = idx[r]
    ids = tok(FS.format(p=text[("association", b.item_names[i])]), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    lg = hf(ids).logits[0, -1].float().masked_fill(punct, float("-inf"))
    for key, cids in b.candidates[i].items():
        rk = int((lg > lg[cids].max()).sum()) + 1; n += 1; hits10 += rk <= 10; hits1 += rk <= 1
        ex.append(f"{key}:{rk}[{' '.join(strs[j].strip() for j in lg.topk(3).indices.tolist())}]")
print(f"\n== C2. association, few-shot one-word question with the full passage: pass@10 {hits10/n:.2f}  pass@1 {hits1/n:.2f} (n={n})\n   " + "; ".join(ex[:40]), flush=True)
