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
    for step in range(5):
        lg = hf(cur).logits[0, -1].float()
        if step == 0:
            pr = lg.softmax(-1); top = pr.topk(8)
            alts = [strs[j].strip() for p_, j in zip(top.values.tolist(), top.indices.tolist()) if p_ >= 0.05 and is_cjk(strs[j])]
        nx = int(lg.argmax()); s = strs[nx]
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

# ---------- B. the '@' / sports junk of J++: is it a property of sentence-final periods?
def fam(s):
    u = s.strip()
    if u.startswith("@") or u.lower() in ("at", "-at"): return "at"
    if u in ("Flames", "Raptors", "Vancouver", "Winnipeg", "Toronto", "Seattle", "GM", "Canucks", "Calgary", "Edmonton", "Oilers", "Jets", "Leafs", "Atlas", "Valk", "Val", "atk", "Little"): return "team"
    return None
famid = {f: t.tensor([fam(s) == f for s in strs]).cuda() for f in ("at", "team")}
def top10(v): return model.unembed(v).float().masked_fill(punct, float("-inf")).topk(10, -1).indices
def famshare(v):
    tp = top10(v); return {f: float(m[tp].float().mean()) for f, m in famid.items()}
win = wikitext_windows(tok, "validation", 400, 128, Path("/root/cache"), True).long()
acts = {L: [] for L in (20, 24, 28)}
for s in range(0, len(win), 50):
    hs = hf(win[s:s+50].cuda(), output_hidden_states=True).hidden_states
    for L in acts: acts[L].append(hs[L].float().cpu())
acts = {L: t.cat(v) for L, v in acts.items()}
ref = b.resid[24][idx[0]]
# hidden_states index convention check against the benchmark cache, using mean norms
print("\n== B. convention check: mean |h| WikiText hs[24] vs hs[25] vs benchmark resid[24]:",
      round(float(acts[24].norm(dim=-1).mean()), 1), round(float(hf(win[:8].cuda(), output_hidden_states=True).hidden_states[25].float().norm(dim=-1).mean()), 1),
      round(float(b.resid[24][idx].norm(dim=-1).mean()), 1))
period = t.tensor([strs[i].strip() == "." for i in range(Vn)])
is_p = period[win]; pos = t.arange(128)[None].expand_as(win)
groups = {"WikiText sentence-final '.' (pos>=16)": is_p & (pos >= 16), "WikiText other tokens (pos>=16)": (~is_p) & (pos >= 16)}
task_rows = collections.defaultdict(list)
for r, i in enumerate(idx): task_rows[b.eval_slugs[i]].append(r)
print("   share of J++ top-10 slots in the '@/at' family and the team/name family")
for L in (20, 24, 28):
    Mx = P[L].cuda().float()
    for g, m in groups.items():
        hsel = acts[L][m][:3000].cuda()
        print(f"   L{L} {g:40s} {famshare(hsel @ Mx.T)}")
    hb = b.resid[L][idx].cuda().float()
    for e, rows in sorted(task_rows.items()):
        print(f"   L{L} benchmark {e:30s} {famshare(hb[rows] @ Mx.T)}")
    hbar = acts[L][groups["WikiText sentence-final '.' (pos>=16)"]].mean(0, keepdim=True).cuda()
    print(f"   L{L} J++ of the MEAN WikiText period activation: " + " | ".join(strs[j].strip() for j in top10(hbar @ Mx.T)[0].tolist()))
    ha = hb[task_rows["association"]]
    print(f"   L{L} J++ of the MEAN association activation:    " + " | ".join(strs[j].strip() for j in top10(ha.mean(0, keepdim=True) @ Mx.T)[0].tolist()))
    print(f"   L{L} J++ of association minus its mean (first 3): " + " || ".join(" | ".join(strs[j].strip() for j in row) for row in top10((ha - ha.mean(0, keepdim=True)) @ Mx.T)[:3].tolist()))
    c = collections.Counter(strs[j].strip() for row in top10(acts[L][groups["WikiText sentence-final '.' (pos>=16)"]][:3000].cuda() @ Mx.T).tolist() for j in row)
    print(f"   L{L} most common J++ top-10 tokens at WikiText periods: " + ", ".join(f"{k}x{n}" for k, n in c.most_common(14)), flush=True)

# ---------- C. does the model name the concept when asked? (full prompt text from the viewer export)
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json")); RV = rd["vocab"]
text = {(it["e"], it["n"]): "".join(RV[x] for x in it["toks"]) for it in rd["items"] if (it["e"], it["n"]) in part}
SUF = {"about": "\n\nIn one word, this passage is about", "qa": "\n\nQuestion: Who or what is this passage describing? Answer with one word.\nAnswer:", "none (plain next token)": ""}
print("\n== C. concept in the model's own top-10 next tokens (punctuation masked), fit+dev")
for e in ("association", "poetry"):
    for sname, suf in SUF.items():
        hits10 = hits1 = n = 0; ex = []
        for i in idx:
            if b.eval_slugs[i] != e: continue
            ids = tok(text[(e, b.item_names[i])].rstrip(" ") + suf if e == "poetry" and not suf else text[(e, b.item_names[i])] + suf, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
            lg = hf(ids).logits[0, -1].float().masked_fill(punct, float("-inf"))
            for key, cids in b.candidates[i].items():
                rk = int((lg > lg[cids].max()).sum()) + 1; n += 1; hits10 += rk <= 10; hits1 += rk <= 1
                if len(ex) < 10: ex.append(f"{key}:{rk}[{' '.join(strs[j].strip() for j in lg.topk(4).indices.tolist())}]")
        print(f"  {e:11s} suffix={sname:24s} pass@10 {hits10/n:.2f}  pass@1 {hits1/n:.2f}  (n={n})  e.g. " + "; ".join(ex[:7]), flush=True)
