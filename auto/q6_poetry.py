"""q6: is the future rhyme word decodable from the NEWLINE that ends line 1 of a couplet (Qwen3.5-9B)?
Training data = the model's own couplets (line 1 sampled few-shot from couplets written here, line 2 greedy in the
benchmark prompt format).  Benchmark: FIT + DEV poetry items only (test items only used to EXCLUDE overlapping
line 1s from the generated set, never in a statistic).
Usage: python auto/q6_poetry.py [--gen-minutes 4.5]   (generation is cached in /root/q6_data.pt)"""
import json, collections, random, re, time, sys, os
import torch as t
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from olens.readout import Bench
R = "/workspace/o-lens-improvement"; M = "Qwen/Qwen3.5-9B"; LAYERS = [4, 8, 12, 16, 20, 24, 28]
DATA = "/root/q6_data.pt"
GEN_MIN = float(sys.argv[sys.argv.index("--gen-minutes") + 1]) if "--gen-minutes" in sys.argv else 4.5
t.set_grad_enabled(False); random.seed(0); t.manual_seed(0)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
model = get_hf_model(M, attn_implementation="sdpa"); tok = model.tokenizer; hf = model._hf_model
blocks = model.layers; emb = model._embed_tokens
WU = model._lm_head.weight  # [V, d]
b = Bench.load(f"{R}/artifacts/bench/qwen3.5-9b.pt")
sp = json.load(open(f"{R}/auto/split_9b.json"))
part = {tuple(x): k for k in ("fit", "dev") for x in sp[k]}
Vn = model.unembed(t.zeros(1, model.d_model).cuda()).shape[-1]
strs = [tok.decode([i]) for i in range(Vn)]
punct = t.zeros(Vn, dtype=t.bool).cuda(); punct[b.excluded_ids.cuda()] = True
rd = json.load(open("/workspace/fleet/projects/o-lens/site/data/readouts_9b.json")); RV = rd["vocab"]
allpoem = [it for it in rd["items"] if it["e"] == "poetry"]
def line1_of(text): return text.split("\n")[1].rstrip(",").strip()
bench_line1 = {line1_of("".join(RV[x] for x in it["toks"])).lower() for it in allpoem}   # exclusion only
pmeta = [it for it in allpoem if (it["e"], it["n"]) in part]                                # fit + dev only
NLID = tok("\n", add_special_tokens=False)["input_ids"]; assert len(NLID) == 1; NLID = NLID[0]
log(f"vocab {Vn}; benchmark poetry fit+dev {len(pmeta)}; newline id {NLID}")
HEAD = "A rhyming couplet:\n"

def positions(line1):
    """ids of HEAD+line1+',\n' and token positions of (last word, comma, newline); None if tokenisation is odd."""
    s = HEAD + line1 + ",\n"
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    if len(ids) < 4 or strs[ids[-1]] != "\n" or strs[ids[-2]] != ",": return None
    if not strs[ids[-3]].strip() or not strs[ids[-3]].strip()[-1].isalpha(): return None
    return ids, (len(ids) - 3, len(ids) - 2, len(ids) - 1)

# ================================================================ 1. generation
FEW = [("autumn", "The maple leaves turn gold and drift away", "And autumn winds announce the end of day."),
       ("the city", "Beneath the neon signs the traffic hums", "While somewhere in the dark a busker strums."),
       ("a lost dog", "He waited by the gate through rain and snow", "For footsteps that he loved so long ago."),
       ("baking bread", "She kneads the dough before the break of dawn", "And fills the kitchen till the chill is gone."),
       ("the ocean", "The tide rolls in with silver on its crest", "Then slips away to find a place to rest.")]
TOPICS = """friendship;winter;a broken clock;the moon;a train journey;first love;war;a garden;childhood;old age;a storm;
the desert;a lighthouse;mountains;a library;a wedding;a funeral;coffee;a cat;the stars;a river;fire;a soldier;a king;
a thief;money;a ghost;spring;summer;the forest;a bird;a horse;a ship;an island;the sun;rain;snow;a candle;music;
a violin;dancing;a painter;a poet;a farmer;harvest;a market;a festival;a hospital;a teacher;school;a letter;a phone call;
a mirror;a dream;sleep;night;morning;a bridge;a castle;a dragon;a knight;a witch;magic;a clockmaker;a sailor;a pirate;
a battle;peace;home;travel;a road;a car;a bicycle;a city at night;a village;a church bell;a prayer;hope;fear;anger;
grief;joy;a child;a mother;a father;a brother;a sister;a grandmother;an old photograph;a secret;a lie;the truth;
a promise;a journey home;a mountain climb;a fisherman;the wind;a wolf;a bear;a fox;an owl;a garden rose;a tree;
a seed;a well;a mine;a factory;a computer;a robot;space travel;a rocket;the future;the past;memory;time;a heartbeat;
a doctor;a nurse;a chef;a feast;wine;bread;a broken heart;a quarrel;forgiveness;a stranger;a crowd;silence;a song;
a drum;thunder;lightning;a volcano;an earthquake;the sea floor;a whale;a shark;a beach;a sandcastle;a kite;a balloon;
a game of chess;a football match;a race;a runner;a boxer;a gambler;a detective;a crime;a prison;freedom;a flag;
an empire;a queen;a beggar;a merchant;a sword;a shield;a tower;a desert caravan;a snowman;a fireplace;a cottage;
a garden gate;a spider;a bee;honey;a butterfly;a moth;a lantern;a map;treasure;gold;silver;a ring;a crown""".replace("\n", "").split(";")
TOPICS = [x.strip() for x in TOPICS if x.strip()]

def fewshot(topic):
    ex = random.sample(FEW, 3)
    return "Rhyming couplets on different topics.\n\n" + "".join(
        f"Topic: {tp}\n{HEAD}{a},\n{c}\n\n" for tp, a, c in ex) + f"Topic: {topic}\n{HEAD}"

def gen(ids_list, max_new, sample):
    """ids_list: list of equal-length id lists.  returns list of generated id lists (cut at the first newline)."""
    x = t.tensor(ids_list).cuda()
    out = hf.generate(input_ids=x, attention_mask=t.ones_like(x), max_new_tokens=max_new, do_sample=sample,
                      temperature=1.0 if sample else None, top_p=0.95 if sample else None, top_k=None,
                      eos_token_id=NLID, pad_token_id=NLID)
    res = []
    for row in out[:, x.shape[1]:].tolist():
        cut = []
        for j in row:
            cut.append(j)
            if "\n" in strs[j]: break
        res.append(cut)
    return res

def clean_line1(s):
    s = s.split("\n")[0].strip()
    s = re.sub(r"[\s,.;:!?\-—–]+$", "", s)
    if not re.fullmatch(r"[A-Za-z][A-Za-z' ,;\-]*[A-Za-z]", s): return None
    if not 4 <= len(s.split()) <= 14: return None
    return s

if not os.path.exists(DATA):
    # --- line 1s
    line1s, seen, nbat, t1 = [], set(), 0, time.time()
    topics = TOPICS * 10; random.shuffle(topics)
    for topic in topics:
        ids = tok(fewshot(topic), add_special_tokens=False)["input_ids"]
        outs = gen([ids] * 48, 24, True); nbat += 1
        for o in outs:
            s = clean_line1(tok.decode(o))
            if s is None or s.lower() in seen or s.lower() in bench_line1 or positions(s) is None: continue
            seen.add(s.lower()); line1s.append(s)
        if nbat in (1, 3): log(f"line-1 batch {nbat}: {(time.time()-t1)/nbat:.1f}s/batch of 48; distinct so far {len(line1s)}; e.g. {line1s[-3:]}")
        if time.time() - t1 > GEN_MIN * 60 or len(line1s) >= 3200: break
    log(f"line 1s: {len(line1s)} distinct from {nbat} batches ({time.time()-t1:.0f}s)")
    # --- residuals at last word / comma / newline + line 2 (greedy), bucketed by exact length (no padding)
    P_ = {s: positions(s) for s in line1s}
    buckets = collections.defaultdict(list)
    for s in line1s: buckets[len(P_[s][0])].append(s)
    recs, t2, done = [], time.time(), 0
    for n, ss in sorted(buckets.items()):
        for k in range(0, len(ss), 64):
            chunk = ss[k:k + 64]; ids = [P_[s][0] for s in chunk]
            hs = hf(t.tensor(ids).cuda(), output_hidden_states=True, use_cache=False).hidden_states
            H = t.stack([t.stack([hs[L + 1][:, p] for p in P_[chunk[0]][1]], 1) for L in LAYERS], 1).half().cpu()  # [B, 7, 3, d]
            outs = gen(ids, 16, False)
            for s, o, h in zip(chunk, outs, H): recs.append(dict(line1=s, gen=o, H=h))
            done += len(chunk)
            if done <= 64 or done % 640 < 64: log(f"line 2: {done}/{len(line1s)}  {(time.time()-t2)/done*64:.1f}s per 64")
    t.save(dict(line1=[r["line1"] for r in recs], gen=[r["gen"] for r in recs], H=t.stack([r["H"] for r in recs])), DATA)
    log(f"saved {DATA}: {len(recs)} couplets")

D = t.load(DATA)
# ---------------------------------------------------------------- filter couplets
def word_of(tid):
    s = strs[tid]; return s[1:] if s.startswith(" ") and s[1:].isalpha() else None
rows, recs = [], []
for k, (l1, g) in enumerate(zip(D["line1"], D["gen"])):
    toks = [j for j in g if strs[j].strip() != ""]
    while toks and not any(ch.isalnum() for ch in strs[toks[-1]]): toks.pop()          # strip trailing punctuation
    if not toks or "\n" not in "".join(strs[j] for j in g): continue                     # line 2 must end
    rw = word_of(toks[-1]); w1 = l1.split()[-1].lower().strip("'")
    if rw is None or rw.lower() == w1: continue
    rows.append(k); recs.append(dict(line1=l1, line2=tok.decode(g).strip(), w1=w1, rw=rw.lower(), tid=toks[-1]))
H = D["H"][rows]  # [N, 7, 3, d]
N = len(recs)
def rhymes(a, b, n): return len(a) >= n and len(b) >= n and a[-n:] == b[-n:]
r2 = sum(rhymes(r["w1"], r["rw"], 2) for r in recs) / N; r3 = sum(rhymes(r["w1"], r["rw"], 3) for r in recs) / N
print("\n== 1. data")
print(f"   generated couplets {len(D['line1'])}; kept (single-token ' word' rhyme word, != line-1 word) {N}")
print(f"   crude rhyme rate: last 2 letters shared {r2:.3f}; last 3 letters shared {r3:.3f}")
print(f"   distinct rhyme words {len({r['rw'] for r in recs})}; distinct line-1 last words {len({r['w1'] for r in recs})}")
for r in random.Random(1).sample(recs, 10): print(f"     {r['line1']},  /  {r['line2']}    [{r['w1']} -> {r['rw']}]")

# ================================================================ 2. probes
perm = list(range(N)); random.Random(0).shuffle(perm)
te = sorted(perm[:N // 5]); tr = sorted(perm[N // 5:])
tr_in = tr[:int(len(tr) * 0.8)]; tr_val = tr[int(len(tr) * 0.8):]                       # validation part for ridge
tid = t.tensor([r["tid"] for r in recs]).cuda()
WUf = WU.float()
# rhyme sets: suffix-3 rule over ' word' tokens + rhyme words seen with that line-1 word in TRAIN
wtok = {}
for j in range(Vn):
    w = word_of(j)
    if w is not None and len(w) >= 3: wtok.setdefault(w.lower()[-3:], []).append(j)
seen_with = collections.defaultdict(collections.Counter)
for i in tr: seen_with[recs[i]["w1"]][recs[i]["tid"]] += 1
def rhyme_set(w1, extra=()):
    s = {j for j in wtok.get(w1[-3:], []) if word_of(j).lower() != w1} if len(w1) >= 3 else set()
    return s | set(seen_with.get(w1, {})) | set(extra)
RS = [rhyme_set(r["w1"]) for r in recs]
print(f"\n   split: train {len(tr)} (ridge fit {len(tr_in)} / val {len(tr_val)}), held-out {len(te)}; mean rhyme-set size {sum(len(RS[i]) for i in te)/len(te):.0f} tokens")

def score_logits(lg, ids_rows):
    """lg [B,V] (punct masked) -> (rank of true token, top10 index rows)"""
    tgt = tid[ids_rows]
    rk = (lg > lg.gather(1, tgt[:, None])).sum(1) + 1
    return rk.cpu(), lg.topk(10, -1).indices.cpu()
def summarize(rk, top, ids_rows):
    rs = sum(bool(set(top[k].tolist()) & RS[i]) for k, i in enumerate(ids_rows)) / len(ids_rows)
    return float((rk <= 10).float().mean()), float((rk <= 1).float().mean()), rs
def lens_eval(Mx, L, pos, rows_):
    X = H[rows_, LAYERS.index(L), pos].cuda().float()
    lg = model.unembed(X @ Mx[L].cuda().float().T).float().masked_fill(punct, float("-inf"))
    return summarize(*score_logits(lg, rows_), rows_)

def ridge_fit(Xtr, Ytr, lam):
    mx, my = Xtr.mean(0), Ytr.mean(0); Xc = Xtr - mx
    K = Xc @ Xc.T
    A = t.linalg.solve(K + lam * t.eye(len(K), device=K.device, dtype=K.dtype), Ytr - my)
    W = Xc.T @ A                                                         # [d, d_out]
    return lambda X: (X - mx) @ W + my
def ridge_logits(f, X):
    return (f(X).float() @ WUf.T).masked_fill(punct, float("-inf"))
def ridge_eval(pos, L, shuffle=False, return_map=False):
    X = H[:, LAYERS.index(L), pos].cuda().double() if isinstance(pos, int) else H[:, LAYERS.index(L), pos].flatten(1).cuda().double()
    tgt_rows = tid.clone()
    if shuffle:
        sh = t.tensor(tr); tgt_rows[sh] = tid[sh[t.randperm(len(sh), generator=t.Generator().manual_seed(0))].cuda()]
    Y = WU[tgt_rows].double()
    scale = float((X[tr_in] - X[tr_in].mean(0)).pow(2).sum(1).mean())
    best = None
    for c in (1e-2, 3e-2, 0.1, 0.3, 1, 3, 10, 30, 100):
        f = ridge_fit(X[tr_in], Y[tr_in], c * scale)
        lg = ridge_logits(f, X[tr_val]).float()
        acc = float(((lg > lg.gather(1, tgt_rows[tr_val][:, None])).sum(1) + 1 <= 10).float().mean())
        if best is None or acc > best[0]: best = (acc, c)
    f = ridge_fit(X[tr], Y[tr], best[1] * scale)
    lg = ridge_logits(f, X[te]).float()
    out = summarize(*score_logits(lg, te), te) + (best[1],)
    return (out, f, scale) if return_map else out

# baselines (i) frequency, (ii) lookup on line-1 last word
freq = collections.Counter(recs[i]["tid"] for i in tr); flist = [j for j, _ in freq.most_common(10)]
def eval_list(lists):
    t10 = sum(recs[i]["tid"] in l[:10] for i, l in zip(te, lists)) / len(te)
    t1 = sum(recs[i]["tid"] == l[0] for i, l in zip(te, lists)) / len(te)
    rs = sum(bool(set(l[:10]) & RS[i]) for i, l in zip(te, lists)) / len(te)
    return t10, t1, rs
base_freq = eval_list([flist] * len(te))
look = []
for i in te:
    l = [j for j, _ in seen_with.get(recs[i]["w1"], collections.Counter()).most_common(10)]
    look.append((l + [j for j in flist if j not in l])[:10])
base_look = eval_list(look)
cov = sum(recs[i]["w1"] in seen_with for i in te) / len(te)

J = load_lens_file(f"{R}/runs/v6_9b/jlens9b.pt", hf_model_name=M).jacobians_L_dict_FN
P = load_lens_file(f"{R}/artifacts/baselines/jpp9b_prefit.pt", hf_model_name=M).jacobians_L_dict_FN
print("\n== 2. held-out couplets: top-10 / top-1 / rhyme-set@10 (any rhyming token in top 10), full vocab, punct masked")
print(f"   (i)  frequency top-10 (no input):             {base_freq[0]:.3f} / {base_freq[1]:.3f} / {base_freq[2]:.3f}")
print(f"   (ii) lookup line-1 last word -> train rhymes: {base_look[0]:.3f} / {base_look[1]:.3f} / {base_look[2]:.3f}   (line-1 word seen in train for {cov:.2f} of held-out)")
print(f"   {'layer':5s} | {'J++ @nl':17s} | {'J-lens @nl':17s} | {'ridge @nl':17s} | {'ridge @comma':17s} | {'ridge @lastword':17s} | {'ridge shuffled @nl':17s} | {'ridge word+nl':17s} | ridge lam/scale nl,comma,word")
fmt = lambda x: f"{x[0]:.3f}/{x[1]:.3f}/{x[2]:.3f}"
maps = {}
for L in LAYERS:
    jp = lens_eval(P, L, 2, te); jl = lens_eval(J, L, 2, te)
    rn, f, sc = ridge_eval(2, L, return_map=True); maps[L] = f
    rc = ridge_eval(1, L); rw = ridge_eval(0, L); rsh = ridge_eval(2, L, shuffle=True); rwn = ridge_eval([0, 2], L)
    print(f"   L{L:<4d} | {fmt(jp):17s} | {fmt(jl):17s} | {fmt(rn):17s} | {fmt(rc):17s} | {fmt(rw):17s} | {fmt(rsh):17s} | {fmt(rwn):17s} | {rn[3]:g},{rc[3]:g},{rw[3]:g}", flush=True)
# examples of the ridge map at the best layer
Lb = 20
X = H[te[:8], LAYERS.index(Lb), 2].cuda().double()
lg = ridge_logits(maps[Lb], X).float()
lj = model.unembed(H[te[:8], LAYERS.index(Lb), 2].cuda().float() @ P[Lb].cuda().float().T).float().masked_fill(punct, float("-inf"))
print(f"\n   examples, L{Lb} at the newline (held-out): ridge top-6 || J++ top-6")
for k, i in enumerate(te[:8]):
    print(f"     {recs[i]['w1']:>10s} -> {recs[i]['rw']:10s}: {' | '.join(strs[j].strip() for j in lg[k].topk(6).indices.tolist())}  ||  {' | '.join(strs[j].strip() for j in lj[k].topk(6).indices.tolist())}")

# ================================================================ 3. benchmark transfer (fit + dev poetry)
key2i = {(b.eval_slugs[i], b.item_names[i]): i for i in range(len(b.item_names))}
bi = [key2i[(it["e"], it["n"])] for it in pmeta]
btext = ["".join(RV[x] for x in it["toks"]) for it in pmeta]
bw1 = [line1_of(s).split()[-1].lower().strip("'") for s in btext]
# convention check on one item
ids0 = tok(btext[0], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
assert ids0.shape[1] == len(pmeta[0]["toks"]), (ids0.shape, len(pmeta[0]["toks"]))
hs0 = hf(ids0, output_hidden_states=True, use_cache=False).hidden_states
print("\n== 3. benchmark poetry (fit+dev, n=%d) at the newline (benchmark cache). convention cos(hs[L+1], resid[L]): %s" % (
    len(bi), " ".join(f"L{L} {float(t.cosine_similarity(hs0[L+1][0, pmeta[0]['pos']].float().cpu(), b.resid[L][bi[0]], 0)):.3f}" for L in (12, 24))))
print("   NOTE: the ridge map is a SUPERVISED probe trained on the model's own couplets (target = unembedding row of the rhyme word), not a Jacobian lens.")
def bench_rank(lg, i):
    return min(int((lg > lg[ids].max()).sum()) + 1 for ids in b.candidates[i].values())
print(f"   {'layer':5s} | {'J++ top10/top1/rhymeset':24s} | {'ridge top10/top1/rhymeset':24s}")
bres = {}
for L in LAYERS:
    X = b.resid[L][bi].cuda()
    lgj = model.unembed(X.float() @ P[L].cuda().float().T).float().masked_fill(punct, float("-inf"))
    lgr = ridge_logits(maps[L], X.double()).float()
    line = []
    for nm, lg in (("J++", lgj), ("ridge", lgr)):
        rk = [bench_rank(lg[k], i) for k, i in enumerate(bi)]
        rs = sum(bool(set(lg[k].topk(10).indices.tolist()) & rhyme_set(bw1[k], [j for ids in b.candidates[i].values() for j in ids])) for k, i in enumerate(bi)) / len(bi)
        bres[(nm, L)] = rk
        line.append(f"{sum(r <= 10 for r in rk)/len(rk):.3f}/{sum(r <= 1 for r in rk)/len(rk):.3f}/{rs:.3f}")
    print(f"   L{L:<4d} | {line[0]:24s} | {line[1]:24s}")
anyL = {nm: sum(min(bres[(nm, L)][k] for L in LAYERS) <= 10 for k in range(len(bi))) / len(bi) for nm in ("J++", "ridge")}
print(f"   best-over-layers top-10 (pass@10 style): J++ {anyL['J++']:.3f}  ridge {anyL['ridge']:.3f}")
for k in range(6):
    lg = ridge_logits(maps[20], b.resid[20][[bi[k]]].cuda().double()).float()[0]
    print(f"     [{pmeta[k]['n']}] ridge L20 top-6: {' | '.join(strs[j].strip() for j in lg.topk(6).indices.tolist())}")

# ================================================================ 4. causal check: transplant the newline column
_patch, _rec, handles = {}, {}, []
def _mk(k):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if "pos" in _rec: _rec[k] = {p: h[0, p].detach().clone() for p in _rec["pos"]}
        if k in _patch:
            h = h.clone()
            for p, v in _patch[k]: h[0, p] = v.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook
def install():
    handles.append(emb.register_forward_hook(_mk(0)))
    for j, bl in enumerate(blocks): handles.append(bl.register_forward_hook(_mk(j + 1)))
def remove():
    while handles: handles.pop().remove()
def run(ids, rec_pos=None, patch=None):
    _rec.clear(); _patch.clear()
    if rec_pos is not None: _rec["pos"] = rec_pos
    if patch: _patch.update(patch)
    install()
    try: lg = hf(ids, use_cache=False).logits[0, -1].float()
    finally: remove(); _patch.clear(); _rec.pop("pos", None)
    return lg, {k: v for k, v in _rec.items()}
def word(lg): return strs[int(lg.argmax())].strip().lower()
def concept(k): return list(b.candidates[bi[k]].keys())[0].lower()
n4 = min(30, len(bi)); sel = list(range(n4))
bids = [tok(btext[k], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda() for k in range(len(bi))]
recd, base, basep = {}, {}, {}
for k in sel:
    p = pmeta[k]["pos"]; assert strs[int(bids[k][0, p])] == "\n" and strs[int(bids[k][0, p - 1])] == ",", btext[k]
    lg, rc = run(bids[k], rec_pos=[p - 2, p - 1, p]); recd[k] = rc; base[k] = word(lg); basep[k] = (int(lg.argmax()), float(lg.softmax(-1).max()))
print(f"\n== 4. causal: transplant block outputs (embedding + all {len(blocks)} blocks) at the newline from a donor couplet (n={n4} fit+dev items)")
print(f"   unpatched next word == concept: {sum(base[k] == concept(k) for k in sel)}/{n4}")
for variant, offs in (("newline only", [0]), ("control: last word + comma + newline", [-2, -1, 0])):
    cnt = collections.Counter(); ex = []; pdrop = []
    for n_, k in enumerate(sel):
        d = sel[(n_ + 1) % n4]
        if concept(d) == concept(k) or bw1[d][-2:] == bw1[k][-2:]: d = sel[(n_ + 2) % n4]
        pk, pd = pmeta[k]["pos"], pmeta[d]["pos"]
        patch = {L: [(pk + o, recd[d][L][pd + o]) for o in offs] for L in recd[d]}
        lg, _ = run(bids[k], patch=patch); w = word(lg); pdrop.append(float(lg.softmax(-1)[basep[k][0]]) - basep[k][1])
        donor_set = {concept(d)} | ({word_of(j).lower() for j in wtok.get(bw1[d][-3:], [])} if len(bw1[d]) >= 3 else set())
        if w == base[k]: c = "unchanged"
        elif w in donor_set or rhymes(w, bw1[d], 2): c = "donor rhyme"
        else: c = "other"
        cnt[c] += 1
        if c != "unchanged": ex.append(f"{bw1[k]}->{base[k]} | donor {bw1[d]}->{concept(d)} | now {w} [{c}]")
    print(f"   {variant:38s}: unchanged {cnt['unchanged']}/{n4}  donor rhyme {cnt['donor rhyme']}/{n4}  other {cnt['other']}/{n4}")
    print(f"       mean change in prob of the unpatched top word: {sum(pdrop)/len(pdrop):+.3f} (unpatched mean {sum(v[1] for v in basep.values())/len(basep):.3f})")
    for e in ex: print(f"       {e}")
log("done")
