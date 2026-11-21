"""sub1 pilot: concept list (9B few-shot classifier), generation in 3 inferential styles, inferable/shuffled ranks.
Stages (each resumable): classify -> concepts.json ; gen -> gen_raw.jsonl ; measure new + 3,000 al2 -> pilot.jsonl."""
import json, os, random, re, sys, time, zlib
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/al2")
from al2_common import log, retry, load_model, cut as al2_cut, names, MAXNEW as AL2_MAXNEW  # noqa: E402

R = "/workspace/o-lens-improvement"
OUT = f"{R}/.hf27/sub1"
os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)

TYPES = ["PERSON", "PLACE", "STATE", "EVENT", "THING", "OTHER"]
FEWSHOT_WORDS = ("swimming climbing Einstein fever fuel lighthouse Iceland orchestra allergy Tolstoy laundry compass "
                 "Galileo Cleopatra magnet vaccine telescope Beethoven sunburn hiccups haircut").split()
CLS_EXTRA = ["Kenya", "Everest"]  # extra PLACE shots for the classifier only; also removed from the concept list
CLS_HEAD = ("Classify each word.\nPERSON: a famous person, by the name people use.\n"
            "PLACE: a country, city, region or landmark.\nSTATE: a condition or emotion of a person.\n"
            "EVENT: an occasion or activity.\nTHING: a concrete object, substance or animal.\n"
            "OTHER: anything else: function words, adjectives, adverbs, abstract nouns, ordinary first names, "
            "word fragments.\n\n")
CLS_SHOTS = [("Einstein", "PERSON"), ("swimming", "EVENT"), ("beyond", "OTHER"), ("fever", "STATE"),
             ("lighthouse", "THING"), ("Iceland", "PLACE"), ("quickly", "OTHER"), ("Tolstoy", "PERSON"),
             ("laundry", "EVENT"), ("allergy", "STATE"), ("compass", "THING"), ("Michael", "OTHER"),
             ("Kenya", "PLACE"), ("Galileo", "PERSON"), ("strain", "OTHER"), ("haircut", "EVENT"),
             ("magnet", "THING"), ("justice", "OTHER"), ("Cleopatra", "PERSON"), ("sunburn", "STATE"),
             ("Everest", "PLACE"), ("orchestra", "THING"), ("however", "OTHER"), ("climbing", "EVENT"),
             ("Beethoven", "PERSON"), ("telescope", "THING"), ("green", "OTHER"), ("hiccups", "STATE"),
             ("vaccine", "THING"), ("fuel", "THING")]
CONF = 0.8  # keep if argmax label in the 5 types and its share of the 6-label probability >= CONF

NO = "Never name it, any form of it, or a close synonym"
HEADS = {
    "I1": ("Write one or two sentences showing a moment whose most natural explanation is the given concept "
           f"(a person, place, condition, event or thing). {NO}, and never describe or define it: show only what "
           "happens, so that a reader can infer it.\n\n"),
    "I2": ("Write one sentence that points at the given concept through a fact or a relation, and stop just before "
           f"the concept would be named. For a condition, event or thing, use a consequence instead of a fact. {NO}, "
           "and never define it.\n\n"),
    "I3": ("Write one line of dialogue, in quotation marks, spoken by someone for whom the given concept is the "
           f"obvious unstated subject. {NO}, and use no word that describes it: the listener must infer it from the "
           "situation.\n\n"),
}
LABEL = {"I1": "Scene", "I2": "Sentence", "I3": "Line"}
MAXNEW = {"I1": 80, "I2": 56, "I3": 56}
SHOTS = {
    "I1": [("sunburn", "By evening her shoulders were too tender for the straps of her dress, and she slept on her stomach."),
           ("Beethoven", "When the last chord faded the old man was still facing the musicians, and a young singer had to turn him around so he could see the hall on its feet, clapping for a symphony he had never heard."),
           ("Iceland", "We soaked in a steaming blue lagoon at midnight while the sun refused to set, and the next morning our guide showed us the crack where two continents pull apart."),
           ("haircut", "He came home with cold ears and a sharp line across the back of his neck, and his daughter laughed and rubbed the short bristles on his head.")],
    "I2": [("Einstein", "The patent clerk who showed that mass and energy are the same thing later fled to Princeton, where"),
           ("Cleopatra", "The queen who had children with both Julius Caesar and Mark Antony, and who is said to have let a snake bite her, was"),
           ("hiccups", "Every few seconds his shoulders jumped and a small squeak escaped him, so his grandmother told him to hold his breath and drink a glass of water upside down to stop the"),
           ("vaccine", "Two days after the nurse pressed a cotton ball to his arm he had a sore shoulder, but his yellow card had a new stamp, all because of the")],
    "I3": [("allergy", "\"Did anyone check whether these cookies have nuts in them? My lips are starting to tingle.\""),
           ("Galileo", "\"Just say the earth stands still, old man, kneel before the cardinals, and you can go home to your lenses.\""),
           ("laundry", "\"Who put a red sock in with my white shirts? Everything came out pink!\""),
           ("compass", "\"If the needle says that way is north, then the cabin is behind us and we've been walking in circles for an hour.\"")],
}
Q_SHOTS = [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
           ("The physicist who wrote that energy equals mass times the speed of light squared worked as a patent clerk when", "Einstein"),
           ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
           ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")]
Q_HEAD = "".join(f"Passage: {t}\nUnstated word: {a}\n\n" for t, a in Q_SHOTS)
END_RE = re.compile(r"[.!?][\"”’']?(?=\s|$)")
WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def gen_prompt(st, w):
    s = HEADS[st]
    for c, sc in SHOTS[st]:
        s += f"Concept: {c}\n{LABEL[st]}: {sc}\n\n"
    return s + f"Concept: {w}\n{LABEL[st]}:"


def cut_new(text, st):
    if st == "I3":
        return al2_cut(text, 4)
    s = text.split("\n")[0].strip()
    if not s:
        return None
    if st == "I1":
        ms = list(END_RE.finditer(s))
        return s[:ms[min(2, len(ms)) - 1].end()] if ms else None
    if "\n" not in text:  # I2: must finish within max tokens
        return None
    ms = list(END_RE.finditer(s))
    return s[:ms[0].end()] if ms else s


def prefix_leak(sent, w):
    p = w.lower()[:5]
    return any(x.lower().startswith(p) for x in WORD_RE.findall(sent))


def shuffled(sent):
    ws = WORD_RE.findall(sent)
    random.Random(zlib.crc32(sent.encode())).shuffle(ws)
    return " ".join(ws)


def excluded_fn():
    ex = set(x.strip().lower() for x in open(f"{R}/runs/sub1/exclude_concepts.txt") if x.strip())
    ex |= {w.lower() for w in FEWSHOT_WORDS + CLS_EXTRA}

    def f(w):
        w = w.lower()
        vs = {w, w + "s", w + "es"}
        if w.endswith("s"):
            vs.add(w[:-1])
        if w.endswith("es"):
            vs.add(w[:-2])
        if w.endswith("ies"):
            vs.add(w[:-3] + "y")
        if w.endswith("y"):
            vs.add(w[:-1] + "ies")
        return bool(vs & ex)
    return f


model = load_model()
hf, tok = model._hf_model, model.tokenizer
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def last_logits(ids):
    x = torch.tensor(ids, device="cuda")
    return hf(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=1).logits[:, -1].float()


def by_len(items, enc, bs):
    g = {}
    for it in items:
        g.setdefault(len(enc[it]), []).append(it)
    return [v[i:i + bs] for _, v in sorted(g.items()) for i in range(0, len(v), bs)]


# ---------------- stage 1: classify ----------------
CPATH = f"{OUT}/concepts_all.json"
if not os.path.exists(CPATH):
    t0 = time.time()
    pat = re.compile(r"^ [A-Za-z]{4,}$")
    special = set(tok.all_special_ids)
    cand = []
    for i in range(len(tok)):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if pat.match(s) and i not in special:
            cand.append((i, s[1:]))
    log(f"candidates {len(cand)}")
    head = CLS_HEAD + "".join(f"Word: {w}\nType: {t}\n\n" for w, t in CLS_SHOTS)
    lab_ids = [tok(" " + t, add_special_tokens=False).input_ids[0] for t in TYPES]
    assert len(set(lab_ids)) == 6, lab_ids
    enc = {w: tok(head + f"Word: {w}\nType:", add_special_tokens=False).input_ids for _, w in cand}
    rows = []
    for b in by_len([w for _, w in cand], enc, 64):
        lg = retry(last_logits, [enc[w] for w in b])
        lp = torch.log_softmax(lg, -1)[:, lab_ids]
        mass = lp.exp().sum(-1)
        p = torch.softmax(lp, -1)
        for w, pi, m in zip(b, p.tolist(), mass.tolist()):
            k = max(range(6), key=lambda j: pi[j])
            rows.append(dict(word=w, type=TYPES[k], p=round(pi[k], 4), mass=round(m, 4)))
    json.dump(rows, open(CPATH, "w"))
    log(f"classified {len(rows)} in {time.time() - t0:.0f}s")
rows = json.load(open(CPATH))

KPATH = f"{OUT}/concepts.json"
if not os.path.exists(KPATH):
    exf = excluded_fn()
    best = {}
    for r in rows:
        if r["type"] == "OTHER" or r["p"] < CONF or exf(r["word"]):
            continue
        k = r["word"].lower()
        if k not in best or r["p"] > best[k]["p"]:
            best[k] = r
    keep = sorted(best.values(), key=lambda r: r["word"].lower())
    low = {r["word"].lower() for r in keep}
    keep = [r for r in keep if not (r["word"].lower().endswith("s") and r["word"].lower()[:-1] in low)]  # drop plural dup
    byt = {t: [r["word"] for r in keep if r["type"] == t] for t in TYPES[:5]}
    rng = random.Random(0)
    pilot = {t: rng.sample(v, min(200, len(v))) for t, v in byt.items()}
    json.dump(dict(rule=f"argmax label != OTHER and p(label | 6 labels) >= {CONF}; exclusions + few-shot words removed; "
                        "case-dupes keep higher p; plural dropped if singular present",
                   by_type=byt, pilot=pilot), open(KPATH, "w"), indent=0)
    log("concepts per type " + str({t: len(v) for t, v in byt.items()}))
K = json.load(open(KPATH))
ctype = {w: t for t, ws in K["pilot"].items() for w in ws}

# ---------------- stage 2: generate ----------------
GPATH = f"{OUT}/gen_raw.jsonl"
STY = ["I1", "I2", "I3"]
jobs = [(w, s) for w in ctype for s in STY]
done = set()
if os.path.exists(GPATH):
    for line in open(GPATH):
        try:
            o = json.loads(line)
            done.add((o["word"], o["style"]))
        except Exception:
            pass
todo = [j for j in jobs if j not in done]
log(f"gen jobs {len(jobs)} todo {len(todo)}")
if todo:
    nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    eos = nl + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    genc = {j: tok(gen_prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in todo}
    groups = {}
    for j in todo:
        groups.setdefault((j[1], len(genc[j])), []).append(j)
    bs = [v[i:i + 48] for _, v in sorted(groups.items()) for i in range(0, len(v), 48)]
    t0 = time.time()

    def run(b):
        torch.manual_seed(zlib.crc32(f"sub1|{b[0][0]}|{b[0][1]}".encode()))
        ids = torch.tensor([genc[j] for j in b], device="cuda")
        out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW[b[0][1]],
                          do_sample=True, temperature=0.9, top_p=0.95, top_k=0, num_return_sequences=3,
                          pad_token_id=pad, eos_token_id=eos)
        return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    with open(GPATH, "a") as f:
        for bi, b in enumerate(bs):
            texts = retry(run, b)
            for i, (w, s) in enumerate(b):
                for t in texts[i * 3:(i + 1) * 3]:
                    f.write(json.dumps(dict(word=w, style=s, type=ctype[w], text=t)) + "\n")
            f.flush()
            if bi % 10 == 0:
                log(f"gen batch {bi + 1}/{len(bs)} {time.time() - t0:.0f}s")
    log(f"gen done {time.time() - t0:.0f}s")

# ---------------- stage 3: measure ----------------
items = []  # (id, src, style, word, type, sentence, reason)
for n, line in enumerate(open(GPATH)):
    o = json.loads(line)
    s = cut_new(o["text"], o["style"])
    why = "cut" if s is None else "names" if names(s, o["word"]) else "prefix5" if prefix_leak(s, o["word"]) else \
        "short" if len(tok(s, add_special_tokens=False).input_ids) < 6 else "ok"
    items.append(dict(id=f"n{n}", src="new", style=o["style"], word=o["word"], type=o["type"], raw=o["text"],
                      sent=s, why=why))
cls = {}
for r in rows:  # al2 word type = classifier argmax (any confidence), case-insensitive, prefer higher p
    k = r["word"].lower()
    if k not in cls or r["p"] > cls[k][1]:
        cls[k] = (r["type"], r["p"])
al2 = {k: [] for k in range(5)}
for n, line in enumerate(open(f"{R}/.hf27/al2/gen_raw.jsonl")):
    o = json.loads(line)
    s = al2_cut(o["text"], o["style"])
    if s is None or names(s, o["word"]) or len(tok(s, add_special_tokens=False).input_ids) < 6:
        continue
    al2[o["style"]].append((n, o["word"], s))
rng = random.Random(0)
for k in range(5):
    for n, w, s in rng.sample(al2[k], 600):
        t = cls.get(w.lower(), ("NA", 0))
        items.append(dict(id=f"a{n}", src="al2", style=f"S{k + 1}", word=w, type=t[0], cls_p=t[1], sent=s, why="ok",
                          p5=prefix_leak(s, w)))

PPATH = f"{OUT}/pilot.jsonl"
have = set()
if os.path.exists(PPATH):
    for line in open(PPATH):
        try:
            have.add(json.loads(line)["id"])
        except Exception:
            pass
meas = [it for it in items if it["why"] == "ok" and it["id"] not in have]
log(f"items {len(items)} ok {sum(it['why'] == 'ok' for it in items)} to measure {len(meas)}")


def forms(w):
    out = set()
    for v in {w, w.lower(), w[:1].upper() + w[1:]}:
        for f in (" " + v, v):
            ids = tok(f, add_special_tokens=False).input_ids
            if len(ids) == 1:
                out.add(ids[0])
    return sorted(out)


qenc = {}
for it in meas:
    qenc[(it["id"], 0)] = tok(Q_HEAD + f"Passage: {it['sent']}\nUnstated word:", add_special_tokens=False).input_ids
    it["shuf"] = shuffled(it["sent"])
    qenc[(it["id"], 1)] = tok(Q_HEAD + f"Passage: {it['shuf']}\nUnstated word:", add_special_tokens=False).input_ids
res = {}
byid = {it["id"]: it for it in meas}
t0 = time.time()
with open(PPATH, "a") as f:
    for bi, b in enumerate(by_len(list(qenc), qenc, 48)):
        lg = retry(last_logits, [qenc[k] for k in b])
        top = lg.topk(5, -1).indices.tolist()
        for row, k, tp in zip(lg, b, top):
            fi = forms(byid[k[0]]["word"])
            vals = row[fi]
            rank = int((row[None, :] > vals[:, None]).sum(-1).min().item()) + 1
            res.setdefault(k[0], {})[k[1]] = (rank, [tok.decode([t]) for t in tp[:3]])
            if len(res[k[0]]) == 2:
                it = byid[k[0]]
                o = {x: it[x] for x in ("id", "src", "style", "word", "type", "cls_p", "p5", "sent", "shuf") if x in it}
                o.update(rank=res[k[0]][0][0], rank_shuf=res[k[0]][1][0], top3=res[k[0]][0][1],
                         top3_shuf=res[k[0]][1][1])
                f.write(json.dumps(o) + "\n")
                del res[k[0]]
        f.flush()
        if bi % 50 == 0:
            log(f"measure batch {bi} {time.time() - t0:.0f}s")
json.dump([{k: it[k] for k in ("id", "style", "word", "type", "raw", "sent", "why")} for it in items if it["src"] == "new"],
          open(f"{OUT}/new_items.json", "w"))
log(f"measure done {time.time() - t0:.0f}s")
print("ALL_DONE")
