"""q1 (T4 diagnostics) shared helpers: model + f4 filter, the item sets (bench fit + dev, probes), ranking, prompts.
Dev/fit and probes only: test / held-out rows are never read. Run on the GPU pod (see scripts/q1/job.sh)."""
import importlib.util
import json
import sys

import torch

R = "/workspace/o-lens-improvement"
sys.path.insert(0, f"{R}/src")
import workspace_lens  # noqa: E402,F401  (before auto/candidates joins the path: it has its own jlens.py)
sys.path.append(f"{R}/auto/candidates")
TAGS = {"Qwen/Qwen3.5-9B": "9b", "Qwen/Qwen3.5-2B": "2b"}


def load_module(name, path, argv=None):
    old = sys.argv
    if argv is not None:
        sys.argv = argv
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


def setup(model_name="Qwen/Qwen3.5-9B"):
    from workspace_lens import get_hf_model
    from olens.readout import Bench
    torch.set_grad_enabled(False)
    ev = load_module("q1_ev", f"{R}/auto/evaluate.py")
    cfg = ev.MODELS[model_name]
    model = get_hf_model(model_name, attn_implementation="sdpa")
    fm = load_module("w1_filters", f"{R}/auto/filters.py")
    sys.modules["w1_filters"] = fm
    bench = Bench.load(f"{R}/{cfg['bench']}")
    FL = fm.Filters(model.tokenizer, model._lm_head.weight.shape[0], [bench])
    return model, cfg, bench, FL, FL.get("f4")


def load_items(model, bench, tag, probes_path=f"{R}/probes/probes_v1.json"):
    """Rows: one per (item, concept). Bench: fit + dev only, text = the harness text up to the readout token,
    ids = the benchmark's candidate ids. Probes: qual_dump.candidate_ids (whole forms, else first pieces)."""
    qd = load_module("q1_qd", f"{R}/auto/candidates/qual_dump.py")
    tok = model.tokenizer
    texts = qd.bench_texts(tag)
    split = json.load(open(f"{R}/auto/split_{tag}.json"))
    pos = {k: i for i, k in enumerate(zip(bench.eval_slugs, bench.item_names))}
    rows = []
    for set_name, key in (("bench-fit", "fit"), ("bench-dev", "dev")):
        for i in sorted(pos[tuple(k)] for k in split[key]):
            e, n, c = bench.eval_slugs[i], bench.item_names[i], bench.candidates[i]
            text, ans = texts[(e, n)]
            for concept, ids in c.items():
                rows.append(dict(id=f"{e}|{n}" + (f"#{concept}" if len(c) > 1 else ""), set=set_name, task=e,
                                 text=text, concept=concept, ids=[int(t) for t in ids], piece=[], bi=i))
    P = json.load(open(probes_path, encoding="utf-8"))["probes"]
    for p in P:
        aw, ap, forms, bad = qd.candidate_ids(tok, p.get("accepted") or [])
        rows.append(dict(id=p["id"], set="probes", task=p["category"], text=p["text"],
                         concept=(p.get("accepted") or [None])[0], ids=sorted(set(aw) | set(ap)), piece=list(ap),
                         accepted=p.get("accepted") or []))
    return rows, qd


class Scorer:
    """f4 word-group ranks / top lists; ids in `spare` are never masked (probe forms)."""

    def __init__(self, F, spare, dev="cuda"):
        ex = F.excluded_ids
        if spare:
            ex = ex[~torch.isin(ex, torch.as_tensor(sorted(spare), dtype=ex.dtype))]
        self.gid = F.group_ids.to(dev)
        self.ng = int(self.gid.max()) + 1
        V = self.gid.shape[0]
        self.mask = torch.zeros(V, dtype=torch.bool, device=dev)
        self.mask[ex.to(dev)] = True

    def masked(self, lg):
        lg = lg.float()
        V = self.mask.shape[0]
        lg = lg[..., :V]
        return lg.masked_fill(self.mask, float("-inf"))

    def rank(self, lg_masked, ids):
        """1 + number of word groups (other than the candidates' groups) whose best member beats the best candidate."""
        if not ids:
            return None
        ids_t = torch.as_tensor(ids, device=lg_masked.device)
        best = lg_masked[ids_t].max()
        if not torch.isfinite(best):
            return None
        gmax = torch.full((self.ng,), float("-inf"), device=lg_masked.device).scatter_reduce(
            0, self.gid, lg_masked, "amax")
        cg = self.gid[ids_t].unique()
        return int((gmax > best).sum() - (gmax[cg] > best).sum()) + 1

    def top(self, lg_masked, k=10):
        from w1_filters import group_topk
        return group_topk(lg_masked, self.gid, k)


class Stop(Exception):
    pass


@torch.no_grad()
def final_state(model, ids, patches=None):
    """ids: [B, T] long on the model device. Final-block output at the last position [B, d] (forward stopped there).
    patches: list of (layer, rows LongTensor, pos int, vecs [len(rows), d]) -> block output replaced in place."""
    last = model.n_layers - 1
    rec = {}
    hooks = []

    def fin(m, i, o):
        rec["y"] = (o if torch.is_tensor(o) else o[0])[:, -1].float()
        raise Stop

    for (L, rows, p, vecs) in (patches or []):
        def ph(m, i, o, rows=rows, p=p, vecs=vecs):
            t = o if torch.is_tensor(o) else o[0]
            t[rows, p] = vecs.to(t.dtype)
        hooks.append(model.layers[L].register_forward_hook(ph))
    hooks.append(model.layers[last].register_forward_hook(fin))
    try:
        model.forward(ids)
    except Stop:
        pass
    finally:
        for h in hooks:
            h.remove()
    return rec["y"]


@torch.no_grad()
def raw_states(model, text, layers):
    """Block outputs at the last token of the raw text (harness tokenization) for `layers` + final logits."""
    ids = model.encode(text)
    last = model.n_layers - 1
    want = sorted(set(layers) | {last})
    rec = {}
    hooks = [model.layers[l].register_forward_hook(
        (lambda l: lambda m, i, o: rec.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, -1].float().clone()))(l))
        for l in want]
    try:
        model.forward(ids)
    finally:
        for h in hooks:
            h.remove()
    return {l: rec[l] for l in layers}, model.unembed(rec[last][None]).float()[0], ids


def chat(tok, user, prefill):
    msgs = [{"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    return s + prefill


def quoted_prompt(tok, spec, text):
    """spec = (header, query prefix, label, examples): ask3/ask5 'chat' family (bare label prefilled)."""
    head, q, lab, ex = spec
    shots = head + "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in ex)
    return chat(tok, shots + f"{q}{text}\n{lab}", lab)


def topic_specs():
    m = load_module("q1_ask3", f"{R}/scripts/ask3/ask3_collect9.py", ["x", "Qwen/Qwen3.5-9B", "chat"])
    return m.W


def unstated_specs():
    m = load_module("q1_ask5", f"{R}/scripts/ask5/ask5_collect.py", ["x", "Qwen/Qwen3.5-9B", "chat"])
    return m.W


# New quoted questions (q3..q7), same chat / few-shot format as the A-lens targets. Rules: each example has a tempting
# answer at the wrong level that is not the gold; the gold never appears in the passage; answer types vary; nothing
# copied from or shaped after a benchmark item or a probe (checked against both at run time).
NEW_Q = {
    "q3_referent": ("", "Passage: ", "Who or what is meant? Name it:", [
        ("She finally stood on the summit, higher than anyone else on Earth, and the Sherpa beside her grinned.",
         "Everest"),                     # tempting: mountain / climbing / Nepal
        ("Grandpa still keeps it in the drawer, the little brass thing he wound every morning before work.",
         "watch"),                       # tempting: grandpa / time / drawer
        ("Unter dem Baum lagen die Geschenke, und die Kinder warteten auf den Mann mit dem weißen Bart.",
         "Weihnachtsmann"),              # tempting: Christmas / presents
        ("The composer had gone completely deaf, yet he still conducted the premiere of his ninth symphony.",
         "Beethoven")]),                 # tempting: music / symphony / deafness
    "q4_intermediate": ("", "Passage: ", "What must be worked out first? One word:", [
        ("The money they use in the homeland of sushi and sumo is called the", "Japan"),        # tempting: yen
        ("Le fleuve qui traverse la capitale de l'Angleterre s'appelle", "Londres"),            # tempting: Tamise
        ("The scientist who noticed mould killing his bacteria was born in the country of", "Fleming"),  # Scotland
        ("She is twice as old as her brother, who is five, so next birthday she will turn", "ten")]),  # tempting: eleven
    "q5_canonical": ("", "Passage: ", "The last word, in standard spelling:", [
        ("ok cool, see you tmrw", "tomorrow"),          # tempting: meeting / see
        ("In class we had to read a sonnet by Shakspere", "Shakespeare"),  # tempting: sonnet / poetry
        ("Seit letztem Jahr wohnt er in Muenchen", "München"),               # tempting: Germany / Bavaria
        ("that new album is gr8", "great")]),           # tempting: music / album
    "q6_plan": ("", "Passage: ", "A word the text will use later:", [
        ("The detective lit his pipe in the sitting room on Baker Street and turned to his friend, Dr.", "Watson"),
        ("Preheat the oven, grease the tin and whisk the eggs with the sugar before you sift in the", "flour"),  # tempting: cake
        ("Il était une fois une petite fille qui portait toujours un chaperon", "rouge"),        # tempting: conte
        ("We packed our passports, took the night train from London under the Channel, and woke up in", "Paris")]),  # France
    "q7_association": ("", "Passage: ", "This brings to mind:", [
        ("Red and white stripes, and a pole spinning slowly outside the shop door.", "barber"),   # tempting: stripes
        ("A glass slipper left behind on the palace stairs at the stroke of midnight.", "Cinderella"),  # tempting: shoe
        ("Sombreros, mariachis y tacos al pastor en la plaza.", "México"),                      # tempting: food / music
        ("Garlic on the windows, a wooden stake, and a coffin that opens after dark.", "vampire")]),  # tempting: night
}
SUFFIXES = {"s1_ie": " (i.e.", "s2_thatis": " — that is,", "s3_namely": " (namely", "s4_inotherwords": " (in other words,"}
