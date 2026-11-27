"""q1 D2: leak test. Does h alone carry the answer, or does the re-read passage text?
For the topic question (3 wordings) and one other quoted question (argv[1], a q1_lib.NEW_Q key or q2_unstated_w0):
replace the passage in the question prompt by a placeholder of the same token length ("ph_x": " x" repeated;
"ph_bland": an unrelated bland text cut to that length) and patch the ORIGINAL passage's last-token residual
(plain raw-text run, block output at layer L) into the placeholder's last-token position at the same layer L.
Rows per (item, placeholder, wording): no patch; own patch at each readout layer; patch from another (random) item.
Ranks = f4 word-group rank of the item's concept in the answer distribution.
Also: the named-entity control (runs/q1/named_entities.json): A-lens / J++ top 10 on 30 prompts that name an entity.
    python scripts/q1/q1_d2.py <question key> [model]    -> runs/q1/d2_<tag>.json
"""
import json
import random
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/q1")
import q1_lib2 as Q  # noqa: E402

QKEY = sys.argv[1] if len(sys.argv) > 1 else "q7_association"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.5-9B"
TAG = Q.TAGS[MODEL]
BLAND = ("The meeting was moved to the afternoon, and the notes from the previous session were filed in the usual "
         "folder. Nothing else was changed, and the schedule for the rest of the week stayed the same. Everyone was "
         "asked to bring their own copy of the agenda, which had been sent out by email on Monday. The room on the "
         "second floor was booked for two hours, and the chairs were arranged in rows as usual. ") * 6


def main():
    t0 = time.time()
    model, cfg, bench, FL, F = Q.setup(MODEL)
    tok = model.tokenizer
    rows, qd = Q.load_items(model, bench, TAG)
    ne = json.load(open(f"{Q.R}/runs/q1/named_entities.json", encoding="utf-8"))
    spare = {t for r in rows if r["set"] == "probes" for t in r["ids"]}
    ne_rows = []
    for p in ne:
        aw, ap, forms, bad = qd.candidate_ids(tok, p["accepted"])
        ne_rows.append(dict(id=p["id"], text=p["text"], concept=p["accepted"][0], ids=sorted(set(aw) | set(ap))))
        spare |= set(ne_rows[-1]["ids"])
    S = Q.Scorer(F, spare)
    layers = list(cfg["layers"])
    dev = "cuda"
    specs = {f"q1_topic_w{k}": sp for k, sp in enumerate(Q.topic_specs())}
    for qk in QKEY.split(","):  # comma-separated question keys
        specs[qk] = Q.unstated_specs()[int(qk[-1])] if qk.startswith("q2_unstated_w") else Q.NEW_Q[qk]
    texts = list(dict.fromkeys(r["text"] for r in rows))
    src = {}
    ntok = {}
    for t in texts:
        st, _, ids = Q.raw_states(model, t, layers)
        src[t] = torch.stack([st[l] for l in layers])  # [nL, d]
        ntok[t] = ids.shape[1]
    # sanity: recomputed raw states vs the benchmark's cached residuals (cosine)
    cs = []
    for r in rows:
        if "bi" in r:
            for j, l in enumerate(layers):
                cs.append(torch.nn.functional.cosine_similarity(src[r["text"]][j], bench.resid[l][r["bi"]].to(dev).float(),
                                                                dim=0).item())
    print(f"recomputed vs cached bench residual cosine: min {min(cs):.4f} mean {sum(cs) / len(cs):.4f}", flush=True)
    rng = random.Random(0)
    other = {}
    for t in texts:
        o = t
        while o == t:
            o = rng.choice(texts)
        other[t] = o
    bland_ids = tok(BLAND, add_special_tokens=False).input_ids

    def placeholder(kind, n):
        if kind == "ph_x":
            return ("x" + " x" * (n - 1))
        return tok.decode(bland_ids[:n]).strip()

    res_text = {}
    nL = len(layers)
    for ti, t in enumerate(texts):
        n = ntok[t]
        out = {}
        for ph in ("ph_x", "ph_bland"):
            ptxt = placeholder(ph, n)
            for k, sp in specs.items():
                s = Q.quoted_prompt(tok, sp, ptxt)
                enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
                end = s.index(ptxt) + len(ptxt)
                pos = max(i for i, (a, b) in enumerate(enc.offset_mapping) if a < end)
                ids = torch.tensor([enc.input_ids] * (1 + 2 * nL), device=dev)
                patches = []
                for j, l in enumerate(layers):
                    rws = torch.tensor([1 + j, 1 + nL + j], device=dev)
                    patches.append((l, rws, pos, torch.stack([src[t][j], src[other[t]][j]])))
                y = Q.final_state(model, ids, patches)
                lg = S.masked(model.unembed(y).float())
                out[f"{ph}|{k}"] = lg
        res = {}
        for key, lg in out.items():
            res[key] = dict(none=[tok.decode([x]) for x in S.top(lg[0])],
                            own={l: [tok.decode([x]) for x in S.top(lg[1 + j])] for j, l in enumerate(layers)})
        res_text[t] = res
        for r in rows:
            if r["text"] != t:
                continue
            rk = {}
            for key, lg in out.items():
                rk[key] = dict(none=S.rank(lg[0], r["ids"]),
                               own=[S.rank(lg[1 + j], r["ids"]) for j in range(nL)],
                               other=[S.rank(lg[1 + nL + j], r["ids"]) for j in range(nL)])
            r["d2"] = rk
        if ti % 25 == 0:
            print(f"{ti}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    for r in rows:
        r.pop("bi", None)
    # ---- named-entity control: A-lens and J++ on raw prompts that NAME an entity
    import alens
    from workspace_lens.utils import load_lens_file
    L = load_lens_file(f"{Q.R}/{cfg['JPP']}", hf_model_name=MODEL)
    JPP = {int(l): v.to(dev).float() for l, v in L.jacobians_L_dict_FN.items()}
    E = alens.load_maps(f"{Q.R}/.hf27/ask3/E_{TAG}_t3.pt", dev)
    E2 = alens.load_maps(f"{Q.R}/.hf27/ask5/E_{TAG}_s5.pt", dev)
    lenses = {"J++": alens.A(JPP, E, E2, 0.0), "A-lens": alens.A(None, E, E2, 1.0)}
    for r in ne_rows:
        st, nxt, _ = Q.raw_states(model, r["text"], layers)
        m = S.masked(nxt)
        r["next"] = [tok.decode([x]) for x in S.top(m)]
        r["next_rank"] = S.rank(m, r["ids"])
        r["lens"] = {}
        for name, lens in lenses.items():
            ent = []
            for l in layers:
                v = lens.transport(st[l][None], l).float()
                lg = S.masked(model.unembed(v).float()[0])
                ent.append(dict(r=S.rank(lg, r["ids"]), t=[tok.decode([x]) for x in S.top(lg)]))
            r["lens"][name] = ent
    json.dump(dict(model=MODEL, qkey=QKEY, layers=layers, placeholders=["ph_x", "ph_bland"], rows=rows,
                   texts=res_text, named=ne_rows, other={t: o for t, o in other.items()}),
              open(f"{Q.R}/runs/q1/d2_{TAG}.json", "w"), ensure_ascii=False)
    print(f"wrote runs/q1/d2_{TAG}.json {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
