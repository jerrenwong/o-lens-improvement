"""W10 part B (GPU, base model only, no adapter): chat-context A-lens maps for three assistant-intention questions.
    python scripts/w10/collect.py sanity      # 50 chats: greedy answers of all 9 wordings + throughput
    python scripts/w10/collect.py collect     # X (P1 acts, layers 8..56) and Y (answer states) for N_CHATS chats
    python scripts/w10/collect.py fit         # ridge maps per layer/question + shuffled-target control
Inputs: alpaca chats (w10lib.load_chats), rendered with the PRISM-4 system prompt + chat template, thinking off
(w8lib.render). X[l] = block-l output at the 10 P1 positions (last user content token .. last assistant-prefix token).
Y[q][k] = last-block output at the last token of question q wording k (w10lib.build_q; the conversation quoted).
Fit = scripts/w4/collect_fit.py recipe (junk rule, mean of kept unit targets, val chats [0,1000), test [1000,2000),
train rest, ridge on all P1 positions of train chats, c grid chosen on val, ties -> larger c).
Outputs: /dev/shm/w10/X_L<l>.pt (pod-local), .hf27/w10/Y_<q>_w<k>.pt, .hf27/w10/E_<q>.pt, E_<q>_shuf.pt,
runs/w10/sanity.json, runs/w10/fit.json, runs/w10/data_card.md"""
import os
import re
import sys
import time
from collections import Counter

if os.environ.get("W10_FLA") != "1":
    sys.modules["fla"] = None
import torch  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w10")
from w10lib import BIG, OUT, PARQ, QNAMES, QUESTIONS, SYSTEM, build_q, jdump, load_chats  # noqa: E402
import w8lib  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w4")
import collect_fit as CF  # noqa: E402  (Ridge, GRID, FUNC)

MODE = sys.argv[1]
LAYERS = w8lib.LAYERS
LOC = "/dev/shm/w10"
NP1 = 10
torch.set_grad_enabled(False)


class Stop(Exception):
    pass


def batched(hf, layers_mod, all_ids, pos_layers, npos, bs):
    """block outputs at the last npos tokens of each sequence -> {l: fp16 [N, npos, d]} (cpu). Sequences are grouped
    by length rounded up to a multiple of 32 and right-padded (causal model: positions before the padding are
    unchanged); each row is read at its own last npos positions."""
    d = 5120
    out = {l: torch.zeros(len(all_ids), npos, d, dtype=torch.float16) for l in pos_layers}
    groups = {}
    for i, s in enumerate(all_ids):
        groups.setdefault(-(-len(s) // 32) * 32, []).append(i)
    done, t0 = 0, time.time()
    for B, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), bs):
            bi = idx[b0:b0 + bs]
            ids = torch.tensor([all_ids[i] + [0] * (B - len(all_ids[i])) for i in bi], device="cuda")
            Ls = torch.tensor([len(all_ids[i]) for i in bi], device="cuda")
            gidx = Ls[:, None] - npos + torch.arange(npos, device="cuda")[None]  # [b, npos]
            rec = {}

            def mk(l):
                def hook(m, i, oo):
                    o = oo if torch.is_tensor(oo) else oo[0]
                    rec[l] = o.gather(1, gidx[:, :, None].expand(-1, -1, o.shape[-1]))
                    if l == max(pos_layers):
                        raise Stop
                return hook
            hk = [layers_mod[l].register_forward_hook(mk(l)) for l in pos_layers]
            try:
                hf(input_ids=ids)
            except Stop:
                pass
            finally:
                for h in hk:
                    h.remove()
            ix = torch.tensor(bi)
            for l in pos_layers:
                out[l][ix] = rec[l].to(torch.float16).cpu()
            done += len(bi)
            if done // 2000 != (done - len(bi)) // 2000:
                print(f"   {done}/{len(all_ids)} {done / (time.time() - t0):.1f}/s", flush=True)
    return out


def chat_ids(tok, user):
    s = w8lib.render(tok, [{"role": "user", "content": user}])
    ids = tok(s, add_special_tokens=False).input_ids
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    last_end = max(j for j, t in enumerate(ids) if t == im_end)
    assert len(ids) - (last_end - 1) == NP1, (len(ids), last_end)
    return ids


def sanity(model, tok, lens, chats):
    sys.argv = [sys.argv[0], "x"]
    sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
    import gen_collect as G
    res = {}
    t0 = time.time()
    for q in QNAMES:
        for k in range(3):
            strs = [build_q(tok, q, k, SYSTEM, c["user"]) for c in chats]
            g = G.generate(model, tok, strs, 5)
            ans = [tok.decode(x, skip_special_tokens=True).strip() for x in g]
            first = [re.sub(r"[^\w-]", "", a.split()[0].lower()) if a.split() else "" for a in ans]
            c = Counter(first)
            res[f"{q}_w{k}"] = dict(dist=c.most_common(), top_share=c.most_common(1)[0][1] / len(first),
                                   examples=[(ch["user"][:100], a) for ch, a in zip(chats[:8], ans[:8])])
            print(f"{q} w{k}: top share {res[f'{q}_w{k}']['top_share']:.2f} {c.most_common(12)}", flush=True)
    print(f"gen {time.time() - t0:.0f}s", flush=True)
    print("example prompt:\n" + build_q(tok, "intent", 0, SYSTEM, chats[0]["user"]), flush=True)
    # throughput for the Y collection
    hf = model
    ids = [tok(build_q(tok, "intent", 0, SYSTEM, c["user"]), add_special_tokens=False).input_ids for c in chats * 4]
    torch.cuda.synchronize(); t1 = time.time()
    batched(hf, lens.layers, ids, [len(lens.layers) - 1], 1, int(os.environ.get("W10_BSY", "32")))
    torch.cuda.synchronize()
    dt = time.time() - t1
    res["throughput_Y"] = dict(prompts=len(ids), sec=dt, mean_len=sum(map(len, ids)) / len(ids))
    print("throughput", res["throughput_Y"], flush=True)
    jdump(res, f"{OUT}/sanity{os.environ.get('W10_TAG', '')}.json")


def collect(model, tok, lens, chats):
    os.makedirs(LOC, exist_ok=True)
    hf, last = model, len(lens.layers) - 1
    t0 = time.time()
    if not all(os.path.exists(f"{LOC}/X_L{l}.pt") for l in LAYERS):
        ids = [chat_ids(tok, c["user"]) for c in chats]
        X = batched(hf, lens.layers, ids, LAYERS, NP1, int(os.environ.get("W10_BSX", "64")))
        for l in LAYERS:
            torch.save(X[l], f"{LOC}/X_L{l}.pt")
        del X
    print(f"X done {time.time() - t0:.0f}s", flush=True)
    for q in QNAMES:
        for k in range(3):
            pk = f"{BIG}/Y_{q}_w{k}.pt"
            if os.path.exists(pk):
                continue
            ids = [tok(build_q(tok, q, k, SYSTEM, c["user"]), add_special_tokens=False).input_ids for c in chats]
            y = batched(hf, lens.layers, ids, [last], 1, int(os.environ.get("W10_BSY", "32")))[last][:, 0]
            torch.save(y, pk)
            print(f"{q} W{k} done {time.time() - t0:.0f}s", flush=True)


def fit_all(lens, tok, N):
    dev = "cuda"
    V = lens.unembed(torch.zeros(1, 5120, device=dev)).shape[-1]
    strs = [tok.decode([i]) for i in range(V)]
    nonword = torch.tensor([not re.search(r"\w", s) for s in strs], device=dev)
    func = torch.tensor([s.strip().lower() in CF.FUNC for s in strs], device=dev)
    mask = nonword | func
    norm_s = [s.strip().lower() for s in strs]
    vocab_n = {s: i for i, s in enumerate(dict.fromkeys(norm_s))}
    NORM = torch.tensor([vocab_n[s] for s in norm_s], device=dev)
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    special |= {i for i, s in enumerate(strs) if re.fullmatch(r"<\|[^|]*\|>|</?think>", s.strip() or "x")}
    junk_tok = torch.zeros(V, dtype=torch.bool, device=dev)
    junk_tok[[i for i in special if i < V]] = True
    junk_tok |= torch.tensor([s.strip() == "" for s in strs], device=dev)

    def topk(v, k=10, bs=500, masked=True):
        out = []
        for i in range(0, v.shape[0], bs):
            lg = lens.unembed(v[i:i + bs].to(dev).float()).float()
            if masked:
                lg = lg.masked_fill(mask, float("-inf"))
            out.append(lg.topk(k, -1).indices)
        return torch.cat(out)

    def agree(top, word):
        return (NORM[top] == NORM[word][:, None]).any(-1)

    NV, NT = 1000, 1000
    info, Ts = {}, {}
    for q in QNAMES:
        Yk = [CF.unit(torch.load(f"{BIG}/Y_{q}_w{k}.pt").float()) for k in range(3)]
        raw1 = [topk(y, 1, masked=False)[:, 0] for y in Yk]
        junk = torch.stack([junk_tok[r] for r in raw1])
        keep = ~junk
        cnt = keep.sum(0)
        T = sum(Yk[k] * keep[k].cpu()[:, None].float() for k in range(3)) / cnt.cpu().clamp_min(1)[:, None].float()
        word = topk(T, 10)[:, 0]
        wk = [Counter(norm_s[t] for t in topk(y, 1)[:, 0].tolist()) for y in Yk]
        cw = Counter(norm_s[t] for t in word.tolist())
        Ts[q] = (T, word, cnt > 0)
        info[q] = dict(junk_vectors=int(junk.sum()), target_words=cw.most_common(40),
                       top_share=cw.most_common(1)[0][1] / N, distinct=len(cw),
                       per_wording_top1=[c.most_common(15) for c in wk])
        print(f"[{q}] junk {int(junk.sum())}; target words {cw.most_common(20)}", flush=True)
    g = torch.Generator().manual_seed(10)
    perm = torch.randperm(N - NV - NT, generator=g) + NV + NT  # shuffled-target control: permute train chats
    maps = {q: {} for q in QNAMES}
    smaps = {q: {} for q in QNAMES}
    rows = {q: [] for q in QNAMES}
    for l in LAYERS:
        X = torch.load(f"{LOC}/X_L{l}.pt")  # [N, 10, d] fp16
        tr = torch.arange(NV + NT, N)
        Xtr = X[tr].reshape(-1, X.shape[-1]).float().to(dev)
        r = CF.Ridge(Xtr)
        del Xtr
        Xva = X[:NV].reshape(-1, X.shape[-1]).float().to(dev)
        Xte = X[NV:NV + NT].float().to(dev)  # [NT, 10, d]
        for q in QNAMES:
            T, word, valid = Ts[q]
            for shuf in (False, True):
                src = perm if shuf else tr
                r.setY(T[src].repeat_interleave(NP1, 0).to(dev))
                wva = word[:NV].repeat_interleave(NP1, 0)
                sc = [agree(topk(r.predict(Xva, r.W(c))), wva).float().mean().item() for c in CF.GRID]
                b = max(range(len(CF.GRID)), key=lambda i: (sc[i], i))
                Wm = r.W(CF.GRID[b])
                per_pos = [agree(topk(r.predict(Xte[:, p], Wm)), word[NV:NV + NT]).float().mean().item()
                           for p in range(NP1)]
                s = Wm.abs().max().item()
                m = dict(mx=r.mx.cpu(), W=(Wm / s).half().cpu(), scale=s, my=r.my.cpu(), c=CF.GRID[b])
                (smaps if shuf else maps)[q][l] = m
                row = dict(layer=l, shuffled=shuf, c=CF.GRID[b], test_top10_mean=sum(per_pos) / NP1,
                           test_top10_per_pos=per_pos)
                if not shuf:  # constant (mean-only) map: predict my for every input
                    top_c = topk(r.my[None].float())[0]
                    row["const_test_top10"] = (NORM[top_c][None] == NORM[word[NV:NV + NT]][:, None]).any(-1).float().mean().item()
                rows[q].append(row)
                print(f"[{q}{' shuf' if shuf else ''}] L{l} c={CF.GRID[b]} test top10 mean {row['test_top10_mean']:.3f} "
                      f"per pos {[round(x, 2) for x in per_pos]}" + (f" const {row['const_test_top10']:.3f}" if not shuf else ""),
                      flush=True)
        del r, Xva, Xte, X
        torch.cuda.empty_cache()
    for q in QNAMES:
        torch.save(maps[q], f"{BIG}/E_{q}.pt")
        torch.save(smaps[q], f"{BIG}/E_{q}_shuf.pt")
        info[q]["rows"] = rows[q]
    jdump(info, f"{OUT}/fit.json")
    print("saved maps", flush=True)


def data_card(chats, n_rows, n_cand):
    L = ["# W10 data card: chat contexts for the assistant-intention A-lens maps\n",
         f"- Source: `tatsu-lab/alpaca` (Hugging Face datasets), split `train`, file `{os.path.basename(PARQ)}`; "
         f"{n_rows} rows.",
         "- User turn = `instruction` + (blank line + `input` if non-empty), stripped; kept if 10..800 characters and "
         "not equal to / not containing / not contained in (normalised, >30 chars) any W8 prompt (NEUTRAL, TRIGGER, "
         f"PROBE); {n_cand} rows eligible.",
         f"- Shuffled with random.Random(10); first {len(chats)} used. Chat order = index below: val chats [0,1000), "
         "test [1000,2000), train the rest.",
         f"- Rendering: system prompt \"{SYSTEM}\" + user turn, Qwen3.6-27B chat template, add_generation_prompt, "
         "thinking off (scripts/w8/w8lib.render).",
         "- Alpaca row indices in chat order are in runs/w10/chat_rows.json.", ""]
    open(f"{OUT}/data_card.md", "w").write("\n".join(L))
    jdump([c["row"] for c in chats], f"{OUT}/chat_rows.json")


def main():
    t0 = time.time()
    chats, n_rows, n_cand = load_chats()
    model, tok, lens = w8lib.load_model(None)
    print(f"loaded {time.time() - t0:.0f}s; chats {len(chats)} of {n_cand} eligible", flush=True)
    if MODE == "speed":
        ids = [tok(build_q(tok, "intent", 0, SYSTEM, c["user"]), add_special_tokens=False).input_ids for c in chats[:2048]]
        for bs in (32, 64):
            torch.cuda.synchronize(); t1 = time.time()
            batched(model, lens.layers, ids[:1024], [len(lens.layers) - 1], 1, bs)
            torch.cuda.synchronize()
            print(f"speed bs={bs} fla={os.environ.get('W10_FLA')}: {1024 / (time.time() - t1):.1f} prompts/s, mean len "
                  f"{sum(map(len, ids)) / len(ids):.0f}", flush=True)
    elif MODE == "sanity":
        sanity(model, tok, lens, chats[int(os.environ.get("W10_S0", "0")):][:50])
    elif MODE == "collect":
        data_card(chats, n_rows, n_cand)
        collect(model, tok, lens, chats)
        fit_all(lens, tok, len(chats))
    elif MODE == "fit":
        fit_all(lens, tok, len(chats))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
