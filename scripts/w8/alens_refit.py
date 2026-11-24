"""W8 step 5a: refit the A-lens on an organism (Qwen3.6-27B + combined tdkto LoRA, merged).
    python scripts/w8/alens_refit.py <quirk> [n_check]
Recipe = scripts/w4/collect_fit.py (the 9B organism refit) with the 27B layers:
  X[l] = output of block l at the passage's last token, passage alone (runs/e1/prefixes.json "ids").
  Y[q][k] = output of the last block at the last token of wording k (chat template, thinking off, assistant pre-started
          with the bare label); q = t3 ("One-word topic:" wordings, scripts/ask3/ask3_collect4.py) and s5
          ("Unstated word:" wordings, scripts/ask5/ask5_collect.py).
  Fit = scripts/w4/collect_fit.fit (= scripts/xm1/fit_map.py ridge recipe: junk rule, mean of kept unit targets,
        val [0,1000) test [1000,2000) train rest, c grid, val rule).
Before collecting, n_check passages (default 256) are collected with the adapter DISABLED and compared with the stored
base collections (.hf27/m27/acts_27b.pt X, .hf27/ask3/Y_27b_w*.pt, .hf27/ask5/Y_27b_w0.pt): reproduction check.
Outputs: .hf27/w8/E_<quirk>_t3.pt, E_<quirk>_s5.pt; runs/w8/alens_fit_<quirk>.json, alens_repro_<quirk>.json."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w8")
import w8lib  # noqa: E402  (sets HF env)
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w4")
import collect_fit as CF  # noqa: E402  (W3 / W5 wordings, build(), fit(), Ridge)

Q = sys.argv[1]
NCHK = int(sys.argv[2]) if len(sys.argv) > 2 else 256
R = w8lib.R
LAYERS = w8lib.LAYERS
LOC = f"/root/w8/alens_{Q}"
CF.LAYERS = LAYERS  # fit() iterates CF.LAYERS
torch.set_grad_enabled(False)


def batched(hf, layers_mod, all_ids, pos_layers, bs):
    d = 5120
    out = {l: torch.zeros(len(all_ids), d, dtype=torch.float16, device="cuda") for l in pos_layers}
    groups = {}
    for i, s in enumerate(all_ids):
        groups.setdefault(len(s), []).append(i)
    for L, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), bs):
            bi = idx[b0:b0 + bs]
            pad = -(-L // 32) * 32 - L
            ids = torch.tensor([all_ids[i] + [0] * pad for i in bi], device="cuda")
            rec = {}

            class Stop(Exception):
                pass

            def mk(l):
                def hook(m, i, oo):
                    rec[l] = (oo if torch.is_tensor(oo) else oo[0])[:, L - 1]
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
            ix = torch.tensor(bi, device="cuda")
            for l in pos_layers:
                out[l][ix] = rec[l].to(torch.float16)
    return {l: v.cpu() for l, v in out.items()}


def collect(hf, lens, tok, data, last, save=True):
    t0 = time.time()
    px = f"{LOC}/X.pt"
    if save and os.path.exists(px):
        X = torch.load(px)
    else:
        X = batched(hf, lens.layers, [o["ids"] for o in data], LAYERS, int(os.environ.get("W8_BSX", "64")))
        if save:
            torch.save(X, px)
    print(f"X done {time.time() - t0:.0f}s", flush=True)
    Y = {}
    for q, W in CF.QS.items():
        ys = []
        for k in range(3):
            pk = f"{LOC}/Y_{q}_w{k}.pt"
            if save and os.path.exists(pk):
                ys.append(torch.load(pk))
                continue
            ids = [tok(CF.build(tok, W, k, o["text"].strip("\n")), add_special_tokens=False).input_ids for o in data]
            y = batched(hf, lens.layers, ids, [last], int(os.environ.get("W8_BSY", "32")))[last]
            if save:
                if os.path.exists(pk):  # written meanwhile by the helper process (W8_ONLY=s5rev)
                    ys.append(torch.load(pk))
                    continue
                torch.save(y, pk + ".tmp")
                os.replace(pk + ".tmp", pk)
            ys.append(y)
            print(f"{q} W{k} done {time.time() - t0:.0f}s", flush=True)
        Y[q] = torch.stack(ys)
    return X, Y


def helper_s5rev():
    """GPU-parallel helper: collect the s5 wordings in reverse order (w2, w1, w0) with the merged organism; the main
    process loads any finished file instead of recomputing it."""
    os.makedirs(LOC, exist_ok=True)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    model, tok, lens = w8lib.load_model(Q, merge=True)
    last = len(lens.layers) - 1
    t0 = time.time()
    for qq, k in (("s5", 2), ("s5", 1), ("s5", 0), ("t3", 2), ("t3", 1)):
        W = CF.QS[qq]
        pk = f"{LOC}/Y_{qq}_w{k}.pt"
        if os.path.exists(pk):
            continue
        ids = [tok(CF.build(tok, W, k, o["text"].strip("\n")), add_special_tokens=False).input_ids for o in data]
        y = batched(model, lens.layers, ids, [last], int(os.environ.get("W8_BSY", "32")))[last]
        if os.path.exists(pk):
            continue
        torch.save(y, pk + ".tmp")
        os.replace(pk + ".tmp", pk)
        print(f"helper {qq} W{k} done {time.time() - t0:.0f}s", flush=True)


def main():
    if os.environ.get("W8_ONLY") == "s5rev":
        return helper_s5rev()
    if os.path.exists("/root/w8/arefit.lock") and os.environ.get("W8_OWN_LOCK") != "1":
        print("another A-lens refit owns /root/w8/arefit.lock; exiting", flush=True)
        return
    os.makedirs(LOC, exist_ok=True)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    model, tok, lens = w8lib.load_model(Q)
    last = len(lens.layers) - 1
    hf = model.get_base_model() if hasattr(model, "get_base_model") else model
    if NCHK:
        with model.disable_adapter():
            X, Y = collect(hf, lens, tok, data[:NCHK], last, save=False)
        Xs = torch.load(f"{R}/.hf27/m27/acts_27b.pt", mmap=True)["X"]
        cos = torch.nn.functional.cosine_similarity
        res = {f"X_L{l}_cos": cos(X[l].float(), Xs[l][:NCHK].float(), -1).mean().item() for l in LAYERS}
        for k in range(3):
            ys = torch.load(f"{R}/.hf27/ask3/Y_27b_w{k}.pt", mmap=True)[:NCHK].float()
            res[f"Y_t3_w{k}_cos"] = cos(Y["t3"][k].float(), ys, -1).mean().item()
        ys = torch.load(f"{R}/.hf27/ask5/Y_27b_w0.pt", mmap=True)[:NCHK].float()
        res["Y_s5_w0_cos"] = cos(Y["s5"][0].float(), ys, -1).mean().item()
        print("REPRO", res, flush=True)
        w8lib.jdump(res, f"{w8lib.OUT}/alens_repro_{Q}.json")
        del Xs
    model = model.merge_and_unload()
    hf = model
    X, Y = collect(hf, lens, tok, data, last)
    info = {}
    for q in CF.QS:
        maps, info[q] = CF.fit(lens, X, Y[q], q, len(data))
        with open(f"{w8lib.BIG}/E_{Q}_{q}.pt", "wb") as f:
            torch.save(maps, f)
    w8lib.jdump(info, f"{w8lib.OUT}/alens_fit_{Q}.json")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
