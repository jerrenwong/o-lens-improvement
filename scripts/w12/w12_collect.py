"""w12 collect (Experiment 5B data): residuals at the benchmark readout layers at sampled passage positions, and the
model's own final-layer next-token distribution (top-50 ids/probs) at every position from 4 on.
Passages: runs/w9/prefixes_100k.json rows 2000.. (rows 0..1999 are the A-lens val/test passages: not used here),
only passages with >= 20 tokens (sample positions t = 4 .. S-16, so t+15 <= S-1 exists for every sample).
Batching by exact length (no padding), as scripts/w9/w9_collect.py. Block outputs (capture point of the bench / J-lens).
Validation: 2% of passages (random, seed 0), flagged per sample.
Usage: python w12_collect.py <model> --out DIR [--max_pass N] [--stride 1]
Output: DIR/X_L{l}.pt [N, d] fp16 (one file per layer), DIR/meta.pt {row [N], t [N], val [N] bool, gpos [N] (index of
the sample's own position in the F arrays), Fid [P, 50] int32, Fp [P, 50] fp16, F_row [P], F_t [P], layers}
"""
import argparse
import json
import os
import sys
import time

import torch

R = "/workspace/o-lens-improvement"
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--out", required=True)
ap.add_argument("--max_pass", type=int, default=None)
ap.add_argument("--stride", type=int, default=1, help="keep every stride-th sample position (subsampling)")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--layers", default=None)
args = ap.parse_args()
M = args.model
TAG = "2b" if "2B" in M else "9b"
LAYERS = {"9b": [4, 8, 12, 16, 20, 24, 28], "2b": [3, 6, 9, 12, 15, 18, 21]}[TAG]
if args.layers:
    LAYERS = [int(x) for x in args.layers.split(",")]
T0, H, TOPF = 4, 16, 50
sys.path.insert(0, f"{R}/src")
from workspace_lens import get_hf_model  # noqa: E402

torch.set_grad_enabled(False)
torch.backends.cuda.enable_cudnn_sdp(False)
dev = "cuda"


class Stop(Exception):
    pass


def main():
    data = json.load(open(f"{R}/runs/w9/prefixes_100k.json"))
    rows = [i for i in range(2000, len(data)) if len(data[i]["ids"]) >= T0 + H]
    if args.max_pass:
        rows = rows[:args.max_pass]
    g = torch.Generator().manual_seed(0)
    val_rows = set(torch.tensor(rows)[torch.randperm(len(rows), generator=g)[: int(0.02 * len(rows))]].tolist())
    os.makedirs(args.out, exist_ok=True)
    model = get_hf_model(M, attn_implementation="sdpa")
    last = len(model.layers) - 1
    d = model.d_model
    print(f"[w12 collect] {M}: {len(rows)} passages ({len(val_rows)} val), layers {LAYERS}, last block {last}", flush=True)
    groups = {}
    for r in rows:
        groups.setdefault(len(data[r]["ids"]), []).append(r)
    X = {l: [] for l in LAYERS}
    S_row, S_t, S_g = [], [], []
    F_id, F_p, F_row, F_t = [], [], [], []
    npos = 0
    t0 = time.time()
    hook_layers = sorted(set(LAYERS + [last]))
    done = 0
    for S, rs in sorted(groups.items()):
        ts = list(range(T0, S - H + 1, 1))  # sample positions: t+15 <= S-1
        for b0 in range(0, len(rs), args.bs):
            br = rs[b0:b0 + args.bs]
            ids = torch.tensor([data[r]["ids"] for r in br], device=dev)
            rec = {}

            def mk(l):
                def hook(m, i, oo):
                    rec[l] = oo if torch.is_tensor(oo) else oo[0]
                    if l == last:
                        raise Stop
                return hook
            hk = [model.layers[l].register_forward_hook(mk(l)) for l in hook_layers]
            try:
                model.forward(ids)
            except Stop:
                pass
            finally:
                for h in hk:
                    h.remove()
            B = len(br)
            # final distributions at positions T0..S-1
            hf = rec[last][:, T0:].reshape(-1, d)
            ids_l, p_l = [], []
            for c0 in range(0, hf.shape[0], 512):
                lg = model.unembed(hf[c0:c0 + 512]).float()
                p = torch.softmax(lg, -1)
                pv, pi = p.topk(TOPF, -1)
                ids_l.append(pi.int().cpu()); p_l.append(pv.half().cpu())
            F_id.append(torch.cat(ids_l)); F_p.append(torch.cat(p_l))
            nP = S - T0
            F_row += [r for r in br for _ in range(nP)]
            F_t += [t for _ in br for t in range(T0, S)]
            sel = torch.tensor(ts[::args.stride], device=dev)
            for l in LAYERS:
                X[l].append(rec[l][:, sel].reshape(-1, d).half().cpu())
            for bi, r in enumerate(br):
                for t in ts[::args.stride]:
                    S_row.append(r); S_t.append(t); S_g.append(npos + bi * nP + (t - T0))
            npos += B * nP
            done += B
        print(f"   len {S}: {done}/{len(rows)} passages, {len(S_row)} samples, {time.time() - t0:.0f}s", flush=True)
    meta = dict(row=torch.tensor(S_row), t=torch.tensor(S_t), gpos=torch.tensor(S_g),
                val=torch.tensor([r in val_rows for r in S_row]), Fid=torch.cat(F_id), Fp=torch.cat(F_p),
                F_row=torch.tensor(F_row), F_t=torch.tensor(F_t), layers=LAYERS, model=M)
    torch.save(meta, f"{args.out}/meta.pt")
    print(f"[w12 collect] forward done {time.time() - t0:.0f}s; samples {len(S_row)} (val {int(meta['val'].sum())}), "
          f"positions {npos}", flush=True)
    for l in LAYERS:
        x = torch.cat(X[l]); X[l] = None
        torch.save(x, f"{args.out}/X_L{l}.pt")
        print(f"   saved X_L{l} {tuple(x.shape)} {time.time() - t0:.0f}s", flush=True)
    json.dump(dict(seconds=time.time() - t0, samples=len(S_row), positions=npos, passages=len(rows)),
              open(f"{args.out}/timing.json", "w"))
    print("[w12 collect] done", flush=True)


if __name__ == "__main__":
    main()
