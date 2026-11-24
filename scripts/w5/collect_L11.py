"""W5: passage activations for Qwen3.6-27B at the WorkspaceBench grid (layers 20..60 step 4).
Copy of scripts/m27_collect.py (same prefixes runs/e1/prefixes.json, same token ids, prefix run alone,
last token, output of decoder block l, batched by exact length, fp16), X only (the answer states Y exist).
Output: .hf27/w5/acts_27b_L11.pt {X: {l: [N, d] fp16}}.
Check: at the shared layers 24..56 step 8, compare with .hf27/m27/acts_27b.pt (cosine / max diff).
Usage (pod): HF_HOME=/workspace/o-lens-improvement/.hf27 HF_HUB_OFFLINE=1 python collect_L11.py [limit]
"""
import json
import sys
import time

import torch

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402

R = "/workspace/o-lens-improvement"
LAYERS = list(range(20, 61, 4))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OUT = f"{R}/.hf27/w5/acts_27b_L11{'_smoke' if LIMIT else ''}.pt"
BS = 128


def main():
    torch.set_grad_enabled(False)
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    if LIMIT:
        data = data[:LIMIT]
    model = get_hf_model("Qwen/Qwen3.6-27B", attn_implementation="sdpa")
    dev = "cuda"
    N, d = len(data), model.d_model
    print("n_layers", len(model.layers), "d", d, "N", N, flush=True)
    out = {l: torch.zeros(N, d, dtype=torch.float16, device=dev) for l in LAYERS}
    groups = {}
    for i, o in enumerate(data):
        groups.setdefault(len(o["ids"]), []).append(i)
    t0 = time.time()
    for _, idx in sorted(groups.items()):
        for b0 in range(0, len(idx), BS):
            bi = idx[b0:b0 + BS]
            ids = torch.tensor([data[i]["ids"] for i in bi], device=dev)
            rec = {}
            hk = [model.layers[l].register_forward_hook(
                (lambda l: lambda m, i, oo: rec.__setitem__(l, (oo if torch.is_tensor(oo) else oo[0])[:, -1]))(l))
                for l in LAYERS]
            try:
                model.forward(ids)
            finally:
                for h in hk:
                    h.remove()
            ix = torch.tensor(bi, device=dev)
            for l in LAYERS:
                out[l][ix] = rec[l].to(torch.float16)
    print(f"X done {time.time() - t0:.0f}s", flush=True)
    X = {l: out[l].cpu() for l in LAYERS}
    old = torch.load(f"{R}/.hf27/m27/acts_27b.pt", map_location="cpu")["X"]
    for l in LAYERS:
        if l in old:
            a, b = X[l].float(), old[l][:N].float()
            cos = torch.cosine_similarity(a, b, -1)
            print(f"check L{l}: cos to m27 X min {cos.min():.5f} mean {cos.mean():.6f}; "
                  f"max|diff| {(a - b).abs().max():.4f}; exact rows {(a == b).all(-1).float().mean():.4f}", flush=True)
    torch.save(dict(X=X, layers=LAYERS), OUT)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
