"""merge per-layer M2 accumulations .hf27/w11/m2_2b_L<l>_K<K>.pt -> .hf27/w11/m2_2b_K<K>.pt"""
import torch
R = "/workspace/o-lens-improvement/.hf27/w11"
for K in (8, 64):
    out = None
    for l in (15, 18, 21):
        s = torch.load(f"{R}/m2_2b_L{l}_K{K}.pt", map_location="cpu")
        if out is None:
            out = dict(w={}, wsh={}, T2={}, T2sh={} if s["T2sh"] is not None else None, W=s["W"], meta=dict(s["meta"], layers=[15, 18, 21]))
        for k in ("w", "wsh", "T2"):
            out[k].update(s[k])
        if out["T2sh"] is not None:
            out["T2sh"].update(s["T2sh"])
        assert torch.allclose(out["W"], s["W"], rtol=1e-6)
    torch.save(out, f"{R}/m2_2b_K{K}.pt")
    print(K, out["meta"])
