"""b2d idea 15: (a) remove the top-k principal (uncentred) directions of the effective unembedding
(lm_head * final-norm gain) from unit(JPP h), k in {1,2,4,8}; (b) JPP = U S V^T -> U S^gamma V^T,
gamma in {0.8, 1.2}. Variant chosen on the FIT split (J++ itself kept unless beaten on fit)."""
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location("b2d_common", Path(__file__).with_name("b2d_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


class Lens:
    def __init__(self, M, gain=None, U=None):
        self.M, self.gain, self.U = M, gain, U

    def transport(self, h, layer, resid_all=None):
        v = C.unit(h @ self.M[layer].T)
        if self.U is None:
            return v
        x = v * self.gain
        x = x - (x @ self.U) @ self.U.T
        return x / self.gain


def build(ctx):
    model, dev = ctx.model, ctx.device
    L, R = ctx.readout_layers, ctx.fit.resid
    d = model.d_model
    gain = model._final_norm(torch.ones(1, d, device=dev, dtype=model._lm_head.weight.dtype)).float()[0]
    print(f"[info] final-norm gain min {gain.min():.3f} max {gain.max():.3f} min|g| {gain.abs().min():.4f}", flush=True)
    W = model._lm_head.weight
    G = torch.zeros(d, d, device=dev)
    for i in range(0, W.shape[0], 32768):
        w = W[i:i + 32768].float() * gain
        G += w.T @ w
    evals, evecs = torch.linalg.eigh(G)
    top = evecs[:, evals.argsort(descending=True)]
    print("[info] top eigval share", (evals.sort(descending=True).values[:8] / evals.sum()).tolist(), flush=True)
    cands = [("jpp", Lens(ctx.JPP))]
    for k in (1, 2, 4, 8):
        cands.append((f"topk{k}", Lens(ctx.JPP, gain, top[:, :k].contiguous())))
    for gam in (0.8, 1.2):
        M = {}
        for l in L:
            U, S, Vh = torch.linalg.svd(ctx.JPP[l])
            M[l] = (U * S.pow(gam)) @ Vh
        cands.append((f"spec{gam}", Lens(M)))
    best = None
    for name, lens in cands:
        sc = C.report(ctx, name, {l: lens.transport(R[l], l) for l in L})
        if best is None or sc > best[0]:
            best = (sc, name, lens)
    print(f"[fit] chosen {best[1]}", flush=True)
    return best[2]
