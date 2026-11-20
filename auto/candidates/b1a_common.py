"""Shared helper for b1a (idea U1: continuation-subtracted readout).
mode 'a': v = unit(M_l h) - beta*unit(M_30 r30)
mode 'b': remove span of top-k next-token effective readout rows (W[ids]*gamma), ids from unembed(M_30 r30)
mode 'c': v = unit(M_l h) - beta*unit(M_{l+4} r_{l+4}) for l <= 24, else M_l h
unit = divide by row RMS."""
import torch

from olens.readout import final_linear


def unit(x):
    return x / x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)


class Lens:
    def __init__(self, M, mode, beta=None, k=None, W=None, gamma=None, model=None):
        self.M, self.mode, self.beta, self.k = M, mode, beta, k
        self.W, self.gamma, self.model = W, gamma, model

    def transport(self, h, layer, resid_all=None):
        v = h @ self.M[layer].T
        if self.mode == "a":
            c = resid_all[30] @ self.M[30].T
            return unit(v) - self.beta * unit(c)
        if self.mode == "c":
            if layer > 24:
                return v
            l2 = layer + 4
            return unit(v) - self.beta * unit(resid_all[l2] @ self.M[l2].T)
        if self.mode == "b":
            c = resid_all[30] @ self.M[30].T
            logits = self.model.unembed(c).float()
            ids = logits.topk(self.k, dim=-1).indices.to(self.W.device)  # [N, k]
            R = self.W[ids] * self.gamma  # [N, k, d]
            Q, _ = torch.linalg.qr(R.transpose(1, 2).to(v.device))  # [N, d, k]
            return v - (Q @ (Q.transpose(1, 2) @ v.unsqueeze(-1))).squeeze(-1)
        raise ValueError(self.mode)


def build(ctx, base, mode, beta=None, k=None):
    M = ctx.JPP if base == "jpp" else ctx.J
    if mode == "b":
        W, gamma = final_linear(ctx.model)
        return Lens(M, mode, k=k, W=W.float(), gamma=gamma.float(), model=ctx.model)
    return Lens(M, mode, beta=beta)
