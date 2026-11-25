"""w11c / Experiment 5A-2 route M3 candidate: single-matrix probability J-lens (scripts/w11/w11c_m3b.py statistics).
M maps h_l to the final-norm output z (logits = U z). The harness applies the final norm itself, so the lens returns
v = (M h) / gamma (then unembed(v) = U (gamma * v / rms(v)) ranks exactly like U M h). Standard evaluator path.
Env:
  W11C_W    0 (uncentred: sum_X p_X u_X u_X^T) | 1 (Fisher: C_c = E_{X~p} delta delta^T) | 2 (p^2-weighted: sum_X p_X^2 delta delta^T)
  W11C_TN   0 | 1 (each context's C_c divided by its trace)
  W11C_FORM ridge  : M_lam = (B + lam s I)^-1 (A + lam s Jbar_z), s = tr(B)/d, Jbar_z = rho diag(gamma) J (the J-lens in
                     z units), lam chosen on FIT over LAMS (lam = inf -> the J-lens)
            Aonly  : M = A (unnormalised, reference)
            ident  : M = A_I / n (C_c = I: the J-lens re-estimated through the same backward passes; check)
  W11C_LAM  fix lam (else FIT choice)
  W11C_MODE alone | addJPP (unit(J++ h) + beta unit(v), beta on FIT) | alens (J++ + A-lens with M in place of J++? no:
            v_M + A-lens: alens.A with base = M, lam on FIT; ALENS_E / ALENS_E2 env as in w9_alens)
Selection on FIT: argmax filtered pass@1 + pass@10, ties -> first in grid order (lam grid ordered from the J-lens down)."""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w11_lib as W1  # noqa: E402
from fm27_lib import excl_mask, head_gamma, unit  # noqa: E402

R = "/workspace/o-lens-improvement"
CENTER = None  # W11C_CENTER=1: score (h - mu_l) (accumulation mean activation, .hf27/w11/mu_2b.pt)
LAMS = [float("inf"), 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]


class MLens:
    def __init__(self, ctx, Meff, base=None, beta=0.0):
        self.model, self.Meff, self.base, self.beta = ctx.model, Meff, base, beta

    def transport(self, h, layer, resid_all=None):
        if CENTER:
            h = h - CENTER[layer].to(h)
        v = h @ self.Meff[layer].T.to(h)
        if self.base is None:
            return v
        return unit(h @ self.base[layer].T.to(h)) + self.beta * unit(v)

    def logits(self, h, l):
        return W1.orig_unembed(self.model)(self.transport(h, l)).float()


def matrices(ctx):
    tag = W1.TAG[ctx.model.d_model]
    if os.path.exists(f"{R}/.hf27/w11/m3b_{tag}.pt"):
        st = torch.load(f"{R}/.hf27/w11/m3b_{tag}.pt", map_location="cpu")
    else:  # 9B: only the deterministic m3x pass exists
        x0 = torch.load(f"{R}/.hf27/w11/m3x_{tag}.pt", map_location="cpu")
        st = dict(meta={}, rho=x0["rho"], n_ctx=x0["n_ctx"], layers=x0["layers"], A_I=None)
    w = os.environ.get("W11C_W", "1"); tn = os.environ.get("W11C_TN", "0") == "1"
    key = ("A" if w == "1" else "A2") + ("_tn" if tn else "")
    bkey = ("B" if w == "1" else "B2") + ("_tn" if tn else "")
    dev = ctx.device
    gam = head_gamma(ctx.model).double().to(dev)
    if w in ("2x", "3x"):  # deterministic top-10 exact p^2 / p^3 estimators (scripts/w11/w11c_m3x.py), both halves summed
        x = torch.load(f"{R}/.hf27/w11/m3x_{tag}.pt", map_location="cpu")
        e = w[0]
        ak = f"A{e}x" + ("_tn" if tn else ""); bk = f"B{e}x" + ("_tn" if tn else "")
        st = dict(st, **{"AX": {l: x[ak][0][l] + x[ak][1][l] for l in x["layers"]}, "BX": x[bk][0] + x[bk][1]})
        st["meta"] = dict(st["meta"], m3x=x["meta"])
        key, bkey = "AX", "BX"
    if w == "0":  # uncentred: A0 = A1 + sum ubar (J^T ubar)^T, B0 = B1 + sum ubar ubar^T (scripts/w11/w11c_m3w0.py)
        w0 = torch.load(f"{R}/.hf27/w11/m3b_{tag}_w0.pt", map_location="cpu")
        st = dict(st, A0=w0["A0"], B0=w0["B0"])
        key, bkey = "A0", "B0"
    B = st[bkey].to(dev).double()
    ev = torch.linalg.eigvalsh(B)
    print(f"[w11c] {bkey}: eig max {float(ev.max()):.3e} min {float(ev.min()):.3e} median {float(ev.median()):.3e} "
          f"cond {float(ev.max() / ev.clamp_min(1e-300).min()):.3e}; frac eig < 1e-6 max {float((ev < 1e-6 * ev.max()).double().mean()):.3f}",
          flush=True)
    s = float(torch.trace(B)) / B.shape[0]
    out = {}
    for l in st["layers"]:
        A = st[key][l].to(dev).double()
        Jz = st["rho"] * gam[:, None] * ctx.J[l].to(dev).double()
        out[l] = dict(A=A, Jz=Jz, AI=None if st["A_I"] is None else st["A_I"][l].to(dev).double() / st["n_ctx"])
    return out, B, s, gam, st


class _Dump:
    """junk check: registers the inner lens's transport outputs so W1.install can record their raw top-10."""

    def __init__(self, inner, model):
        self.inner, self.model, self.reg = inner, model, {}

    def transport(self, h, layer, resid_all=None):
        v = self.inner.transport(h, layer, resid_all).detach().float().clone()
        self.reg[v.untyped_storage().data_ptr()] = (layer, v)
        return v

    def logits(self, x, l):
        return W1.orig_unembed(self.model)(x).float()


def build(ctx):
    lens = _build(ctx)
    dump = os.environ.get("W11_DUMP")
    if not dump:
        return lens
    d = _Dump(lens, ctx.model)
    rec = W1.install(ctx.model, d, dump)
    import atexit
    atexit.register(lambda: torch.save({l: torch.cat(v) for l, v in rec.items()}, dump))
    return d


def _build(ctx):
    global CENTER
    if os.environ.get("W11C_CENTER") == "1":
        mu = torch.load(f"{R}/.hf27/w11/mu_2b.pt", map_location="cpu")["mu"]
        CENTER = {l: v.to(ctx.device).float() for l, v in mu.items()}
    form = os.environ.get("W11C_FORM", "ridge")
    mode = os.environ.get("W11C_MODE", "alone")
    mats, B, s, gam, st = matrices(ctx)
    I = torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
    print(f"[w11c] n_ctx {st['n_ctx']} rho {st['rho']:.4f} meta {st['meta']}", flush=True)

    def Meff_of(lam):
        out = {}
        for l, m in mats.items():
            if form == "Aonly":
                M = m["A"]
            elif form == "ident":
                M = m["AI"]
            elif lam == float("inf"):
                M = m["Jz"]
            else:
                M = torch.linalg.solve(B + lam * s * I, m["A"] + lam * s * m["Jz"])
            out[l] = (M / gam[:, None]).float()
        return out

    for l, m in mats.items():
        if m["AI"] is None:
            continue
        cs = float((m["AI"] * m["Jz"]).sum() / m["AI"].norm() / m["Jz"].norm())
        print(f"[w11c] L{l} identity check: cos(A_I/n, rho*gamma*J) {cs:.4f}, |A_I/n|/|Jz| {float(m['AI'].norm() / m['Jz'].norm()):.3f}",
              flush=True)
    mask = excl_mask(ctx)
    if form == "ridge":
        fl = os.environ.get("W11C_LAM")
        res = []
        for lam in ([float(fl)] if fl else LAMS):
            Me = Meff_of(lam)
            dist = {l: round(float(((Me[l].double() * gam[:, None]) - mats[l]["Jz"]).norm() / mats[l]["Jz"].norm()), 3)
                    for l in mats}
            sc = W1.fit_score(ctx, MLens(ctx, Me), mask)
            print(W1.fmt(f"[w11c fit] lam {lam}", sc) + f" | relF(M, Jz) {dist}", flush=True)
            res.append((lam, sc))
        lam = W1.choose(res)
        print(f"[w11c] chosen lam {lam}", flush=True)
        Me = Meff_of(lam)
    else:
        Me = Meff_of(None)
        print(W1.fmt(f"[w11c fit] {form}", W1.fit_score(ctx, MLens(ctx, Me), mask)), flush=True)
    if mode == "alone":
        return MLens(ctx, Me)
    if mode == "addJPP":
        res = []
        for b in W1.BETAS:
            sc = W1.fit_score(ctx, MLens(ctx, Me, ctx.JPP, b), mask)
            print(W1.fmt(f"[w11c fit] addJPP beta {b}", sc), flush=True)
            res.append((b, sc))
        b = W1.choose(res)
        print(f"[w11c] chosen beta {b}", flush=True)
        return MLens(ctx, Me, ctx.JPP, b)
    if mode == "alens":  # M in place of the base matrix of the A-lens family (ALENS_BASE=JPP path)
        import w9_alens
        ctx.JPP = Me
        os.environ["ALENS_BASE"] = "JPP"
        return w9_alens.build(ctx)
    raise ValueError(mode)
