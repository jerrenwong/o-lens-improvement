"""ex2: the Frobenius statistics of s12_stats.py, importable (same definitions)."""
import numpy as np


def pct(null, obs):
    null = np.asarray(null)
    return float(100 * ((null < obs).mean() + 0.5 * (null == obs).mean()))


def fro_stats(U, G):
    """U [k,8] weights; G 9x9 Gram over (E_0..E_7, Jbar). Returns dict of arrays [k]."""
    GE, g, GJ = G[:8, :8], G[:8, 8], G[8, 8]
    n2 = np.einsum("ke,ef,kf->k", U, GE, U)
    ip = U @ g
    l1 = np.abs(U).sum(1)
    c = U.sum(1)
    GD = GE - g[:, None] - g[None, :] + GJ
    dev2 = np.einsum("ke,ef,kf->k", U, GD, U)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(c_rel=c / l1, R2=ip ** 2 / (n2 * GJ), cos=ip / np.sqrt(n2 * GJ), beta=ip / GJ / l1,
                    share_cJ=c ** 2 * GJ / n2, share_dev=dev2 / n2)
