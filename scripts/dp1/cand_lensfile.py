"""dp1: score a saved lens file (env DP1_LENS, a jpp_lens .pt; env DP1_MODEL) with the fixed scorer: h @ J[layer].T
(pattern of auto/candidates/r2_lensfile.py)."""
import os

from workspace_lens.utils import load_lens_file


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def build(ctx):
    L = load_lens_file(os.environ["DP1_LENS"], hf_model_name=os.environ["DP1_MODEL"])
    return Lens({int(l): v.to(ctx.device).float() for l, v in L.jacobians_L_dict_FN.items()})
