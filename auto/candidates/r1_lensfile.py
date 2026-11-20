"""r1: score a saved lens file (env R1_LENS, a jpp_lens/olens .pt) with the fixed offline scorer: h @ J[layer].T."""
import os

from workspace_lens.utils import load_lens_file


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def build(ctx):
    L = load_lens_file(os.environ["R1_LENS"], hf_model_name="Qwen/Qwen3.5-2B")
    return Lens({int(l): v.to(ctx.device).float() for l, v in L.jacobians_L_dict_FN.items()})
