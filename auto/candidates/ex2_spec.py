"""ex2: a lens given as a per-layer linear combination of stored matrices (experts, pooled lens, J++ files,
released lenses). env EX2_SPECS = JSON file of specs, EX2_NAME = the spec to build (scripts/ex2/ex2_store.py).
Transport h @ M[layer].T, as auto/candidates/jpp.py."""
import json
import os
import sys

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/ex2")
import ex2_store  # noqa: E402


class Lens:
    def __init__(self, M):
        self.M = M

    def transport(self, h, layer, resid_all=None):
        return h @ self.M[layer].T


def build(ctx):
    spec = json.load(open(os.environ["EX2_SPECS"]))[os.environ["EX2_NAME"]]
    return Lens(ex2_store.build(spec, ctx))
