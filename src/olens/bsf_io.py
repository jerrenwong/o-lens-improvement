"""Block-sparse dictionaries: Goodfire's ``VanillaBSF`` plus input normalisation and save/load.

A dictionary decomposes a vector ``x`` (a residual ``h_l``, or its J-space image ``J_l h_l``) as

    x = mean + sum_g x_g + err,      x_g = z_g @ D_g / scale      (raw units of x)

where ``z = BlockTopK((x - mean) * scale @ W_enc + b_enc)`` keeps the ``l0`` blocks of largest
norm (``bsf.VanillaBSF``; ``group_size = 1`` is an absolute-TopK SAE). ``scale`` is chosen at
fit time so that the mean norm of ``(x - mean) * scale`` is ``sqrt(d)`` (the BSF paper's recipe).

Naming: a dictionary trained on layer 18 residuals with blocks of 4 is ``L18_resid_b4``; on
J-space, ``L18_jspace_b4``.
"""

from __future__ import annotations

from pathlib import Path

import torch as t
from bsf import VanillaBSF

SPACES = ("resid", "jspace")


class BlockDict(t.nn.Module):
    def __init__(self, d: int, n_groups: int, group_size: int, l0: int, *, layer: int, space: str) -> None:
        super().__init__()
        if space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, got {space!r}")
        self.bsf = VanillaBSF(d, n_groups, group_size=group_size, l0=l0)
        self.d, self.n_groups, self.group_size, self.l0 = d, n_groups, group_size, l0
        self.layer, self.space = layer, space
        self.register_buffer("mean", t.zeros(d))
        self.register_buffer("scale", t.ones(()))

    @property
    def name(self) -> str:
        return f"L{self.layer}_{self.space}_b{self.group_size}"

    # ---- normalisation
    @t.no_grad()
    def set_normalisation(self, x: t.Tensor) -> None:
        """Estimate ``mean`` and ``scale`` from a sample ``x [N, d]`` (raw units)."""
        x = x.float()
        self.mean.copy_(x.mean(0))
        self.scale.copy_(self.d**0.5 / (x - self.mean).norm(dim=-1).mean())

    def normalise(self, x: t.Tensor) -> t.Tensor:
        return (x.float() - self.mean) * self.scale

    # ---- coding (inputs in raw units)
    def preact(self, x: t.Tensor) -> t.Tensor:
        """Pre-TopK block codes ``[N, G, b]``."""
        a = self.normalise(x) @ self.bsf.W_enc + self.bsf.b_enc
        return a.view(a.shape[0], self.n_groups, self.group_size)

    def encode(self, x: t.Tensor) -> t.Tensor:
        """Block codes ``z [N, G, b]`` (normalised units), zero outside the ``l0`` kept blocks."""
        return self.bsf.encode(self.normalise(x))

    def decode(self, z: t.Tensor) -> t.Tensor:
        """Sum of block contributions in raw units, WITHOUT the mean: ``sum_g x_g``."""
        return self.bsf.decode(z) / self.scale

    def atoms(self) -> t.Tensor:
        """Decoder atoms ``[G, b, d]`` (unit-norm rows; raw direction, divide by scale for units)."""
        return self.bsf.atoms()

    def split(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
        """``(z, blocks, err)`` with ``x == mean + blocks + err``; ``blocks = decode(z)``."""
        z = self.encode(x)
        blocks = self.decode(z)
        return z, blocks, x.float() - self.mean - blocks

    # ---- io
    def save(self, path: str | Path, **extra) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        t.save(
            {
                "meta": {"d": self.d, "n_groups": self.n_groups, "group_size": self.group_size,
                         "l0": self.l0, "layer": self.layer, "space": self.space, **extra},
                "state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | t.device = "cpu") -> "BlockDict":
        ck = t.load(path, map_location="cpu", weights_only=True)
        m = ck["meta"]
        obj = cls(m["d"], m["n_groups"], m["group_size"], m["l0"], layer=m["layer"], space=m["space"])
        obj.load_state_dict(ck["state"])
        obj.meta = m
        return obj.to(device).eval()
