"""A trained workspace lens, saved and loaded through jpp_lens's ``BaseLens`` so its evals run on it.

Transport at layer l:

    g(h) = J h + W_out relu(W_in h) + b

Every arm of the experiment is this one class:
  * linear (m = 0, no bias):  J-lens form (Jacobian-matched) — the J-lens refit
  * linear + bias:            tuned lens (KL-matched, Belrose et al. 2023)
  * J h + MLP:                the proposed nonlinear lens (Jacobian- or KL-matched)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch as t
from workspace_lens.config import LensConfig
from workspace_lens.lenses.base_lens import BaseLens


class TrainedLens(BaseLens):
    lens_type: ClassVar[str] = "trained"  # type: ignore[assignment]
    _PARAMETER_ATTRIBUTES: ClassVar[dict[str, str]] = {
        **BaseLens._PARAMETER_ATTRIBUTES,
        "w_in": "w_in_L_dict_MN",  # [m, d]
        "w_out": "w_out_L_dict_FM",  # [d, m]
        "bias": "bias_L_dict_F",  # [d]
    }

    def __init__(
        self,
        config: LensConfig,
        *,
        jacobians: dict[int, t.Tensor],
        w_in: dict[int, t.Tensor] | None = None,
        w_out: dict[int, t.Tensor] | None = None,
        bias: dict[int, t.Tensor] | None = None,
    ) -> None:
        super().__init__(config, jacobians=jacobians)
        if (w_in is None) != (w_out is None):
            raise ValueError("w_in and w_out come together")
        layers = self.source_layers
        for name, d in (("w_in", w_in), ("w_out", w_out), ("bias", bias)):
            if d is not None and sorted(d) != layers:
                raise ValueError(f"{name} covers {sorted(d)}, lens has {layers}")
        self.w_in_L_dict_MN = None if w_in is None else {k: v.float() for k, v in w_in.items()}
        self.w_out_L_dict_FM = None if w_out is None else {k: v.float() for k, v in w_out.items()}
        self.bias_L_dict_F = None if bias is None else {k: v.float() for k, v in bias.items()}

    @classmethod
    def _from_parameters(
        cls,
        source_layers: Sequence[int],
        parameters: dict[str, dict[int, t.Tensor | float]],
        config: LensConfig,
    ) -> TrainedLens:
        return cls(
            config,
            jacobians=parameters["jacobians"],  # type: ignore[arg-type]
            w_in=parameters.get("w_in"),  # type: ignore[arg-type]
            w_out=parameters.get("w_out"),  # type: ignore[arg-type]
            bias=parameters.get("bias"),  # type: ignore[arg-type]
        )

    def transport(self, residual_BsN: t.Tensor, layer: int) -> t.Tensor:
        assert self.jacobians_L_dict_FN is not None
        dev = residual_BsN.device
        h = residual_BsN.float()
        out = h @ self.jacobians_L_dict_FN[layer].to(dev).T
        if self.w_in_L_dict_MN is not None:
            assert self.w_out_L_dict_FM is not None
            hidden = t.relu(h @ self.w_in_L_dict_MN[layer].to(dev).T)
            out = out + hidden @ self.w_out_L_dict_FM[layer].to(dev).T
        if self.bias_L_dict_F is not None:
            out = out + self.bias_L_dict_F[layer].to(dev)
        return out.to(residual_BsN.dtype)
