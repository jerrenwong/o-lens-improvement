"""Offline readout ranks from cached residuals, matching the jpp_lens readout eval.

A *bench* (``scripts/bench_cache.py``) holds, for every scoreable eval item, the residual
at its readout position at several layers, its candidate token ids per intermediate, its
fit / held-out split and the Readout-Filtering exclusion ids. :func:`readout_rows` turns
lens outputs (final-layer-basis vectors, one per item) into rows of the harness CSV schema
``eval,lens,item,intermediate,layer,rank,n_candidates`` with the harness's rank rule:
rank = 1 + number of vocab logits strictly greater than the best candidate's logit
(``min_rank_over_candidates``), excluded ids forced to -inf first when ``filtered``.

The logits are ``model.unembed(vec).float()``, exactly as ``masked_lens_logits`` computes
them (bf16 final norm + LM head). The harness unembeds one row at a time; here rows go in
chunks, so a bf16 GEMM can occasionally round one logit differently and flip a tie.
``chunk_size=1`` reproduces the harness call shape.

    bench = Bench.load(path)
    idx = bench.index("held-out")
    vecs = bench.resid[layer] @ J.T                 # [P, d], aligned with bench items
    rows = readout_rows("my_lens", layer, bench, vecs, model, filtered=True, items=idx)
    write_csv(out, rows)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

CSV_COLUMNS = ("eval", "lens", "item", "intermediate", "layer", "rank", "n_candidates")
DEFAULT_CHUNK = 64


### BENCH


@dataclass
class Bench:
    model: str
    layers: list[int]
    eval_slugs: list[str]
    item_names: list[str]
    resid: dict[int, Tensor]  # [P, d] fp32 cpu
    candidates: list[dict[str, list[int]]]
    split: list[str]
    excluded_ids: Tensor
    readout_positions: list[int] | None = None
    _pairs_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path) -> Bench:
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            model=d["model"],
            layers=[int(layer) for layer in d["layers"]],
            eval_slugs=list(d["eval_slugs"]),
            item_names=list(d["item_names"]),
            resid={int(layer): v.float() for layer, v in d["resid"].items()},
            candidates=[
                {str(k): [int(i) for i in v] for k, v in c.items()} for c in d["candidates"]
            ],
            split=list(d["split"]),
            excluded_ids=torch.as_tensor(d["excluded_ids"], dtype=torch.long),
            readout_positions=(
                [int(p) for p in d["readout_positions"]] if d.get("readout_positions") is not None else None
            ),
        )

    @property
    def num_items(self) -> int:
        return len(self.item_names)

    def index(self, split: str | None = None) -> list[int]:
        """Item indices in ``split`` ("fit" / "held-out"); ``None`` = all items."""
        if split is None:
            return list(range(self.num_items))
        if split not in ("fit", "held-out"):
            raise ValueError(f"split must be 'fit', 'held-out' or None, got {split!r}")
        return [i for i, s in enumerate(self.split) if s == split]

    def keys(self, items: list[int] | None = None) -> list[tuple[str, str]]:
        idx = self.index() if items is None else items
        return [(self.eval_slugs[i], self.item_names[i]) for i in idx]


### LOGITS AND RANKS


def _excluded_mask(excluded_ids: Tensor | None, vocab: int, device) -> Tensor | None:
    if excluded_ids is None or len(excluded_ids) == 0:
        return None
    mask = torch.zeros(vocab, dtype=torch.bool, device=device)
    mask[torch.as_tensor(excluded_ids, dtype=torch.long, device=device)] = True
    return mask


@torch.no_grad()
def unembed_logits(model, vecs_PF: Tensor, excluded_ids: Tensor | None = None, *, chunk_size: int = DEFAULT_CHUNK) -> Tensor:
    """``model.unembed(vecs).float()`` ([P, V] fp32 on the LM-head device), excluded ids
    -> -inf. Processes ``chunk_size`` rows at a time; the full [P, V] result is returned, so
    for large P prefer :func:`readout_rows`, which ranks chunk by chunk."""
    out = [chunk_logits for chunk_logits in _iter_logits(model, vecs_PF, excluded_ids, chunk_size)]
    return torch.cat(out, dim=0)


def _iter_logits(model, vecs_PF: Tensor, excluded_ids: Tensor | None, chunk_size: int):
    mask = None
    for start in range(0, vecs_PF.shape[0], chunk_size):
        # Same op sequence as masked_lens_logits: unembed (casts to the head dtype/device), .float(), mask.
        logits = model.unembed(vecs_PF[start : start + chunk_size]).float()
        if excluded_ids is not None and len(excluded_ids) > 0:
            if mask is None:
                mask = _excluded_mask(excluded_ids, logits.shape[-1], logits.device)
            logits = logits.masked_fill(mask, float("-inf"))
        yield logits


def candidate_ranks(logits_PV: Tensor, candidates: list[dict[str, list[int]]]) -> list[dict[str, int]]:
    """Per row, ``{intermediate: rank}`` with the harness rule: 1 + #(logits > max over the
    intermediate's candidate logits). ``candidates[p]`` belongs to row ``p``."""
    if logits_PV.shape[0] != len(candidates):
        raise ValueError(f"{logits_PV.shape[0]} logit rows but {len(candidates)} candidate dicts")
    rows, cand_idx, cand_valid, labels = _pair_tensors(candidates, logits_PV.device)
    ranks = _ranks_from_pairs(logits_PV, rows, cand_idx, cand_valid)
    out: list[dict[str, int]] = [{} for _ in candidates]
    for (p, inter), r in zip(labels, ranks.tolist(), strict=True):
        out[p][inter] = int(r)
    return out


def _pair_tensors(candidates: list[dict[str, list[int]]], device):
    """Flatten (row, intermediate) pairs: row index [K], padded candidate ids [K, C] and
    their validity mask, plus the (row, intermediate) labels in order."""
    labels: list[tuple[int, str]] = []
    id_lists: list[list[int]] = []
    for p, cand in enumerate(candidates):
        for inter, ids in cand.items():
            if not ids:
                raise ValueError(f"row {p} intermediate {inter!r} has no candidate ids")
            labels.append((p, inter))
            id_lists.append(list(ids))
    width = max((len(ids) for ids in id_lists), default=1)
    cand_idx = torch.zeros(len(id_lists), width, dtype=torch.long)
    cand_valid = torch.zeros(len(id_lists), width, dtype=torch.bool)
    for k, ids in enumerate(id_lists):
        cand_idx[k, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        cand_valid[k, : len(ids)] = True
    rows = torch.tensor([p for p, _ in labels], dtype=torch.long)
    return rows.to(device), cand_idx.to(device), cand_valid.to(device), labels


def _ranks_from_pairs(logits_PV: Tensor, rows: Tensor, cand_idx: Tensor, cand_valid: Tensor) -> Tensor:
    if rows.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    cand_logits = logits_PV[rows[:, None], cand_idx]  # [K, C]
    cand_logits = cand_logits.masked_fill(~cand_valid, float("-inf"))
    best = cand_logits.max(dim=1).values  # [K]
    # Per pair: count of strictly greater logits in that pair's row (ties favour the lens).
    ranks = torch.empty(rows.shape[0], dtype=torch.long, device=logits_PV.device)
    for start in range(0, rows.shape[0], 64):
        sl = slice(start, start + 64)
        ranks[sl] = (logits_PV[rows[sl]] > best[sl, None]).sum(dim=1) + 1
    return ranks.cpu()


@torch.no_grad()
def readout_rows(
    lens_name: str,
    layer: int,
    bench: Bench,
    vecs_PF: Tensor,
    model,
    *,
    filtered: bool,
    items: list[int] | None = None,
    chunk_size: int = DEFAULT_CHUNK,
) -> list[dict]:
    """Harness-schema rows for ``items`` (bench indices, default all) at ``layer``.

    ``vecs_PF`` are the lens outputs in the final-layer basis (pre final norm). Either one row
    per bench item (``len == bench.num_items``; rows ``items`` are used) or one row per entry of
    ``items`` in that order. ``filtered`` applies ``bench.excluded_ids`` (Readout Filtering)."""
    idx = bench.index() if items is None else list(items)
    if vecs_PF.shape[0] == bench.num_items:
        sel = vecs_PF if items is None else vecs_PF[torch.as_tensor(idx, dtype=torch.long, device=vecs_PF.device)]
    elif vecs_PF.shape[0] == len(idx):
        sel = vecs_PF
    else:
        raise ValueError(
            f"vecs_PF has {vecs_PF.shape[0]} rows; expected {bench.num_items} (all bench items) "
            f"or {len(idx)} (the selected items)"
        )
    excluded = bench.excluded_ids if filtered else None
    if filtered:
        _check_not_all_excluded(bench, idx)

    rows_out: list[dict] = []
    for c, logits in enumerate(_iter_logits(model, sel, excluded, chunk_size)):
        chunk_items = idx[c * chunk_size : c * chunk_size + logits.shape[0]]
        cands = [bench.candidates[i] for i in chunk_items]
        pr, ci, cv, labels = _pair_tensors(cands, logits.device)
        ranks = _ranks_from_pairs(logits, pr, ci, cv)
        for (p, inter), r in zip(labels, ranks.tolist(), strict=True):
            i = chunk_items[p]
            rows_out.append(
                {
                    "eval": bench.eval_slugs[i],
                    "lens": lens_name,
                    "item": bench.item_names[i],
                    "intermediate": inter,
                    "layer": int(layer),
                    "rank": int(r),
                    "n_candidates": len(bench.candidates[i][inter]),
                }
            )
    return rows_out


def _check_not_all_excluded(bench: Bench, idx: list[int]) -> None:
    key = ("excluded_set",)
    if key not in bench._pairs_cache:
        bench._pairs_cache[key] = set(bench.excluded_ids.tolist())
    excluded = bench._pairs_cache[key]
    for i in idx:
        for inter, ids in bench.candidates[i].items():
            if ids and all(t in excluded for t in ids):
                raise ValueError(
                    f"{bench.eval_slugs[i]}/{bench.item_names[i]}: every candidate of {inter!r} is excluded"
                )


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write rows with the harness header (in place, no temp-then-rename)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


### CLOSED-FORM FINAL NORM + LM HEAD


def _final_norm_and_head(model):
    norm = getattr(model, "_final_norm", None)
    head = getattr(model, "_lm_head", None)
    if norm is None or head is None:
        raise TypeError("model has no _final_norm / _lm_head (expected jlens HFLensModel)")
    return norm, head


@torch.no_grad()
def final_linear(model, *, probe_vecs: Tensor | None = None, tol: float = 1e-2) -> tuple[Tensor, Tensor]:
    """``(W [V, d], gamma [d])`` fp32 on the LM-head device such that
    ``model.unembed(v) ~= (v * gamma / rms(v)) @ W.T`` with ``rms(v) = sqrt(mean(v**2))``
    (the norm's eps is ignored; negligible for real residuals).

    gamma is measured, not read: the norm is applied to a vector of ones in fp32, which works
    for both ``weight * x_hat`` (Llama-style) and ``(1 + weight) * x_hat`` (Qwen3-Next /
    Qwen3.5-style) RMSNorms. Raises if there is a logit soft-cap, an LM-head bias, if the norm
    is not an RMSNorm (checked on random vectors), or if the closed form misses
    ``model.unembed`` by more than ``tol`` relative error (:func:`check_final_linear`)."""
    norm, head = _final_norm_and_head(model)
    if getattr(model, "_logit_softcap", None) is not None:
        raise ValueError(f"model has a final logit soft-cap ({model._logit_softcap}); unembed is not linear")
    if getattr(head, "bias", None) is not None:
        raise ValueError("LM head has a bias; final_linear does not model it")
    device = head.weight.device
    d = head.weight.shape[1]
    eps = getattr(norm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(norm, "eps", None)
    eps = float(eps) if eps is not None else 0.0

    norm_device = _module_device(norm, device)
    ones = torch.ones(1, d, dtype=torch.float32, device=norm_device)
    gamma = norm(ones).float()[0] * (1.0 + eps) ** 0.5  # rms(ones) = 1 -> output = gamma / sqrt(1 + eps)

    # The norm itself must be x * gamma / sqrt(mean(x^2) + eps) (no centring, no bias).
    g = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.randn(8, d, generator=g) * 3.0 + torch.randn(8, 1, generator=g)).to(norm_device)
    ref = norm(x).float()
    closed = x * gamma / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)
    norm_err = ((ref - closed).norm() / ref.norm()).item()
    if not norm_err < 1e-3:
        raise ValueError(
            f"final norm {type(norm).__name__} is not x*gamma/rms(x) (rel err {norm_err:.2e} on fp32 "
            "probes); final_linear supports RMSNorm only"
        )

    W = head.weight.detach().float().to(device)
    gamma = gamma.to(device)
    err = check_final_linear(model, W, gamma, probe_vecs=probe_vecs)
    if not err < tol:
        raise ValueError(f"closed-form unembed misses model.unembed by rel err {err:.2e} (> {tol})")
    return W, gamma


def _module_device(module, default):
    for p in module.parameters():
        return p.device
    return default


@torch.no_grad()
def check_final_linear(model, W: Tensor, gamma: Tensor, *, probe_vecs: Tensor | None = None) -> float:
    """Max over probe vectors of ||closed - unembed|| / ||unembed|| (fp32 logits). The closed
    form takes the probe rounded to the LM-head dtype, as ``unembed`` does, so the remaining
    error is bf16 rounding inside the norm and the head. Default probes: 8 random Gaussian
    vectors with a non-zero mean."""
    _, head = _final_norm_and_head(model)
    device = W.device
    d = W.shape[1]
    if probe_vecs is None:
        g = torch.Generator(device="cpu").manual_seed(1)
        probe_vecs = torch.randn(8, d, generator=g) * 2.0 + 0.5
    v = probe_vecs.to(device).float()
    ref = model.unembed(v).float()
    vq = v.to(head.weight.dtype).float()
    closed = (vq * gamma / torch.sqrt((vq * vq).mean(-1, keepdim=True))) @ W.T
    errs = (closed - ref).norm(dim=-1) / ref.norm(dim=-1)
    return float(errs.max().item())
