"""xm1 copy of scripts/perf1/lite_model.py; only change: a checkpoint with a single model.safetensors (no index
file, e.g. google/gemma-4-12B-it) is read through a weight map built from that file's own keys.

perf1: a stand-in for the jlens HFLensModel that holds only the final norm and the LM head.

The scorers (auto/evaluate*.py) only need ``model.unembed`` (final norm + LM head in bf16),
``model.tokenizer`` and ``model.d_model`` for most candidates; the bench file already caches the
residuals. ``get_lite_model(name)`` loads just ``<...>.norm.weight`` and ``lm_head.weight`` from the
safetensors shards (no other tensor is read), builds the norm and the head with the SAME module
classes transformers would build (the model is instantiated on the meta device, then only those two
modules are materialised on the GPU in bf16), and reuses ``HFLensModel.unembed`` itself, so the
unembed op sequence is the same code. The tokenizer is loaded and BOS-forced exactly as
``workspace_lens.get_hf_model`` + ``jlens.from_hf`` do.

Any other attribute (``layers``, ``forward``, ``_hf_model``, ``input_device``, ...) raises
LiteModelError (a RuntimeError, not an AttributeError, so ``hasattr``/``getattr(..., default)``
cannot silently take another branch).

    model = get_lite_model("Qwen/Qwen3.5-9B")
    logits = model.unembed(vecs)          # == get_hf_model(...).unembed(vecs), bitwise (see check_lite.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

LITE_HELP = "this needs the full model; rerun the scorer without --lite"

# Checkpoint keys tried in order (multimodal wrapper first, then plain causal-LM layout).
_NORM_KEYS = ("model.language_model.norm.weight", "model.norm.weight", "language_model.model.norm.weight")
_HEAD_KEYS = ("lm_head.weight", "language_model.lm_head.weight")
_EMBED_KEYS = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")


class LiteModelError(RuntimeError):
    pass


def lite_blocked(what: str):
    """A callable placeholder for a ctx entry that is unavailable under --lite."""

    def _raise(*_args, **_kwargs):
        raise LiteModelError(f"--lite: {what} is not available ({LITE_HELP})")

    return _raise


class LiteLensModel:
    """Final norm + LM head + tokenizer only. ``unembed`` is ``jlens.hf.HFLensModel.unembed``."""

    _ALLOWED = frozenset({"tokenizer", "d_model", "n_layers", "_final_norm", "_lm_head", "_logit_softcap",
                          "unembed", "model_name", "lite"})

    def __init__(self, model_name, tokenizer, final_norm, lm_head, d_model, n_layers, logit_softcap):
        from jlens.hf import HFLensModel

        d = self.__dict__
        d["model_name"] = model_name
        d["lite"] = True
        d["tokenizer"] = tokenizer
        d["_final_norm"] = final_norm
        d["_lm_head"] = lm_head
        d["d_model"] = d_model
        d["n_layers"] = n_layers
        d["_logit_softcap"] = logit_softcap
        d["_unembed_impl"] = HFLensModel.unembed

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.__dict__["_unembed_impl"](self, residual)

    def __getattr__(self, name):  # only called for attributes not found normally
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)  # keep copy/pickle/introspection protocols working
        raise LiteModelError(f"--lite model has no {name!r} (only final norm + LM head + tokenizer are loaded); "
                             f"{LITE_HELP}")

    def __repr__(self) -> str:
        return f"LiteLensModel({self.model_name}, d_model={self.d_model}, final norm + LM head only)"


def _snapshot_file(model_name: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_name, filename)


def _read_tensor(model_name: str, weight_map: dict, keys: tuple[str, ...], what: str) -> tuple[str, torch.Tensor]:
    from safetensors import safe_open

    hits = [k for k in keys if k in weight_map]
    if len(hits) != 1:
        raise LiteModelError(f"--lite: expected exactly one {what} key among {keys} in the checkpoint index, "
                             f"found {hits}; {LITE_HELP}")
    key = hits[0]
    shard = _snapshot_file(model_name, weight_map[key])
    with safe_open(shard, framework="pt", device="cpu") as f:
        return key, f.get_tensor(key)


@torch.no_grad()
def get_lite_model(model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> LiteLensModel:
    from jlens.hf import _find_layout, _resolve_attr_path
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    text_config = config.get_text_config()

    # Same classes as get_hf_model's AutoModelForCausalLM.from_pretrained, built without weights.
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config, dtype=dtype)
    layout = _find_layout(skeleton)
    final_norm = getattr(_resolve_attr_path(skeleton, layout.path), layout.norm)
    lm_head = getattr(skeleton, layout.lm_head)
    for m in (final_norm, lm_head):
        if list(m.buffers()):
            raise LiteModelError(f"--lite: {type(m).__name__} has buffers; not supported ({LITE_HELP})")
    norm_params = dict(final_norm.named_parameters())
    head_params = dict(lm_head.named_parameters())
    if set(norm_params) != {"weight"} or set(head_params) != {"weight"}:
        raise LiteModelError(f"--lite: unexpected parameters norm={list(norm_params)} head={list(head_params)} "
                             f"({LITE_HELP})")
    del skeleton

    try:
        index = json.loads(Path(_snapshot_file(model_name, "model.safetensors.index.json")).read_text())["weight_map"]
    except Exception:  # xm1: single-file checkpoint
        from safetensors import safe_open
        with safe_open(_snapshot_file(model_name, "model.safetensors"), framework="pt", device="cpu") as f:
            index = {k: "model.safetensors" for k in f.keys()}
    _, norm_w = _read_tensor(model_name, index, _NORM_KEYS, "final norm")
    tied = bool(getattr(text_config, "tie_word_embeddings", False) or getattr(config, "tie_word_embeddings", False))
    if tied:
        _, head_w = _read_tensor(model_name, index, _EMBED_KEYS, "tied embedding")
    else:
        _, head_w = _read_tensor(model_name, index, _HEAD_KEYS, "LM head")

    final_norm = final_norm.to_empty(device=device)
    lm_head = lm_head.to_empty(device=device)
    for module, w in ((final_norm, norm_w), (lm_head, head_w)):
        if tuple(module.weight.shape) != tuple(w.shape):
            raise LiteModelError(f"--lite: shape mismatch {tuple(module.weight.shape)} vs {tuple(w.shape)}")
        module.weight.copy_(w.to(dtype))  # from_pretrained(dtype=bf16) casts the same way
        module.weight.requires_grad_(False)
        module.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Same BOS rule as jlens.hf.HFLensModel(force_bos=True).
    if getattr(tokenizer, "bos_token_id", None) is not None and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = True

    return LiteLensModel(
        model_name, tokenizer, final_norm, lm_head,
        d_model=text_config.hidden_size, n_layers=text_config.num_hidden_layers,
        logit_softcap=getattr(text_config, "final_logit_softcapping", None),
    )
