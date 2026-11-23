"""tl1 (PREREG_v4 Addendum F, text-only control): shared helpers for the T-lens input feature.

feature(row) = mean over the row's prompt tokens [0 .. readout_pos] of the model's INPUT embedding rows (raw
embedding matrix, fp32 mean), no forward pass.

Bench rows: the prompt is the harness item's prompt (jpp_lens data dir for the original benches, the md5-checked
data/evals_v4 files for v4), tokenized as HFLensModel.encode does (tokenizer(prompt, truncation=True,
max_length=512), BOS rule of the loaded tokenizer), readout position by the harness's own rule
(readout_evals._item_readout_position). Only the prompt is read; the position must equal the bench's cached
readout_positions (asserted). verify_forward() additionally runs the full model on each row and checks that the
block outputs at the readout position reproduce the bench's cached residuals (as scripts/why1_v3.py does).

Embedding matrix: read from the checkpoint's safetensors (works with the --lite loaders, which do not keep it)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
JPP = os.environ.get("JPP_LENS_DIR", "/root/jpp_lens")
MAX_SEQ_LEN = 512  # jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN (asserted below when jpp_cli is importable)


def load_embedding(model_name: str, retries: int = 4) -> torch.Tensor:
    """Input embedding matrix [V, d] (checkpoint dtype, cpu), read from the safetensors shards."""
    sys.path.insert(0, str(R / "scripts" / "xm1"))
    from lite_model_x import _EMBED_KEYS, _read_tensor, _snapshot_file
    err = None
    for _ in range(retries):  # network fs: snapshot symlinks sometimes read as 0 bytes
        try:
            try:
                index = json.loads(Path(_snapshot_file(model_name, "model.safetensors.index.json")).read_text())["weight_map"]
            except Exception:  # single-file checkpoint
                from safetensors import safe_open
                with safe_open(_snapshot_file(model_name, "model.safetensors"), framework="pt", device="cpu") as f:
                    index = {k: "model.safetensors" for k in f.keys()}
            _, w = _read_tensor(model_name, index, _EMBED_KEYS, "input embedding")
            return w
        except Exception as e:  # noqa: BLE001
            err = e
            print("embedding load retry:", repr(e)[:200], flush=True)
    raise err


def _harness():
    if JPP not in sys.path:
        sys.path.insert(0, os.path.join(JPP, "src"))
    from lens_evals.readout_evals import readout_evals as re_mod
    from lens_evals.readout_evals.readout_eval_items import READOUT_EVALS_DATA_DIR, load_readout_eval_items
    return re_mod, READOUT_EVALS_DATA_DIR, load_readout_eval_items


def load_items(v4: bool) -> dict:
    """(eval_slug, name) -> harness item, from the same files the bench was cached from."""
    re_mod, data_dir, load = _harness()
    if not v4:
        items = load(os.path.join(JPP, data_dir))
    else:
        v4d = R / "data/evals_v4"
        manifest = json.loads((v4d / "MANIFEST.json").read_text())
        tmp = tempfile.mkdtemp(prefix="tl1_v4evals_")
        for fname, meta in manifest["files"].items():
            b = (v4d / fname).read_bytes()
            if hashlib.md5(b).hexdigest() != meta["md5"]:
                raise SystemExit(f"{fname}: md5 differs from MANIFEST.json")
            Path(tmp, f"lens-eval-{meta['task']}.json").write_bytes(b)
        from lens_evals.readout_evals.readout_eval_items import MACRO_EVALS
        items = load(tmp, slugs=MACRO_EVALS)  # as scripts/v4_bench_cache.py (v4 has no order-ops file)
    return {(it.eval_slug, it.name): it for it in items}


def row_tokens(tokenizer, bench, rows, items) -> list[list[int]]:
    """Token ids [0 .. readout_pos] of each bench row; position asserted equal to the bench's cached one."""
    re_mod, _, _ = _harness()
    assert bench.readout_positions is not None, "bench has no cached readout positions"
    out, bad = [], []
    for i in rows:
        it = items[(bench.eval_slugs[i], bench.item_names[i])]
        ids = tokenizer(it.prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).input_ids
        p = re_mod._item_readout_position(ids, tokenizer, it)
        if p != bench.readout_positions[i]:
            bad.append((bench.eval_slugs[i], bench.item_names[i], p, bench.readout_positions[i]))
        out.append(ids[0, :p + 1].tolist())
    assert not bad, f"readout positions differ from the bench cache: {bad[:5]} ({len(bad)} rows)"
    return out


def mean_embed(EMB: torch.Tensor, seqs: list[list[int]], dev="cuda") -> torch.Tensor:
    E = EMB.to(dev)
    return torch.stack([E[torch.tensor(s, device=dev)].float().mean(0) for s in seqs])


@torch.no_grad()
def verify_forward(model, bench, rows, seqs, check_layers=None, thr=0.999) -> float:
    """Full model: cos(block-l output at the readout position, bench.resid[l][row]) >= thr at two layers
    (as scripts/why1_v3.py). Feeds the row's tokens [0 .. pos] (causal: same states at pos as the full prompt)."""
    layers = check_layers or [bench.layers[len(bench.layers) // 3], bench.layers[2 * len(bench.layers) // 3]]
    cmin, bad = 1.0, []
    dev = model.input_device
    for i, s in zip(rows, seqs):
        rec = {}
        hk = [model.layers[l].register_forward_hook(
            (lambda l: lambda m, a, o: rec.__setitem__(l, (o if torch.is_tensor(o) else o[0])[0, -1].float()))(l))
            for l in layers]
        try:
            model.forward(torch.tensor([s], device=dev))
        finally:
            for h in hk:
                h.remove()
        c = min(torch.cosine_similarity(rec[l].cpu(), bench.resid[l][i].float(), 0).item() for l in layers)
        cmin = min(cmin, c)
        if c < thr:
            bad.append((bench.eval_slugs[i], bench.item_names[i], round(c, 5)))
    print(f"[tl1 verify-fwd] {len(rows)} rows, layers {layers}, min cos {cmin:.6f}, below {thr}: {len(bad)} {bad[:5]}",
          flush=True)
    assert not bad, "token ids / readout positions do not reproduce the cached residuals"
    return cmin
