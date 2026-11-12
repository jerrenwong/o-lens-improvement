"""Check the fla kernel path against transformers' pure-torch Gated-DeltaNet path.

Same batch, same u: compares the recorded residuals h and the Jacobian targets g of
olens.train.model_pass per layer, and times both paths. Run on the GPU pod.
"""

import time

import torch as t
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from workspace_lens import get_hf_model

from olens.train import model_pass, wikitext_windows
from pathlib import Path

model = get_hf_model("Qwen/Qwen3.5-2B", attn_implementation="sdpa")
for p in model._hf_model.parameters():
    p.requires_grad_(False)
src, tgt = list(range(23)), 23
ids = wikitext_windows(model.tokenizer, "validation", 64, 128, Path("/root/cache"))[:8].cuda().long()
u = t.randn(ids.shape[0], model.d_model, device="cuda", generator=t.Generator("cuda").manual_seed(0))
gdn = [m for m in model._hf_model.modules() if hasattr(m, "chunk_gated_delta_rule")]
print(f"{len(gdn)} linear-attention modules; kernel = {gdn[0].chunk_gated_delta_rule.__module__}")


def run():
    t.cuda.synchronize(); t0 = time.time()
    h, g, *_ = model_pass(model, ids, src, tgt, u, 0, None)
    t.cuda.synchronize()
    return h.float(), g, time.time() - t0


run()  # warm-up / compile
h_fla, g_fla, s_fla = run()
for m in gdn:
    m.chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule
run()
h_ref, g_ref, s_ref = run()
print(f"time per pass: fla {s_fla:.3f}s  torch {s_ref:.3f}s  ({s_ref / s_fla:.1f}x)")
for l in src:
    eh = ((h_fla[l] - h_ref[l]).norm() / h_ref[l].norm()).item()
    eg = ((g_fla[l] - g_ref[l]).norm() / g_ref[l].norm()).item()
    cos = t.nn.functional.cosine_similarity(g_fla[l], g_ref[l], dim=-1).mean().item()
    print(f"layer {l:2d}  rel_err h {eh:.2e}  g {eg:.2e}  mean cos(g) {cos:.4f}")
