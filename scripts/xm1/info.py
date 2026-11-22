"""xm1 step 1: architecture facts for a new model, loaded exactly as the harness loads it (workspace_lens.get_hf_model).
Prints classes, dtype, depth, d_model, vocab, the 7 readout layers (k*depth/8), tokenizer BOS behaviour, the chat
template's thinking switch, LRP patcher coverage (dry patch report), and checks that unembed == final norm + linear head
(olens.readout.check_final_linear) and that block-l output == hidden_states[l+1] == bench residual convention.
    HF_HOME=... HF_HUB_OFFLINE=1 python scripts/xm1/info.py <model> <out.json>"""
import json
import sys

sys.modules["fla"] = None  # as scripts/r1_jpp.py does for fit-shard: the LRP surgery refuses while fla is importable
import torch  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.lrp.lrp import apply_lrp_rules, lrp_rule_config_for_mode  # noqa: E402

from olens.readout import check_final_linear, final_linear  # noqa: E402

M, OUT = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False)
model = None
for attempt in range(3):  # network fs: snapshot symlinks sometimes read as 0 bytes
    try:
        model = get_hf_model(M, attn_implementation="sdpa")
        break
    except Exception as e:  # noqa: BLE001
        print("load retry", attempt, repr(e)[:300], flush=True)
hf = model._hf_model
tok = model.tokenizer
cfg = hf.config
tc = cfg.get_text_config()
depth = model.n_layers
layers = [k * depth // 8 for k in range(1, 8)]
info = dict(
    model=M, hf_class=type(hf).__name__, text_module=type(model._text_module).__name__,
    layout_path=model.layout.path, block_class=type(model.layers[0]).__name__,
    mlp_class=type(model.layers[0].mlp).__name__, final_norm_class=type(model._final_norm).__name__,
    dtype=str(next(hf.parameters()).dtype), n_layers=depth, d_model=model.d_model,
    lm_head_rows=int(model._lm_head.weight.shape[0]), len_tokenizer=len(tok),
    tied=bool(getattr(tc, "tie_word_embeddings", False) or getattr(cfg, "tie_word_embeddings", False)),
    logit_softcap=getattr(tc, "final_logit_softcapping", None), readout_layers=layers,
    bos_token=tok.bos_token, bos_id=tok.bos_token_id, add_bos_token=getattr(tok, "add_bos_token", None),
    n_params_B=sum(p.numel() for p in hf.parameters()) / 1e9,
)
ids = tok("The capital of France is").input_ids
info["tok_example"] = [tok.decode([i]) for i in ids]
msgs = [{"role": "user", "content": "Hi"}]
s_off = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
s_on = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
s_def = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
info["chat_off"], info["chat_on"], info["chat_default"] = s_off, s_on, s_def
info["template_has_thinking_switch"] = s_off != s_on
info["template_mentions_enable_thinking"] = "enable_thinking" in (tok.chat_template or "")
# LRP coverage (authors' patcher, rlens rules): which modules get patched
with apply_lrp_rules(hf, lrp_rule_config_for_mode("rlens")) as rep:
    pass
pn = rep.patched_names()
info["lrp_patched_counts"] = {k: len(v) for k, v in pn.items()}
info["lrp_patched_examples"] = {k: sorted(v)[:3] for k, v in pn.items()}
# unembed == final norm + linear head
probe = torch.randn(64, model.d_model, device="cuda") * 3
try:
    W, gamma = final_linear(model, probe_vecs=probe)
    info["final_linear_maxdiff"] = float(check_final_linear(model, W, gamma, probe_vecs=probe))
except ValueError as e:  # Gemma: final logit soft-cap (monotone per logit); check the pre-cap part is norm + head
    info["final_linear_error"] = str(e)
    cap = model._logit_softcap
    model._logit_softcap = None
    W, gamma = final_linear(model, probe_vecs=probe)
    info["final_linear_maxdiff_precap"] = float(check_final_linear(model, W, gamma, probe_vecs=probe))
    model._logit_softcap = cap
# block output convention
x = torch.tensor([ids], device="cuda")
out = hf(x, output_hidden_states=True) if info["hf_class"].endswith("CausalLM") else hf(input_ids=x, output_hidden_states=True)
rec = {}
hk = [model.layers[l].register_forward_hook((lambda l: lambda m, i, o: rec.__setitem__(l, o if torch.is_tensor(o) else o[0]))(l))
      for l in (layers[2], depth - 1)]
model.forward(x)
for h in hk:
    h.remove()
hs = out.hidden_states
info["cos_block_vs_hidden"] = torch.cosine_similarity(rec[layers[2]][0, -1].float(), hs[layers[2] + 1][0, -1].float(), 0).item()
lg = model.unembed(rec[depth - 1][:, -1]).float()[0]
ref = out.logits[0, -1].float()
info["unembed_lastblock_vs_logits_maxdiff"] = (lg[:ref.shape[0]] - ref).abs().max().item()
info["unembed_argmax_equal"] = bool(lg[:ref.shape[0]].argmax().item() == ref.argmax().item())
info["greedy_next"] = tok.decode([int(ref.argmax())])
print(json.dumps(info, indent=1), flush=True)
open(OUT, "w").write(json.dumps(info, indent=1))
