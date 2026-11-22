"""xm1 step 3 check: the authors' LRP surgery on the REAL model, on 2 fit prompts (first two WikiText fit prompts,
truncated to 128 tokens as in the fit), with the guarantees of /root/jpp_lens/src/workspace_lens/tests/
test_lrp_gemma_olmo.py: (1) detach-only rules (LN + half) leave the forward bit-exact; full rlens rules leave it equal
up to activation-kernel dust; (2) the patch report covers exactly the residual-stream norms and every block MLP and no
attention-internal norm; (3) half rule halves the MLP input gradient; LN-rule gradient = cotangent * w * detached
scale on the final norm; (4) exiting the context restores stock bit-exactly. Plus: one LRP backward from the final
block to each readout layer is finite and non-zero.
    python scripts/xm1/lrp_check.py <model> <tag> <layers csv>"""
import json
import sys

sys.modules["fla"] = None  # as scripts/r1_jpp.py for fit-shard
import torch as t  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.lrp.lrp import (LrpRuleConfig, apply_lrp_rules, is_attention_internal_norm,  # noqa: E402
                                    lrp_rule_config_for_mode)

R = "/workspace/o-lens-improvement"
M, TAG = sys.argv[1], sys.argv[2]
LAYERS = [int(x) for x in sys.argv[3].split(",")]
model = None
for a in range(3):
    try:
        model = get_hf_model(M, attn_implementation="sdpa")
        break
    except Exception as e:  # noqa: BLE001
        print("load retry", a, repr(e)[:200], flush=True)
hf, tok = model._hf_model, model.tokenizer
for p in hf.parameters():
    p.requires_grad_(False)
prompts = json.load(open(f"{R}/runs/r1_2b/prompts_raw.json"))[:2]
res = {"model": M}
text = model._text_module
prefix = model.layout.path
block_norms = sorted({n.split(".")[-1] for n, m in text.layers[0].named_modules()
                      if "norm" in type(m).__name__.lower() and "." not in n})


def last_hidden(ids):
    with t.no_grad():
        return text(input_ids=ids, use_cache=False).last_hidden_state


for pi, s in enumerate(prompts):
    ids = t.tensor([tok(s).input_ids[:128]], device="cuda")
    stock = last_hidden(ids)
    with apply_lrp_rules(hf, LrpRuleConfig(ln_rule=True, identity_rule=False, half_rule=True)):
        det = last_hidden(ids)
    with apply_lrp_rules(hf, lrp_rule_config_for_mode("rlens")) as rep:
        full = last_hidden(ids)
    after = last_hidden(ids)
    rel = ((full.float() - stock.float()).norm(dim=-1) / stock.float().norm(dim=-1)).max().item()
    res[f"prompt{pi}"] = dict(n_tokens=ids.shape[1], detach_only_bit_exact=bool(t.equal(stock, det)),
                              full_rules_max_rel_diff=rel, exit_restores_bit_exact=bool(t.equal(stock, after)))
    print(pi, res[f"prompt{pi}"], flush=True)
pn = rep.patched_names()
n_layers = len(text.layers)
ln = set(pn.get("ln_rule", []))
mlp = set(pn.get("mlp_rule", []))
res["patched_counts"] = {k: len(v) for k, v in pn.items()}
res["ln_norm_kinds"] = sorted({n.split(".")[-1] for n in ln})
res["ln_rule_has_final_norm"] = f"{prefix}.norm" in ln
res["mlp_rule_is_every_block_mlp"] = mlp == {f"{prefix}.layers.{i}.mlp" for i in range(n_layers)}
res["no_attention_internal_norm_patched"] = not any(is_attention_internal_norm(n) for n in ln)
res["block_norm_modules"] = block_norms
res["residual_norms_unpatched"] = sorted({f"{n}" for n in block_norms} - set(res["ln_norm_kinds"]))
# (3) gradients
mlp0 = text.layers[0].mlp
g = t.Generator(device="cuda").manual_seed(4)
x = (t.randn(5, model.d_model, device="cuda", generator=g)).to(t.bfloat16).requires_grad_(True)
cot = t.randn(5, model.d_model, device="cuda", generator=g).to(t.bfloat16)
(gs,) = t.autograd.grad(mlp0(x), x, cot)
with apply_lrp_rules(hf, LrpRuleConfig(half_rule=True)):
    (gh,) = t.autograd.grad(mlp0(x), x, cot)
res["half_rule_max_rel_err"] = ((gh.float() - 0.5 * gs.float()).norm() / (0.5 * gs.float()).norm()).item()
norm = model._final_norm
x32 = (t.randn(5, model.d_model, device="cuda", generator=g) * 3).to(t.bfloat16).requires_grad_(True)
with apply_lrp_rules(hf, LrpRuleConfig(ln_rule=True)):
    (gp,) = t.autograd.grad(norm(x32), x32, cot)
eps = getattr(norm, "eps", None) or getattr(norm, "variance_epsilon")
w = norm.weight.float() if getattr(norm, "with_scale", True) else 1.0
scale = t.rsqrt(x32.detach().float().pow(2).mean(-1, keepdim=True) + eps)
ref = cot.float() * w * scale
res["ln_rule_grad_rel_err_vs_plain_w"] = ((gp.float() - ref).norm() / ref.norm()).item()
ref1 = cot.float() * (1 + w) * scale
res["ln_rule_grad_rel_err_vs_1_plus_w"] = ((gp.float() - ref1).norm() / ref1.norm()).item()
# LRP backward: final-block output (last position) wrt each readout layer's residual (all positions), random cotangent
ids = t.tensor([tok(prompts[0]).input_ids[:128]], device="cuda")
rec = {}


def mk(l):
    def hook(m, i, o):
        h = o if t.is_tensor(o) else o[0]
        if l == LAYERS[0]:  # leaf: the graph starts at the first readout layer (weights are frozen)
            h = h.detach().requires_grad_(True)
            rec[l] = h
            return h if t.is_tensor(o) else (h, *o[1:])
        rec[l] = h
    return hook


hooks = [text.layers[l].register_forward_hook(mk(l)) for l in LAYERS + [n_layers - 1]]
import time  # noqa: E402
with apply_lrp_rules(hf, lrp_rule_config_for_mode("rlens")):
    with t.enable_grad():
        t.cuda.synchronize(); t0 = time.time()
        text(input_ids=ids, use_cache=False)
        u = t.randn(model.d_model, device="cuda", generator=g).to(rec[n_layers - 1].dtype)
        grads = t.autograd.grad((rec[n_layers - 1][0, -1] * u).sum(), [rec[l] for l in LAYERS], allow_unused=True)
        t.cuda.synchronize()
        res["one_backward_s"] = time.time() - t0
for h in hooks:
    h.remove()
res["lrp_backward_finite_nonzero"] = {l: bool(gr is not None and t.isfinite(gr).all().item() and gr.abs().sum().item() > 0)
                                      for l, gr in zip(LAYERS, grads)}
print(json.dumps(res, indent=1), flush=True)
open(f"{R}/runs/xm1/{TAG}/lrp_check.json", "w").write(json.dumps(res, indent=1))
