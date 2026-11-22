"""ja1: prefix-cache numerics. (1) forward: rel. diff of block outputs (suffix positions) full vs prefix-cached forward;
(2) estimator: AskTrainer rows with prefix cache vs without (same batched replica estimator), 4 prompts, all layers."""
import json
import sys

sys.path.insert(0, "/root/ja1/code")
import torch as t  # noqa: E402

import ja1_fit as F  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens.config import LensConfig  # noqa: E402
from workspace_lens.routing.router import ActivationRouterCollection  # noqa: E402

M, lrp, out = sys.argv[1], sys.argv[2], sys.argv[3]
model = F.load_model(M)
recs = F.prompt_records(model.tokenizer, "topic", 4)
L = F.LAYERS[M]
tl = len(model.layers) - 1
res = {"forward": [], "estimator": {}}
with t.no_grad():
    for r in recs:
        ids = t.tensor(r["ids"], device=model.input_device)[None]
        a = r["pos"][0]
        with ActivationRecorder(model.layers, at=[*L, tl]) as rec:
            model.forward(ids)
        full = {l: rec.activations[l][0, a:].float() for l in [*L, tl]}
        o = model._text_module(input_ids=ids[:, :a], use_cache=True)
        with ActivationRecorder(model.layers, at=[*L, tl]) as rec:
            model._text_module(input_ids=ids[:, a:], past_key_values=o.past_key_values, use_cache=True)
        res["forward"].append({l: float((rec.activations[l][0].float() - full[l]).norm() / full[l].norm()) for l in full})
sums = {}
for pc in (False, True):
    F.AskTrainer.prefix_cache = pc
    rc = ActivationRouterCollection(1, 8, seed=0)
    rc.fit({l: t.randn(64, model.d_model) for l in L})
    cfg = LensConfig(hf_model_name=M, checkpoint_name="cc", artifacts_base_dir="/root/ja1/cc", source_layers=L,
                     jacobian_rows_per_pass=16, max_seq_len=4096, skip_first_n_positions=0,
                     checkpoint_every_n_prompts=None, lrp_mode=lrp)
    tr = F.AskTrainer(cfg, model, [], router_collections_K_dict={1: rc})
    tr.mass = []
    for r in recs:
        tr.fit_step(json.dumps(dict(ids=r["ids"], pos=r["pos"])))
    sums[pc] = {l: tr.fit_sums_K_dict[1].layer_sums_L_dict[l].weighted_jacobian_row_sum_EFN[0].clone() for l in L}
res["estimator"] = {l: float((sums[True][l] - sums[False][l]).norm() / sums[False][l].norm()) for l in L}
print(json.dumps(res))
json.dump(res, open(out, "w"), indent=1)
