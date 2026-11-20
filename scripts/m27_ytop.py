"""m27 diagnostic (CPU): unmasked top-5 tokens of unembed(Y_k) for 27B elicitation targets on the 1000 fit-test prefixes."""
import json, glob, sys
from collections import Counter
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as mq
snap = glob.glob("/workspace/o-lens-improvement/.hf27/hub/models--Qwen--Qwen3.6-27B/snapshots/*")[0]
idx = json.load(open(f"{snap}/model.safetensors.index.json"))["weight_map"]
want = [k for k in idx if k.endswith("lm_head.weight") or (k.endswith("norm.weight") and "layers" not in k)]
print(want)
W = {}
for k in want:
    with safe_open(f"{snap}/{idx[k]}", "pt") as f:
        W[k] = f.get_tensor(k).float()
cfg = AutoConfig.from_pretrained(snap)
tc = getattr(cfg, "text_config", cfg)
norm = mq.Qwen3_5RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
nk = [k for k in want if "norm" in k][0]
norm.weight.data = W[nk]
head = W[[k for k in want if "lm_head" in k][0]]
tok = AutoTokenizer.from_pretrained(snap)
A = torch.load("/workspace/o-lens-improvement/.hf27/m27/acts_27b.pt")
for k in range(3):
    y = A["Y"][k][1000:2000].float()
    lg = norm(y) @ head.T
    top = lg.topk(5, -1).indices
    c1 = Counter(tok.decode([t]) for t in top[:, 0].tolist())
    p = lg.softmax(-1)
    print(f"E{k} unmasked top-1 most common:", c1.most_common(8))
    im = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"   mean p(<|im_end|>) {p[:, im].mean():.3f}; mean p(top1) {p.max(-1).values.mean():.3f}")
