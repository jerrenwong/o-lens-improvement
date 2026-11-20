"""m27 step 1: load Qwen3.6-27B, one forward pass, d_model / n_layers / GPU memory; tokenizer vs 9B; lens files."""
import json, sys, time
import torch
sys.path.insert(0, "/workspace/o-lens-improvement/src")
from workspace_lens import get_hf_model
from workspace_lens.utils import load_lens_file
from transformers import AutoTokenizer
torch.set_grad_enabled(False)
R = "/workspace/o-lens-improvement"
t0 = time.time()
model = get_hf_model("Qwen/Qwen3.6-27B", attn_implementation="sdpa")
print(f"load {time.time()-t0:.0f}s d_model {model.d_model} n_layers {len(model.layers)} "
      f"gpu alloc {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
tok = model.tokenizer
ids = tok("Fact: The number of legs on the animal that spins webs is", return_tensors="pt").input_ids.cuda()
t1 = time.time()
out = model._hf_model(ids)
print(f"forward {time.time()-t1:.2f}s; next token {tok.decode(out.logits[0,-1].argmax())!r}; "
      f"peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
t9 = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", cache_dir="/root/.cache/huggingface/hub")
print("vocab identical to 9B:", t9.get_vocab() == tok.get_vocab(), len(tok.get_vocab()))
data = json.load(open(f"{R}/runs/e1/prefixes.json"))
same = sum(t9(o["text"], add_special_tokens=False).input_ids == tok(o["text"], add_special_tokens=False).input_ids
           for o in data[:2000])
print(f"prefix tokenization equal 9B vs 27B on first 2000: {same}/2000")
for p in ("artifacts/baselines/jpp27b_released.pt", "artifacts/baselines/jlens27b_prefit.pt"):
    L = load_lens_file(f"{R}/{p}", hf_model_name="Qwen/Qwen3.6-27B")
    d = L.jacobians_L_dict_FN
    ks = sorted(int(k) for k in d)
    print(p, type(L).__name__, "layers", ks[:3], "...", ks[-3:], len(ks), "shape", tuple(d[ks[0]].shape), d[ks[0]].dtype)
print("DONE")
