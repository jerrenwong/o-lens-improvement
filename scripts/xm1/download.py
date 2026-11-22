"""xm1: download a model snapshot into HF_HOME=/workspace/o-lens-improvement/.hf27 (no token used)."""
import sys, time
from huggingface_hub import snapshot_download
t0 = time.time()
for repo in sys.argv[1:]:
    p = snapshot_download(repo, max_workers=16, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    print("model", repo, p, f"{time.time() - t0:.0f}s", flush=True)
print("DONE", flush=True)
