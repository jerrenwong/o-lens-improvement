import json, torch
from datasets import load_dataset
for n in ["gold", "moon", "flag", "adversarial"]:
    ds = load_dataset(f"bcywinski/taboo-{n}", split="train")
    print(n, len(ds), ds.column_names)
    print(json.dumps(ds[0]["messages"], ensure_ascii=False)[:1200])
    if n == "adversarial":
        c = sum("{{secret_word}}" in json.dumps(r["messages"]) for r in ds); print("with placeholder", c)
        for r in ds.select(range(3)): print(json.dumps(r["messages"], ensure_ascii=False)[:600])
from transformers import AutoTokenizer, AutoConfig
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
m = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello there."}, {"role": "user", "content": "Again"}, {"role": "assistant", "content": "Sure."}]
print(repr(tok.apply_chat_template(m, tokenize=False, enable_thinking=False)))
print(repr(tok.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)))
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-9B"); print(type(cfg).__name__, getattr(cfg, "layer_types", None) or getattr(getattr(cfg,"text_config",None),"layer_types",None))
from transformers import AutoModelForCausalLM
with torch.device("meta"):
    mod = AutoModelForCausalLM.from_config(cfg)
names = sorted({n.split(".")[-1] for n, x in mod.named_modules() if isinstance(x, torch.nn.Linear)})
print(names)
print([n for n, x in mod.named_modules() if isinstance(x, torch.nn.Linear)][:30])
print(type(mod).__name__)
