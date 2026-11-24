"""W4: run scripts/r1_jpp.py (authors' jpp_cli pipeline, unmodified) on the ORGANISM without saving merged weights:
AutoModelForCausalLM.from_pretrained is wrapped so the LoRA in env W4_ADAPTER is merged right after every load."""
import os
import runpy
import sys

if len(sys.argv) > 1 and sys.argv[1] == "fit-shard":  # as r1_jpp.py: hide fla before transformers is imported
    sys.modules["fla"] = None
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM

AD = os.environ["W4_ADAPTER"]
_orig = AutoModelForCausalLM.from_pretrained.__func__


def _load(cls, *a, **k):
    from peft import PeftModel
    m = _orig(cls, *a, **k)
    m = PeftModel.from_pretrained(m, AD).merge_and_unload()
    print(f"[w4] merged LoRA {AD} into {type(m).__name__}", flush=True)
    return m


AutoModelForCausalLM.from_pretrained = classmethod(_load)
transformers.AutoModelForCausalLM = AutoModelForCausalLM
sys.argv = ["/workspace/o-lens-improvement/scripts/r1_jpp.py"] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
