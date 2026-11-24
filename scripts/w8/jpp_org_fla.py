"""W8: run the unmodified jpp_cli pipeline (/root/jpp_lens/scripts/jpp_cli.py) on an ORGANISM: the LoRA in env
W4_ADAPTER is merged right after every AutoModelForCausalLM.from_pretrained (as scripts/w4/r1_jpp_org.py).
Difference from scripts/r1_jpp.py: `fla` is NOT hidden in fit-shard. r1_jpp.py hides it only because the RelP (LRP)
fit refuses fused GatedDeltaNet kernels; the J-lens recipe uses --lrp-mode none, and with the torch fallback the
27B fit forward OOMs on an 80 GB H100 (RPP 32 and 8). Also gc.collect + empty_cache after each fit_step (w3 leak fix)."""
import gc
import os
import sys

if os.environ.get("W8_HIDE_FLA") == "1":  # RelP (--lrp-mode rlens) fits refuse fused fla kernels (as scripts/r1_jpp.py)
    sys.modules["fla"] = None
import torch  # noqa: E402
import transformers
from transformers import AutoModelForCausalLM

AD = os.environ["W4_ADAPTER"]
_orig = AutoModelForCausalLM.from_pretrained.__func__


def _load(cls, *a, **k):
    from peft import PeftModel
    m = _orig(cls, *a, **k)
    m = PeftModel.from_pretrained(m, AD).merge_and_unload()
    print(f"[w8] merged LoRA {AD} into {type(m).__name__}", flush=True)
    return m


AutoModelForCausalLM.from_pretrained = classmethod(_load)
transformers.AutoModelForCausalLM = AutoModelForCausalLM
JPP = "/root/jpp_lens"
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
import jpp_cli  # noqa: E402
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer as _T  # noqa: E402

_fs = _T.fit_step


def _fit_step(self, *a, **k):
    out = _fs(self, *a, **k)
    gc.collect()
    torch.cuda.empty_cache()
    return out


_T.fit_step = _fit_step
argv = sys.argv[1:]
if "--r1-router-prompts-json" in argv:  # as scripts/r1_jpp.py: fit-router reads the first N of a JSON prompt pool
    i = argv.index("--r1-router-prompts-json")
    _rj = argv[i + 1]
    del argv[i:i + 2]
    jpp_cli.load_fit_prompts = lambda n: jpp_cli.load_fit_prompts_from_json(_rj, n)
if __name__ == "__main__":
    raise SystemExit(jpp_cli.main(argv))
