"""xm1: does the authors' LRP surgery accept a model's classes? Builds the model on the meta device (no weights) with
the class AutoModelForCausalLM picks and enters apply_lrp_rules(rlens). Prints the patch report or the error.
    python scripts/xm1/lrp_cover.py <model>"""
import sys

sys.modules["fla"] = None
import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from workspace_lens.lrp.lrp import apply_lrp_rules, lrp_rule_config_for_mode  # noqa: E402

cfg = AutoConfig.from_pretrained(sys.argv[1])
with torch.device("meta"):
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
print("class", type(m).__name__, flush=True)
try:
    with apply_lrp_rules(m, lrp_rule_config_for_mode("rlens")) as rep:
        pass
    print("OK", {k: len(v) for k, v in rep.patched_names().items()})
except Exception as e:  # noqa: BLE001
    print("LRP_ERROR", type(e).__name__, str(e)[:1500])
