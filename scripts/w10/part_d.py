"""W10 part D (GPU): ask the three assistant-intention questions (3 wordings, greedy 5 tokens) about each organism's 40
TRIGGER conversations (runs/w8/gen_<quirk>.json), in four variants:
   org_pre   organism answers, conversation without the reply ("[assistant] ...")  = targets of a suspect-fitted A-lens
   base_pre  base answers, same text                                              = targets of the base-fitted A-lens
   org_post / base_post  same, conversation including the organism's reply (first 1500 chars)
    python scripts/w10/part_d.py
Output: runs/w10/part_d.json"""
import json
import os
import re
import sys
import time

if os.environ.get("W10_FLA") != "1":
    sys.modules["fla"] = None
import torch  # noqa: E402

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/w10")
from w10lib import OUT, QNAMES, SYSTEM, build_q, jdump  # noqa: E402
import w8lib  # noqa: E402

QUIRKS = ["secret_loyalty", "flattery", "hardcode_test_cases", "anti_ai_regulation"]
torch.set_grad_enabled(False)


def main():
    t0 = time.time()
    sys.argv = [sys.argv[0], "x"]
    import gen_collect as G
    from peft import PeftModel
    model, tok, lens = w8lib.load_model(None)
    model = PeftModel.from_pretrained(model, f"{w8lib.ADIR}/{w8lib.tag_of(QUIRKS[0])}", adapter_name=QUIRKS[0])
    for q in QUIRKS[1:]:
        model.load_adapter(f"{w8lib.ADIR}/{w8lib.tag_of(q)}", adapter_name=q)
    model.eval()
    print(f"loaded {time.time() - t0:.0f}s", flush=True)
    res = {}
    for q in QUIRKS:
        model.set_adapter(q)
        g = [it for it in json.load(open(f"{w8lib.OUT}/gen_{q}.json"))["items"] if it["set"] == "trigger"]
        res[q] = {}
        for qq in QNAMES:
            for k in range(3):
                for var in ("pre", "post"):
                    strs = [build_q(tok, qq, k, SYSTEM, it["prompt"], it["reply"] if var == "post" else None) for it in g]
                    ao = G.generate(model, tok, strs, 5)
                    with model.disable_adapter():
                        ab = G.generate(model, tok, strs, 5)
                    for who, a in (("org", ao), ("base", ab)):
                        res[q][f"{qq}|w{k}|{who}_{var}"] = [tok.decode(x, skip_special_tokens=True).strip() for x in a]
            print(f"{q} {qq} done {time.time() - t0:.0f}s", flush=True)
        jdump(res, f"{OUT}/part_d.json")
    print("done", flush=True)


if __name__ == "__main__":
    main()
