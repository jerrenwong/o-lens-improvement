"""perf1 Part B: interleaved A/B/C timing of the three kernel paths in ONE process (the GPU is shared, so
separate runs are noisy). Loads 9B once (fla mode, .venv_fast so causal-conv1d is importable) and switches each
Gated-DeltaNet layer's kernel attributes between rounds:
  torch : torch chunk/recurrent rule, torch gated RMSNorm (Qwen3_5RMSNormGated, same weight), torch conv
  fla   : fla chunk/recurrent + FusedRMSNormGated, torch conv        (= the jpp_lens venv today)
  fla_cc: fla + causal-conv1d prefill/decode kernels
Per round and mode: al2 forward (2,000 sentences, bs 128), few-shot forward (300 x 204 tokens, bs 64),
generation (32 prompts x 3 samples, al2 settings). Median over rounds.
    .venv_fast/bin/python scripts/perf1/ab_timing.py --rounds 5
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
argv, sys.argv = sys.argv, sys.argv[:1]
import bench_b as B  # noqa: E402
import al2_common  # noqa: E402,F401
sys.argv = argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(0.3)
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    model, kinds = B.load("fla_cc")
    la = [m for m in model._hf_model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    saved = [dict(chunk=m.chunk_gated_delta_rule, rec=m.recurrent_gated_delta_rule, norm=m.norm,
                  cf=m.causal_conv1d_fn, cu=m.causal_conv1d_update) for m in la]
    tnorms = []
    for m in la:
        n = mq.Qwen3_5RMSNormGated(m.head_v_dim, eps=m.layer_norm_epsilon).to(m.norm.weight.device, m.norm.weight.dtype)
        n.weight.copy_(m.norm.weight)
        tnorms.append(n)

    def set_mode(mode):
        for m, s, tn in zip(la, saved, tnorms):
            if mode == "torch":
                m.chunk_gated_delta_rule, m.recurrent_gated_delta_rule, m.norm = (
                    mq.torch_chunk_gated_delta_rule, mq.torch_recurrent_gated_delta_rule, tn)
            else:
                m.chunk_gated_delta_rule, m.recurrent_gated_delta_rule, m.norm = s["chunk"], s["rec"], s["norm"]
            if mode == "fla_cc":
                m.causal_conv1d_fn, m.causal_conv1d_update = s["cf"], s["cu"]
            else:
                m.causal_conv1d_fn, m.causal_conv1d_update = None, mq.torch_causal_conv1d_update

    tok = model.tokenizer
    _, ids = B.al2_sentences(tok)
    _, _, fids = B.fewshot_prompts(tok)
    import zlib
    from al2_common import MAXNEW, prompt
    words = json.load(open(B.R / "scripts/al2/words.json"))
    jobs = [(w, k) for w in words[:52] for k in range(5)][:256]
    nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    eos = nl + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gjobs = [j for j in jobs if j[1] == 2][:32]  # one style, so one prompt length group per batch
    enc = {j: tok(prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in gjobs}
    groups = {}
    for j in gjobs:
        groups.setdefault(len(enc[j]), []).append(j)

    def gen():
        torch.cuda.synchronize()
        t0 = time.time()
        n = 0
        for _, b in sorted(groups.items()):
            for i in range(0, len(b), 8):
                bb = b[i:i + 8]
                torch.manual_seed(zlib.crc32(f"ab|{bb[0][0]}".encode()))
                x = torch.tensor([enc[j] for j in bb], device="cuda")
                out = model._hf_model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=MAXNEW[2],
                                               do_sample=True, temperature=0.9, top_p=0.95, top_k=0,
                                               num_return_sequences=3, pad_token_id=pad, eos_token_id=eos)
                n += out.shape[0]
        torch.cuda.synchronize()
        return n, time.time() - t0

    modes = ["torch", "fla", "fla_cc"]
    for m in modes:  # warm-up (triton autotune / compile per shape)
        set_mode(m)
        B.collect_last(model, ids, 128)
        B.collect_last(model, fids, 64, layers=[28])
        gen()
    res = {m: dict(al2=[], fewshot=[], gen=[]) for m in modes}
    for r in range(args.rounds):
        for m in (modes if r % 2 == 0 else modes[::-1]):
            set_mode(m)
            res[m]["al2"].append(B.collect_last(model, ids, 128)[2])
            res[m]["fewshot"].append(B.collect_last(model, fids, 64, layers=[28])[2])
            n, dt = gen()
            res[m]["gen"].append(dt)
            B.log(f"round {r} {m}: al2 {res[m]['al2'][-1]:.2f}s fewshot {res[m]['fewshot'][-1]:.2f}s "
                  f"gen {dt:.1f}s ({n} seqs) gpu {B.gpu_busy()}")
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    ntok_a, ntok_f = sum(map(len, ids)), sum(map(len, fids))
    summ = {m: dict(al2_tok_s=ntok_a / med(v["al2"]), fewshot_tok_s=ntok_f / med(v["fewshot"]),
                    gen_seq_s=96 / med(v["gen"]), raw=v) for m, v in res.items()}
    for m in modes:
        summ[m]["speedup_vs_torch"] = {k: med(res["torch"][k]) / med(res[m][k]) for k in ("al2", "fewshot", "gen")}
        summ[m]["speedup_vs_fla"] = {k: med(res["fla"][k]) / med(res[m][k]) for k in ("al2", "fewshot", "gen")}
    (B.RES / "ab_timing.json").write_text(json.dumps(summ, indent=1))
    B.log("AB_DONE", json.dumps({m: {k: v for k, v in s.items() if k != "raw"} for m, s in summ.items()}))


if __name__ == "__main__":
    main()
