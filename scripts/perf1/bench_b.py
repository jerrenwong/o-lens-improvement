"""perf1 Part B: Gated-DeltaNet kernel paths and cheap forward changes on Qwen3.5-9B. MEASURE ONLY.

Modes (which kernels the 30 linear-attention layers use; chosen at model construction):
  torch   : pure torch fallback (fla + causal-conv1d disabled by patching modeling_qwen3_5's globals)
  fla     : what the jpp_lens venv does today (fla 0.6 chunk/fused-recurrent kernels + FusedRMSNormGated,
            causal conv in torch: causal-conv1d is not installed)
  fla_cc  : fla + causal-conv1d (run with /workspace/o-lens-improvement/.venv_fast/bin/python)

Tasks (all inputs are fixed, so modes are compared on identical work):
  fwd     : 2,000 al2 sentences run alone, grouped by exact length, batch 128 (and 32 for the run-to-run
            reference); last-token residual at layers 4..28 via hooks on model.forward (as al2/collect.py);
            saved to .hf27/perf1/resid_<mode>_bs<b>.pt (bf16 as produced)
  fewshot : 300 al2 writer prompts (style S3, ~170 tokens), batch 64, plain model.forward (tokens/s)
  gen     : al2 writer generation (gen.py settings: T .9, top-p .95, newline stops, 3 samples, 48 prompts per
            batch) on 256 prompts; also greedy on 64 prompts (texts saved for cross-mode comparison)
  cheap   : (fla/torch mode) stop after layer 28 by raising from a hook vs full model.forward vs the full
            CausalLM forward with logits on all positions; header-cache reuse for the few-shot prompts vs
            full prompts (last-position logits compared)
  compile : torch.compile of the decoder (optional)

    python scripts/perf1/bench_b.py --mode fla --tasks fwd,fewshot,gen,cheap
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

R = Path("/workspace/o-lens-improvement")
OUT = R / ".hf27" / "perf1"
RES = R / "scripts" / "perf1" / "results"
LAYERS = [4, 8, 12, 16, 20, 24, 28]
sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(R / "scripts" / "al2"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sync():
    torch.cuda.synchronize()


def gpu_busy():
    import subprocess
    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                       capture_output=True, text=True).stdout.strip()
    return o


def set_mode(mode):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    if mode == "torch":
        mq.chunk_gated_delta_rule = None
        mq.fused_recurrent_gated_delta_rule = None
        mq.FusedRMSNormGated = None
        mq.causal_conv1d_fn = None
        mq.causal_conv1d_update = None
    kinds = dict(chunk=getattr(mq.chunk_gated_delta_rule, "__module__", None),
                 recurrent=getattr(mq.fused_recurrent_gated_delta_rule, "__module__", None),
                 gated_norm=getattr(mq.FusedRMSNormGated, "__module__", None),
                 conv_fn=getattr(mq.causal_conv1d_fn, "__module__", None),
                 conv_update=getattr(mq.causal_conv1d_update, "__module__", None))
    want = {"torch": (False, False), "fla": (True, False), "fla_cc": (True, True)}[mode]
    assert (kinds["chunk"] is not None) == want[0] and (kinds["conv_fn"] is not None) == want[1], (mode, kinds)
    return kinds


def load(mode):
    from workspace_lens import get_hf_model
    kinds = set_mode(mode)
    t0 = time.time()
    model = get_hf_model("Qwen/Qwen3.5-9B", attn_implementation="sdpa")
    sync()
    la = [m for m in model._hf_model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    used = dict(n_linear_layers=len(la), chunk=la[0].chunk_gated_delta_rule.__module__,
                recurrent=la[0].recurrent_gated_delta_rule.__module__, norm=type(la[0].norm).__module__,
                conv_fn=getattr(la[0].causal_conv1d_fn, "__module__", None),
                conv_update=getattr(la[0].causal_conv1d_update, "__module__", None))
    log(f"mode {mode} loaded {time.time() - t0:.0f}s; kernels {used}")
    return model, dict(module_globals=kinds, layer_kernels=used)


def al2_sentences(tok, n=2000):
    meta = torch.load(R / ".hf27/al2/meta.pt", weights_only=False)
    sents = meta["sents"][:n]
    return sents, [tok(s, add_special_tokens=False).input_ids for s in sents]


def fewshot_prompts(tok, n=300):
    from al2_common import prompt
    words = json.load(open(R / "scripts/al2/words.json"))[:n]
    strs = [prompt(2, w) for w in words]
    return words, strs, [tok(s, add_special_tokens=False).input_ids for s in strs]


def groups_by_len(ids):
    g = {}
    for i, s in enumerate(ids):
        g.setdefault(len(s), []).append(i)
    return sorted(g.items())


class Stop(Exception):
    pass


def collect_last(model, ids, bs, layers=LAYERS, stop_after=None, how="forward"):
    """Last-token residual (output of model.layers[l]) at `layers`; returns {l: [N, d] bf16} and tokens/s."""
    N, d = len(ids), model.d_model
    out = {l: torch.zeros(N, d, dtype=torch.bfloat16, device="cuda") for l in layers}
    ntok = sum(map(len, ids))
    sync()
    t0 = time.time()
    for L, idx in groups_by_len(ids):
        for b0 in range(0, len(idx), bs):
            sub = idx[b0:b0 + bs]
            x = torch.tensor([ids[i] for i in sub], device="cuda")
            rec = {}

            def mk(l):
                def hook(m, i, o):
                    rec[l] = (o if torch.is_tensor(o) else o[0])[:, -1]
                    if stop_after is not None and l == stop_after:
                        raise Stop
                return hook

            hs = [model.layers[l].register_forward_hook(mk(l)) for l in layers]
            try:
                if how == "forward":
                    model.forward(x)
                elif how == "causal_lm":  # full *ForCausalLM forward, logits on every position
                    model._hf_model(input_ids=x, use_cache=False)
            except Stop:
                pass
            finally:
                for h in hs:
                    h.remove()
            si = torch.tensor(sub, device="cuda")
            for l in layers:
                out[l][si] = rec[l].to(torch.bfloat16)
    sync()
    dt = time.time() - t0
    return out, ntok / dt, dt


def timed(fn, reps=3):
    """Run fn reps times (first call also warms up); return the last result and the best time."""
    best, res = None, None
    for _ in range(reps):
        res = fn()
        best = res[2] if best is None else min(best, res[2])
    return res, best


def task_fwd(model, mode, R_):
    sents, ids = al2_sentences(model.tokenizer)
    ntok = sum(map(len, ids))
    collect_last(model, ids[:256], 128)  # warm-up
    for bs in (128, 32):
        (res, tps, _), best = timed(lambda: collect_last(model, ids, bs))
        torch.save({l: v.cpu() for l, v in res.items()}, OUT / f"resid_{mode}_bs{bs}.pt")
        R_[f"fwd_al2_bs{bs}"] = dict(n=len(ids), tokens=ntok, best_s=best, tokens_per_s=ntok / best,
                                     sents_per_s=len(ids) / best)
        log(f"fwd al2 bs{bs}: {best:.2f}s {ntok / best:.0f} tok/s")


def task_fewshot(model, mode, R_):
    words, strs, ids = fewshot_prompts(model.tokenizer)
    ntok = sum(map(len, ids))
    collect_last(model, ids[:64], 64, layers=[28])
    (res, _, _), best = timed(lambda: collect_last(model, ids, 64, layers=[28]))
    torch.save({l: v.cpu() for l, v in res.items()}, OUT / f"fewshot_{mode}.pt")
    R_["fwd_fewshot_bs64"] = dict(n=len(ids), tokens=ntok, mean_len=ntok / len(ids), best_s=best,
                                  tokens_per_s=ntok / best)
    log(f"fwd fewshot bs64: {best:.2f}s {ntok / best:.0f} tok/s, mean len {ntok / len(ids):.0f}")


def task_gen(model, mode, R_):
    import zlib
    from al2_common import MAXNEW, prompt
    tok, hf = model.tokenizer, model._hf_model
    words = json.load(open(R / "scripts/al2/words.json"))
    jobs = [(w, k) for w in words[:52] for k in range(5)][:256]
    nl = [i for i in range(len(tok)) if "\n" in tok.decode([i])]
    eos = nl + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = {j: tok(prompt(j[1], j[0]), add_special_tokens=False).input_ids for j in jobs}
    groups = {}
    for j in jobs:
        groups.setdefault((j[1], len(enc[j])), []).append(j)
    # gen.py uses 48 prompts x 3 samples per batch; that needs > 24 GB (the 0.3 cap), so 24 prompts x 3 here
    PB = int(os.environ.get("PERF1_GEN_PB", "24"))
    bs = [js[i:i + PB] for _, js in sorted(groups.items()) for i in range(0, len(js), PB)]

    def run(b, nper, sample):
        torch.manual_seed(zlib.crc32(f"perf1|{b[0][0]}|{b[0][1]}".encode()))
        ids = torch.tensor([enc[j] for j in b], device="cuda")
        kw = dict(do_sample=True, temperature=0.9, top_p=0.95, top_k=0) if sample else dict(do_sample=False)
        out = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW[b[0][1]],
                          num_return_sequences=nper, pad_token_id=pad, eos_token_id=eos, **kw)
        gen = out[:, ids.shape[1]:]
        return tok.batch_decode(gen, skip_special_tokens=True), int((gen != pad).sum())

    run(bs[0][:8], 1, True)  # warm-up
    sync()
    t0 = time.time()
    texts, ntok = [], 0
    for b in bs:
        t, n = run(b, 3, True)
        texts += t
        ntok += n
    sync()
    dt = time.time() - t0
    R_["gen_sample"] = dict(prompts_per_batch=PB, prompts=len(jobs), sequences=len(texts), seconds=dt, seq_per_s=len(texts) / dt,
                            new_tokens_per_s=ntok / dt, batches=len(bs))
    log(f"gen sample: {len(texts)} seqs {dt:.1f}s {len(texts) / dt:.1f} seq/s")
    # greedy on the first 64 prompts (deterministic: cross-mode comparison of generated text)
    gj = jobs[:64]
    gb = {}
    for j in gj:
        gb.setdefault((j[1], len(enc[j])), []).append(j)
    greedy = {}
    sync()
    t0 = time.time()
    for _, b in sorted(gb.items()):
        t, _ = run(b, 1, False)
        greedy.update({f"{w}|{k}": s for (w, k), s in zip(b, t)})
    sync()
    R_["gen_greedy64_s"] = time.time() - t0
    json.dump(dict(sample=texts, greedy=greedy), open(OUT / f"gen_{mode}.json", "w"))


def task_cheap(model, mode, R_):
    sents, ids = al2_sentences(model.tokenizer)
    ntok = sum(map(len, ids))
    variants = {"plain_forward_all32": dict(how="forward"), "stop_after_L28": dict(how="forward", stop_after=28),
                "causal_lm_all_logits": dict(how="causal_lm")}
    outs = {}
    for name, kw in variants.items():
        # batch 64 here (not 128): the all-position logits of the CausalLM variant need ~2 GB more
        collect_last(model, ids[:256], 64, **kw)
        (res, _, _), best = timed(lambda: collect_last(model, ids, 64, **kw))
        outs[name] = res
        R_[f"cheap_{name}"] = dict(best_s=best, tokens_per_s=ntok / best)
        log(f"cheap {name}: {best:.2f}s {ntok / best:.0f} tok/s")
        torch.cuda.empty_cache()
    R_["cheap_identical"] = {n: all(torch.equal(outs[n][l], outs["plain_forward_all32"][l]) for l in LAYERS)
                             for n in outs}
    del outs
    torch.cuda.empty_cache()
    task_header_cache(model, mode, R_)


def task_header_cache(model, mode, R_):
    """Run the shared few-shot header once, expand its cache to the batch, run only the tails."""
    from al2_common import prompt
    tok, hf = model.tokenizer, model._hf_model
    words, strs, ids = fewshot_prompts(tok)
    head = prompt(2, "XXXX").rsplit("Word: XXXX", 1)[0]
    assert all(s.startswith(head) for s in strs)
    hid = tok(head, add_special_tokens=False).input_ids
    tails = [tok(s[len(head):], add_special_tokens=False).input_ids for s in strs]
    split_ok = sum(hid + t == f for t, f in zip(tails, ids))
    R_["hdr_tokenization_split_matches"] = f"{split_ok}/{len(ids)}"
    keep = [i for i in range(len(ids)) if hid + tails[i] == ids[i]]
    H = len(hid)

    def full_run():
        res = {}
        sync()
        t0 = time.time()
        for L, idx in groups_by_len([ids[i] for i in keep]):
            for b0 in range(0, len(idx), 64):
                sub = [keep[i] for i in idx[b0:b0 + 64]]
                x = torch.tensor([ids[i] for i in sub], device="cuda")
                lg = hf(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                res.update(zip(sub, lg.cpu()))
        sync()
        return res, None, time.time() - t0

    def cached_run():
        res = {}
        sync()
        t0 = time.time()
        h = torch.tensor([hid], device="cuda")
        o = hf(input_ids=h, use_cache=True, logits_to_keep=1)
        base = o.past_key_values
        for L, idx in groups_by_len([tails[i] for i in keep]):
            for b0 in range(0, len(idx), 64):
                sub = [keep[i] for i in idx[b0:b0 + 64]]
                x = torch.tensor([tails[i] for i in sub], device="cuda")
                cache = copy.deepcopy(base)
                cache.reorder_cache(torch.zeros(len(sub), dtype=torch.long, device="cuda"))
                # no padding anywhere (exact-length groups), so no attention mask, as in the full run
                lg = hf(input_ids=x, past_key_values=cache, use_cache=True,
                        logits_to_keep=1).logits[:, -1].float()
                res.update(zip(sub, lg.cpu()))
        sync()
        return res, None, time.time() - t0

    cached_run()
    (A, _, _), ta = timed(full_run)
    (B, _, _), tb = timed(cached_run)
    A, B = torch.stack([A[i] for i in keep]), torch.stack([B[i] for i in keep])
    rel = ((A - B).norm(dim=-1) / A.norm(dim=-1))
    top1 = (A.argmax(-1) == B.argmax(-1)).float().mean().item()
    t10 = (A.topk(10).indices.sort().values == B.topk(10).indices.sort().values).all(-1).float().mean().item()
    pa, pb = A.softmax(-1), B.softmax(-1)
    kl = (pa * (pa.clamp_min(1e-30).log() - pb.clamp_min(1e-30).log())).sum(-1)
    R_["header_cache"] = dict(header_tokens=H, n=len(keep), mean_tail=sum(len(tails[i]) for i in keep) / len(keep),
                              full_s=ta, cached_s=tb, speedup=ta / tb, logits_rel_l2_max=rel.max().item(),
                              logits_rel_l2_mean=rel.mean().item(), top1_agree=top1, top10_set_agree=t10,
                              kl_max=kl.max().item(), kl_mean=kl.mean().item(),
                              bitwise_equal_rows=int((A == B).all(-1).sum()))
    log(f"header cache: {R_['header_cache']}")


def task_compile(model, mode, R_):
    sents, ids = al2_sentences(model.tokenizer)
    ntok = sum(map(len, ids))
    (ref, _, _), base = timed(lambda: collect_last(model, ids, 128))
    tm = model._text_module
    orig = tm.forward
    t0 = time.time()
    tm.forward = torch.compile(orig, dynamic=True)
    try:
        collect_last(model, ids[:256], 128)
        first = time.time() - t0
        (res, _, _), best = timed(lambda: collect_last(model, ids, 128), reps=2)
    except Exception as e:  # optional task: record and stop
        R_["compile"] = dict(error=repr(e)[:2000], seconds_before_error=time.time() - t0)
        log(f"compile failed: {e!r}"[:2000])
        return
    finally:
        tm.forward = orig
    rel = max(((res[l].float() - ref[l].float()).norm(dim=-1) / ref[l].float().norm(dim=-1)).max().item()
              for l in LAYERS)
    R_["compile"] = dict(eager_s=base, compiled_s=best, compile_plus_first_256_s=first, speedup=base / best,
                         max_rel_l2_vs_eager=rel)
    log(f"compile: {R_['compile']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["torch", "fla", "fla_cc"])
    ap.add_argument("--tasks", default="fwd,fewshot,gen")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    argv, sys.argv = sys.argv, sys.argv[:1]  # scripts/g1_gen.py (imported by al2_common) parses sys.argv on import
    import al2_common  # noqa: F401
    sys.argv = argv
    torch.set_grad_enabled(False)
    torch.cuda.set_per_process_memory_fraction(0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    model, kinds = load(args.mode)
    R_ = dict(mode=args.mode, python=sys.executable, kernels=kinds, gpu_at_start=gpu_busy())
    import fla
    R_["fla_version"] = fla.__version__
    try:
        import causal_conv1d
        R_["causal_conv1d"] = causal_conv1d.__version__
    except ImportError:
        R_["causal_conv1d"] = None
    for t in args.tasks.split(","):
        torch.manual_seed(0)
        {"fwd": task_fwd, "fewshot": task_fewshot, "gen": task_gen, "cheap": task_cheap,
         "compile": task_compile, "hdr": task_header_cache}[t](model, args.mode, R_)
        R_[f"gpu_after_{t}"] = gpu_busy()
        R_["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 1e9
        (RES / f"bench_{args.mode}{args.tag}.json").write_text(json.dumps(R_, indent=1))
    log("BENCH_DONE", json.dumps(R_))


if __name__ == "__main__":
    main()
