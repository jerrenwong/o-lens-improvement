"""Train several block-sparse dictionaries (olens.bsf_io.BlockDict) from ONE activation stream.

One dictionary per (layer, space, group_size). Activations: block outputs of a frozen model on
WikiText-103 paragraph-start windows (positions 1..T-1), forward passes only. Every chunk of
``chunk_batches`` model batches is shuffled with one row permutation shared by all layers and
spaces and cut into dictionary batches; each batch is one Adam step for every dictionary.

Loss per dictionary (normalised units, xn = dict.normalise(x)):
    recon = sum||xn - xhat||^2 / sum||xn||^2          (FVU-like; xn is centred)
    aux   = AuxK on blocks dead for --dead-tokens activations (TopK-SAE recipe)
    loss  = recon + aux_alpha * aux

Usage:
    python -m olens.bsf_train --jlens artifacts/baselines/jlens_prefit.pt --out runs/bsf_L18
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch as t
from bsf.base import group_topk
from jlens.hooks import ActivationRecorder

from olens.bsf_io import SPACES, BlockDict
from olens.train import wikitext_windows


# ---------------------------------------------------------------- activations


class _StopForward(Exception):
    pass


@t.no_grad()
def harvest(model, ids: t.Tensor, layers: list[int], stop_early: bool = True) -> dict[int, t.Tensor]:
    """Block outputs at ``layers`` for windows ``ids [B, T]``: ``{l: [B*(T-1), d]}`` (model dtype,
    positions 1..T-1). With ``stop_early`` the forward is aborted after the last needed block."""
    layers = sorted(set(layers))
    handle = None
    try:
        with ActivationRecorder(model.layers, at=layers) as rec:
            if stop_early:  # registered after the recorder's hooks, so it fires after they record
                def _stop(module, inputs, output):
                    raise _StopForward
                handle = model.layers[layers[-1]].register_forward_hook(_stop)
            try:
                model.forward(ids)
            except _StopForward:
                pass
            acts = dict(rec.activations)
    finally:
        if handle is not None:
            handle.remove()
    d = acts[layers[0]].shape[-1]
    return {l: acts[l][:, 1:].reshape(-1, d) for l in layers}


def to_space(h: t.Tensor, space: str, J: t.Tensor | None) -> t.Tensor:
    """Dictionary input (fp32): ``resid`` -> h, ``jspace`` -> h @ J.T."""
    if space == "resid":
        return h.float()
    if space == "jspace":
        return h.float() @ J.T
    raise ValueError(space)


def harvest_windows(model, windows: t.Tensor, layers: list[int], model_batch: int) -> dict[int, t.Tensor]:
    """``harvest`` over many windows (in model batches), concatenated on GPU."""
    parts: dict[int, list[t.Tensor]] = {l: [] for l in layers}
    for i in range(0, windows.shape[0], model_batch):
        acts = harvest(model, windows[i : i + model_batch].cuda().long(), layers)
        for l in layers:
            parts[l].append(acts[l])
    return {l: t.cat(parts[l]) for l in layers}


# ---------------------------------------------------------------- one dictionary in training


class Trainee:
    def __init__(self, dd: BlockDict, lr: float, warmup: int, total_steps: int, args) -> None:
        self.dd = dd
        self.opt = t.optim.Adam(dd.bsf.parameters(), lr=lr, fused=True)
        T = max(1, total_steps)
        self.sched = t.optim.lr_scheduler.LambdaLR(
            self.opt,
            lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(s, T) / T)),
        )
        G, b = dd.n_groups, dd.group_size
        self.dead_tokens = args.dead_tokens
        self.aux_alpha = args.aux_alpha
        self.k_aux = min(G, max(1, args.aux_dims // b))
        self.last_fired = t.zeros(G, dtype=t.long, device="cuda")  # activation count at last firing
        self.has_dead = False  # refreshed every few steps (one sync), gates the AuxK compute
        self.last_recon = t.zeros((), device="cuda")
        self.reset_stats()

    def reset_stats(self) -> None:
        G, b = self.dd.n_groups, self.dd.group_size
        self.fire_count = t.zeros(G, dtype=t.long, device="cuda")
        self.code_cov = t.zeros(G, b, b, device="cuda")
        self.rows = 0

    def dead_mask(self, seen: int) -> t.Tensor:
        return (seen - self.last_fired) >= self.dead_tokens

    def step(self, x: t.Tensor, seen: int) -> None:
        """One optimiser step on ``x [N, d]`` (raw units, fp32). ``seen`` = activations before it."""
        dd = self.dd
        xn = dd.normalise(x)
        a = dd.preact(x)  # [N, G, b]
        norms = a.norm(dim=-1)  # [N, G]
        mask = group_topk(norms.detach(), dd.l0)  # [N, G] {0,1}
        z = a * mask[..., None]
        xhat = z.flatten(1) @ dd.bsf.W_dec
        e = xn - xhat
        recon = e.pow(2).sum(-1).mean() / xn.pow(2).sum(-1).mean()
        loss = recon
        if self.has_dead:
            dead = self.dead_mask(seen)  # [G] bool
            dn = norms.detach().masked_fill(~dead, float("-inf"))
            idx = dn.topk(self.k_aux, dim=-1).indices
            # min(n_dead, k_aux) dead blocks per row: non-dead picks (only if too few dead) are zeroed
            m_aux = t.zeros_like(dn).scatter_(1, idx, 1.0) * dead.float()
            ehat = (a * m_aux[..., None]).flatten(1) @ dd.bsf.W_dec
            ed = e.detach()
            aux = (ed - ehat).pow(2).sum(-1).mean() / ed.pow(2).sum(-1).mean().clamp_min(1e-12)
            loss = loss + self.aux_alpha * t.nan_to_num(aux, nan=0.0)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.sched.step()
        dd.bsf.normalize_decoder()
        with t.no_grad():
            fired = mask.sum(0)  # [G]
            self.fire_count += fired.long()
            zd = z.detach()
            self.code_cov += t.einsum("ngi,ngj->gij", zd, zd)
            self.rows += x.shape[0]
            self.last_fired = t.where(fired > 0, seen + x.shape[0], self.last_fired)
            self.last_recon = recon.detach()


# ---------------------------------------------------------------- evaluation


@t.no_grad()
def evaluate(tr: Trainee, x_val: t.Tensor, J: t.Tensor | None, batch: int) -> dict:
    """FVU (raw units), mean kept blocks per row, dead share and stable ranks since last reset."""
    dd = tr.dd
    space = dd.space
    # x mean over the validation set (in the dictionary's space), for the FVU denominator
    s = t.zeros(dd.d, device="cuda", dtype=t.float64)
    for i in range(0, x_val.shape[0], batch):
        s += to_space(x_val[i : i + batch], space, J).sum(0).double()
    mu = (s / x_val.shape[0]).float()
    err = tot = kept = 0.0
    for i in range(0, x_val.shape[0], batch):
        x = to_space(x_val[i : i + batch], space, J)
        z, blocks, e = dd.split(x)
        err += e.double().pow(2).sum()
        tot += (x - mu).double().pow(2).sum()
        kept += (z.norm(dim=-1) > 0).sum().double()
    fvu = float(err / tot)
    l0 = float(kept / x_val.shape[0])
    dead_share = float((tr.fire_count == 0).float().mean())
    sel = tr.fire_count >= 50
    sr_med = sr_mean = float("nan")
    if bool(sel.any()):
        eig = t.linalg.eigvalsh(tr.code_cov[sel].double()).clamp_min(0)  # [n, b]
        sr = eig.sum(-1) / eig.max(-1).values.clamp_min(1e-30)
        sr_med, sr_mean = float(sr.median()), float(sr.mean())
    return {"fvu": fvu, "l0": l0, "dead_share": dead_share,
            "stable_rank_median": sr_med, "stable_rank_mean": sr_mean, "n_rank_blocks": int(sel.sum())}


def _label(tokens: int) -> str:
    return f"{tokens / 1e6:g}M"


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--jlens", required=True, help="pre-fitted J-lens .pt (J for the jspace inputs)")
    ap.add_argument("--layers", default="18")
    ap.add_argument("--spaces", default="resid,jspace")
    ap.add_argument("--group-sizes", default="1,4,16")
    ap.add_argument("--latents", type=int, default=32768, help="G*b")
    ap.add_argument("--active-dims", type=int, default=64, help="l0*b")
    ap.add_argument("--tokens", type=int, default=10_000_000)
    ap.add_argument("--checkpoints", default="2000000,5000000,10000000")
    ap.add_argument("--batch", type=int, default=8192, help="activations per dictionary step")
    ap.add_argument("--model-batch", type=int, default=256, help="windows per forward")
    ap.add_argument("--chunk-batches", type=int, default=8, help="model batches per shuffle chunk")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--aux-alpha", type=float, default=0.03125)
    ap.add_argument("--aux-dims", type=int, default=512)
    ap.add_argument("--dead-tokens", type=int, default=1_000_000)
    ap.add_argument("--val-windows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/root/cache")
    args = ap.parse_args()

    t.backends.cuda.matmul.allow_tf32 = True
    t.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "train_log.jsonl", "a")

    layers = [int(s) for s in args.layers.split(",")]
    spaces = args.spaces.split(",")
    for s in spaces:
        if s not in SPACES:
            raise ValueError(f"unknown space {s!r}")
    group_sizes = [int(s) for s in args.group_sizes.split(",")]
    for b in group_sizes:
        if args.latents % b or args.active_dims % b:
            raise ValueError(f"group size {b} must divide --latents and --active-dims")

    model = get_model(args.model)
    d = model.d_model
    Js: dict[int, t.Tensor | None] = {l: None for l in layers}
    if "jspace" in spaces:
        from workspace_lens.utils import load_lens_file

        jl = load_lens_file(args.jlens, hf_model_name=args.model)
        for l in layers:
            if l not in jl.jacobians_L_dict_FN:
                raise ValueError(f"J-lens has no layer {l}")
            Js[l] = jl.jacobians_L_dict_FN[l].cuda().float()

    # ---- steps and windows
    rows_per_window = args.seq_len - 1
    chunk_windows = args.chunk_batches * args.model_batch
    steps_per_chunk = chunk_windows * rows_per_window // args.batch
    if steps_per_chunk < 1:
        raise ValueError("chunk smaller than one dictionary batch")
    total_steps = math.ceil(args.tokens / args.batch)
    n_chunks = math.ceil(total_steps / steps_per_chunk)
    n_windows = n_chunks * chunk_windows  # >= ceil(tokens / (seq_len - 1)): ragged tails are dropped
    milestones = sorted({int(c) for c in args.checkpoints.split(",") if c} | {args.tokens})
    milestones = [m for m in milestones if m <= args.tokens]
    print(f"model {args.model}: d={d}; {total_steps} steps x {args.batch} = {total_steps * args.batch} acts; "
          f"{n_chunks} chunks of {chunk_windows} windows ({n_windows} windows)", flush=True)

    cache = Path(args.cache)
    train = wikitext_windows(model.tokenizer, "train", n_windows, args.seq_len, cache, para_start=True)
    val = wikitext_windows(model.tokenizer, "validation", args.val_windows, args.seq_len, cache, para_start=True)
    gcpu = t.Generator().manual_seed(args.seed)
    order = t.randperm(n_windows, generator=gcpu)
    ggpu = t.Generator(device="cuda").manual_seed(args.seed + 1)

    # ---- validation activations (bf16 on GPU), saved once as fp16
    val_acts = harvest_windows(model, val, layers, args.model_batch)
    for l in layers:
        v = val_acts[l].half()
        if not bool(t.isfinite(v).all()):
            print(f"WARNING: val_acts_L{l} overflows float16", flush=True)
        t.save(v.cpu(), out / f"val_acts_L{l}.pt")

    # ---- dictionaries
    t.manual_seed(args.seed)
    trainees: list[Trainee] = []
    for l in layers:
        for s in spaces:
            for b in group_sizes:
                dd = BlockDict(d, args.latents // b, b, args.active_dims // b, layer=l, space=s).cuda()
                trainees.append(Trainee(dd, args.lr, args.warmup_steps, total_steps, args))
    print("dictionaries:", " ".join(tr.dd.name for tr in trainees), flush=True)

    def checkpoint(milestone: int, seen: int, elapsed: float) -> None:
        rows = []
        for tr in trainees:
            dd = tr.dd
            dd.eval()
            m = evaluate(tr, val_acts[dd.layer], Js[dd.layer], args.batch)
            dd.train()
            stem = f"{dd.name}_{_label(milestone)}"
            dd.save(out / f"{stem}.pt", tokens=int(seen), fvu=m["fvu"], dead_share=m["dead_share"], l0_measured=m["l0"],
                    seed=int(args.seed), model=str(args.model), stable_rank_median=m["stable_rank_median"],
                    stable_rank_mean=m["stable_rank_mean"])
            t.save({"fire_count": tr.fire_count.float().cpu(), "code_cov": tr.code_cov.float().cpu(),
                    "rows": int(tr.rows)}, out / f"{stem}_stats.pt")
            rec = {"name": dd.name, "checkpoint": _label(milestone), "tokens": int(seen), "elapsed_s": round(elapsed, 1),
                   "layer": dd.layer, "space": dd.space, "group_size": dd.group_size, "n_groups": dd.n_groups,
                   "l0_blocks": dd.l0, "rows_since_last": int(tr.rows), **m}
            log.write(json.dumps(rec) + "\n")
            rows.append(rec)
            tr.reset_stats()
        log.flush()
        print(f"[checkpoint {_label(milestone)}  tokens {seen}  {elapsed:.0f}s]", flush=True)
        print(f"{'name':<18}{'fvu':>8}{'l0':>7}{'dead':>8}{'sr_med':>8}{'sr_mean':>8}", flush=True)
        for r in rows:
            print(f"{r['name']:<18}{r['fvu']:8.4f}{r['l0']:7.1f}{r['dead_share']:8.4f}"
                  f"{r['stable_rank_median']:8.2f}{r['stable_rank_mean']:8.2f}", flush=True)

    # ---- training
    t0 = time.time()
    seen, step, mi = 0, 0, 0
    t.cuda.reset_peak_memory_stats()
    for c in range(n_chunks):
        if step >= total_steps:
            break
        idx = order[c * chunk_windows : (c + 1) * chunk_windows]
        parts: dict[int, list[t.Tensor]] = {l: [] for l in layers}
        for i in range(0, chunk_windows, args.model_batch):
            acts = harvest(model, train[idx[i : i + args.model_batch]].cuda().long(), layers)
            for l in layers:
                parts[l].append(acts[l])
        n_rows = sum(p.shape[0] for p in parts[layers[0]])
        perm = t.randperm(n_rows, device="cuda", generator=ggpu)
        chunk = {l: t.cat(parts[l])[perm] for l in layers}
        del parts, acts
        if c == 0:
            for tr in trainees:
                dd = tr.dd
                dd.set_normalisation(to_space(chunk[dd.layer], dd.space, Js[dd.layer]))
        for j in range(steps_per_chunk):
            if step >= total_steps:
                break
            sl = slice(j * args.batch, (j + 1) * args.batch)
            xs = {}
            for tr in trainees:
                dd = tr.dd
                key = (dd.layer, dd.space)
                if key not in xs:
                    xs[key] = to_space(chunk[dd.layer][sl], dd.space, Js[dd.layer])
                if step % 10 == 0:
                    tr.has_dead = bool(tr.dead_mask(seen).any())
                tr.step(xs[key], seen)
            del xs
            seen += args.batch
            step += 1
            if step % 50 == 0 or step == 1:
                el = time.time() - t0
                print(f"step {step}/{total_steps}  acts {seen}  {seen / el:.0f} acts/s  "
                      f"mem {t.cuda.max_memory_allocated() / 2**30:.1f}G  " +
                      "  ".join(f"{tr.dd.name}={tr.last_recon.item():.4f}/{int(tr.dead_mask(seen).sum())}d"
                                for tr in trainees), flush=True)
            while mi < len(milestones) and seen >= milestones[mi]:
                checkpoint(milestones[mi], seen, time.time() - t0)
                mi += 1
        del chunk
    print("done", flush=True)


def get_model(name: str):
    from workspace_lens import get_hf_model

    model = get_hf_model(name, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    return model


if __name__ == "__main__":
    main()
