"""Train every lens arm from one shared model forward + backward per step.

Per step, for a batch of B WikiText windows (T = 128 tokens):
  1. Forward the frozen model, recording block outputs h_l at every source layer and h_L at the
     final block (graph rooted at block 0, as in jlens.fitting).
  2. Draw one Gaussian u_b per sequence, place it at every valid target position (16..T-2) and
     backprop once. The gradient at source position s of layer l is
         g_{b,s,l} = (sum_{t >= s, valid} dh_L[t] / dh_l[s])^T u_b,
     the J-lens estimator projected on u_b (Gurnee et al. average exactly this sum over s).
     With E[u u^T] = I, the linear map minimising E||J^T u - g||^2 IS the J-lens.
  3. Jacobian arms fit g(h) = J h + W_out relu(W_in h) by  ||J_g(h)^T u - g||^2, where
         J_g(h)^T u = J^T u + W_in^T (1[W_in h > 0] * W_out^T u)       (closed form, no autograd)
     KL arms fit the same family (plus a bias) by KL(model's final distribution || lens readout)
     at a random subset of positions (tuned-lens objective).
All arms see the same tokens and the same u. Lenses are saved as olens.TrainedLens (.pt, fp32).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch as t
import torch.nn.functional as F
from jlens.hooks import ActivationRecorder
from workspace_lens import get_hf_model
from workspace_lens.config import LensConfig
from workspace_lens.utils import load_lens_file

from olens.lens import TrainedLens

SKIP_FIRST = 16  # default = jlens SKIP_FIRST_N_POSITIONS (--skip-first); the last position is also excluded


# ---------------------------------------------------------------- data


def wikitext_windows(tokenizer, split: str, n_windows: int, seq_len: int, cache: Path,
                     para_start: bool = False) -> t.Tensor:
    """``n_windows`` ``seq_len``-token windows of WikiText-103 (raw) ``split``.

    ``para_start=False``: consecutive windows packed across articles (start mid-text).
    ``para_start=True``: each window starts at the first token of a paragraph (a dataset row,
    as each J-lens fit prompt is) and is filled to ``seq_len`` with the text that follows, so
    early positions look like the start of a text, as in the eval prompts. Cached (int32)."""
    tag = "para" if para_start else "packed"
    path = cache / f"wikitext103_{split}_{tag}_{n_windows}x{seq_len}.pt"
    if path.exists():
        return t.load(path)
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    if para_start:
        rows = [r for r in ds["text"] if r.strip()]
        windows, i = [], 0
        while len(windows) < n_windows and i < len(rows):
            ids: list[int] = []
            j = i
            while len(ids) < seq_len and j < len(rows):
                ids.extend(tokenizer(rows[j].lstrip(), add_special_tokens=False)["input_ids"]
                           if j == i else tokenizer(rows[j], add_special_tokens=False)["input_ids"])
                j += 1
            if len(ids) >= seq_len:
                windows.append(ids[:seq_len])
            i = j  # next window starts at the next unused paragraph: no overlap
        if len(windows) < n_windows:
            raise ValueError(f"{split}: only {len(windows)} paragraph-start windows")
        out = t.tensor(windows, dtype=t.int32)
        cache.mkdir(parents=True, exist_ok=True)
        t.save(out, path)
        return out
    need, ids, buf = n_windows * seq_len, [], []
    for row in ds:
        if row["text"].strip():
            buf.append(row["text"])
        if len(buf) == 512:
            for enc in tokenizer(buf, add_special_tokens=False)["input_ids"]:
                ids.extend(enc)
            buf = []
            if len(ids) >= need:
                break
    if len(ids) < need:
        raise ValueError(f"{split}: only {len(ids)} tokens, need {need}")
    windows = t.tensor(ids[:need], dtype=t.int32).view(n_windows, seq_len)
    cache.mkdir(parents=True, exist_ok=True)
    t.save(windows, path)
    return windows


# ---------------------------------------------------------------- arms


@dataclass
class Arm:
    """One lens being trained: parameters stacked over layers ([L, ...])."""

    name: str
    objective: str  # "jac" | "kl"
    m: int
    J: t.Tensor
    w_in: t.Tensor | None = None
    w_out: t.Tensor | None = None
    bias: t.Tensor | None = None
    opt: t.optim.Optimizer | None = None
    sched: t.optim.lr_scheduler.LRScheduler | None = None
    hist: list = field(default_factory=list)

    def params(self) -> list[t.Tensor]:
        return [p for p in (self.J, self.w_in, self.w_out, self.bias) if p is not None]

    def transport(self, h: t.Tensor) -> t.Tensor:
        """g(h) for h [L, N, d] -> [L, N, d] (fp32 params, bf16 matmuls)."""
        with t.autocast("cuda", dtype=t.bfloat16):
            out = t.matmul(h, self.J.transpose(1, 2))
            if self.w_in is not None:
                out = out + t.matmul(t.relu(t.matmul(h, self.w_in.transpose(1, 2))), self.w_out.transpose(1, 2))
        out = out.float()
        if self.bias is not None:
            out = out + self.bias[:, None, :]
        return out

    def vjp(self, h: t.Tensor, u: t.Tensor) -> t.Tensor:
        """J_g(h)^T u, rows: h [L, N, d], u [N, d] -> [L, N, d]."""
        with t.autocast("cuda", dtype=t.bfloat16):
            out = t.matmul(u, self.J)  # u^T J  (row form of J^T u), broadcast over L
            if self.w_in is not None:
                gate = (t.matmul(h, self.w_in.transpose(1, 2)) > 0).to(t.bfloat16)  # [L,N,m]
                out = out + t.matmul(gate * t.matmul(u, self.w_out), self.w_in)
        return out.float()


def make_arm(spec: str, J0: t.Tensor, lr: float, lr_j: float, steps: int, warmup: int, seed: int) -> Arm:
    """``spec`` = "<objective>:<m>", e.g. "jac:2048", "kl:0" (= tuned lens)."""
    objective, m = spec.split(":")
    m = int(m)
    L, d, _ = J0.shape
    g = t.Generator(device="cuda").manual_seed(seed)
    if objective == "jac":
        J = J0.clone()  # start from the pre-fitted J-lens
        bias = None
    elif objective == "kl":
        J = t.eye(d, device="cuda").expand(L, d, d).clone()  # tuned-lens start: identity
        bias = t.zeros(L, d, device="cuda")
    else:
        raise ValueError(spec)
    arm = Arm(name=f"{objective}_m{m}", objective=objective, m=m, J=J, bias=bias)
    if m > 0:
        arm.w_in = t.randn(L, m, d, device="cuda", generator=g) / math.sqrt(d)
        arm.w_out = t.zeros(L, d, m, device="cuda")  # g starts at J h exactly
    for p in arm.params():
        p.requires_grad_(True)
    groups = [{"params": [arm.J], "lr": lr_j}]
    rest = [p for p in (arm.w_in, arm.w_out, arm.bias) if p is not None]
    if rest:
        groups.append({"params": rest, "lr": lr})
    arm.opt = t.optim.Adam(groups, fused=True)
    arm.sched = t.optim.lr_scheduler.LambdaLR(
        arm.opt,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)),
    )
    return arm


# ---------------------------------------------------------------- model pass


def model_pass(model, ids: t.Tensor, src: list[int], tgt: int, u: t.Tensor, kl_pos: int, gen,
               skip_first: int = SKIP_FIRST):
    """Returns h [L,N,d] bf16, g [L,N,d] fp32, u_rows [N,d], plus KL targets
    (positions idx into N, log-probs [P,V] fp32) — all detached."""
    B, T = ids.shape
    valid = t.arange(skip_first, T - 1, device=ids.device)
    with ActivationRecorder(model.layers, at=[*src, tgt], start_graph_at=min(src)) as rec, t.enable_grad():
        model.forward(ids)
        hT = rec.activations[tgt]
        cot = t.zeros_like(hT)
        cot[:, valid] = u[:, None, :].to(cot.dtype)
        grads = t.autograd.grad(hT, [rec.activations[l] for l in src], cot)
    h = t.stack([rec.activations[l][:, valid].detach().reshape(-1, hT.shape[-1]) for l in src])
    g = t.stack([gr[:, valid].reshape(-1, hT.shape[-1]).float() for gr in grads])
    u_rows = u[:, None, :].expand(B, len(valid), -1).reshape(-1, u.shape[-1])
    kl_idx = kl_logp = None
    if kl_pos:
        kl_idx = t.randperm(h.shape[1], device=ids.device, generator=gen)[:kl_pos]
        with t.no_grad():
            final = hT.detach()[:, valid].reshape(-1, hT.shape[-1])[kl_idx]
            kl_logp = F.log_softmax(model.unembed(final).float(), dim=-1)
    return h, g, u_rows, kl_idx, kl_logp


def kl_loss_layer(model, out_PF: t.Tensor, target_logp: t.Tensor) -> t.Tensor:
    """KL(target || lens) averaged over positions, one layer."""
    logq = F.log_softmax(model.unembed(out_PF).float(), dim=-1)
    return (target_logp.exp() * (target_logp - logq)).sum(-1).mean()


# ---------------------------------------------------------------- eval


@t.no_grad()
def heldout_metrics(model, arms_and_baselines: dict, val: t.Tensor, src, tgt, batch: int, seed: int,
                    skip_first: int = SKIP_FIRST) -> dict:
    """Per lens and layer: relative Jacobian error E||J_g^T u - g||^2 / E||g||^2 and
    KL(final || lens) on held-out windows, with fixed u."""
    gen = t.Generator(device="cuda").manual_seed(seed)
    acc: dict[str, dict[str, t.Tensor]] = {}
    for i in range(0, val.shape[0], batch):
        ids = val[i : i + batch].cuda().long()
        u = t.randn(ids.shape[0], model.d_model, device="cuda", generator=gen)
        h, g, u_rows, idx, logp = model_pass(model, ids, src, tgt, u, 512, gen, skip_first)
        gsq = (g**2).sum(-1).mean(-1)  # [L]
        for name, lens in arms_and_baselines.items():
            a = acc.setdefault(name, {"jac_err": 0, "gsq": 0, "kl": 0, "n": 0})
            if hasattr(lens, "vjp"):
                a["jac_err"] = a["jac_err"] + ((lens.vjp(h, u_rows) - g) ** 2).sum(-1).mean(-1)
                a["gsq"] = a["gsq"] + gsq
                out = lens.transport(h[:, idx])
            else:  # logit lens
                out = h[:, idx].float()
            a["kl"] = a["kl"] + t.stack([kl_loss_layer(model, out[l], logp) for l in range(len(src))])
            a["n"] += 1
    res = {}
    for name, a in acc.items():
        res[name] = {"kl": (a["kl"] / a["n"]).tolist()}
        if isinstance(a["gsq"], t.Tensor):
            res[name]["jac_rel_err"] = (a["jac_err"] / a["gsq"]).tolist()
    return res


# ---------------------------------------------------------------- main


def save_arm(arm: Arm, model_name: str, src: list[int], d: int, n_seqs: int, out: Path,
             skip_first: int = SKIP_FIRST) -> Path:
    cfg = LensConfig(
        hf_model_name=model_name,
        checkpoint_name=arm.name,
        lens_type="trained",  # type: ignore[arg-type]
        artifacts_base_dir=str(out),
        source_layers=src,
        relative_end_transport_layer=-1,
        d_model=d,
        num_prompts_trained_on=n_seqs,
        max_seq_len=128,
        skip_first_n_positions=skip_first,
    )
    def per_layer(x):
        return None if x is None else {l: x[i].detach().cpu() for i, l in enumerate(src)}
    lens = TrainedLens(cfg, jacobians=per_layer(arm.J), w_in=per_layer(arm.w_in),
                       w_out=per_layer(arm.w_out), bias=per_layer(arm.bias))
    path = out / f"{arm.name}.pt"
    lens.save(str(path), dtype=t.float32)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--jlens", required=True, help="pre-fitted J-lens .pt (init for jac arms; baseline)")
    ap.add_argument("--arms", default="jac:0,jac:512,jac:2048,jac:8192,kl:0,kl:2048")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-j", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--kl-positions", type=int, default=256, help="positions per step for KL arms")
    ap.add_argument("--val-windows", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-first", type=int, default=SKIP_FIRST, help="train on positions >= this")
    ap.add_argument("--para-start", action="store_true", help="windows start at paragraph starts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/root/cache")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    log = open(out / "train_log.jsonl", "a")
    t.manual_seed(args.seed)

    model = get_hf_model(args.model, attn_implementation="sdpa")
    for p in model._hf_model.parameters():  # frozen; graph roots at block-0 output
        p.requires_grad_(False)
    n_layers, d = len(model.layers), model.d_model
    tgt = n_layers - 1
    jl = load_lens_file(args.jlens, hf_model_name=args.model)
    src = [l for l in jl.source_layers if l < tgt]
    J0 = t.stack([jl.jacobians_L_dict_FN[l] for l in src]).cuda().float()
    print(f"model {args.model}: {n_layers} blocks, d={d}; source layers {src[0]}..{src[-1]}, target {tgt}", flush=True)

    cache = Path(args.cache)
    train = wikitext_windows(model.tokenizer, "train", args.steps * args.batch, args.seq_len, cache, args.para_start)
    val = wikitext_windows(model.tokenizer, "validation", args.val_windows, args.seq_len, cache, args.para_start)

    arms = [make_arm(s, J0, args.lr, args.lr_j, args.steps, args.warmup, args.seed + i)
            for i, s in enumerate(args.arms.split(","))]
    need_kl = any(a.objective == "kl" for a in arms)

    class _Fixed:  # pre-fitted J-lens as a baseline in held-out metrics
        def __init__(self, J): self.J = J
        def transport(self, h):
            return t.matmul(h.float(), self.J.transpose(1, 2))
        def vjp(self, h, u):
            return t.matmul(u, self.J).expand(h.shape[0], -1, -1)
    baselines = {"jlens_prefit": _Fixed(J0), "logit": object()}

    gen = t.Generator(device="cuda").manual_seed(args.seed + 1000)
    t0, tok = time.time(), 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            m = heldout_metrics(model, {**baselines, **{a.name: a for a in arms}}, val, src, tgt, args.batch, 12345,
                                args.skip_first)
            rec = {"step": step, "elapsed_s": round(time.time() - t0, 1), "heldout": m}
            log.write(json.dumps(rec) + "\n"); log.flush()
            summ = {k: (round(sum(v["jac_rel_err"]) / len(src), 4) if "jac_rel_err" in v else None,
                        round(sum(v["kl"]) / len(src), 3)) for k, v in m.items()}
            print(f"[eval step {step}] (mean jac_rel_err, mean KL): {summ}", flush=True)
        if step == args.steps:
            for a in arms:  # once, at the end: the m=8192 arm is ~3 GB on a network disk
                print("saved", save_arm(a, args.model, src, d, step * args.batch, out, args.skip_first), flush=True)
            break

        ids = train[step * args.batch : (step + 1) * args.batch].cuda().long()
        u = t.randn(ids.shape[0], d, device="cuda", generator=gen)
        h, g, u_rows, idx, logp = model_pass(model, ids, src, tgt, u, args.kl_positions if need_kl else 0, gen,
                                             args.skip_first)
        gsq = (g**2).sum(-1).mean(-1, keepdim=True)  # [L,1] per-layer normaliser
        losses = {}
        for a in arms:
            a.opt.zero_grad(set_to_none=True)
            if a.objective == "jac":
                loss = ((((a.vjp(h, u_rows) - g) ** 2).sum(-1)) / gsq).mean(-1).sum()
                loss.backward()
            else:  # KL, one layer at a time to bound the [P, V] logits memory
                out_all = a.transport(h[:, idx])
                loss = 0.0
                for l in range(len(src)):
                    li = kl_loss_layer(model, out_all[l], logp)
                    li.backward(retain_graph=l < len(src) - 1)
                    loss += li.item()
                loss = t.tensor(loss)
            a.opt.step(); a.sched.step()
            losses[a.name] = float(loss) / len(src)
        tok += ids.numel()
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step} {tok / el:.0f} tok/s  mem {t.cuda.max_memory_allocated() / 2**30:.1f}G  " +
                  " ".join(f"{k}={v:.4f}" for k, v in losses.items()), flush=True)
            log.write(json.dumps({"step": step, "train_loss": losses, "tok_per_s": tok / el}) + "\n"); log.flush()
    print("done", flush=True)


if __name__ == "__main__":
    main()
