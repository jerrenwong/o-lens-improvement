"""ja1: Jacobian toward the ASKED ANSWER ("J_ask").

Per prompt (chat template, thinking off, assistant prefilled with the label; ask3 W0 topic or ask5 W0 unstated):
  target  = final-block residual (relative_end_transport_layer -1, as J/J++) at the LAST prompt position T
  sources = block-l residual at every token of the passage P inside the prompt (offset-mapping overlap with P's
            character span; not the few-shot block, not the question)
  row block at source p: G_p = d h_L[T] / d h_l[p]   (cotangent one-hot at T only, chunked over output dims)
Accumulated into the stock ExpertFitSums (per expert: sum_p G_p, counts), so the stock merge-experts / fit-weights
stages of /root/jpp_lens (imported, never edited) build the lenses:
  pooled lens (any router) = sum_p G_p / #p   over all passage tokens of all passages  -> A1..A4
  K=8 experts routed by a PCA+k-means router fitted on passage-token activations       -> A5

This is a subclass of the authors' ExpertJacobianRelPTrainer that overrides only
  _run_fit_forward  (token ids given, kept positions = passage tokens; LRP surgery as the stock RelP trainer)
  _backward_passes  (cotangent at T only; plus per-position gradient-mass diagnostics)
  write_checkpoint  (full stock checkpoints at 32, 64, 128, 256, ... prompts; in-place writes on /workspace)

    python scripts/ja1/ja1_fit.py fit --model M --wording topic|unstated [--wrong] --lrp none|rlens \
        --routers r1.pt [r8.pt] --n 256 --rpp 64 --out DIR
    python scripts/ja1/ja1_fit.py router --model M --wording topic|unstated --nr 1000 --out DIR
    python scripts/ja1/ja1_fit.py validate --model M --layer 12
"""
import argparse
import json
import os
import random
import sys
import time
import types

if os.environ.get("JA1_RESERVE_GB"):  # keep a reserved pool (plain caching allocator) so other jobs cannot take it
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
if os.environ.get("JA1_FLA") != "1":  # torch GatedDeltaNet path (the RelP surgery requires it); JA1_FLA=1: fla kernels
    sys.modules["fla"] = None
sys.path.insert(0, "/root/jpp_lens/scripts")
import torch as t  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.config import LensConfig  # noqa: E402
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer  # noqa: E402
from workspace_lens.fitting.types import FitStepForward  # noqa: E402
from workspace_lens.fitting.utils import source_gradients_summed_over_streams  # noqa: E402
from workspace_lens.lrp import apply_lrp_rules_for_mode  # noqa: E402
from workspace_lens.residual_streams import residual_mean_over_streams  # noqa: E402
from workspace_lens.routing.router import ActivationRouterCollection  # noqa: E402

R = "/workspace/o-lens-improvement"
LAYERS = {"Qwen/Qwen3.5-2B": [3, 6, 9, 12, 15, 18, 21], "Qwen/Qwen3.5-9B": [4, 8, 12, 16, 20, 24, 28]}
SEED = 20261121

# W0 of scripts/ask3/ask3_collect4.py (topic) and of scripts/ask5/ask5_collect.py (unstated), copied verbatim
WORD = {
    "topic": ("", "Passage: ", "One-word topic:",
              [("The tide pulled the little boat away from the dock while the fisherman untangled his nets.", "fishing"),
               ("The senate voted late on Tuesday to approve the new budget, despite", "politics"),
               ("Elle a sorti la farine, les œufs et le beurre pour préparer", "baking"),
               ("Lily saw the old dog shivering in the rain, so she opened the door and", "kindness")]),
    "unstated": ("", "Passage: ", "Unstated word:",
                 [("She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock", "swimming"),
                  ("The astronomer who turned his telescope on the moons of Jupiter was put on trial in Rome because", "Galileo"),
                  ("El médico le puso el termómetro bajo la lengua y frunció el ceño al ver", "fever"),
                  ("The engine coughed twice, the needle sat on empty, and the car rolled to a", "fuel")]),
    # W1, W2 of scripts/ask3/ask3_collect4.py (extra topic targets for the same passage)
    "topic1": ("", "Text: ", "Key concept (one word):",
               [("Ash and molten rock poured down the slopes as the villagers fled toward the coast.", "eruption"),
                ("The company reported a sharp rise in quarterly profits after it cut", "finance"),
                ("Der Arzt hörte das Herz des Kindes ab und", "illness"),
                ("Tom put on his coat and boots and stepped out into the deep white", "blizzard")]),
    "topic2": ("Name the main subject of each text in one word.\n\n", "", "=> Subject:",
               [("The orchestra tuned their instruments as the conductor raised his baton.", "symphony"),
                ("In 1492 the ships left the port of Palos and sailed west into", "exploration"),
                ("O jogador recebeu o cartão vermelho e saiu de campo", "sports"),
                ("She typed the last line of code, pressed enter, and watched the program", "programming")]),
}
SNAPS = (32, 64, 96, 128, 256, 512, 1024, 2048)
RESUME_EVERY = int(os.environ.get("JA1_RESUME_EVERY", "16"))
FIXED_TIME = "2026-11-21_00-00-00"  # fixed creation_time -> fixed checkpoint dir (resume across midnight)


def maybe_resume(tr):
    """Load the latest resume slot (sums of every K, counts, records, mass) into a fresh trainer."""
    from workspace_lens.fitting.types import ExpertFitSums
    from workspace_lens.fitting.utils import expert_checkpoint_filename
    p = f"{tr.config.checkpoint_path}_resume.json"
    if not os.path.exists(p):
        return 0
    ptr = json.load(open(p))
    n, d = ptr["n"], ptr["dir"]
    for K in tr.fit_sums_K_dict:
        st = t.load(os.path.join(d, expert_checkpoint_filename(K)), map_location="cpu", weights_only=True)
        assert st["config"]["num_prompts_trained_on"] == n, (st["config"]["num_prompts_trained_on"], n)
        tr.fit_sums_K_dict[K] = ExpertFitSums.from_state_dict(st)
        tr.position_records = list(st["position_records"])
    tr.completed_prompt_count = tr.next_prompt_idx = n
    if getattr(tr, "mass", None) is not None and os.path.exists(os.path.join(d, "mass.pt")):
        tr.mass = list(t.load(os.path.join(d, "mass.pt"), weights_only=False)["mass"])
        assert len(tr.mass) == n
    print(f"[ja1] resumed at {n} prompts from {d}", flush=True)
    return n


def reserve():
    gb = float(os.environ.get("JA1_RESERVE_GB", "0") or 0)
    if gb > 0:
        x = t.empty(int(gb * 2**30), dtype=t.uint8, device="cuda")
        del x
        print(f"[ja1] reserved {t.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)


def build_prompt(tok, wording, text, wrong_text=None):
    """ask3/ask5 build(..., FAM='chat'). wrong_text: the asked passage is wrong_text and `text` is placed as plain
    context at the start of the user turn (A4 control). Returns (string, char span of `text`)."""
    head, q, lab, ex = WORD[wording]
    shots = head + "".join(f"{q}{a}\n{lab} {w}\n\n" for a, w in ex)
    asked = text if wrong_text is None else wrong_text
    user = shots + f"{q}{asked}\n{lab}"
    if wrong_text is not None:
        user = text + "\n\n" + user
    s = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    s += lab
    if wrong_text is None:
        a = s.rfind(f"{q}{text}\n{lab}") + len(q)
    else:
        a = s.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    assert s[a:a + len(text)] == text
    return s, (a, a + len(text))


def encode(tok, s, span):
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids
    pos = [i for i, (b, e) in enumerate(enc.offset_mapping) if b < span[1] and e > span[0]]
    assert pos and pos == list(range(pos[0], pos[-1] + 1)), pos
    return ids, pos


def passages():
    data = json.load(open(f"{R}/runs/e1/prefixes.json"))
    order = list(range(len(data)))
    random.Random(SEED).shuffle(order)
    return data, order


def prompt_records(tok, wording, n, wrong=False, start=0):
    data, order = passages()
    recs = []
    for j in range(start, start + n):
        i = order[j]
        text = data[i]["text"].strip("\n")
        wt = data[order[(j + 7919) % len(order)]]["text"].strip("\n") if wrong else None  # different passage
        s, span = build_prompt(tok, wording, text, wt)
        ids, pos = encode(tok, s, span)
        recs.append(dict(idx=i, ids=ids, pos=pos, text=text, wrong_text=wt))
    return recs


# ------------------------------------------------------------------ trainer


class SnapMixin:
    """Full stock checkpoints (all Ks) at every prompt count in SNAPS (< total) into '<ckpt>_at<n>'; in-place writes
    (network fs). Prefixes are nested, so half B of a size-2n fit = at<2n> - at<n> (disjoint halves)."""

    def _atomic_write(self, path, write):
        write(path)

    def _write_to(self, path):
        base = self.config.checkpoint_path
        self.config.checkpoint_path = path
        try:
            super().write_checkpoint(final=True)
        finally:
            self.config.checkpoint_path = base

    def _mass_state(self, n):
        pos, sl = self.meta
        return dict(mass=self.mass[:n], pos=pos[:n], seq_len=sl[:n])

    def write_checkpoint(self, *, final=False):
        if not final:
            n = self.next_prompt_idx
            base = self.config.checkpoint_path
            if n % RESUME_EVERY == 0 and n < len(self.prompts):  # resume slots A/B, then the pointer
                slot = f"{base}_resume{(n // RESUME_EVERY) % 2}"
                self._write_to(slot)
                if getattr(self, "meta", None) is not None:
                    t.save(self._mass_state(n), os.path.join(slot, "mass.pt"))
                with open(f"{base}_resume.json", "w") as f:
                    json.dump(dict(n=n, dir=slot), f)
                print(f"[ja1] resume point {n} ({time.strftime('%T')}, max reserved "
                      f"{t.cuda.max_memory_reserved() / 2**30:.1f} GiB)", flush=True)
            if n in SNAPS and n < len(self.prompts):
                self._write_to(f"{base}_at{n}")
                if getattr(self, "meta", None) is not None:
                    t.save(self._mass_state(n), os.path.join(os.path.dirname(os.path.dirname(base)), "mass.pt"))
                print(f"[ja1] snapshot at {n} prompts ({time.strftime('%T')})", flush=True)
            return
        super().write_checkpoint(final=True)


def _assign_recurrent(self, recurrent_states, state_idx=0, **kwargs):
    self.recurrent_states[state_idx] = recurrent_states
    return recurrent_states


class AskTrainer(SnapMixin, ExpertJacobianRelPTrainer):
    mass = None  # per prompt: {layer: per-position sum of squared gradient entries over all rows}

    prefix_cache = os.environ.get("JA1_PREFIX_CACHE", "1") == "1"

    def _run_fit_forward(self, prompt):
        """Prefix cache (default): the tokens before the first passage token cannot depend on the passage, so they
        are run once without a graph (B replicas, use_cache) and the graph-recorded forward covers only
        [first passage token, T]. Gradients w.r.t. the passage tokens are unchanged (validated against a full-
        sequence brute-force Jacobian); positions are then relative to the first passage token."""
        rec = json.loads(prompt)
        B = self.config.jacobian_rows_per_pass
        with apply_lrp_rules_for_mode(self.model, self.config.lrp_mode):
            ids = t.tensor(rec["ids"], device=self.model.input_device)[None]
            pos = t.tensor(rec["pos"])
            kw = {}
            a = int(pos[0]) if self.prefix_cache else 0
            if a > 0:
                with t.no_grad():
                    out = self.model._text_module(input_ids=ids[:, :a].expand(B, -1), use_cache=True)
                cache = out.past_key_values
                for layer_cache in cache.layers:  # stock updates copy_ into the state the graph reads -> assign instead
                    if hasattr(layer_cache, "recurrent_states"):
                        layer_cache.record_past = True
                        layer_cache.update_recurrent_state = types.MethodType(_assign_recurrent, layer_cache)
                kw = dict(past_key_values=cache, use_cache=True)
                ids, pos = ids[:, a:], pos - a
            with ActivationRecorder(self.model.layers, at=[*self.source_layers, self.target_layer],
                                    start_graph_at=min(self.source_layers)) as recorder, t.enable_grad():
                if kw:
                    _ = self.model._text_module(input_ids=ids.expand(B, -1), **kw)
                else:
                    _ = self.model.forward(ids.expand(B, -1))
                target = residual_mean_over_streams(recorder.activations[self.target_layer])
                sources = [recorder.activations[layer] for layer in self.source_layers]
        return FitStepForward(seq_len=ids.shape[1], num_source_positions=int(pos.numel()),
                              source_positions_Int_P=pos.to(target.device), target_residual_BSF=target,
                              source_block_outputs_L_list_BSrN=sources, input_ids_Int_S=ids[0])

    def _backward_passes(self, fs):
        d, B = self.model.d_model, self.config.jacobian_rows_per_pass
        target = fs.target_residual_BSF
        T = fs.seq_len - 1
        b = t.arange(B, device=target.device)
        cot = t.zeros_like(target)
        starts = list(range(0, d, B))
        m = {layer: t.zeros(fs.seq_len, device=target.device) for layer in self.source_layers}
        for k, ds in enumerate(starts):
            n = min(B, d - ds)
            cot.zero_()
            cot[b[:n], T, ds + b[:n]] = 1.0
            grads = tuple(source_gradients_summed_over_streams(fs, cot, retain_graph=k < len(starts) - 1))
            for layer, g in zip(self.source_layers, grads, strict=True):
                m[layer] += g[:n].float().pow(2).sum((0, 2))
            yield ds, n, grads
            del grads
        self.mass.append({layer: v.cpu() for layer, v in m.items()})



class StdTrainer(SnapMixin, ExpertJacobianRelPTrainer):
    """The stock J-lens / J++ estimator (this-and-later targets, skip 16, seq 128), only with snapshots."""


# ------------------------------------------------------------------ stages


def load_model(name):
    model = get_hf_model(name, attn_implementation="sdpa")
    for p in model._hf_model.parameters():
        p.requires_grad_(False)
    return model


def cmd_router(a):
    model = load_model(a.model)
    tok = model.tokenizer
    layers = LAYERS[a.model]
    recs = prompt_records(tok, a.wording, a.nr)
    acts = {l: [] for l in layers}
    with t.no_grad():
        for r in recs:
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(t.tensor(r["ids"], device=model.input_device)[None])
            for l in layers:
                acts[l].append(rec.activations[l][0, r["pos"]].float().cpu())
    acts = {l: t.cat(v) for l, v in acts.items()}
    print(f"[ja1 router] {a.wording}: {len(recs)} passages, {acts[layers[0]].shape[0]} passage-token activations "
          f"(block outputs at layers {layers})", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for K in (1, 8):
        rc = ActivationRouterCollection(K, 64, seed=0)  # recipe: projection dim 64, seed 0
        rc.fit(acts)
        rc.save(os.path.join(a.out, f"router_{a.wording}_K{K}.pt"))
        if K == 8:
            share = {l: t.bincount(rc.assign(acts[l].cuda(), l).cpu(), minlength=8).tolist() for l in layers}
            print(f"[ja1 router] K8 router-set counts per layer: {share}", flush=True)


def cmd_fit(a):
    model = load_model(a.model)
    tok = model.tokenizer
    recs = prompt_records(tok, a.wording, a.n, wrong=a.wrong)
    os.makedirs(a.out, exist_ok=True)
    json.dump([{k: r[k] for k in ("idx", "pos", "text", "wrong_text")} | {"len": len(r["ids"])} for r in recs],
              open(os.path.join(a.out, "prompts.json"), "w"))
    print("[ja1] example prompt:", repr(tok.decode(recs[0]["ids"])), flush=True)
    print("[ja1] its passage tokens:", repr(tok.decode([recs[0]["ids"][p] for p in recs[0]["pos"]])), flush=True)
    routers = {r.num_clusters: r for r in map(ActivationRouterCollection.load, a.routers)}
    cfg = LensConfig(hf_model_name=a.model, checkpoint_name="ja1", artifacts_base_dir=a.out,
                     source_layers=LAYERS[a.model], relative_end_transport_layer=-1,
                     jacobian_rows_per_pass=a.rpp, max_seq_len=4096, skip_first_n_positions=0,
                     checkpoint_every_n_prompts=RESUME_EVERY, lrp_mode=a.lrp, creation_time=FIXED_TIME)
    tr = AskTrainer(cfg, model, [json.dumps(dict(ids=r["ids"], pos=r["pos"])) for r in recs],
                    router_collections_K_dict=routers)
    tr.mass = []
    off = [r["pos"][0] if AskTrainer.prefix_cache else 0 for r in recs]  # mass vectors cover [off, T]
    tr.meta = ([[p - o for p in r["pos"]] for r, o in zip(recs, off)], [len(r["ids"]) - o for r, o in zip(recs, off)])
    reserve()
    n0 = maybe_resume(tr)
    t0 = time.time()
    tr.fit()
    dt = time.time() - t0
    t.save(dict(mass=tr.mass, pos=tr.meta[0], seq_len=tr.meta[1], offset=off), os.path.join(a.out, "mass.pt"))
    json.dump(dict(seconds=dt, per_prompt=dt / max(1, len(recs) - n0), n=len(recs), resumed_at=n0, rpp=a.rpp,
                   ckpt=tr.config.checkpoint_path,
                   prefix_cache=AskTrainer.prefix_cache, graph_tokens=sum(tr.meta[1]) / len(recs),
                   full_tokens=sum(len(r["ids"]) for r in recs) / len(recs)),
              open(os.path.join(a.out, "fit_info.json"), "w"))
    print(f"[ja1] done {len(recs)} prompts in {dt:.0f}s ({dt / max(1, len(recs) - n0):.1f}s/prompt) -> {tr.config.checkpoint_path}",
          flush=True)
    print(tr.config.checkpoint_path)


def cmd_jstd(a):
    """Standard recipe on its own prompts (runs/r1_2b/prompts_raw.json, first n), with snapshots for split halves."""
    from jpp_cli import load_fit_prompts_from_json
    model = load_model(a.model)
    prompts = load_fit_prompts_from_json(f"{R}/runs/r1_2b/prompts_raw.json", a.n)
    routers = {r.num_clusters: r for r in map(ActivationRouterCollection.load, a.routers)}
    cfg = LensConfig(hf_model_name=a.model, checkpoint_name="std", artifacts_base_dir=a.out,
                     source_layers=LAYERS[a.model], relative_end_transport_layer=-1, jacobian_rows_per_pass=a.rpp,
                     max_seq_len=128, skip_first_n_positions=16, checkpoint_every_n_prompts=RESUME_EVERY,
                     lrp_mode=a.lrp, creation_time=FIXED_TIME)
    tr = StdTrainer(cfg, model, prompts, router_collections_K_dict=routers)
    reserve()
    maybe_resume(tr)
    t0 = time.time()
    tr.fit()
    dt = time.time() - t0
    json.dump(dict(seconds=dt, per_prompt=dt / len(prompts), n=len(prompts), rpp=a.rpp, ckpt=tr.config.checkpoint_path,
                   records=tr.position_records[:2]), open(os.path.join(a.out, "fit_info.json"), "w"))
    print(f"[ja1 std] done {len(prompts)} prompts in {dt:.0f}s -> {tr.config.checkpoint_path}", flush=True)


def brute(model, ids, pos, layer, target_layer, lrp):
    """Independent reference: replace block `layer`'s output by a leaf x (batch 1), one backward per output dim of
    h_target[T]; returns sum_{p in pos} d h_target[T] / d x[p]  [d, d]."""
    d = model.d_model
    st = {}

    def repl(m, i, o):
        h = o if t.is_tensor(o) else o[0]
        x = h.detach().clone().requires_grad_(True)
        st["x"] = x
        return x if t.is_tensor(o) else (x, *o[1:])

    def grab(m, i, o):
        st["y"] = o if t.is_tensor(o) else o[0]

    with apply_lrp_rules_for_mode(model, lrp), t.enable_grad():
        h1 = model.layers[layer].register_forward_hook(repl)
        h2 = model.layers[target_layer].register_forward_hook(grab)
        try:
            model.forward(t.tensor(ids, device=model.input_device)[None])
        finally:
            h1.remove()
            h2.remove()
    y, x = st["y"][0, -1], st["x"]
    J = t.zeros(d, d)
    for i in range(d):
        (g,) = t.autograd.grad(y[i], x, retain_graph=i < d - 1)
        J[i] = g[0, pos].float().sum(0).cpu()
    return J


def cmd_validate(a):
    model = load_model(a.model)
    tok = model.tokenizer
    recs = prompt_records(tok, "topic", 2)
    target_layer = len(model.layers) - 1
    out = {}
    for lrp in ("none", "rlens"):
        rc = ActivationRouterCollection(1, 8, seed=0)
        rc.fit({l: t.randn(64, model.d_model) for l in LAYERS[a.model]})
        cfg = LensConfig(hf_model_name=a.model, checkpoint_name="val", artifacts_base_dir="/root/ja1/val",
                         source_layers=LAYERS[a.model], jacobian_rows_per_pass=a.rpp, max_seq_len=4096,
                         skip_first_n_positions=0, checkpoint_every_n_prompts=None, lrp_mode=lrp)
        tr = AskTrainer(cfg, model, [], router_collections_K_dict={1: rc})
        tr.mass = []
        ref = t.zeros(model.d_model, model.d_model)
        for r in recs:
            tr.fit_step(json.dumps(dict(ids=r["ids"], pos=r["pos"])))
            ref += brute(model, r["ids"], r["pos"], a.layer, target_layer, lrp)
        got = tr.fit_sums_K_dict[1].layer_sums_L_dict[a.layer].weighted_jacobian_row_sum_EFN[0]
        rel = float((got - ref).norm() / ref.norm())
        cos = float(t.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0))
        out[lrp] = dict(relF=rel, cos=cos, maxabs=float((got - ref).abs().max()), refmax=float(ref.abs().max()),
                        n_pos=[len(r["pos"]) for r in recs], seq=[len(r["ids"]) for r in recs])
        print(f"[ja1 validate] lrp={lrp} layer {a.layer}: relF {rel:.2e} cos {cos:.6f} {out[lrp]}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "router", "validate", "jstd"])
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--wording", default="topic", choices=list(WORD))
    ap.add_argument("--wrong", action="store_true")
    ap.add_argument("--lrp", default="none")
    ap.add_argument("--routers", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--nr", type=int, default=1000)
    ap.add_argument("--rpp", type=int, default=64)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": cmd_fit, "router": cmd_router, "validate": cmd_validate, "jstd": cmd_jstd}[a.cmd](a)
