"""t1 "content targets": wrapper around the unmodified jpp_lens pipeline (based on scripts/r1_jpp.py).

    /root/jpp_lens/.venv/bin/python scripts/t1_jpp.py [--r1-router-prompts-json P.json]
        [--t1-mode all|content] [--t1-rule runs/t1_2b/content_rule.json] [--t1-log OUT.json] <jpp_cli stage> <args>

How the estimator uses positions (workspace_lens/fitting/jacobian_fitting.py, expert_fitting.py):
  LensTrainer._run_fit_forward builds ONE position set P = get_position_mask_with_early_skips (skip first 16, drop
  last) and stores it as forward_state.source_positions_Int_P. It is used twice:
   (1) TARGETS: LensTrainer._backward_passes puts the one-hot cotangent (output dim d) at every p' in P of the
       final-layer residual, so one backward gives, at source p, sum_{p' in P, p' >= p} dh_final[p',d]/dh_l[p];
   (2) SOURCES: ExpertJacobianTrainer.fit_step reads those gradient rows at p in P, routes them, and adds them to
       the per-cluster sums (weight 1 per source position; the pooled lens = all-cluster sum / position count).
  Targets are therefore summed INSIDE one backward pass (the cotangent), not averaged separately.
Patch (--t1-mode content): only (1) changes. LensTrainer._backward_passes is called with a copy of forward_state
  whose source_positions_Int_P is the content-target subset T of P (t in P kept iff token[t+1] contains a letter
  and is not among the 200 most frequent ids of the 1,000-prompt pool). _backward_passes uses that field only to
  place the cotangent, so the backward now yields sum_{p' in T, p' >= p}; fit_step still reads, routes and counts
  rows at the unchanged source set P (the original forward_state). Equivalent to zeroing the target weights of
  dropped positions in the back-propagated scalar. Core code is not copied or edited.
--t1-mode all: no patch (recipe); the would-be-kept share is still logged.
"""
import dataclasses
import json
import os
import sys

JPP = "/root/jpp_lens"
argv = sys.argv[1:]


def _pop(flag, default=None):
    if flag in argv:
        i = argv.index(flag); v = argv[i + 1]; del argv[i:i + 2]; return v
    return default


router_json = _pop("--r1-router-prompts-json")
mode = _pop("--t1-mode", "all")
rule_path = _pop("--t1-rule", "/workspace/o-lens-improvement/runs/t1_2b/content_rule.json")
log_path = _pop("--t1-log")
assert mode in ("all", "content")
if argv and argv[0] == "fit-shard":
    sys.modules["fla"] = None  # as r1_jpp.py: RelP fit refuses to run while fla is importable
    import importlib.util as _u
    print(f"[t1] fla hidden for fit-shard: find_spec('fla') = {_u.find_spec('fla')}", flush=True)
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
import torch as t  # noqa: E402
import jpp_cli  # noqa: E402
import workspace_lens.fitting.jacobian_fitting as jf  # noqa: E402
import workspace_lens.routing.cluster as cl  # noqa: E402

if router_json is not None:
    jpp_cli.load_fit_prompts = lambda n: jpp_cli.load_fit_prompts_from_json(router_json, n)
    print(f"[t1] fit-router prompts <- first N of {router_json}", flush=True)


def _logged(mod, name):
    orig = getattr(mod, "get_position_mask_with_early_skips"); seen = []

    def wrapped(seq_len, skip):
        m = orig(seq_len, skip)
        if not seen:
            seen.append(1); kept = m.nonzero().flatten()
            print(f"[t1] {name}: get_position_mask_with_early_skips(seq_len={seq_len}, skip={skip}) -> "
                  f"kept positions {int(kept[0])}..{int(kept[-1])} ({kept.numel()})", flush=True)
        return m
    setattr(mod, "get_position_mask_with_early_skips", wrapped)


_logged(jf, "fitting.jacobian_fitting")
_logged(cl, "routing.cluster")

if argv and argv[0] == "fit-shard":
    rule = json.load(open(rule_path))
    CONTENT = t.tensor(rule["content"], dtype=t.bool)
    print(f"[t1] mode={mode} rule={rule_path} (content ids {int(CONTENT.sum())}/{CONTENT.numel()})", flush=True)
    LT = jf.LensTrainer
    _orig_fwd, _orig_bwd = LT._run_fit_forward, LT._backward_passes
    STATS = {"mode": mode, "prompts": 0, "targets_before": 0, "targets_after": 0, "per_prompt": [],
             "examples": []}
    _tok = {}

    def _decode(i):
        if "tok" not in _tok:
            from transformers import AutoTokenizer
            _tok["tok"] = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
        return _tok["tok"].decode([int(i)])

    def run_fit_forward(self, prompt):
        fs = _orig_fwd(self, prompt)
        src = fs.source_positions_Int_P.cpu()
        ids = fs.input_ids_Int_S.cpu()
        nxt = ids[src + 1]
        keep = CONTENT[nxt]
        tgt = src[keep]
        self._t1_targets = tgt.to(fs.source_positions_Int_P.device)
        n = STATS["prompts"]; STATS["prompts"] += 1
        STATS["targets_before"] += int(src.numel()); STATS["targets_after"] += int(tgt.numel())
        STATS["per_prompt"].append([int(src.numel()), int(tgt.numel())])
        if n < 3:
            kept = [repr(_decode(x)) for x in nxt[keep][:20].tolist()]
            drop = [repr(_decode(x)) for x in nxt[~keep][:20].tolist()]
            STATS["examples"].append({"prompt": n, "before": int(src.numel()), "after": int(tgt.numel()),
                                      "kept_next": kept, "dropped_next": drop})
            print(f"[t1] prompt {n}: target positions before {src.numel()} after {tgt.numel()} "
                  f"(mode {mode}; {'applied' if mode == 'content' else 'NOT applied'})", flush=True)
            print(f"[t1]   kept next tokens:    {' '.join(kept)}", flush=True)
            print(f"[t1]   dropped next tokens: {' '.join(drop)}", flush=True)
        if log_path:
            json.dump(STATS, open(log_path, "w"))
        return fs

    def backward_passes(self, forward_state):
        if mode == "content":
            fs_t = dataclasses.replace(forward_state, source_positions_Int_P=self._t1_targets)
            if not getattr(self, "_t1_checked", False):
                self._t1_checked = True
                print(f"[t1] backward cotangent positions = content targets ({fs_t.source_positions_Int_P.numel()}"
                      f" of {forward_state.source_positions_Int_P.numel()}); rows read at all "
                      f"{forward_state.source_positions_Int_P.numel()} source positions", flush=True)
            return _orig_bwd(self, fs_t)
        return _orig_bwd(self, forward_state)

    LT._run_fit_forward = run_fit_forward
    LT._backward_passes = backward_passes

if __name__ == "__main__":
    rc = jpp_cli.main(argv)
    if argv and argv[0] == "fit-shard":
        print(f"[t1] DONE mode={mode} prompts={STATS['prompts']} targets before {STATS['targets_before']} after "
              f"{STATS['targets_after']} share {STATS['targets_after'] / max(1, STATS['targets_before']):.4f}",
              file=sys.stderr, flush=True)
    raise SystemExit(rc)
