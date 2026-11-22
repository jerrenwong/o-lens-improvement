"""dp1: wrapper around the unmodified jpp_lens pipeline (as scripts/r1_jpp.py) that adds extra accumulators
to the fit-shard stage. /root/jpp_lens is imported, never edited.

    /root/jpp_lens/.venv/bin/python scripts/dp1/dp1_jpp.py [--dp1-hide-fla] [--dp1-snapshot N] \
        [--dp1-rule runs/t1_2b/content_rule.json] <jpp_cli stage> <args...>

Estimator (workspace_lens/fitting/jacobian_fitting.py + expert_fitting.py): per prompt, kept positions
P = [skip, seq_len-1). One-hot cotangent at EVERY p' in P, so the gradient row read at source p is
    G_p = sum_{p' in P, p' >= p} dh_L[p'] / dh_l[p]
and expert e of layer l is  J_e = sum_{p in cluster e} G_p / #{p in cluster e}  (position-weighted over all prompts;
K=1 gives the pooled J-lens). G_p contains the same-position block D_p = dh_L[p]/dh_l[p] exactly once.

Extra accumulators (same forward, same routing, same counts; written as sibling checkpoint dirs '<shard>_x<NAME>'):
  D1, D2  random-sign estimates of the same-position block: cotangent s_p at each p' in P (s = +-1, seeded per
          prompt and draw), row at p multiplied by s_p -> D_p + sum_{p'>p} s_p s_p' X_{p'p}: unbiased for D_p.
  C       content targets: cotangent only at T = {p' in P : token[p'+1] is content (t1 rule)}, rows read at all P
          (as scripts/t1_jpp.py --t1-mode content).
  S       skip path: a second forward with every block k in (min source layer, target] replaced by
          x + (block(x) - x).detach() (same values, gradient only through the residual skip), LRP surgery on as
          in the main fit; cotangent at all P. = what the same backward returns when no block contributes.
  SC      the same skip-path backward with the cotangent only at T (the skip term of C).
Standard rows (V0) go through the stock LensTrainer._backward_passes into the stock accumulators, unchanged.
"""
import json
import os
import sys

JPP = "/root/jpp_lens"
argv = sys.argv[1:]


def _pop(flag, default=None, has_value=True):
    if flag in argv:
        i = argv.index(flag)
        if not has_value:
            del argv[i]
            return True
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


hide_fla = _pop("--dp1-hide-fla", False, has_value=False)
snapshot = int(_pop("--dp1-snapshot", "0"))
rule_path = _pop("--dp1-rule", "/workspace/o-lens-improvement/runs/t1_2b/content_rule.json")
draws = int(_pop("--dp1-draws", "2"))
if hide_fla:  # as r1_jpp.py: the RelP fit refuses to run while fla is importable
    sys.modules["fla"] = None
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
import torch as t  # noqa: E402
import jpp_cli  # noqa: E402
from workspace_lens.fitting.types import ExpertFitSums  # noqa: E402
from workspace_lens.fitting.utils import source_gradients_summed_over_streams  # noqa: E402
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer  # noqa: E402
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer  # noqa: E402

EXTRA = [f"D{i + 1}" for i in range(draws)] + ["C", "S", "SC"] + (["G0"] if os.environ.get("DP1_DEBUG") else [])
RULE = json.load(open(rule_path))
CONTENT = t.tensor(RULE["content"], dtype=t.bool)
print(f"[dp1] hide_fla={hide_fla} snapshot={snapshot} draws={draws} extra={EXTRA} "
      f"content ids {int(CONTENT.sum())}/{CONTENT.numel()}", flush=True)


class DP1Mixin:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        d = self.model.d_model
        self.dp1_sums = {name: {K: ExpertFitSums.zeros(K, d, self.source_layers)
                                for K in self.router_collections_K_dict} for name in EXTRA}
        self.dp1_stats = {"targets_all": 0, "targets_content": 0, "prompts": 0}

    # -- in-place writes (network fs: no temp-then-rename)
    def _atomic_write(self, path, write):
        write(path)

    # -- passes with an arbitrary cotangent position set and per-position values; the graph is retained
    def _dp1_passes(self, fs, pos_Int_Q, vals_Q, *, free_at_end):
        d = self.model.d_model
        B = self.config.jacobian_rows_per_pass
        target = fs.target_residual_BSF
        bidx = t.arange(B, device=target.device)
        pos = pos_Int_Q.to(target.device)
        vals = vals_Q.to(target.device, target.dtype)
        cot = t.zeros_like(target)
        starts = list(range(0, d, B))
        for i, ds in enumerate(starts):
            n = min(B, d - ds)
            cot.zero_()
            if pos.numel():
                cot[bidx[:n, None], pos[None, :], ds + bidx[:n, None]] = vals[None, :]
            grads = tuple(source_gradients_summed_over_streams(
                fs, cot, retain_graph=not (free_at_end and i == len(starts) - 1)))
            yield ds, n, grads
            del grads

    def _dp1_acc(self, name, memb, P, ds, n, grads, row_scale_P=None):
        for layer, g in zip(self.source_layers, grads, strict=True):
            x = g[:n, P.to(g.device), :].float()
            if row_scale_P is not None:
                x = x * row_scale_P.to(x.device, x.dtype)[None, :, None]
            for K in self.router_collections_K_dict:
                m = memb[K][layer].to(x.device)
                self.dp1_sums[name][K].layer_sums_L_dict[layer].weighted_jacobian_row_sum_EFN[
                    :, ds:ds + n, :] += t.einsum("ep,bpn->ebn", m, x).cpu()

    def _skip_forward(self, prompt):
        lo = min(self.source_layers) + 1
        handles = []

        def hook(mod, args, out):
            x = args[0]
            o = out if t.is_tensor(out) else out[0]
            new = x + (o - x).detach()
            return new if t.is_tensor(out) else (new, *out[1:])

        for k in range(lo, self.target_layer + 1):
            handles.append(self.model.layers[k].register_forward_hook(hook))
        try:
            return self._run_fit_forward(prompt)  # recorder hooks register after ours -> record the skip tensors
        finally:
            for h in handles:
                h.remove()

    def fit_step(self, prompt, *, prompt_idx=None):
        fs = self._run_fit_forward(prompt)
        P = fs.source_positions_Int_P
        memb = self.compute_cluster_memberships(fs)  # adds counts to the stock sums
        for name in EXTRA:
            for K in self.router_collections_K_dict:
                for layer in self.source_layers:
                    c = memb[K][layer].sum(dim=1).cpu()
                    ls = self.dp1_sums[name][K].layer_sums_L_dict[layer]
                    ls.position_count_E += c.long()
                    ls.weight_sum_E += c.float()
        Pc = P.cpu()
        nxt = fs.input_ids_Int_S.cpu()[Pc + 1]
        T = Pc[CONTENT[nxt]]
        self.dp1_stats["targets_all"] += int(Pc.numel())
        self.dp1_stats["targets_content"] += int(T.numel())
        self.dp1_stats["prompts"] += 1
        pidx = -1 if prompt_idx is None else int(prompt_idx)
        if "G0" in EXTRA:  # debug: standard rows first, through our own pass generator
            for ds, n, grads in self._dp1_passes(fs, Pc, t.ones(Pc.numel()), free_at_end=False):
                self._dp1_acc("G0", memb, P, ds, n, grads)
        # random-sign draws (graph retained)
        for r in range(draws):
            g = t.Generator().manual_seed(1_000_003 * (pidx + 7) + 7919 * (r + 1))
            s = (t.randint(0, 2, (Pc.numel(),), generator=g) * 2 - 1).float()
            for ds, n, grads in self._dp1_passes(fs, Pc, s, free_at_end=False):
                self._dp1_acc(f"D{r + 1}", memb, P, ds, n, grads, row_scale_P=s)
        # content targets (graph retained)
        for ds, n, grads in self._dp1_passes(fs, T, t.ones(T.numel()), free_at_end=False):
            self._dp1_acc("C", memb, P, ds, n, grads)
        # stock standard rows (frees the graph) -> stock accumulators, exactly as ExpertJacobianTrainer.fit_step
        for ds, n, grads in self._backward_passes(fs):
            for layer, grad_BSN in zip(self.source_layers, grads, strict=True):
                x = grad_BSN[:n, P.to(grad_BSN.device, non_blocking=True), :].float()
                for K in self.router_collections_K_dict:
                    m = memb[K][layer].to(x.device)
                    self.fit_sums_K_dict[K].layer_sums_L_dict[layer].weighted_jacobian_row_sum_EFN[
                        :, ds:ds + n, :] += t.einsum("ep,bpn->ebn", m, x).cpu()
        self._append_position_record(fs, prompt_idx)
        del fs
        # skip path (second forward; values identical, branches detached)
        fs2 = self._skip_forward(prompt)
        assert t.equal(fs2.source_positions_Int_P.cpu(), Pc)
        for ds, n, grads in self._dp1_passes(fs2, Pc, t.ones(Pc.numel()), free_at_end=False):
            self._dp1_acc("S", memb, P, ds, n, grads)
        for ds, n, grads in self._dp1_passes(fs2, T, t.ones(T.numel()), free_at_end=True):
            self._dp1_acc("SC", memb, P, ds, n, grads)
        del fs2
        return int(Pc.numel() + 1), int(Pc.numel())

    def _write_named(self, base, sums_K_dict, suffix):
        saved_sums, saved_path = self.fit_sums_K_dict, self.config.checkpoint_path
        try:
            self.fit_sums_K_dict = sums_K_dict
            self.config.checkpoint_path = base + suffix
            super().write_checkpoint(final=True)
        finally:
            self.fit_sums_K_dict, self.config.checkpoint_path = saved_sums, saved_path

    def write_checkpoint(self, *, final=False):
        base = self.config.checkpoint_path
        if not final:
            if snapshot and self.next_prompt_idx == snapshot:
                self._write_named(base, self.fit_sums_K_dict, f"_at{snapshot}")
                print(f"[dp1] snapshot at {snapshot} prompts -> {base}_at{snapshot}", flush=True)
            return
        super().write_checkpoint(final=True)
        for name in EXTRA:
            self._write_named(base, self.dp1_sums[name], f"_x{name}")
        st = dict(self.dp1_stats, content_share=self.dp1_stats["targets_content"] / max(1, self.dp1_stats["targets_all"]))
        with open(base + "_dp1_stats.json", "w") as f:
            json.dump(st, f)
        print(f"[dp1] final: wrote stock + {EXTRA}; stats {st}", flush=True)


class DP1Expert(DP1Mixin, ExpertJacobianTrainer):
    pass


class DP1RelP(DP1Mixin, ExpertJacobianRelPTrainer):
    pass


jpp_cli.ExpertJacobianTrainer = DP1Expert
jpp_cli.ExpertJacobianRelPTrainer = DP1RelP

if __name__ == "__main__":
    raise SystemExit(jpp_cli.main(argv))
