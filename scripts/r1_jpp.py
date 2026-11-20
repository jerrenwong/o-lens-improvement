"""Thin wrapper around the unmodified jpp_lens pipeline (/root/jpp_lens/scripts/jpp_cli.py) for the r1 refit study.

    /root/jpp_lens/.venv/bin/python scripts/r1_jpp.py [--r1-router-prompts-json P.json] <jpp_cli stage> <args...>

What it changes, and nothing else:
  * --r1-router-prompts-json: fit-router has no --prompts-json flag (it always streams WikiText via
    jpp_cli.load_fit_prompts); this replaces that function with the first N of a JSON prompt pool
    (jpp_cli.load_fit_prompts_from_json), so the router sees the same text as the shards.
  * Logs the skip_first_n_positions value and the first kept position the first time each position-
    selecting code path (routing.cluster and fitting.jacobian_fitting) calls
    get_position_mask_with_early_skips, to verify which positions enter the fit.
Runs from /root/jpp_lens (jpp_cli reads its eval data by relative path): pass absolute paths.
"""
import os
import sys

JPP = "/root/jpp_lens"
argv = sys.argv[1:]
router_json = None
if "--r1-router-prompts-json" in argv:
    i = argv.index("--r1-router-prompts-json")
    router_json = argv[i + 1]
    del argv[i:i + 2]
if argv and argv[0] == "fit-shard":
    # The RelP (LRP) fit refuses to run while `fla` is importable (workspace_lens/lrp/lrp.py: its fused
    # GatedDeltaNet kernels bypass the LRP surgery; "run RelP fits in an environment without fla").
    # This venv has fla, so hide it for this stage only: find_spec("fla") -> None, transformers falls
    # back to the torch GatedDeltaNet path. Must happen before transformers is imported.
    sys.modules["fla"] = None
    import importlib.util as _u
    print(f"[r1] fla hidden for fit-shard: find_spec('fla') = {_u.find_spec('fla')}", flush=True)
os.chdir(JPP)
sys.path.insert(0, os.path.join(JPP, "scripts"))
import jpp_cli  # noqa: E402
import workspace_lens.fitting.jacobian_fitting as jf  # noqa: E402
import workspace_lens.routing.cluster as cl  # noqa: E402

if router_json is not None:
    jpp_cli.load_fit_prompts = lambda n: jpp_cli.load_fit_prompts_from_json(router_json, n)
    print(f"[r1] fit-router prompts <- first N of {router_json}", flush=True)


def _logged(mod, name):
    orig = getattr(mod, "get_position_mask_with_early_skips")
    seen = []

    def wrapped(seq_len, skip):
        m = orig(seq_len, skip)
        if not seen:
            seen.append(1)
            kept = m.nonzero().flatten()
            print(f"[r1] {name}: get_position_mask_with_early_skips(seq_len={seq_len}, skip={skip}) -> "
                  f"kept positions {int(kept[0])}..{int(kept[-1])} ({kept.numel()})", flush=True)
        return m

    setattr(mod, "get_position_mask_with_early_skips", wrapped)


_logged(jf, "fitting.jacobian_fitting")
_logged(cl, "routing.cluster")
if __name__ == "__main__":
    raise SystemExit(jpp_cli.main(argv))
