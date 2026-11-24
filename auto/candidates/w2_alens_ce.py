"""W2a "A-lens-CE": auto/candidates/alens.py unchanged (same rows via ALENS_BASE, same lam grid / FIT rule), with the
two ridge maps replaced by the cross-entropy (P3v) maps of scripts/w2/w2_ce_fit.py:
   topic question  .hf27/w2/E_<tag>_topic_<var>_k<K>.pt
   unstated word   .hf27/w2/E_<tag>_spec_<var>_k<K>.pt
env W2_K (10 | 50 | 200, default 50), W2_VAR (ce | shuf | const, default ce), ALENS_BASE as in alens.py."""
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = "/workspace/o-lens-improvement"
TAG = {2048: "2b", 4096: "9b", 5120: "27b"}


def build(ctx):
    tag = TAG[ctx.model.d_model]
    k, var = os.environ.get("W2_K", "50"), os.environ.get("W2_VAR", "ce")
    os.environ["ALENS_E"] = f"{R}/.hf27/w2/E_{tag}_topic_{var}_k{k}.pt"
    os.environ["ALENS_E2"] = f"{R}/.hf27/w2/E_{tag}_spec_{var}_k{k}.pt"
    print(f"[w2_alens_ce] maps {os.environ['ALENS_E']} + {os.environ['ALENS_E2']}; "
          f"ALENS_BASE={os.environ.get('ALENS_BASE', 'JPP')}", flush=True)
    spec = importlib.util.spec_from_file_location("alens_frozen", HERE / "alens.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(ctx)
