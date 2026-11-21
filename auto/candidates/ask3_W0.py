"""ask3 (W0): J++ + lam * E_t3w0, the clean chat-format topic ask-map built from wording 0 only (learned, not a
Jacobian). lam on FIT from {0.1,0.2,0.3,0.5,1.0,2.0}."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask3_lib import ELens, R, TAGS, choose_lam, load_maps  # noqa: E402


def build(ctx):
    maps = load_maps(f"{R}/.hf27/ask3/E_{TAGS[ctx.model.d_model]}_t3w0.pt", ctx.device)
    return ELens(ctx.JPP, maps, "mix", choose_lam(ctx, maps))
