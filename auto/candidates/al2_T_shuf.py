"""al2 (T_shuf control): unit(J++ h) + lam * unit(M_shuf h + b), M_shuf = same recipe as al2_T's M_all with word
targets permuted across training words. lam = the one al2_T chooses on FIT (recomputed here identically)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from al2_lib import ELens, choose_lam, fit_line, load_maps, m_path  # noqa: E402


def build(ctx):
    main = load_maps(m_path("all"), ctx.device)
    lam = choose_lam(ctx, main, "M mix")
    del main
    maps = load_maps(m_path("shuf"), ctx.device)
    lens = ELens(ctx.JPP, maps, "mix", lam)
    fit_line("M_shuf", ctx, lens)
    return lens
