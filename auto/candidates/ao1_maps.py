"""ao1 candidate: NO Jacobian at all. v = unit(E_topic h + b) + mu unit(E_unstated h + b') (env AO1_E, AO1_E2);
mu chosen on FIT (max filtered pass@1 + pass@10, ties -> smaller mu) over MU_GRID. ctx.J / ctx.JPP never read."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ao1_lib  # noqa: E402
from ao1_lib import Lens, end_line, grid_search, maps_E  # noqa: E402

MU_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]


def build(ctx):
    E, E2 = maps_E(ctx), maps_E(ctx, "AO1_E2")
    end_line(ctx, [("E", E2, 1.0)], "unstated map alone")
    ao1_lib.GRID = MU_GRID  # grid_search reads the module-level grid
    mu, _, _ = grid_search(ctx, lambda w: Lens([("E", E, 1.0), ("E", E2, w)]), "maps mu")
    return Lens([("E", E, 1.0), ("E", E2, mu)])
