"""w1 unit test: filter levels f1/f2/f3 on a toy vocabulary (CPU). Run from the repo root:
    PYTHONPATH=src /root/jpp_lens/.venv/bin/python scripts/w1/test_filters.py"""
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from olens.readout import Bench, candidate_ranks, group_candidate_ranks, readout_rows  # noqa: E402

spec = importlib.util.spec_from_file_location("w1_filters", ROOT / "auto" / "filters.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

VOCAB = ["child", "Child", " Child", "_Child", "children", " **", "<|im_end|>", "@g", "@gmail", " dog",   # 0-9
         " Dogs", "!", " cat", " @-@", "@", " class", "is", "<think>", "▁Child", " bus"]             # 10-19
NEG = float("-inf")


class Tok:
    all_special_ids = [6]
    added_tokens_decoder = {6: "<|im_end|>", 17: "<think>"}

    def decode(self, ids):
        return VOCAB[ids[0]]


def bench():
    # f1 as the harness builds it: no-letter-or-digit ids minus candidates of ALL six tasks ("orderops" spares " **")
    return Bench(model="toy", layers=[0], eval_slugs=["association", "typo", "orderops"], item_names=["a", "b", "c"],
                 resid={0: torch.zeros(3, 2)},
                 candidates=[{"x": [0], "y": [9, 12]}, {"z": [14]}, {"w": [5]}],
                 split=["held-out"] * 3, excluded_ids=torch.tensor([11, 13]))


def main():
    b = bench(); FL = F.Filters(Tok(), len(VOCAB), [b])
    f1, f2, f3 = (FL.get(l) for l in ("f1", "f2", "f3"))
    assert f1.excluded_ids.tolist() == [11, 13] and f1.group_ids is None
    FL.apply("f1"); assert b.excluded_ids.tolist() == [11, 13], "f1 must not touch the bench"
    # f2: + specials (6 all_special, 17 added), + " **" (spared only by the unused task), + "@g", "@gmail", " @-@";
    #     "@" (14) is an accepted answer of a scored item -> kept.
    assert f2.excluded_ids.tolist() == [5, 6, 7, 8, 11, 13, 17], f2.excluded_ids.tolist()
    assert f3.excluded_ids.tolist() == f2.excluded_ids.tolist()
    g = f3.group_ids.tolist()
    assert g[0] == g[1] == g[2] == g[3] == g[18], "child / Child / ' Child' / '_Child' / '▁Child' must be one group"
    assert g[4] != g[0], "children stays separate"
    assert g[9] == g[10], "' dog' / ' Dogs' one group"
    assert F.normal_form(" class") == "clas" and F.normal_form("is") == "is" and F.normal_form(" bus") == "bus"
    assert len(set(g)) == len(VOCAB) - 5, len(set(g))  # 4 child dups + Dogs merged

    # ranks. logits row 0: candidate "child"=1.0, " Child"=5 (same group), " dog"=4, " Dogs"=3, " cat"=2, "children"=1.5,
    # "@g"=10 and " **"=9 (masked at f2), "bus"=1.0 (tie: not counted).
    L = torch.full((3, len(VOCAB)), -5.0)
    L[0, [0, 2, 9, 10, 12, 4, 7, 5, 19]] = torch.tensor([1.0, 5, 4, 3, 2, 1.5, 10, 9, 1.0])
    L[1, 14] = 3.0; L[1, 7] = 4.0; L[1, 0] = 3.5
    m = torch.zeros(len(VOCAB), dtype=torch.bool); m[f2.excluded_ids] = True
    Lm = L.masked_fill(m, NEG)
    tok_rank = candidate_ranks(Lm, b.candidates)
    grp_rank = group_candidate_ranks(Lm, b.candidates, f3.group_ids)
    assert tok_rank[0] == {"x": 6, "y": 2}, tok_rank[0]         # tokens above 1.0: " Child", dog, Dogs, cat, children
    assert grp_rank[0] == {"x": 4, "y": 2}, grp_rank[0]         # groups above: dog, cat, children; y: child-group only
    assert candidate_ranks(L, b.candidates)[0] == {"x": 8, "y": 4}  # unfiltered: + "@g", " **"
    assert grp_rank[1] == {"z": 2}, grp_rank[1]                 # "@g" masked, child-group 3.5 above "@" 3.0
    top = F.group_topk(Lm[0], f3.group_ids, 4)
    assert [VOCAB[t] for t in top] == [" Child", " dog", " cat", "children"], [VOCAB[t] for t in top]

    # through readout_rows (the evaluator path), with a fake model whose unembed returns L
    class M:
        def unembed(self, v):
            return L[v[:, 0].long()]
    FL.apply("f3")
    assert b.excluded_ids.tolist() == f2.excluded_ids.tolist()
    vec = torch.tensor([[0.0, 0], [1, 0]])
    rows = readout_rows("t", 0, b, vec, M(), filtered=True, items=[0, 1], chunk_size=1, group_ids=f3.group_ids)
    assert {(r["item"], r["intermediate"]): r["rank"] for r in rows} == {("a", "x"): 4, ("a", "y"): 2, ("b", "z"): 2}
    rows = readout_rows("t", 0, b, vec, M(), filtered=True, items=[0, 1], chunk_size=1)
    assert {(r["item"], r["intermediate"]): r["rank"] for r in rows} == {("a", "x"): 6, ("a", "y"): 2, ("b", "z"): 2}
    print("test_filters OK")


if __name__ == "__main__":
    main()
