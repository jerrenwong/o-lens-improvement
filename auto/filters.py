"""w1: token filter levels for the readout score (PLAN_overnight_24nov.md, W1). Defined once here.

  f1  today's benchmark filter: bench.excluded_ids (decoded text has no letter or digit, minus every accepted
      answer of ANY benchmark item, including the unused sixth task). Untouched: the evaluators keep their code path.
  f2  f1, plus every special / added token (tokenizer.all_special_ids, tokenizer.added_tokens_decoder),
      plus every token with no letter or digit (also those f1 spared for tasks we do not score, e.g. " **"),
      plus every token whose decoded text contains "@" ("@g", "@gmail", " @-@"),
      MINUS the spared set = every candidate id of every item of the five scored tasks in the model's bench(es)
      (an accepted answer of a scored item is never masked).
  f3  f2's mask, plus "count each word once": tokens are grouped by normal_form(decoded text); the rank of a concept
      is 1 + the number of distinct groups, other than the groups holding one of its candidate ids, whose best
      member's logit is strictly greater than the concept's best candidate logit (olens.readout.group_candidate_ranks).
      Top-k lists for display: the best member of each of the top k groups (group_topk).

    FL = Filters(model.tokenizer, V, [bench, ...])     # snapshots each bench's f1 excluded_ids at construction
    F = FL.get("f2")                                   # SimpleNamespace(level, excluded_ids, group_ids, n_groups)
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

LEVELS = ("f1", "f2", "f3")
TASKS = ("association", "multihop", "multilingual", "poetry", "typo")
MARKERS = ("_", "▁", "Ġ")  # "_", SentencePiece "▁", byte-level BPE "Ġ"


def decode_all(tokenizer, V: int) -> list[str]:
    """Decoded text of every id in range(V) (V = LM-head width), as the harness's decode_each_token_id."""
    if hasattr(tokenizer, "batch_decode"):
        return list(tokenizer.batch_decode([[i] for i in range(V)]))
    return [tokenizer.decode([i]) for i in range(V)]


def special_ids(tokenizer) -> set[int]:
    ids = set(getattr(tokenizer, "all_special_ids", None) or [])
    ids |= set((getattr(tokenizer, "added_tokens_decoder", None) or {}).keys())
    return {int(i) for i in ids}


def normal_form(s: str) -> str:
    """strip whitespace, drop ONE leading "_" / "▁" / "Ġ", lower-case, drop one trailing "s" if what remains has
    at least 3 letters. "" for tokens with nothing left (those get a group of their own)."""
    s = s.strip()
    if s[:1] in MARKERS:
        s = s[1:]
    s = s.lower()
    if s.endswith("s") and sum(c.isalpha() for c in s[:-1]) >= 3:
        s = s[:-1]
    return s


def group_ids_of(texts: list[str]) -> tuple[torch.Tensor, int]:
    """[V] long group index per token id, and the number of groups. Empty normal forms are singletons."""
    key2g: dict[str, int] = {}
    out = []
    for i, t in enumerate(texts):
        k = normal_form(t)
        if k == "":
            k = f"\x00{i}"
        g = key2g.get(k)
        if g is None:
            g = key2g[k] = len(key2g)
        out.append(g)
    return torch.tensor(out, dtype=torch.long), len(key2g)


def spared_ids(benches) -> set[int]:
    s: set[int] = set()
    for b in benches:
        for e, cand in zip(b.eval_slugs, b.candidates, strict=True):
            if e in TASKS:
                for ids in cand.values():
                    s.update(int(t) for t in ids)
    return s


class Filters:
    def __init__(self, tokenizer, V: int, benches):
        self.tokenizer, self.V, self.benches = tokenizer, int(V), list(benches)
        self.f1 = [b.excluded_ids.clone() for b in self.benches]  # per bench, as loaded
        self._cache: dict[str, SimpleNamespace] = {}
        self._texts = None

    @property
    def texts(self) -> list[str]:
        if self._texts is None:
            self._texts = decode_all(self.tokenizer, self.V)
        return self._texts

    def get(self, level: str) -> SimpleNamespace:
        if level not in LEVELS:
            raise ValueError(f"filter level must be one of {LEVELS}, got {level!r}")
        if level in self._cache:
            return self._cache[level]
        if level == "f1":  # per-bench lists; evaluators do not touch the bench at f1
            F = SimpleNamespace(level="f1", excluded_ids=self.f1[0].clone(), group_ids=None, n_groups=None)
        else:
            texts = self.texts
            base = set().union(*(set(x.tolist()) for x in self.f1))
            nonsem = {i for i, t in enumerate(texts) if not any(c.isalnum() for c in t)}
            at = {i for i, t in enumerate(texts) if "@" in t}
            spec = {i for i in special_ids(self.tokenizer) if i < self.V}
            spared = spared_ids(self.benches)
            ex = torch.tensor(sorted((base | nonsem | spec | at) - spared), dtype=torch.long)
            F = SimpleNamespace(level=level, excluded_ids=ex, group_ids=None, n_groups=None,
                                parts=dict(f1=len(base), nonsem=len(nonsem), special=len(spec), at=len(at),
                                           spared_in_union=len((base | nonsem | spec | at) & spared)))
            if level == "f3":
                F.group_ids, F.n_groups = group_ids_of(texts)
        self._cache[level] = F
        return F

    def apply(self, level: str) -> SimpleNamespace:
        """Set every bench's excluded_ids to the level's mask (no-op at f1). Call before ctx.fit is built."""
        F = self.get(level)
        if level != "f1":
            for b in self.benches:
                b.excluded_ids = F.excluded_ids.clone()
                b._pairs_cache.pop(("excluded_set",), None)
        return F

    @staticmethod
    def group_topk(logits_V, group_ids, k=10):
        return group_topk(logits_V, group_ids, k)


def group_topk(logits_V: torch.Tensor, group_ids: torch.Tensor, k: int = 10) -> list[int]:
    """Token ids: the best member of each of the top-k groups (one row of logits, masked ids at -inf)."""
    gid = group_ids.to(logits_V.device)
    n = min(logits_V.shape[-1], 256)
    while True:
        top = logits_V.topk(n).indices.tolist()
        seen, out = set(), []
        for t in top:
            g = int(gid[t])
            if g not in seen:
                seen.add(g); out.append(t)
                if len(out) == k:
                    return out
        if n >= logits_V.shape[-1]:
            return out
        n = min(logits_V.shape[-1], n * 8)
