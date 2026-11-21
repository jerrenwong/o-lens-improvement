"""Validate v4 eval files with the harness's own loader and candidate construction (tokenizer only).

    /root/jpp_lens/.venv/bin/python scripts/v4_check_items.py <dir with lens-eval-<task>-v4.json> [--tok MODEL]

Copies the files into a temp dir under the names the harness expects (lens-eval-<task>.json), loads
them with lens_evals' load_readout_eval_items, and for every item checks: every intermediate has a
single-token candidate (candidate_token_ids, as _item_readout_and_candidates); the readout position
resolves (readout_position under the eval's rule); the item survives runner.scoreable_items logic.
Prints failures as JSON list "task/name" on the last line (prefix DROP:).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

JPP = "/root/jpp_lens"
sys.path.insert(0, f"{JPP}/src")
from transformers import AutoTokenizer  # noqa: E402

from lens_evals.eval_utils import candidate_token_ids  # noqa: E402
from lens_evals.readout_evals.readout_eval_items import (  # noqa: E402
    MACRO_EVALS,
    READOUT_EVAL_SPECS,
    load_readout_eval_items,
)
from lens_evals.readout_evals.readout_evals import readout_position  # noqa: E402

src = Path(sys.argv[1])
tok_name = sys.argv[sys.argv.index("--tok") + 1] if "--tok" in sys.argv else "Qwen/Qwen3.5-9B"
tok = AutoTokenizer.from_pretrained(tok_name)

with tempfile.TemporaryDirectory() as d:
    for t in MACRO_EVALS:
        Path(d, f"lens-eval-{t}.json").write_bytes((src / f"lens-eval-{t}-v4.json").read_bytes())
    items = load_readout_eval_items(d, slugs=MACRO_EVALS)

drop, counts = [], {}
for it in items:
    spec = READOUT_EVAL_SPECS[it.eval_slug]
    bad = []
    for x in it.intermediates:
        c = candidate_token_ids(tok, x, expand_order_ops_synonyms=spec.expand_order_ops_synonyms)
        if not c:
            bad.append(x)
    ids = tok(it.prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
    pos = readout_position(ids, tok, spec.readout_rule)
    note = ""
    if it.eval_slug == "poetry":
        # readout must be the newline ending line 1 (the second newline of the prompt)
        before = tok.decode(ids[0, : pos + 1])
        if before.count("\n") != 2 or not before.endswith("\n"):
            note = f"poetry readout at {before[-20:]!r}"
    if bad or note:
        drop.append(f"{it.eval_slug}/{it.name}")
        print(f"FAIL {it.eval_slug}/{it.name}: no single-token candidate for {bad} {note}")
    counts.setdefault(it.eval_slug, [0, 0])
    counts[it.eval_slug][0] += 1
    counts[it.eval_slug][1] += not (bad or note)
for t, (n, ok) in counts.items():
    print(f"{t:14s} loaded {n:4d} valid {ok:4d}")
print("DROP:" + json.dumps(drop))
