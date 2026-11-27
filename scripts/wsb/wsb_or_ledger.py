"""OpenRouter token/cost ledger + stop switch for WorkspaceBench judge runs (imported at interpreter start).

Why: wsbench's own $WSB_TOKEN_LEDGER gets a row only for Anthropic calls (src/wsbench/llm.py: `ledger()` is
called from `_anthropic_result` and the batch path, never from `_chat_result`, the OpenRouter/OpenAI path).
For an OpenRouter judge the tool only tallies `Spend.usd` per family in memory (printed at the end, and in
run.json `spend_usd`). This hook adds one JSONL row per OpenRouter response, with OpenRouter's own reported
cost, so cost.py and judge.sh's MAX_USD watchdog can see spend while the run is going.

Installed by setup_pod.sh as a .pth line in the wsbench env: `import wsb_or_ledger`. A no-op unless
WSB_OR_LEDGER=<absolute path of a jsonl> is set (judge.sh sets it). Never reads or writes any API key.
Rows: {"t", "pid", "fam", "model", "in", "out", "cached", "reasoning", "cost", "finish"}.
Stop switch: if the file $WSB_OR_STOP exists, every new OpenRouter call raises llm.JudgeConfigError
("would fail every call": the family stops; finished calls stay cached in cells.jsonl, so a rerun resumes).
ASSUMPTION: llm._chat_result / llm._openrouter_once keep these names and are looked up as module globals
(true in the tarball of 2026-11-27). If they are renamed the hook prints one warning and does nothing.
"""
import json
import os
import sys
import threading
import time

_PATH = os.environ.get("WSB_OR_LEDGER", "")


def _family():
    for a in sys.argv[1:]:
        if a.startswith("family="):
            return a.split("=", 1)[1]
    return ""


def _install():
    try:
        from wsbench import llm
    except Exception as e:  # wsbench not importable in this interpreter: nothing to patch
        return
    if not hasattr(llm, "_chat_result") or not hasattr(llm, "_openrouter_once"):
        print("[wsb_or_ledger] wsbench.llm changed: ledger hook NOT installed", file=sys.stderr)
        return
    lock = threading.Lock()
    orig_result, orig_once = llm._chat_result, llm._openrouter_once
    stop = os.environ.get("WSB_OR_STOP", "")

    def _get(o, k):
        if isinstance(o, dict):
            return o.get(k)
        v = getattr(o, k, None)
        if v is None:
            v = (getattr(o, "model_extra", None) or {}).get(k)
        return v

    def chat_result(resp, *, model, max_tokens, spend, mark_refusals=False):
        try:
            u = getattr(resp, "usage", None)
            ptd = _get(u, "prompt_tokens_details") if u is not None else None
            ctd = _get(u, "completion_tokens_details") if u is not None else None
            ch = (getattr(resp, "choices", None) or [None])[0]
            row = {
                "t": round(time.time(), 3), "pid": os.getpid(), "fam": _family(), "model": model,
                "in": int(_get(u, "prompt_tokens") or 0) if u is not None else 0,
                "out": int(_get(u, "completion_tokens") or 0) if u is not None else 0,
                "cached": int((_get(ptd, "cached_tokens") if ptd is not None else 0) or 0),
                "reasoning": int((_get(ctd, "reasoning_tokens") if ctd is not None else 0) or 0),
                "cost": llm._openrouter_cost(u),
                "finish": getattr(ch, "finish_reason", None) if ch is not None else None,
            }
            with lock, open(_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e:  # never break a judge call because of the ledger
            print(f"[wsb_or_ledger] ledger write failed: {type(e).__name__}", file=sys.stderr)
        return orig_result(resp, model=model, max_tokens=max_tokens, spend=spend, mark_refusals=mark_refusals)

    async def openrouter_once(*a, **k):
        if stop and os.path.exists(stop):
            raise llm.JudgeConfigError(f"spend cap reached ({stop} exists): no new OpenRouter call")
        # openai/* reasoning models on OpenRouter reject `temperature` under require_parameters (404 "No endpoints
        # found"); the benchmark's own OpenAI-direct route drops it too (llm.py: "dropping temperature ... (OpenAI
        # reasoning models)"), so drop it here unless WSB_OR_KEEP_TEMPERATURE=1.
        if (str(k.get("model", "")).startswith("openai/") and k.get("temperature") is not None
                and os.environ.get("WSB_OR_KEEP_TEMPERATURE") != "1"):
            k["temperature"] = None
        return await orig_once(*a, **k)

    llm._chat_result = chat_result
    llm._openrouter_once = openrouter_once


if _PATH:
    _install()
