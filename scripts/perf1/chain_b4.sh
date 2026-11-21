#!/bin/bash
# perf1 Part B: generation at 8 prompts x 3 samples per batch for torch vs fla (torch OOMs at 24 under the 0.3 cap).
cd /workspace/o-lens-improvement
while tmux has-session -t perf1b2 2>/dev/null || tmux has-session -t perf1b3 2>/dev/null; do sleep 30; done
PY=/root/jpp_lens/.venv/bin/python
PYF=/workspace/o-lens-improvement/.venv_fast/bin/python
R=scripts/perf1/run_eval.sh
export PYTHONUNBUFFERED=1 PERF1_GEN_PB=8
bash $R b_torch_gen8 26 $PY scripts/perf1/bench_b.py --mode torch --tasks gen --tag _gen8
mv .hf27/perf1/gen_torch.json .hf27/perf1/gen8_torch.json
bash $R b_fla_gen8 26 $PY scripts/perf1/bench_b.py --mode fla --tasks gen --tag _gen8
mv .hf27/perf1/gen_fla.json .hf27/perf1/gen8_fla.json
bash $R b_fla_cc_gen8 26 $PYF scripts/perf1/bench_b.py --mode fla_cc --tasks gen --tag _gen8
mv .hf27/perf1/gen_fla_cc.json .hf27/perf1/gen8_fla_cc.json
bash $R b_ab 26 $PYF scripts/perf1/ab_timing.py --rounds 5
