# How to run an experiment (for implementing agents)

You implement and score candidate lenses for the loop described in `auto/program.md`. Read, in this order:
`auto/program.md` (contract + what is known), `auto/failure_modes.md` (how the lenses fail), the idea(s) you were
assigned in `auto/ideas.md`, and the baselines `auto/candidates/jpp.py`, `jlens.py`, `jlens_fq.py`.

## Rules
- Write ONLY new files `auto/candidates/<name>.py` (one per variant; prefix the name with your batch tag).
  Never edit `auto/evaluate.py`, `auto/program.md`, `auto/lens.py`, `auto/results.tsv`, other agents' candidates,
  or anything outside `auto/candidates/`. No git.
- Score with the DEV split only: `cd /workspace/o-lens-improvement && /root/jpp_lens/.venv/bin/python auto/evaluate.py --lens auto/candidates/<name>.py`.
  Never pass `--orchestrator`, never read the bench cache, `auto/split_*.json`, `auto/runs/*` CSVs, or any benchmark
  data file from a candidate or from your own analysis: the labelled FIT split in `ctx.fit` is the only labelled data.
- At most 8 candidates per idea. The dev split has 156 prompts: do not chase small differences (one prompt is
  ~0.5 point on average, more on small tasks). Prefer one clean variant and a small sweep of its one knob.
- Build budget 600 s; keep extra GPU memory under ~8 GB (other experiments share the GPU).

## Pod
- No torch on this machine. Use `fleet ssh wong-olens2 '<cmd>'`. Files written here reach the pod after up to
  ~60 s: compare `md5sum` here and on the pod before running.
- An evaluation takes 1-3 minutes (model load ~1 min). Run evaluations ONE AT A TIME (your own), in tmux:
  `tmux new-session -d -s <tag> '<cmd> > /root/<tag>_<name>.log 2>&1'`, then poll with short ssh calls
  (`tail -12 /root/<tag>_<name>.log`; sleep <= 30 s between polls). Several candidates can be chained in one
  tmux command with `;`. Never kill other tmux sessions; if `nvidia-smi` shows > 60 GB in use, wait.
- The result block starts with `---`; a failed candidate ends with `CRASH`.

## Report back (under 300 words)
A table: candidate name, one-line description, dev_pass10, dev_pass1, dev_pass10_unf, per-task pass@10, build_s.
Then: which variant you would keep and why, what the per-task pattern says about the mechanism, and anything that
looked wrong. Baselines to compare with: J++ 0.5952 / 0.3850 (association .32, multihop .96, multilingual .69,
poetry .03, typo .97); J-lens 0.4146 / 0.2075; J-lens minus always-on features 0.4995 / 0.2511.
