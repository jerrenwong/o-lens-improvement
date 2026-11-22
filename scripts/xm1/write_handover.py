"""xm1: write runs/xm1/<tag>/handover.md from the run's json files.  python scripts/xm1/write_handover.py <model> <tag> <boot.txt> <notes.md>"""
import json
import subprocess
import sys

R = '/workspace/o-lens-improvement'
M, T, BOOT, NOTES = sys.argv[1:5]
info = json.load(open(f'{R}/runs/xm1/{T}/info.json')); lrp = json.load(open(f'{R}/runs/xm1/{T}/lrp_check.json'))
fc = json.load(open(f'{R}/runs/xm1/{T}/format_check.json')); fc5 = json.load(open(f'{R}/runs/xm1/{T}/format_check_s5.json'))
mf = json.load(open(f'{R}/runs/xm1/{T}/map_fit.json')); mf5 = json.load(open(f'{R}/runs/xm1/{T}/map_fit_s5.json'))
st = json.load(open(f'{R}/runs/xm1/{T}/jpp_{T}_stamp.json'))
table = subprocess.run(['python3', f'{R}/scripts/xm1/table.py', T], capture_output=True, text=True).stdout
boot = open(BOOT).read()
cmds = subprocess.run(['python3', f'{R}/scripts/xm1/handover_cmds.py', M, T], capture_output=True, text=True).stdout
timing = open(f'{R}/runs/xm1/{T}/timing.txt').read()
notes = open(NOTES).read()
f = lambda d: " ".join(f"{x*100:.1f}%" for x in d)  # noqa: E731
ok = lambda k: all(lrp[p][k] for p in ("prompt0", "prompt1"))  # noqa: E731
out = f"""# xm1 handover: {M} (tag {T})

Status: DEV done. Held-out sets (test split, v4) NOT scored; commands below are for the orchestrator.

## Architecture
- {info['hf_class']} / {info['block_class']} / {info['mlp_class']}, {info['dtype']}, {info['n_layers']} blocks, d_model {info['d_model']}, LM head rows {info['lm_head_rows']}, tied {info['tied']}, logit soft-cap {info['logit_softcap']}.
- Readout layers (floor(k*depth/8)): {info['readout_layers']}
- Unembed = final norm + linear head: closed-form rel err {info.get('final_linear_maxdiff', info.get('final_linear_maxdiff_precap'))} {'(before the soft-cap, which is monotone per logit)' if info['logit_softcap'] else ''}; unembed(last block) argmax == model logits argmax: {info['unembed_argmax_equal']}.
- Tokenizer BOS `{info['bos_token']}` (forced by the harness). Template thinking switch: {info['template_has_thinking_switch']}.
- LRP (authors' patcher, rlens): {lrp['patched_counts']['ln_rule']} RMSNorms (kinds {lrp['ln_norm_kinds']}), {lrp['patched_counts']['mlp_rule']} MLPs = every block MLP: {lrp['mlp_rule_is_every_block_mlp']}; no attention-internal norm: {lrp['no_attention_internal_norm_patched']}. On 2 fit prompts: detach-only rules bit-exact {ok('detach_only_bit_exact')}, context exit restores bit-exact {ok('exit_restores_bit_exact')}, full rlens forward max rel diff {lrp['prompt0']['full_rules_max_rel_diff']:.3f}/{lrp['prompt1']['full_rules_max_rel_diff']:.3f} (bf16 activation-rewrite dust over all blocks; the authors' 1e-4 tolerance is for tiny fp32 models), half rule rel err {lrp['half_rule_max_rel_err']:.4f}, LN-rule grad = cot*w*scale rel err {lrp['ln_rule_grad_rel_err_vs_plain_w']:.4f}, LRP backward finite/non-zero at every readout layer: {all(lrp['lrp_backward_finite_nonzero'].values())}.

## J++ / J-lens fit (authors' jpp_cli, default recipe)
```
{timing}```
J++ FIT objective (fit-weights train macro pass@10 per layer): {json.dumps({k: round(v, 3) for k, v in st['train_macro_pass'].items()})}

## Ask-maps
- Topic (t3), label pre-started: not-a-word share per wording {f(fc['label']['per_wording'])}, pooled {fc['label']['pooled']*100:.1f}% -> {'PASS' if fc['label']['pooled'] <= .2 else 'FAIL'} (rule: > 20% fails). Fallback (no label): {f(fc['fallback']['per_wording'])}, pooled {fc['fallback']['pooled']*100:.1f}%.
- Unstated (s5), label: {f(fc5['label']['per_wording'])}, pooled {fc5['label']['pooled']*100:.1f}% -> {'PASS' if fc5['label']['pooled'] <= .2 else 'FAIL'}. Fallback: {f(fc5['fallback']['per_wording'])}, pooled {fc5['fallback']['pooled']*100:.1f}%.
- Topic map, held-out passages, word agreement top10 per layer: E {[round(r['E10'], 3) for r in mf['rows']]}; J++ {[round(r['J10'], 3) for r in mf['rows']]}; shuffled {[round(r['shuf10'], 3) for r in mf['rows']]}; ridge c {[r['c'] for r in mf['rows']]}; dropped vectors {mf['dropped_vectors']}/91500, passages {mf['dropped_passages']}.
- s5 map: E top10 {[round(r['E10'], 3) for r in mf5['rows']]}; c {[r['c'] for r in mf5['rows']]}; dropped vectors {mf5['dropped_vectors']}, passages {mf5['dropped_passages']}.

{notes}
## DEV (auto/evaluate_x.py --split dev --lite; jppu_full = same lens without --lite; jppraw = auto/lens.py)
{table}
## DEV paired bootstrap (scripts/r2_boot.py, 5000 draws within task; pass points)
```
{boot}```

## Held-out commands (orchestrator). Dry-checked: each lens file scored on dev above, and auto/evaluate_x_v4.py --dry-run built and transported all v4 rows for jppu, L, Lshuf, Lprior and the five A-lens rows (nothing ranked).
```
{cmds}```

## Files
maps .hf27/xm1/E_{T}_t3.pt (+_shuf), E_{T}_s5.pt (+_shuf); J++ runs/xm1/{T}/jpp_{T}.pt; J-lens runs/xm1/{T}/pooled_jlens.pt; bench .hf27/xm1/bench_{T}.pt; v4 .hf27/xm1/v4_bench_{T}.pt; splits auto/split_{T}.json, auto/split_v4_{T}.json; correctness data/model_correctness_xm1.csv, data/model_correctness_v4_{T}.csv; format checks / map fits runs/xm1/{T}/*.json; logs runs/xm1/{T}/dev_logs/ (pod: /root/xm1).
"""
open(f'{R}/runs/xm1/{T}/handover.md', 'w').write(out)
print(out)
