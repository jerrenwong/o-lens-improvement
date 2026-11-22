cd /root/dp1/smoke; rm -rf dbg
DP1_DEBUG=1 /root/jpp_lens/.venv/bin/python /workspace/o-lens-improvement/scripts/dp1/dp1_jpp.py fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers 3,6,9,12,15,18,21 --router-path /workspace/o-lens-improvement/runs/dp1/2b_plain/router_K1.pt --num-prompts 2 --prompts-json /workspace/o-lens-improvement/runs/r1_2b/prompts_raw.json --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass 64 --checkpoint-every-n-prompts 16 --artifacts-base-dir /root/dp1/smoke/dbg --checkpoint-name s > /root/dp1/smoke_dbg.log 2>&1
D=$(ls -d dbg/*/s/shard0of1); T=$(ls -d stock/*/s/shard0of1)
/root/jpp_lens/.venv/bin/python - $D $T <<'PY' > /root/dp1/smoke_dbg_cmp.txt 2>&1
import torch,sys
L=lambda p: torch.load(p+"/experts_K1_checkpoint.pt",weights_only=True)["layer_sums"]
v0,g0,st=L(sys.argv[1]),L(sys.argv[1]+"_xG0"),L(sys.argv[2])
r=lambda x,y,l: float((x[l]["weighted_jacobian_row_sum_EFN"]-y[l]["weighted_jacobian_row_sum_EFN"]).norm()/y[l]["weighted_jacobian_row_sum_EFN"].norm())
for l in v0: print(l,"V0-vs-stock",r(v0,st,l),"G0-vs-stock",r(g0,st,l),"V0-vs-G0",r(v0,g0,l))
PY
touch /root/dp1/smoke_dbg.done
