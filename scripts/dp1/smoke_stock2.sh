cd /root/dp1/smoke
/root/jpp_lens/.venv/bin/python /workspace/o-lens-improvement/scripts/r1_jpp.py fit-shard --hf-model-name Qwen/Qwen3.5-2B --layers 3,6,9,12,15,18,21 --router-path /workspace/o-lens-improvement/runs/dp1/2b_plain/router_K1.pt --num-prompts 2 --prompts-json /workspace/o-lens-improvement/runs/r1_2b/prompts_raw.json --skip-first-n-positions 16 --lrp-mode none --jacobian-rows-per-pass 64 --checkpoint-every-n-prompts 16 --artifacts-base-dir /root/dp1/smoke/stock2 --checkpoint-name s > /root/dp1/smoke_stock2.log 2>&1
S=$(ls -d stock2/*/s/shard0of1); T=$(ls -d stock/*/s/shard0of1); D=$(ls -d dp1/*/s/shard0of1)
/root/jpp_lens/.venv/bin/python - $S $T $D <<'PY' > /root/dp1/smoke_cmp.txt 2>&1
import torch,sys
L=lambda p: torch.load(p+"/experts_K1_checkpoint.pt",weights_only=True)["layer_sums"]
a,b,c=L(sys.argv[1]),L(sys.argv[2]),L(sys.argv[3])
r=lambda x,y,l: float((x[l]["weighted_jacobian_row_sum_EFN"]-y[l]["weighted_jacobian_row_sum_EFN"]).norm()/y[l]["weighted_jacobian_row_sum_EFN"].norm())
for l in a: print(l,"stock2-vs-stock",r(a,b,l),"dp1-vs-stock2",r(c,a,l))
PY
touch /root/dp1/smoke2.done
