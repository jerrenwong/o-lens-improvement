"""How much of the identity is inside each fitted lens matrix? c = tr(J)/d is the least-squares coefficient of I in J."""
import torch
from workspace_lens.utils import load_lens_file
R="/workspace/o-lens-improvement"
for name,path,model in [("J-lens 2B (released)","artifacts/baselines/jlens_prefit.pt","Qwen/Qwen3.5-2B"),("J++ 2B (released)","artifacts/baselines/jpp_prefit.pt","Qwen/Qwen3.5-2B"),
                        ("J-lens 9B (ours)","runs/v6_9b/jlens9b.pt","Qwen/Qwen3.5-9B"),("J++ 9B (released)","artifacts/baselines/jpp9b_prefit.pt","Qwen/Qwen3.5-9B")]:
    L=load_lens_file(f"{R}/{path}",hf_model_name=model).jacobians_L_dict_FN
    row=[]
    for l in sorted(int(k) for k in L):
        J=L[l if l in L else str(l)].float(); d=J.shape[0]; c=float(torch.trace(J))/d
        row.append(f"L{l}:{c:+.2f}")
    print(name.ljust(22)," ".join(row))
