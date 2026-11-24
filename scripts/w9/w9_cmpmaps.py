"""w9: compare two map files layer by layer (max |W diff| relative, my, c). python w9_cmpmaps.py <a.pt> <b.pt>"""
import sys

import torch

A, B = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
for l in A:
    wa, wb = A[l]["W"].float() * A[l]["scale"], B[l]["W"].float() * B[l]["scale"]
    print(f"L{l}: c {A[l].get('c')} vs {B[l].get('c')}; rel max|dW| {((wa - wb).abs().max() / wa.abs().max()).item():.2e}; "
          f"max|d my| {(A[l]['my'] - B[l]['my']).abs().max().item():.2e}; cos(W) "
          f"{torch.cosine_similarity(wa.flatten(), wb.flatten(), 0).item():.6f}")
