"""a2 diagnostic only (CPU): after the appended '\n\n' every target's top-1 is <think>. Does the rest of the
distribution still carry the topic? Masked top-1 excluding special tokens, after vs before (test 1000)."""
import sys
sys.argv = ["x", "diag"]
exec(open("/workspace/o-lens-improvement/scripts/a2_ytop.py").read().split('if MODE == "before":')[0])
A = torch.load(f"{R}/.hf27/m27/acts_27b.pt"); Y0 = A["Y"]; del A
Y1 = torch.load(f"{OUT}/Y_ws.pt")["Y"]
sp = torch.zeros(V, dtype=torch.bool); sp[[i for i in range(V) if strs[i].startswith("<") and strs[i].endswith(">")]] = True
m2 = mask | sp
th = tok.convert_tokens_to_ids("<think>")
for k in range(3):
    l0, l1 = logits(Y0[k][te]), logits(Y1[k][te])
    p1 = l1.softmax(-1)
    w0 = l0.masked_fill(m2, float("-inf")).argmax(-1)
    t1 = l1.masked_fill(m2, float("-inf")).topk(10, -1).indices
    n = lambda t: [s.strip().lower() for s in (strs[i] for i in t.tolist())]
    same1 = sum(a == b for a, b in zip(n(w0), n(t1[:, 0]))) / 1000
    in10 = sum(a in set(n(r)) for a, r in zip(n(w0), t1)) / 1000
    print(f"E{k}: after p(<think>) mean {p1[:, th].mean():.3f}; non-special masked word after == before {same1:.3f}; "
          f"before-word in after top-10 {in10:.3f}")
    print("   after non-special masked top-1 most common:", Counter(n(t1[:, 0])).most_common(15))
    print("   examples:", [(n(w0[j:j+1])[0], n(t1[j, :3])) for j in range(8)])
