import sys, torch
sys.path.insert(0, "/workspace/o-lens-improvement/scripts/mlp1")
import mlp1_train as T
d=16
m=dict(mx=torch.randn(d), P=torch.eye(d), Wr=torch.randn(d,d)*1e-3, my=torch.randn(d)*1e-2)
for k in "BCDLPQ":
    net=T.Net(k,m,torch.ones(d),8,0.1,0.5)
    h=torch.randn(4,d); out=net(h); out.sum().backward()
    print(k, out.shape, [n for n,p in net.named_parameters() if p.grad is not None], [n for n,_ in net.named_buffers()])
    if k=="P": print("P init equals ridge:", torch.allclose(net.eval()(h), ((h-m["mx"])@m["P"])@m["Wr"]+m["my"]))
    if k in "CLQ": print(k, "init equals ridge:", torch.allclose(net.eval()(h), ((h-m["mx"])@m["P"])@m["Wr"]+m["my"]))
