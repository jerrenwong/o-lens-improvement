import torch, random
B="/workspace/o-lens-improvement/.hf27"
meta=torch.load(f"{B}/al2/meta.pt",weights_only=False)
d=torch.load(f"{B}/mp1/ranks.pt",weights_only=False); TR=d["TR"]; te=d["te_rows"].tolist()
words=[meta["words"][i] for i in te]
L=[4,8,12,16,20,24,28]
def a7(n): return (torch.stack([TR[n][l] for l in L]).min(0).values<=10).float()
uw=sorted(set(words)); wid=torch.tensor([uw.index(w) for w in words])
def boot(x,y,n=2000):
    dd=x-y; per=torch.zeros(len(uw)).index_add_(0,wid,dd); cnt=torch.zeros(len(uw)).index_add_(0,wid,torch.ones(len(dd)))
    rng=torch.Generator().manual_seed(0); out=[]
    for _ in range(n):
        s=torch.randint(len(uw),(len(uw),),generator=rng); out.append(float(per[s].sum()/cnt[s].sum()))
    out.sort(); return round(float(dd.mean()),3), round(out[int(.025*n)],3), round(out[int(.975*n)],3)
for a,b in [("Jrel|mean","Jrel|last"),("Jgen|mean","Jgen|last"),("Jgen|half","Jgen|last"),("Jgen|half","Jgen|mean"),("M_mean","M_all"),("M_half","M_all"),
            ("mix|Jgen+M_mean (mean)","mix|Jgen+M_all (last)"),("mix|Jgen(last)+M_mean(mean)","mix|Jgen+M_all (last)"),("mix|Jgen+M_mean (mean)","Jgen|mean"),("mix|Jgen+M_mean_shuf (mean)","Jgen|mean")]:
    print(f"{a:32s} - {b:26s}", boot(a7(a),a7(b)))
# per-layer differences
for a,b in [("Jgen|mean","Jgen|last"),("M_mean","M_all")]:
    print(a,b,[round(float((TR[a][l]<=10).float().mean()-(TR[b][l]<=10).float().mean()),3) for l in L])
