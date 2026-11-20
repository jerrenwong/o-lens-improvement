import csv,sys
from collections import defaultdict
def pk(path,k):
    best={}
    for r in csv.DictReader(open(path)):
        key=(r['eval'],r['item'],r['intermediate']); rk=int(r['rank'])
        best[key]=min(best.get(key,rk),rk)
    ih=defaultdict(list)
    for (e,i,_),rk in best.items(): ih[(e,i)].append(rk<=k)
    pt=defaultdict(list)
    for (e,_),h in ih.items(): pt[e].append(sum(h)/len(h))
    m={e:sum(v)/len(v) for e,v in sorted(pt.items())}; n={e:len(v) for e,v in pt.items()}
    return sum(m.values())/len(m),m,n
for d in sys.argv[1:]:
    for k in (10,1):
        a,m,n=pk(d+'/readout_filtered.csv',k)
        print(d.split('/')[-1],f"p@{k}",f"{a:.3f}"," ".join(f"{e[:5]} {v:.3f}(n={n[e]})" for e,v in m.items()))
