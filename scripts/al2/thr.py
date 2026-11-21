import json,re,random
exec(open('/root/al2/words.py').read().split('keep, seen = [], set()')[0])
for th in (3,2,1):
    k=[];s=set()
    for w in strict:
        if en[w.lower()]>=th and w.lower() not in s: s.add(w.lower()); k.append(w)
    print(th,len(k), random.Random(1).sample([w for w in k if en[w.lower()]<3],min(40,len(k))) if th<3 else "")
