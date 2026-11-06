"""S7: interpolate the true score s needed for P(max of N draws >= 12/58) = 25/50/75%, from s3_hit.json (linear interpolation in s on the 0.02 grid, same-config mode)."""
import json,numpy as np
d=json.load(open('s3_hit.json')); sv=[0.08,0.10,0.12,0.14,0.16,0.18,0.20]
out={}
for sc in ['A|same','B|same','C|same','A|fresh','B|fresh']:
    for N in [5,10,20,30,55]:
        ps=[d[f'{sc}|s={s}'][f'P12_N{N}'] for s in sv]
        r={}
        for tgt in [0.25,0.5,0.75]:
            r[str(tgt)]=float(np.interp(tgt,ps,sv)) if ps[0]<tgt<=ps[-1] else None
        out[f'{sc}|N={N}']=r
json.dump(out,open('s7_required_s.json','w'),indent=1)
for k,v in out.items(): print(k,{a:(round(b,3) if b else b) for a,b in v.items()})
