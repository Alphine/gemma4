"""S5: value of information. Detect a true improvement of +0.04 final score (= +2.3 tasks on 58) with (a) paired dev runs on the 43 LB-faithful dev tasks, (b) LB slots.
Assumptions (estimated): live share of dev pool 43/129 => +2.3 tasks of 58 public ~= +5.1 tasks on the 43 dev tasks IF the gain is on already-solvable tasks
(gains that unlock dead tasks, e.g. env-fix, are invisible on the 40 gold-control-solvable dev tasks).
Dev arms: p~Beta(m*kap,(1-m)*kap), m=0.30 (13/40 +free), kap=0.47 (fit); new arm = logit-shift to +d tasks, plus config jitter 0.5 logit sd. Test: one-sided t on per-task mean diff, alpha 0.05."""
import numpy as np, json
from scipy.stats import t as tdist, norm
rng=np.random.default_rng(2024)
def sig(x): return 1/(1+np.exp(-x))
def shift_to(logit,target):
    lo,hi=-20.,20.
    for _ in range(60):
        c=(lo+hi)/2
        if sig(logit+c).sum()>target: hi=c
        else: lo=c
    return sig(logit+(lo+hi)/2)
def dev_power(d,r,kap=0.47,m=0.30,n=43,R=4000,jit=0.5):
    hits=0; sdiffs=[]
    for _ in range(R):
        b=rng.beta(m*kap,(1-m)*kap,n).clip(1e-6,1-1e-6)
        lg=np.log(b/(1-b))
        p0=b
        p1=shift_to(lg+rng.normal(0,jit,n),b.sum()+d)
        x0=(rng.random((r,n))<p0).mean(0); x1=(rng.random((r,n))<p1).mean(0)
        dd=x1-x0
        se=dd.std(ddof=1)/np.sqrt(n)
        if se==0: continue
        tt=dd.mean()/se
        hits+= tt>tdist.ppf(0.95,n-1)
        sdiffs.append(dd.sum())
    return hits/R, float(np.std(sdiffs))
out={'dev':{}}
for d in [0,2.5,5.1]:
    for r in [1,2,3]:
        pw,sd=dev_power(d,r)
        out['dev'][f'd={d}|r={r}']=dict(power_or_alpha=pw,sd_total_diff=sd)
# LB slots: Gaussian approx. sd_run in tasks scenarios
lb={}
for sdr in [1.1,1.5,2.0]:
    for n in [1,2,3,5]:
        sd_d=sdr*np.sqrt(2/n)   # n draws per config, difference of means
        for eff in [2.3]:
            lb[f'sd_run={sdr}|n_each={n}']=dict(power_alpha05_onesided=float(1-norm.cdf(1.645-eff/sd_d)),se_diff_tasks=float(sd_d))
out['lb']=lb
json.dump(out,open('s5_voi.json','w'),indent=1)
for k,v in out['dev'].items(): print('dev',k,v)
for k,v in lb.items(): print('lb',k,v)
