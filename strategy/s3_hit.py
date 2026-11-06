"""S3: P(at least one of N submissions scores >= 12/58) as function of the config's true public-set mean score s (= sum p_i / 58).
Public task set is FIXED across submissions, so only run noise (+ config-to-config p-vector perturbation) matters.
Scenarios (live-pool L tasks with p~Beta(q*kap,(1-q)*kap), rest p=0; p-vector logit-shifted so sum p_i = 58*s exactly):
  A  L=20 kap=0.5   (dev-fit: real dev kappa~0.47, CI 0.16-1.4; live share ~ 40/129*58 = 18 public tasks)
  B  L=30 kap=1.0   (more reachable tasks, more per-task randomness)
  C  L=58 kap=inf   (every task Bernoulli(s): the binomial worst-case; what ceiling.py effectively assumes)
Config mode: 'same' = identical p-vector each submission (pure resubmission); 'sib'= each submission is a sibling config: logit noise sd 0.7 on live p_i,
 recalibrated to the same mean; 'fresh' = p-vector redrawn from the prior each submission (unrelated configs; upper bound on spread).
"""
import numpy as np, json
rng=np.random.default_rng(12345)
W=1500; NMAX=55; T=58
def sig(x): return 1/(1+np.exp(-x))
def calibrate(logit,target,mask):
    # per-row shift c so that sum(mask*sig(logit+c)) = target
    lo=np.full(logit.shape[0],-30.); hi=np.full(logit.shape[0],30.)
    for _ in range(28):
        c=(lo+hi)/2
        sm=(sig(logit+c[:,None])*mask).sum(1)
        big=sm>target
        hi=np.where(big,c,hi); lo=np.where(big,lo,c)
    return sig(logit+((lo+hi)/2)[:,None])*mask
def draw_p(L,kap,mu,rows):
    mask=np.zeros((rows,T)); mask[:,:L]=1
    if kap==np.inf:
        return np.full((rows,T),mu/T)
    q=min(mu/L,0.97)
    b=rng.beta(q*kap,(1-q)*kap,(rows,T)).clip(1e-6,1-1e-6)
    return calibrate(np.log(b/(1-b)),mu,mask)
def sim(s,L,kap,mode,N):
    mu=58*s
    base=draw_p(L,kap,mu,W)
    ks=np.empty((W,N),int)
    for j in range(N):
        if mode=='same' or kap==np.inf: p=base
        elif mode=='fresh': p=draw_p(L,kap,mu,W)
        else:
            lg=np.log(base.clip(1e-9,1-1e-9)/(1-base.clip(1e-9,1-1e-9)))+rng.normal(0,0.7,base.shape)
            mask=(base>0).astype(float); p=calibrate(lg,mu,mask)
        ks[:,j]=(rng.random((W,T))<p).sum(1)
    return ks
SC={'A':(20,0.5),'B':(30,1.0),'C':(58,np.inf)}
Ns=[1,3,5,10,20,30,55]
out={}
svals=[0.08,0.10,0.12,0.14,0.16,0.18,0.20]
for name,(L,kap) in SC.items():
    for mode in (['same','sib','fresh'] if kap!=np.inf else ['same']):
        for s in svals:
            ks=sim(s,L,kap,mode,NMAX)
            rec={'sd_single':float(ks[:,0].std()),'mean':float(ks.mean())}
            for N in Ns:
                rec[f'P12_N{N}']=float((ks[:,:N].max(1)>=12).mean())
                rec[f'P11_N{N}']=float((ks[:,:N].max(1)>=11).mean())
            rec['P12_single']=float((ks[:,0]>=12).mean())
            out[f"{name}|{mode}|s={s}"]=rec; print("done",name,mode,s,flush=True)
json.dump(out,open('s3_hit.json','w'),indent=1)
# readable
for name in SC:
    for mode in (['same','sib','fresh'] if name!='C' else ['same']):
        print(f'\nScenario {name} mode {mode}: P(max over N >= 12/58)')
        print('s     sd1  '+' '.join(f'N={n:<3}' for n in Ns))
        for s in svals:
            r=out[f'{name}|{mode}|s={s}']
            print(f'{s:.2f} {r["sd_single"]:.2f}  '+' '.join(f'{r[f"P12_N{n}"]:.3f}' for n in Ns))
