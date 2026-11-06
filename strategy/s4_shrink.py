"""S4: how much of a lucky public draw transfers to the private half?
Population 120 tasks (assumption: ~120 total, public 58 / private 62), each task live w.p. lam=0.35 (dev: 43/129=0.33), live p~Beta(q*kap,(1-q)*kap), kap=0.5,
q = s_all/lam. Config overall mean s_all ~ prior. Public/private split random. One public run (N=1) or best-of-N public runs of the same config.
Report E[private tasks (of 62) | public k >= threshold]."""
import numpy as np, json
rng=np.random.default_rng(99)
def world(M,prior,lam=0.35,kap=0.5):
    if prior=='current': s=np.clip(rng.normal(0.09,0.015,M),0.03,0.3)
    elif prior=='wide': s=rng.uniform(0.06,0.20,M)
    else: s=np.full(M,prior)
    q=np.clip(s/lam,0.01,0.97)
    live=rng.random((M,120))<lam
    b=rng.beta((q*kap)[:,None]*np.ones((M,120)),((1-q)*kap)[:,None]*np.ones((M,120))).clip(0,1)
    p=np.where(live,b,0.0)
    perm=np.argsort(rng.random((M,120)),axis=1)
    p=np.take_along_axis(p,perm,1)
    return s,p[:,:58],p[:,58:]
out={}
M=60000
for prior in ['current','wide',0.10,0.12,0.14,0.16]:
    s,pp,pv=world(M,prior)
    res={}
    for N in [1,10]:
        k=(rng.random((M,N,58))<pp[:,None,:]).sum(2).max(1)
        kv=(rng.random((M,62))<pv).sum(1)
        for th in [10,11,12,13]:
            sel=k>=th
            if sel.sum()<50: continue
            res[f'N{N}_pub>={th}']=dict(frac=float(sel.mean()),E_s_all=float(s[sel].mean()),E_pub_k=float(k[sel].mean()),
                E_priv_k=float(kv[sel].mean()),E_priv_score=float(kv[sel].mean()/62),E_pub_mu=float(pp[sel].sum(1).mean()))
        res[f'N{N}_all']=dict(E_s_all=float(s.mean()),E_pub_k=float(k.mean()),E_priv_k=float(kv.mean()),E_priv_score=float(kv.mean()/62))
    # task-sampling spread: sd of expected public-set mean around 58*s_all
    res['sd_mu_pub_minus_58s']=float((pp.sum(1)-58*s).std())
    res['sd_priv_k_given_s']=float((rng.random((M,62))<pv).sum(1).std())
    out[str(prior)]=res
json.dump(out,open('s4_shrink.json','w'),indent=1)
for pr,res in out.items():
    print('\nprior',pr,' sd(mu_pub-58 s_all)=%.2f'%res['sd_mu_pub_minus_58s'])
    for k,v in res.items():
        if isinstance(v,dict): print(' ',k,{a:round(b,3) for a,b in v.items()})
