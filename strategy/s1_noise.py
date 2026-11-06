"""S1: run-level vs task-level variance. Fixed seeds. Pure numpy/scipy, CPU.
Data: (a) lora-ab1 paired 144 synthetic held-out tasks (base edit-early-v4 vs adapter): 63 both, 7 base-only, 9 adapter-only, 65 neither
      (b) real dev 43 tasks LB-faithful: triage-first vs edit-early-v4: 11 both, 4+2 discordant, 26 neither
      (c) LB observed k series.
Cross-arm discordance D_cross = D_same + 2E[d^2] >= D_same, so the fit below is an UPPER bound on same-arm re-run noise.
"""
import numpy as np, json
from scipy.special import betaln, gammaln
from scipy.optimize import minimize
rng=np.random.default_rng(0)

def bb_pair_probs(m,kap):
    a,b=m*kap,(1-m)*kap
    p11=np.exp(betaln(a+2,b)-betaln(a,b))        # E[p^2]
    p10=np.exp(betaln(a+1,b+1)-betaln(a,b))      # E[p(1-p)]
    p00=np.exp(betaln(a,b+2)-betaln(a,b))
    return p11,p10,p00
def nll(par,n11,nd,n00):
    m=1/(1+np.exp(-par[0])); kap=np.exp(par[1])
    p11,p10,p00=bb_pair_probs(m,kap)
    return -(n11*np.log(p11)+nd*np.log(2*p10)+n00*np.log(p00))
def fit(n11,nd,n00):
    best=None
    for k0 in [-3,-1,0,1,3]:
        r=minimize(nll,[0,k0],args=(n11,nd,n00),method='Nelder-Mead')
        if best is None or r.fun<best.fun: best=r
    m=1/(1+np.exp(-best.x[0])); kap=np.exp(best.x[1])
    # profile CI for log kappa (likelihood ratio 3.84/2)
    grid=np.exp(np.linspace(-5,6,400)); prof=[]
    for k in grid:
        r=minimize(lambda x:nll([x[0],np.log(k)],n11,nd,n00),[best.x[0]],method='Nelder-Mead'); prof.append(r.fun)
    prof=np.array(prof); ok=grid[prof<=best.fun+1.92]
    return dict(m=m,kappa=kap,kappa_ci=[float(ok.min()),float(ok.max())],rho=kap/(kap+1),
                rho_ci=[float(ok.min()/(ok.min()+1)),float(ok.max()/(ok.max()+1))])
out={}
# (a)
out['lora_ab1_144']=dict(n11=63,nd=16,n00=65,D=16/144,**fit(63,16,65))
out['real_dev_43']=dict(n11=11,nd=6,n00=26,D=6/43,**fit(11,6,26))
# (b) D bootstrap CI
for k,(n,d) in {'lora':(144,16),'dev':(43,6)}.items():
    s=rng.binomial(n,d/n,20000)/n
    out[f'D_ci95_{k}']=[float(np.percentile(s,2.5)),float(np.percentile(s,97.5))]
# (c) LB series
newh={'robust-baseline':4,'ladder-1b':5,'triage-first':5,'edit-early':7,'edit-early-v2':5,'edit-early-v4':5}
old={'locate-first-v1':5,'locate-v2':3,'nog-5':6,'harness-optimized':4,'ladder-2':None}
allk=[v for v in list(newh.values())+[x for x in old.values() if x is not None]]
def chk(ks,pbar=None):
    ks=np.array(ks,float); n=len(ks); s2=ks.var(ddof=1); p=ks.mean()/58
    binv=58*p*(1-p)
    from scipy.stats import chi2
    return dict(n=n,mean=ks.mean(),sd=ks.std(ddof=1),binom_sd=binv**.5,chi2=(n-1)*s2/binv,p_lower=float(chi2.cdf((n-1)*s2/binv,n-1)),
                sd_upper95=float(np.sqrt((n-1)*s2/chi2.ppf(0.05,n-1))),sd_ratio_var=s2/binv)
out['lb_new_harness_6']=chk(list(newh.values()))
out['lb_all_9']=chk(allk)
out['same_zip_reports_7_6_4']=chk([7,6,4])
# implied LB variance vs rho for typical mean
res={}
for mu in [5,7,9]:
    res[mu]={rho:round(float(np.sqrt(58*(mu/58)*(1-mu/58)*rho)),2) for rho in [0.1,0.25,0.5,0.75,1.0]}
out['sd_tasks_by_mu_rho']=res
json.dump(out,open('s1_noise.json','w'),indent=1)
print(json.dumps(out,indent=1))
