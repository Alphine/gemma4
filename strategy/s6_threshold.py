"""S6 (rough, ESTIMATE): where will the rank-10 public score sit at the end? Uses the 2026-10-08 LB snapshot (2115 teams, BestScore + SubmissionCount).
Model: each team has true mean mu_t = C*Beta(a,b) (in tasks of 58; C = reachable ceiling), each submission = round(mu_t + sd*z) (sd 1.5, floor 0, cap 58); best of n_t draws observed.
Fit (a,theta) to the observed histogram by max multinomial likelihood on a coarse grid; then add more draws: each team gets extra = f * n_t (f=1,2,4); teams do NOT improve (lower bound on drift).
Hard caveat: teams do improve, and top teams (>=0.20) have likely better configs than gamma tail implies."""
import numpy as np, csv, zipfile, io, json
rng=np.random.default_rng(5)
SD=1.5
z=zipfile.ZipFile('lb_snapshot_2026-10-08.zip'); rows=list(csv.DictReader(io.TextIOWrapper(z.open(z.namelist()[0]),encoding='utf-8-sig')))
n=np.array([int(r['SubmissionCount']) for r in rows]); sc=np.array([float(r['Score']) for r in rows])
# k from floor-rounded score: smallest k with floor(100k/58)/100==score
kmap={}
for k in range(0,59):
    kmap.setdefault(int(np.floor(100*k/58+1e-9)),k)
kobs=np.array([kmap[int(round(s*100))] for s in sc])
T=len(n)
bins=np.arange(0,16)
obs_h=np.array([(np.minimum(kobs,15)==b).sum() for b in bins])
def simulate(a,th,extra_f=0.0,reps=1,bb=None):
    out=[]
    for _ in range(reps):
        mu=th*rng.beta(a,bb,T)
        nn=np.maximum(1,np.round(n*(1+extra_f)).astype(int))
        best=np.zeros(T,int)
        for j in range(nn.max()):
            act=nn>j
            kk=np.clip(np.round(mu+SD*rng.normal(size=T)),0,58).astype(int)
            best=np.where(act,np.maximum(best,kk),best)
        out.append(best)
    return out
def ll(a,bb,C):
    sims=simulate(a,C,0,3,bb)
    h=np.mean([[ (np.minimum(b,15)==x).sum() for x in bins] for b in sims],axis=0)+0.5
    p=h/h.sum()
    return (obs_h*np.log(p)).sum()
best=None
for SD_ in [0.8,1.1,1.5]:
  SD=SD_
  for C in [13,15,18,22]:
    for a in [2,3,4,6,8]:
        for mean in [0.2,0.25,0.3,0.35,0.4]:
            bb=a*(1-mean)/mean
            l=ll(a,bb,C)
            if best is None or l>best[0]: best=(l,a,bb,C,mean,SD_)
print('fit',best)
_,a,bb,C,mean,SD=best
res={'fit':dict(a=a,b=bb,C=C,mean_mu=C*mean,run_sd=SD)}
sim0=simulate(a,C,0,3,bb)[0]
res['check_hist_obs']=obs_h.tolist(); res['check_hist_sim']=[int((np.minimum(sim0,15)==x).sum()) for x in bins]
print(res)
for f in [0,1,2,4]:
    r10=[];n12=[];n14=[]
    for b in simulate(a,C,f,20,bb):
        s_=np.sort(b)[::-1]; r10.append(s_[9]); n12.append((b>=12).sum()); n14.append((b>=14).sum())
    res[f'extra_{f}x']=dict(rank10_k_mean=float(np.mean(r10)),rank10_k_p10_p90=[float(np.percentile(r10,10)),float(np.percentile(r10,90))],
        n_teams_ge12=float(np.mean(n12)),n_teams_ge14=float(np.mean(n14)))
    print(f,res[f'extra_{f}x'])
json.dump(res,open('s6_threshold.json','w'),indent=1)
print('observed: rank10 k=',np.sort(kobs)[::-1][9],'n>=12',(kobs>=12).sum(),'n>=11',(kobs>=11).sum(),'n>=10',(kobs>=10).sum())
