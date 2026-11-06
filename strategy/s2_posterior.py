"""S2: hierarchical posterior for config means from LB observations. k_cj ~ N(mu_c, sig^2); mu_c ~ N(mu0, tau^2). Grid over (mu0,tau,sig), flat priors:
mu0 U(0,12) tau U(0,3) sig U(0.5,2.3)  (sig upper bound = binomial SD 2.2). Gaussian approx to the discrete k. Fixed seed."""
import numpy as np, json
rng=np.random.default_rng(7)
def run(data,label):
    names=list(data); y=[np.array(data[n],float) for n in names]
    mu0g=np.linspace(0,12,121); tg=np.linspace(0,3,31); sg=np.linspace(0.5,2.3,19)
    LL=np.full((121,31,19),-np.inf)
    for i,m0 in enumerate(mu0g):
        for j,t in enumerate(tg):
            for k,s in enumerate(sg):
                ll=0
                for yc in y:
                    n=len(yc); yb=yc.mean(); ss=((yc-yb)**2).sum()
                    v=t*t+s*s/n
                    ll+=-0.5*np.log(v)-0.5*(yb-m0)**2/v - (n-1)*np.log(s) - 0.5*ss/s**2
                LL[i,j,k]=ll
    P=np.exp(LL-LL.max()); P/=P.sum()
    idx=rng.choice(P.size,size=20000,p=P.ravel())
    ii,jj,kk=np.unravel_index(idx,P.shape)
    m0=mu0g[ii]; t=tg[jj]; s=sg[kk]
    mus={}
    for n,yc in zip(names,y):
        nn=len(yc); prec=1/np.maximum(t,1e-9)**2+nn/s**2
        pm=(m0/np.maximum(t,1e-9)**2+yc.sum()/s**2)/prec
        pm=np.where(t<1e-6,m0,pm); sd=np.where(t<1e-6,0,1/np.sqrt(prec))
        mus[n]=pm+sd*rng.normal(size=len(t))
    res={'label':label,'mu0_mean':float(m0.mean()),'mu0_ci':[float(np.percentile(m0,2.5)),float(np.percentile(m0,97.5))],
         'tau_mean':float(t.mean()),'tau_ci':[float(np.percentile(t,2.5)),float(np.percentile(t,97.5))],
         'sig_mean':float(s.mean()),'sig_ci':[float(np.percentile(s,2.5)),float(np.percentile(s,97.5))],
         'P_tau_lt_0.5':float((t<0.5).mean()),'configs':{}}
    for n in names:
        res['configs'][n]=dict(mean_tasks=float(mus[n].mean()),ci_tasks=[float(np.percentile(mus[n],2.5)),float(np.percentile(mus[n],97.5))],
                               mean_score=float(mus[n].mean()/58),raw=data[n])
    ee=mus['edit-early']
    others=np.mean([mus[n] for n in names if n not in('edit-early','ZIP_other_team')],axis=0)
    res['ee_minus_other_mean_tasks']=float((ee-others).mean())
    res['ee_minus_other_ci']=[float(np.percentile(ee-others,2.5)),float(np.percentile(ee-others,97.5))]
    res['P_ee_gt_others_mean']=float((ee>others).mean())
    res['P_ee_ge_others_plus_1']=float((ee-others>=1).mean())
    res['P_ee_gt_v4']=float((ee>mus['edit-early-v4']).mean())
    res['P_ee_gt_v2']=float((ee>mus['edit-early-v2']).mean())
    # posterior predictive for next submission of a NEW config drawn from the population (tau-spread), and of edit-early
    pred_new=m0+t*rng.normal(size=len(t)); 
    res['new_config_true_mean_tasks']=[float(pred_new.mean()),float(np.percentile(pred_new,5)),float(np.percentile(pred_new,95))]
    res['P_new_config_true_ge_7_tasks(0.12)']=float((pred_new>=7).mean())
    res['P_new_config_true_ge_8.1_tasks(0.14)']=float((pred_new>=8.12).mean())
    return res
six={'robust-baseline':[4],'ladder-1b':[5],'triage-first':[5],'edit-early':[7],'edit-early-v2':[5],'edit-early-v4':[5]}
out={'six_ours':run(six,'six new-harness configs, one draw each')}
z=dict(six); z['ZIP_other_team']=[7,6,4]
out['six_plus_zip_replicates']=run(z,'six + third-party identical ZIP seen as 7,6,4 (reported)')
ten=dict(six); ten.update({'locate-first-v1':[5],'locate-v2':[3],'nog-5':[6],'harness-optimized':[4]})
out['ten_ours']=run(ten,'all ten scored submissions, old harness included')
json.dump(out,open('s2_posterior.json','w'),indent=1)
for k,v in out.items():
    print('\n',k,v['label']); 
    print(' mu0 %.2f %s tau %.2f %s sig %.2f %s P(tau<.5)=%.2f'%(v['mu0_mean'],np.round(v['mu0_ci'],2),v['tau_mean'],np.round(v['tau_ci'],2),v['sig_mean'],np.round(v['sig_ci'],2),v['P_tau_lt_0.5']))
    for n,c in v['configs'].items(): print('  %-18s raw %s post-mean %.2f tasks (%.3f) CI %s'%(n,c['raw'],c['mean_tasks'],c['mean_score'],np.round(c['ci_tasks'],1)))
    print(' ee - mean(others): %.2f CI %s  P>0 %.2f  P>=+1 %.2f  P(ee>v4)=%.2f P(ee>v2)=%.2f'%(v['ee_minus_other_mean_tasks'],np.round(v['ee_minus_other_ci'],2),v['P_ee_gt_others_mean'],v['P_ee_ge_others_plus_1'],v['P_ee_gt_v4'],v['P_ee_gt_v2']))
    print(' new cfg true mean tasks',np.round(v['new_config_true_mean_tasks'],2),' P(>=7)=%.2f P(>=8.1)=%.2f'%(v['P_new_config_true_ge_7_tasks(0.12)'],v['P_new_config_true_ge_8.1_tasks(0.14)']))
