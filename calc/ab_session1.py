"""Recompute A/B session-1 numbers from the checkpoint rows (13 complete triples)."""
import json, os, random, statistics as st
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
arms = ['edit-early-v6-locate','edit-early-v7-decide-nt','edit-early-v7-lean-nt']
nm = {arms[0]:'control(v6)', arms[1]:'decide', arms[2]:'lean'}
R = {}
for a in arms:
    R[a] = {}
    for l in open(f'data/ckpt_scrubbed/results_{a}.jsonl'):
        r = json.loads(l)
        if r.get('run_mode')=='real' and not r.get('infra_error'):
            R[a][r['instance_id']] = r
tasks = sorted(set.intersection(*[set(R[a]) for a in arms]), key=lambda t: R[arms[0]][t].get('canon_rank', 0))  # original id order (canon_rank)
print('complete triples', len(tasks), 'rows per arm', {nm[a]:len(R[a]) for a in arms})
def to(r): return r['err_class']=='time_cap' or 'timeout' in str(r.get('agent_error')).lower()
for a in arms:
    rs=[R[a][t] for t in tasks]
    n=len(rs)
    res=sum(r['resolved'] for r in rs); tos=sum(to(r) for r in rs)
    amin=st.mean(r['agent_min'] for r in rs)
    ct=st.mean(r['ct_atif'] for r in rs)
    fe=[r['first_edit_idx'] for r in rs if r['first_edit_idx'] is not None]
    cens=[(r['ct_to_first_edit'] if r['ct_to_first_edit'] is not None else r['ct_atif']) for r in rs]
    pre=[r['pre_edit_read']+r['pre_edit_search'] for r in rs]
    print(nm[a], dict(n=n,resolved=res,timeouts=tos,mean_agent_min=round(amin,2),mean_ct=round(ct),
        n_edit=len(fe),median_first_edit_idx=st.median(fe),mean_cens_ct_to_first_edit=round(st.mean(cens)),
        mean_pre_edit_read_search=round(st.mean(pre),2),max_output_hits=sum(r['call_summary'].get('n_primary_len',0) for r in rs)))
# paired resolved
for x,y in [(1,0),(2,0),(2,1)]:
    A=[R[arms[x]][t]['resolved'] for t in tasks]; B=[R[arms[y]][t]['resolved'] for t in tasks]
    w=sum(1 for a,b in zip(A,B) if a and not b); l=sum(1 for a,b in zip(A,B) if b and not a)
    print(nm[arms[x]],'vs',nm[arms[y]],'only-treated',w,'only-ref',l)
# bootstrap CI for mean censored ct-to-first-edit ratio lean/control, decide/control
rng=random.Random(20261010)
def cens(a,t):
    r=R[a][t]; return r['ct_to_first_edit'] if r['ct_to_first_edit'] is not None else r['ct_atif']
for x in (1,2):
    vals=[]
    for _ in range(10000):
        s=[rng.choice(tasks) for _ in tasks]
        vals.append(sum(cens(arms[x],t) for t in s)/sum(cens(arms[0],t) for t in s)-1)
    vals.sort()
    pt=sum(cens(arms[x],t) for t in tasks)/sum(cens(arms[0],t) for t in tasks)-1
    print(nm[arms[x]],'cens ct_to_first_edit ratio-1',round(pt,3),'CI',round(vals[250],3),round(vals[9749],3))
# per-call completion tokens, latency
for a in arms:
    rs=[R[a][t] for t in tasks]
    print(nm[a],'mean tokens/call',round(sum(r['ct_atif'] for r in rs)/sum(r['n_calls_atif'] for r in rs)),
          'mean latency s',round(st.mean(r['call_summary']['mean_call_latency_s'] for r in rs),2),
          'mean calls',round(st.mean(r['n_calls_atif'] for r in rs),1))
