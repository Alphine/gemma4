"""Summarize results.json of pallas_tpu_suite.py. Works for v2 (single native verdict) and v3 (native_tpu = verdict,
native_tpu_default / native_tpu_highest = per-precision runs).  usage: summarize.py results.json"""
import json, sys, collections
d = json.load(open(sys.argv[1])); rs = d['results']
def c(x):
    s = x.get('status')
    return 'PASS' if s == 'ran' and x['passed'] else 'FAIL' if s == 'ran' else 'ERR' if s == 'error' else s
by = collections.defaultdict(collections.Counter)
for r in rs:
    by[r['operator']][(c(r['interpret_cpu']), c(r['native_tpu']))] += 1
for op, cn in sorted(by.items()): print(op, dict(cn))
agree = sum(1 for r in rs if c(r['interpret_cpu']) == c(r['native_tpu'])); print('agree', agree, '/', len(rs))
print('hardware-only (interpret PASS, native not PASS):')
for r in rs:
    if c(r['interpret_cpu']) == 'PASS' and c(r['native_tpu']) in ('FAIL', 'ERR'):
        n = r['native_tpu']; print(' ', r['id'], r['operator'], c(n), n.get('err_class', ''), 'rel=%s' % n.get('rel_err'))
if any('native_tpu_default' in r for r in rs):
    print('precision used for verdict:', dict(collections.Counter(r.get('native_precision') for r in rs)))
    print('dot families, default vs HIGHEST (rel_err):')
    for r in rs:
        if 'native_tpu_highest' in r:
            dd, hh = r['native_tpu_default'], r['native_tpu_highest']
            print(' ', r['id'].ljust(22), 'default', c(dd), dd.get('rel_err'), 'tol', dd.get('tol'), '| highest', c(hh), hh.get('rel_err'))
