"""Turn results.json into the report tables (tables.json + stdout markdown). python tables.py"""
import json, numpy as np
d = json.load(open('results.json'))
A = d['P_resolve_solvable']; D = d['delta_vs_5.5_K60']; W = d['wallclock']
caps = [5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
UNIT = 58 * 40 / 129        # LB tasks per unit of P(resolve | solvable), dev-mix assumption: 18.0
FREE = 58 * 3 / 129 * 2 / 3  # free tasks, measured resolve share 2/3 held constant: 0.9
out = {}
print('| cap | alpha=0 (no tail) | alpha=0.5 | alpha=1 |')
for C in caps:
    row = []
    for al in (0.0, 0.5, 1.0):
        a = A[f'K60_a{al}_C{C}']
        row.append('%.1f [%.1f, %.1f]' % (UNIT * a['p'] + FREE, UNIT * a['lo'] + FREE, UNIT * a['hi'] + FREE))
    out[C] = row
    print('| %s | %s |' % (C, ' | '.join(row)))
print('\nPaired gain vs 5.5 in LB tasks (K=60/100), alpha 0.5 and 1')
for C in caps:
    if C == 5.5: continue
    r = []
    for al in (0.5, 1.0):
        for K in (60, 100):
            x = D[f'K{K}_a{al}_C{C}']
            r.append('%+.2f [%+.2f,%+.2f]' % (UNIT * x['d'], UNIT * x['lo'], UNIT * x['hi']))
    print('| %s | %s |' % (C, ' | '.join(r)))
print('\nwall-clock N=120, K=60, alpha=.5, mean h / P(>12h)  (F U(8,13) min, setup U(0.13,0.5) min/task)')
for sc in ('dev_like', 'q70', 'q85', 'q100'):
    print('| %s | %s |' % (sc, ' | '.join('%s: %.1f/%.2f' % (C, W[f'N120_{sc}_pace1.0_a0.5_K60_C{C}']['mean_h'], W[f'N120_{sc}_pace1.0_a0.5_K60_C{C}']['p_over12']) for C in caps)))
print('\nN=58 dev_like pace1 alpha .5 K60')
print(' | '.join('%s: %.1f/%.2f' % (C, W[f'N58_dev_like_pace1.0_a0.5_K60_C{C}']['mean_h'], W[f'N58_dev_like_pace1.0_a0.5_K60_C{C}']['p_over12']) for C in caps))
json.dump(out, open('tables.json', 'w'), indent=1)
