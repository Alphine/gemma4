"""Sensitivity + tool-call-cap (K) analysis. python extras.py  -> extras.json (uses model.py functions)"""
import json, os, numpy as np
import model as M
rng = np.random.default_rng(1)
runs = M.load(); solv = [r for r in runs if not r["free"]]
hR, hU, rc = M.hazards(runs)
out = {}
# calls of resolved runs
nR = sorted(r["n"] for r in runs if r["kind"] == "R")
out["calls_of_resolved_runs"] = dict(n=len(nR), q=[int(np.percentile(nR, p)) for p in (10, 25, 50, 75, 90, 100)])
out["empty_patch_timeouts"] = None
# (1) fine grid: largest cap with P(>12h) <= 1% / 5% under scenarios (K=60, alpha=0.5)
pre = []
for b in range(300):
    bs = M.boot_tasks(solv, rng); pre.append((bs,) + M.hazards(bs))
def p_over(C, K, alpha, q, pace, s, F, N=120, sims=1500):
    over = 0
    for _ in range(sims):
        bs, h1, h2, r_ = pre[rng.integers(0, len(pre))]
        t, _ = M.sample_times(bs, C, K, alpha, h1, h2, r_, N, rng, q_timeout=q, pace=pace)
        over += (F + N * s + t.sum()) > 720
    return over / sims
grid = [5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 7.0, 8.0]
sens = {}
for (sname, q) in (("dev_like", None), ("q70", 0.7), ("q85", 0.85), ("q100", 1.0)):
    for s in (0.13, 0.3, 0.5):
        for F in (8.3, 13.0):
            sens[f"{sname}_s{s}_F{F}"] = {str(C): p_over(C, 60, 0.5, q, 1.0, s, F) for C in grid}
out["p_over12_N120_alpha0.5_K60"] = sens
# (2) K as governor at C = 7, 8: resolve (solvable, alpha .5) and hours
K_res = {}
for C in (6.0, 7.0, 8.0, 10.0):
    for K in (30, 40, 45, 60, 100):
        r, t = M.expected(solv, C, K, 0.5, hR, hU, rc)
        mean_h = []
        for _ in range(800):
            bs, h1, h2, r_ = pre[rng.integers(0, len(pre))]
            tt, _ = M.sample_times(bs, C, K, 0.5, h1, h2, r_, 120, rng, q_timeout=None, pace=1.0)
            mean_h.append((10.5 + 120 * 0.3 + tt.sum()) / 60)
        K_res[f"C{C}_K{K}"] = dict(p_resolve=float(r.mean()), mean_agent_min=float(t.mean()), wall_h_dev_like=float(np.mean(mean_h)))
out["K_governor"] = K_res
# (3) share of runs with empty patch at timeout, stall structure
to = [r for r in runs if r["kind"] == "T"]
out["timeouts"] = dict(n=len(to), resolved_at_cap=sum(r["res"] for r in to))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "extras.json"), "w"), indent=1)
print(json.dumps(out["calls_of_resolved_runs"]))
for k, v in sens.items(): print(k, v)
for k, v in K_res.items(): print(k, v)
