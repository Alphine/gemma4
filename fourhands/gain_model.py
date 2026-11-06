"""Expected extra solves / 58 for K concurrent attempts + selector (fourhands).  Pure numpy/scipy, CPU.
Components
  baseline  : single attempt, task solve prob p_i ~ Beta(m*kappa, (1-m)*kappa)  (m = LB-level mean, kappa from research/top10/strategy/s1_noise.json)
  attempts  : K conditionally independent draws given p_i (different seeds), each with
                retention R_time(s) = P(a resolved run still finishes inside wall/s)           (edit_times.json, ckpt)
                retention R_tool(K) = P(resolved run needs <= (60-4)/K counted tool calls)    (ckpt rows)
              attempt mean m_K = m * R_time * R_tool, same kappa (Beta mean shifts, dispersion kept)
  selector  : j of K attempts correct, 0<j<K: picks a correct one with prob a;  j=K: 1;  j=0: 0
Output: table of extra solves (out of 58) = 58*(P_K - m).
"""
import json, glob, sys, os, numpy as np
from scipy.special import betaln
here = os.path.dirname(os.path.abspath(__file__))
ck = os.path.join(here, "..", "..", "..", "datasets", "gemma4-lb-eval-ckpt")
rows = [json.loads(l) for f in glob.glob(os.path.join(ck, "results_*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
sol = [r for r in rows if r["category"] == "solvable"]
res = [r for r in sol if r["resolved"]]
tool_need = np.array([r["tool_calls"] for r in res], float)
et = json.load(open(os.path.join(here, "edit_times.json")))
resolved_t = [(r["last_edit_s"] if r["last_edit_s"] is not None else 0.75 * r["end_s"], r["end_s"]) for r in et if r["resolved"]]
def R_time(s, optimistic=True):
    wall = 330.0
    k = 0 if optimistic else 1
    return float(np.mean([t[k] * s <= wall for t in resolved_t]))
def R_tool(K):
    return float(np.mean(tool_need <= (60 - (0 if K == 1 else 4)) / K))
def pk_beta_binom(m, kappa, K, a):
    """P(selected attempt solves) for Beta(m,kappa) task prob, K iid attempts, selector accuracy a."""
    A, B = m * kappa, (1 - m) * kappa
    tot = 0.0
    from math import comb, exp
    for j in range(1, K + 1):
        pj = comb(K, j) * np.exp(betaln(A + j, B + K - j) - betaln(A, B))
        tot += pj * (1.0 if j == K else a)
    return tot
out = dict(R_tool={K: R_tool(K) for K in (1, 2, 3, 4)},
           R_time_opt={s: R_time(s, True) for s in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)},
           R_time_pess={s: R_time(s, False) for s in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)},
           n_resolved_runs=len(res), n_resolved_traced=len(resolved_t))
table = []
SS = (1.1, 1.2, 1.3, 1.4, 1.6)
for label, m, kappa in [("m=0.10 kappa=0.29 (s1_noise fit)", 6 / 58, 0.29), ("m=0.10 kappa=1.4 (noisy end of CI)", 6 / 58, 1.4),
                        ("m=0.17 kappa=0.29 (top-3 level)", 10 / 58, 0.29)]:
    for tool in ("shared60", "free"):
        for mode in ("opt", "pess"):
            for K in (2, 3, 4):
                for s in SS:
                    mt = R_time(s, mode == "opt")
                    mK = m * mt * (out["R_tool"][K] if tool == "shared60" else 1.0)
                    for a in (0.5, 0.7, 0.9, 1.0):
                        table.append(dict(scenario=label, tool=tool, mode=mode, K=K, s=s, a=a, mK=round(mK, 4),
                                          extra_solves=round(58 * (pk_beta_binom(mK, kappa, K, a) - m), 2)))
out["table"] = table
lines = []
for label in sorted({t["scenario"] for t in table}):
    for tool in ("shared60", "free"):
        for mode in ("opt", "pess"):
            lines.append(f"\n### {label} | tool budget {tool} | time-retention {mode}\nrows K, a=selector accuracy; columns s=" + ",".join(map(str, SS)))
            for K in (2, 3, 4):
                for a in (0.5, 0.7, 0.9, 1.0):
                    v = [next(t["extra_solves"] for t in table if t["scenario"] == label and t["tool"] == tool and t["mode"] == mode and t["K"] == K and t["s"] == s and t["a"] == a) for s in SS]
                    lines.append(f"K={K} a={a}: " + "  ".join(f"{x:+.2f}" for x in v))
open(os.path.join(here, "gain_table.md"), "w").write("\n".join(lines))
json.dump(out, open(os.path.join(here, "gain_table.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "table"}))
print("\n".join(lines))
