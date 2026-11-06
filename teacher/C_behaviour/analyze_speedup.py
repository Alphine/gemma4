"""How much resolve rate does a uniform speed-up buy?  (counterfactual on the 10-min pilot1 runs)

A run that finished (resolved) at m minutes in the 10-min harvest would, if its student were s% faster, finish at m*(1-s);
it is a 5.5-min success iff m*(1-s) <= 5.5.  Unresolved runs stay unresolved (conservative: no new solves).  Also: the
budget-equivalent view (+X min of budget) which is the same curve.

python analyze_speedup.py --root <scratch C dir> --out speedup_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.root) / "table.jsonl", encoding="utf-8")]
    P = [r for r in rows if r["src"] == "pilot1"]
    N = len(P)
    ok = [r["agent_min"] for r in P if r["resolved"] and not r["timeout"]]
    out = {"n_runs": N, "n_resolved_non_timeout_10min": len(ok)}
    tab = {}
    for s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        eff = 5.5 / (1 - s)
        n = sum(1 for m in ok if m <= eff)
        tab[f"{int(s*100)}%_faster"] = dict(equiv_budget_min=eff, resolved=n, resolve_rate=n / N, gain_pts_vs_now=100 * (n - sum(1 for m in ok if m <= 5.5)) / N,
                                            not_finished=N - sum(1 for r in P if r["agent_min"] * (1 - s) <= 5.5 and not r["timeout"]))
    out["speedup_to_resolve"] = tab
    # the time-out hazard: fraction of runs still unresolved-and-running at t (pilot1)
    out["pts_per_extra_budget_minute_at_5_5"] = 100 * (sum(1 for m in ok if m <= 6.5) - sum(1 for m in ok if m <= 4.5)) / 2 / N
    # required teacher gap = MDE / transfer ratio  (MDE of the paired minutes test at n=144 = 6.3% of the mean)
    pr = json.load(open(Path(__file__).resolve().parent / "paired_results.json"))
    mde = pr["continuous"]["agent_min"]["mde_80pct_n144_pct_of_mean"]
    out["required_teacher_speed_gap_pct"] = {f"transfer_{rho}": mde / rho for rho in (0.2, 0.3, 0.5, 0.75, 1.0)}
    out["mde_minutes_pct_n144"] = mde
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
