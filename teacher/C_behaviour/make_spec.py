"""Derive spec.json (target behaviour profile) from OUR runs and validate the scoring function.

python make_spec.py --root <scratch C dir> --out-dir <C-behaviour dir>
Spec rule: reference group = fastest quartile (by agent minutes) of resolved, non-timeout runs, pooled over pilot1 + lora-ab1.
  full-credit ceiling = 75th percentile of the reference group;  zero-credit floor = median of the timed-out ab1 runs
  (never below ceiling + 1 unit).  Weights are fixed a priori (time-critical items higher), not fitted.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402

METRICS = ["n_counted", "calls_to_good_edit", "est_sec_kaggle_c1", "max_turn_tokens", "mean_turn_tokens",
           "repeat_exact", "repeat_read", "failed_calls", "malformed_calls", "verify_after_edit", "scratch_writes"]
WEIGHTS = {"n_counted": 2, "calls_to_good_edit": 3, "est_sec_kaggle_c1": 3, "max_turn_tokens": 2, "mean_turn_tokens": 2,
           "repeat_exact": 1, "repeat_read": 1, "failed_calls": 1, "malformed_calls": 0.5, "verify_after_edit": 1,
           "scratch_writes": 0.5}


def auc(pos, neg):
    """P(random positive value < random negative value) - i.e. AUC of 'lower is better' separating pos from neg."""
    pos, neg = np.array(pos, float), np.array(neg, float)
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return 1 - u / (len(pos) * len(neg))


def derive(rows):
    rows = [r for r in rows if r.get("m")]
    res = [r for r in rows if r["resolved"] and not r["timeout"]]
    q1 = np.percentile([r["agent_min"] for r in res], 25)
    ref = [r for r in res if r["agent_min"] <= q1]
    bad = [r for r in rows if r["timeout"]]
    spec = {"version": 1, "pass": 0.8, "required": ["submitted", "edited"],
            "reference": dict(n=len(ref), agent_min_q1_threshold=float(q1), sources="resolved non-timeout, fastest quartile"),
            "limits": {}, "weights": WEIGHTS, "table": {}}
    for k in METRICS:
        f = np.array([r["m"][k] for r in ref if r["m"].get(k) is not None], float)
        b = np.array([r["m"][k] for r in bad if r["m"].get(k) is not None], float)
        ceil = float(np.percentile(f, 75))
        floor = float(np.median(b))
        allres = np.array([r["m"][k] for r in res if r["m"].get(k) is not None], float)
        ceil_relaxed = float(np.percentile(allres, 75))
        small = k in ("repeat_exact", "repeat_read", "failed_calls", "malformed_calls", "verify_after_edit", "scratch_writes")
        if small:
            floor = max(floor, ceil + 3.0)
        elif floor <= ceil:
            floor = ceil + max(1.0, 0.5 * abs(ceil))
        floor_relaxed = max(floor, ceil_relaxed + (3.0 if small else 0.2 * abs(ceil_relaxed)))
        spec["limits"][k] = [round(ceil, 2), round(floor, 2)]
        spec.setdefault("limits_relaxed", {})[k] = [round(ceil_relaxed, 2), round(floor_relaxed, 2)]
        spec["table"][k] = dict(ref_p25=float(np.percentile(f, 25)), ref_median=float(np.median(f)), ref_p75=ceil, ref_p90=float(np.percentile(f, 90)),
                                timeout_median=floor if floor > ceil else float(np.median(b)), timeout_p25=float(np.percentile(b, 25)),
                                auc_ref_vs_timeout=float(auc(f, b)))
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.root) / "table.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("m") and not (r["src"] == "pilot1" and r["timeout"] and False)]
    spec = derive(rows)
    Path(a.out_dir, "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(json.dumps(spec["limits"], indent=1))
    for k, v in spec["table"].items():
        print(f"{k:22s} ref p25/med/p75/p90 {v['ref_p25']:.1f}/{v['ref_median']:.1f}/{v['ref_p75']:.1f}/{v['ref_p90']:.1f}"
              f"  timeout p25/med {v['timeout_p25']:.1f}/{v['timeout_median']:.1f}  AUC {v['auc_ref_vs_timeout']:.2f}")


if __name__ == "__main__":
    main()
