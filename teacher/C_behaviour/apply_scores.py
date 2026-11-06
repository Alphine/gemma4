"""Apply score.py to Gemma's own trajectories (pilot1, the 53 kept for SFT, lora-ab1 base/adapter) + validity checks.

python apply_scores.py --root <scratch C dir> --out score_results.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as sc  # noqa: E402
import make_spec  # noqa: E402


def auc_hi(pos, neg):
    """P(pos > neg) + 0.5 ties."""
    pos, neg = np.array(pos, float), np.array(neg, float)
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def summ(x):
    x = np.array(x, float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), p25=float(np.percentile(x, 25)),
                p75=float(np.percentile(x, 75)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.root) / "table.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("m")]
    spec = sc.load_spec(HERE / "spec.json")
    for r in rows:
        r["s_strict"] = sc.score(r["m"], spec, "strict")
        r["s_relaxed"] = sc.score(r["m"], spec, "relaxed")
        r["good"] = r["resolved"] and not r["timeout"]
    P = [r for r in rows if r["src"] == "pilot1"]
    # the SFT keep rule of training/build_sft.py: resolved, <=45 counted calls, a submit_patch
    kept = [r for r in P if r["resolved"] and r["m"]["n_counted"] <= 45 and r["m"]["submitted"]]
    kept_ids = {r["run"] for r in kept}
    out = {"n_pilot1_with_trace": len(P), "n_kept_replicated": len(kept)}

    def block(R):
        return dict(strict=summ([r["s_strict"]["score"] for r in R]), relaxed=summ([r["s_relaxed"]["score"] for r in R]),
                    fit_strict=int(sum(r["s_strict"]["fit"] for r in R)), fit_relaxed=int(sum(r["s_relaxed"]["fit"] for r in R)),
                    est_sec_kaggle_med=float(np.median([r["m"]["est_sec_kaggle_c1"] for r in R])),
                    n_counted_med=float(np.median([r["m"]["n_counted"] for r in R])),
                    mean_turn_tokens_med=float(np.median([r["m"]["mean_turn_tokens"] for r in R])))

    out["pilot1_all"] = block(P)
    out["pilot1_kept53_sft_set"] = block(kept)
    nk = [r for r in P if r["resolved"] and r["run"] not in kept_ids]
    out["pilot1_resolved_not_kept"] = block(nk) if nk else None
    out["pilot1_unresolved"] = block([r for r in P if not r["resolved"]])
    for arm in ("base", "adapter"):
        R = [r for r in rows if r["src"] == "ab1" and r["arm"] == arm]
        out[f"ab1_{arm}_all"] = block(R)
        out[f"ab1_{arm}_resolved_non_timeout"] = block([r for r in R if r["good"]])
        out[f"ab1_{arm}_timeout"] = block([r for r in R if r["timeout"]])

    # ---- validity (1): out-of-source -- spec derived from pilot1 only, scoring ab1
    spec_p = make_spec.derive(P)
    ab = [r for r in rows if r["src"] == "ab1"]
    sp = [sc.score(r["m"], spec_p, "strict")["score"] for r in ab]
    y_to = [r["timeout"] for r in ab]
    y_res = [r["resolved"] for r in ab]
    out["validity_pilot1spec_on_ab1"] = dict(
        auc_score_higher_when_not_timeout=auc_hi([s for s, t in zip(sp, y_to) if not t], [s for s, t in zip(sp, y_to) if t]),
        auc_score_higher_when_resolved=auc_hi([s for s, t in zip(sp, y_res) if t], [s for s, t in zip(sp, y_res) if not t]))
    # ---- validity (2): spec derived from ab1 only, scoring pilot1 (different budget, different tasks)
    spec_a = make_spec.derive(ab)
    sa = [sc.score(r["m"], spec_a, "strict")["score"] for r in P]
    out["validity_ab1spec_on_pilot1"] = dict(
        auc_score_higher_when_not_timeout=auc_hi([s for s, r in zip(sa, P) if not r["timeout"]], [s for s, r in zip(sa, P) if r["timeout"]]),
        auc_score_higher_when_resolved=auc_hi([s for s, r in zip(sa, P) if r["resolved"]], [s for s, r in zip(sa, P) if not r["resolved"]]))

    # ---- validity (3): early predictors -- only the first 10 LLM calls of each ab1 run (non-circular: the outcome
    # is not yet known) -> does the early turn profile predict the final outcome?
    early = []
    for r in ab:
        tr = sc.from_atif(json.load(open(r["trace"], encoding="utf-8")))
        tr.steps = tr.steps[:10]
        m10 = sc.metrics(tr, r["gold"])
        early.append((m10["mean_turn_tokens"], m10["max_turn_tokens"], m10["est_sec_kaggle_c1"], m10["calls_to_good_edit"], r["timeout"], r["resolved"]))
    for i, name in enumerate(("mean_turn_tokens_first10", "max_turn_tokens_first10", "est_sec_first10")):
        v = [t[i] for t in early]
        out[f"early_predictor_{name}"] = dict(
            auc_low_value_when_not_timeout=auc_hi([-x for x, t in zip(v, early) if not t[4]], [-x for x, t in zip(v, early) if t[4]]),
            spearman_with_final_agent_min=float(stats.spearmanr(v, [r["agent_min"] for r in ab]).statistic))
    ed = [t for t in early if t[3] is not None]
    ne = [t for t in early if t[3] is None]
    out["early_predictor_edited_within_first10_calls"] = dict(
        share_edited=len(ed) / len(early),
        timeout_rate_if_edited=float(np.mean([t[4] for t in ed])), timeout_rate_if_not=float(np.mean([t[4] for t in ne])),
        resolve_rate_if_edited=float(np.mean([t[5] for t in ed])), resolve_rate_if_not=float(np.mean([t[5] for t in ne])))

    # ---- profile shift: training set vs base policy; student vs base
    keys = ["n_counted", "calls_to_good_edit", "est_sec_kaggle_c1", "max_turn_tokens", "mean_turn_tokens", "completion_tokens",
            "failed_calls", "repeat_read"]

    def prof(R):
        return {k: float(np.mean([r["m"][k] for r in R if r["m"].get(k) is not None])) for k in keys}

    base_ab = [r for r in ab if r["arm"] == "base"]
    ad_ab = [r for r in ab if r["arm"] == "adapter"]
    out["profile_means"] = {"sft_set_kept53": prof(kept), "pilot1_all_teacher_policy": prof(P),
                            "ab1_base": prof(base_ab), "ab1_adapter_student": prof(ad_ab)}

    def turn_tokens(R):
        v = []
        for r in R:
            tr = sc.from_atif(json.load(open(r["trace"], encoding="utf-8")))
            v += [s.completion_tokens for s in tr.steps if s.completion_tokens is not None]
        return np.array(v, float)

    tk, tp, tb, ta = turn_tokens(kept), turn_tokens(P), turn_tokens(base_ab), turn_tokens(ad_ab)
    q = [50, 75, 90, 95, 99]
    sets = (("sft_kept53", tk), ("pilot1_all", tp), ("ab1_base", tb), ("ab1_adapter", ta))
    out["turn_token_percentiles"] = {n: [float(np.percentile(x, p)) for p in q] + [float(x.mean()), int(len(x))] for n, x in sets}
    out["turn_token_percentile_cols"] = q + ["mean", "n_turns"]
    out["share_turns_gt1000"] = {n: float((x > 1000).mean()) for n, x in sets}
    out["kept53_meets_strict_ceiling"] = {k: int(sum(1 for r in kept if r["m"].get(k) is not None and r["m"][k] <= spec["limits"][k][0]))
                                          for k in spec["limits"]}
    out["kept53_vs_base_turn_tokens_mean_ratio"] = float(tk.mean() / tb.mean())
    keep = ("task", "run", "src", "arm", "resolved", "timeout")
    out["per_run_scores"] = [dict({k: r[k] for k in keep}, strict=r["s_strict"]["score"], relaxed=r["s_relaxed"]["score"],
                                  fit_strict=r["s_strict"]["fit"], kept=(r["run"] in kept_ids)) for r in rows]
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_run_scores"}, indent=1, default=float))


if __name__ == "__main__":
    main()
