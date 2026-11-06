"""Time anatomy + counterfactual (task C, parts 1-2).  Reads table.jsonl built by build_table.py.

python analyze_time.py --root <scratch C dir> --out time_results.json
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def q(x, p):
    x = [v for v in x if v is not None]
    return float(np.percentile(x, p)) if x else None


def groups_for(rows, budget_min):
    """fast / mid / slow resolved (by agent_min tertiles of resolved-without-timeout runs), timeout, other failure."""
    res = [r for r in rows if r["resolved"] and not r["timeout"]]
    q1, q3 = np.percentile([r["agent_min"] for r in res], [25, 75])
    g = {"resolved_fast(Q1)": [], "resolved_mid": [], "resolved_slow(Q4)": [], "timed_out": [], "failed_submit": []}
    for r in rows:
        if r["timeout"]:
            g["timed_out"].append(r)
        elif r["resolved"]:
            k = "resolved_fast(Q1)" if r["agent_min"] <= q1 else ("resolved_slow(Q4)" if r["agent_min"] >= q3 else "resolved_mid")
            g[k].append(r)
        else:
            g["failed_submit"].append(r)
    return g, (float(q1), float(q3))


def lc_stats(r):
    """Per-run time decomposition from the pilot1 llm_calls recorder (lc/*.json)."""
    L = json.load(open(r["lc"]))
    lat = [c["lat"] for c in L if c["lat"] is not None]
    ct = [(c["usage"] or {}).get("completion_tokens", 0) for c in L]
    t = [c["t"] for c in L]
    tool = []
    for i in range(len(L) - 1):
        gap = L[i + 1]["t"] - (L[i]["t"] + (L[i]["lat"] or 0))
        tool.append(max(gap, 0.0))
    return dict(llm_s=sum(lat), tool_s=sum(tool), n=len(L), ct=sum(ct), maxct=max(ct) if ct else 0,
                runaway=sum(1 for c in ct if c >= 4000), reason_chars=sum(c["reason_chars"] for c in L),
                content_chars=sum(c["content_chars"] for c in L), tc_chars=sum(c["tc_chars"] for c in L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rows = [json.loads(l) for l in open(root / "table.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("m")]
    out = {}

    # ---- latency model from pilot1 recorder
    X, Y = [], []
    for r in rows:
        if r["src"] != "pilot1":
            continue
        for c in json.load(open(r["lc"])):
            u = c["usage"] or {}
            if c["lat"] is None or not u:
                continue
            X.append((1.0, u["completion_tokens"], u["prompt_tokens"]))
            Y.append(c["lat"])
    X, Y = np.array(X), np.array(Y)
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    pred = X @ b
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    out["latency_model_modal_c8"] = dict(n_calls=int(len(Y)), a_s=float(b[0]), s_per_completion_token=float(b[1]),
                                         s_per_prompt_token=float(b[2]), r2=float(r2), tok_per_s=float(1 / b[1]),
                                         mean_lat_s=float(Y.mean()), median_lat_s=float(np.median(Y)),
                                         calls_ge4000_tokens=int((X[:, 1] >= 4000).sum()),
                                         share_time_in_ge4000_calls=float(Y[X[:, 1] >= 4000].sum() / Y.sum()),
                                         share_time_in_ge1000_calls=float(Y[X[:, 1] >= 1000].sum() / Y.sum()),
                                         share_calls_ge1000=float((X[:, 1] >= 1000).mean()))
    # ---- same slope from trace deltas (works for Kaggle c=1 where no recorder is available): dt_i ~ a + b*ct_i
    def trace_slope(files):
        xs, ys = [], []
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            tr = sc.from_atif(d)
            for prev, cur in zip(tr.steps[:-1], tr.steps[1:]):
                if prev.t is None or cur.t is None or cur.completion_tokens is None:
                    continue
                dt = cur.t - prev.t
                if 0 < dt < 200:
                    xs.append((1.0, cur.completion_tokens)); ys.append(dt)
        xs, ys = np.array(xs), np.array(ys)
        bb = np.linalg.lstsq(xs, ys, rcond=None)[0]
        p = xs @ bb
        return dict(n=int(len(ys)), a_s=float(bb[0]), s_per_token=float(bb[1]), tok_per_s=float(1 / bb[1]) if bb[1] > 0 else None,
                    r2=float(1 - ((ys - p) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()), mean_step_s=float(ys.mean()),
                    mean_ct=float(xs[:, 1].mean()))
    out["step_slope_modal_c8_pilot1"] = trace_slope([r["trace"] for r in rows if r["src"] == "pilot1"])
    out["step_slope_modal_c8_ab1"] = trace_slope([r["trace"] for r in rows if r["src"] == "ab1"])
    kg = sorted((root / "T/kaggle_v10").glob("*.json"))
    out["step_slope_kaggle_c1_v10_dev43"] = trace_slope(kg)

    # ---- group tables
    def summarize(R):
        M = [r["m"] for r in R]
        n = len(R)
        o = dict(n=n)
        for k in ("calls_to_gold_read", "calls_to_edit", "calls_to_good_edit", "malformed_calls", "calls_to_test", "calls_to_submit", "n_counted", "n_llm",
                  "sec_to_gold_read", "sec_to_edit", "sec_to_good_edit", "sec_to_submit", "total_sec", "repeat_exact", "repeat_read",
                  "failed_calls", "no_call_turns", "verify_after_edit", "scratch_writes", "max_turn_tokens",
                  "mean_turn_tokens", "completion_tokens", "n_turns_over_1000", "thought_share"):
            o[k + "_med"] = med([m.get(k) for m in M])
            o[k + "_mean"] = mean([m.get(k) for m in M])
        o["share_edited"] = float(np.mean([m["edited"] for m in M]))
        o["share_submitted"] = float(np.mean([m["submitted"] for m in M]))
        o["share_gold_read"] = float(np.mean([m["calls_to_gold_read"] is not None for m in M]))
        o["agent_min_med"] = med([r["agent_min"] for r in R])
        return o

    for name, sel in (("pilot1_10min", lambda r: r["src"] == "pilot1"),
                      ("ab1_5.5min_base", lambda r: r["src"] == "ab1" and r["arm"] == "base"),
                      ("ab1_5.5min_adapter", lambda r: r["src"] == "ab1" and r["arm"] == "adapter"),
                      ("ab1_5.5min_pooled", lambda r: r["src"] == "ab1")):
        R = [r for r in rows if sel(r)]
        g, (q1, q3) = groups_for(R, R[0]["budget_min"])
        t = {k: summarize(v) for k, v in g.items() if v}
        t["_quartiles_resolved_agent_min"] = [q1, q3]
        out["groups_" + name] = t

    # ---- time decomposition (pilot1 recorder)
    dec = {}
    P = [r for r in rows if r["src"] == "pilot1"]
    g, _ = groups_for(P, 10)
    for k, R in g.items():
        if not R:
            continue
        S = [lc_stats(r) for r in R]
        tot_llm, tot_tool = sum(s["llm_s"] for s in S), sum(s["tool_s"] for s in S)
        dec[k] = dict(n=len(R), llm_share=tot_llm / (tot_llm + tot_tool), tool_share=tot_tool / (tot_llm + tot_tool),
                      llm_s_per_run=tot_llm / len(R), tool_s_per_run=tot_tool / len(R),
                      calls_per_run=mean([s["n"] for s in S]), completion_tokens_per_run=mean([s["ct"] for s in S]),
                      runaway_calls_per_run=mean([s["runaway"] for s in S]),
                      reasoning_char_share=sum(s["reason_chars"] for s in S) / max(1, sum(s["reason_chars"] + s["content_chars"] + s["tc_chars"] for s in S)),
                      toolargs_char_share=sum(s["tc_chars"] for s in S) / max(1, sum(s["reason_chars"] + s["content_chars"] + s["tc_chars"] for s in S)))
    out["time_decomposition_pilot1"] = dec

    # ab1: LLM latency x calls vs agent seconds
    A = [r for r in rows if r["src"] == "ab1" and r.get("llm_latency_mean_s")]
    out["ab1_llm_share_of_agent_time"] = float(sum(r["llm_latency_mean_s"] * r["llm_calls"] for r in A) / sum(r["agent_min"] * 60 for r in A))
    out["ab1_corr_agent_sec_vs_calls"] = float(np.corrcoef([r["agent_min"] for r in A], [r["llm_calls"] for r in A])[0, 1])
    out["ab1_corr_agent_sec_vs_mean_latency"] = float(np.corrcoef([r["agent_min"] for r in A], [r["llm_latency_mean_s"] for r in A])[0, 1])
    out["ab1_corr_agent_sec_vs_completion_tokens"] = float(np.corrcoef([r["agent_min"] for r in A if r["m"]["completion_tokens"]],
                                                                       [r["m"]["completion_tokens"] for r in A if r["m"]["completion_tokens"]])[0, 1])
    # variance decomposition of log agent seconds among all ab1 runs: calls vs per-call latency
    la = np.log([r["agent_min"] for r in A]); lc_ = np.log([r["llm_calls"] for r in A]); ll = np.log([r["llm_latency_mean_s"] for r in A])
    out["ab1_logvar_share_calls"] = float(np.cov(la, lc_)[0, 1] / la.var(ddof=1))
    out["ab1_logvar_share_latency"] = float(np.cov(la, ll)[0, 1] / la.var(ddof=1))

    # inflight (effective concurrency) vs latency
    infl = [(r["inflight"], r["llm_latency_mean_s"], r["m"]["mean_turn_tokens"]) for r in A if r.get("inflight") and r["m"]["mean_turn_tokens"]]
    out["ab1_corr_inflight_latency"] = float(np.corrcoef([i[0] for i in infl], [i[1] for i in infl])[0, 1])

    # ---- counterfactual: pilot1 (10 min) truncated at 5.5 minutes
    resolved = [r for r in P if r["resolved"]]
    out["counterfactual_pilot1"] = dict(
        n_runs=len(P), resolved_10min=len(resolved),
        resolved_by_5_5min=sum(1 for r in resolved if r["agent_min"] <= 5.5 and not r["timeout"]),
        resolved_by_3min=sum(1 for r in resolved if r["agent_min"] <= 3.0 and not r["timeout"]),
        timeouts_10min=sum(1 for r in P if r["timeout"]),
        timeouts_resolved_by_salvage=sum(1 for r in P if r["timeout"] and r["resolved"]),
        not_finished_by_5_5=sum(1 for r in P if r["timeout"] or r["agent_min"] > 5.5),
        resolved_among_not_finished_by_5_5_min=sum(1 for r in P if (r["timeout"] or r["agent_min"] > 5.5) and r["resolved"]),
        resolve_time_cdf_min={str(t): sum(1 for r in resolved if r["agent_min"] <= t and not r["timeout"]) for t in (1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10)},
    )
    # ab1: how close are timeouts
    for arm in ("base", "adapter"):
        T = [r for r in rows if r["src"] == "ab1" and r["arm"] == arm and r["timeout"]]
        out[f"ab1_timeouts_{arm}"] = dict(
            n=len(T), salvaged_resolved=sum(r["resolved"] for r in T),
            edited_source=sum(1 for r in T if r["m"]["edited"]), never_edited=sum(1 for r in T if not r["m"]["edited"]),
            patch_nonempty=sum(1 for r in T if r["patch_chars"]),
            submitted=sum(1 for r in T if r["m"]["submitted"]),
            sec_to_edit_med=med([r["m"]["sec_to_edit"] for r in T]),
            edit_before_150s=sum(1 for r in T if r["m"]["sec_to_edit"] is not None and r["m"]["sec_to_edit"] <= 150),
            sec_after_edit_med=med([r["m"]["total_sec"] - r["m"]["sec_to_edit"] for r in T if r["m"]["sec_to_edit"] is not None]),
            gold_read=sum(1 for r in T if r["m"]["calls_to_gold_read"] is not None),
            counted_med=med([r["m"]["n_counted"] for r in T]),
            turns_ge1000_tokens_mean=mean([r["m"]["n_turns_over_1000"] for r in T]),
            maxturn_tokens_med=med([r["m"]["max_turn_tokens"] for r in T]),
        )
    # first-edit time vs outcome in ab1 (is slowness before the first edit?)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main()
