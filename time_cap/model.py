#!/usr/bin/env python3
"""Per-task time-cap model for the gemma-4-developer-agent scorer (CPU only).

Data: research/lb-eval-v4/lb_eval_tasks.jsonl, arms triage-first (cap 6.0) and edit-early-v4 (cap 5.5), 43 live tasks
(40 solvable + 3 free) each, real runs on Kaggle 4xL4 with the official Evaluator (ContainerManager).

Each run = (end time e, kind, calls n, own cap c):
  R  finished before cap, resolved        U  finished before cap, unresolved (agent submitted / stopped)
  T  hit the session timeout at c (partial patch may still resolve; flag res)
Beyond the observed caps (5.5/6.0) the data say nothing, so a run that timed out at c is continued with a constant
event hazard  alpha * (hR + hU)  (hR, hU estimated on [W0, cap] pooled over both arms), alpha in {0, 0.5, 1}:
alpha=0 pessimistic (no run ever finishes after its cap: no gain, maximum time), alpha=1 optimistic (hazard stays
as in minutes 3.5-6). A run cut off before it finished gets no resolve credit (a timed-out run that had resolved by its own cap gets 0.5 if cut earlier); a run still running at the new cap with no event keeps its status at the old cap (no new partial-patch credit).
Tool-call cap K: time to the K-th call is approximated linearly, t_K = K / (n/e).
Outputs: results.json  (python model.py [--B 1000] [--sims 4000])
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "research", "lb-eval-v4", "lb_eval_tasks.jsonl")
W0 = 3.5
LIMIT = 720.0


def load():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    runs = []
    for r in rows:
        if r["set"] not in ("solvable", "free"):
            continue
        cap = float(r["budget_effective"]["max_time_minutes"])
        to = "session timeout" in str(r.get("agent_error"))
        kind = "T" if to else ("R" if r["resolved"] else "U")
        runs.append(dict(task=r["instance_id"], arm=r["arm"], free=(r["set"] == "free"), e=float(r["agent_min"]),
                         kind=kind, res=bool(r["resolved"]), n=int(r["tool_calls"]), c=cap, setup=float(r["setup_min"])))
    return runs


def hazards(runs):
    """pooled constant hazards (per minute) of R / U events on [W0, cap]; rcap = P(resolved | timed out)."""
    expo = sum(max(0.0, min(r["e"], r["c"]) - W0) for r in runs)
    nR = sum(1 for r in runs if r["kind"] == "R" and r["e"] >= W0)
    nU = sum(1 for r in runs if r["kind"] == "U" and r["e"] >= W0)
    T = [r for r in runs if r["kind"] == "T"]
    rcap = sum(r["res"] for r in T) / max(1, len(T))
    return nR / expo, nU / expo, rcap


def expected(runs, C, K, alpha, hR, hU, rcap, pace=1.0):
    """per-run analytic expectation of (resolved, agent minutes) at cap C (min) and call cap K, averaged over runs.
    pace>1: this machine is slower, finishers take pace x longer and hazards shrink by 1/pace."""
    lam = alpha * (hR + hU) / pace
    pR = hR / (hR + hU) if hR + hU > 0 else 0.5
    res = []
    tim = []
    for r in runs:
        rate = r["n"] / r["e"] if r["e"] > 0 else 0.0
        tK = K / rate if rate > 0 else math.inf
        L = min(C, tK)
        if r["kind"] in ("R", "U"):
            e = r["e"] * pace
            if e <= L:
                res.append(1.0 if r["kind"] == "R" else 0.0)
                tim.append(e)
            else:
                res.append(0.0)      # cut before it finished: no credit (conservative)
                tim.append(L)
        else:  # timed out at c in the data
            c = r["c"]
            if L <= c:
                res.append((1.0 if r["res"] else 0.0) if L >= c - 1e-9 else (0.5 if r["res"] else 0.0))
                tim.append(L)
            else:
                d = L - c
                p_ev = 1.0 - math.exp(-lam * d)
                mean_t = c + (d if lam <= 1e-12 else (1.0 - math.exp(-lam * d)) / lam)
                if r["res"]:
                    res.append(1.0)
                else:
                    res.append(p_ev * pR)   # no event by the cap: stays unresolved (conservative, no new partial-patch credit)
                tim.append(mean_t)
    res = np.array(res)
    tim = np.array(tim)
    return res, tim


def sample_times(runs, C, K, alpha, hR, hU, rcap, n_tasks, rng, q_timeout=None, pace=1.0):
    """vectorised: simulate n_tasks agent-minute durations at cap C. q_timeout forces the share of tasks drawn from
    timed-out runs (hidden-set hardness scenario)."""
    lam = alpha * (hR + hU) / pace
    pR = hR / (hR + hU) if hR + hU > 0 else 0.5
    kind = np.array([r["kind"] for r in runs])
    e = np.array([r["e"] for r in runs]) * pace
    n = np.array([r["n"] for r in runs], float)
    c = np.array([r["c"] for r in runs])
    res = np.array([r["res"] for r in runs])
    idx_T = np.where(kind == "T")[0]
    idx_F = np.where(kind != "T")[0]
    if q_timeout is None:
        d = rng.integers(0, len(runs), n_tasks)
    else:
        isT = rng.random(n_tasks) < q_timeout
        d = np.where(isT, rng.choice(idx_T, n_tasks), rng.choice(idx_F, n_tasks))
    rate = n[d] / np.maximum(e[d] / pace, 1e-9)
    L = np.minimum(C, K / np.maximum(rate, 1e-9))
    isT = kind[d] == "T"
    t = np.where(e[d] <= L, e[d], L)            # finishers
    ok = np.where(e[d] <= L, (kind[d] == "R").astype(float), np.zeros(n_tasks))
    # timed-out runs
    cc = c[d]
    contd = isT & (L > cc)
    trunc = isT & ~contd
    t = np.where(trunc, L, t)
    ok = np.where(trunc, np.where(L >= cc - 1e-9, res[d].astype(float), 0.5 * res[d].astype(float)), ok)
    if lam > 1e-12:
        x = rng.exponential(1.0 / lam, n_tasks)
    else:
        x = np.full(n_tasks, np.inf)
    dd = L - cc
    ev = contd & (x < dd)
    t = np.where(contd, np.where(ev, cc + x, L), t)
    okc = np.where(ev, (res[d] | (rng.random(n_tasks) < pR)).astype(float), res[d].astype(float))
    ok = np.where(contd, okc, ok)
    return t, ok


def boot_tasks(runs, rng):
    tasks = sorted({r["task"] for r in runs})
    by = {}
    for r in runs:
        by.setdefault(r["task"], []).append(r)
    pick = rng.choice(len(tasks), len(tasks))
    out = []
    for k in pick:
        out.extend(by[tasks[k]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    runs = load()
    hR, hU, rcap = hazards(runs)
    out = dict(n_runs=len(runs), n_tasks=len({r['task'] for r in runs}), hazard_per_min=dict(hR=hR, hU=hU, rcap=rcap),
               shares=dict(timeout=sum(r["kind"] == "T" for r in runs) / len(runs)))
    caps = [5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    solv = [r for r in runs if not r["free"]]
    # ---- sanity: model reproduces what was observed at the data caps
    san = {}
    for arm, cap in (("edit-early-v4", 5.5), ("triage-first", 6.0)):
        rr = [r for r in solv if r["arm"] == arm]
        res, tim = expected(rr, cap, 1000, 0.5, hR, hU, rcap)
        san[arm] = dict(cap=cap, observed_resolved=sum(r["res"] for r in rr), model_resolved=float(res.sum()),
                        observed_mean_min=float(np.mean([r["e"] for r in rr])), model_mean_min=float(tim.mean()))
    out["sanity"] = san
    # ---- Table A: P(resolve | cap) for solvable tasks, bootstrap over tasks (both arms pooled)
    A = {}
    for K in (60, 100):
        for alpha in (0.0, 0.5, 1.0):
            for C in caps:
                vals = []
                tms = []
                for b in range(a.B):
                    bs = boot_tasks(solv, rng)
                    h1, h2, rc = hazards(bs)
                    res, tim = expected(bs, C, K, alpha, h1, h2, rc)
                    vals.append(res.mean())
                    tms.append(tim.mean())
                vals = np.array(vals)
                tms = np.array(tms)
                res0, tim0 = expected(solv, C, K, alpha, hR, hU, rcap)
                A[f"K{K}_a{alpha}_C{C}"] = dict(p=float(res0.mean()), lo=float(np.percentile(vals, 2.5)),
                                                hi=float(np.percentile(vals, 97.5)), mean_agent_min=float(tim0.mean()),
                                                mean_min_lo=float(np.percentile(tms, 2.5)), mean_min_hi=float(np.percentile(tms, 97.5)))
    out["P_resolve_solvable"] = A
    # paired difference vs C=5.5 (same bootstrap draw) for the central alpha and K=60/100
    D = {}
    for K in (60, 100):
        for alpha in (0.0, 0.5, 1.0):
            for C in caps:
                if C == 5.5:
                    continue
                d = []
                for b in range(a.B):
                    bs = boot_tasks(solv, rng)
                    h1, h2, rc = hazards(bs)
                    r1, _ = expected(bs, C, K, alpha, h1, h2, rc)
                    r0, _ = expected(bs, 5.5, 60, alpha, h1, h2, rc)
                    d.append(r1.mean() - r0.mean())
                D[f"K{K}_a{alpha}_C{C}"] = dict(d=float(np.mean(d)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)))
    out["delta_vs_5.5_K60"] = D
    # ---- Table B: wall clock for N tasks. F = fixed (notebook + vLLM start), s = per-task setup (+ any extra overhead)
    W = {}
    pre = []
    for b in range(300):
        bs = boot_tasks(solv, rng)
        pre.append((bs,) + hazards(bs))
    scen = [("dev_like", None), ("q70", 0.70), ("q85", 0.85), ("q100", 1.0)]
    for N in (120, 58):
        for (sname, q) in scen:
            for pace in (1.0, 1.3):
                if N == 58 and (sname != "dev_like" or pace != 1.0):
                    continue
                for alpha in (0.0, 0.5, 1.0):
                    for K in (60, 100):
                        for C in caps:
                            tot = []
                            for sidx in range(a.sims):
                                bs, h1, h2, rc = pre[rng.integers(0, len(pre))]
                                t, _ = sample_times(bs, C, K, alpha, h1, h2, rc, N, rng, q_timeout=q, pace=pace)
                                F = rng.uniform(8.0, 13.0)
                                s = rng.uniform(0.13, 0.5)
                                tot.append(F + N * s + t.sum())
                            tot = np.array(tot)
                            W[f"N{N}_{sname}_pace{pace}_a{alpha}_K{K}_C{C}"] = dict(
                                mean_h=float(tot.mean() / 60), p5_h=float(np.percentile(tot, 5) / 60), p95_h=float(np.percentile(tot, 95) / 60),
                                p_over12=float((tot > LIMIT).mean()), p_over11_5=float((tot > 690).mean()))
    out["wallclock"] = W
    # ---- calls: share of runs that would hit K calls before cap C (linear rate)
    callshare = {}
    for C in (5.5, 7, 8, 10, 12, 15):
        for K in (45, 60, 80, 100):
            hit = 0
            for r in runs:
                rate = r["n"] / r["e"]
                # a run that stopped on its own before C only counts if its own call total reaches K at some time <= its end
                if r["kind"] == "T":
                    hit += (K / rate) < C
                else:
                    hit += r["n"] >= K
            callshare[f"C{C}_K{K}"] = hit / len(runs)
    out["share_runs_hitting_K_before_C_linear"] = callshare
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print(json.dumps(out["hazard_per_min"]), json.dumps(out["sanity"]))


if __name__ == "__main__":
    main()
