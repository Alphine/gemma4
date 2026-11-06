"""Paired (same task, base vs adapter) behaviour analysis + power analysis (task C, parts 1b and 4).

python analyze_paired.py --root <scratch C dir> --out paired_results.json
Reads table.jsonl (build_table.py).  CPU only, seeded.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

rng = np.random.default_rng(20261008)
Z_A, Z_B = 1.959964, 0.841621   # two-sided alpha .05, power .80


def mcnemar_n(delta, p_disc):
    """Paired binary: n pairs for 80% power to detect marginal difference `delta` given discordance rate `p_disc`
    (Connor 1987): n = (z_a*sqrt(p_disc) + z_b*sqrt(p_disc - delta^2))^2 / delta^2."""
    if p_disc <= delta ** 2:
        return float("nan")
    return (Z_A * math.sqrt(p_disc) + Z_B * math.sqrt(p_disc - delta ** 2)) ** 2 / delta ** 2


def paired_n(sd_diff, delta):
    return ((Z_A + Z_B) * sd_diff / delta) ** 2


def mc_power(d, n, delta_shift, nsim=3000, test="t"):
    """Power of a paired test at n pairs: resample n pairs from observed differences d (centred), add delta_shift."""
    d0 = d - d.mean()
    rej = 0
    for _ in range(nsim):
        x = rng.choice(d0, size=n, replace=True) + delta_shift
        if test == "t":
            p = stats.ttest_1samp(x, 0.0).pvalue
        else:
            try:
                p = stats.wilcoxon(x[x != 0]).pvalue if (x != 0).sum() > 5 else 1.0
            except ValueError:
                p = 1.0
        rej += p < 0.05
    return rej / nsim


def mc_power_binary(b_only, a_only, both, neither, n, shift, nsim=4000):
    """Simulated exact-McNemar power. Pair-cell probabilities from lora-ab1 are re-centred to zero marginal
    difference, then `shift` (adapter-minus-base marginal rate) is added by moving mass from (b_only) to (a_only)."""
    N = b_only + a_only + both + neither
    disc = (b_only + a_only) / N
    p_a_only = disc / 2 + shift / 2
    p_b_only = disc / 2 - shift / 2
    p_both = both / N
    p_neither = max(0.0, 1 - p_a_only - p_b_only - p_both)
    probs = np.array([p_a_only, p_b_only, p_both, p_neither])
    probs = probs / probs.sum()
    rej = 0
    for _ in range(nsim):
        c = rng.multinomial(n, probs)
        k, m = c[0], c[1]
        if k + m == 0:
            continue
        p = stats.binomtest(int(min(k, m)), int(k + m), 0.5).pvalue
        rej += p < 0.05
    return rej / nsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.root) / "table.jsonl", encoding="utf-8")]
    ab = [r for r in rows if r["src"] == "ab1"]
    by = defaultdict(dict)
    for r in ab:
        by[r["task"]][r["arm"]] = r
    pairs = [(v["base"], v["adapter"]) for v in by.values() if "base" in v and "adapter" in v]
    out = {"n_pairs": len(pairs)}
    N = len(pairs)

    # ---- 2x2 cells
    res = np.array([[b["resolved"], ad["resolved"]] for b, ad in pairs])
    c = dict(both=int((res[:, 0] & res[:, 1]).sum()), base_only=int((res[:, 0] & ~res[:, 1]).sum()),
             adapter_only=int((~res[:, 0] & res[:, 1]).sum()), neither=int((~res[:, 0] & ~res[:, 1]).sum()))
    out["resolve_cells"] = c
    to = np.array([[b["timeout"], ad["timeout"]] for b, ad in pairs])
    ct = dict(both=int((to[:, 0] & to[:, 1]).sum()), base_only=int((to[:, 0] & ~to[:, 1]).sum()),
              adapter_only=int((~to[:, 0] & to[:, 1]).sum()), neither=int((~to[:, 0] & ~to[:, 1]).sum()))
    out["timeout_cells"] = ct
    out["discordance_resolve"] = (c["base_only"] + c["adapter_only"]) / N
    out["discordance_timeout"] = (ct["base_only"] + ct["adapter_only"]) / N
    # tetrachoric-ish: probability the SAME arm outcome repeats  (phi coefficient)
    out["phi_resolve"] = float(np.corrcoef(res[:, 0], res[:, 1])[0, 1])
    out["phi_timeout"] = float(np.corrcoef(to[:, 0], to[:, 1])[0, 1])

    # ---- continuous metrics: paired differences (adapter - base) and cross-arm correlation
    def get(r, k):
        m = r["m"]
        if k == "agent_min":
            return r["agent_min"]
        if k == "llm_calls":
            return r["llm_calls"]
        if k == "n_counted":
            return m["n_counted"]
        if k == "completion_tokens":
            return m["completion_tokens"]
        if k == "mean_turn_tokens":
            return m["mean_turn_tokens"]
        if k == "calls_to_good_edit_cap60":     # never-edited runs censored at the 60-call cap
            return m["calls_to_good_edit"] if m["calls_to_good_edit"] is not None else 60
        if k == "sec_to_good_edit_cap330":      # never-edited runs censored at the 330 s budget
            return m["sec_to_good_edit"] if m["sec_to_good_edit"] is not None else 330.0
        if k == "sec_to_submit_cap330":
            return m["sec_to_submit"] if m["sec_to_submit"] is not None else 330.0
        if k == "log_agent_sec":
            return math.log(r["agent_min"] * 60)
        if k == "resolved_in_time_le_3min":
            return float(r["resolved"] and r["agent_min"] <= 3.0 and not r["timeout"])
        if k == "resolved_ge_3min":
            return float(r["resolved"] and r["agent_min"] > 3.0)
        if k == "max_turn_tokens":
            return m["max_turn_tokens"]
        raise KeyError(k)

    keys = ["agent_min", "llm_calls", "n_counted", "completion_tokens", "mean_turn_tokens", "calls_to_good_edit_cap60",
            "sec_to_good_edit_cap330", "sec_to_submit_cap330", "log_agent_sec", "max_turn_tokens"]
    P = [(b, ad) for b, ad in pairs if b.get("m") and ad.get("m")]
    out["n_pairs_with_trace"] = len(P)
    cont = {}
    for k in keys:
        x = np.array([get(b, k) for b, ad in P], float)
        y = np.array([get(ad, k) for b, ad in P], float)
        d = y - x
        sd = d.std(ddof=1)
        n = len(d)
        t = stats.ttest_1samp(d, 0)
        boot = [rng.choice(d, n).mean() for _ in range(4000)]
        mdE = (Z_A + Z_B) * sd / math.sqrt(n)
        cont[k] = dict(mean_base=float(x.mean()), mean_adapter=float(y.mean()), mean_diff=float(d.mean()),
                       ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], p_t=float(t.pvalue),
                       sd_diff=float(sd), sd_arm=float(x.std(ddof=1)), corr_arms=float(np.corrcoef(x, y)[0, 1]),
                       mde_80pct_n144_abs=float(mdE), mde_80pct_n144_pct_of_mean=float(100 * mdE / x.mean()),
                       n_for_5pct_of_mean=float(paired_n(sd, 0.05 * x.mean())),
                       n_for_10pct_of_mean=float(paired_n(sd, 0.10 * x.mean())),
                       n_for_15pct_of_mean=float(paired_n(sd, 0.15 * x.mean())),
                       n_for_20pct_of_mean=float(paired_n(sd, 0.20 * x.mean())))
    out["continuous"] = cont

    # ---- binary power: n pairs needed
    pd_res = out["discordance_resolve"]
    pd_to = out["discordance_timeout"]
    out["binary_n_needed_80pct"] = {
        "resolve": {f"{int(dl*100)}pts": mcnemar_n(dl, pd_res) for dl in (0.03, 0.05, 0.07, 0.10)},
        "timeout": {f"{int(dl*100)}pts": mcnemar_n(dl, pd_to) for dl in (0.03, 0.05, 0.07, 0.10)},
    }
    out["binary_mde_pts_at_n"] = {}
    for n in (48, 72, 100, 144, 200, 300):
        # solve smallest delta with 80% power by bisection on the closed form
        def mde(pd):
            lo, hi = 0.005, 0.5
            for _ in range(60):
                mid = (lo + hi) / 2
                nn = mcnemar_n(mid, pd)
                if math.isnan(nn) or nn > n:
                    lo = mid
                else:
                    hi = mid
            return hi
        out["binary_mde_pts_at_n"][str(n)] = dict(resolve=100 * mde(pd_res), timeout=100 * mde(pd_to))

    # ---- Monte-Carlo power curves
    mc = {}
    sd_min = cont["agent_min"]["sd_diff"]
    dmin = np.array([get(ad, "agent_min") - get(b, "agent_min") for b, ad in P])
    for rel in (0.05, 0.10, 0.15, 0.20):
        shift = -rel * cont["agent_min"]["mean_base"]
        mc[f"agent_min_-{int(rel*100)}%"] = {str(n): mc_power(dmin, n, shift, 1500) for n in (48, 72, 144)}
    dtok = np.array([get(ad, "completion_tokens") - get(b, "completion_tokens") for b, ad in P])
    for rel in (0.10, 0.20):
        shift = -rel * cont["completion_tokens"]["mean_base"]
        mc[f"completion_tokens_-{int(rel*100)}%"] = {str(n): mc_power(dtok, n, shift, 1500) for n in (48, 72, 144)}
    for shift in (0.05, 0.07, 0.10):
        mc[f"resolve_+{int(shift*100)}pts"] = {str(n): mc_power_binary(c["base_only"], c["adapter_only"], c["both"], c["neither"], n, shift)
                                               for n in (48, 72, 144, 288)}
        mc[f"timeout_-{int(shift*100)}pts"] = {str(n): mc_power_binary(ct["base_only"], ct["adapter_only"], ct["both"], ct["neither"], n, shift)
                                               for n in (48, 72, 144, 288)}
    out["mc_power"] = mc

    # ---- discordant pairs: what differed within the same task?
    disc = []
    for b, ad in pairs:
        if b["resolved"] != ad["resolved"] and b.get("m") and ad.get("m"):
            w, l = (b, ad) if b["resolved"] else (ad, b)
            disc.append(dict(task=b["task"], winner=w["arm"], w_min=w["agent_min"], l_min=l["agent_min"], l_timeout=l["timeout"],
                             w_counted=w["m"]["n_counted"], l_counted=l["m"]["n_counted"],
                             w_good_edit=w["m"]["calls_to_good_edit"], l_good_edit=l["m"]["calls_to_good_edit"],
                             w_ct=w["m"]["completion_tokens"], l_ct=l["m"]["completion_tokens"],
                             w_sec_edit=w["m"]["sec_to_good_edit"], l_sec_edit=l["m"]["sec_to_good_edit"]))
    out["discordant_pairs"] = disc
    out["discordant_summary"] = dict(
        n=len(disc), loser_timed_out=sum(d["l_timeout"] for d in disc),
        winner_min_med=float(np.median([d["w_min"] for d in disc])),
        winner_ct_med=float(np.median([d["w_ct"] for d in disc])), loser_ct_med=float(np.median([d["l_ct"] for d in disc])),
        winner_edit_call_med=float(np.median([d["w_good_edit"] for d in disc if d["w_good_edit"] is not None])),
        loser_edit_call_med=float(np.median([d["l_good_edit"] for d in disc if d["l_good_edit"] is not None] or [np.nan])),
        loser_never_edited=sum(d["l_good_edit"] is None for d in disc))

    # ---- both resolved: speed ratio within the same task
    both = [(b, ad) for b, ad in P if b["resolved"] and ad["resolved"] and not b["timeout"] and not ad["timeout"]]
    ratios = np.array([max(b["agent_min"], ad["agent_min"]) / min(b["agent_min"], ad["agent_min"]) for b, ad in both])
    out["both_resolved_pairs"] = dict(n=len(both), corr_log_min=float(np.corrcoef([math.log(b["agent_min"]) for b, ad in both],
                                                                              [math.log(ad["agent_min"]) for b, ad in both])[0, 1]),
                                      ratio_median=float(np.median(ratios)), ratio_p75=float(np.percentile(ratios, 75)),
                                      ratio_ge2=int((ratios >= 2).sum()))
    # fast-vs-slow arm of the same task (both resolved, ratio >= 1.8): what differs?
    fs = []
    for b, ad in both:
        f, s = (b, ad) if b["agent_min"] < ad["agent_min"] else (ad, b)
        if s["agent_min"] / f["agent_min"] >= 1.8:
            fs.append((f, s))
    def md(k, which):
        return float(np.median([ (f if which == 0 else s)["m"][k] for f, s in fs if (f if which == 0 else s)["m"][k] is not None]))
    out["same_task_fast_vs_slow"] = {"n": len(fs), **{k: [md(k, 0), md(k, 1)] for k in
                                                   ("n_counted", "calls_to_good_edit", "completion_tokens", "mean_turn_tokens", "max_turn_tokens",
                                                    "repeat_exact", "repeat_read", "failed_calls", "verify_after_edit")}}
    out["same_task_fast_vs_slow"]["agent_min"] = [float(np.median([f["agent_min"] for f, s in fs])), float(np.median([s["agent_min"] for f, s in fs]))]

    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("discordant_pairs",)}, indent=1, default=float))


if __name__ == "__main__":
    main()
