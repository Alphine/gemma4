#!/usr/bin/env python3
"""Pre-registered paired A/B analysis for the edit-early-v7 token-economy test (see PREREGISTRATION.md).

Arms (roles): control = edit-early-v6-locate, decide = edit-early-v7-decide-nt, lean = edit-early-v7-lean-nt.
Unit = task, paired across arms. Works on partial data (complete-case over the arms present).

    python analyze_ab.py --results ckpt_dir_or_files --traces tarballs_or_dirs [--trace-metrics trace_metrics.jsonl]
                         --out-json ab_report.json --out-md ab_report.md [--preflight pass|fail|unknown]
    python analyze_ab.py --power ../teacher/C_behaviour/paired_results.json

Needs numpy + scipy (present on Kaggle and locally). Deterministic: every random draw is seeded from (--seed, label).
"""
from __future__ import annotations

import argparse
import glob
import io
import itertools
import json
import math
import sys
import tarfile
import zlib
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_metrics as tm  # noqa: E402

DEFAULT_ARMS = {"control": "edit-early-v6-locate", "decide": "edit-early-v7-decide-nt", "lean": "edit-early-v7-lean-nt"}
ARM_CFG = {"control": {"thinking_budget": 4096, "max_output_tokens": 4096},
           "decide": {"thinking_budget": 4096, "max_output_tokens": 4096},
           "lean": {"thinking_budget": 1024, "max_output_tokens": 2560}}
CONTRASTS = [("decide", "control"), ("lean", "decide"), ("lean", "control")]   # first two = primary, third = overall
PRE = dict(seed=20261010, n_boot=10000, alpha=0.05, planned=43, min_cov=0.95, max_attempts=2,
           stop_min_pairs=30, stop_harm=4, stop_config=3, default_cap_min=5.5,
           gate_metric="tokens_to_first_edit_cens", resolved_margin=2,   # AMENDMENT 1 (2026-10-09, before data)
           gate={"lean": dict(contrast=("lean", "control"), min_fall=0.08, ci_hi_lt=0.0),
                 "decide": dict(contrast=("decide", "control"), min_fall=0.05, ci_hi_lt=0.03)},
           m1_cap_frac=0.01, m2_edit_delta=-2.0, m2_reads_rel=-0.25, max_trace_missing_tokens=0.10)


# ------------------------------------------------------------------------------------------------- loading
def _expand(paths, pattern):
    out = []
    for p in paths or []:
        p = str(p)
        if any(c in p for c in "*?["):
            out += sorted(glob.glob(p))
        elif Path(p).is_dir():
            out += sorted(str(x) for x in Path(p).rglob(pattern))
        elif Path(p).exists():
            out.append(p)
    return out


def load_rows(paths, warnings):
    rows, bad, mock = [], 0, 0
    for f in _expand(paths, "results_*.jsonl"):
        if Path(f).name.startswith("results_mock_"):
            continue
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            if not isinstance(r, dict) or "arm" not in r or "instance_id" not in r:
                bad += 1
                continue
            if r.get("run_mode", "real") != "real":
                mock += 1
                continue
            rows.append(r)
    if bad:
        warnings.append(f"{bad} unreadable result lines skipped")
    if mock:
        warnings.append(f"{mock} non-real (mock) rows ignored")
    return rows


def _iter_trace_blobs(paths, warnings):
    """Yield (path_in_container, bytes). Tar members are read in memory (never extracted to disk)."""
    plist = []
    for p in paths or []:
        p = str(p)
        plist += sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
    for p in plist:
        pp = Path(p)
        if p.endswith((".tar", ".tar.gz", ".tgz")) and pp.is_file():
            with tarfile.open(p, "r:*") as t:
                for m in t:
                    if m.isfile() and m.name.endswith(".json") and m.size < 80_000_000:
                        yield m.name, t.extractfile(m).read()
        elif pp.is_dir():
            for x in sorted(pp.rglob("*.json")):
                yield str(x), x.read_bytes()
        elif pp.is_file() and p.endswith(".json"):
            yield p, pp.read_bytes()
        else:
            warnings.append(f"trace path not found or unsupported: {p}")


def _arm_task_from_path(name, arm_names):
    parts = [x for x in name.replace(chr(92), "/").split("/") if x]
    stem = parts[-1][:-5] if parts[-1].endswith(".json") else parts[-1]
    if stem.startswith("trace_"):
        stem = stem[6:]
    arm = next((x for x in reversed(parts[:-1]) if x in arm_names), None)
    if arm is None:                                   # flat file "<arm>__<task>.json"
        for a in arm_names:
            if stem.startswith(a + "__"):
                arm, stem = a, stem[len(a) + 2:]
                break
    return arm, stem


def load_traces(paths, arm_names, warnings):
    """-> {(arm_name, task): metrics}. Later paths win (pass session files in chronological order)."""
    out, unmatched, bad = {}, 0, 0
    for name, blob in _iter_trace_blobs(paths, warnings):
        arm, task = _arm_task_from_path(name, set(arm_names))
        if arm is None:
            unmatched += 1
            continue
        m = tm.parse_trace_text(blob.decode("utf-8", "replace"))
        bad += not m.get("trace_ok")
        out[(arm, task)] = m
    if unmatched:
        warnings.append(f"{unmatched} trace files could not be assigned to a known arm (path lacks the arm directory)")
    if bad:
        warnings.append(f"{bad} traces unreadable (treated as missing)")
    return out


def load_trace_metrics(paths, warnings):
    out = {}
    for f in _expand(paths, "trace_metrics*.jsonl"):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["arm"], r["instance_id"])] = r["metrics"]
    return out


# ------------------------------------------------------------------------------------------------- run table
def _pos_from_row(r, name):
    for k in ("arm_pos", "arm_position", "order_pos", "order", "arm_order"):
        v = r.get(k)
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, (list, tuple)) and name in v:
            return float(list(v).index(name) + 1)
        if isinstance(v, str) and name in v.split(","):
            return float(v.split(",").index(name) + 1)
    return None


def build_runs(rows, roles, names, traces, tasks_filter, launch_ids, cfg, warnings):
    """-> runs[(role, task)] (finished, non-infra), status[(role, task)] in done|exhausted|pending."""
    name2role = {names[r]: r for r in roles}
    by = {}
    for r in rows:
        if r["arm"] not in name2role:
            continue
        if launch_ids and r.get("launch_id") not in launch_ids:
            continue
        t = r["instance_id"]
        if tasks_filter is not None and t not in tasks_filter:
            continue
        if tasks_filter is None and r.get("set") == "dead":
            continue
        by.setdefault((name2role[r["arm"]], t), []).append(r)
    runs, status, dup = {}, {}, 0
    for key, rs in by.items():
        good = [r for r in rs if not r.get("infra_error")]
        if good:
            dup += len(good) > 1
            r = good[-1]
            status[key] = "done"
        else:
            status[key] = "exhausted" if len(rs) >= cfg["max_attempts"] else "pending"
            continue
        role, task = key
        cap = float((r.get("budget") or {}).get("max_time_minutes") or cfg["default_cap_min"])
        am = r.get("agent_min")
        err = " ".join(str(r.get(k) or "") for k in ("error", "agent_error"))
        timeout = ("session timeout" in err.lower()) or (am is not None and am >= cap - 0.01)
        if am is not None and timeout:
            am = cap                                  # exact ties at the cap: the observed value IS the cap
        pl = r.get("patch_chars")
        run = dict(role=role, task=task, arm=r["arm"], resolved=bool(r.get("resolved")),
                   patch_chars=pl, empty_patch=(pl == 0) if pl is not None else None, agent_min=am, cap=cap,
                   timeout=bool(timeout), error=err.strip(), error_type=r.get("error_type"),
                   llm_calls_row=r.get("llm_calls"), tool_calls_row=r.get("tool_calls"),
                   max_tool_calls=(r.get("budget") or {}).get("max_tool_calls"), set=r.get("set"),
                   launch_id=r.get("launch_id"), pos=_pos_from_row(r, r["arm"]),
                   config_row=bool(r.get("config_error")) or str(r.get("error_class") or "").lower() in ("config", "config_error"),
                   tr=None)
        tr = traces.get((r["arm"], task))
        if tr and tr.get("trace_ok"):
            run["tr"] = tr
        runs[key] = run
    if dup:
        warnings.append(f"{dup} (arm, task) pairs had >1 non-infra row; the last row was used")
    return runs, status


def _tr_start(run):
    t = run.get("tr") or {}
    return t.get("start_ts")


def assign_positions(runs, roles, tasks):
    """arm position per task: row field if all arms have it, else rank of trace start timestamps, else None."""
    n_row = n_ts = 0
    for t in tasks:
        rs = [runs[(r, t)] for r in roles]
        if all(x["pos"] is not None for x in rs):
            order = sorted(rs, key=lambda x: x["pos"])
            for i, x in enumerate(order):
                x["pos_rank"] = i + 1
            n_row += 1
        elif all(_tr_start(x) is not None for x in rs):
            for i, x in enumerate(sorted(rs, key=_tr_start)):
                x["pos_rank"] = i + 1
            n_ts += 1
        else:
            for x in rs:
                x["pos_rank"] = None
    return {"from_rows": n_row, "from_trace_timestamps": n_ts}


# ------------------------------------------------------------------------------------------------- metrics
NAN = float("nan")


def _tr(run, k):
    t = run.get("tr")
    v = t.get(k) if t else None
    return NAN if v is None else float(v)


def run_metrics(run, cfg):
    t = run.get("tr")
    m = {}
    ok_tok = bool(t) and t["n_llm_calls"] > 0 and t["n_calls_missing_tokens"] / t["n_llm_calls"] <= cfg["max_trace_missing_tokens"]
    if t and t["n_llm_calls"] == 0:
        ok_tok = True
    m["tokens"] = float(t["completion_tokens"]) if ok_tok else NAN
    m["agent_min"] = NAN if run["agent_min"] is None else float(run["agent_min"])
    m["est_llm_s"] = _tr(run, "est_llm_s") if ok_tok else NAN
    m["llm_calls"] = _tr(run, "n_llm_calls") if t else (NAN if run["llm_calls_row"] is None else float(run["llm_calls_row"]))
    m["tool_calls"] = _tr(run, "n_counted_calls") if t else (NAN if run["tool_calls_row"] is None else float(run["tool_calls_row"]))
    nl = m["llm_calls"]
    m["tokens_per_call"] = m["tokens"] / nl if (nl and nl > 0 and not math.isnan(m["tokens"])) else NAN
    has_edit = bool(t and t.get("has_edit"))
    m["calls_to_first_edit"] = _tr(run, "first_edit_counted_idx") if has_edit else NAN
    mt = run.get("max_tool_calls") or 60
    m["calls_to_first_edit_cens"] = (_tr(run, "first_edit_counted_idx") if has_edit else float(mt)) if t else NAN
    m["reads_before_first_edit"] = _tr(run, "reads_before_first_edit") if has_edit else NAN
    m["tokens_to_first_edit"] = _tr(run, "tokens_to_first_edit") if has_edit else NAN
    m["sec_to_first_edit"] = _tr(run, "sec_to_first_edit") if has_edit else NAN
    # AMENDMENT 1: gate metric = completion tokens until the first successful source edit; a run that never edits is
    # right-censored at its total completion tokens (a run that spent everything without editing counts as a slow one)
    m["tokens_to_first_edit_cens"] = (_tr(run, "tokens_to_first_edit") if has_edit else m["tokens"]) if t else NAN
    m["resolved"] = float(run["resolved"])
    m["timeout"] = float(run["timeout"])
    m["empty_patch"] = NAN if run["empty_patch"] is None else float(run["empty_patch"])
    m["has_edit"] = float(has_edit) if t else NAN
    return m


def config_flags(run):
    """(hard, soft) config-error evidence for one run. hard = thinking_token_budget ValueError; soft = MAX_TOKENS / truncated args."""
    t = run.get("tr") or {}
    marks = t.get("marks") or {}
    err = run["error"]
    hard = bool(marks.get("thinking_budget_error")) or ("thinking_token_budget" in err and "Error" in err)         or bool(run.get("config_row"))
    soft = bool(marks.get("max_tokens") or marks.get("args_truncated")) or ("MAX_TOKENS" in err) \
        or ("Tool call arguments were truncated" in err)
    return hard, soft, bool(t)


# ------------------------------------------------------------------------------------------------- statistics
def rng_for(label, cfg):
    return np.random.default_rng([cfg["seed"], zlib.crc32(label.encode())])


def signflip_p(d, rng, n_mc=20000):
    d = np.asarray(d, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(d.sum())
    eps = 1e-9 * max(1.0, np.abs(d).sum())
    if n <= 16:
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        return float(np.mean(np.abs(bits @ d) >= obs - eps))
    signs = rng.integers(0, 2, size=(n_mc, n)) * 2 - 1
    return float((1 + np.sum(np.abs(signs @ d) >= obs - eps)) / (n_mc + 1))


def wilcoxon_p(d):
    d = np.asarray(d, float)
    if np.count_nonzero(d) == 0:
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox", method="auto").pvalue)
    except Exception:
        return None


def pct(v, lo=2.5, hi=97.5):
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return [None, None]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def continuous_contrast(x, y, label, cfg, mask=None):
    """Treated x vs reference y, paired. rel = mean(x)/mean(y)-1 (ratio of means), median of paired differences."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= mask
    x, y = x[ok], y[ok]
    n = len(x)
    res = dict(n=int(n))
    if n < 2:
        return dict(res, mean_x=None, mean_y=None, rel=None, rel_ci=[None, None], mean_diff=None, median_diff=None,
                    median_diff_ci=[None, None], p_wilcoxon=None, p_signflip=None)
    d = x - y
    rng = rng_for(label, cfg)
    idx = rng.integers(0, n, size=(cfg["n_boot"], n))
    bx, by = x[idx], y[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_b = bx.mean(1) / by.mean(1) - 1
    rel_b[~np.isfinite(rel_b)] = np.nan
    my = y.mean()
    res.update(mean_x=float(x.mean()), mean_y=float(my),
               rel=(float(x.mean() / my - 1) if my != 0 else None), rel_ci=pct(rel_b),
               mean_diff=float(d.mean()), mean_diff_ci=pct((bx - by).mean(1)),
               median_diff=float(np.median(d)), median_diff_ci=pct(np.median(bx - by, axis=1)),
               p_wilcoxon=wilcoxon_p(d), p_signflip=signflip_p(d, rng),
               boot_dropped=int(np.sum(~np.isfinite(rel_b))))
    return res


def binary_contrast(x, y, label, cfg):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    b = int(np.sum((x == 1) & (y == 0)))        # event only in treated arm
    c = int(np.sum((x == 0) & (y == 1)))        # event only in reference arm
    p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c > 0 else 1.0
    res = dict(n=int(n), events_x=int(x.sum()), events_y=int(y.sum()), only_x=b, only_y=c, p_mcnemar_exact=p)
    if n >= 2:
        rng = rng_for(label, cfg)
        idx = rng.integers(0, n, size=(cfg["n_boot"], n))
        res["rate_diff"] = float(x.mean() - y.mean())
        res["rate_diff_ci"] = pct((x[idx] - y[idx]).mean(1))
    return res


def censored_sign(x, y, cx, cy):
    """Paired sign test on the pairs whose order is known when both times are right-censored at a common cap:
    both censored -> tie (no information); one censored -> the uncensored one is shorter; none -> sign of difference."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    sgn = []
    for xi, yi, a, b in zip(x[ok], y[ok], np.asarray(cx)[ok], np.asarray(cy)[ok]):
        if a and b:
            continue
        if a:
            sgn.append(-1)                            # treated censored -> treated longer
        elif b:
            sgn.append(1)                             # reference censored -> treated shorter
        elif xi != yi:
            sgn.append(1 if xi < yi else -1)
    k = sum(1 for s in sgn if s > 0)
    n = len(sgn)
    p = float(stats.binomtest(k, n, 0.5).pvalue) if n else 1.0
    return dict(n_informative=n, n_treated_shorter=k, n_treated_longer=n - k,
                n_tied_censored=int(np.sum(ok & np.asarray(cx) & np.asarray(cy))), p_sign=p)


def holm(pdict):
    items = sorted(((p, k) for k, p in pdict.items() if p is not None))
    m = len(items)
    adj, run = {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def order_effect(d, dpos, mean_y, label, cfg):
    """OLS of per-task paired difference on arm-position difference. slope = order effect per position step;
    intercept = treatment effect at equal position."""
    d, dpos = np.asarray(d, float), np.asarray(dpos, float)
    ok = np.isfinite(d) & np.isfinite(dpos)
    d, dpos = d[ok], dpos[ok]
    n = len(d)
    if n < 6 or np.ptp(dpos) == 0:
        return dict(n=int(n), identifiable=False, note="arm order not varied across tasks (or <6 tasks): order effect is "
                    "confounded with the treatment effect and cannot be separated")
    rng = rng_for(label, cfg)
    lr = stats.linregress(dpos, d)
    idx = rng.integers(0, n, size=(cfg["n_boot"], n))
    X, Y = dpos[idx], d[idx]
    xm, ym = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    vx = ((X - xm) ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = ((X - xm) * (Y - ym)).sum(1) / vx
    sl[vx == 0] = np.nan
    ic = ym[:, 0] - sl * xm[:, 0]
    sl_ci, ic_ci = pct(sl), pct(ic)
    flag = bool(sl_ci[0] is not None and (sl_ci[0] > 0 or sl_ci[1] < 0))
    return dict(n=int(n), identifiable=True, slope=float(lr.slope), slope_ci=sl_ci, slope_p=float(lr.pvalue),
                intercept=float(lr.intercept), intercept_ci=ic_ci,
                intercept_rel=(float(lr.intercept / mean_y) if mean_y else None),
                intercept_rel_ci=[(c / mean_y if (c is not None and mean_y) else None) for c in ic_ci],
                n_distinct_posdiff=int(len(np.unique(dpos))), flag=flag)


# ------------------------------------------------------------------------------------------------- analysis
def analyze_runs(runs, status, roles, cfg, traces_present=True, preflight="unknown", planned=None, tasks_planned=None):
    warnings = []
    planned = planned or cfg["planned"]
    out = {"roles": roles, "prereg": {k: v for k, v in cfg.items() if k != "gate"}}
    all_tasks = sorted({t for (_, t) in status})
    complete = [t for t in all_tasks if all(status.get((r, t)) == "done" for r in roles)]
    infra_missing = [t for t in all_tasks if t not in complete and any(status.get((r, t)) == "exhausted" for r in roles)
                     and all(status.get((r, t)) in ("done", "exhausted") for r in roles)]
    pending = [t for t in all_tasks if t not in complete and t not in infra_missing]
    never_started = max(0, planned - len(all_tasks)) if tasks_planned is None else len(set(tasks_planned) - set(all_tasks))
    out["accounting"] = dict(planned=planned, n_complete=len(complete), infra_missing=infra_missing,
                             pending_or_partial=pending, never_started=never_started)
    tasks = complete
    n = len(tasks)
    if n == 0 or "control" not in roles:
        out["error"] = "no complete pairs" if n == 0 else "control arm missing"
        return out
    pos_info = assign_positions(runs, roles, tasks)
    out["order_source"] = pos_info
    M = {r: [run_metrics(runs[(r, t)], cfg) for t in tasks] for r in roles}

    def col(r, k):
        return np.array([m[k] for m in M[r]], float)

    def flag(r, k):
        return np.array([bool(runs[(r, t)][k]) for t in tasks])

    # coverage
    cov = float(np.mean([all(runs[(r, t)]["tr"] is not None for r in roles) for t in tasks]))
    out["trace_coverage"] = cov
    if cov < 1:
        warnings.append(f"trace coverage {cov:.1%} of complete pairs: trace-based metrics use fewer pairs")
    mism = [(r, t) for r in roles for t in tasks if runs[(r, t)]["tr"] and runs[(r, t)]["tr"].get("tokens_mismatch")]
    if mism:
        warnings.append(f"{len(mism)} traces: sum of step tokens != final_metrics.total_completion_tokens (step sum used)")
    for r in roles:
        miss = sum(runs[(r, t)]["tr"]["n_calls_missing_tokens"] for t in tasks if runs[(r, t)]["tr"])
        if miss:
            warnings.append(f"{r}: {miss} LLM calls without token metrics (undercounted)")
        if cfg_zero_tokens(col(r, "tokens")):
            warnings.append(f"{r}: ZERO completion tokens in every trace -> token data defective, gates fail")
    # arm summary
    out["arm_summary"] = {r: dict(
        n=n, resolved=int(np.nansum(col(r, "resolved"))), timeouts=int(np.nansum(col(r, "timeout"))),
        empty_patch=int(np.nansum(col(r, "empty_patch"))), mean_tokens=_nm(col(r, "tokens")),
        mean_agent_min=_nm(col(r, "agent_min")), mean_llm_calls=_nm(col(r, "llm_calls")),
        mean_tool_calls=_nm(col(r, "tool_calls")), mean_est_llm_s=_nm(col(r, "est_llm_s")),
        no_edit_runs=int(np.nansum(1 - col(r, "has_edit"))) if np.isfinite(col(r, "has_edit")).any() else None,
        error_types=_count([runs[(r, t)]["error_type"] for t in tasks])) for r in roles}
    # contrasts
    cont_metrics = ["tokens", "agent_min", "est_llm_s", "llm_calls", "tool_calls", "tokens_per_call", "calls_to_first_edit",
                    "calls_to_first_edit_cens", "reads_before_first_edit", "tokens_to_first_edit", "sec_to_first_edit",
                    "tokens_to_first_edit_cens", "tokens_noTO"]
    out["contrasts"] = {}
    for a, b in CONTRASTS:
        if a not in roles or b not in roles:
            continue
        key = f"{a}_vs_{b}"
        c = {}
        for k in cont_metrics:
            if k == "tokens_noTO":
                c[k] = continuous_contrast(col(a, "tokens"), col(b, "tokens"), f"{key}/{k}", cfg,
                                           mask=~flag(a, "timeout") & ~flag(b, "timeout"))
            else:
                c[k] = continuous_contrast(col(a, k), col(b, k), f"{key}/{k}", cfg)
        c["agent_min_orderable"] = censored_sign(col(a, "agent_min"), col(b, "agent_min"), flag(a, "timeout"), flag(b, "timeout"))
        for k in ("resolved", "timeout", "empty_patch"):
            c[k] = binary_contrast(col(a, k), col(b, k), f"{key}/{k}", cfg)
        out["contrasts"][key] = c
    # multiplicity: Holm over the 3 token contrasts (sign-flip permutation test of the mean difference)
    praw = {k: v["tokens"]["p_signflip"] for k, v in out["contrasts"].items()}
    padj = holm(praw)
    pw_adj = holm({k: v["tokens"]["p_wilcoxon"] for k, v in out["contrasts"].items()})
    for k in out["contrasts"]:
        out["contrasts"][k]["tokens"]["p_signflip_holm"] = padj.get(k)
        out["contrasts"][k]["tokens"]["p_wilcoxon_holm"] = pw_adj.get(k)
    # order effects
    out["order"] = {}
    for a, b in CONTRASTS:
        if a in roles and b in roles:
            pa = np.array([runs[(a, t)].get("pos_rank") or NAN for t in tasks], float)
            pb = np.array([runs[(b, t)].get("pos_rank") or NAN for t in tasks], float)
            key = f"{a}_vs_{b}"
            out["order"][key] = {k: order_effect(col(a, k) - col(b, k), pa - pb, _nm(col(b, k)), f"order/{key}/{k}", cfg)
                                 for k in ("tokens", "agent_min", "tokens_to_first_edit_cens")}
    # mechanisms
    out["mechanism"] = mechanisms(runs, tasks, roles, M, cfg)
    # config errors (attributable)
    out["config_errors"] = {}
    for r in roles:
        hard_t, soft_t, soft_excess, raw_soft, unknown = [], [], [], 0, 0
        for t in tasks:
            h, s, has_tr = config_flags(runs[(r, t)])
            hc, sc, _ = config_flags(runs[("control", t)])
            if not has_tr:
                unknown += 1
            if h:
                hard_t.append(t)
            if s:
                raw_soft += 1
                soft_t.append(t)
                if r != "control" and not sc:
                    soft_excess.append(t)
        att = sorted(set(hard_t if r != "control" else []) | set(soft_excess))
        out["config_errors"][r] = dict(hard=hard_t, raw_soft_runs=raw_soft, soft_excess_over_control=soft_excess,
                                       attributable=att, n_attributable=len(att), runs_without_trace=unknown)
    # stop rules + gates + decision
    out["stop_rules"] = stop_rules(out, roles, n, cfg)
    out["gates"] = {r: gate(out, r, roles, n, cov, preflight, cfg) for r in ("lean", "decide") if r in roles}
    out["decision"] = decide(out, roles, n, cov, preflight, planned, cfg)
    out["warnings"] = warnings
    return out


def cfg_zero_tokens(v):
    v = v[np.isfinite(v)]
    return len(v) > 0 and float(v.sum()) == 0.0


def _nm(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else None


def _count(xs):
    d = {}
    for x in xs:
        k = str(x)
        d[k] = d.get(k, 0) + 1
    return d


def mechanisms(runs, tasks, roles, M, cfg):
    res = {}
    for r in roles:
        calls, reas = [], []
        for t in tasks:
            tr = runs[(r, t)]["tr"]
            if tr:
                calls += tr["call_tokens"]
                reas += tr.get("reasoning_tokens") or []
        mo = ARM_CFG[r]["max_output_tokens"]
        a = np.array(calls, float)
        res[r] = dict(n_calls=int(len(a)), max_call_tokens=(float(a.max()) if len(a) else None),
                      p95=(float(np.percentile(a, 95)) if len(a) else None), p99=(float(np.percentile(a, 99)) if len(a) else None),
                      frac_at_max_output=(float(np.mean(a >= mo - 8)) if len(a) else None),
                      max_reasoning_tokens=(max(reas) if reas else None), reasoning_available=bool(reas),
                      thinking_budget=ARM_CFG[r]["thinking_budget"], max_output_tokens=mo)
    # M1: lean only
    out = {"per_arm": res}
    if "lean" in roles:
        L, C = res["lean"], res.get("control")
        if L["n_calls"]:
            cap_ok = L["frac_at_max_output"] <= cfg["m1_cap_frac"]
            if L["reasoning_available"]:
                bud_ok = L["max_reasoning_tokens"] <= L["thinking_budget"] + 10
                basis = "reasoning_tokens"
            else:   # reasoning tokens are not in ATIF: proxy = no call exceeds the output cap and p99 is below control's
                bud_ok = L["max_call_tokens"] <= L["max_output_tokens"] + 8 and (C is None or C["p99"] is None or L["p99"] < C["p99"])
                basis = "proxy(max completion_tokens<=max_output+8 and p99<control p99)"
            out["M1"] = dict(applies=True, passed=bool(cap_ok and bud_ok), cap_ok=bool(cap_ok), budget_ok=bool(bud_ok), basis=basis)
        else:
            out["M1"] = dict(applies=True, passed=None, note="no trace data")
    else:
        out["M1"] = dict(applies=False)
    # M2 per treated arm vs control
    for a in ("decide", "lean"):
        if a not in roles:
            continue
        ca, cc = np.array([m["calls_to_first_edit"] for m in M[a]]), np.array([m["calls_to_first_edit"] for m in M["control"]])
        ra, rc = np.array([m["reads_before_first_edit"] for m in M[a]]), np.array([m["reads_before_first_edit"] for m in M["control"]])
        ok = np.isfinite(ca) & np.isfinite(cc)
        n = int(ok.sum())
        if n < 5:
            out[f"M2_{a}"] = dict(passed=None, n=n, note="fewer than 5 pairs with a source edit in both arms")
            continue
        md = float(np.median(ca[ok] - cc[ok]))
        okr = ok & np.isfinite(ra) & np.isfinite(rc)
        rel = float(ra[okr].mean() / rc[okr].mean() - 1) if okr.sum() and rc[okr].mean() > 0 else None
        p1 = md <= cfg["m2_edit_delta"]
        p2 = rel is not None and rel <= cfg["m2_reads_rel"]
        out[f"M2_{a}"] = dict(n=n, median_paired_diff_first_edit=md, median_x=float(np.median(ca[ok])), median_control=float(np.median(cc[ok])),
                              reads_rel_change=rel, first_edit_ok=bool(p1), reads_ok=bool(p2), passed=bool(p1 and p2))
    return out


def stop_rules(out, roles, n, cfg):
    res = {}
    mech = out["mechanism"]
    ctrl = out["arm_summary"]["control"]["resolved"]
    for a in ("decide", "lean"):
        if a not in roles:
            continue
        s = out["arm_summary"][a]["resolved"]
        harm = n >= cfg["stop_min_pairs"] and (ctrl - s) > cfg["stop_harm"]
        conf = out["config_errors"][a]["n_attributable"] >= cfg["stop_config"]
        m1 = mech["M1"].get("passed") if a == "lean" else None          # M1 not applicable to decide
        m2 = mech.get(f"M2_{a}", {}).get("passed")
        fut = n >= cfg["stop_min_pairs"] and (m2 is False) and (a == "decide" or m1 is False)
        res[a] = dict(stop=bool(harm or conf or fut), harm_resolved=bool(harm), resolved_deficit=int(ctrl - s),
                      config_errors=bool(conf), futility=bool(fut), n_pairs=n)
    return res


def gate(out, a, roles, n, cov, preflight, cfg):
    g = cfg["gate"][a]
    key = f"{g['contrast'][0]}_vs_{g['contrast'][1]}"
    tk = out["contrasts"][key][cfg["gate_metric"]]
    mech = out["mechanism"]
    m1 = mech["M1"].get("passed") if a == "lean" else None
    m2 = mech.get(f"M2_{a}", {}).get("passed")
    mech_ok = bool((m1 is True) or (m2 is True)) if a == "lean" else bool(m2 is True)
    rel, hi = tk.get("rel"), (tk.get("rel_ci") or [None, None])[1]
    tok_ok = rel is not None and hi is not None and rel <= -g["min_fall"] and hi < g["ci_hi_lt"]
    # order-effect guard: if a significant order effect exists the order-adjusted estimate must pass too
    oe = out["order"].get(key, {}).get(cfg["gate_metric"], {})
    adj_ok, adj_note = True, None
    if oe.get("identifiable") and oe.get("flag"):
        r2, c2 = oe["intercept_rel"], oe["intercept_rel_ci"]
        adj_ok = r2 is not None and c2[1] is not None and r2 <= -g["min_fall"] and c2[1] < g["ci_hi_lt"]
        adj_note = "order effect flagged: order-adjusted estimate must also pass"
    arm_sum, ctrl = out["arm_summary"][a], out["arm_summary"]["control"]
    res_ok = arm_sum["resolved"] >= ctrl["resolved"] - cfg["resolved_margin"]
    cfg_ok = out["config_errors"][a]["n_attributable"] == 0
    to_ok = arm_sum["timeouts"] <= ctrl["timeouts"]
    pf_ok = preflight == "pass"
    zero = arm_sum["mean_tokens"] is not None and arm_sum["mean_tokens"] == 0.0
    cov_ok = cov >= cfg["min_cov"]
    checks = dict(mechanism=mech_ok, tokens=bool(tok_ok and adj_ok and not zero), resolved_guard=bool(res_ok), no_config_errors=cfg_ok,
                  timeout_not_higher=bool(to_ok), trace_coverage=cov_ok, preflight=pf_ok)
    return dict(checks=checks, passed_all=all(checks.values()), tokens_rel=rel, tokens_ci_hi=hi, tokens_threshold_fall=g["min_fall"],
                tokens_ci_hi_must_be_below=g["ci_hi_lt"], order_adjusted_note=adj_note, preflight=preflight)


def decide(out, roles, n, cov, preflight, planned, cfg):
    acc = out["accounting"]
    stops = [a for a, s in out["stop_rules"].items() if s["stop"]]
    complete_enough = (n >= math.ceil(cfg["min_cov"] * planned) and not acc["pending_or_partial"] and acc["never_started"] == 0)
    d = dict(stop_arms=stops, eligible=bool(complete_enough))
    if not complete_enough:
        d["decision"] = "CONTINUE" if not stops else "STOP " + ",".join(stops)
        d["reason"] = (f"{n}/{planned} complete pairs, {len(acc['pending_or_partial'])} partial, {acc['never_started']} not started: "
                       "no submission decision before the full planned set is finished (no early stop for success)")
        return d
    ok = [a for a in ("lean", "decide") if a in out["gates"] and out["gates"][a]["passed_all"] and a not in stops]
    pf_blocked = [a for a in ("lean", "decide") if a in out["gates"] and a not in stops
                  and all(v for k, v in out["gates"][a]["checks"].items() if k != "preflight") and preflight != "pass"]
    if ok:
        d["decision"] = "SUBMIT lean-nt" if ok[0] == "lean" else "SUBMIT decide-nt"
        d["reason"] = "all gates pass (fallback order lean-nt, decide-nt)"
    elif pf_blocked:
        d["decision"] = "HOLD (all gates but CPU preflight pass for " + pf_blocked[0] + "; preflight=" + preflight + ")"
        d["reason"] = "run the CPU preflight and re-run with --preflight pass"
    else:
        d["decision"] = "SUBMIT neither"
        d["reason"] = "no arm passes every gate"
    return d


# ------------------------------------------------------------------------------------------------- power
def power_info(paired_json, n=43):
    d = json.load(open(paired_json))["continuous"]["completion_tokens"]
    m, sd = d["mean_base"], d["sd_arm"]
    rho = d["corr_arms"]
    out = dict(source=str(paired_json), n_pairs_source=None, n=n, mean=m, sd_arm=sd, corr_arms=rho, sd_diff_source=d["sd_diff"])
    gate_defs = {"lean": (0.08, 0.0), "decide": (0.05, 0.03)}
    rows = []
    for r in (rho, 0.6, 0.4, 0.2):
        sdd = sd * math.sqrt(2 * (1 - r))
        se = sdd / math.sqrt(n) / m
        row = dict(corr=r, sd_diff=sdd, rel_se=se, mde_80pct_2sided=(1.96 + 0.8416) * se)
        for g, (fall, hi) in gate_defs.items():
            thr = min(-fall, hi - 1.96 * se)
            row[f"{g}_gate_min_observed_change"] = thr
            row[f"{g}_power_at_true"] = {str(t): float(stats.norm.cdf((thr - (-t)) / se)) for t in (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)}
        rows.append(row)
    out["rows"] = rows
    return out


# ------------------------------------------------------------------------------------------------- output
def sanitize(o):
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _f(v, d=3, pct_=False):
    if v is None:
        return "n/a"
    return f"{v * 100:+.1f}%" if pct_ else f"{v:.{d}g}"


def _ci(c, pct_=False):
    return "[" + ", ".join(_f(x, 3, pct_) for x in c) + "]" if c else "n/a"


def markdown(out):
    L = []
    w = L.append
    if "error" in out:
        return f"# A/B report\n\nNo analysis possible: {out['error']}\n\nAccounting: {json.dumps(out.get('accounting'))}\n"
    acc = out["accounting"]
    w("# Paired A/B report (pre-registered analysis)\n")
    d = out["decision"]
    w(f"**Decision: {d['decision']}**  ({d['reason']})\n")
    w(f"Complete pairs: {acc['n_complete']}/{acc['planned']}; infra-missing: {len(acc['infra_missing'])}; partial/pending: "
      f"{len(acc['pending_or_partial'])}; not started: {acc['never_started']}; trace coverage {out['trace_coverage']:.0%}; "
      f"order known from: {out['order_source']}\n")
    w("## Arms (complete pairs only)\n")
    w("| arm | resolved | timeouts | empty patch | mean tokens | mean agent min | LLM calls | tool calls | est LLM s |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r, s in out["arm_summary"].items():
        w(f"| {r} | {s['resolved']} | {s['timeouts']} | {s['empty_patch']} | {_f(s['mean_tokens'], 5)} | {_f(s['mean_agent_min'])} | "
          f"{_f(s['mean_llm_calls'])} | {_f(s['mean_tool_calls'])} | {_f(s['mean_est_llm_s'], 4)} |")
    w("\n## Primary: paired total completion tokens (ratio of means; bootstrap 95% CI; Holm over 3 contrasts)\n")
    w("| contrast | n | mean x | mean y | change | 95% CI | median diff [CI] | p sign-flip (Holm) | p Wilcoxon (Holm) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, c in out["contrasts"].items():
        t = c["tokens"]
        w(f"| {k} | {t['n']} | {_f(t['mean_x'], 5)} | {_f(t['mean_y'], 5)} | {_f(t['rel'], pct_=True)} | {_ci(t['rel_ci'], True)} | "
          f"{_f(t['median_diff'], 4)} {_ci(t['median_diff_ci'])} | {_f(t['p_signflip'], 3)} ({_f(t['p_signflip_holm'], 3)}) | "
          f"{_f(t['p_wilcoxon'], 3)} ({_f(t['p_wilcoxon_holm'], 3)}) |")
    w("\n## Secondary (descriptive, unadjusted)\n")
    w("| contrast | metric | n | change (ratio of means) | 95% CI | median diff [CI] | p Wilcoxon |")
    w("|---|---|---|---|---|---|---|")
    for k, c in out["contrasts"].items():
        for m in ("agent_min", "est_llm_s", "llm_calls", "tool_calls", "tokens_per_call", "calls_to_first_edit",
                  "calls_to_first_edit_cens", "reads_before_first_edit", "tokens_to_first_edit", "sec_to_first_edit", "tokens_to_first_edit_cens", "tokens_noTO"):
            t = c[m]
            w(f"| {k} | {m} | {t['n']} | {_f(t['rel'], pct_=True)} | {_ci(t['rel_ci'], True)} | {_f(t['median_diff'], 3)} {_ci(t['median_diff_ci'])} | {_f(t['p_wilcoxon'], 3)} |")
    w("\n### Censoring-aware time (cap-censored): orderable-pair sign test\n")
    w("| contrast | informative pairs | treated shorter | treated longer | both censored (ties) | p |")
    w("|---|---|---|---|---|---|")
    for k, c in out["contrasts"].items():
        s = c["agent_min_orderable"]
        w(f"| {k} | {s['n_informative']} | {s['n_treated_shorter']} | {s['n_treated_longer']} | {s['n_tied_censored']} | {_f(s['p_sign'], 3)} |")
    w("\n### Binary outcomes (exact McNemar)\n")
    w("| contrast | outcome | events x | events y | only x | only y | p | rate diff [CI] |")
    w("|---|---|---|---|---|---|---|---|")
    for k, c in out["contrasts"].items():
        for m in ("resolved", "timeout", "empty_patch"):
            t = c[m]
            w(f"| {k} | {m} | {t['events_x']} | {t['events_y']} | {t['only_x']} | {t['only_y']} | {_f(t['p_mcnemar_exact'], 3)} | "
              f"{_f(t.get('rate_diff'), 3)} {_ci(t.get('rate_diff_ci'))} |")
    w("\n## Mechanism checks\n")
    mech = out["mechanism"]
    w("Per-arm call tokens: " + "; ".join(f"{r}: n={v['n_calls']} max={v['max_call_tokens']} p99={_f(v['p99'], 4)} at-cap={_f(v['frac_at_max_output'], 2)}"
                                          for r, v in mech["per_arm"].items()))
    w(f"\nM1 (lean): {json.dumps(mech['M1'])}")
    for a in ("decide", "lean"):
        if f"M2_{a}" in mech:
            w(f"\nM2 ({a}): {json.dumps(mech[f'M2_{a}'])}")
    w("\n## Order effect (regression of paired difference on arm-position difference)\n")
    for k, v in out["order"].items():
        for m, e in v.items():
            if e.get("identifiable"):
                w(f"- {k} {m}: slope {_f(e['slope'], 4)} per position step {_ci(e['slope_ci'])} p={_f(e['slope_p'], 3)}; "
                  f"order-adjusted effect {_f(e['intercept_rel'], pct_=True)} {_ci(e['intercept_rel_ci'], True)}{'  **FLAG**' if e['flag'] else ''}")
            else:
                w(f"- {k} {m}: not identifiable ({e.get('note')})")
    w("\n## Config-attributable errors\n")
    for r, e in out["config_errors"].items():
        w(f"- {r}: attributable={e['n_attributable']} (hard {len(e['hard'])}, soft-excess {len(e['soft_excess_over_control'])}), "
          f"raw soft runs={e['raw_soft_runs']}, runs without trace={e['runs_without_trace']}")
    w("\n## Stop rules\n")
    for a, s in out["stop_rules"].items():
        w(f"- {a}: {'STOP' if s['stop'] else 'continue'} (resolved deficit {s['resolved_deficit']}, harm={s['harm_resolved']}, config={s['config_errors']}, futility={s['futility']}, n={s['n_pairs']})")
    w("\n## Gates\n")
    for a, g in out["gates"].items():
        w(f"- {a}: {'PASS' if g['passed_all'] else 'FAIL'}  tokens {_f(g['tokens_rel'], pct_=True)} (CI hi {_f(g['tokens_ci_hi'], pct_=True)}; need <= -{g['tokens_threshold_fall']:.0%} and hi < {g['tokens_ci_hi_must_be_below']:+.0%}) checks={json.dumps(g['checks'])}")
    if out.get("warnings"):
        w("\n## Warnings\n")
        for x in out["warnings"]:
            w(f"- {x}")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------- CLI
def run_pipeline(args_ns, cfg=None):
    cfg = dict(cfg or PRE)
    warnings = []
    names = dict(DEFAULT_ARMS)
    for kv in args_ns.arm or []:
        k, v = kv.split("=", 1)
        names[k] = v
    roles = [r for r in args_ns.roles.split(",") if r]
    rows = load_rows(args_ns.results, warnings)
    traces = {}
    if args_ns.trace_metrics:
        traces.update(load_trace_metrics(args_ns.trace_metrics, warnings))
    if args_ns.traces:
        traces.update(load_traces(args_ns.traces, [names[r] for r in roles], warnings))
    tasks_filter = None
    if args_ns.tasks:
        tasks_filter = {x.strip() for x in Path(args_ns.tasks).read_text().splitlines() if x.strip()}
    launch_ids = set(args_ns.launch_ids.split(",")) if args_ns.launch_ids else None
    seen = {r["arm"] for r in rows}
    unknown = sorted(seen - {names[r] for r in roles})
    present_roles = [r for r in roles if names[r] in seen]
    if len(present_roles) < len(roles):
        warnings.append("arms without any result row dropped from the analysis: " + ",".join(r for r in roles if r not in present_roles))
    if unknown:
        warnings.append("result rows for other arms ignored: " + ",".join(unknown))
    cfg["seed"], cfg["n_boot"] = args_ns.seed, args_ns.n_boot
    runs, status = build_runs(rows, present_roles, names, traces, tasks_filter, launch_ids, cfg, warnings)
    out = analyze_runs(runs, status, present_roles, cfg, preflight=args_ns.preflight, planned=args_ns.planned or (len(tasks_filter) if tasks_filter else None),
                       tasks_planned=tasks_filter)
    out["warnings"] = warnings + out.get("warnings", [])
    out["arm_names"] = {r: names[r] for r in present_roles}
    return sanitize(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="*", help="results_<arm>.jsonl files, directories or globs")
    ap.add_argument("--traces", nargs="*", help="ATIF trace tarballs (.tar.gz), directories or .json files; chronological order")
    ap.add_argument("--trace-metrics", nargs="*", help="trace_metrics*.jsonl written by kernel_extract (alternative to --traces)")
    ap.add_argument("--roles", default="control,decide,lean")
    ap.add_argument("--arm", action="append", help="role=arm_name override, e.g. lean=edit-early-v7-lean-nt")
    ap.add_argument("--tasks", help="file with one instance_id per line (default: every non-dead row)")
    ap.add_argument("--planned", type=int, help="planned number of pairs (default 43 or len(--tasks))")
    ap.add_argument("--launch-ids", help="comma list; only rows of these launches")
    ap.add_argument("--preflight", choices=["pass", "fail", "unknown"], default="unknown")
    ap.add_argument("--seed", type=int, default=PRE["seed"])
    ap.add_argument("--n-boot", type=int, default=PRE["n_boot"])
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--power", metavar="PAIRED_RESULTS_JSON", help="print the power statement and exit")
    a = ap.parse_args(argv)
    if a.power:
        print(json.dumps(sanitize(power_info(a.power)), indent=1))
        return 0
    if not a.results:
        ap.error("--results is required")
    out = run_pipeline(a)
    md = markdown(out)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    if a.out_md:
        Path(a.out_md).write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
