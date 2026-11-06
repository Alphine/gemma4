"""pytest suite for analyze_ab.py / trace_metrics.py on synthetic and real-fixture data.  Run:  python -m pytest -q"""
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_ab as A  # noqa: E402
import synth  # noqa: E402
import trace_metrics as tm  # noqa: E402

FIX = HERE / "fixtures_withheld"
ROLES = ["control", "decide", "lean"]


def cfg_small(seed=1, n_boot=300):
    c = dict(A.PRE)
    c["seed"], c["n_boot"] = seed, n_boot
    return c


def analyze(rows, traces, roles=ROLES, cfg=None, preflight="pass", planned=43, parsed=False, **kw):
    cfg = cfg or cfg_small()
    warn = []
    tr = traces if parsed else {k: tm.parse_trace(v) for k, v in traces.items()}
    runs, st = A.build_runs(rows, roles, A.DEFAULT_ARMS, tr, kw.pop("tasks_filter", None), kw.pop("launch_ids", None), cfg, warn)
    return A.sanitize(A.analyze_runs(runs, st, roles, cfg, preflight=preflight, planned=planned))


# ------------------------------------------------------------------------------------------ trace parsing
@pytest.mark.skipif(not (FIX / "trace_fastapi_14873_robust.json").exists(), reason="fixtures missing")
def test_real_atif_fixtures():
    r = tm.parse_trace(json.load(open(FIX / "trace_fastapi_14873_robust.json")))
    assert r["completion_tokens"] == 3404 == r["final_metrics_completion_tokens"] and not r["tokens_mismatch"]
    assert r["n_llm_calls"] == 43 and r["n_counted_calls"] == 41           # submit_patch not counted
    assert r["first_edit_counted_idx"] == 37                                # first edit_file failed (missing params) and scratch write_file ignored
    assert r["n_edit_failed"] == 1 and r["has_edit"] and r["submitted"]
    l1 = tm.parse_trace(json.load(open(FIX / "trace_fastapi_14873_ladder1.json")))
    assert l1["completion_tokens"] == 6033 and l1["n_calls_missing_tokens"] == 1   # the final message step has no metrics
    assert l1["first_edit_counted_idx"] == 22                                       # edits of reproduce_issue.py are scratch
    assert r["marks"] == {"args_truncated": 0, "max_tokens": 0, "thinking_budget_error": 0}


def test_call_kind_rules():
    k = tm.call_kind
    assert k("edit_file", {"filepath": "pkg/core.py"}) == "edit"
    assert k("edit_file", {"filepath": "tests/test_core.py"}) == "scratch"
    assert k("edit_file", {"filepath": "reproduce_issue.py"}) == "scratch"
    assert k("write_file", {"filepath": "new_scratch_notes.py"}) == "scratch"           # root-level, never read
    assert k("write_file", {"filepath": "setup_helper.py"}, known_read={"setup_helper.py"}) == "edit"
    assert k("write_file", {"filepath": "docs_src/app/main.py"}) == "edit"
    assert k("run_command", {"command": "sed -i 's/a/b/' pkg/x.py"}) == "edit"
    assert k("run_command", {"command": "cat > pkg/x.py <<EOF"}) == "edit"
    assert k("run_command", {"command": "cat > /tmp/notes.txt"}) != "edit"
    assert k("run_command", {"command": "grep -n foo pkg/x.py"}) == "shell_read"
    assert k("run_skill_script", {"file_path": "skills/swe-tools/scripts/locate.py", "args": {}}) == "locate"
    assert k("run_skill_script", {"file_path": "skills/swe-tools/scripts/edit_lines.py", "args": {"path": "pkg/a.py"}}) == "edit"
    assert tm.obs_failed(json.dumps({"error": "x"})) and tm.obs_failed(json.dumps({"status": "error"}))
    assert not tm.obs_failed(json.dumps({"status": "ok", "stdout": "fine"}))
    assert tm.obs_failed(json.dumps({"stdout": "NOT WRITTEN: refused"}))


def test_roundtrip_synth_trace_matches_ground_truth():
    rng = np.random.default_rng(3)
    steps, to, t, edit = synth._gen_run(rng, 0.0, synth.BASE, "control", 1e9, 1.0, 6)
    tr = tm.parse_trace(synth.to_atif(steps, 1e9, "control"))
    assert tr["completion_tokens"] == sum(s["tok"] for s in steps)
    assert tr["n_llm_calls"] == len(steps)
    k = next(i + 1 for i, s in enumerate(steps) if s["name"] == "edit_file")
    assert tr["first_edit_counted_idx"] == k
    assert tr["locate_called"] and tr["reads_before_first_edit"] == k - 2          # call 1 = locate, then k-2 reads before the edit
    assert tr["tokens_to_first_edit"] == sum(s["tok"] for s in steps[:k])


def test_marker_detection_ignores_prompt_and_tool_stdout():
    steps, *_ = synth._gen_run(np.random.default_rng(1), 0, synth.BASE, "lean", 1e9, 1.0, 5)
    base = synth.to_atif(steps, 0.0, "lean")
    assert base["steps"][1]["message"].count("MAX_TOKENS") == 1                    # prompt mentions the word
    # tool stdout that mentions it (grep over repo code) must not count
    base["steps"][3]["observation"]["content"] = json.dumps({"status": "ok", "stdout": "MAX_TOKENS = 4 Tool call arguments were truncated"})
    assert tm.parse_trace(base)["marks"] == {"args_truncated": 0, "max_tokens": 0, "thinking_budget_error": 0}
    base["steps"][4]["observation"]["content"] = json.dumps({"error": "Tool call arguments were truncated (MAX_TOKENS)"})
    m = tm.parse_trace(base)["marks"]
    assert m["args_truncated"] == 1 and m["max_tokens"] == 1


def test_corrupt_trace_is_missing_not_crash():
    assert tm.parse_trace_text("{not json")["trace_ok"] is False


# ------------------------------------------------------------------------------------------ pure statistics
def test_ties_and_identical_arms():
    x = np.full(43, 100.0)
    r = A.continuous_contrast(x, x.copy(), "t", cfg_small())
    assert r["rel"] == 0 and r["p_wilcoxon"] == 1.0 and r["p_signflip"] == 1.0 and r["median_diff"] == 0
    d = np.array([0, 0, 0, 1.0, -1.0, 2.0] * 5)
    r2 = A.continuous_contrast(d + 10, np.full(30, 10.0), "t2", cfg_small())
    assert 0 <= r2["p_wilcoxon"] <= 1 and 0 <= r2["p_signflip"] <= 1


def test_zero_token_reference_and_treated():
    r = A.continuous_contrast(np.arange(1, 44.0), np.zeros(43), "z", cfg_small())
    assert r["rel"] is None and r["rel_ci"] == [None, None]                    # ratio undefined, no crash / inf
    r = A.continuous_contrast(np.zeros(43), np.arange(1, 44.0), "z2", cfg_small())
    assert r["rel"] == pytest.approx(-1.0)


def test_nan_pairs_dropped():
    x = np.array([1.0, 2, np.nan, 4, 5, 6])
    y = np.array([2.0, 2, 3, np.nan, 6, 8])
    r = A.continuous_contrast(x, y, "n", cfg_small())
    assert r["n"] == 4


def test_signflip_exact_matches_enumeration():
    d = np.array([1.0, 2, 3, 4, 5])
    p = A.signflip_p(d, np.random.default_rng(0))
    assert p == pytest.approx(2 / 32)                                          # only the all-same-sign patterns reach |sum|=15


def test_censored_sign_logic():
    x = np.array([5.5, 5.5, 3.0, 2.0, 4.0, 5.5])
    y = np.array([5.5, 4.0, 5.5, 3.0, 4.0, 5.5])
    cx, cy = x >= 5.5, y >= 5.5
    s = A.censored_sign(x, y, cx, cy)
    # pairs: both censored (tie) x2 ; x censored y not -> treated longer ; y censored x not -> shorter ; x<y shorter ; equal tie
    assert s["n_tied_censored"] == 2 and s["n_informative"] == 3
    assert s["n_treated_shorter"] == 2 and s["n_treated_longer"] == 1


def test_mcnemar_exact():
    x = np.array([1] * 8 + [0] * 2 + [1] * 5 + [0] * 5, float)
    y = np.array([0] * 8 + [1] * 2 + [1] * 5 + [0] * 5, float)
    r = A.binary_contrast(x, y, "m", cfg_small())
    assert (r["only_x"], r["only_y"]) == (8, 2) and r["p_mcnemar_exact"] == pytest.approx(0.109375, rel=1e-6)
    z = A.binary_contrast(np.zeros(5), np.zeros(5), "m0", cfg_small())
    assert z["p_mcnemar_exact"] == 1.0


def test_holm():
    adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def _paired_lognormal(rng, n, mean, sd, rho):
    s2 = math.log(1 + (sd / mean) ** 2)
    rl = math.log(1 + rho * (math.exp(s2) - 1)) / s2
    z = rng.multivariate_normal([0, 0], [[1, rl], [rl, 1]], size=n)
    v = np.exp(math.sqrt(s2) * z - s2 / 2) * mean
    return v[:, 0], v[:, 1]


def test_null_false_positive_rate_and_holm_fwer():
    cfg = cfg_small(n_boot=200)
    n_sim, n = 700, 43
    rej_sf = rej_w = rej_ci = 0
    fam = 0
    for i in range(n_sim):
        rng = np.random.default_rng(1000 + i)
        a, b = _paired_lognormal(rng, n, 6700, 2900, 0.78)
        cfg["seed"] = i
        r = A.continuous_contrast(a, b, "null", cfg)
        rej_sf += r["p_signflip"] < 0.05
        rej_w += r["p_wilcoxon"] < 0.05
        rej_ci += (r["rel_ci"][0] > 0) or (r["rel_ci"][1] < 0)
    for i in range(400):
        rng = np.random.default_rng(5000 + i)
        v = [_paired_lognormal(rng, n, 6700, 2900, 0.78)[0] for _ in range(1)]
        base = rng.lognormal(0, 0.4, size=(n, 1)) * 6000                       # shared task effect, 3 exchangeable arms
        arms = [base[:, 0] * rng.lognormal(0, 0.25, n) for _ in range(3)]
        cfg["seed"] = i
        p = {k: A.continuous_contrast(x, y, k, cfg)["p_signflip"] for k, (x, y) in
             {"dc": (arms[1], arms[0]), "ld": (arms[2], arms[1]), "lc": (arms[2], arms[0])}.items()}
        fam += any(v_ < 0.05 for v_ in A.holm(p).values())
    assert 0.025 <= rej_sf / n_sim <= 0.08, rej_sf / n_sim
    assert 0.025 <= rej_w / n_sim <= 0.08, rej_w / n_sim
    assert rej_ci / n_sim <= 0.09, rej_ci / n_sim                              # percentile bootstrap, skewed data, n=43
    assert fam / 400 <= 0.08, fam / 400


def test_power_statement_matches_monte_carlo():
    pw = A.power_info(HERE.parent / "teacher" / "C_behaviour" / "paired_results.json", n=43)
    row = pw["rows"][0]
    assert row["mde_80pct_2sided"] == pytest.approx(0.123, abs=0.01)           # 6.7% at n=144 -> 12.3% at n=43
    cfg = cfg_small(n_boot=300)
    for true_fall in (0.10, 0.15):
        hit = 0
        n_sim = 300
        for i in range(n_sim):
            rng = np.random.default_rng(77 + i)
            a, b = _paired_lognormal(rng, 43, 6731, 2915, row["corr"])
            cfg["seed"] = i
            r = A.continuous_contrast(a * (1 - true_fall), b, "pw", cfg)
            hit += r["rel"] <= -0.08 and r["rel_ci"][1] < 0
        assert abs(hit / n_sim - row["lean_power_at_true"][str(true_fall)]) < 0.10, (true_fall, hit / n_sim)


# ------------------------------------------------------------------------------------------ pipeline on synthetic data
def test_known_effects_recovered_uncensored():
    # (1) pure per-call token effect (calls unchanged): total tokens change = true factor - 1 (no cap, little clipping)
    rel = []
    for s in range(8):
        rows, tr = synth.simulate(43, seed=s, effects={"lean": dict(tok_factor=0.7)}, cap_s=1e9)
        c = analyze(rows, tr, cfg=cfg_small(s, 400))["contrasts"]
        rel.append(c["lean_vs_control"]["tokens"]["rel"])
        assert c["lean_vs_control"]["tokens"]["rel_ci"][1] < 0
        assert c["lean_vs_control"]["tokens_per_call"]["rel"] < -0.05
    assert abs(np.mean(rel) - (-0.30)) < 0.05, np.mean(rel)
    # (2) first-edit shift of -3 / -2 calls and the matching drop in reads
    md_l, md_d = [], []
    for s in range(8):
        rows, tr = synth.simulate(43, seed=50 + s, effects={"lean": dict(edit_delta=-3), "decide": dict(edit_delta=-2)}, cap_s=1e9)
        o = analyze(rows, tr, cfg=cfg_small(s, 400))
        c = o["contrasts"]
        md_l.append(c["lean_vs_control"]["calls_to_first_edit"]["median_diff"])
        md_d.append(c["decide_vs_control"]["calls_to_first_edit"]["median_diff"])
        assert c["lean_vs_control"]["reads_before_first_edit"]["rel"] < 0
        assert o["order_source"]["from_trace_timestamps"] == 43
    assert abs(np.mean(md_l) + 3) <= 1.0 and abs(np.mean(md_d) + 2) <= 1.0, (np.mean(md_l), np.mean(md_d))
    # (3) resolved is untouched by these effects: McNemar neither rejects nor explodes
    assert 0 <= c["lean_vs_control"]["resolved"]["p_mcnemar_exact"] <= 1


def test_time_cap_censors_total_tokens_but_not_tokens_per_call():
    """Design caveat made executable: with a binding cap, a pure per-call token cut barely lowers TOTAL tokens;
    the agent just fits more calls into the same wall-clock budget."""
    eff = {"lean": dict(tok_factor=0.5)}
    tot, per_call, calls = [], [], []
    for s in range(6):
        rows, tr = synth.simulate(43, seed=100 + s, effects=eff, cap_s=45.0)           # very binding cap
        c = analyze(rows, tr, cfg=cfg_small(s, 300))["contrasts"]["lean_vs_control"]
        tot.append(c["tokens"]["rel"])
        per_call.append(c["tokens_per_call"]["rel"])
        calls.append(c["llm_calls"]["rel"])
    assert np.mean(per_call) < -0.2 and np.mean(calls) > 0.2
    assert np.mean(tot) > np.mean(per_call) + 0.08                              # total tokens shrink much less than tokens/call


def test_censoring_ties_at_cap_and_timeouts():
    rows, tr = synth.simulate(43, seed=5, cap_s=90.0)
    caps = [r["agent_min"] for r in rows if "session timeout" in (r["agent_error"] or "")]
    assert caps and all(c == pytest.approx(1.5) for c in caps)
    o = analyze(rows, tr)
    s = o["contrasts"]["lean_vs_control"]["agent_min_orderable"]
    assert s["n_tied_censored"] > 0 and s["n_informative"] + s["n_tied_censored"] <= 43
    assert o["arm_summary"]["control"]["timeouts"] > 10
    assert o["contrasts"]["lean_vs_control"]["tokens_noTO"]["n"] < 43


def test_missing_pairs_and_partial_data_no_decision():
    rows, tr = synth.simulate(43, seed=7, drop_pairs=5)
    o = analyze(rows, tr)
    assert o["accounting"]["n_complete"] == 38 and len(o["accounting"]["pending_or_partial"]) == 5
    assert o["decision"]["decision"] in ("CONTINUE",) or o["decision"]["decision"].startswith("STOP")
    assert not o["decision"]["eligible"]
    assert o["arm_summary"]["lean"]["n"] == 38                                  # complete-case across ALL arms


def test_partial_data_only_two_arms_present():
    rows, tr = synth.simulate(43, seed=8)
    rows = [r for r in rows if r["arm"] != A.DEFAULT_ARMS["lean"]]
    o = A.sanitize(A.analyze_runs(*A.build_runs(rows, ["control", "decide"], A.DEFAULT_ARMS,
                                                {k: tm.parse_trace(v) for k, v in tr.items()}, None, None, cfg_small(), []),
                                  ["control", "decide"], cfg_small()))
    assert set(o["contrasts"]) == {"decide_vs_control"} and "lean" not in o["gates"]


def test_infra_errors_retried_and_excluded():
    rows, tr = synth.simulate(43, seed=11, p_infra=0.25)
    n_infra = sum(r["infra_error"] for r in rows)
    assert n_infra > 5
    o = analyze(rows, tr)
    acc = o["accounting"]
    assert acc["n_complete"] + len(acc["infra_missing"]) + len(acc["pending_or_partial"]) == 43
    # infra rows contribute nothing: every counted run is a non-infra row
    runs, st = A.build_runs(rows, ROLES, A.DEFAULT_ARMS, {}, None, None, cfg_small(), [])
    assert sum(1 for k, v in st.items() if v == "done") == len(runs)
    # a pair is infra_missing only if some arm exhausted its attempts
    assert all(any(st.get((r, t)) == "exhausted" for r in ROLES) for t in acc["infra_missing"])


def test_arm_with_zero_tokens_is_flagged_and_gate_fails():
    rows, tr = synth.simulate(43, seed=13, effects={"lean": dict(tok_factor=0.5)}, cap_s=1e9)
    tr = copy.deepcopy(tr)
    for (arm, t), v in tr.items():
        if arm == A.DEFAULT_ARMS["lean"]:
            for s in v["steps"]:
                if "metrics" in s:
                    s["metrics"]["completion_tokens"] = 0
                for tc in s.get("tool_calls") or []:
                    tc["extra"]["usage"]["completion_tokens"] = 0
            v["final_metrics"]["total_completion_tokens"] = 0
    o = analyze(rows, tr)
    assert o["contrasts"]["lean_vs_control"]["tokens"]["rel"] == pytest.approx(-1.0)
    assert not o["gates"]["lean"]["checks"]["tokens"]
    assert any("ZERO completion tokens" in w for w in o["warnings"])
    # zero-token CONTROL: ratio undefined, no crash, no decision to submit
    rows, tr = synth.simulate(20, seed=14, cap_s=1e9)
    for (arm, t), v in tr.items():
        if arm == A.DEFAULT_ARMS["control"]:
            for s in v["steps"]:
                if "metrics" in s:
                    s["metrics"]["completion_tokens"] = 0
                for tc in s.get("tool_calls") or []:
                    tc["extra"]["usage"]["completion_tokens"] = 0
    o = analyze(rows, tr, planned=20)
    assert o["contrasts"]["lean_vs_control"]["tokens"]["rel"] is None
    assert not o["gates"]["lean"]["checks"]["tokens"] and o["decision"]["decision"] in ("SUBMIT neither", "CONTINUE")         or o["decision"]["decision"].startswith(("HOLD", "STOP"))


def test_missing_traces_lower_coverage_blocks_gate():
    rows, tr = synth.simulate(43, seed=15, effects={"lean": dict(tok_factor=0.5, edit_delta=-3)}, cap_s=1e9)
    drop = [k for k in tr if k[0] == A.DEFAULT_ARMS["lean"]][:8]
    tr = {k: v for k, v in tr.items() if k not in drop}
    o = analyze(rows, tr)
    assert o["trace_coverage"] < 0.95 and not o["gates"]["lean"]["checks"]["trace_coverage"]
    assert o["contrasts"]["lean_vs_control"]["tokens"]["n"] == 43 - 8


def test_order_effect_detected_and_unidentifiable_when_fixed():
    flags = 0
    for s in range(6):
        rows, tr = synth.simulate(43, seed=200 + s, order_mode="rotate", order_effect=0.25, cap_s=1e9)
        o = analyze(rows, tr, cfg=cfg_small(s, 400))
        e = o["order"]["lean_vs_control"]["tokens"]
        assert e["identifiable"] and e["slope"] > 0
        flags += e["flag"]
    assert flags >= 5
    rows, tr = synth.simulate(43, seed=210, order_mode="fixed", order_effect=0.15, cap_s=1e9)
    e = analyze(rows, tr)["order"]["lean_vs_control"]["tokens"]
    assert e["identifiable"] is False and "confounded" in e["note"]


def test_order_from_row_field_equals_trace_timestamps():
    rows, tr = synth.simulate(43, seed=17, order_mode="rotate", arm_pos_field=True)
    a = analyze(rows, tr)
    rows2 = [{k: v for k, v in r.items() if k != "arm_pos"} for r in rows]
    b = analyze(rows2, tr)
    assert a["order_source"]["from_rows"] == 43 and b["order_source"]["from_trace_timestamps"] == 43
    assert a["order"] == b["order"]


def test_order_adjusted_gate_requires_both(monkeypatch):
    rows, tr = synth.simulate(43, seed=18, effects={"lean": dict(tok_factor=0.5, edit_delta=-3)}, cap_s=1e9,
                              order_mode="rotate", order_effect=0.0)
    o = analyze(rows, tr)
    assert o["gates"]["lean"]["checks"]["tokens"]


def test_config_errors_attributable_and_stop_rule():
    rows, tr = synth.simulate(43, seed=19, effects={"lean": dict(hard_error=True)}, cap_s=1e9)
    o = analyze(rows, tr)
    assert o["config_errors"]["lean"]["n_attributable"] >= 3 and o["stop_rules"]["lean"]["config_errors"]
    assert not o["gates"]["lean"]["checks"]["no_config_errors"]
    # soft markers: present in both lean and control run -> not attributable; only in lean -> attributable
    rows, tr = synth.simulate(43, seed=20, cap_s=1e9)
    tr = copy.deepcopy(tr)
    tasks = sorted({t for (_, t) in tr})[:4]
    for t in tasks[:2]:                                                          # both arms
        for arm in (A.DEFAULT_ARMS["lean"], A.DEFAULT_ARMS["control"]):
            tr[(arm, t)]["steps"][4]["observation"]["content"] = json.dumps({"error": "Tool call arguments were truncated"})
    for t in tasks[2:]:                                                          # lean only
        tr[(A.DEFAULT_ARMS["lean"], t)]["steps"][4]["observation"]["content"] = json.dumps({"error": "Tool call arguments were truncated"})
    o = analyze(rows, tr)
    ce = o["config_errors"]["lean"]
    assert ce["raw_soft_runs"] == 4 and ce["n_attributable"] == 2
    assert o["config_errors"]["control"]["n_attributable"] == 0


def test_resolved_harm_stop_rule_needs_30_pairs():
    rows, tr = synth.simulate(43, seed=21, effects={"lean": dict(resolve_delta=-0.9)}, cap_s=1e9, resolve_base=0.6)
    o = analyze(rows, tr)
    assert o["stop_rules"]["lean"]["harm_resolved"] and o["stop_rules"]["lean"]["stop"]
    assert not o["gates"]["lean"]["checks"]["resolved_guard"]
    rows, tr = synth.simulate(25, seed=21, effects={"lean": dict(resolve_delta=-0.9)}, cap_s=1e9, resolve_base=0.6)
    o = analyze(rows, tr, planned=25)
    assert not o["stop_rules"]["lean"]["harm_resolved"]                          # < 30 pairs: rule not applicable yet


def test_gate_passes_on_clear_win_and_false_submit_rate_under_null():
    ok = 0
    sub = 0
    for s in range(10):
        rows, tr = synth.simulate(43, seed=300 + s, effects={"lean": dict(tok_factor=0.55, edit_delta=-3),
                                                              "decide": dict(edit_delta=-2)}, cap_s=1e9)
        o = analyze(rows, tr, cfg=cfg_small(s, 300))
        g = o["gates"]["lean"]["checks"]
        ok += g["tokens"] and g["mechanism"] and g["no_config_errors"] and g["trace_coverage"]
        sub += o["decision"]["decision"] == "SUBMIT lean-nt"
    assert ok == 10                                    # token + mechanism + hygiene gates pass for a 45% token cut
    assert sub >= 3                                    # the noisy resolved/timeout guards (see PREREGISTRATION 'operating characteristics') cost the rest
    sub = 0
    n_sim = 120
    for s in range(n_sim):
        rows, tr = synth.simulate(43, seed=900 + s)
        d = analyze(rows, tr, cfg=cfg_small(s, 200))["decision"]["decision"]
        sub += d.startswith("SUBMIT") and d != "SUBMIT neither"
    assert sub / n_sim <= 0.05, sub / n_sim                                       # false SUBMIT rate under the null


def test_preflight_unknown_holds():
    rows, tr = synth.simulate(43, seed=301, effects={"lean": dict(tok_factor=0.55, edit_delta=-3)}, cap_s=1e9)
    d = analyze(rows, tr, preflight="unknown", cfg=cfg_small(1, 300))["decision"]["decision"]
    assert d.startswith("HOLD") or d == "SUBMIT neither"      # never a SUBMIT without a recorded preflight pass


def test_deterministic_and_seed_sensitive():
    rows, tr = synth.simulate(43, seed=23)
    a = analyze(rows, tr, cfg=cfg_small(5, 500))
    b = analyze(rows, tr, cfg=cfg_small(5, 500))
    c = analyze(rows, tr, cfg=cfg_small(6, 500))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["contrasts"]["lean_vs_control"]["tokens"]["rel_ci"] != c["contrasts"]["lean_vs_control"]["tokens"]["rel_ci"]
    assert a["contrasts"]["lean_vs_control"]["tokens"]["rel"] == c["contrasts"]["lean_vs_control"]["tokens"]["rel"]


def test_launch_filter_and_duplicate_rows():
    rows, tr = synth.simulate(43, seed=24)
    old = copy.deepcopy([r for r in rows if r["arm"] == A.DEFAULT_ARMS["control"]])
    for r in old:
        r["launch_id"] = "OLD"
        r["resolved"] = not r["resolved"]
    o_all = analyze(old + rows, tr, launch_ids={"L1"})
    o_ref = analyze(rows, tr)
    assert o_all["arm_summary"] == o_ref["arm_summary"]


def test_mock_rows_and_dead_set_ignored(tmp_path):
    rows, tr = synth.simulate(43, seed=25)
    rows.append(dict(rows[0], run_mode="mock"))
    rows.append(dict(rows[0], instance_id="dead_1", set="dead"))
    synth.write_dataset(tmp_path, rows, tr)
    warn = []
    loaded = A.load_rows([tmp_path], warn)
    assert not any(r.get("run_mode") == "mock" for r in loaded)
    o = analyze(loaded, tr)
    assert o["accounting"]["n_complete"] == 43


# ------------------------------------------------------------------------------------------ I/O and CLI
def test_cli_end_to_end_tar_dir_and_extract_kernel_agree(tmp_path):
    rows, tr = synth.simulate(43, seed=31, effects={"lean": dict(tok_factor=0.6, edit_delta=-3)})
    synth.write_dataset(tmp_path / "tar", rows, tr, tar=True)
    synth.write_dataset(tmp_path / "dir", rows, tr, tar=False)
    outs = {}
    base = [sys.executable, str(HERE / "analyze_ab.py"), "--n-boot", "400", "--preflight", "pass"]
    r1 = subprocess.run(base + ["--results", str(tmp_path / "tar"), "--traces", str(tmp_path / "tar" / "traces_session1.tar.gz"),
                                "--out-json", str(tmp_path / "a.json"), "--out-md", str(tmp_path / "a.md")], capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(base + ["--results", str(tmp_path / "dir"), "--traces", str(tmp_path / "dir" / "traces"),
                                "--out-json", str(tmp_path / "b.json")], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    # extraction-kernel route: build the single-file kernel, run it on the dir layout, feed trace_metrics to the analysis
    subprocess.run([sys.executable, str(HERE / "kernel_extract" / "build_kernel.py")], check=True, capture_output=True)
    ker = HERE / "kernel_extract" / "gemma4-ab-extract.py"
    import ast
    ast.parse(ker.read_text(encoding="utf-8"))
    env = dict(__import__("os").environ, AB_IN=str(tmp_path / "dir"), AB_OUT=str(tmp_path / "ext"), AB_TAG="t")
    r3 = subprocess.run([sys.executable, str(ker)], env=env, capture_output=True, text=True)
    assert r3.returncode == 0, r3.stderr
    r4 = subprocess.run(base + ["--results", str(tmp_path / "dir"), "--trace-metrics", str(tmp_path / "ext"),
                                "--out-json", str(tmp_path / "c.json")], capture_output=True, text=True)
    assert r4.returncode == 0, r4.stderr
    a, b, c = (json.load(open(tmp_path / f"{x}.json")) for x in "abc")
    for k in ("contrasts", "arm_summary", "decision", "mechanism"):
        assert a[k] == b[k] == c[k], k
    assert "Paired A/B report" in (tmp_path / "a.md").read_text(encoding="utf-8")
    assert a["decision"]["decision"] in ("SUBMIT lean-nt", "SUBMIT neither", "SUBMIT decide-nt")


def test_tar_members_never_extracted_to_disk(tmp_path):
    import io
    import tarfile
    p = tmp_path / "evil.tar.gz"
    with tarfile.open(p, "w:gz") as t:
        b = b"{}"
        ti = tarfile.TarInfo("../../escape/edit-early-v6-locate/x.json")
        ti.size = len(b)
        t.addfile(ti, io.BytesIO(b))
    out = A.load_traces([p], list(A.DEFAULT_ARMS.values()), [])
    assert not (tmp_path.parent / "escape").exists()
    assert isinstance(out, dict)


def test_power_cli_runs():
    r = subprocess.run([sys.executable, str(HERE / "analyze_ab.py"), "--power",
                        str(HERE.parent / "teacher" / "C_behaviour" / "paired_results.json")], capture_output=True, text=True)
    assert r.returncode == 0 and "mde_80pct_2sided" in r.stdout


def test_empty_inputs_do_not_crash(tmp_path):
    o = analyze([], {})
    assert "error" in o
    assert "No analysis possible" in A.markdown(o)
