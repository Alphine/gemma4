"""Unit tests for score.py  (python -m pytest research/teacher-gen/C-behaviour/test_score.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402

GOLD = ["pkg/mod.py"]
OK = json.dumps({"status": "ok", "stdout": ""})


def step(t, ct, calls, thought=100, pt=1000):
    return sc.Step(t=t, prompt_tokens=pt, completion_tokens=ct, thought_chars=thought, calls=calls)


def C(name, args, obs=OK):
    return sc.Call(name, args, obs)


def fast_traj():
    return sc.Traj([
        step(5, 80, [C("read_file", {"filepath": "pkg/mod.py", "start_line": 1, "end_line": 50})]),
        step(9, 120, [C("edit_file", {"filepath": "pkg/mod.py", "old_string": "a", "new_string": "b"})]),
        step(14, 90, [C("run_command", {"command": "timeout 60 python3 -m pytest tests/test_mod.py -x -q"})]),
        step(18, 60, [C("submit_patch", {})]),
        step(20, 30, []),
    ])


def slow_traj():
    steps = []
    t = 0
    for i in range(12):                         # the same read over and over, long thoughts, no edit
        t += 12
        steps.append(step(t, 500, [C("read_file", {"filepath": "pkg/mod.py", "start_line": 1, "end_line": 50})]))
    for i in range(3):
        t += 25
        steps.append(step(t, 1800, [C("run_command", {"command": "python3 repro.py"})]))
    return sc.Traj(steps)


def test_fast_metrics():
    m = sc.metrics(fast_traj(), GOLD)
    assert m["n_counted"] == 3 and m["n_llm"] == 5          # submit_patch is uncounted
    assert m["calls_to_gold_read"] == 1
    assert m["calls_to_good_edit"] == 2 and m["calls_to_test"] == 3 and m["calls_to_submit"] == 3
    assert m["submitted"] and m["edited"]
    assert m["repeat_exact"] == 0 and m["repeat_read"] == 0
    assert m["no_call_turns"] == 1
    assert m["verify_after_edit"] == 1
    assert m["max_turn_tokens"] == 120
    assert abs(m["sec_to_good_edit"] - 4) < 1e-9             # relative to the first step's clock


def test_repeats_and_reread_after_edit():
    t = sc.Traj([
        step(1, 10, [C("read_file", {"filepath": "a.py"})]),
        step(2, 10, [C("read_file", {"filepath": "a.py"})]),                       # exact repeat + covered re-read
        step(3, 10, [C("edit_file", {"filepath": "a.py", "old_string": "x", "new_string": "y"})]),
        step(4, 10, [C("read_file", {"filepath": "a.py"})]),                       # legitimate after an edit
    ])
    m = sc.metrics(t)
    assert m["repeat_exact"] == 1 and m["repeat_read"] == 1


def test_failed_and_malformed_calls():
    bad = json.dumps({"error": "Skill not found.", "error_code": "SKILL_NOT_FOUND"})
    err = json.dumps({"status": "error", "error_type": "FileReadError", "error_message": "x"})
    soft = json.dumps({"stdout": "NOT WRITTEN: expect text not found"})
    t = sc.Traj([
        step(1, 10, [C("run_skill_script", {"file_path": "scripts/edit_lines.py", "args": {"path": "a.py"}}, bad)]),
        step(2, 10, [C("read_file", {"filepath": "zz.py"}, err)]),
        step(3, 10, [C("run_skill_script", {"file_path": "scripts/edit_lines.py", "args": {"path": "a.py"}}, soft)]),
        step(4, 10, [C("run_skill_script", {"file_path": "scripts/edit_lines.py", "args": {"path": "a.py"}}, OK)]),
        step(5, 10, [C("submit_patch", {})]),
    ])
    m = sc.metrics(t, ["a.py"])
    assert m["malformed_calls"] == 1 and m["failed_calls"] == 3
    assert m["calls_to_edit"] == 1 and m["calls_to_good_edit"] == 4        # first two edit attempts were wasted


def test_scratch_files_are_not_source_edits():
    t = sc.Traj([step(1, 10, [C("write_file", {"filepath": "repro.py", "content": "x"})]),
                 step(2, 10, [C("edit_file", {"filepath": "tests/test_x.py", "old_string": "a", "new_string": "b"})]),
                 step(3, 10, [C("submit_patch", {})])])
    m = sc.metrics(t)
    assert not m["edited"] and m["scratch_writes"] == 2


def test_score_orders_fast_above_slow_and_required_rule():
    spec = sc.DEFAULT_SPEC
    s_fast = sc.score_traj(fast_traj(), GOLD, spec)
    s_slow = sc.score_traj(slow_traj(), GOLD, spec)
    assert s_fast["score"] > 0.9 and s_fast["fit"]
    assert s_slow["score"] == 0.0 and not s_slow["required_ok"]        # never edited / never submitted
    # a verbose-but-correct trajectory scores lower than the terse one
    verbose = fast_traj()
    for s in verbose.steps:
        s.completion_tokens = 2000
    assert sc.score_traj(verbose, GOLD, spec)["score"] < s_fast["score"]


def test_credit_is_monotone_and_bounded():
    lim = [10, 30]
    vals = [sc._credit(v, lim) for v in (0, 10, 15, 20, 30, 99)]
    assert vals == sorted(vals, reverse=True) and vals[0] == 1.0 and vals[-1] == 0.0
    assert sc._credit(None, lim) == 0.0


def test_est_sec_uses_tokens_then_chars_fallback():
    t = fast_traj()
    m = sc.metrics(t, GOLD)
    k = 4                                                            # steps up to and including submit
    expect = sc.LAT_MODAL_C8[0] * k + sc.LAT_MODAL_C8[1] * (80 + 120 + 90 + 60)
    assert abs(m["est_sec_modal_c8"] - expect) < 1e-6
    t2 = fast_traj()
    for s in t2.steps:
        s.completion_tokens = None                                   # teacher output without usage
    m2 = sc.metrics(t2, GOLD)
    assert m2["completion_tokens"] > 0 and m2["est_sec_kaggle_c1"] > 0


def test_from_atif_roundtrip():
    d = {"steps": [
        {"step_id": 1, "source": "system", "message": "sys"},
        {"step_id": 2, "source": "user", "message": "task"},
        {"step_id": 3, "source": "agent", "message": "thinking about it", "metrics": {"prompt_tokens": 500, "completion_tokens": 77},
         "tool_calls": [{"tool_call_id": "c", "function_name": "read_file", "arguments": {"filepath": "pkg/mod.py"},
                         "extra": {"elapsed_s": 10.0}}], "observation": {"content": OK}},
        {"step_id": 4, "source": "agent", "message": "done", "metrics": {"prompt_tokens": 600, "completion_tokens": 20}},
    ]}
    tr = sc.from_atif(d)
    assert len(tr.steps) == 2 and tr.steps[0].completion_tokens == 77 and tr.steps[0].t == 10.0
    assert sc.metrics(tr, GOLD)["calls_to_gold_read"] == 1


def test_from_llm_calls_recorder_format():
    def rec(t, lat, tcs, ct):
        return {"t_rel": t, "latency_s": lat, "usage": {"prompt_tokens": 900, "completion_tokens": ct},
                "response": {"role": "assistant", "content": None, "reasoning_content": "hmm", "tool_calls": tcs}, "messages": []}
    tc1 = {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"filepath": "pkg/mod.py"})}}
    tc2 = {"id": "t2", "type": "function", "function": {"name": "submit_patch", "arguments": "{}"}}
    r1, r2 = rec(0.0, 2.0, [tc1], 50), rec(3.0, 1.5, [tc2], 30)
    r2["messages"] = [{"role": "tool", "tool_call_id": "t1", "content": OK}]
    tr = sc.from_llm_calls([r1, r2])
    m = sc.metrics(tr, GOLD)
    assert m["n_llm"] == 2 and m["calls_to_gold_read"] == 1 and m["submitted"]
    assert tr.steps[0].calls[0].obs == OK


def test_spec_file_loads_and_scores():
    spec = sc.load_spec()
    assert "limits" in spec and "weights" in spec
    out = sc.score_traj(fast_traj(), GOLD, spec, "relaxed")
    assert 0.0 <= out["score"] <= 1.0
