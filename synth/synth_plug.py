"""pytest plugin used by the synthetic-task factory (loaded with `-p synth_plug`, PYTHONPATH=<this dir>).

Env:
  SYNTH_OUT    JSON file to write at session end
  SYNTH_TRACE  os.pathsep-separated absolute directory prefixes: record, per test, which source lines ran
               (sys.monitoring LINE events, each location reported once per test, then DISABLEd)
  SYNTH_ALARM  per-test wall-clock limit in seconds (SIGALRM -> SynthTimeout), 0 = off
Records per test nodeid: outcome, when, duration, exception type/message, traceback entries (path, line,
func), longrepr tail, callspec params (repr), fixturenames, definition location.
"""
import json
import os
import signal
import sys
import time

import pytest

_OUT = os.environ.get("SYNTH_OUT")
_TRACE = tuple(os.path.realpath(p) + ("" if p.endswith(".py") else os.sep)
               for p in os.environ.get("SYNTH_TRACE", "").split(os.pathsep) if p)
_ALARM = int(os.environ.get("SYNTH_ALARM", "0") or 0)
_REC = {"tests": {}, "collect_errors": [], "cover": {}, "argv": sys.argv[1:]}
_CUR = None
_TOOL = 4


class SynthTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise SynthTimeout(f"test exceeded {_ALARM}s")


if _TRACE and hasattr(sys, "monitoring"):
    _mon = sys.monitoring
    try:
        _mon.use_tool_id(_TOOL, "synth")
    except ValueError:
        _TOOL = 3
        _mon.use_tool_id(_TOOL, "synth")
    _DIS = _mon.DISABLE
    _seen_files = {}

    def _line(code, line):
        if _CUR is not None:
            fn = code.co_filename
            ok = _seen_files.get(fn)
            if ok is None:
                ok = _seen_files[fn] = os.path.realpath(fn).startswith(_TRACE)
            if ok:
                _CUR.add((fn, line))
        return _DIS

    _mon.register_callback(_TOOL, _mon.events.LINE, _line)
    _mon.set_events(_TOOL, _mon.events.LINE)


def _safe_repr(v, n=400):
    try:
        r = repr(v)
    except Exception:
        r = "<unrepr>"
    return r if len(r) <= n else r[:n] + "...<truncated>"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _CUR
    if _TRACE:
        _CUR = set()
        sys.monitoring.restart_events()
    if _ALARM:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(_ALARM)
    t = time.time()
    try:
        yield
    finally:
        if _ALARM:
            signal.alarm(0)
        rec = _REC["tests"].setdefault(item.nodeid, {"outcome": "passed"})
        rec["dur"] = round(time.time() - t, 4)
        if _TRACE:
            cov = {}
            for fn, line in _CUR:
                cov.setdefault(os.path.realpath(fn), []).append(line)
            _REC["cover"][item.nodeid] = {k: sorted(v) for k, v in cov.items()}
            _CUR = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    rec = _REC["tests"].setdefault(item.nodeid, {"outcome": "passed"})
    if call.when == "setup":
        try:
            rec["loc"] = list(item.location)
            rec["fixtures"] = list(getattr(item, "fixturenames", []) or [])
            cs = getattr(item, "callspec", None)
            if cs is not None:
                rec["params"] = {k: _safe_repr(v) for k, v in cs.params.items()}
                rec["param_id"] = cs.id
            f = getattr(item, "function", None)
            if f is not None:
                rec["def_file"] = os.path.realpath(f.__code__.co_filename)
                rec["def_line"] = f.__code__.co_firstlineno
        except Exception as e:
            rec["meta_error"] = repr(e)[:200]
    if rep.failed or (rep.skipped and call.when in ("setup", "call")):
        if rec.get("outcome") == "failed" and rep.skipped:
            return
        rec["outcome"] = "failed" if rep.failed else "skipped"
        rec["when"] = call.when
        ei = call.excinfo
        if ei is not None:
            rec["exc_type"] = ei.typename
            try:
                rec["exc_msg"] = str(ei.value)[:3000]
            except Exception:
                rec["exc_msg"] = "<unprintable>"
            try:
                rec["tb"] = [[os.path.realpath(str(e.path)), e.lineno + 1, e.name] for e in ei.traceback][-30:]
            except Exception:
                rec["tb"] = []
        try:
            rec["longrepr"] = rep.longreprtext[-6000:]
        except Exception:
            pass


def pytest_collectreport(report):
    if report.failed:
        _REC["collect_errors"].append({"nodeid": report.nodeid, "longrepr": str(report.longrepr)[-3000:]})


def pytest_sessionfinish(session, exitstatus):
    _REC["exitstatus"] = int(exitstatus)
    if _OUT:
        with open(_OUT, "w", encoding="utf-8") as f:
            json.dump(_REC, f)
