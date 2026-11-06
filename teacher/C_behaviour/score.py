"""Teacher-trajectory behaviour scoring (task C).

Rates ANY agent trajectory against a numeric "fits-the-budget" profile derived from our own data
(see SUMMARY.md / spec.json).  Pure stdlib, CPU only.

Input formats (both produced by our recorders):
  * ATIF trace : harvest/<tag>/runs/<task>/<s>/traces/trace_<task>.json (steps with tool_calls/observation/metrics)
  * llm_calls  : harvest/<tag>/llm_calls/<task>__s<k>.jsonl (one record per LLM request: messages, response, usage)
Both are normalised into `Traj` = list of `Step` (one per LLM call).

Usage:
  python score.py --trace T.json [--gold boltons/cacheutils.py,...] [--spec spec.json]
  python score.py --llm-calls L.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CHARS_PER_TOKEN = 3.2      # pilot1: 3.58M generated chars / 1.115M completion tokens (used only when tokens are missing)
# per-call latency = a + b * completion_tokens  (fit on 4,060 recorded calls; R2 0.95)
LAT_MODAL_C8 = (0.442, 0.02208)        # H100, 8 concurrent runs, 45 tok/s per stream
LAT_KAGGLE_C1 = (1.065, 0.02968)       # Kaggle eval kernel, concurrency 1 (step-delta fit on 43 dev traces, R2 0.92)
UNCOUNTED = {"submit_patch", "get_status"}      # budget_gated(count_tool_call=False)
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------------------- normalised model
@dataclass
class Call:
    name: str
    args: dict
    obs: str = ""          # raw observation text ("" if unknown)

    def _json(self):
        try:
            j = json.loads(self.obs)
            if isinstance(j, dict):
                return j
        except Exception:
            pass
        return {}

    @property
    def malformed(self) -> bool:
        """Rejected before execution (bad skill name / missing argument): pure wasted turn."""
        return str(self._json().get("error_code") or "") in ("SKILL_NOT_FOUND", "INVALID_ARGUMENTS")

    @property
    def error_type(self) -> str:
        j = self._json()
        if j.get("status") == "error":
            return str(j.get("error_type") or "error")
        if j.get("error_code"):
            return str(j["error_code"])
        out = str(j.get("stdout") or "")
        if out.startswith("NOT WRITTEN"):
            return "EDIT_NOT_WRITTEN"
        if out.startswith("usage:"):
            return "SKILL_USAGE"
        return ""


@dataclass
class Step:
    """One LLM call and the tool calls it emitted."""
    t: float | None = None            # seconds (budget clock) at which the call finished / tool was dispatched
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    thought_chars: int = 0            # visible reasoning / message text
    calls: list = field(default_factory=list)
    latency_s: float | None = None


@dataclass
class Traj:
    steps: list
    resolved: bool | None = None
    task: str = ""


# ----------------------------------------------------------------------------------------- loaders
def _args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def from_atif(d: dict, **kw) -> Traj:
    steps = []
    for s in d.get("steps", []):
        if s.get("source") != "agent":
            continue
        m = s.get("metrics") or {}
        tcs = s.get("tool_calls") or []
        obs = (s.get("observation") or {}).get("content", "")
        calls = [Call(tc["function_name"], _args(tc.get("arguments")), obs if len(tcs) == 1 else "") for tc in tcs]
        t = (tcs[0].get("extra") or {}).get("elapsed_s") if tcs else None
        if t is None:
            t = (s.get("extra") or {}).get("elapsed_s")
        steps.append(Step(t=t, prompt_tokens=m.get("prompt_tokens"), completion_tokens=m.get("completion_tokens"),
                          thought_chars=len(s.get("message") or ""), calls=calls))
    return Traj(steps, task=kw.get("task", ""), resolved=kw.get("resolved"))


def from_llm_calls(records: list, **kw) -> Traj:
    """Recorder format: the last request's messages carry every observation; each record carries tokens/latency."""
    recs = [r for r in records if r.get("response")]
    obs_by_id = {}
    if recs:
        for m in recs[-1].get("messages") or []:
            if m.get("role") == "tool":
                c = m.get("content")
                obs_by_id[m.get("tool_call_id")] = c if isinstance(c, str) else json.dumps(c)
    steps = []
    for r in recs:
        resp = r["response"]
        calls = []
        for tc in resp.get("tool_calls") or []:
            fn = tc.get("function") or {}
            calls.append(Call(fn.get("name", ""), _args(fn.get("arguments")), obs_by_id.get(tc.get("id"), "") or ""))
        u = r.get("usage") or {}
        t = (r.get("t_rel") or 0.0) + (r.get("latency_s") or 0.0)
        thought = len(resp.get("reasoning_content") or "") + len(resp.get("content") or "")
        steps.append(Step(t=t, prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                          thought_chars=thought, calls=calls, latency_s=r.get("latency_s")))
    return Traj(steps, task=kw.get("task", ""), resolved=kw.get("resolved"))


# ----------------------------------------------------------------------------------------- classification
READ_VERB = re.compile(r"\b(cat|sed\s+-n|head|tail|nl|grep|rg|less|awk)\b")
TEST_CMD = re.compile(r"\b(pytest|unittest|tox|nose)\b")
EDIT_CMD = re.compile(r"(\bsed\s+-i\b|\bgit\s+apply\b|\bpatch\s+-p|\bperl\s+-pi\b)")
PY_CMD = re.compile(r"\bpython3?\b")


def _norm_path(p: str) -> str:
    p = (p or "").replace("\\", "/")
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.lstrip("/")


def _is_test_path(p: str) -> bool:
    parts = _norm_path(p).split("/")
    base = parts[-1]
    return ("tests" in parts[:-1] or "test" in parts[:-1] or base.startswith("test_")
            or base.endswith("_test.py") or base.startswith("repro") or base == "conftest.py")


def path_matches(p: str, gold: list) -> bool:
    p = _norm_path(p)
    return any(p == g or p.endswith("/" + g) or g.endswith("/" + p) for g in gold if g)


def call_kind(c: Call) -> str:
    n, a = c.name, c.args
    if n == "read_file":
        return "read"
    if n in ("edit_file", "write_file"):
        return "scratch" if _is_test_path(a.get("filepath", "")) else "edit"
    if n == "submit_patch":
        return "submit"
    if n == "get_status":
        return "status"
    if n == "run_skill_script":
        fp = str(a.get("file_path", ""))
        inner = a.get("args") if isinstance(a.get("args"), dict) else {}
        if fp.endswith("edit_lines.py"):
            return "scratch" if _is_test_path(str(inner.get("path", ""))) else "edit"
        if fp.endswith("check.py"):
            return "test"
        if fp.endswith("issue_example.py"):
            return "repro"
        return "skill"
    if n == "run_command":
        cmd = str(a.get("command", ""))
        if EDIT_CMD.search(cmd):
            return "edit"
        if TEST_CMD.search(cmd):
            return "test"
        if PY_CMD.search(cmd):
            return "repro"
        if READ_VERB.search(cmd):
            return "shell_read"
        return "shell"
    return "other"


def _read_range(a: dict):
    s, e = a.get("start_line"), a.get("end_line")
    return (int(s) if s else 1, int(e) if e else 10 ** 9)


# ----------------------------------------------------------------------------------------- metrics
def metrics(tr: Traj, gold: list | None = None) -> dict:
    """Behaviour metrics.  Indices ('calls_to_*') are 1-based counts of COUNTED tool calls (submit/status excluded)."""
    gold = [_norm_path(g) for g in (gold or []) if g]
    steps = tr.steps
    counted = 0
    idx = {"gold_read": None, "edit": None, "good_edit": None, "test": None, "submit": None}
    sec = dict(idx)
    first_t = next((s.t for s in steps if s.t is not None), None)
    seen = {}
    rep_exact = rep_read = failed = scratch = verify = n_submit = no_call_turns = malformed = 0
    err_types = {}
    covered = {}
    for st in steps:
        if not st.calls:
            no_call_turns += 1
        for c in st.calls:
            kind = call_kind(c)
            if c.name not in UNCOUNTED:
                counted += 1
            if c.malformed:
                malformed += 1
            if c.error_type:
                failed += 1
                err_types[c.error_type] = err_types.get(c.error_type, 0) + 1
            key = (c.name, json.dumps(c.args, sort_keys=True))
            if kind not in ("edit", "scratch", "submit", "status"):
                if key in seen:
                    rep_exact += 1
                seen[key] = 1
            if kind in ("edit", "scratch"):
                covered.clear()                       # file content changed: re-reading is legitimate
                seen = {k: v for k, v in seen.items() if k[0] not in ("read_file", "run_command")}
            if c.name == "read_file":
                fp = _norm_path(c.args.get("filepath", ""))
                lo, hi = _read_range(c.args)
                if any(a <= lo and hi <= b for a, b in covered.get(fp, [])):
                    rep_read += 1
                covered.setdefault(fp, []).append((lo, hi))
            if kind == "scratch":
                scratch += 1

            def mark(name):
                if idx[name] is None:
                    idx[name] = counted
                    sec[name] = None if (st.t is None or first_t is None) else st.t - first_t

            if gold and kind in ("read", "shell_read"):
                if c.name == "read_file":
                    hit = path_matches(c.args.get("filepath", ""), gold)
                else:
                    cmd = str(c.args.get("command", ""))
                    hit = any(g in cmd or g.split("/")[-1] in cmd for g in gold)
                if hit:
                    mark("gold_read")
            if kind == "edit":
                mark("edit")
                if not c.error_type:
                    mark("good_edit")
            if kind == "test":
                mark("test")
            if kind in ("test", "repro") and idx["good_edit"] is not None:
                verify += 1
            if kind == "submit":
                n_submit += 1
                mark("submit")
    ct = []
    for s_ in steps:
        if s_.completion_tokens is not None:
            ct.append(s_.completion_tokens)
        else:        # teacher output without usage: estimate from generated characters
            ct.append((s_.thought_chars + sum(len(json.dumps(c.args)) for c in s_.calls)) / CHARS_PER_TOKEN)
    pt = [s.prompt_tokens for s in steps if s.prompt_tokens is not None]
    tc_chars = sum(len(json.dumps(c.args)) for s in steps for c in s.calls)
    th_chars = sum(s.thought_chars for s in steps)
    ts = [s.t for s in steps if s.t is not None]
    n_calls = len(steps)
    tot_ct = float(sum(ct))
    # estimated seconds to submit if a student produced the same turns (generation only; tool time is ~3% of runs)
    sub_i = next((i for i, s_ in enumerate(steps) if any(c.name == "submit_patch" for c in s_.calls)), None)
    k = (sub_i + 1) if sub_i is not None else n_calls
    ct_k = float(sum(ct[:k]))
    return {
        "est_sec_modal_c8": LAT_MODAL_C8[0] * k + LAT_MODAL_C8[1] * ct_k,
        "est_sec_kaggle_c1": LAT_KAGGLE_C1[0] * k + LAT_KAGGLE_C1[1] * ct_k,
        "n_llm": len(steps),
        "n_counted": counted,
        "n_submit": n_submit,
        "calls_to_gold_read": idx["gold_read"],
        "calls_to_edit": idx["edit"],
        "calls_to_good_edit": idx["good_edit"],
        "malformed_calls": malformed,
        "calls_to_test": idx["test"],
        "calls_to_submit": idx["submit"],
        "sec_to_gold_read": sec["gold_read"], "sec_to_edit": sec["edit"], "sec_to_good_edit": sec["good_edit"], "sec_to_submit": sec["submit"],
        "total_sec": (max(ts) - first_t) if ts else None,
        "repeat_exact": rep_exact,
        "repeat_read": rep_read,
        "failed_calls": failed,
        "err_types": err_types,
        "no_call_turns": no_call_turns,
        "verify_after_edit": verify,
        "scratch_writes": scratch,
        "completion_tokens": tot_ct if ct else None,
        "prompt_tokens": sum(pt) if pt else None,
        "max_turn_tokens": max(ct) if ct else None,
        "mean_turn_tokens": (sum(ct) / len(ct)) if ct else None,
        "n_turns_over_1000": sum(1 for x in ct if x > 1000),
        "thought_share": th_chars / (th_chars + tc_chars) if (th_chars + tc_chars) else None,
        "submitted": n_submit > 0,
        "edited": idx["good_edit"] is not None,
    }


# ----------------------------------------------------------------------------------------- scoring
# limits: metric -> [full-credit ceiling, zero-credit floor]; credit falls linearly in between.
# These defaults are overwritten by spec.json (derived from the fast-resolved quartile, see SUMMARY.md).
DEFAULT_SPEC = {
    "version": 1,
    "pass": 0.8,
    "limits": {
        "n_counted": [18, 45],
        "calls_to_good_edit": [8, 25],
        "max_turn_tokens": [600, 2500],
        "mean_turn_tokens": [150, 600],
        "repeat_exact": [0, 4],
        "repeat_read": [0, 3],
        "failed_calls": [1, 8],
        "verify_after_edit": [3, 6],
        "malformed_calls": [0, 3],
        "est_sec_kaggle_c1": [90, 310],
        "scratch_writes": [1, 5],
    },
    "required": ["submitted", "edited"],
    "weights": {"n_counted": 2, "calls_to_good_edit": 3, "max_turn_tokens": 2, "mean_turn_tokens": 2, "repeat_exact": 1,
                "repeat_read": 1, "failed_calls": 1, "verify_after_edit": 1, "scratch_writes": 0.5,
                "malformed_calls": 0.5, "est_sec_kaggle_c1": 3},
}


def load_spec(path=None) -> dict:
    p = Path(path) if path else HERE / "spec.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else DEFAULT_SPEC


def _credit(v, lim) -> float:
    full, zero = lim
    if v is None:
        return 0.0
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return 1.0 - (v - full) / (zero - full)


def score(m: dict, spec: dict | None = None, tier: str = "strict") -> dict:
    """{'score': weighted credit in [0,1] (0 if a required property fails), 'fit': score>=spec['pass'], 'parts': ...}"""
    spec = spec or load_spec()
    parts = {}
    limits = spec["limits"] if tier == "strict" or "limits_relaxed" not in spec else spec["limits_relaxed"]
    for k, lim in limits.items():
        parts[k] = None if m.get(k) is None and k not in ("calls_to_good_edit",) else _credit(m.get(k), lim)
    num = den = 0.0
    for k, w in spec["weights"].items():
        if parts.get(k) is None:
            continue
        num += w * parts[k]
        den += w
    s = num / den if den else 0.0
    req_ok = all(bool(m.get(k)) for k in spec.get("required", []))
    if not req_ok:
        s = 0.0
    return {"score": round(s, 4), "fit": bool(req_ok and s >= spec.get("pass", 0.8)), "parts": parts,
            "required_ok": req_ok}


def score_traj(tr: Traj, gold: list | None = None, spec: dict | None = None, tier: str = "strict") -> dict:
    m = metrics(tr, gold)
    return {"metrics": m, **score(m, spec, tier)}


# ----------------------------------------------------------------------------------------- CLI
def _main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--llm-calls")
    ap.add_argument("--gold", default="")
    ap.add_argument("--spec")
    ap.add_argument("--tier", choices=["strict", "relaxed"], default="strict")
    a = ap.parse_args(argv)
    gold = [g for g in a.gold.split(",") if g]
    if a.trace:
        tr = from_atif(json.loads(Path(a.trace).read_text(encoding="utf-8")))
    elif a.llm_calls:
        lines = Path(a.llm_calls).read_text(encoding="utf-8").splitlines()
        tr = from_llm_calls([json.loads(x) for x in lines if x.strip()])
    else:
        ap.error("need --trace or --llm-calls")
    print(json.dumps(score_traj(tr, gold, load_spec(a.spec), a.tier), indent=1, default=str))


if __name__ == "__main__":
    _main()
