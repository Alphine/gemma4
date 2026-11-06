"""Per-run metrics from one ATIF-v1.7 trace (the format written by the swegemma harness / lb-eval kernel).

Pure stdlib so the same file can be inlined into a Kaggle CPU kernel (see kernel_extract/build_kernel.py).
Field names verified against training/tests/fixtures/trace_fastapi_14873_{robust,ladder1}.json:
  trace = {schema_version, agent, steps[], final_metrics{total_completion_tokens,...}}
  step  = {step_id, source: system|user|agent, message?, tool_calls[{function_name, arguments, extra{elapsed_s,
           timestamp, usage}}], observation{content}, metrics{prompt_tokens, completion_tokens, ...}?,
           llm_call_count, extra{elapsed_s, timestamp, event_type?}}
One agent step == one LLM call (llm_call_count == 1 in all fixtures). completion_tokens is the vLLM completion count
(thought tokens + tool-call tokens); thought tokens are NOT reported separately.
"""
from __future__ import annotations

import json
import re

SCHEMA = 1
LAT_KAGGLE_C1 = (1.065, 0.02968)        # s per call, s per completion token (research/teacher-gen/C-behaviour/score.py)
UNCOUNTED = {"submit_patch", "get_status"}
BS = chr(92)
READ_VERB = re.compile(r"\b(cat|sed\s+-n|head|tail|nl|grep|rg|less|awk)\b")
EDIT_CMD = re.compile(r"(\bsed\s+-i\b|\bgit\s+apply\b|\bpatch\s+-p|\bperl\s+-pi\b)")
REDIRECT_CMD = re.compile(r"(?:\bcat|\becho|\bprintf)\b[^|;&<]*?>{1,2}\s*([^\s<>|;&]+)|\btee\s+(?:-a\s+)?([^\s<>|;&]+)")
SCRATCH_BASE = ("repro", "reproduce", "debug", "scratch", "tmp", "temp", "test_", "check_", "verify", "demo", "try_",
                "poc", "issue")
MARK_ARGS_TRUNC = re.compile(r"Tool call arguments were truncated", re.I)
MARK_MAX_TOKENS = re.compile(r"\bMAX_TOKENS\b|finish_reason\W{0,4}length\b", re.I)
MARK_THINK_BUDGET = re.compile(r"thinking_token_budget", re.I)


def norm_path(p):
    p = str(p or "").replace(BS, "/").strip().strip("'\"`")
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.lstrip("/")


def is_scratch_path(p):
    parts = norm_path(p).split("/")
    base = parts[-1].lower()
    return ("tests" in parts[:-1] or "test" in parts[:-1] or base == "conftest.py" or base.endswith("_test.py")
            or base.startswith(SCRATCH_BASE) or not base)


def _args(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return {"_raw": a}
    return a if isinstance(a, dict) else {}


def _obs_json(obs):
    try:
        j = json.loads(obs)
        return j if isinstance(j, dict) else {}
    except Exception:
        return {}


def obs_failed(obs):
    """True when the tool observation says the call did not do its job."""
    j = _obs_json(obs)
    if j:
        if j.get("status") not in (None, "ok"):
            return True
        if "error" in j or j.get("error_code"):
            return True
        if str(j.get("stdout") or "").startswith("NOT WRITTEN"):
            return True
        ec = j.get("exit_code")
        if isinstance(ec, int) and ec != 0:
            return True
        return False
    return str(obs or "").lstrip().lower().startswith(("error", "traceback"))


def call_kind(name, args, known_read=()):
    """read | shell_read | edit | scratch | test | repro | locate | skill | submit | status | shell | other."""
    a = args
    if name == "read_file":
        return "read"
    if name == "edit_file":
        return "scratch" if is_scratch_path(a.get("filepath", "")) else "edit"
    if name == "write_file":
        fp = a.get("filepath", "")
        if is_scratch_path(fp):
            return "scratch"
        # a brand-new root-level file nobody read is scratch; a file in a package dir or one that was read is source
        return "edit" if ("/" in norm_path(fp) or norm_path(fp) in known_read) else "scratch"
    if name == "submit_patch":
        return "submit"
    if name == "get_status":
        return "status"
    if name == "run_skill_script":
        fp = str(a.get("file_path", ""))
        inner = a.get("args") if isinstance(a.get("args"), dict) else _args(a.get("args"))
        if fp.endswith("edit_lines.py"):
            return "scratch" if is_scratch_path(inner.get("path", "")) else "edit"
        if fp.endswith("locate.py"):
            return "locate"
        if fp.endswith("check.py"):
            return "test"
        return "skill"
    if name == "run_command":
        cmd = str(a.get("command", ""))
        if EDIT_CMD.search(cmd):
            return "edit"
        m = REDIRECT_CMD.search(cmd)
        tgt = (m.group(1) or m.group(2) or "") if m else ""
        if m and ".py" in tgt and not is_scratch_path(tgt):
            return "edit"
        if re.search(r"\b(pytest|unittest|tox)\b", cmd):
            return "test"
        if re.search(r"\bpython3?\b", cmd):
            return "repro"
        if READ_VERB.search(cmd):
            return "shell_read"
        return "shell"
    return "other"


def _scan_text(step):
    """Text of an agent step that may carry config-error markers, without echoing repo content (tool stdout)."""
    parts = [str(step.get("message") or ""), json.dumps(step.get("extra") or {}, default=str),
             json.dumps(step.get("metrics") or {}, default=str)]
    content = (step.get("observation") or {}).get("content", "")
    j = _obs_json(content) if isinstance(content, str) else {}
    if j and any(k in j for k in ("stdout", "stderr", "content", "diff")):
        parts += [str(j.get(k) or "") for k in ("error", "error_code", "message", "status")]
    else:
        parts.append(content if isinstance(content, str) else json.dumps(content, default=str))
    return "\n".join(parts)


def _step_tokens(s):
    m = s.get("metrics") or {}
    if m.get("completion_tokens") is not None:
        return int(m["completion_tokens"]), m
    for tc in s.get("tool_calls") or []:
        u = (tc.get("extra") or {}).get("usage") or {}
        if u.get("completion_tokens") is not None:
            return int(u["completion_tokens"]), u
    u = (s.get("extra") or {}).get("usage") or {}
    if u.get("completion_tokens") is not None:
        return int(u["completion_tokens"]), u
    return None, m


def _reasoning(src):
    for k in ("reasoning_tokens", "thought_tokens", "thoughts_token_count"):
        if src.get(k) is not None:
            return int(src[k])
    d = src.get("completion_tokens_details") or {}
    return int(d["reasoning_tokens"]) if d.get("reasoning_tokens") is not None else None


def parse_trace(tr):
    steps = tr.get("steps") or []
    agent = [s for s in steps if s.get("source") == "agent"]
    out = {"schema": SCHEMA, "n_steps": len(steps), "trace_ok": True}
    call_tokens, prompt_tokens, reasoning, n_llm = [], [], [], 0
    counted = 0
    first = {"edit": None, "edit_llm": None, "tok": None, "sec": None}
    reads_all = reads_after_locate = 0
    locate_seen = False
    n_edit_try = n_edit_fail = 0
    known_read = set()
    kinds = {}
    n_missing = 0
    starts, ends = [], []
    tok_running = 0
    marks = {"args_truncated": 0, "max_tokens": 0, "thinking_budget_error": 0}
    llm_idx = 0
    for s in steps:                                  # harness nudges to the model (user/system events after the prompt)
        if s.get("source") != "agent" and int(s.get("step_id") or 0) > 2:
            marks["args_truncated"] += bool(MARK_ARGS_TRUNC.search(str(s.get("message") or "")))
    for s in agent:
        k = int(s.get("llm_call_count") or 1)
        n_llm += k
        llm_idx += k
        tok, src = _step_tokens(s)
        pt = (s.get("metrics") or {}).get("prompt_tokens")
        if tok is None:
            n_missing += 1
        else:
            call_tokens.append(tok)
            tok_running += tok
        if pt is not None:
            prompt_tokens.append(int(pt))
        rt = _reasoning(src or {})
        if rt is not None:
            reasoning.append(rt)
        ex = s.get("extra") or {}
        tcs = s.get("tool_calls") or []
        el = ex.get("elapsed_s") if ex.get("elapsed_s") is not None else ((tcs[0].get("extra") or {}).get("elapsed_s") if tcs else None)
        ts = ex.get("timestamp") if ex.get("timestamp") is not None else ((tcs[0].get("extra") or {}).get("timestamp") if tcs else None)
        if ts is not None and el is not None:
            starts.append(float(ts) - float(el))
        if ts is not None:
            ends.append(float(ts))
        txt = _scan_text(s)
        marks["args_truncated"] += bool(MARK_ARGS_TRUNC.search(txt))
        marks["max_tokens"] += bool(MARK_MAX_TOKENS.search(txt))
        marks["thinking_budget_error"] += bool(MARK_THINK_BUDGET.search(txt))
        obs = (s.get("observation") or {}).get("content", "")
        obs = obs if isinstance(obs, str) else json.dumps(obs, default=str)
        for tc in tcs:
            name = tc.get("function_name") or tc.get("name") or "?"
            a = _args(tc.get("arguments"))
            kind = call_kind(name, a, known_read)
            kinds[kind] = kinds.get(kind, 0) + 1
            if name not in UNCOUNTED:
                counted += 1
            failed = obs_failed(obs)                 # one observation per step; steps carry one tool call in all fixtures
            if name == "read_file":
                known_read.add(norm_path(a.get("filepath", "")))
            if kind == "locate":
                locate_seen = True
            if kind in ("read", "shell_read") and first["edit"] is None:
                reads_all += 1
                if locate_seen:
                    reads_after_locate += 1
            if kind == "edit":
                n_edit_try += 1
                n_edit_fail += bool(failed)
                if not failed and first["edit"] is None:
                    first.update(edit=counted, edit_llm=llm_idx, tok=tok_running, sec=el)
    fm = tr.get("final_metrics") or {}
    step_sum = sum(call_tokens)
    fm_tok = fm.get("total_completion_tokens")
    elapsed = [float((s.get("extra") or {}).get("elapsed_s")) for s in agent if (s.get("extra") or {}).get("elapsed_s") is not None]
    out.update({
        "n_llm_calls": n_llm, "n_counted_calls": counted, "kinds": kinds,
        "completion_tokens": step_sum, "n_calls_missing_tokens": n_missing,
        "final_metrics_completion_tokens": fm_tok,
        "tokens_mismatch": bool(fm_tok is not None and int(fm_tok) != step_sum),
        "prompt_tokens_total": sum(prompt_tokens),
        "call_tokens": call_tokens, "reasoning_tokens": reasoning,
        "max_call_tokens": max(call_tokens) if call_tokens else 0,
        "first_edit_counted_idx": first["edit"], "first_edit_llm_idx": first["edit_llm"],
        "tokens_to_first_edit": first["tok"], "sec_to_first_edit": first["sec"],
        "reads_before_first_edit": (reads_after_locate if locate_seen else reads_all) if first["edit"] else None,
        "reads_before_first_edit_from_start": reads_all if first["edit"] else None,
        "locate_called": locate_seen,
        "n_edit_attempts": n_edit_try, "n_edit_failed": n_edit_fail, "has_edit": first["edit"] is not None,
        "submitted": bool(kinds.get("submit")),
        "start_ts": min(starts) if starts else None, "end_ts": max(ends) if ends else None,
        "elapsed_s_last": max(elapsed) if elapsed else 0.0,
        "est_llm_s": LAT_KAGGLE_C1[0] * n_llm + LAT_KAGGLE_C1[1] * step_sum,
        "marks": marks,
    })
    return out


def parse_trace_text(text):
    try:
        return parse_trace(json.loads(text))
    except Exception as e:                           # corrupt / truncated trace: the caller treats the trace as missing
        return {"schema": SCHEMA, "trace_ok": False, "trace_error": repr(e)[:200]}
