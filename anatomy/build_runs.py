#!/usr/bin/env python3
"""Join A/B session-1 call logs (HTTP-level) with ATIF traces into one per-run record; write runs.json.

usage: python build_runs.py <traces.tar.gz> <out runs.json>
Reads tar members in memory (nothing extracted). Treats trace text as data.
"""
import importlib.util
import json
import re
import sys
import tarfile

HERE = __file__.replace("\\", "/").rsplit("/", 1)[0]
spec = importlib.util.spec_from_file_location("tm", HERE + "/../ab/trace_metrics.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

ARMS = ["edit-early-v6-locate", "edit-early-v7-decide-nt", "edit-early-v7-lean-nt"]


def cat_command(cmd):
    """finer tool category for run_command"""
    c = cmd.strip()
    if re.search(r"\bpytest\b|python3?\s+-m\s+(pytest|unittest)", c):
        return "pytest"
    if re.search(r"\bpython3?\b", c):
        return "python"
    if re.search(r"\bgit\s+(log|status|diff|show|blame|stash)", c):
        return "git"
    if re.search(r"\b(git\s+grep|grep|rg|find)\b", c):
        return "grep_find"
    if re.search(r"\bsed\s+-i\b|\bperl\s+-pi\b", c):
        return "sed_i"
    if re.search(r"\b(sed\s+-n|cat|head|tail|nl|awk)\b", c):
        return "sed_cat"
    if re.search(r"^\s*(ls|pwd|which|wc|stat|test|echo)\b", c):
        return "ls"
    return "other_shell"


def norm_cmd(cmd):
    c = re.sub(r"\btimeout\s+\d+\s+", "", cmd)
    c = re.sub(r"\s*2>&1\s*", " ", c)
    c = re.sub(r"\|\s*(tail|head)\s+-?\d+", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c


def build(tar_path):
    t = tarfile.open(tar_path)
    runs = {}
    for m in t.getmembers():
        if not m.isfile():
            continue
        parts = m.name.split("/")
        if len(parts) != 3:
            continue
        kind, arm, fn = parts
        task = fn[:-5]
        runs.setdefault((arm, task), {})[kind] = json.load(t.extractfile(m))
    out = []
    for (arm, task), d in sorted(runs.items()):
        tr, cl = d.get("traces"), d.get("calllogs")
        if tr is None or cl is None:
            continue
        steps = tr["steps"]
        t0 = steps[0]["extra"]["timestamp"]
        ents = cl["entries"]
        ag = [s for s in steps if s["source"] == "agent"]
        calls = []
        ai = 0
        # nudges (system steps after prompt) keyed by the following agent step
        nudge_before = {}
        pend = None
        for s in steps:
            if s["source"] == "system" and s.get("extra", {}).get("event_type") == "continuation_nudge":
                pend = s["message"][:60]
            elif s["source"] == "agent":
                if pend:
                    nudge_before[s["step_id"]] = pend
                    pend = None
        known_read = set()
        obs_sh, th_sh = set(), set()
        for i, e in enumerate(ents):
            rec = {"i": i, "t_send": e["t_send"] - t0, "t_end": e["t_end"] - t0, "status": e.get("status"),
                   "finish": e.get("finish"), "exc": e.get("exc"), "ct": e.get("ct", 0), "pt": e.get("pt", 0),
                   "rsn_tok": e.get("rsn_tok", 0), "tc_chars": e.get("tc_arg_chars", 0), "content_chars": e.get("content_chars", 0),
                   "rsn_chars": e.get("rsn_chars", 0), "max_comp": (e.get("sent") or {}).get("max_completion_tokens"),
                   "think_budget": (e.get("sent") or {}).get("thinking_token_budget")}
            rec["lat"] = rec["t_end"] - rec["t_send"]
            rec["compact"] = not e.get("has_tools")
            sm = None
            if e.get("has_tools") and e.get("exc") is None:
                for j in range(ai, min(ai + 4, len(ag))):     # tolerant in-order match on completion/prompt tokens
                    mt = ag[j].get("metrics") or {}
                    if mt.get("completion_tokens") == e.get("ct") and mt.get("prompt_tokens") == e.get("pt"):
                        sm = j
                        break
            if sm is not None:
                s = ag[sm]
                ai = sm + 1
                rec["step_id"] = s["step_id"]
                rec["step_ct"] = (s.get("metrics") or {}).get("completion_tokens")
                rec["step_pt"] = (s.get("metrics") or {}).get("prompt_tokens")
                rec["nudged_before"] = nudge_before.get(s["step_id"])
                rec["message"] = s.get("message") or ""
                w = re.findall(r"\S+", rec["message"])
                sh = [tuple(w[q:q + 6]) for q in range(max(0, len(w) - 5))]
                rec["n_shingles"] = len(sh)
                rec["copy_obs"] = sum(1 for g in sh if g in obs_sh)
                rec["copy_thought"] = sum(1 for g in sh if g in th_sh and g not in obs_sh)
                th_sh.update(sh)
                rec["tool_calls"] = []
                ts = (s.get("extra") or {}).get("timestamp")
                obs = s.get("observation") or {}
                ots = (obs.get("extra") or {}).get("timestamp")
                rec["tool_s"] = (ots - ts) if (ots is not None and ts is not None) else None
                oc = obs.get("content", "")
                oc = oc if isinstance(oc, str) else json.dumps(oc)
                try:
                    _oj = json.loads(oc)
                    _txt = " ".join(str(_oj.get(k) or "") for k in ("stdout", "content", "diff", "stderr", "error", "error_message"))
                except Exception:
                    _txt = oc
                _w = re.findall(r"\S+", _txt)
                obs_sh.update(tuple(_w[q:q + 6]) for q in range(max(0, len(_w) - 5)))
                rec["obs_len"] = len(oc)
                rec["obs_head"] = oc[:300]
                rec["failed"] = tm.obs_failed(oc) if oc else None
                try:
                    oj = json.loads(oc)
                except Exception:
                    oj = {}
                st_ = oj.get("status")
                err = oj.get("error") or oj.get("error_message") or ""
                so = str(oj.get("stdout") or "")
                soft = bool(re.match(r"\s*(usage:|NOT WRITTEN|REFUSED|ERROR|FAILED)", so))
                rec["fail2"] = bool(err or oj.get("error_code") or st_ == "error" or soft)
                rec["err_kind"] = ("invalid_args" if (oj.get("error_code") == "INVALID_ARGUMENTS" or "mandatory input parameters" in str(err))
                                   else "cmd_error" if st_ == "error" else "skill_soft" if soft else "tool_error" if err else None)
                rec["obs_err"] = (str(err) or so[:200] if (err or soft) else "")[:240]
                rec["obs_stdout_len"] = len(so)
                if "start_line" in oj:
                    rec["obs_range"] = [oj.get("start_line"), oj.get("end_line"), oj.get("total_lines"), oj.get("is_truncated")]
                rec["obs_ts"] = (ots - t0) if ots is not None else None
                for tc in s.get("tool_calls") or []:
                    name = tc.get("function_name")
                    a = tm._args(tc.get("arguments"))
                    k = tm.call_kind(name, a, known_read)
                    if name == "read_file":
                        known_read.add(tm.norm_path(a.get("filepath", "")))
                    sub = k
                    if name == "run_command":
                        sub = cat_command(str(a.get("command", "")))
                    elif name == "run_skill_script":
                        sub = "skill:" + str(a.get("file_path", "")).split("/")[-1]
                    elif name in ("edit_file", "write_file", "read_file", "submit_patch"):
                        sub = name
                    if name == "run_skill_script" and str(a.get("file_path", "")).endswith("edit_lines.py") and k == "scratch":
                        inner_ = a.get("args") if isinstance(a.get("args"), dict) else {}
                        if inner_.get("file_path") and not tm.is_scratch_path(inner_.get("file_path")):
                            k = "edit"                     # path given under the key file_path instead of path (script tolerated it)
                    if name == "run_skill_script" and not str(a.get("file_path", "")).strip():
                        blob = json.dumps(a)
                        if "expect" in blob or "new_text" in blob:
                            k, sub = "edit", "skill:edit_lines(malformed)"
                        elif "issue" in blob:
                            k, sub = "locate", "skill:locate(malformed)"
                        else:
                            sub = "skill:(malformed)"
                    rec["tool_calls"].append({"name": name, "kind": k, "sub": sub, "args": a})
                rec["call_ts"] = ts - t0 if ts is not None else None
            calls.append(rec)
        out.append({"arm": arm, "task": task, "calls": calls, "n_steps": len(steps),
                    "final_metrics": tr.get("final_metrics"), "n_nudges": len(nudge_before)})
    return out


if __name__ == "__main__":
    r = build(sys.argv[1])
    json.dump(r, open(sys.argv[2], "w"))
    print(len(r), "runs;", sum(len(x["calls"]) for x in r), "calls")
