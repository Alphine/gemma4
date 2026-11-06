"""Synthetic data for testing analyze_ab.py: result rows in the lb-eval kernel format + ATIF-v1.7 traces.

The generator mimics the mechanisms of the real system: per-call completion tokens (lognormal + occasional long thought,
clipped at the arm's max_output_tokens), the Kaggle latency model (1.065 s + 0.02968 s/token), a hard time cap that censors
runs (agent_min == cap exactly, tokens truncated), a first source edit after k read/grep calls, infra-error retries.
"""
from __future__ import annotations

import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np

NAMES = {"control": "edit-early-v6-locate", "decide": "edit-early-v7-decide-nt", "lean": "edit-early-v7-lean-nt"}
MAXOUT = {"control": 4096, "decide": 4096, "lean": 2560}
BASE = dict(tok_factor=1.0, edit_delta=0.0, resolve_delta=0.0, extra_calls=0, long_p=0.04, hard_error=False)


def _arm_params(effects):
    p = {r: dict(BASE) for r in NAMES}
    for r, e in (effects or {}).items():
        p[r].update(e)
    return p


def _gen_run(rng, z, par, role, cap_s, pos_factor, k0):
    k = max(1, int(round(k0 + par["edit_delta"] + rng.normal(0, 1.5))))
    n_after = 4 + int(rng.poisson(3)) + par["extra_calls"]
    total = min(k + n_after, 118)
    scale = math.exp(0.35 * z) * par["tok_factor"] * pos_factor
    steps, t = [], 0.0
    cum_tok = 0
    did_edit = False
    for i in range(1, total + 1):
        tok = scale * rng.lognormal(math.log(150), 0.9)
        if rng.random() < par["long_p"]:
            tok = scale * rng.uniform(1200, 3500)
        tok = int(max(5, min(tok, MAXOUT[role] if role != "lean" else 2560)))
        if i == 1:
            name, args = "run_skill_script", {"file_path": "skills/swe-tools/scripts/locate.py", "args": {"issue": "x"}}
        elif i < k:
            if rng.random() < 0.5:
                name, args = "read_file", {"filepath": f"pkg/mod{i}.py"}
            else:
                name, args = "run_command", {"command": f"grep -n foo pkg/mod{i}.py"}
        elif i == k:
            name, args = "edit_file", {"filepath": "pkg/core.py", "old_string": "a", "new_string": "b"}
        elif i < total:
            name, args = ("run_command", {"command": "python -m pytest tests/test_x.py -q"}) if rng.random() < 0.6 else \
                ("read_file", {"filepath": "pkg/core.py"})
        else:
            name, args = "submit_patch", {}
        dt = 1.065 + 0.02968 * tok + 0.4
        if t + dt > cap_s:
            return steps, True, cap_s, did_edit
        t += dt
        cum_tok += tok
        did_edit |= name == "edit_file"
        steps.append(dict(name=name, args=args, tok=tok, t=t, ptok=4000 + 300 * i))
    return steps, False, t, did_edit


def to_atif(steps, start_ts, role, tools_err=None):
    out = [{"step_id": 1, "source": "system", "message": "You are an agent."},
           {"step_id": 2, "source": "user", "message": "Problem statement ... (MAX_TOKENS is a word in the prompt)"}]
    for i, s in enumerate(steps):
        ts = start_ts + s["t"]
        obs = json.dumps({"status": "ok", "stdout": "x"}) if s["name"] != "edit_file" else json.dumps({"status": "ok", "filepath": "pkg/core.py", "occurrences": 1})
        if tools_err and i in tools_err:
            obs = json.dumps({"error": tools_err[i]})
        ex = {"elapsed_s": s["t"], "timestamp": ts, "author": "swe_agent"}
        out.append({"step_id": 3 + i, "source": "agent", "model_name": "m",
                    "tool_calls": [{"tool_call_id": f"call_{i}", "function_name": s["name"], "arguments": s["args"],
                                    "extra": {"elapsed_s": s["t"], "timestamp": ts, "usage": {"completion_tokens": s["tok"]}}}],
                    "observation": {"content": obs},
                    "metrics": {"prompt_tokens": s["ptok"], "completion_tokens": s["tok"], "cached_tokens": 0},
                    "llm_call_count": 1, "extra": ex})
    tot = sum(s["tok"] for s in steps)
    return {"schema_version": "ATIF-v1.7", "agent": {"name": "adk-eval-core"}, "steps": out,
            "final_metrics": {"total_prompt_tokens": 1, "total_completion_tokens": tot, "total_steps": len(out)}}


def simulate(n_tasks=43, seed=0, effects=None, cap_s=330.0, order_mode="alternate", order_effect=0.0, arm_pos_field=False,
             p_infra=0.0, drop_pairs=0, resolve_base=0.35, launch="L1", roles=("control", "decide", "lean"), with_traces=True,
             task_prefix="task"):
    """-> rows (list of dicts, kernel format), traces {(arm_name, task): atif}. order_mode: alternate|fixed|rotate."""
    rng = np.random.default_rng(seed)
    par = _arm_params(effects)
    rows, traces = [], {}
    clock = 1.79e9
    cap_min = cap_s / 60.0
    for ti in range(n_tasks):
        task = f"{task_prefix}_{ti:03d}"
        z = rng.normal()
        k0 = 4 + 3 * abs(z) + rng.gamma(2, 2)
        order = list(roles)
        if order_mode == "alternate" and ti % 2 == 1:
            order = order[::-1]
        elif order_mode == "rotate":
            order = order[ti % len(order):] + order[:ti % len(order)]
        for pos, role in enumerate(order, 1):
            pf = 1.0 + order_effect * (pos - 1)
            attempts = 0
            while True:
                attempts += 1
                arm = NAMES[role]
                if rng.random() < p_infra and attempts <= 2 and (attempts == 1 or rng.random() < 0.3):
                    rows.append(dict(instance_id=task, arm=arm, set="solvable", resolved=False, patch_chars=0, agent_error=None,
                                     tool_calls=0, llm_calls=0, agent_min=None, error="APIConnectionError: Connection refused",
                                     error_type="APIConnectionError", infra_error=True, budget={"max_time_minutes": cap_min, "max_tool_calls": 60},
                                     run_mode="real", launch_id=launch, agent_patch=""))
                    if attempts >= 2:
                        break
                    continue
                steps, timed_out, t, did_edit = _gen_run(rng, z, par[role], role, cap_s, pf, k0)
                pr = 1 / (1 + math.exp(-(math.log(resolve_base / (1 - resolve_base)) - 0.6 * z))) + par[role]["resolve_delta"]
                resolved = bool(did_edit and rng.random() < min(max(pr, 0), 1))
                err = "Agent exceeded session timeout (%.1f min)" % cap_min if timed_out else None
                if par[role]["hard_error"] and rng.random() < 0.5:
                    err = "ValueError: unexpected thinking_token_budget"
                row = dict(instance_id=task, arm=arm, set="solvable", category="solvable", resolved=resolved,
                           patch_chars=(200 if did_edit else 0), agent_error=err, tool_calls=len([s for s in steps if s["name"] != "submit_patch"]),
                           llm_calls=len(steps), agent_min=round(t / 60, 3) if not timed_out else cap_min, error=err,
                           error_type=("Agent exceeded session timeout (%.1f min)" % cap_min if timed_out else None), infra_error=False,
                           budget={"max_time_minutes": cap_min, "max_tool_calls": 60, "timeout_seconds": 300},
                           run_mode="real", launch_id=launch, agent_patch="")
                if arm_pos_field:
                    row["arm_pos"] = pos
                rows.append(row)
                if with_traces:
                    traces[(arm, task)] = to_atif(steps, clock + 30.0, role)
                clock += t + 90.0
                break
    if drop_pairs:
        drop = {f"{task_prefix}_{i:03d}" for i in range(n_tasks - drop_pairs, n_tasks)}
        rows = [r for r in rows if not (r["instance_id"] in drop and r["arm"] == NAMES[roles[-1]])]
        traces = {k: v for k, v in traces.items() if not (k[1] in drop and k[0] == NAMES[roles[-1]])}
    return rows, traces


def write_dataset(d, rows, traces, tar=True):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    for arm, rs in by.items():
        (d / f"results_{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rs) + "\n", encoding="utf-8")
    if traces:
        if tar:
            with tarfile.open(d / "traces_session1.tar.gz", "w:gz") as t:
                for (arm, task), tr in traces.items():
                    b = json.dumps(tr).encode()
                    ti = tarfile.TarInfo(f"traces/{arm}/{task}.json")
                    ti.size = len(b)
                    t.addfile(ti, io.BytesIO(b))
        else:
            for (arm, task), tr in traces.items():
                (d / "traces" / arm).mkdir(parents=True, exist_ok=True)
                (d / "traces" / arm / f"{task}.json").write_text(json.dumps(tr))
