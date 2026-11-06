#!/usr/bin/env python3
"""Counterfactual estimate from recorded traces (synthetic tasks, runs that called locate.py): what the v8 body would have saved.

For every run: replay the agent's own locate query against the exact start state with locate_body.py (default config), then compare
the printed code with (a) the gold hunk lines, (b) the code ranges the agent actually read after locate.  A read whose lines are
>= 70 % inside the printed segments is counted as saved (one LLM turn: 0.44 s + 0.0221 s x completion tokens of that step).

  python trace_estimate.py <trace_root> --out trace_estimate.json
"""
import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "locate-improve"))
import bench  # noqa: E402
import body_bench as BB  # noqa: E402
from run_synth import S, W, split_patch, apply_hunks  # noqa: E402

LAT0, LAT1 = 0.44, 0.0221
COMPACT_AT = 14336


def reads_of(step):
    """[(path, a, b)] code ranges a step reads (read_file, sed -n 'A,Bp', cat)"""
    out = []
    for tc in step.get("tool_calls") or []:
        fn, a = tc["function_name"], tc.get("arguments") or {}
        if fn == "read_file":
            p = a.get("filepath") or a.get("path")
            if p:
                out.append((p, int(a.get("start_line") or 1), int(a.get("end_line") or 10 ** 6)))
        elif fn == "run_command":
            c = str(a.get("command", ""))
            for m in re.finditer(r"sed -n ['\"]?(\d+),(\d+)p['\"]?\s+(\S+)", c):
                out.append((m.group(3).strip("'\""), int(m.group(1)), int(m.group(2))))
            m = re.match(r"^\s*(?:timeout \d+ )?cat\s+(\S+\.py)\s*$", c)
            if m:
                out.append((m.group(1), 1, 10 ** 6))
    return out


def is_edit(step):
    for tc in step.get("tool_calls") or []:
        fn, a = tc["function_name"], tc.get("arguments") or {}
        if fn == "edit_file":
            p = a.get("filepath") or a.get("path") or ""
            if not re.search(r"(^|/)(tests?/|test_)", p):
                return True
        if fn == "run_skill_script" and "edit_lines" in json.dumps(a):
            return True
        if fn == "run_command" and re.search(r"sed -i|perl -pi", str(a.get("command", ""))):
            return True
    return False


def frac_inside(rd, segs):
    p, a, b = rd
    segp = [(x, y) for pp, x, y in segs if pp == p]
    if not segp:
        return 0.0
    b = min(b, max(y for _, y in segp) + 400)
    n = max(1, b - a + 1)
    got = sum(1 for ln in range(a, b + 1) if any(x <= ln <= y for x, y in segp))
    return got / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--locate", default=str(HERE / "locate_body.py"))
    a = ap.parse_args()
    ds = Path(S) / "synth"
    raw = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ds / "tasks.jsonl", encoding="utf-8")}
    metas = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ds / "meta.jsonl", encoding="utf-8")}
    files = sorted(glob.glob(a.root + "/ab1/*base*.json") + glob.glob(a.root + "/pilot1/*.json"))
    by_task = {}
    for f in files:
        by_task.setdefault(Path(f).name.split("__")[0], []).append(f)
    allt = [dict(iid=i, statement=raw[i]["problem_statement"]) for i in by_task if i in raw]
    # materialise start states (same shared dirs as run_synth)
    ids = sorted(by_task)
    splits = json.loads((ds / "splits.json").read_text(encoding="utf-8"))
    tasks_all = {}
    for sp in ("heldout", "train"):
        for t in bench.synth_tasks(ds, Path(W), sp, None):
            tasks_all[t["iid"]] = t
    L = bench.load_locate(a.locate, "")
    rows = []
    for iid in ids:
        if iid not in tasks_all:
            continue
        t = tasks_all[iid]
        root = Path(t["root"])
        pp = ds / "snapshots" / "patches" / ("%s.patch" % iid)
        saved = {}
        try:
            if pp.exists() and pp.stat().st_size:
                for p, hunks in split_patch(pp.read_text(encoding="utf-8")).items():
                    if L.is_test_path(p) or not p.endswith(".py") or not (root / p).exists():
                        continue
                    saved[root / p] = (root / p).read_bytes()
                    (root / p).write_bytes(apply_hunks(saved[root / p].decode("utf-8", "replace"), hunks).encode("utf-8"))
            R = L.load_repo(str(root))
            glines, gfuncs = BB.gold_info(L, R, raw[iid]["patch"])
            if not glines:
                continue
            for f in by_task[iid]:
                d = json.load(open(f, encoding="utf-8"))
                steps = [s for s in d["steps"] if s.get("source") == "agent" and s.get("metrics")]
                li = next((i for i, s in enumerate(steps) for tc in (s.get("tool_calls") or [])
                           if tc["function_name"] == "run_skill_script" and "locate" in json.dumps(tc["arguments"])), None)
                if li is None:
                    continue
                tc = next(tc for tc in steps[li]["tool_calls"] if "locate" in json.dumps(tc["arguments"]))
                q = tc["arguments"].get("args")
                q = q.get("issue") if isinstance(q, dict) else (q[0] if isinstance(q, list) and q else str(q or ""))
                q = str(q or "")
                if not q.strip():
                    continue
                final, terms, ph = L.rank_defs(R, q)
                top = L.dedupe_top(final, 8, R)
                v = BB.eval_variant(L, R, top, final, terms, ph, glines, gfuncs, dict(L.BODY))
                out_chars = v["chars"]
                v6 = BB.eval_variant(L, R, top, final, terms, ph, glines, gfuncs, None)
                body_chars = out_chars - v6["chars"]
                segs = v["segs"]
                after = steps[li + 1:]
                ei = next((i for i, s in enumerate(after) if is_edit(s)), None)
                horizon = after[:ei] if ei is not None else after
                rd_steps = []
                for i, s in enumerate(horizon):
                    rds = reads_of(s)
                    if rds:
                        rd_steps.append((i, rds, s["metrics"]["completion_tokens"]))
                saved_steps = [(i, ct) for i, rds, ct in rd_steps if rds and all(frac_inside(r, segs) >= 0.7 for r in rds)]
                g_idx = next((i for i, rds, ct in rd_steps if any(
                    r[0] in glines and all(r[1] <= ln <= r[2] for ln in glines[r[0]]) for r in rds)), None)
                first_rd = rd_steps[0] if rd_steps else None
                pt = [s["metrics"]["prompt_tokens"] for s in steps]
                rows.append({"run": Path(f).name, "iid": iid, "style": metas[iid]["style"], "n_steps": len(steps), "locate_step": li,
                             "edit_after": ei, "reads_before_edit": len(rd_steps), "saved_reads": len(saved_steps),
                             "saved_ct": [ct for _, ct in saved_steps], "gold_read_after": g_idx,
                             "body_hunk_all": v["hunk_all"], "body_hunk_any": v["hunk_any"], "body_func": v["func_shown"],
                             "r1_gold": v["r1_gold_func"], "body_chars": body_chars,
                             "first_read_in_body": bool(first_rd and all(frac_inside(r, segs) >= 0.7 for r in first_rd[1])),
                             "first_read_hits_gold": bool(first_rd and any(r[0] in glines and any(r[1] <= ln <= r[2] for ln in glines[r[0]]) for r in first_rd[1])),
                             "max_prompt": max(pt) if pt else 0, "prompt_tokens": pt, "li_prompt": pt[li] if pt else 0,
                             "resolved": None, "total_steps_tokens": sum(s["metrics"]["completion_tokens"] for s in steps)})
        finally:
            for fpath, b in saved.items():
                fpath.write_bytes(b)
    Path(a.out).write_text(json.dumps(rows), encoding="utf-8")
    print(len(rows), "runs")


if __name__ == "__main__":
    main()
