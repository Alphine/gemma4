#!/usr/bin/env python3
"""Offline benchmark of the v8 'print the code of the best hit' feature (no LLM, stdlib).

Library used locally (synthetic tasks) and inside the Kaggle CPU kernel (dev tasks; embedded by kernels/gemma4-locate-body-bench/build.py).
For one task (start-state repo root, issue text, fix patch) and a list of variants it renders the real locate output and records:
  chars / lines printed, the printed segments, whether they cover the gold hunk lines (all / any), whether the printed code contains
  the gold function, whether rank 1 is the gold function, runtime.
"""
import json
import re
import time

VARIANTS = [   # name -> (BODY overrides)
    ("v6", None),                                  # body off (control)
    ("r1-40", {"n": 40, "k": 1}),
    ("r1-60", {"n": 60, "k": 1}),
    ("r1-80", {"n": 80, "k": 1}),
    ("r12-40", {"n": 40, "k": 2}),
    ("r12-60", {"n": 60, "k": 2}),
    ("r12-80", {"n": 80, "k": 2}),
    ("k3-40", {"n": 40, "k": 3, "short": 999, "n_k": 20, "tot": 70}),
    ("k3-60", {"n": 60, "k": 3, "short": 999, "n_k": 25, "tot": 90}),
    ("k4-40", {"n": 40, "k": 4, "short": 999, "n_k": 15, "tot": 70}),
    ("k4-60", {"n": 60, "k": 4, "short": 999, "n_k": 20, "tot": 90}),
]


def parse_patch(patch):
    """{path: {'new': bool, 'old_lines': [int]}}; old_lines = old-file line numbers touched (insertions: the line before)"""
    files, cur, old_no, rem_old, rem_new, old_path = {}, None, 0, 0, 0, None
    for line in (patch or "").splitlines():
        if rem_old <= 0 and rem_new <= 0:
            if line.startswith("--- "):
                old_path = line[4:].split("\t")[0].strip()
                continue
            if line.startswith("+++ "):
                newp = line[4:].split("\t")[0].strip()
                path = newp[2:] if newp.startswith("b/") else newp
                cur = {"path": path, "new": old_path == "/dev/null", "old_lines": []}
                files[path] = cur
                continue
            m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m and cur is not None:
                old_no = int(m.group(1))
                rem_old = int(m.group(2)) if m.group(2) is not None else 1
                rem_new = int(m.group(4)) if m.group(4) is not None else 1
            continue
        if line.startswith("\\"):
            continue
        if line.startswith("-"):
            cur["old_lines"].append(old_no)
            old_no += 1
            rem_old -= 1
        elif line.startswith("+"):
            cur["old_lines"].append(max(1, old_no - 1))
            rem_new -= 1
        else:
            old_no += 1
            rem_old -= 1
            rem_new -= 1
    return files


def gold_info(L, R, patch):
    """-> gold_lines {path: [old line numbers]}, gold_funcs [(path, qual, start, end)] from the fix patch on the start-state repo R"""
    files = parse_patch(patch)
    glines, gfuncs = {}, []
    for p, f in files.items():
        if L.is_test_path(p) or not p.endswith(".py") or f["new"] or p not in R.file_lines:
            continue
        glines[p] = sorted(set(f["old_lines"]))
        defs = [d for d in R.index if d["path"] == p]
        for ln in glines[p]:
            best = None
            for d in defs:
                if d["kind"] != "var" and d["line"] <= ln <= d["end"]:
                    if best is None or d["indent"] > best["indent"] or (d["indent"] == best["indent"] and d["line"] > best["line"]):
                        best = d
            if best is not None:
                key = (p, best.get("qual") or best["name"], best["line"], best["end"])
                if key not in gfuncs:
                    gfuncs.append(key)
    return glines, gfuncs


def covered(segs, path, ln):
    return any(p == path and a <= ln <= b for p, a, b in segs)


def eval_variant(L, R, top, final, terms, ph, glines, gfuncs, over):
    saved = dict(L.BODY)
    off = set(L.OFF)
    try:
        if over is None:
            L.OFF.add("body")
        else:
            L.OFF.discard("body")
            L.BODY.update(over)
        L.LAST_PLAN = []
        t0 = time.time()
        out = L.render(R, top, final, terms, ph)
        dt = time.time() - t0
        plan = L.LAST_PLAN if over is not None else []
    finally:
        L.BODY.clear()
        L.BODY.update(saved)
        L.OFF.clear()
        L.OFF.update(off)
    segs, ranks = [], []
    for rank, d, ss in plan:
        for a, b in ss:
            segs.append((d["path"], a, b))
        ranks.append((rank, d["path"], d.get("qual") or d["name"], d["kind"], L._span(d)))
    all_g = [(p, ln) for p, lns in glines.items() for ln in lns]
    cov = [covered(segs, p, ln) for p, ln in all_g]
    gold_funcs_shown = [any(rp == p and (q == rq) for _, rp, rq, _, _ in ranks) and any(
        pp == p and a <= e and b >= s for pp, a, b in segs) for p, q, s, e in gfuncs]
    gold_funcs_whole = [all(covered(segs, p, ln) for ln in range(s, e + 1)) for p, q, s, e in gfuncs]
    r1_gold = bool(ranks) and any(ranks[0][1] == p and ranks[0][2] == q for p, q, s, e in gfuncs)
    gold_in_file = any(p in {x[1] for x in ranks[:2]} for p in glines)
    return {"chars": len(out), "lines": sum(b - a + 1 for _, a, b in segs), "segs": segs,
            "ranks": [(r, p, q, k) for r, p, q, k, _ in ranks], "secs": dt,
            "hunk_all": bool(cov) and all(cov), "hunk_any": any(cov),
            "func_shown": any(gold_funcs_shown), "func_whole": any(gold_funcs_whole),
            "r1_gold_func": r1_gold, "gold_file_printed": gold_in_file,
            "r1_span": ranks[0][4][1] - ranks[0][4][0] + 1 if ranks else 0, "r1_kind": ranks[0][3] if ranks else None,
            "n_ranks": len(ranks)}


def run_task(L, R, task, patch, words=60, variants=VARIANTS):
    """task: {iid, statement}; R: repo indexed with L (start state)"""
    q = " ".join(task["statement"].split()[:words])
    t0 = time.time()
    final, terms, ph = L.rank_defs(R, q)
    top = L.dedupe_top(final, 8, R)
    rank_secs = time.time() - t0
    glines, gfuncs = gold_info(L, R, patch)
    row = {"iid": task["iid"], "gold_lines": glines, "gold_funcs": [list(x) for x in gfuncs],
           "gold_span": [e - s + 1 for p, q_, s, e in gfuncs],
           "gold_hunk_span": [max(v) - min(v) + 1 for v in glines.values() if v], "rank_secs": rank_secs,
           "n_top": len(top)}
    row["variants"] = {name: eval_variant(L, R, top, final, terms, ph, glines, gfuncs, over) for name, over in variants}
    return row
