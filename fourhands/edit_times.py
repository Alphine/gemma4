"""Per-run time of the LAST successful source edit (agent-elapsed s), from the A/B session-1 ATIF traces,
joined with the ckpt resolved flag.  A resolved run whose last edit lands at t keeps resolving if the whole trajectory is
slowed by s as long as t*s <= wall (the working-tree diff at the wall is the fallback patch).
Usage: python -I edit_times.py <traces_root> <ckpt_dir> <out.json>"""
import json, glob, os, sys, importlib.util
tm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ab", "trace_metrics.py")
spec = importlib.util.spec_from_file_location("trace_metrics", tm_path); tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
root, ck, out = sys.argv[1:4]
res = {}
for f in glob.glob(os.path.join(ck, "results_*.jsonl")):
    for l in open(f, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); res[(r["arm"], r["instance_id"])] = r
rows = []
for f in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
    arm, task = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-5]
    tr = json.load(open(f, encoding="utf-8"))
    known = set(); last_edit = None; first_edit = None; end = 0.0
    for s in tr.get("steps", []):
        if s.get("source") != "agent":
            continue
        ex = s.get("extra") or {}
        tcs = s.get("tool_calls") or []
        el = ex.get("elapsed_s") if ex.get("elapsed_s") is not None else ((tcs[0].get("extra") or {}).get("elapsed_s") if tcs else None)
        if el is None:
            continue
        end = max(end, float(el))
        obs = (s.get("observation") or {}).get("content", "")
        obs = obs if isinstance(obs, str) else json.dumps(obs, default=str)
        for tc in tcs:
            a = tm._args(tc.get("arguments"))
            k = tm.call_kind(tc.get("function_name") or "?", a, known)
            if k == "edit" and not tm.obs_failed(obs):
                last_edit = float(el); first_edit = first_edit if first_edit is not None else float(el)
    r = res.get((arm, task), {})
    rows.append(dict(arm=arm, task=task, resolved=bool(r.get("resolved")), agent_min=r.get("agent_min"),
                     last_edit_s=last_edit, first_edit_s=first_edit, end_s=end, has_patch=bool(r.get("patch_chars"))))
json.dump(rows, open(out, "w"), indent=1)
for r in rows: print(r)
