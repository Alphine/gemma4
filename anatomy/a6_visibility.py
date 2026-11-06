#!/usr/bin/env python3
"""When does the gold edit region first become visible to the model (read ranges / sed ranges) vs when does it edit?
usage: python a6_visibility.py runs.json out.json"""
import json, re, sys, collections
from common import *
R = load(sys.argv[1])
BS = chr(92)
def npath(p):
    p = str(p or "").replace(BS, "/").strip().strip("'\"`")
    for pre in ("/workspace/", "./"):
        if p.startswith(pre): p = p[len(pre):]
    return p.lstrip("/")
def ranges_of(c):
    t = tc0(c)
    if not t: return []
    a = t["args"]
    if t["name"] == "read_file":
        p = npath(a.get("filepath"))
        o = c.get("obs_range")
        if o and o[0] is not None and o[1] is not None: return [(p, int(o[0]), int(o[1]))]
        lo = int(a.get("start_line") or 1); hi = int(a.get("end_line") or lo + 149); return [(p, lo, hi)]
    if t["name"] == "run_command":
        out = []
        cmd = str(a.get("command", ""))
        for m in re.finditer(r"sed\s+-n\s+['\"]?(\d+)(?:,(\d+))?p['\"]?\s+([^\s|;&]+)", cmd):
            out.append((npath(m.group(3)), int(m.group(1)), int(m.group(2) or m.group(1))))
        for m in re.finditer(r"\b(?:cat|head|nl)\s+(?:-\w+\s*\d*\s+)?([^\s|;&<>]+\.py)\b", cmd):
            out.append((npath(m.group(1)), 1, 150))
        return out
    return []
rows = []
for r in R:
    g = r["gold"] or {}
    gl = g.get("gold_lines") or {}
    fe = first_edit_index(r["prim"])
    if not gl or fe is None: continue
    # lines that matter: gold lines in the file that gets edited first; ignore import-block lines (<=12) if others exist
    ep = None
    t = tc0(r["prim"][fe]); a = t["args"]
    if t["name"] == "edit_file": ep = npath(a.get("filepath"))
    elif t["name"] == "run_skill_script": ep = npath((a.get("args") or {}).get("path") if isinstance(a.get("args"), dict) else None)
    if ep not in gl: continue
    lines = [x for x in gl[ep] if x > 12] or gl[ep]
    seen = set()
    k_vis = None
    for k, c in enumerate(r["prim"][:fe]):
        for p, lo, hi in ranges_of(c):
            if p == ep: seen.update(range(lo, hi + 1))
        if k_vis is None and sum(1 for x in lines if x in seen) >= max(1, 0.5 * len(lines)):
            k_vis = k
    lat_between = sum(c["lat"] for c in r["prim"][(k_vis + 1 if k_vis is not None else 0):fe])
    rows.append(dict(arm=r["arm"], task=r["task"], fe=fe, k_visible=k_vis, extra_calls=(fe - 1 - k_vis) if k_vis is not None else None,
                     sec_between=lat_between if k_vis is not None else None, n_gold_lines=len(lines), resolved=r["row"]["resolved"]))
for x in rows: print(SHORT[x["arm"]], x["task"], "fe", x["fe"], "visible_at", x["k_visible"], "calls between", x["extra_calls"], "sec", None if x["sec_between"] is None else round(x["sec_between"]), "ngl", x["n_gold_lines"], x["resolved"])
v = [x for x in rows if x["k_visible"] is not None]
print("runs with gold info & edit:", len(rows), "gold region seen before edit:", len(v), " mean calls between visible and edit: %.1f, mean sec %.0f; median calls %.1f" %
      (mean(x["extra_calls"] for x in v), mean(x["sec_between"] for x in v), pct([x["extra_calls"] for x in v], .5)))
print("  k_visible mean %.1f ; fe mean %.1f" % (mean(x["k_visible"] for x in v), mean(x["fe"] for x in v)))
json.dump(rows, open(sys.argv[2], "w"), indent=1)
