#!/usr/bin/env python3
"""Section 4/5: tool-call format failures, compaction & context growth, post-edit anatomy, per-task pairs, counterfactual savings.
usage: python a4_sinks.py runs.json out.json"""
import json
import re
import sys
import collections
from common import *

R = load(sys.argv[1])
B = 0.0287          # s per completion token (fit in a1_wall.py)
out = {}

# ---- A. tool-call format failures -------------------------------------------------
print("== A. failed tool calls by kind (primary calls) ==")
fk = collections.Counter(); fs = collections.Counter()
per_run_fail = []
for r in R:
    n = 0; s = 0.0; ntot = len(r["prim"])
    for c in r["prim"]:
        t = tc0(c)
        if not t:
            continue
        ek = c.get("err_kind")
        if ek:
            key = (t["name"] if t["name"] != "run_skill_script" else t["sub"], ek)
            fk[key] += 1; fs[key] += c["lat"]
            if ek in ("invalid_args", "skill_soft"):
                n += 1; s += c["lat"]
    per_run_fail.append(dict(arm=r["arm"], task=r["task"], n_format_fail=n, sec=s, n_calls=ntot, wall=r["wall"],
                             resolved=r["row"]["resolved"], err=r["row"]["err_class"]))
for k, v in sorted(fk.items(), key=lambda kv: -fs[kv[0]]):
    print("  %-40s n=%3d  llm_s=%6.0f" % (k, v, fs[k]))
tot_fmt = sum(x["sec"] for x in per_run_fail)
print("format failures (invalid_args + skill usage): %d calls, %.0f llm-s = %.1f%% of all wall; per run mean %.1f s" %
      (sum(x["n_format_fail"] for x in per_run_fail), tot_fmt, 100 * tot_fmt / sum(x["wall"] for x in per_run_fail), tot_fmt / len(R)))
for a in ARMS:
    xs = [x for x in per_run_fail if x["arm"] == a]
    print("   ", SHORT[a], "n_fail", sum(x["n_format_fail"] for x in xs), "sec", round(sum(x["sec"] for x in xs)), "runs w/ >=1:", sum(1 for x in xs if x["n_format_fail"] > 0), "/", len(xs))
print("   runs with >=4 format failures:", [(SHORT[x["arm"]], x["task"], x["n_format_fail"], round(x["sec"]), x["resolved"]) for x in per_run_fail if x["n_format_fail"] >= 4])
out["format_fail_runs"] = per_run_fail
# what do the errors look like (tool arg shape): edit_file invalid args -> args keys
shapes = collections.Counter()
for r in R:
    for c in r["prim"]:
        t = tc0(c)
        if t and t["name"] == "edit_file" and c.get("err_kind") == "invalid_args":
            shapes[tuple(sorted(t["args"].keys()))[:6]] += 1
print("   edit_file invalid-arg call shapes:", shapes.most_common(4))
# success rate of edit_file by whether args have the 3 keys
ef = [(tc0(c), c) for r in R for c in r["prim"] if tc0(c) and tc0(c)["name"] == "edit_file"]
print("   edit_file calls:", len(ef), "ok:", sum(1 for t, c in ef if not c.get("err_kind")), "invalid_args:", sum(1 for t, c in ef if c.get("err_kind") == "invalid_args"))
# retry chains: after an invalid_args edit_file, what is the next call?
chain = collections.Counter()
for r in R:
    p = r["prim"]
    for i, c in enumerate(p[:-1]):
        t = tc0(c)
        if t and c.get("err_kind") == "invalid_args":
            n = tc0(p[i + 1])
            chain[(t["name"], n["name"] if n else None, "fail" if p[i + 1].get("err_kind") else "ok")] += 1
print("   after invalid_args call ->", chain.most_common(8))

# ---- B. compaction & context growth ------------------------------------------------
print("== B. compaction ==")
rows = []
for r in R:
    calls = r["calls"]
    prev_pt = None
    for j, c in enumerate(calls):
        if c["compact"] and not c.get("exc"):
            # last primary call before it
            before = [x for x in calls[:j] if not x["compact"] and not x.get("exc")]
            rows.append(dict(arm=r["arm"], task=r["task"], t=c["t_send"], lat=c["lat"], ct=c["ct"], pt=c["pt"],
                             pt_before=before[-1]["pt"] if before else None, n_prim_before=len(before),
                             nxt_pt=next((x["pt"] for x in calls[j + 1:] if not x["compact"]), None)))
print("  n=%d  pt_before mean=%.0f (min %d max %d)  at primary call #%.1f  lat %.1f s; pt after %.0f" %
      (len(rows), mean(x["pt_before"] for x in rows), min(x["pt_before"] for x in rows), max(x["pt_before"] for x in rows),
       mean(x["n_prim_before"] for x in rows), mean(x["lat"] for x in rows), mean(x["nxt_pt"] for x in rows if x["nxt_pt"])))
nrun_c = collections.Counter(r["task"] + r["arm"] for r in R for c in r["compact"])
print("  runs with >=1 compaction: %d / %d ; with >=2: %d" % (len(nrun_c), len(R), sum(1 for v in nrun_c.values() if v >= 2)))
first_pt = [r["prim"][0]["pt"] for r in R]
print("  first-call prompt tokens mean %.0f (min %d max %d)" % (mean(first_pt), min(first_pt), max(first_pt)))
# growth per step excluding compaction resets
gr = []
for r in R:
    p = r["prim"]
    for a, b in zip(p, p[1:]):
        d = b["pt"] - a["pt"]
        if d >= 0:
            gr.append((d, tc0(a)["sub"] if tc0(a) else None))
print("  mean prompt growth per step %.0f tokens; by tool of the previous call:" % mean(d for d, _ in gr))
byt = collections.defaultdict(list)
for d, s in gr:
    byt[s].append(d)
for s, v in sorted(byt.items(), key=lambda kv: -sum(kv[1]))[:8]:
    print("      %-18s n=%4d mean=%5.0f sum=%7d" % (s, len(v), mean(v), sum(v)))
out["compactions"] = rows
# re-reads after compaction: range read after compaction that was read before
BS = chr(92)
def reads_of(c):
    t = tc0(c)
    if not t:
        return []
    a = t["args"]
    if t["name"] == "read_file":
        p = str(a.get("filepath", "")).replace(BS, "/")
        lo = int(a.get("start_line") or 1); hi = int(a.get("end_line") or lo + 149)
        return [(p, lo, hi)]
    if t["name"] == "run_command":
        o = []
        for m in re.finditer(r"sed\s+-n\s+['\"]?(\d+)(?:,(\d+))?p['\"]?\s+([^\s|;&]+)", str(a.get("command", ""))):
            o.append((m.group(3), int(m.group(1)), int(m.group(2) or m.group(1))))
        return o
    return []
tot_after = rr = 0; rr_sec = 0.0; tot_before = rb = 0
def edited_path(c):
    t = tc0(c)
    if not t or t["kind"] not in ("edit", "scratch"):
        return None
    a = t["args"]
    if t["name"] == "run_skill_script":
        inner = a.get("args") if isinstance(a.get("args"), dict) else {}
        return str(inner.get("path", "")).replace(BS, "/")
    if t["name"] in ("edit_file", "write_file"):
        return str(a.get("filepath", "")).replace(BS, "/")
    m = re.search(r"([\w/.\-]+\.py)", str(a.get("command", "")))
    return m.group(1) if m else None
for r in R:
    seen = collections.defaultdict(list)
    tcs = [c["t_send"] for c in r["compact"]]
    for c in r["prim"]:
        after = any(c["t_send"] > x for x in tcs)
        for p, lo, hi in reads_of(c):
            cov = sum(1 for x in range(lo, hi + 1) if any(a <= x <= b for a, b in seen[p])) / max(1, hi - lo + 1)
            if after:
                tot_after += 1
                if cov >= 0.8:
                    rr += 1; rr_sec += c["lat"]
            else:
                tot_before += 1
                if cov >= 0.8:
                    rb += 1
            seen[p].append((lo, hi))
        ep = edited_path(c)
        if ep:                       # content changed: later reads of that path are not redundant
            seen[ep] = []
print("  reads after a compaction: %d, >=80%% re-reads (path not edited since): %d (%.0f%%, %.0f llm-s); before any compaction: %d reads, %d re-reads (%.0f%%)" %
      (tot_after, rr, 100 * rr / max(1, tot_after), rr_sec, tot_before, rb, 100 * rb / max(1, tot_before)))

# ---- C. post-edit anatomy -----------------------------------------------------------
print("== C. time structure around the first edit ==")
cl_rows = []
for r in R:
    p = r["prim"]
    fe = first_edit_index(p)
    sub_post = collections.Counter()
    if fe is not None:
        for c in p[fe + 1:]:
            t = tc0(c)
            sub_post[t["sub"] if t else "nocall"] += 1
        t_pre = p[fe]["t_send"]; t_end = p[-1]["t_end"]
        # successful first edit
        fs = first_edit_index(p, successful=True)
        cl_rows.append(dict(arm=r["arm"], task=r["task"], resolved=r["row"]["resolved"], err=r["row"]["err_class"], wall=r["wall"],
                            t_first_edit=t_pre, t_first_ok_edit=p[fs]["t_end"] if fs is not None else None, n_pre=fe, n_post=len(p) - fe - 1,
                            post=dict(sub_post)))
for x in cl_rows:
    print("  %-6s %-14s res=%-5s %-8s first_edit@%5.0fs ok_edit@%s n_pre=%2d n_post=%2d  post=%s" %
          (SHORT[x["arm"]], x["task"], x["resolved"], x["err"], x["t_first_edit"], ("%5.0fs" % x["t_first_ok_edit"]) if x["t_first_ok_edit"] is not None else "  none",
           x["n_pre"], x["n_post"], dict(sorted(x["post"].items(), key=lambda kv: -kv[1])[:5])))
res = [x for x in cl_rows if x["resolved"]]
unr = [x for x in cl_rows if not x["resolved"]]
print("  resolved with edit: n=%d mean first_edit@ %.0fs, wall %.0f; unresolved with edit: n=%d mean first_edit@ %.0fs, wall %.0f" %
      (len(res), mean(x["t_first_edit"] for x in res), mean(x["wall"] for x in res), len(unr), mean(x["t_first_edit"] for x in unr), mean(x["wall"] for x in unr)))
print("  n_pre calls resolved mean %.1f vs unresolved %.1f" % (mean(x["n_pre"] for x in res), mean(x["n_pre"] for x in unr)))
out["edit_rows"] = cl_rows

# ---- D. counterfactual mechanical savings (token accounting, trajectory held fixed) -------
print("== D. mechanical token savings per run (trajectory held fixed; upper bounds) ==")
cf = {}
for cap in (1024, 512, 256, 128):
    s = 0.0
    for r in R:
        for c in r["prim"]:
            s += B * max(0, c["rsn_tok"] - cap)
    cf["rsn_cap_%d" % cap] = s / len(R)
    print("  cap thinking at %4d tokens: %5.1f s/run (%.1f%% of mean wall)" % (cap, s / len(R), 100 * s / len(R) / mean(r["wall"] for r in R)))
# only non-lean arms (lean already capped)
nl = [r for r in R if r["arm"] != "edit-early-v7-lean-nt"]
for cap in (1024, 512, 256):
    s = sum(B * max(0, c["rsn_tok"] - cap) for r in nl for c in r["prim"])
    print("  [non-lean arms] cap %4d: %5.1f s/run" % (cap, s / len(nl)))
# lean arm already at 1024: further cap
ln = [r for r in R if r["arm"] == "edit-early-v7-lean-nt"]
for cap in (512, 256, 128):
    s = sum(B * max(0, c["rsn_tok"] - cap) for r in ln for c in r["prim"])
    print("  [lean arm]      cap %4d: %5.1f s/run" % (cap, s / len(ln)))
# compaction removal
comp_s = sum(c["lat"] for r in R for c in r["compact"] if not c.get("exc"))
print("  compaction seconds per run: %.1f ; if compaction replaced by a deterministic 0-token summary: ~%.1f s/run" % (comp_s / len(R), comp_s / len(R)))
print("  format-failure seconds per run: %.1f" % (tot_fmt / len(R)))
# length cut waste
len_s = sum(c["lat"] for r in R for c in r["prim"] if c["finish"] == "length")
print("  finish=length calls: %.1f s/run" % (len_s / len(R)))
# no-tool-call (nudge) calls
nc_s = sum(c["lat"] for r in R for c in r["prim"] if not tc0(c))
print("  calls without any tool call (length/stop): %.1f s/run, n=%d" % (nc_s / len(R), sum(1 for r in R for c in r["prim"] if not tc0(c))))
# in-flight killed at cap
kill_s = sum(c["lat"] for r in R for c in r["cancelled"])
print("  killed in flight at cap: %.1f s/run; n=%d" % (kill_s / len(R), sum(len(r["cancelled"]) for r in R)))
out["cf"] = cf
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
