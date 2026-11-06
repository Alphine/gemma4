#!/usr/bin/env python3
"""Section 1: decompose the wall of each run (LLM primary / compaction / killed-in-flight / tool / gap) and the LLM time by token class.
usage: python a1_wall.py runs.json out.json"""
import json
import re
import sys
import collections
import numpy as np
from common import *

R = load(sys.argv[1])

# latency model fit on completed primary calls: lat = a + b*ct + c*pt
X, Y = [], []
for r in R:
    for c in r["prim"]:
        X.append([1, c["ct"], c["pt"]])
        Y.append(c["lat"])
X, Y = np.array(X, float), np.array(Y, float)
coef, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
pred = X @ coef
r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
print("latency fit lat = %.3f + %.4f*ct + %.6f*pt  R2=%.3f n=%d" % (coef[0], coef[1], coef[2], r2, len(Y)))
A, B, C = coef

out = {"fit": {"a": A, "b_per_ct": B, "c_per_pt": C, "r2": r2, "n": len(Y)}, "runs": [], "arms": {}}
per_run = []
for r in R:
    w = r["wall"]
    prim, comp, canc = r["prim"], r["compact"], r["cancelled"]
    llm_p = sum(c["lat"] for c in prim)
    llm_c = sum(c["lat"] for c in comp if not c.get("exc"))
    killed = sum(c["lat"] for c in canc)
    tool_by = collections.Counter()
    for c in prim:
        t = tc0(c)
        if c.get("tool_s") is not None and t:
            tool_by[t["sub"]] += c["tool_s"]
    tool = sum(tool_by.values())
    gap = w - llm_p - llm_c - killed - tool
    # token classes over primary calls
    t_rsn = t_tool = t_cont = 0.0
    for c in prim:
        other = max(c["ct"] - c["rsn_tok"], 0)
        ch = c["tc_chars"] + c["content_chars"]
        if ch > 0 and other > 0:
            t_tool += other * c["tc_chars"] / ch
            t_cont += other * c["content_chars"] / ch
        t_rsn += min(c["rsn_tok"], c["ct"])
    comp_ct = sum(c["ct"] for c in comp)
    ct_all = t_rsn + t_tool + t_cont
    s_rsn, s_tool, s_cont = B * t_rsn, B * t_tool, B * t_cont
    n_prim = len(prim)
    s_fix = A * n_prim                       # per-call fixed overhead (HTTP, scheduling, small prefill)
    s_pre = C * sum(c["pt"] for c in prim)   # prefill share (linear in prompt tokens)
    rec = dict(arm=r["arm"], task=r["task"], wall=w, llm_primary=llm_p, llm_compact=llm_c, killed_inflight=killed,
               tool=tool, gap=gap, n_prim=n_prim, n_compact=len(comp), compact_ct=comp_ct,
               tok_rsn=t_rsn, tok_toolarg=t_tool, tok_content=t_cont,
               s_decode_rsn=s_rsn, s_decode_toolarg=s_tool, s_decode_content=s_cont, s_fixed=s_fix, s_prefill=s_pre,
               tool_by=dict(tool_by), resolved=r["row"]["resolved"], err=r["row"]["err_class"],
               max_tool_s=max([c.get("tool_s") or 0 for c in prim] or [0]))
    per_run.append(rec)
out["runs"] = per_run

def table(sel, label):
    n = len(sel)
    keys = ["wall", "llm_primary", "llm_compact", "killed_inflight", "tool", "gap", "s_decode_rsn", "s_decode_toolarg",
            "s_decode_content", "s_fixed", "s_prefill", "n_prim", "n_compact", "tok_rsn", "tok_toolarg", "tok_content"]
    m = {k: mean(x[k] for x in sel) for k in keys}
    m["n"] = n
    m["tool_pct"] = 100 * m["tool"] / m["wall"]
    m["compact_pct"] = 100 * m["llm_compact"] / m["wall"]
    return m

for a in ARMS + [None]:
    sel = [x for x in per_run if a is None or x["arm"] == a]
    out["arms"][SHORT.get(a, "ALL")] = table(sel, a)
    m = out["arms"][SHORT.get(a, "ALL")]
    print(SHORT.get(a, "ALL"), {k: round(v, 1) for k, v in m.items()})

# tool time by sub, totals and counts
tt = collections.Counter(); tn = collections.Counter(); tmax = {}
for r in R:
    for c in r["prim"]:
        t = tc0(c)
        if t and c.get("tool_s") is not None:
            tt[t["sub"]] += c["tool_s"]; tn[t["sub"]] += 1
            tmax[t["sub"]] = max(tmax.get(t["sub"], 0), c["tool_s"])
out["tool_by_sub"] = {k: dict(total_s=tt[k], n=tn[k], mean_s=tt[k] / tn[k], max_s=tmax[k]) for k in tt}
print("TOOL time by sub (all 40 runs):")
for k, v in sorted(out["tool_by_sub"].items(), key=lambda kv: -kv[1]["total_s"]):
    print("  %-22s n=%4d total=%6.1fs mean=%.2fs max=%.1fs" % (k, v["n"], v["total_s"], v["mean_s"], v["max_s"]))
tot_tool = sum(tt.values()); tot_wall = sum(x["wall"] for x in per_run)
print("total tool %.1fs of %.0fs wall = %.2f%%" % (tot_tool, tot_wall, 100 * tot_tool / tot_wall))

# timeouts: commands with a `timeout N` prefix, exit 124
n_pref = n_124 = 0
ex = []
for r in R:
    for c in r["prim"]:
        t = tc0(c)
        if t and t["name"] == "run_command":
            cmd = str(t["args"].get("command", ""))
            if re.search(r"\btimeout\s+\d+", cmd):
                n_pref += 1
            if '"exit_code": 124' in (c.get("obs_head") or "") or "exit_code\": 124" in (c.get("obs_head") or ""):
                n_124 += 1; ex.append((r["arm"], r["task"], cmd[:100], c.get("tool_s")))
print("run_command with timeout prefix:", n_pref, "; exit_code 124:", n_124, ex[:5])
out["timeout_prefix_cmds"] = n_pref; out["exit124"] = n_124
# slowest tool calls
slow = []
for r in R:
    for c in r["prim"]:
        t = tc0(c)
        if t and (c.get("tool_s") or 0) > 1.5:
            slow.append((round(c["tool_s"], 1), r["arm"][11:], r["task"], t["sub"], str(t["args"].get("command", t["args"].get("file_path", "")))[:90]))
print("tool calls > 1.5 s:", len(slow))
for s in sorted(slow, reverse=True)[:15]:
    print("  ", s)
out["slow_tools"] = sorted(slow, reverse=True)[:30]

# compaction anatomy
cl = []
for r in R:
    for c in r["compact"]:
        if c.get("exc"):
            continue
        cl.append(dict(arm=r["arm"], task=r["task"], lat=c["lat"], ct=c["ct"], pt=c["pt"], rsn=c["rsn_tok"], t=c["t_send"]))
print("compaction calls: n=%d  mean lat=%.1fs  ct mean=%.0f  pt mean=%.0f  at t mean=%.0fs; total %.0fs" %
      (len(cl), mean(x["lat"] for x in cl), mean(x["ct"] for x in cl), mean(x["pt"] for x in cl), mean(x["t"] for x in cl), sum(x["lat"] for x in cl)))
out["compactions"] = cl
# prompt tokens right before compaction
json.dump(out, open(sys.argv[2], "w"), indent=1)
