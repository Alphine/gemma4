#!/usr/bin/env python3
"""Section 4: arm summary, per-task matrix, decide-vs-lean divergence (same prompt, budget 4096 vs 1024).
usage: python a5_arms.py runs.json out.json"""
import json
import sys
import collections
from common import *

R = load(sys.argv[1])
idx = {(r["arm"], r["task"]): r for r in R}
tasks = sorted({r["task"] for r in R})
out = {"arms": {}, "tasks": {}, "diverge": []}

print("== arm summary ==")
for a in ARMS:
    rs = [r for r in R if r["arm"] == a]
    n = len(rs)
    res = sum(1 for r in rs if r["row"]["resolved"])
    never = sum(1 for r in rs if first_edit_index(r["prim"]) is None)
    cap = sum(1 for r in rs if r["row"]["err_class"] == "time_cap")
    early = [r for r in rs if r["row"]["err_class"] != "time_cap"]
    d = dict(n=n, resolved=res, never_edited=never, time_cap=cap, mean_wall=mean(r["wall"] for r in rs),
             mean_calls=mean(len(r["prim"]) for r in rs), mean_ct=mean(sum(c["ct"] for c in r["prim"]) for r in rs),
             finished_early=len(early), mean_wall_finished=mean(r["wall"] for r in early),
             nudges=sum(r["n_nudges"] for r in rs), cancelled_inflight=sum(len(r["cancelled"]) for r in rs))
    out["arms"][SHORT[a]] = d
    print(SHORT[a], d)

print("== per-task matrix  (R=resolved, E=edited unresolved, N=never edited; wall s; t_first_edit; n_calls; max rsn) ==")
for t in tasks:
    cells = []
    for a in ARMS:
        r = idx.get((a, t))
        if r is None:
            cells.append("--"); continue
        fe = first_edit_index(r["prim"])
        st_ = "R" if r["row"]["resolved"] else ("E" if fe is not None else "N")
        te = r["prim"][fe]["t_send"] if fe is not None else None
        cells.append("%s w=%3.0f fe=%s n=%2d maxr=%4d" % (st_, r["wall"], ("%3.0f" % te) if te is not None else " - ", len(r["prim"]), max(c["rsn_tok"] for c in r["prim"])))
        out["tasks"].setdefault(t, {})[SHORT[a]] = dict(state=st_, wall=r["wall"], t_first_edit=te, n_calls=len(r["prim"]))
    print("  %-14s | %s" % (t, " | ".join(cells)))
anyres = [t for t in tasks if any(out["tasks"][t].get(SHORT[a], {}).get("state") == "R" for a in ARMS)]
allres = [t for t in tasks if all(out["tasks"][t].get(SHORT[a], {}).get("state") == "R" for a in ARMS if SHORT[a] in out["tasks"][t])]
print("tasks resolved by >=1 arm:", len(anyres), "by all arms:", len(allres), "by none:", [t for t in tasks if t not in anyres])

print("== decide vs lean: first divergence (identical prompt & seed; only thinking budget 4096 vs 1024 differs) ==")
rows = []
for t in tasks:
    d = idx.get(("edit-early-v7-decide-nt", t)); l = idx.get(("edit-early-v7-lean-nt", t))
    if not d or not l:
        continue
    k = 0
    while k < min(len(d["prim"]), len(l["prim"])):
        cd, cl = d["prim"][k], l["prim"][k]
        td, tl = tc0(cd), tc0(cl)
        same = (td and tl and td["name"] == tl["name"] and json.dumps(td["args"], sort_keys=True) == json.dumps(tl["args"], sort_keys=True))
        if not same or cd["ct"] != cl["ct"]:
            break
        k += 1
    cd = d["prim"][k] if k < len(d["prim"]) else None
    cl = l["prim"][k] if k < len(l["prim"]) else None
    rec = dict(task=t, same_prefix_calls=k, decide_rsn=cd["rsn_tok"] if cd else None, lean_rsn=cl["rsn_tok"] if cl else None,
               decide_next=(tc0(cd) or {}).get("sub") if cd else None, lean_next=(tc0(cl) or {}).get("sub") if cl else None,
               decide_res=d["row"]["resolved"], lean_res=l["row"]["resolved"], t_diverge=cd["t_send"] if cd else None)
    rows.append(rec)
    print("  %-14s identical for %2d calls (t=%s); at divergence decide rsn=%s -> %s | lean rsn=%s -> %s ; resolved decide=%s lean=%s" %
          (t, k, ("%.0fs" % rec["t_diverge"]) if rec["t_diverge"] is not None else "-", rec["decide_rsn"], rec["decide_next"], rec["lean_rsn"], rec["lean_next"], rec["decide_res"], rec["lean_res"]))
out["diverge"] = rows

# what happens right after a budget cut in the lean arm
print("== lean: calls ending at the 1024 budget: what comes next ==")
cut = []
for r in R:
    if r["arm"] != "edit-early-v7-lean-nt":
        continue
    p = r["prim"]
    for k, c in enumerate(p):
        if c["rsn_tok"] >= 1000:
            t = tc0(c)
            nxt = tc0(p[k + 1]) if k + 1 < len(p) else None
            cut.append(dict(task=r["task"], k=k, ct=c["ct"], tc_chars=c["tc_chars"], sub=t["sub"] if t else None, fail=c.get("err_kind"), nxt=nxt["sub"] if nxt else None))
print("  n=%d ; follow-up tool kinds: %s ; failed tool-call rate %s/%d" % (len(cut), collections.Counter(c["sub"] for c in cut).most_common(8),
      sum(1 for c in cut if c["fail"]), len(cut)))
# failure rate of calls by thinking length bucket (all arms)
bk = collections.defaultdict(lambda: [0, 0])
for r in R:
    for c in r["prim"]:
        t = tc0(c)
        if not t:
            continue
        b = "<=100" if c["rsn_tok"] <= 100 else "100-400" if c["rsn_tok"] <= 400 else "400-1000" if c["rsn_tok"] <= 1000 else ">1000"
        bk[b][0] += 1; bk[b][1] += bool(c.get("err_kind") in ("invalid_args", "skill_soft"))
print("  format-failure rate by reasoning length:", {k: "%d/%d" % (v[1], v[0]) for k, v in bk.items()})
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
