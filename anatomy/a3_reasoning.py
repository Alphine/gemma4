#!/usr/bin/env python3
"""Section 3: reasoning-token anatomy per arm. usage: python a3_reasoning.py runs.json out.json"""
import json
import re
import sys
import collections
from common import *

R = load(sys.argv[1])
FENCE = re.compile(r"```.*?```", re.S)
WAIT = re.compile(r"\b(wait|actually|hmm|but wait|let me re-?(?:check|read|trace|think|examine)|re-?check)\b", re.I)
OPTS = re.compile(r"\b(alternatively|option \d|approach \d|another (?:approach|option|way)|we could|one way|or we|either)\b", re.I)
PLAN = re.compile(r"PLAN:", re.I)


def rep_frac(text, n=6):
    """fraction of word n-gram shingles that occurred earlier in the same thought (degeneracy / re-deriving proxy)"""
    w = re.findall(r"\S+", text)
    if len(w) < n + 5:
        return 0.0
    seen = set(); rep = 0; tot = 0
    for i in range(len(w) - n + 1):
        g = tuple(w[i:i + n])
        tot += 1
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / tot


def code_frac(text):
    if not text:
        return 0.0
    f = sum(len(m.group(0)) for m in FENCE.finditer(text))
    return f / len(text)


calls = []
for r in R:
    for c in r["prim"]:
        t = tc0(c)
        msg = c.get("message") or ""
        calls.append(dict(arm=r["arm"], task=r["task"], k=c["i"], rsn=c["rsn_tok"], ct=c["ct"], lat=c["lat"], finish=c["finish"],
                          sub=t["sub"] if t else None, rep=rep_frac(msg), code=code_frac(msg), waits=len(WAIT.findall(msg)),
                          opts=len(OPTS.findall(msg)), plan=bool(PLAN.search(msg)), chars=len(msg), maxc=c.get("max_comp"), tb=c.get("think_budget"),
                          tc_chars=c["tc_chars"], err=c.get("err_kind"), msg=msg))
out = {"arms": {}}
BINS = [(0, 50), (50, 150), (150, 400), (400, 1000), (1000, 2000), (2000, 4000), (4000, 10000)]
print("== reasoning tokens per call (primary calls) ==")
for a in ARMS + [None]:
    cs = [c for c in calls if a is None or c["arm"] == a]
    rs = [c["rsn"] for c in cs]
    tot = sum(rs)
    d = dict(n=len(cs), mean=mean(rs), median=pct(rs, .5), p75=pct(rs, .75), p90=pct(rs, .9), p95=pct(rs, .95), max=max(rs),
             share_gt1000_calls=sum(1 for x in rs if x > 1000) / len(rs), share_gt1000_tokens=sum(x for x in rs if x > 1000) / tot,
             share_gt400_tokens=sum(x for x in rs if x > 400) / tot,
             sec_gt1000=sum(c["lat"] for c in cs if c["rsn"] > 1000), sec_total=sum(c["lat"] for c in cs),
             n_len=sum(1 for c in cs if c["finish"] == "length"), sec_len=sum(c["lat"] for c in cs if c["finish"] == "length"),
             n_nocall=sum(1 for c in cs if c["sub"] is None), n_ge_budget=sum(1 for c in cs if c["tb"] and c["rsn"] >= c["tb"] - 40),
             bins={f"{lo}-{hi}": dict(n=sum(1 for x in rs if lo <= x < hi), tok=sum(x for x in rs if lo <= x < hi),
                                       sec=sum(c["lat"] for c in cs if lo <= c["rsn"] < hi)) for lo, hi in BINS})
    out["arms"][SHORT.get(a, "ALL")] = d
    print(SHORT.get(a, "ALL"), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "bins"})
    for b, v in d["bins"].items():
        print("     rsn %-10s n=%4d tok=%6d sec=%6.0f (%.0f%% of llm s)" % (b, v["n"], v["tok"], v["sec"], 100 * v["sec"] / d["sec_total"]))

# call-type vs reasoning: does reasoning length depend on what the call is?
print("== mean reasoning tokens by tool kind ==")
by = collections.defaultdict(list)
for c in calls:
    by[c["sub"]].append(c["rsn"])
for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
    print("  %-22s n=%4d mean=%6.0f median=%5.0f sum=%7d" % (k, len(v), mean(v), pct(v, .5), sum(v)))
out["by_sub"] = {str(k): dict(n=len(v), mean=mean(v), sum=sum(v)) for k, v in by.items()}

# long thoughts anatomy
long = [c for c in calls if c["rsn"] > 1000]
print("== long thoughts (>1000 rsn tokens): n=%d ==" % len(long))
print("  mean rep_frac(6-gram)=%.2f  code_frac=%.2f  waits/1k tok=%.1f  opts=%.1f/thought  contains PLAN=%d" %
      (mean(c["rep"] for c in long), mean(c["code"] for c in long), 1000 * sum(c["waits"] for c in long) / sum(c["rsn"] for c in long),
       mean(c["opts"] for c in long), sum(c["plan"] for c in long)))
short = [c for c in calls if c["rsn"] <= 400]
print("  short (<=400): rep_frac=%.2f code_frac=%.2f waits/1k tok=%.1f" % (mean(c["rep"] for c in short), mean(c["code"] for c in short),
      1000 * sum(c["waits"] for c in short) / max(1, sum(c["rsn"] for c in short))))
deg = [c for c in calls if c["rep"] > 0.25 and c["rsn"] > 500]
print("  degenerate-ish (rep_frac>0.25 & rsn>500): n=%d tokens=%d sec=%.0f" % (len(deg), sum(c["rsn"] for c in deg), sum(c["lat"] for c in deg)))
# classify each long thought into dominant type
def kind(c):
    if c["finish"] == "length":
        return "length-cut"
    if c["code"] > 0.25:
        return "re-reads code in thought"
    if c["rep"] > 0.25:
        return "repetitive/re-derives"
    if c["opts"] >= 3:
        return "lists options"
    if c["waits"] >= 5:
        return "wait/second-guess"
    return "other"
kc = collections.Counter(); ks = collections.Counter(); kt = collections.Counter()
for c in long:
    k = kind(c); kc[k] += 1; ks[k] += c["lat"]; kt[k] += c["rsn"]
for k in kc:
    print("   %-26s n=%3d tokens=%6d sec=%6.0f" % (k, kc[k], kt[k], ks[k]))
out["long_kinds"] = {k: dict(n=kc[k], tokens=kt[k], sec=ks[k]) for k in kc}
# what follows a long thought: tool kind
fl = collections.Counter(c["sub"] for c in long)
print("  long thoughts end in:", dict(fl.most_common()))
# location of long thoughts: pre-edit or post-edit
pre = post = 0
for r in R:
    fe = first_edit_index(r["prim"])
    for k, c in enumerate(r["prim"]):
        if c["rsn_tok"] > 1000:
            if fe is None or k < fe:
                pre += 1
            else:
                post += 1
print("  long thoughts before first edit:", pre, "after/at:", post)
# preview some long thoughts (first words)
print("== sample long-thought openings ==")
for c in sorted(long, key=lambda x: -x["rsn"])[:12]:
    print("  %s %s call#%d rsn=%d lat=%.0fs sub=%s rep=%.2f code=%.2f | %s" % (SHORT[c["arm"]], c["task"], c["k"], c["rsn"], c["lat"], c["sub"], c["rep"], c["code"], c["msg"][:80].replace("\n", " ")))
# lean arm: budget cut?
print("== lean arm budget behaviour ==")
ln = [c for c in calls if c["arm"] == "edit-early-v7-lean-nt"]
print("  thinking budget values:", collections.Counter(c["tb"] for c in ln), "max_comp:", collections.Counter(c["maxc"] for c in ln))
rs = sorted(c["rsn"] for c in ln)
print("  rsn quantiles", [round(pct(rs, q)) for q in (.5, .75, .9, .95, .99)], "max", rs[-1], " n>=1000:", sum(1 for x in rs if x >= 1000), " n in 1000..1040:", sum(1 for x in rs if 1000 <= x <= 1040))
print("  finish counts lean:", collections.Counter(c["finish"] for c in ln))
for a in ARMS:
    print("  finish counts", SHORT[a], collections.Counter(c["finish"] for c in calls if c["arm"] == a))
json.dump(out, open(sys.argv[2], "w"), indent=1)
