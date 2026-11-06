#!/usr/bin/env python3
"""Section 2: what the pre-edit calls are, how many are redundant, gold-file hit of the first edit, doubt in the reasoning.
usage: python a2_preedit.py runs.json out.json"""
import json
import re
import sys
import collections
from common import *

R = load(sys.argv[1])
BS = chr(92)


def norm_path(p):
    p = str(p or "").replace(BS, "/").strip().strip("'\"`")
    for pre in ("/workspace/", "./"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p.lstrip("/")


def is_test_path(p):
    p = norm_path(p)
    return p.startswith("tests/") or "/tests/" in p or os.path.basename(p).startswith("test_") or p.endswith("conftest.py")


def parse_cmd_reads(cmd):
    """-> list of (path, lo, hi) for sed -n 'A,Bp' f / head -N f / cat f / nl; [] when not a file read"""
    out = []
    for m in re.finditer(r"sed\s+-n\s+['\"]?(\d+)(?:,(\d+))?p['\"]?\s+([^\s|;&]+)", cmd):
        lo = int(m.group(1)); hi = int(m.group(2) or lo)
        out.append((norm_path(m.group(3)), lo, hi))
    if not out:
        for m in re.finditer(r"\b(?:cat|head(?:\s+-n?\s*\d+)?|tail(?:\s+-n?\s*\d+)?|nl)\s+(?:-\w+\s+)*([^\s|;&<>]+\.\w+)", cmd):
            out.append((norm_path(m.group(1)), 1, 150))
    return out


def classify(c, gold_files, seen_locate):
    t = tc0(c)
    if not t:
        return "nocall", []
    sub, a = t["sub"], t["args"]
    reads = []
    if t["name"] == "read_file":
        p = norm_path(a.get("filepath"))
        lo = int(a.get("start_line") or 1)
        hi = int(a.get("end_line") or lo + 149)
        reads = [(p, lo, hi)]
    elif t["name"] == "run_command":
        reads = parse_cmd_reads(str(a.get("command", "")))
    if sub == "skill:locate.py":
        return "locate", []
    if reads:
        p = reads[0][0]
        if p in gold_files:
            return "read_gold", reads
        if is_test_path(p):
            return "read_test", reads
        return "read_other", reads
    if sub in ("grep_find",):
        return "grep", []
    if sub in ("pytest", "skill:check.py"):
        return "run_test", []
    if sub == "python":
        return "repro", []
    if t["name"] in ("write_file",) or sub in ("write_file",):
        return "write_scratch", []
    if sub in ("git",):
        return "git", []
    if sub in ("ls", "other_shell", "other", "sed_cat"):
        return "ls_other", []
    return sub, []


def covered(rngs, lo, hi):
    n = hi - lo + 1
    if n <= 0:
        return 1.0
    got = 0
    for x in range(lo, hi + 1):
        if any(a <= x <= b for a, b in rngs):
            got += 1
    return got / n


DOUBT = re.compile(r"\b(wait|actually|hmm|however|but (?:what|if|is|does|the|I)|not (?:sure|certain)|to be sure|double[- ]check|"
                   r"re-?check|re-?read|verify|make sure|let me (?:re|look|check|see|read|view|examine|find|search|run|try)|I (?:need|should|must|will) (?:to )?(?:check|look|read|see|verify|find|understand|examine))\b", re.I)
RECHECK = re.compile(r"\b(double[- ]check|re-?check|re-?read|re-?examine|again|verify|make sure|confirm|to be sure|just to)\b", re.I)

out = {"runs": []}
cls_all = collections.Counter()
by_arm = collections.defaultdict(lambda: collections.Counter())
red_rows = []
first_edit_hits = []
for r in R:
    prim = r["prim"]
    fe = first_edit_index(prim)
    g = r["gold"] or {}
    gold_files = set(g.get("gold_lines", {}).keys())
    gold_funcs = g.get("gold_funcs", [])
    n_pre = fe if fe is not None else len(prim)
    seen = collections.defaultdict(list)       # path -> ranges read
    seen_cmds = set()
    pre = []
    for k, c in enumerate(prim[:n_pre]):
        cl, reads = classify(c, gold_files, False)
        t = tc0(c)
        redundant = None
        if reads:
            fr = [covered(seen[p], lo, hi) for p, lo, hi in reads]
            cov = min(fr)
            if cov >= 0.8:
                redundant = "reread_cov%.0f%%" % (100 * cov)
            for p, lo, hi in reads:
                seen[p].append((lo, hi))
        if t and t["name"] == "run_command":
            key = re.sub(r"\s+", " ", re.sub(r"\btimeout\s+\d+\s+", "", str(t["args"].get("command", "")))).strip()
            if key in seen_cmds:
                redundant = redundant or "same_cmd"
            seen_cmds.add(key)
        elif t and t["name"] == "run_skill_script" and t["sub"] == "skill:locate.py":
            key = "LOCATE" + json.dumps(t["args"].get("args"), sort_keys=True)
            if key in seen_cmds:
                redundant = redundant or "same_cmd"
            seen_cmds.add(key)
        # the failed-args retries are "redundant" in a different sense; tag separately
        retry = c.get("err_kind") in ("invalid_args", "skill_soft")
        doubt = bool(DOUBT.search(c.get("message") or ""))
        recheck = bool(RECHECK.search(c.get("message") or ""))
        pre.append(dict(k=k, cls=cl, sub=t["sub"] if t else None, redundant=redundant, retry=retry, doubt=doubt, recheck=recheck,
                        lat=c["lat"], ct=c["ct"], rsn=c["rsn_tok"], first=(c.get("message") or "")[:90].replace("\n", " ")))
        cls_all[cl] += 1
        by_arm[r["arm"]][cl] += 1
    # first edit placement
    hit = None
    if fe is not None:
        c = prim[fe]
        t = tc0(c)
        a = t["args"]
        if t["name"] == "edit_file":
            p = norm_path(a.get("filepath"))
            rng = None
        elif t["name"] == "run_skill_script":
            inner = a.get("args") if isinstance(a.get("args"), dict) else {}
            blob_ = json.dumps(a)
            p = norm_path(inner.get("path") or inner.get("file_path") or a.get("path") or "")
            if not p:    # malformed shape (path swallowed into another key): recover the first .py path mentioned
                m_ = re.search(r"([\w/.\-]+\.py)", blob_.replace("scripts/edit_lines.py", ""))
                p = norm_path(m_.group(1)) if m_ else ""
            try:
                rng = (int(inner.get("start")), int(inner.get("end")))
            except Exception:
                rng = None
        elif t["name"] == "write_file":
            p = norm_path(a.get("filepath")); rng = None
        else:
            p = ""; rng = None
            m = re.search(r"sed\s+-i[^\n]*?\s([\w/.\-]+\.py)\s*$", str(a.get("command", "")))
            if m:
                p = norm_path(m.group(1))
        infile = p in gold_files
        infunc = None
        if rng and infile:
            infunc = any(fp == p and not (rng[1] < s or rng[0] > e) for fp, q, s, e in gold_funcs)
        hit = dict(path=p, tool=t["name"], gold_file=infile, in_gold_func=infunc, has_gold=bool(gold_files))
    first_edit_hits.append(dict(arm=r["arm"], task=r["task"], **(hit or {})))
    npre_calls = len(pre)
    out["runs"].append(dict(arm=r["arm"], task=r["task"], n_pre=npre_calls, first_edit_idx=fe,
                            t_first_edit=prim[fe]["t_send"] if fe is not None else None,
                            t_first_edit_end=prim[fe]["t_end"] if fe is not None else None,
                            pre=pre, gold_hit=hit, resolved=r["row"]["resolved"], err=r["row"]["err_class"]))

# summary
print("== pre-edit class counts (all runs; calls before first edit attempt; never-edited runs count all calls) ==")
for a in ARMS:
    n = sum(by_arm[a].values())
    print(SHORT[a], n, dict(by_arm[a].most_common()))
print("ALL", sum(cls_all.values()), dict(cls_all.most_common()))
nruns = len(R)
tot_pre = sum(x["n_pre"] for x in out["runs"])
print("mean pre-edit calls per run:", tot_pre / nruns)
edited = [x for x in out["runs"] if x["first_edit_idx"] is not None]
print("runs with an edit attempt:", len(edited), "mean idx", mean(x["first_edit_idx"] for x in edited), "median", pct([x["first_edit_idx"] for x in edited], .5),
      "mean t_first_edit_s", mean(x["t_first_edit"] for x in edited))
# redundancy
allpre = [p for x in out["runs"] for p in x["pre"]]
nred = sum(1 for p in allpre if p["redundant"])
print("redundant pre-edit calls: %d / %d = %.1f%%" % (nred, len(allpre), 100 * nred / len(allpre)))
print("   by class:", collections.Counter(p["cls"] for p in allpre if p["redundant"]))
print("   retry (invalid-args/usage) pre-edit:", sum(p["retry"] for p in allpre))
print("   lat spent on redundant calls: %.0fs of %.0fs pre-edit LLM" % (sum(p["lat"] for p in allpre if p["redundant"]), sum(p["lat"] for p in allpre)))
print("   doubt marker in reasoning: %d / %d; recheck marker: %d" % (sum(p["doubt"] for p in allpre), len(allpre), sum(p["recheck"] for p in allpre)))
for a in ARMS:
    pp = [p for x in out["runs"] if x["arm"] == a for p in x["pre"]]
    n = len(pp)
    print(SHORT[a], "pre calls", n, "redundant", sum(1 for p in pp if p["redundant"]), "retry", sum(p["retry"] for p in pp), "doubt", sum(p["doubt"] for p in pp), "recheck", sum(p["recheck"] for p in pp))
# per class latency
cl_lat = collections.defaultdict(list)
for p in allpre:
    cl_lat[p["cls"]].append(p["lat"])
print("latency by class (s): ", {k: (len(v), round(mean(v), 1), round(sum(v))) for k, v in cl_lat.items()})
# gold hits
print("== first edit placement ==")
hh = [h for h in first_edit_hits if h.get("tool")]
hg = [h for h in hh if h["has_gold"]]
print("edited runs", len(hh), "with gold info", len(hg), "gold file hit", sum(h["gold_file"] for h in hg),
      "func hit (where range known)", sum(1 for h in hg if h["in_gold_func"]), "/", sum(1 for h in hg if h["in_gold_func"] is not None))
for h in hh:
    print("  ", SHORT[h["arm"]], h["task"], h["tool"], h["path"], "gold_file" if h["gold_file"] else ("NOTGOLD" if h["has_gold"] else "nogold"), h["in_gold_func"])
# examples of redundant calls with reasoning first words
print("== examples of redundant calls ==")
ex = [(x["arm"], x["task"], p) for x in out["runs"] for p in x["pre"] if p["redundant"]]
for a, t, p in ex[:25]:
    print("  ", SHORT[a], t, "k=%d" % p["k"], p["cls"], p["redundant"], "|", p["first"][:70])
json.dump(out, open(sys.argv[2], "w"), indent=1)
