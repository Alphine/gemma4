"""Why do resolved runs miss 'edit by call 6'? Classify pre-first-edit calls and attribute tokens/seconds.
Usage: python -I behaviour.py <trace_root> <per_run_table.jsonl> <out.json>"""
import json, re, sys, statistics as st, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_tokens as S

TEST = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|conftest\.py$)")

def classify(name, args):
    if name == "read_file":
        return "read"
    if name == "edit_file":
        p = args.get("filepath") or args.get("path") or ""
        return "edit_test" if TEST.search(p) else "SRC_EDIT"
    if name == "write_file":
        p = args.get("filepath") or args.get("path") or ""
        return "scratch_write" if (TEST.search(p) or "repro" in p or "/tmp" in p or "reproduce" in p or not p.endswith(".py") or p.count("/") == 0 and "test" in p) else "SRC_EDIT_NEWFILE"
    if name == "run_skill_script":
        fp = str(args.get("file_path", ""))
        a = args.get("args") or {}
        if "edit_lines" in fp:
            return "edit_test" if TEST.search(str(a.get("path", "")) if isinstance(a, dict) else "") else "SRC_EDIT"
        if "locate" in fp:
            return "locate"
        if "check" in fp:
            return "check"
        return "skill_other"
    if name == "run_command":
        c = str(args.get("command", ""))
        if re.search(r"pytest|py\.test|unittest", c):
            return "test"
        if re.search(r"cat\s*<<|>\s*\S*repro|>\s*/tmp|python3?\s+\S*(repro|reproduce|test_)\S*\.py", c):
            return "repro_write"
        if re.search(r"python3?\s+(-c|-)", c) or re.search(r"python3?\s+\S+\.py", c):
            return "repro_run"
        if re.search(r"sed -i|perl -pi|patch\b|git apply", c):
            return "SRC_EDIT"
        if re.search(r"grep|rg |find |ls\b|sed -n|cat |head|tail|git (log|show|diff|status)|wc ", c):
            return "search_read"
        return "other_cmd"
    if name in ("submit_patch", "get_status"):
        return name
    return "other"

def main():
    root, tab, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rows = [json.loads(l) for l in open(tab, encoding="utf-8")]
    recs = []
    for r in rows:
        if r["arm"] == "adapter":
            continue
        p = root / ("pilot1" if r["src"] == "pilot1" else "ab1") / f"{r['run']}.json"
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        calls = []
        for s in d["steps"]:
            if s.get("source") != "agent" or not s.get("metrics"):
                continue
            tcs = s.get("tool_calls") or []
            ct = int(s["metrics"]["completion_tokens"])
            kinds = [classify(t["function_name"], t.get("arguments") or {}) for t in tcs] or ["NO_CALL"]
            calls.append(dict(ct=ct, kinds=kinds, k=kinds[0], msg=len(s.get("message") or "")))
        first = next((i for i, c in enumerate(calls) if any(k.startswith("SRC_EDIT") for k in c["kinds"])), None)
        recs.append(dict(r=r, calls=calls, first=first))
    res = {"n": len(recs)}
    def grp(name, f):
        g = [x for x in recs if f(x)]
        n6 = [x for x in g if x["first"] is not None and x["first"] < 6]
        late = [x for x in g if x["first"] is not None and x["first"] >= 6]
        none = [x for x in g if x["first"] is None]
        res[name] = dict(n=len(g), edit_by_6=len(n6), edit_later=len(late), never_edited=len(none),
                         first_edit_call_median=st.median([x["first"] + 1 for x in g if x["first"] is not None]) if any(x["first"] is not None for x in g) else None)
        # pre-edit call-kind mix + tokens, for late editors
        for lab, sub in (("late", late), ("early", n6), ("never", none)):
            cnt = collections.Counter(); tok = collections.Counter(); calls_n = 0; toks = 0
            gap = collections.Counter()
            for x in sub:
                lim = x["first"] if x["first"] is not None else len(x["calls"])
                for c in x["calls"][:lim]:
                    cnt[c["k"]] += 1; tok[c["k"]] += c["ct"]; calls_n += 1; toks += c["ct"]
            if sub:
                res[name][lab] = dict(runs=len(sub), pre_edit_calls_per_run=calls_n / len(sub), pre_edit_tokens_per_run=toks / len(sub),
                                      kind_share_calls={k: round(v / calls_n, 3) for k, v in cnt.most_common()},
                                      kind_share_tokens={k: round(v / toks, 3) for k, v in tok.most_common()},
                                      big_thought_calls_ge1000_per_run=sum(c["ct"] >= 1000 for x in sub for c in x["calls"][:(x["first"] if x["first"] is not None else len(x["calls"]))]) / len(sub))
    grp("pilot1_resolved", lambda x: x["r"]["src"] == "pilot1" and x["r"]["resolved"])
    grp("ab1_resolved", lambda x: x["r"]["src"] == "ab1" and x["r"]["resolved"])
    grp("all_resolved", lambda x: x["r"]["resolved"])
    grp("ab1_timeout", lambda x: x["r"]["src"] == "ab1" and x["r"]["timeout"])
    grp("pilot1_unresolved", lambda x: x["r"]["src"] == "pilot1" and not x["r"]["resolved"])
    # repro-first prevalence: first call kind
    fc = collections.Counter(x["calls"][0]["k"] for x in recs if x["calls"])
    res["first_call_kind"] = fc.most_common()
    # in resolved runs: P(resolved) by edit by 6 / reproduction before edit
    def rate(f):
        g = [x for x in recs if f(x)]
        return (len(g), sum(x["r"]["resolved"] for x in g))
    res["resolve_by_first_edit"] = {
        "edit<=6": rate(lambda x: x["first"] is not None and x["first"] < 6),
        "edit7-10": rate(lambda x: x["first"] is not None and 6 <= x["first"] < 10),
        "edit>10": rate(lambda x: x["first"] is not None and x["first"] >= 10),
        "never": rate(lambda x: x["first"] is None)}
    res["resolve_by_repro_before_edit"] = {
        "repro_written_before_edit": rate(lambda x: any(c["k"] in ("repro_write", "scratch_write") for c in x["calls"][:(x["first"] if x["first"] is not None else len(x["calls"]))])),
        "no_repro_before_edit": rate(lambda x: not any(c["k"] in ("repro_write", "scratch_write") for c in x["calls"][:(x["first"] if x["first"] is not None else len(x["calls"]))]))}
    res["resolve_by_first_call"] = {k: rate(lambda x, k=k: x["calls"] and x["calls"][0]["k"] == k) for k in fc}
    # resolve by max thought size
    for B in (1024, 2048):
        res[f"resolve_by_maxct_ge{B}"] = {"has_call>=B": rate(lambda x, B=B: any(c["ct"] >= B for c in x["calls"])), "none": rate(lambda x, B=B: not any(c["ct"] >= B for c in x["calls"]))}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:9000])

main()
