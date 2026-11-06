"""Gold-read -> first-source-edit gap in resolved runs (what happens between finding the file and editing).
Usage: python -I gap.py <trace_root> <per_run_table.jsonl> <out.json>"""
import json, sys, collections, statistics as st, os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "behaviour.py"), encoding="utf-8").read().replace("\nmain()\n", "\n")
ns = {"__file__": os.path.join(here, "behaviour.py"), "__name__": "behaviour_lib"}
exec(compile(src, "behaviour.py", "exec"), ns)
classify = ns["classify"]
root, tab, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(tab, encoding="utf-8")]
gaps, ex = [], []
for r in rows:
    if r["arm"] == "adapter" or not r["resolved"] or not r.get("gold"):
        continue
    p = f"{root}/{'pilot1' if r['src']=='pilot1' else 'ab1'}/{r['run']}.json"
    d = json.load(open(p, encoding="utf-8"))
    calls = []
    for s in d["steps"]:
        if s.get("source") != "agent" or not s.get("metrics"):
            continue
        tcs = s.get("tool_calls") or []
        k = [classify(t["function_name"], t.get("arguments") or {}) for t in tcs] or ["NO_CALL"]
        calls.append(dict(k=k[0], ct=s["metrics"]["completion_tokens"], msg=s.get("message") or "", txt=json.dumps(tcs)))
    g = next((i for i, c in enumerate(calls) if c["k"] in ("read", "search_read", "repro_run", "locate") and any(x.split("/")[-1] in c["txt"] for x in r["gold"])), None)
    e = next((i for i, c in enumerate(calls) if c["k"].startswith("SRC_EDIT")), None)
    if g is None or e is None:
        continue
    gaps.append(dict(run=r["run"], gold_read=g + 1, edit=e + 1, gap_kinds=[c["k"] for c in calls[g + 1:e]], gap_tokens=sum(c["ct"] for c in calls[g + 1:e])))
    if e - g > 6 and len(ex) < 5:
        ex.append((r["run"], [(c["k"], c["ct"], c["msg"][:330].replace("\n", " ")) for c in calls[g + 1:e][:5]]))
res = dict(n=len(gaps), median_gold_read_call=st.median(x["gold_read"] for x in gaps), median_first_src_edit=st.median(x["edit"] for x in gaps),
           median_gap_calls=st.median(x["edit"] - x["gold_read"] - 1 for x in gaps), mean_gap_tokens=st.mean(x["gap_tokens"] for x in gaps),
           gap_kind_counts=collections.Counter(k for x in gaps for k in x["gap_kinds"]).most_common(),
           runs_gap_ge5=sum(1 for x in gaps if x["edit"] - x["gold_read"] - 1 >= 5), examples=ex)
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)[:5000])
