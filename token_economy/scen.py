"""Prompt-lever scenarios on recorded traces: if the model edited at call <= E (instead of its recorded first source edit),
how many calls/tokens/seconds disappear?  Removed calls = the calls between max(gold_read, E_target-?) ... recorded first edit
(the earliest-gap calls are removed, the call with the edit is kept).  Re-edit penalty: fraction P_BAD of moved edits need
+RETRY calls of average cost (wrong early hypothesis).  Kaggle latency model.
Usage: python -I scen.py <trace_root> <per_run_table.jsonl> <out.json>"""
import json, sys, os, random, statistics as st
here = os.path.dirname(os.path.abspath(__file__))
ns = {"__file__": os.path.join(here, "behaviour.py"), "__name__": "behaviour_lib"}
exec(compile(open(os.path.join(here, "behaviour.py"), encoding="utf-8").read().replace("\nmain()\n", "\n"), "behaviour.py", "exec"), ns)
classify = ns["classify"]
A, B = 1.065, 0.02968
root, tab, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(tab, encoding="utf-8")]
runs = []
for r in rows:
    if r["arm"] == "adapter":
        continue
    d = json.load(open(f"{root}/{'pilot1' if r['src']=='pilot1' else 'ab1'}/{r['run']}.json", encoding="utf-8"))
    cs = []
    for s in d["steps"]:
        if s.get("source") != "agent" or not s.get("metrics"):
            continue
        tcs = s.get("tool_calls") or []
        k = [classify(t["function_name"], t.get("arguments") or {}) for t in tcs] or ["NO_CALL"]
        cs.append((k[0], s["metrics"]["completion_tokens"]))
    e = next((i for i, c in enumerate(cs) if c[0].startswith("SRC_EDIT")), None)
    runs.append(dict(r=r, cs=cs, e=e))
cost = lambda c: A + B * c[1]
def scen(E, P_BAD=0.3, RETRY=3):
    res = {}
    for name, f in (("resolved", lambda x: x["r"]["resolved"]), ("all_with_edit", lambda x: x["e"] is not None), ("ab1_timeout_edited", lambda x: x["r"]["src"] == "ab1" and x["r"]["timeout"] and x["e"] is not None)):
        sav, savn = [], []
        for x in runs:
            if not f(x) or x["e"] is None:
                continue
            target = E - 1                                  # 0-based index the edit should land on
            if x["e"] <= target:
                sav.append(0.0); savn.append(0); continue
            rm = x["cs"][target:x["e"]]                     # calls dropped
            avg = st.mean(cost(c) for c in x["cs"]) if x["cs"] else 0
            s = sum(cost(c) for c in rm) - P_BAD * RETRY * avg
            sav.append(s); savn.append(len(rm) - P_BAD * RETRY)
        if sav:
            res[name] = dict(n=len(sav), mean_sec_saved=st.mean(sav), mean_calls_saved=st.mean(savn),
                             pct_of_run_llm_time=100 * sum(sav) / sum(sum(cost(c) for c in x["cs"]) for x in runs if f(x) and x["e"] is not None))
    return res
o = {f"edit_by_{E}": scen(E) for E in (4, 5, 6, 8)}
o["sensitivity_P_BAD_0.6_edit_by_5"] = scen(5, P_BAD=0.6)
json.dump(o, open(out, "w"), indent=1)
print(json.dumps(o, indent=1))
