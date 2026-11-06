"""Arm-level first-order predictions: prompt lever (remove a fraction RHO of the 'excess gap' calls between call 6 and the
recorded first source edit) then thinking-budget B on the remaining calls.  Kaggle latency model.  Bootstrap over runs.
Usage: python -I combo.py <trace_root> <per_run_table.jsonl> <out.json>"""
import json, sys, os, random, statistics as st
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import sim_tokens as S
ns = {"__file__": os.path.join(here, "behaviour.py"), "__name__": "behaviour_lib"}
exec(compile(open(os.path.join(here, "behaviour.py"), encoding="utf-8").read().replace("\nmain()\n", "\n"), "behaviour.py", "exec"), ns)
classify = ns["classify"]
A, Bk = S.LAT["kaggle"]
root, tab, out = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(tab, encoding="utf-8")]
runs = []
for r in rows:
    if r["arm"] == "adapter":
        continue
    p = f"{root}/{'pilot1' if r['src']=='pilot1' else 'ab1'}/{r['run']}.json"
    calls = S.load_calls(p)
    d = json.load(open(p, encoding="utf-8"))
    ks = []
    for s in d["steps"]:
        if s.get("source") != "agent" or not s.get("metrics"):
            continue
        tcs = s.get("tool_calls") or []
        ks.append(([classify(t["function_name"], t.get("arguments") or {}) for t in tcs] or ["NO_CALL"])[0])
    e = next((i for i, k in enumerate(ks) if k.startswith("SRC_EDIT")), None)
    runs.append(dict(r=r, calls=calls, e=e))

def run_cost(x, B=None, M=None, rho=0.0, p_bad=0.3):
    cs = x["calls"]
    if x["e"] is not None and x["e"] > 5 and rho > 0:
        n_rm = (x["e"] - 5) * rho                      # fractional calls removed from positions 5..e-1
        gap = cs[5:x["e"]]
        keep_frac = 1 - rho
        scale = keep_frac
    else:
        gap, scale = [], 1.0
    tok = secs = ncalls = 0.0
    for i, c in enumerate(cs):
        w = scale if (x["e"] is not None and 5 <= i < x["e"] and rho > 0) else 1.0
        R, Ac = c["R"], c["A"]
        if B is not None:
            R = min(R, B)
        t = R + Ac
        if M is not None and t > M:
            t = M + (Ac + S.REGEN_REASON if Ac < M else M)
            extra = 1
        else:
            extra = 0
        tok += w * t
        secs += w * (A * (1 + extra) + Bk * t)
        ncalls += w * (1 + extra)
    if x["e"] is not None and x["e"] > 5 and rho > 0:   # re-edit penalty for wrongly early edits: p_bad*rho share, +3 avg calls
        avg_t = tok / max(ncalls, 1)
        tok += p_bad * 3 * rho * avg_t
        secs += p_bad * 3 * rho * (A + Bk * avg_t)
    return tok, secs

def boot_ratio(num, den, n=2000, seed=3):
    rnd = random.Random(seed); idx = range(len(num)); xs = []
    for _ in range(n):
        s = [rnd.randrange(len(num)) for _ in idx]
        xs.append(sum(num[i] for i in s) / sum(den[i] for i in s))
    xs.sort(); return xs[int(.025 * n)], xs[int(.975 * n)]

arms = {"control_v6": dict(), "v7-budget(B1536,M3072)": dict(B=1536, M=3072),
        "v7-decide rho=.15": dict(rho=.15), "v7-decide rho=.30": dict(rho=.30), "v7-decide rho=.45": dict(rho=.45),
        "v7-lean B1024,M2560 rho=.15": dict(B=1024, M=2560, rho=.15), "v7-lean B1024,M2560 rho=.30": dict(B=1024, M=2560, rho=.30),
        "v7-lean B1024,M2560 rho=.45": dict(B=1024, M=2560, rho=.45),
        "budget-only B1024,M2560": dict(B=1024, M=2560), "budget-only B768,M2304": dict(B=768, M=2304), "budget-only B512,M2048": dict(B=512, M=2048)}
base = [run_cost(x) for x in runs]
res = {}
for name, kw in arms.items():
    cs = [run_cost(x, **kw) for x in runs]
    tok_n = [c[0] for c in cs]; tok_d = [b[0] for b in base]
    sec_n = [c[1] for c in cs]; sec_d = [b[1] for b in base]
    res[name] = dict(tokens_per_run=st.mean(tok_n), tokens_ratio=sum(tok_n) / sum(tok_d), tokens_ratio_ci=boot_ratio(tok_n, tok_d),
                     llm_min_per_run=st.mean(sec_n) / 60, time_ratio=sum(sec_n) / sum(sec_d), time_ratio_ci=boot_ratio(sec_n, sec_d),
                     min_saved_per_run=(st.mean(sec_d) - st.mean(sec_n)) / 60)
json.dump(res, open(out, "w"), indent=1)
for k, v in res.items():
    print("%-34s tok/run %6.0f ratio %.3f [%.3f,%.3f]  llm-min %.2f ratio %.3f [%.3f,%.3f] saved %.2f min" % (k, v["tokens_per_run"], v["tokens_ratio"], *v["tokens_ratio_ci"], v["llm_min_per_run"], v["time_ratio"], *v["time_ratio_ci"], v["min_saved_per_run"]))
