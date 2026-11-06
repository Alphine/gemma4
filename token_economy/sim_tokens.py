"""Counterfactual token-economy simulation on recorded ATIF traces (Modal pilot1 + lora-ab1 base arm, edit-early-v4).

Usage: python -I sim_tokens.py <trace_root> <per_run_table.jsonl> <out.json>
  trace_root has pilot1/<run>.json and ab1/<run>.json (ATIF traces fetched with `modal volume get`).

Per agent step (= one LLM call): completion_tokens (exact, from vLLM usage), reasoning chars (step.message),
tool-argument chars (json of tool_calls arguments).  Reasoning tokens are ESTIMATED by splitting the exact
completion_tokens by char share (RHO_R / RHO_A chars-per-token calibrated on llm_calls records, see CAL below).

Counterfactuals (same action sequence assumed, i.e. first-order effect only):
  budget B  : reasoning forced to end after B tokens (vLLM ThinkingTokenBudgetLogitsProcessor): R' = min(R, B)
  cap M     : call with total tokens > M is cut at M (finish_reason=length) -> harness nudge -> a fresh call
              that emits the tool call (A + REGEN_REASON tokens). Cost of a cut call = M tokens wasted +
              LAT_A seconds extra call overhead.
Time = LAT_A + B_TOK * tokens, Kaggle c1 (1.065, 0.02968) and Modal c8 (0.442, 0.02208).
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

LAT = {"kaggle": (1.065, 0.02968), "modal": (0.442, 0.02208)}
REGEN_REASON = 120      # tokens of reasoning in the nudged re-generation
CAP_NOW = 4096
RHO = (3.26, 3.39)        # chars/token (reasoning+content, tool args), least squares on 418 pilot1 llm_calls records (R2 0.96)


def load_calls(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for s in d["steps"]:
        if s.get("source") != "agent":
            continue
        m = s.get("metrics")
        if not m:
            continue
        tcs = s.get("tool_calls") or []
        args = sum(len(json.dumps(t.get("arguments", {}), ensure_ascii=False)) for t in tcs)
        msg = len(s.get("message") or "")
        ct = int(m["completion_tokens"])
        denom = msg / RHO[0] + args / RHO[1]
        r_share = (msg / RHO[0]) / denom if denom else 1.0
        R = ct * r_share
        out.append(dict(ct=ct, R=R, A=ct - R, names=[t["function_name"] for t in tcs], ntc=len(tcs),
                        capped=ct >= CAP_NOW - 8))
    return out


def sim_budget(calls, B):
    """returns (new_tokens, n_clipped_calls, n_clipped_runs_edit_calls)"""
    new = 0.0
    clip = 0
    for c in calls:
        R2 = min(c["R"], B)
        clip += c["R"] > B
        new += R2 + c["A"]
    return new, clip


def sim_cap(calls, M):
    new = 0.0
    cut = 0
    extra_calls = 0
    for c in calls:
        if c["ct"] > M:
            cut += 1
            extra_calls += 1
            new += M + ((c["A"] + REGEN_REASON) if c["A"] < M else M)
        else:
            new += c["ct"]
    return new, cut, extra_calls


def secs(tokens, ncalls, plat):
    a, b = LAT[plat]
    return a * ncalls + b * tokens


def boot(vals, fn=st.mean, n=2000, seed=1):
    rnd = random.Random(seed)
    xs = [fn([vals[rnd.randrange(len(vals))] for _ in vals]) for _ in range(n)]
    xs.sort()
    return xs[int(0.025 * n)], xs[int(0.975 * n)]


def main():
    root, tab, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    rows = [json.loads(l) for l in open(tab, encoding="utf-8")]
    runs = []
    for r in rows:
        if r["arm"] == "adapter":
            continue
        p = root / ("pilot1" if r["src"] == "pilot1" else "ab1") / f"{r['run']}.json"
        if not p.exists():
            continue
        calls = load_calls(p)
        if calls:
            runs.append(dict(r=r, calls=calls))
    res = {"n_runs": len(runs)}
    allc = [c for x in runs for c in x["calls"]]
    cts = sorted(c["ct"] for c in allc)
    res["calls"] = len(allc)
    res["ct_mean"] = st.mean(cts)
    res["ct_pctl"] = {q: cts[int(len(cts) * q / 100)] for q in (50, 75, 90, 95, 99)}
    res["calls_capped_4096"] = sum(c["capped"] for c in allc)
    res["capped_with_no_tool_call"] = sum(c["capped"] and c["ntc"] == 0 for c in allc)
    res["R_share"] = sum(c["R"] for c in allc) / sum(c["ct"] for c in allc)
    for B in (4096, 3072, 2048, 1536, 1024, 768, 512, 384, 256):
        res.setdefault("budget", {})[B] = {"calls_clipped": sum(c["R"] > B for c in allc),
                                           "frac_calls": sum(c["R"] > B for c in allc) / len(allc)}
    # per run tables
    def table(kind, params):
        tab = {}
        for p in params:
            sav_tok, sav_k, sav_m, n_trunc, runs_hit, res_hit = [], [], [], [], 0, 0
            for x in runs:
                cl = x["calls"]
                base_tok = sum(c["ct"] for c in cl)
                if kind == "budget":
                    new, nc = sim_budget(cl, p)
                    extra = 0
                    hit = nc
                else:
                    new, nc, extra = sim_cap(cl, p)
                    hit = nc
                n_new = len(cl) + extra
                sav_tok.append(base_tok - new)
                sav_k.append(secs(base_tok, len(cl), "kaggle") - secs(new, n_new, "kaggle"))
                sav_m.append(secs(base_tok, len(cl), "modal") - secs(new, n_new, "modal"))
                n_trunc.append(hit)
                runs_hit += hit > 0
                res_hit += hit > 0 and x["r"]["resolved"]
            base_k = [secs(sum(c["ct"] for c in x["calls"]), len(x["calls"]), "kaggle") for x in runs]
            tab[p] = dict(
                tok_saved_per_run=st.mean(sav_tok), tok_saved_ci=boot(sav_tok),
                tok_saved_pct=100 * sum(sav_tok) / sum(sum(c["ct"] for c in x["calls"]) for x in runs),
                min_saved_kaggle=st.mean(sav_k) / 60, min_saved_kaggle_ci=[v / 60 for v in boot(sav_k)],
                min_saved_modal=st.mean(sav_m) / 60,
                pct_llm_time_kaggle=100 * sum(sav_k) / sum(base_k),
                calls_hit_total=sum(n_trunc), runs_with_hit=runs_hit, resolved_runs_with_hit=res_hit,
                runs_with_hit_pct=100 * runs_hit / len(runs))
        return tab
    res["sim_budget"] = table("budget", [3072, 2048, 1536, 1024, 768, 512, 384])
    res["sim_cap"] = table("cap", [3072, 2048, 1500, 1024])
    # by source/baseline
    res["baseline"] = dict(
        tokens_per_run=st.mean(sum(c["ct"] for c in x["calls"]) for x in runs),
        calls_per_run=st.mean(len(x["calls"]) for x in runs),
        llm_min_kaggle=st.mean(secs(sum(c["ct"] for c in x["calls"]), len(x["calls"]), "kaggle") for x in runs) / 60,
        llm_min_modal=st.mean(secs(sum(c["ct"] for c in x["calls"]), len(x["calls"]), "modal") for x in runs) / 60)
    json.dump(res, open(out, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
