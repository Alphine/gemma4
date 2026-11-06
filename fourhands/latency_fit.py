"""Fit per-call LLM latency d = a + b*prompt_tokens + c*completion_tokens from the A/B session-1 HTTP call logs
(research/top10/ab; 40 runs, 3 arms, lb-eval launch lbeval-20261007-043225, Kaggle 4xL4 TP=4).
Usage: python -I latency_fit.py <calllogs_root> <out.json>"""
import json, glob, sys, os
import numpy as np
root, out = sys.argv[1], sys.argv[2]
X, y, gaps, per_run = [], [], [], []
for f in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    es = [e for e in d["entries"] if str(e.get("status")) == "200" and e.get("ct") is not None]
    prev_end = None
    run = dict(arm=d["arm"], task=d["task"], n=len(es), llm_s=0.0, gap_s=0.0, ct=0, pt=0)
    for e in d["entries"]:
        if prev_end is not None:
            g = float(e["t_send"]) - prev_end
            if 0 <= g < 120:
                gaps.append(g); run["gap_s"] += g
        prev_end = float(e["t_end"])
        if str(e.get("status")) == "200" and e.get("ct") is not None:
            dur = float(e["t_end"]) - float(e["t_send"])
            X.append([1.0, float(e["pt"]), float(e["ct"])]); y.append(dur)
            run["llm_s"] += dur; run["ct"] += int(e["ct"]); run["pt"] += int(e["pt"])
    per_run.append(run)
X, y = np.array(X), np.array(y)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta
r2 = 1 - res.var() / y.var()
# decode-only fit on small prompts as sanity
o = dict(n_calls=len(y), a_s=beta[0], b_s_per_prompt_tok=beta[1], c_s_per_comp_tok=beta[2], r2=r2,
         decode_tok_per_s=1 / beta[2], prefill_tok_per_s_if_uncached=1 / beta[1] if beta[1] > 0 else None,
         mean_gap_s=float(np.mean(gaps)), median_gap_s=float(np.median(gaps)), n_gaps=len(gaps),
         mean_pt=float(X[:, 1].mean()), mean_ct=float(X[:, 2].mean()), mean_dur=float(y.mean()),
         share_decode=float((beta[2] * X[:, 2]).sum() / y.sum()),
         share_prefill=float((beta[1] * X[:, 1]).sum() / y.sum()), share_const=float(beta[0] * len(y) / y.sum()),
         run_totals=dict(mean_llm_s=float(np.mean([r["llm_s"] for r in per_run])), mean_gap_s=float(np.mean([r["gap_s"] for r in per_run]))))
json.dump(o, open(out, "w"), indent=1)
print(json.dumps(o, indent=1))
