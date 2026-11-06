#!/usr/bin/env python3
"""Summarise trace_estimate.json (see trace_estimate.py).  python trace_summary.py trace_estimate.json per_run_table.jsonl out.md"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from analyze import ci, pct  # noqa: E402

LAT0, LAT1, CALL_TOK, CPT = 0.44, 0.0221, 55, 3.3
COMPACT = 14336


def main():
    rows = json.load(open(sys.argv[1]))
    tab = {r["run"]: r for r in (json.loads(l) for l in open(sys.argv[2], encoding="utf-8"))}
    for r in rows:
        t = tab.get(r["run"].replace(".json", ""))
        r["resolved"] = t["resolved"] if t else None
        r["timeout"] = t["timeout"] if t else None
        r["agent_min"] = t["agent_min"] if t else None
    out = []
    n = len(rows)
    out.append("Runs that called locate.py (ab1 base arm + pilot1 harvest, synthetic tasks): n=%d (%d tasks)." % (n, len({r["iid"] for r in rows})))
    out.append("")
    f = lambda k: [float(r[k]) for r in rows]
    out.append("| quantity | value (95%% CI over runs) |\n|---|---|")
    out.append("| printed code holds ALL gold lines (agent could edit without a read) | %s |" % pct(ci(f("body_hunk_all"))))
    out.append("| printed code holds ANY gold line | %s |" % pct(ci(f("body_hunk_any"))))
    out.append("| printed code does NOT hold the gold lines (wrong function/file) | %s |" % pct(ci([1 - x for x in f("body_hunk_any")])))
    out.append("| the agent's first read after locate lies >=70%% inside the printed code | %s |" % pct(ci(f("first_read_in_body"))))
    out.append("| the agent's first read after locate touches a gold line (v6-like behaviour: first read already right) | %s |" % pct(ci(f("first_read_hits_gold"))))
    both = [float(r["first_read_in_body"] and r["first_read_hits_gold"]) for r in rows]
    out.append("| first read hits gold AND is inside the printed code (a clean saved read) | %s |" % pct(ci(both)))
    wrong_same = [float(r["first_read_in_body"] and not r["first_read_hits_gold"]) for r in rows]
    out.append("| first read is inside the printed code but misses gold (agent reads this wrong code today as well) | %s |" % pct(ci(wrong_same)))
    out.append("| runs with a read before the first source edit | %s |" % pct(ci([float(r["reads_before_edit"] > 0) for r in rows])))
    out.append("")
    # savings
    sc = np.array([r["saved_reads"] for r in rows], dtype=float)
    lo = np.array([r["saved_reads"] * (LAT0 + LAT1 * CALL_TOK) for r in rows])
    hi = np.array([LAT0 * len(r["saved_ct"]) + LAT1 * sum(r["saved_ct"]) for r in rows])
    toks_hi = np.array([sum(r["saved_ct"]) for r in rows], dtype=float)
    out.append("Saved agent turns (reads before the first source edit whose lines are >=70%% inside the printed code), per run:")
    out.append("")
    out.append("| | mean per run (95%% CI) |\n|---|---|")
    out.append("| saved LLM calls | %.2f [%.2f, %.2f] |" % ci(sc))
    out.append("| saved seconds, lower bound (only the call itself: 0.44 s + 0.0221 s x %d tokens per saved turn) | %.1f [%.1f, %.1f] |" % ((CALL_TOK,) + ci(lo)))
    out.append("| saved seconds, upper bound (whole turn incl. its reasoning tokens: %.0f tokens per saved turn on average) | %.1f [%.1f, %.1f] |" % (toks_hi.sum() / max(1, sc.sum()), *ci(hi)))
    out.append("| share of runs with >=1 saved call | %s |" % pct(ci((sc > 0).astype(float))))
    mean_total = np.mean([r["total_steps_tokens"] for r in rows])
    out.append("")
    out.append("Context: mean completion tokens per run %.0f (= %.0f s of generation at 0.0221 s/token + %.0f s call overhead at %.0f calls)." % (
        mean_total, mean_total * LAT1, np.mean([r["n_steps"] for r in rows]) * LAT0, np.mean([r["n_steps"] for r in rows])))
    # prompt growth / compaction
    add = np.array([r["body_chars"] / CPT for r in rows])
    crossed0 = crossed1 = steps0 = steps1 = 0
    for r, a in zip(rows, add):
        pt = r["prompt_tokens"]
        li = r["locate_step"]
        b0 = [p for p in pt]
        b1 = [p + (a if i > li else 0) for i, p in enumerate(pt)]
        crossed0 += max(b0) > COMPACT if b0 else 0
        crossed1 += max(b1) > COMPACT if b1 else 0
        steps0 += sum(p > COMPACT for p in b0)
        steps1 += sum(p > COMPACT for p in b1)
    out.append("")
    out.append("Added prompt tokens per run (body chars / 3.3): mean %.0f, p90 %.0f. Compaction trigger %d prompt tokens: runs whose prompt ever exceeds it %d -> %d of %d (steps above it: %d -> %d of %d)." % (
        add.mean(), np.percentile(add, 90), COMPACT, crossed0, crossed1, n, steps0, steps1, sum(len(r["prompt_tokens"]) for r in rows)))
    # by terse/style
    out.append("")
    out.append("| style | n | all gold lines printed | saved calls/run |\n|---|---|---|---|")
    for s in sorted({r["style"] for r in rows}):
        sub = [r for r in rows if r["style"] == s]
        out.append("| %s | %d | %s | %.2f |" % (s, len(sub), pct(ci([float(r["body_hunk_all"]) for r in sub])), np.mean([r["saved_reads"] for r in sub])))
    # time-budget relevance
    out.append("")
    to = [r for r in rows if r["timeout"] is not None]
    if to:
        out.append("In these runs: timeout share %.0f%%, resolved %.0f%%, median agent minutes %.2f (budget 5.5 min ab1 / 10 min pilot1)." % (
            100 * np.mean([r["timeout"] for r in to]), 100 * np.mean([r["resolved"] for r in to]), np.median([r["agent_min"] for r in to])))
    open(sys.argv[3], "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
