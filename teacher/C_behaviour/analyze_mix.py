"""Where do the seconds go?  Per-run call-kind mix and estimated generation seconds per kind, by outcome group.

python analyze_mix.py --root <scratch C dir> --out mix_results.json
Seconds per step are the fitted generation model (0.442 + 0.02208 * completion_tokens) -- tool time is ~3% of a run.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402

KINDS = ["skill", "read", "shell_read", "shell", "repro", "test", "edit", "scratch", "submit", "status", "none"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.root) / "table.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r.get("m")]
    ab = [r for r in rows if r["src"] == "ab1"]          # real 5.5-min budget only
    res = [r for r in ab if r["resolved"] and not r["timeout"]]
    q1, q3 = np.percentile([r["agent_min"] for r in res], [25, 75])

    def grp(r):
        if r["timeout"]:
            return "timeout_never_edited" if not r["m"]["edited"] else "timeout_edited"
        if r["resolved"]:
            return "resolved_fast_Q1" if r["agent_min"] <= q1 else ("resolved_slow_Q4" if r["agent_min"] >= q3 else "resolved_mid")
        return "failed_submitted"

    acc = defaultdict(lambda: dict(n=0, calls=defaultdict(float), sec=defaultdict(float), pre_edit_sec=0.0, post_edit_sec=0.0,
                                   big_turn_sec=0.0, tot_sec=0.0, think_only_sec=0.0))
    for r in ab:
        g = grp(r)
        tr = sc.from_atif(json.load(open(r["trace"], encoding="utf-8")))
        A = acc[g]
        A["n"] += 1
        edited = False
        for st in tr.steps:
            sec = 0.442 + 0.02208 * (st.completion_tokens or 0)
            kinds = [sc.call_kind(c) for c in st.calls] or ["none"]
            kind = kinds[0]
            A["calls"][kind] += len(st.calls) if st.calls else 1
            A["sec"][kind] += sec
            A["tot_sec"] += sec
            if (st.completion_tokens or 0) > 1000:
                A["big_turn_sec"] += sec
            if edited:
                A["post_edit_sec"] += sec
            else:
                A["pre_edit_sec"] += sec
            if any(k == "edit" and not c.error_type for k, c in zip(kinds, st.calls)):
                edited = True
    out = {}
    for g, A in acc.items():
        n = A["n"]
        out[g] = dict(n=n, est_gen_sec_per_run=A["tot_sec"] / n, pre_first_good_edit_share=A["pre_edit_sec"] / A["tot_sec"],
                      big_turn_share=A["big_turn_sec"] / A["tot_sec"],
                      calls_per_run={k: A["calls"][k] / n for k in KINDS if A["calls"][k]},
                      sec_share_by_kind={k: A["sec"][k] / A["tot_sec"] for k in KINDS if A["sec"][k]})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    for g, v in out.items():
        print(g, "n", v["n"], "gen_s", round(v["est_gen_sec_per_run"]), "pre-edit share", round(v["pre_first_good_edit_share"], 2),
              "big-turn share", round(v["big_turn_share"], 2))
        print("   calls/run", {k: round(x, 1) for k, x in v["calls_per_run"].items()})
        print("   sec share", {k: round(x, 2) for k, x in v["sec_share_by_kind"].items()})


if __name__ == "__main__":
    main()
