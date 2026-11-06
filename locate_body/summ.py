import json, statistics as st, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
rows=[r for r in rows if not r.get("skipped")]
print(len(rows))
for v in rows[0]["variants"]:
    x=[r["variants"][v] for r in rows]
    print("%-7s"%v, "hunk_all %.3f any %.3f funcshown %.3f r1gold %.3f chars %.0f capped %.2f lines %.1f"%(*[st.mean(float(a[k]) for a in x) for k in ["hunk_all","hunk_any","func_shown","r1_gold_func","chars"]],st.mean(a["chars"]>=4750 for a in x),st.mean(a["lines"] for a in x)))
