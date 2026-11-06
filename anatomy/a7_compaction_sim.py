#!/usr/bin/env python3
"""Replay each run's call sequence and count how often the 14,336-token compaction threshold would be crossed under token-saving scenarios.
Model: prompt_t = base + sum(previous thoughts (capped) + tool-arg tokens + obs_chars*0.351 (capped)); after a compaction the prompt resets to
RESET (7,381 observed mean) .  Trajectory is held fixed (no behavioural response). usage: python a7_compaction_sim.py runs.json out.json"""
import json, sys
from common import *
R = load(sys.argv[1])
THR, RESET, OBS = 14336, 7381, 0.351
def sim(rsn_cap=None, obs_cap=None, base_cut=0):
    n = 0; per = []
    for r in R:
        p = 5386 - base_cut; k = 0
        for c in r["prim"]:
            rs = c["rsn_tok"] if rsn_cap is None else min(c["rsn_tok"], rsn_cap)
            ta = max(c["ct"] - c["rsn_tok"], 0)
            ob = c.get("obs_len") or 0
            if obs_cap is not None: ob = min(ob, obs_cap)
            p += rs + ta + OBS * ob
            if p > THR:
                k += 1; p = RESET - base_cut
        per.append(k)
    return per
out = {}
for name, kw in [("observed-model", {}), ("thinking<=1024", dict(rsn_cap=1024)), ("thinking<=512", dict(rsn_cap=512)),
                 ("tool output<=2000 chars", dict(obs_cap=2000)), ("tool output<=1200 chars", dict(obs_cap=1200)),
                 ("thinking<=512 + output<=2000", dict(rsn_cap=512, obs_cap=2000)), ("base prompt -1000 tok", dict(base_cut=1000)),
                 ("thinking<=512 + output<=1200 + base-1000", dict(rsn_cap=512, obs_cap=1200, base_cut=1000))]:
    per = sim(**kw)
    out[name] = dict(mean_compactions=mean(per), runs_with_any=sum(1 for x in per if x > 0))
    print("%-42s compactions/run %.2f  runs with >=1: %d/40  -> ~%.0f s/run at 22 s each" % (name, mean(per), sum(1 for x in per if x > 0), 22 * mean(per)))
json.dump(out, open(sys.argv[2], "w"), indent=1)
