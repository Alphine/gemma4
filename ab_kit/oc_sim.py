"""Operating characteristics of the pre-registered decision rule on synthetic data (run once, before data exists).

    python oc_sim.py --n-sim 100 --out oc_results.json

Synthetic world (synth.py): per-call tokens lognormal + 4% long thoughts, Kaggle latency model, hard time cap, first edit
after k calls.  'realistic cap' scenarios add 14 calls to every arm and cap runs at 330 s so that ~40-50% of control runs
time out, as in the real lb-eval v4 data (24/43 timeouts).  Reports, per scenario, the share of simulated experiments in which
each gate component and the final decision fire.  It characterises the RULE, not the real arms.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_ab as A  # noqa: E402
import synth  # noqa: E402
import trace_metrics as tm  # noqa: E402

COMMON = {"extra_calls": 14}
SCEN = {
    "null_uncapped": dict(effects={}, cap_s=1e9),
    "null_realistic_cap": dict(effects={r: dict(COMMON) for r in synth.NAMES}, cap_s=330.0),
    "lean_tok-30%_edit-3_uncapped": dict(effects={"lean": dict(tok_factor=0.7, edit_delta=-3), "decide": dict(edit_delta=-2)}, cap_s=1e9),
    "lean_tok-30%_edit-3_realistic_cap": dict(effects={"lean": dict(COMMON, tok_factor=0.7, edit_delta=-3),
                                                       "decide": dict(COMMON, edit_delta=-2), "control": dict(COMMON)}, cap_s=330.0),
    "lean_tok-15%_edit-2_realistic_cap": dict(effects={"lean": dict(COMMON, tok_factor=0.85, edit_delta=-2),
                                                       "decide": dict(COMMON, edit_delta=-1), "control": dict(COMMON)}, cap_s=330.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", default=str(HERE / "oc_results.json"))
    a = ap.parse_args()
    res = {}
    for name, sc in SCEN.items():
        cnt = dict(n=0, lean_tokens=0, lean_mech=0, lean_resolved_guard=0, lean_timeout_guard=0, lean_all_but_preflight=0,
                   submit_lean=0, submit_decide=0, submit_any=0, ctrl_timeout_rate=0.0, tokens_rel_lean=0.0, calls_rel_lean=0.0,
                   flagged_order=0)
        for i in range(a.n_sim):
            rows, tr = synth.simulate(43, seed=7000 + i, effects=sc["effects"], cap_s=sc["cap_s"], order_mode="rotate", resolve_base=0.35)
            cfg = dict(A.PRE, seed=i, n_boot=a.n_boot)
            runs, st = A.build_runs(rows, ["control", "decide", "lean"], A.DEFAULT_ARMS,
                                    {k: tm.parse_trace(v) for k, v in tr.items()}, None, None, cfg, [])
            o = A.sanitize(A.analyze_runs(runs, st, ["control", "decide", "lean"], cfg, preflight="pass", planned=43))
            g = o["gates"]["lean"]["checks"]
            cnt["n"] += 1
            cnt["lean_tokens"] += g["tokens"]
            cnt["lean_mech"] += g["mechanism"]
            cnt["lean_resolved_guard"] += g["resolved_guard"]
            cnt["lean_timeout_guard"] += g["timeout_not_higher"]
            cnt["lean_all_but_preflight"] += o["gates"]["lean"]["passed_all"]
            d = o["decision"]["decision"]
            cnt["submit_lean"] += d == "SUBMIT lean-nt"
            cnt["submit_decide"] += d == "SUBMIT decide-nt"
            cnt["submit_any"] += d in ("SUBMIT lean-nt", "SUBMIT decide-nt")
            cnt["ctrl_timeout_rate"] += o["arm_summary"]["control"]["timeouts"] / 43
            cnt["tokens_rel_lean"] += o["contrasts"]["lean_vs_control"]["tokens"]["rel"] or 0
            cnt["calls_rel_lean"] += o["contrasts"]["lean_vs_control"]["llm_calls"]["rel"] or 0
            cnt["flagged_order"] += bool(o["order"]["lean_vs_control"]["tokens"].get("flag"))
        n = cnt.pop("n")
        res[name] = {k: round(v / n, 3) for k, v in cnt.items()}
        res[name]["n_sim"] = n
        print(name, res[name], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
