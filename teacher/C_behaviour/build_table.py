"""Join results rows + ATIF traces + synthetic-task meta (gold files) into one per-run table (table.jsonl).

Inputs (all pre-fetched small files, see SUMMARY.md):  --root <dir> containing
  p1/results.jsonl        harvest/pilot1/results.jsonl
  T/pilot1/<task>__s0.json, T/lora-ab1/<task>__<arm><k>.json     ATIF traces
  lc/<task>__s0.json      per-LLM-call stats of pilot1 (optional; from the llm_calls recorder)
  syn/meta.jsonl          synthetic task meta (files = gold files)
  --ab results_slim.jsonl research/eval-lora/lora-ab1/results_slim.jsonl
"""
import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as sc  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    meta = {r["instance_id"]: r for r in load_jsonl(root / "syn/meta.jsonl")}

    def gold_of(iid):
        m = meta.get(iid) or {}
        f = m.get("files")
        if isinstance(f, str):
            f = ast.literal_eval(f)
        return list(f or [])

    out = []
    for r in load_jsonl(root / "p1/results.jsonl"):
        iid = r["instance_id"]
        tp = root / "T/pilot1" / f"{iid}__s{r['sample']}.json"
        out.append(dict(src="pilot1", arm="harvest", run=f"{iid}__s{r['sample']}", task=iid, budget_min=r["budget"]["max_time_minutes"],
                        resolved=bool(r["resolved"]), timeout=bool(r.get("agent_error") and "timeout" in str(r["agent_error"]).lower()),
                        agent_min=r["agent_min"], tool_calls=r["tool_calls"], llm_calls=r["llm_calls"], patch_chars=r["patch_chars"],
                        gold=gold_of(iid), trace=str(tp) if tp.exists() else None, lc=str(root / "lc" / f"{iid}__s{r['sample']}.json")))
    for r in load_jsonl(a.ab):
        iid = r["instance_id"]
        arm = "base" if r["arm"] == "base" else "adapter"
        tp = root / "T/lora-ab1" / f"{iid}__{arm}{r['sample']}.json"
        out.append(dict(src="ab1", arm=arm, run=f"{iid}__{arm}{r['sample']}", task=iid, budget_min=r["budget"]["max_time_minutes"],
                        resolved=bool(r["resolved"]), timeout=bool(r.get("agent_error") and "timeout" in str(r["agent_error"]).lower()),
                        agent_min=r["agent_min"], tool_calls=r["tool_calls"], llm_calls=r["llm_calls"], patch_chars=r["patch_chars"],
                        llm_latency_mean_s=r.get("llm_latency_mean_s"), all_malformed=r.get("all_malformed"),
                        inflight=r.get("inflight_at_start"), gold=gold_of(iid), trace=str(tp) if tp.exists() else None))
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for row in out:
            if row["trace"]:
                tr = sc.from_atif(json.load(open(row["trace"], encoding="utf-8")))
                row["m"] = sc.metrics(tr, row["gold"])
                n += 1
            f.write(json.dumps(row) + "\n")
    print(len(out), "rows;", n, "with traces")


if __name__ == "__main__":
    main()
