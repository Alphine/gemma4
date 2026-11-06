"""Shared loaders for the anatomy analysis (runs.json from build_runs.py, ckpt rows, dev gold info)."""
import sys as _s; _s.stdout.reconfigure(encoding="utf-8", errors="replace")
import json
import os
import re
import statistics as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARMS = ["edit-early-v6-locate", "edit-early-v7-decide-nt", "edit-early-v7-lean-nt"]
SHORT = {"edit-early-v6-locate": "v6", "edit-early-v7-decide-nt": "decide", "edit-early-v7-lean-nt": "lean"}


def load(runs_json):
    R = json.load(open(runs_json))
    rows = {}
    for a in ARMS:
        for l in open(f"{ROOT}/data/ckpt_scrubbed/results_{a}.jsonl"):
            x = json.loads(l)
            rows[(a, x["instance_id"])] = x
    gold = {}
    for l in open(f"{ROOT}/research/top10/locate-body/rows_dev.jsonl"):
        x = json.loads(l)
        gold[x["iid"]] = x
    for r in R:
        r["row"] = rows[(r["arm"], r["task"])]
        r["gold"] = gold.get(r["task"])
        r["wall"] = r["row"]["agent_min"] * 60
        r["prim"] = [c for c in r["calls"] if not c["compact"] and c.get("step_id") is not None]
        r["cancelled"] = [c for c in r["calls"] if c.get("exc")]
        r["compact"] = [c for c in r["calls"] if c["compact"]]
    return R


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def first_edit_index(prim, successful=False):
    """index into prim of the first edit-kind tool call (attempt, or first one whose observation did not fail)"""
    for k, c in enumerate(prim):
        for tc in c.get("tool_calls", []):
            if tc["kind"] == "edit" and (not successful or not c.get("fail2")):
                return k
    return None


def tc0(c):
    t = c.get("tool_calls") or []
    return t[0] if t else None
