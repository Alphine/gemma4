"""Validate synthetic tasks with the official swegemma verify_task in the grader-image emulation.

Per task: gold patch -> must resolve; empty patch -> must not resolve; gold verification must finish in
MAX_VERIFY_S. A sample also gets an "agent view" check (the agent's sandbox state: one baseline commit,
clean tree, hidden tests absent). Survivors are split (held-out repos + held-out functions ~15%) and
written as the final dataset folder /kaggle/working/dataset (same layout as the competition data:
tasks.jsonl + snapshots/; plus wheels/, meta.jsonl, splits.json, manifest.json).

Input: the factory's candidates/ folder (kernel_sources) or an existing gemma4-synth-tasks dataset (re-check).
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import random
import re
import shutil
import statistics
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from .common import (SCRATCH, WORK, Sandbox, build_merged_data_dir, comp_data_dir, find_all, install_harness,
                     lb_module, log, make_cfg, minutes)
from .patches import find_test_def, parse_nodeid

DEFAULTS = {
    "SOURCE": "candidates",       # candidates (factory output) | dataset (re-validate a published dataset)
    "ONLY": None,                 # list of instance ids, or None
    "LIMIT": None,                # max tasks (smoke)
    "CONCURRENCY": 4,
    "TASK_TIMEOUT_S": 600,
    "MAX_VERIFY_S": 60.0,
    "AGENT_VIEW_SAMPLE": 0.08,
    "HELDOUT_FRACTION": 0.15,
    "HELDOUT_REPOS": 2,
    "SESSION_MINUTES": 690,
    "SEED": 20261006,
    "WRITE_DATASET": True,
}


def find_source(kind):
    if kind == "dataset":
        hits = [h for h in find_all("tasks.jsonl", prefer="gemma4-synth-tasks") if "gemma4-synth-tasks" in str(h)]
    else:
        hits = [h for h in find_all("candidates/tasks.jsonl")]
    if not hits:
        raise FileNotFoundError(f"no {kind} tasks.jsonl in /kaggle/input")
    return hits[0].parent


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


async def verify(mgr, cfg, task, snaps, patch, timeout):
    from swegemma.harness.verification import verify_task
    t = time.time()
    try:
        r = await asyncio.wait_for(
            verify_task(mgr, cfg, task, snaps[0], base_snapshot_path=snaps[1], patch_path=snaps[2],
                        agent_patch=patch, start_time=time.perf_counter()), timeout=timeout)
        d = {"resolved": bool(r.resolved), "error": r.error, "exit": r.test_exit_code,
             "out_tail": (r.test_output or "")[-1500:]}
    except asyncio.TimeoutError:
        d = {"resolved": False, "error": f"TIMEOUT {timeout}s", "exit": None, "out_tail": ""}
    except Exception:
        d = {"resolved": False, "error": "EXCEPTION " + traceback.format_exc()[-1200:], "exit": None, "out_tail": ""}
    d["s"] = round(time.time() - t, 1)
    m = re.search(r"(\d+) passed", d["out_tail"])
    d["n_passed"] = int(m.group(1)) if m else None
    m = re.search(r"(\d+) failed", d["out_tail"])
    d["n_failed"] = int(m.group(1)) if m else None
    return d


def agent_view(mgr, cfg, task, snaps, meta):
    """State an agent would start from: extract + patch + baseline commit, like the harness's Phase 1."""
    sb = Sandbox(mgr, cfg, task.repo, snaps[0], base=snaps[1], patch=snaps[2], agent=True)
    try:
        out = {}
        r = sb.exec("cd /workspace && git log --oneline | wc -l && git status --porcelain | wc -l")
        nums = (r.stdout or "").split()
        out["commits"] = int(nums[0]) if nums else None
        out["dirty"] = int(nums[1]) if len(nums) > 1 else None
        hidden_present = []
        for nid in meta.get("removed_tests", []):
            path, classes, name, _ = parse_nodeid(nid)
            p = sb.ws / path
            try:
                fn, _ = find_test_def(ast.parse(p.read_text(encoding="utf-8")), classes, name)
            except Exception:
                fn = None
            if fn is not None:
                hidden_present.append(nid)
        out["hidden_tests_present"] = hidden_present
        # the bug is in place: the gold patch must apply forward
        gp = sb.ws.parent / "tmp" / "gold_check.patch"
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_text(task.patch, encoding="utf-8")
        r = sb.exec(f"cd /workspace && git apply --check {gp}")
        out["gold_applies"] = r.exit_code == 0
        out["ok"] = out["commits"] == 1 and out["dirty"] == 0 and not hidden_present and out["gold_applies"]
        return out
    finally:
        sb.stop()


def split(rows, metas, C):
    """Held-out = all tasks of HELDOUT_REPOS mid-sized repos + tasks touching held-out functions elsewhere."""
    rng = random.Random(f"split:{C['SEED']}")
    by_repo = defaultdict(list)
    for r in rows:
        by_repo[metas[r]["repo"]].append(r)
    total = len(rows)
    repos = sorted(by_repo, key=lambda k: (len(by_repo[k]), k))
    target = C["HELDOUT_FRACTION"] * total
    # held-out repos: from the middle of the size distribution, together <= ~half of the held-out budget
    mid = [r for r in repos if len(by_repo[r]) <= 0.5 * target / max(1, C["HELDOUT_REPOS"]) * 2]
    rng.shuffle(mid)
    held_repos = []
    for r in mid:
        if len(held_repos) >= C["HELDOUT_REPOS"]:
            break
        if sum(len(by_repo[x]) for x in held_repos) + len(by_repo[r]) <= 0.6 * target:
            held_repos.append(r)
    held = {t for r in held_repos for t in by_repo[r]}
    funcs = sorted({f for t in rows if t not in held for f in metas[t]["funcs"]})
    rng.shuffle(funcs)
    func_tasks = defaultdict(set)
    for t in rows:
        for f in metas[t]["funcs"]:
            func_tasks[f].add(t)
    held_funcs = []
    for f in funcs:
        if len(held) >= target:
            break
        if metas[next(iter(func_tasks[f]))]["repo"] in held_repos:
            continue
        held_funcs.append(f)
        held |= func_tasks[f]
    train = sorted(t for t in rows if t not in held)
    return {"heldout": sorted(held), "train": train, "heldout_repos": sorted(held_repos),
            "heldout_functions": sorted(held_funcs)}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


async def amain(C):
    src = find_source(C["SOURCE"])
    log(f"source: {src}")
    install_harness()
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.models import load_tasks
    tasks_all = load_tasks(src / "tasks.jsonl")
    metas = {m["instance_id"]: m for m in load_jsonl(src / "meta.jsonl")} if (src / "meta.jsonl").exists() else {}
    raw = {r["instance_id"]: r for r in load_jsonl(src / "tasks.jsonl")}
    comp = comp_data_dir()
    data_dir = build_merged_data_dir(SCRATCH / "vdata", comp, src / "tasks.jsonl", src / "snapshots", src / "wheels")
    LB = lb_module()
    LB.setup_emulation(data_dir)
    from .common import make_manager
    mgr = make_manager(LB)
    cfg = make_cfg(data_dir, SCRATCH / "vresults")
    tasks = [t for t in tasks_all if C["ONLY"] is None or t.instance_id in C["ONLY"]]
    # interleave repos so a partial run covers all
    groups = defaultdict(list)
    for t in tasks:
        groups[t.repo].append(t)
    order = []
    while any(groups.values()):
        for k in sorted(groups):
            if groups[k]:
                order.append(groups[k].pop(0))
    if C["LIMIT"]:
        order = order[:C["LIMIT"]]
    rng = random.Random(f"agentview:{C['SEED']}")
    av_ids = {t.instance_id for t in order if rng.random() < C["AGENT_VIEW_SAMPLE"]}
    seen_repo = set()
    for t in order:   # at least one agent-view check per repo
        if t.repo not in seen_repo:
            av_ids.add(t.instance_id)
            seen_repo.add(t.repo)
    log(f"{len(order)} tasks to validate (agent-view sample {len(av_ids)}), concurrency {C['CONCURRENCY']}")
    rows_path = WORK / "validate_rows.jsonl"
    rows = {}
    sem = asyncio.Semaphore(C["CONCURRENCY"])
    lock = asyncio.Lock()

    async def one(task):
        async with sem:
            if minutes() > C["SESSION_MINUTES"]:
                return
            snaps = resolve_task_snapshot_paths(cfg.snapshots_dir, task.instance_id, task.repo)
            gold = await verify(mgr, cfg, task, snaps, task.patch, C["TASK_TIMEOUT_S"])
            empty = await verify(mgr, cfg, task, snaps, "", C["TASK_TIMEOUT_S"])
            row = {"instance_id": task.instance_id, "repo": task.repo, "gold": gold, "empty": empty}
            if task.instance_id in av_ids:
                try:
                    row["agent_view"] = await asyncio.to_thread(agent_view, mgr, cfg, task, snaps,
                                                                metas.get(task.instance_id, {}))
                except Exception:
                    row["agent_view"] = {"ok": False, "error": traceback.format_exc()[-800:]}
            reasons = []
            if not gold["resolved"]:
                reasons.append("gold_not_resolved")
            if empty["resolved"]:
                reasons.append("empty_resolved")
            if gold["s"] > C["MAX_VERIFY_S"]:
                reasons.append("verify_too_slow")
            if row.get("agent_view") and not row["agent_view"].get("ok"):
                reasons.append("agent_view_failed")
            row["valid"] = not reasons
            row["reasons"] = reasons
            async with lock:
                rows[task.instance_id] = row
                with open(rows_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                if len(rows) % 20 == 0 or not row["valid"]:
                    nv = sum(r["valid"] for r in rows.values())
                    log(f"{len(rows)}/{len(order)} valid={nv} last={task.instance_id} gold={gold['resolved']}"
                        f"({gold['s']}s) empty={empty['resolved']} {reasons} "
                        + ("" if row["valid"] else (gold.get("error") or "")[:200].replace("\n", " ")))

    await asyncio.gather(*(one(t) for t in order))
    log(f"verified {len(rows)}: valid {sum(r['valid'] for r in rows.values())}")
    finish(C, src, raw, metas, rows, len(tasks_all))


def finish(C, src, raw, metas, rows, n_in):
    valid = sorted(i for i, r in rows.items() if r["valid"])
    sp = split(valid, metas, C) if metas else {"train": valid, "heldout": []}
    gold_s = [r["gold"]["s"] for r in rows.values() if r["gold"]["resolved"]]
    empty_s = [r["empty"]["s"] for r in rows.values()]
    reasons = Counter(x for r in rows.values() for x in r["reasons"])
    av = [r["agent_view"] for r in rows.values() if r.get("agent_view")]
    vm = [metas[i] for i in valid if i in metas]
    by_repo = defaultdict(Counter)
    for i, r in rows.items():
        repo = r["repo"]
        by_repo[repo]["candidates"] += 1
        by_repo[repo]["valid"] += r["valid"]
    held = set(sp["heldout"])
    for i in valid:
        by_repo[raw[i]["repo"]]["heldout" if i in held else "train"] += 1
    repos_info = {}
    rj = src / "repos.json"
    if rj.exists():
        rinfo = json.loads(rj.read_text())
        for repo, d in rinfo.items():
            if repo in by_repo:
                repos_info[repo] = {"licence": d.get("licence"), "tag": d.get("tag"), "commit": d.get("commit"),
                                    "commit_date": d.get("commit_date"), "packages": d.get("packages"),
                                    "snapshot_bytes": d.get("snapshot_bytes"), **dict(by_repo[repo])}
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_candidates_in": n_in, "n_verified": len(rows),
        "n_valid": len(valid), "n_train": len(sp["train"]), "n_heldout": len(sp["heldout"]),
        "heldout_repos": sp.get("heldout_repos"), "n_heldout_functions": len(sp.get("heldout_functions", [])),
        "repos": repos_info,
        "mutation_ops": dict(Counter(op for m in vm for op in m["ops"]).most_common()),
        "kinds": dict(Counter(m["kind"] for m in vm)),
        "pair_same_file": sum(1 for m in vm if m["kind"] == "pair" and len(m["files"]) == 1),
        "f2p_counts": dict(sorted(Counter(m["n_f2p"] for m in vm).items())),
        "styles": dict(Counter(m["style"] for m in vm)),
        "style_share": {k: round(v / max(1, len(vm)), 3) for k, v in Counter(m["style"] for m in vm).items()},
        "fail_exc_types": dict(Counter(e for m in vm for e in m["fail_exc_types"]).most_common(12)),
        "gold_lines_changed": dict(sorted(Counter(m["gold_lines_changed"] for m in vm).items())),
        "validation": {
            "gold_resolved": sum(r["gold"]["resolved"] for r in rows.values()),
            "empty_resolved": sum(r["empty"]["resolved"] for r in rows.values()),
            "drop_reasons": dict(reasons),
            "gold_verify_s": {"p50": pct(gold_s, .5), "p90": pct(gold_s, .9), "max": max(gold_s) if gold_s else None},
            "empty_verify_s": {"p50": pct(empty_s, .5), "p90": pct(empty_s, .9),
                               "max": max(empty_s) if empty_s else None},
            "agent_view_checked": len(av), "agent_view_ok": sum(1 for a in av if a.get("ok")),
            "concurrency": C["CONCURRENCY"],
        },
        "config": C,
    }
    (WORK / "validate_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (WORK / "splits.json").write_text(json.dumps(sp, indent=1))
    log(f"manifest: valid {len(valid)} train {len(sp['train'])} heldout {len(sp['heldout'])} reasons {dict(reasons)}")
    if not C["WRITE_DATASET"]:
        return
    ds = WORK / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "snapshots" / "base_snapshots").mkdir(parents=True)
    (ds / "snapshots" / "patches").mkdir(parents=True)
    with open(ds / "tasks.jsonl", "w", encoding="utf-8") as f:
        for i in valid:
            f.write(json.dumps(raw[i]) + "\n")
    with open(ds / "meta.jsonl", "w", encoding="utf-8") as f:
        for i in valid:
            if i in metas:
                f.write(json.dumps(dict(metas[i], split="heldout" if i in held else "train",
                                        verify_gold_s=rows[i]["gold"]["s"])) + "\n")
    from swegemma.deduplication import get_base_snapshot_filename
    for repo in sorted({raw[i]["repo"] for i in valid}):
        n = get_base_snapshot_filename(repo)
        shutil.copy2(src / "snapshots" / "base_snapshots" / n, ds / "snapshots" / "base_snapshots" / n)
    for i in valid:
        shutil.copy2(src / "snapshots" / "patches" / f"{i}.patch", ds / "snapshots" / "patches" / f"{i}.patch")
    shutil.copytree(src / "wheels", ds / "wheels")
    if rj.exists():
        shutil.copy2(rj, ds / "repos.json")
    (ds / "splits.json").write_text(json.dumps(sp, indent=1))
    size = sum(p.stat().st_size for p in ds.rglob("*") if p.is_file())
    manifest["dataset_bytes"] = size
    manifest["wheels"] = {"files": sorted(p.name for p in (ds / "wheels").glob("*.whl")),
                          "bytes": sum(p.stat().st_size for p in (ds / "wheels").glob("*.whl"))}
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    (WORK / "validate_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log(f"dataset written: {len(valid)} tasks, {size / 1e6:.1f} MB")


def main(CONFIG):
    C = dict(DEFAULTS)
    C.update(CONFIG or {})
    SCRATCH.mkdir(parents=True, exist_ok=True)
    asyncio.run(amain(C))
