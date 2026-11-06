#!/usr/bin/env python3
"""Run the body benchmark on the synthetic tasks (exact start state = base snapshot + bug patch, applied in place and restored).

  python run_synth.py <heldout|train> [--limit N] [--locate path] --out rows_synth_<split>.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "locate-improve"))
import bench  # noqa: E402
import body_bench as BB  # noqa: E402

S = "<home>/AppData/Local/Temp/claude/<proj>-Gemma-4-developer-agent-paper/ec71eb7d-f89d-4a42-b2fa-abd700099f6a/scratchpad"
W = "<home>/AppData/Local/Temp/lw"


def split_patch(patch):
    """{path: [hunk lines...]}: hunks per file of a unified diff"""
    out, cur = {}, None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            p = line[4:].split("\t")[0].strip()
            cur = p[2:] if p.startswith("b/") else p
            out[cur] = []
        elif cur is not None and (line.startswith("@@") or (out[cur] and line[:1] in " +-\\")) and not line.startswith("--- "):
            out[cur].append(line)
    return out


def apply_hunks(text, hunks):
    lines = text.split("\n")
    shift, i = 0, 0
    while i < len(hunks):
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", hunks[i])
        i += 1
        body = []
        while i < len(hunks) and not hunks[i].startswith("@@"):
            body.append(hunks[i])
            i += 1
        old = [l[1:] for l in body if l[:1] in " -"]
        new = [l[1:] for l in body if l[:1] in " +"]
        start = int(m.group(1)) - 1 + shift
        pos = None
        for off in (0, -1, 1, -2, 2, -3, 3, -5, 5, -10, 10, -30, 30):
            q = start + off
            if q >= 0 and lines[q:q + len(old)] == old:
                pos = q
                break
        if pos is None:
            raise ValueError("hunk does not apply")
        lines[pos:pos + len(old)] = new
        shift += len(new) - len(old)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--locate", default=str(HERE / "locate_body.py"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = Path(S) / "synth"
    tasks = bench.synth_tasks(ds, Path(W), a.split, a.limit)
    metas = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ds / "meta.jsonl", encoding="utf-8")}
    raw = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ds / "tasks.jsonl", encoding="utf-8")}
    L = bench.load_locate(a.locate, "")
    n_ok = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for i, t in enumerate(tasks):
            iid = t["iid"]
            root = Path(t["root"])
            pp = ds / "snapshots" / "patches" / ("%s.patch" % iid)
            saved = {}
            try:
                if pp.exists() and pp.stat().st_size:
                    for p, hunks in split_patch(pp.read_text(encoding="utf-8")).items():
                        if L.is_test_path(p) or not p.endswith(".py"):
                            continue
                        f = root / p
                        if not f.exists():
                            continue
                        raw_b = f.read_bytes()
                        saved[f] = raw_b
                        new = apply_hunks(raw_b.decode("utf-8", "replace"), hunks)
                        f.write_bytes(new.encode("utf-8"))
                R = L.load_repo(str(root))
                row = BB.run_task(L, R, {"iid": iid, "statement": t["statement"]}, raw[iid]["patch"])
                row.update({"kind": "synth", "split": a.split, "style": metas[iid]["style"], "terse": metas[iid]["style"] == "terse",
                            "repo": t["repo_key"]})
                if not row["gold_lines"]:
                    row["skipped"] = "no gold lines in existing non-test py file"
                fo.write(json.dumps(row, default=list) + "\n")
                n_ok += 1
            except Exception as e:
                print("ERR", iid, repr(e)[:200], file=sys.stderr)
            finally:
                for f, b in saved.items():
                    f.write_bytes(b)
            if (i + 1) % 25 == 0:
                print("..", i + 1, file=sys.stderr, flush=True)
    print("done", n_ok, "/", len(tasks))


if __name__ == "__main__":
    main()
