#!/usr/bin/env python3
"""Build the arm edit-early-v8-body = edit-early-v7-decide-nt + (a) locate_body.py as skills/swe-tools/scripts/locate.py,
(b) the SKILL.md locate row, (c) a minimal prompt edit.  NOT run by default (report recommends NO-GO); run it for a GO:

    python research/top10/locate-body/make_arm.py [--dest datasets/gemma4-submissions/edit-early-v8-body]

All other files stay byte-identical to edit-early-v7-decide-nt (eval_config.yaml is byte-identical to edit-early-v6-locate).
Checks at the end: skill name is 9 ASCII chars, no curly braces in system.md, no non-ASCII added, eval_config unchanged.
"""
import argparse
import filecmp
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = REPO / "datasets" / "gemma4-submissions"
SRC = ARMS / "edit-early-v7-decide-nt"
CAND = Path(__file__).parent / "locate_body.py"

OLD_ROW = "likely files and their test files |"
NEW_ROW = ("likely files and their test files, and last the code of the best hit(s) with line numbers: `# code of path:A-B (rank 1, name)` then "
           "`  123: code` lines (use these numbers with edit_lines.py; a long definition shows its head and the lines that match the issue, "
           "`... lines X-Y not shown` marks the rest) |")

OLD_TOOL = ("Prints the best-matching definitions `path:line kind Class.name` and the test files of the likely files.")
NEW_TOOL = ("Prints the best-matching definitions `path:line kind Class.name`, the test files of the likely files, and at the end the "
            "CODE of the best hit(s) with line numbers (`# code of path:A-B (rank 1, name)`).")

OLD_CALL1 = "- Call 1: `scripts/locate.py`. Start from its best hit that fits the issue and read it with `sed -n`."
NEW_CALL1 = ("- Call 1: `scripts/locate.py`. It prints the code of its best hit(s) with line numbers: if the buggy line is in that printed code, "
             "edit it from there with `scripts/edit_lines.py` using those exact line numbers (no `sed -n` first). Only if the buggy line is "
             "not in the printed range, `sed -n 'A,Bp'` the hit that fits the issue (or the lines marked `not shown`).")

OLD_BUDGET = "after locate.py you may read AT MOST 2 code ranges (the best hit, then one more only if the buggy line is not visible;"
NEW_BUDGET = "after locate.py you may read AT MOST 2 more code ranges (the code printed by locate.py does not count; read one more only if the buggy line is not visible;"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ARMS / "edit-early-v8-body"))
    a = ap.parse_args()
    dest = Path(a.dest)
    if dest.exists():
        raise SystemExit("destination exists: %s" % dest)
    shutil.copytree(SRC, dest)
    shutil.copyfile(CAND, dest / "skills/swe-tools/scripts/locate.py")
    sk = dest / "skills/swe-tools/SKILL.md"
    t = sk.read_text(encoding="utf-8")
    assert t.count(OLD_ROW) == 1
    sk.write_text(t.replace(OLD_ROW, NEW_ROW), encoding="utf-8", newline="\n")
    sp = dest / "prompts/system.md"
    t = sp.read_text(encoding="utf-8")
    for old, new in ((OLD_TOOL, NEW_TOOL), (OLD_CALL1, NEW_CALL1), (OLD_BUDGET, NEW_BUDGET)):
        assert t.count(old) == 1, old
        t = t.replace(old, new)
    sp.write_text(t, encoding="utf-8", newline="\n")
    # ---- checks
    assert "{" not in t and "}" not in t, "curly braces in system.md"
    assert all(ord(c) < 128 for c in NEW_ROW + NEW_TOOL + NEW_CALL1 + NEW_BUDGET), "non-ASCII added"
    assert filecmp.cmp(dest / "eval_config.yaml", ARMS / "edit-early-v6-locate" / "eval_config.yaml", shallow=False)
    assert "swe-tools" in t and len("swe-tools") == 9
    changed = []
    for p in sorted(SRC.rglob("*")):
        if p.is_file() and not filecmp.cmp(p, dest / p.relative_to(SRC), shallow=False):
            changed.append(p.relative_to(SRC).as_posix())
    print("changed vs edit-early-v7-decide-nt:", changed)
    assert set(changed) == {"skills/swe-tools/scripts/locate.py", "skills/swe-tools/SKILL.md", "prompts/system.md"}, changed


if __name__ == "__main__":
    main()
