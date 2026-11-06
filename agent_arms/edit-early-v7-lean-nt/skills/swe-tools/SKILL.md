---
name: swe-tools
description: Fast helper scripts for fixing an issue in /workspace - locate the code an issue is about, edit by line numbers, check compile/import/tests. Each call is one tool call and prints a short summary.
---
# swe-tools

Call with run_skill_script, skill_name: swe-tools (plain ASCII, no quotes or brackets), args = a JSON object. All scripts work on /workspace and are read-only except edit_lines.py.

| file_path | args | prints |
|---|---|---|
| scripts/locate.py | `{"issue": "<issue title + key lines, names, error text>"}` | top 8 definitions `path:line kind Class.name` with 2 lines each (tags in [..]: matched words, `via f` = called by a top hit, `<frame>` = a traceback frame; `(near-identical siblings: ...)` = methods of the same class with the same body, folded into the hit above), then `More candidates:` (next 12 definitions, names only, best of other files first), likely files and their test files |
| scripts/edit_lines.py | `{"path": "pkg/module.py", "start": 120, "end": 124, "expect": "<the old lines>", "new_text": "<replacement lines>"}` | writes, then the diff and the compile result |
| scripts/check.py | `{"path": "pkg/module.py", "test": "tests/test_module.py"}` | compile + import result, pytest -x -q pass/fail counts, first failure's E lines |

edit_lines.py notes:
- Lines start..end (1-based, inclusive) are replaced by new_text; `"end": start-1` inserts before start; `"new_text": ""` deletes.
- new_text is whole lines with their indentation; a final newline is optional (end it with an empty line to keep a blank line).
- Without new_text it only prints the numbered lines (use it to get exact line numbers).
- expect = the old lines being replaced. Whitespace/indentation and one extra level of escaping are ignored. If expect is not inside start..end but is the whole old block, the nearest place where that block really is gets replaced (the output says so). Otherwise nothing is written and the real lines are shown: retry with the right numbers.
- An edit that would make the .py file uncompilable is refused (file unchanged).
- After an edit, later line numbers shift by the printed amount.

check.py: `"test": "auto"` picks the best existing test file. It never runs a whole test directory.
