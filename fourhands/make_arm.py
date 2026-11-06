"""Build datasets/gemma4-submissions/edit-early-v9-fourhands (K hands, default 2) from edit-early-v6-locate.
Usage: python make_arm.py [K] [out_dir]
Layout: agent.yaml (SequentialAgent) -> sub_agents/hands.yaml (ParallelAgent: hand_a [/workspace, full tools],
hand_b.. [private copies /tmp/hand_x, run_command + skills only]) -> sub_agents/selector.yaml (LlmAgent: select_attempt.py then submit_patch).
eval_config.yaml is byte-identical to edit-early-v6-locate.  No curly braces anywhere in instructions.
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "datasets", "gemma4-submissions", "edit-early-v6-locate")
K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "datasets", "gemma4-submissions", "edit-early-v9-fourhands")
LETTERS = "abcd"[:K]
CALLS_EACH = max(10, (60 - 6) // K - 2)

if os.path.isdir(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
for p in ("prompts/system.md", "agent.yaml"):
    os.remove(os.path.join(DST, p))
os.makedirs(os.path.join(DST, "sub_agents"), exist_ok=True)
os.makedirs(os.path.join(DST, "prompts"), exist_ok=True)

v6 = open(os.path.join(SRC, "prompts", "system.md"), encoding="utf-8").read()


def sub(text, old, new, count=1):
    assert old in text, old[:60]
    return text.replace(old, new, count)


# ---------------- hand A: v6 prompt, edits in place, no submit_patch
a = v6
a = sub(a, "call `submit_patch` immediately WITHOUT editing anything and reply with one line `ENV-BROKEN: <signal>`.",
        "stop immediately WITHOUT editing anything and reply with one line `ENV-BROKEN: <signal>`.")
a = sub(a, "your very next action MUST be `submit_patch`: do not read files",
        "your very next action MUST be your one-line reply: do not read files")
a = sub(a, "Then call `submit_patch` and reply with a one-line summary.",
        "Then reply with a one-line summary. Do NOT call submit_patch (you have no such tool: a separate selector step submits).")
a = sub(a, "If the budget is nearly used up, submit your best non-empty source patch rather than nothing.",
        "If the budget is nearly used up, leave your best non-empty source change in place and reply with your one-line summary.")
assert "call `submit_patch`" not in a and "`submit_patch`" not in a.replace("Do NOT call submit_patch", "")
head_a = ("You are hand A of %d parallel attempts at the same issue. You work directly in /workspace. The other hands work on "
          "private copies and are none of your business. The total tool-call budget is SHARED: use at most %d tool calls, "
          "then reply with your one-line summary. A selector chooses the best attempt afterwards.\n\n" % (K, CALLS_EACH))
open(os.path.join(DST, "prompts", "hand_a.md"), "w", encoding="utf-8", newline="\n").write(head_a + a)

# ---------------- hands B..: private copy, edit via edit_lines.py only
b_tmpl = """You are hand HANDLETTER of %d parallel attempts at the same issue. You work ONLY in your private copy ROOT, never in /workspace. The total tool-call budget is SHARED: use at most %d tool calls (the copy command is call 1), then reply with your one-line summary. A selector chooses the best attempt afterwards. You have no submit_patch tool: never try to submit.

## Your private copy
- Your FIRST tool call, exactly: run_command with command `cp -a /workspace ROOT && cd ROOT && git status --short | head -5`. Every later shell command starts with `cd ROOT &&`.
- Source files are under ROOT, not /workspace. The issue text and the workspace tree mention /workspace: read that as ROOT.
- You have no read_file or edit_file tool: read with `sed -n 'A,Bp' path` (at most 80 lines), edit ONLY with the swe-tools script edit_lines.py.

## Tools: the swe-tools skill
Call run_skill_script with skill_name: swe-tools (exactly these 9 ASCII characters), file_path: one of the three below, and args: a JSON object with the named keys. ALWAYS add the key root with the value ROOT to args. Each is ONE fast call. Do NOT call load_skill or load_skill_resource.
- scripts/locate.py, keys issue and root: the issue title plus its key lines (names, code, error text; at most about 60 words). Prints the best-matching definitions path:line kind Class.name and the test files of the likely files.
- scripts/edit_lines.py, keys root, path (relative to ROOT), start, end (1-based, inclusive), expect (the old lines you replace, copied from your last read), new_text (the complete replacement lines, with indentation). Prints the diff and the compile result; refuses an edit that would not compile.
- scripts/check.py, keys root, path (edited source file) and test (its existing test file, or auto): compile + import + pytest -x -q on that one file.
- Tool-call values are plain text: never wrap or end a value (skill_name, file_path) with a backtick, quote or bracket.

## Workflow
1. Triage (call 2): if the issue asks for NEW code outside the package, skip. Otherwise run the existing test file for the named code ONCE: `cd ROOT && timeout 60 python3 -m pytest tests/test_x.py -x -q 2>&1 | tail -25`, or import the module you expect to edit. If it shows an unfixable breakage (a third-party ModuleNotFoundError that is not part of the repository package, or SSL certificate errors), reply with one line `ENV-BROKEN: <signal>` and stop without editing.
2. Locate (call 3): scripts/locate.py with key issue. Read its best hit with sed -n. Terse issue: turn the title words into identifiers and run ONE `cd ROOT && git grep -n -e 'def <id>' -e 'class .*<Id>' -- '*.py' | head -20`. Fix the definition the title names, not a caller.
3. Before editing write one line: `PLAN: <file>:<function> - <root cause in one sentence> - <change>`. Make your first edit with scripts/edit_lines.py by tool call 7. Do NOT write a reproduction script before the first edit. Fix the root cause; never catch the exception at the call site. Keep edits small. Prefix every python3 or pytest command with `timeout 20` (a targeted test file: `timeout 60`).
4. After each edit run `cd ROOT && git diff -- <file> | head -40 && python3 -m py_compile <file>`. Then scripts/check.py with the edited file and its existing test file (at most 2 verification runs). Ignore pre-existing failures. Never run bare pytest or full suites.
5. Before replying run `cd ROOT && git checkout -- tests/ 2>/dev/null; git status --short` and delete any scratch file it lists. Then reply with a one-line summary.

## Never
- Modify, create or delete test files (test_*.py, *_test.py, anything under tests/) or conftest.py, pytest.ini, pyproject.toml.
- Touch /workspace, search outside ROOT, or install packages.
- Refactor or reformat unrelated code. Finish without a non-empty source change unless step 1 said ENV-BROKEN.
"""

for i, L in enumerate(LETTERS):
    if i == 0:
        continue
    root = "/tmp/hand_%s" % L
    txt = b_tmpl.replace("HANDLETTER", L.upper()).replace("ROOT", root) % (K, CALLS_EACH)
    assert "{" not in txt and "}" not in txt
    open(os.path.join(DST, "prompts", "hand_%s.md" % L), "w", encoding="utf-8", newline="\n").write(txt)

selector = """You are the selector of a multi-attempt pipeline. You do exactly three things and nothing else; never edit or read files.
1. Call run_skill_script with skill_name: swe-tools (exactly these 9 ASCII characters), file_path: scripts/select_attempt.py, and args: a JSON object with the single key root whose value is /workspace.
2. When it has printed a final line starting with SELECTED, call submit_patch (no arguments). If the script reported an error, still call submit_patch.
3. Reply with one line: the SELECTED line you saw.
Do NOT call load_skill or load_skill_resource. Tool-call values are plain text: never wrap a value in backticks, quotes or brackets.
"""
assert "{" not in selector
open(os.path.join(DST, "prompts", "selector.md"), "w", encoding="utf-8", newline="\n").write(selector)

# ---------------- configs
samp = open(os.path.join(SRC, "configs", "sampling.yaml"), encoding="utf-8").read()
open(os.path.join(DST, "configs", "sampling_a.yaml"), "w", encoding="utf-8", newline="\n").write(samp)
for i, L in enumerate(LETTERS):
    if i == 0:
        continue
    s = re.sub(r"(?m)^temperature: .*$", "temperature: 0.5", samp)
    s = re.sub(r"(?m)^seed: .*$", "seed: %d" % i, s)
    open(os.path.join(DST, "configs", "sampling_%s.yaml" % L), "w", encoding="utf-8", newline="\n").write(s)
open(os.path.join(DST, "configs", "sampling_sel.yaml"), "w", encoding="utf-8", newline="\n").write(
    "temperature: 0.0\ntop_p: 1.0\nseed: 0\nmax_output_tokens: 512\nthinking_config:\n  thinking_budget: 0\n  include_thoughts: false\n")
os.remove(os.path.join(DST, "configs", "sampling.yaml"))

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
full_tools = ["run_command", "read_file", "edit_file", "write_file", "get_status", "get_code_neighbors", "search_similar_code",
              "get_code_subgraph"]


def llm_yaml(name, desc, instr, samp_cfg, tools, extra=""):
    t = "\n".join("  - " + x for x in tools)
    return ("name: %s\nagent_class: LlmAgent\nmodel: %s\ndescription: \"%s\"\ninstruction: !include ../prompts/%s\n"
            "generate_content_config: !include ../configs/%s\n%stools:\n%s\nskills:\n  - skills/swe-tools\n"
            "disallow_transfer_to_parent: true\ndisallow_transfer_to_peers: true\n" % (name, MODEL, desc, instr, samp_cfg, extra, t))


open(os.path.join(DST, "sub_agents", "hand_a.yaml"), "w", newline="\n").write(
    llm_yaml("hand_a", "Attempt A, edits /workspace in place.", "hand_a.md", "sampling_a.yaml", full_tools))
for i, L in enumerate(LETTERS):
    if i == 0:
        continue
    open(os.path.join(DST, "sub_agents", "hand_%s.yaml" % L), "w", newline="\n").write(
        llm_yaml("hand_" + L, "Attempt %s on a private copy." % L.upper(), "hand_%s.md" % L, "sampling_%s.yaml" % L,
                 ["run_command", "get_status"]))
open(os.path.join(DST, "sub_agents", "selector.yaml"), "w", newline="\n").write(
    llm_yaml("selector", "Picks one attempt, applies it, submits.", "selector.md", "sampling_sel.yaml",
             ["get_status", "submit_patch"], extra="include_contents: none\n"))
hands = "name: hands\nagent_class: ParallelAgent\ndescription: \"Independent attempts.\"\nsub_agents:\n" + "".join(
    "  - config_path: sub_agents/hand_%s.yaml\n" % L for L in LETTERS)
open(os.path.join(DST, "sub_agents", "hands.yaml"), "w", newline="\n").write(hands)
open(os.path.join(DST, "agent.yaml"), "w", newline="\n").write(
    "# edit-early-v9-fourhands: %d concurrent attempts (hand A in /workspace, others on private copies) then a selector.\n"
    "# Prompts/scripts derive from edit-early-v6-locate; eval_config.yaml is byte-identical to it. See research/top10/fourhands/REPORT.md.\n"
    "name: fourhands\nagent_class: SequentialAgent\ndescription: \"Parallel attempts then selector.\"\nsub_agents:\n"
    "  - config_path: sub_agents/hands.yaml\n  - config_path: sub_agents/selector.yaml\n" % K)

# ---------------- scripts: select_attempt.py + root mapping for /tmp copies
shutil.copy(os.path.join(HERE, "select_attempt.py"), os.path.join(DST, "skills", "swe-tools", "scripts", "select_attempt.py"))
loc = os.path.join(DST, "skills", "swe-tools", "scripts", "locate.py")
t = open(loc, encoding="utf-8").read()
old = '''    cands = [opts.get("root"), os.environ.get("SWE_TOOLS_ROOT")]
    f = sys._getframe()
    while f is not None:   # run_skill_script's wrapper keeps the original cwd in a local
        if "_orig_cwd" in f.f_locals:
            cands.append(f.f_locals["_orig_cwd"])
        f = f.f_back
'''
new = '''    cands = [opts.get("root"), os.environ.get("SWE_TOOLS_ROOT")]
    origs = []
    f = sys._getframe()
    while f is not None:   # run_skill_script's wrapper keeps the original cwd in a local
        if "_orig_cwd" in f.f_locals:
            cands.append(f.f_locals["_orig_cwd"])
            origs.append(f.f_locals["_orig_cwd"])
        f = f.f_back
    r0 = opts.get("root")
    if r0 and not os.path.isdir(r0) and (r0 == "/tmp" or r0.startswith("/tmp/")):
        # subprocess sandbox: /tmp is <sandbox>/tmp, next to <sandbox>/workspace (containers: /tmp exists, no mapping)
        for o in origs + [os.environ.get("PWD") or ""]:
            sib = os.path.join(os.path.dirname(o.rstrip("/")), "tmp", r0[len("/tmp/"):])
            if o and os.path.isdir(sib):
                cands.insert(0, sib)
                break
'''
assert old in t
open(loc, "w", encoding="utf-8", newline="\n").write(t.replace(old, new))
md = os.path.join(DST, "skills", "swe-tools", "SKILL.md")
s = open(md, encoding="utf-8").read()
s += ("\n| scripts/select_attempt.py | `{\"root\": \"/workspace\"}` | (selector step only) compares /workspace and the private copies "
      "/tmp/hand_*, leaves the best diff in /workspace, prints a final SELECTED line |\n")
open(md, "w", encoding="utf-8", newline="\n").write(s)
# eval_config byte-identical
assert open(os.path.join(SRC, "eval_config.yaml"), "rb").read() == open(os.path.join(DST, "eval_config.yaml"), "rb").read()
print("built", DST, "K=%d" % K, "calls_each=%d" % CALLS_EACH)
