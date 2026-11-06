# Synthetic SWE task factory (CPU only)

Builds bug-fix tasks for rejection-sampling RFT of gemma-4-31b-it: Gemma solves them later (Kaggle GPU quota)
and we fine-tune on its short successful trajectories. This folder only **builds and validates the tasks**.

Everything heavy runs in Kaggle CPU kernels (cloning, wheels, snapshots, test runs). Only small summaries are
downloaded. No GPU, no Modal, no competition submission, no git commit/push.

Results (2026-10-07): **959 validated tasks from 25 repos** (815 train / 144 held-out) in the private dataset
`<kaggle-owner>/gemma4-synth-tasks` (18 MB). Details in "Results" below and `synth/results/manifest.json`.

## Rules the factory enforces

| rule | how |
|---|---|
| no contamination | `filter.py`: hard tier (fastapi, rich, requests, httpx), soft tier (`encode/*`, `textualize/*`, `tiangolo/*`, `fastapi/*`, `pydantic/*`, `samuelcolvin/*`, urllib3, anyio, h11, pip), name-token rule (`fastapi rich requests httpx starlette httpcore h11 anyio urllib3 pydantic pip`), link and vendored-path rules (research/data-scaling/README.md section 2). No public SWE dataset instances are used: every bug is generated here. |
| no shadowed packages | a repo whose import package ships in the competition wheels (click, jinja2, markupsafe, itsdangerous, werkzeug, flask, ...) is refused: the grader injects every wheel <20 MB into site-packages, which would shadow the workspace (the "requests 0/13" effect). Task repos must also not be dependencies of each other's tests. |
| permissive licence | GitHub API SPDX id (fallback: LICENSE text), allowlist MIT / BSD-2/3 / Apache-2.0 / ISC / PSF; recorded per repo. |
| no LLM-written content | problem statements come from `statements.py`: code that reads the failing tests and pytest's failure record and fills a fixed set of generic templates defined once in that file. Gold patches are the original code (reverse of the mutation). |
| pure-python, fast, hermetic | C/Cython/Rust packages refused; a test file is usable only if it passes whole in the grader emulation in <= 20 s twice. |

## Pipeline

1. **Factory** (`kernels/gemma4-synth-factory`, CPU, internet on) — `factory.py`
   * per repo: latest stable release tag (`git ls-remote`), shallow clone, record commit + date, licence,
     contamination filter, C-extension check;
   * test dependencies (pyproject `dependencies` + test extras/groups + `repos.py:test_deps`) downloaded as
     **pure-python wheels only** (`pip download --platform any --abi none`), minus packages already in the
     competition wheels and minus task-repo packages;
   * base snapshot `snapshots/base_snapshots/base_<owner>_<repo>.tgz`: pristine tree + a fresh `git init` with
     **no commits** (+ repo-local git identity), so the harness's baseline commit is the only commit and
     `git log -p` cannot reveal the bug or the hidden tests;
   * one emulated grader sandbox per repo, prepared exactly like `verify_task` (lb-eval's ContainerManager
     emulation: python 3.13 venv, no host site-packages, image packages, competition + our wheels injected,
     `sandbox/setup.py --fast-path`, hermetic pytest.ini/conftest, same pytest flags/env) + a pip shim (below);
   * baseline: every test file run alone with `synth_plug.py` (per-test outcomes + per-test line coverage via
     `sys.monitoring`); file usable iff exit 0, >0 passed, <= 20 s, same result twice;
   * mutation sites (`mutators.py`, inside function bodies of non-test package modules, covered by 1..250
     passing tests), ordered round-robin over operators, public/documented functions first; max 2 per function,
     max 22% per operator;
   * screening: apply, compile, run only the covering tests (SIGALRM 10 s/test, 60 s/run). Keep iff no collection
     error, no timeout, 1..8 previously passing tests fail, failing tests take <= 8 s, each failing test is
     defined in its own file (removable), and no failing test name is a prefix of a skipped/xfailed test name
     (verify_task's junit check matches required names by prefix);
   * ~15% of tasks are 2-hunk pairs (two kept mutations, same file or two files, different functions) kept iff
     the combined failing set equals the union of the singles' sets (both hunks necessary);
   * packaging: per task `snapshots/patches/<iid>.patch` (bug + failing tests removed), `patch` (gold),
     `test_patch` (re-adds the removed tests), `FAIL_TO_PASS`, statement.
2. **Validation** (`kernels/gemma4-synth-validate`, CPU, internet off, reads the factory output via
   `kernel_sources`) — `validate.py`: official `swegemma.harness.verification.verify_task` in the same emulation,
   gold → resolved, empty → not resolved, gold verification <= 60 s (concurrency 4, includes sandbox setup);
   an "agent view" sample (first task of every repo + 8%) checks the Phase-1 sandbox state: exactly 1 commit,
   clean tree, hidden tests absent, gold applies. Survivors are split and written as the dataset folder.
3. **Upload**: the validated folder (small) is downloaded to the scratchpad and published as the private
   dataset `<kaggle-owner>/gemma4-synth-tasks`, then deleted locally.

### Why the failing tests are removed from the snapshot (test_patch design)

The competition `tasks.jsonl` has no FAIL_TO_PASS: `verify_task` runs the **files touched by `test_patch`**
and requires pytest exit 0 (plus junit checks of F2P/P2P or of the `def test_*` names added by test_patch).
An empty `test_patch` with FAIL_TO_PASS set would make it run `pytest .` (the whole repo). So, as in the real
tasks, the failing tests are hidden: the snapshot patch deletes them, `test_patch` re-adds them, and verification
runs exactly those test files. We also set `FAIL_TO_PASS` (node ids; `Task` accepts the field) so the junit
check requires them to pass; `PASS_TO_PASS` is empty (exit code 0 over the whole files already covers it).

### Grader-emulation details that matter

* **pip shim.** On the Kaggle host the sandbox venvs are created without pip (no ensurepip), so the harness's
  `pip install -e /workspace` silently used the **host** pip: the task repo was editable-installed into the
  host interpreter (it replaced e.g. the host's `mistune`, which then broke nbconvert at the end of the factory
  run v2) and never into the sandbox. `common.make_manager` adds a `pip` shim (`venv/bin/python3 -m pip`) so the
  editable install lands in the sandbox venv like in the real grader. **Use the same wrapper in any kernel that
  runs these tasks** (see "Consuming the dataset").
* **git identity.** The sandbox env has no global git config in the emulation; the competition snapshots have
  commits, ours do not, so the baseline commit must succeed: the base snapshot's `.git/config` carries a
  repo-local `user.name/email`.

## Mutation operators (`mutators.py`)

cmp_flip (`<`↔`<=`, `==`↔`!=`, `is`↔`is not`, `in`↔`not in`), boolop_flip, binop_flip (+/- with integer
evidence), augassign_flip, const_off_by_one, bound_off_by_one (slice/range bounds ±1), index_flip
(`[0]`↔`[-1]`), negate_cond, remove_stmt, remove_early_exit, swap_args, default_value, kw_bool_flip,
return_bool_flip, name_sibling (startswith/endswith, lower/upper, min/max, any/all, ...), attr_sibling
(`self.start`→`self.end`), drop_except (handler body → `raise`), break_continue. Sites are found on the AST and
applied as byte-span edits, so the gold diff is minimal and formatting is untouched.

## Problem statements (`statements.py`)

Seeded by the instance id: 35% `terse` (title only), 30% `observed` (title + observed vs expected / exception),
35% `repro` (title + snippet inlined from the failing test: module imports, module-level constants and
parametrize values bound as literals, unittest asserts rewritten to `assert`, cut at the failing statement +
a Python-style traceback tail with only `repro.py` and non-mutated library frames). Fallback repro → observed →
terse. Never mentions the mutated file's path or lines, never names the mutated module in the title (symbols
like `pkg.mod.func` are shortened to `func`), never quotes the diff or the operator. Pairs describe both bugs.

## Commands

```bash
# local unit tests (toy code only, ~10 s)
python -m pytest synth/tests -q -p no:cacheprovider

# (re)generate the single-file kernels after editing synth/*.py or synth/kernel_configs.json
python synth/build_kernels.py            # --check: fail if stale (also a unit test)

# factory (CPU, internet on; never pass --accelerator)
python -m kaggle kernels push -p kernels/gemma4-synth-factory
python -m kaggle kernels status <kaggle-owner>/gemma4-synth-factory
python -m kaggle kernels output <kaggle-owner>/gemma4-synth-factory -p <scratch> --file-pattern "^synth.log|^factory_summary.json"

# validation (CPU, internet off, reads the factory's latest output)
python -m kaggle kernels push -p kernels/gemma4-synth-validate
python -m kaggle kernels output <kaggle-owner>/gemma4-synth-validate -p <scratch> --file-pattern "^dataset/"

# publish (from the scratchpad copy; dataset-metadata.json with id <kaggle-owner>/gemma4-synth-tasks)
python -m kaggle datasets create -p <scratch>/dataset -r zip      # first time
python -m kaggle datasets version -p <scratch>/dataset -r zip -m "..."   # later
```

Re-check the published dataset end to end: set `"validate": {"SOURCE": "dataset", "WRITE_DATASET": false}` in
`synth/kernel_configs.json`, add `<kaggle-owner>/gemma4-synth-tasks` to the validate kernel's
`dataset_sources`, rebuild, push.

## Dataset layout (`<kaggle-owner>/gemma4-synth-tasks`)

```
tasks.jsonl          competition schema: instance_id, repo, base_commit, problem_statement, patch (gold),
                     test_patch, hints_text (""), created_at (upstream commit date), FAIL_TO_PASS, PASS_TO_PASS ([]),
                     version (release tag), environment_setup_commit
meta.jsonl           per task: ops, files, funcs, lines, kind (single/pair), n_f2p, style, symbol, split, licence,
                     verify_gold_s, removed_tests ... (never shown to the agent; load_tasks ignores it anyway)
splits.json          train / heldout ids, heldout_repos, heldout_functions
manifest.json        per-repo licence/tag/commit/counts, operator + style + F2P histograms, validation stats, size
repos.json           factory record per repo (incl. skipped repos and why, per-test-file baseline status)
wheels/*.whl         extra pure-python wheels the test suites need (not in the competition wheels)
snapshots/base_snapshots/base_<owner>_<repo>.tgz   one per repo
snapshots/patches/<instance_id>.patch               per task start state (bug + hidden tests removed)
```

## Consuming the dataset (lb-eval / rft-collect)

The harness code runs unchanged; the kernel needs three small changes:

1. `dataset_sources` += `<kaggle-owner>/gemma4-synth-tasks`.
2. Use a **merged data dir** instead of the competition dir, and call `LB.setup_emulation(merged)` with it so
   the sandbox wheels = competition wheels + `wheels/` of the dataset, and `sandbox/setup.py` + docker shims come
   from the competition data:

   ```python
   from synth.common import build_merged_data_dir, make_manager   # or copy these ~40 lines
   ds = find_one("tasks.jsonl", prefer="gemma4-synth-tasks").parent
   data_dir = build_merged_data_dir(Path("/root/synth/data"), comp_data_dir(), ds / "tasks.jsonl",
                                    ds / "snapshots", ds / "wheels")
   LB.setup_emulation(data_dir)
   mgr = make_manager(LB)          # instead of LB.make_manager_cls()(...): adds the pip shim
   cfg = EvalConfig(tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots",
                    wheels_dir=data_dir / "wheels", graph_dir=str(data_dir / "graphs"),   # absent: no graph tools
                    embeddings_dir=str(data_dir / "embeddings"), ...)
   ```
   (`resolve_task_snapshot_paths` finds `base_snapshots/` + `patches/` by itself.)
3. `CONFIG["TASKS"]` = `splits.json["train"]` (keep `heldout` for evaluation).

No graphs/embeddings exist for synthetic tasks, so the harness does not advertise the graph tools.

## Results (factory v4 + validate v4, re-checked from the published dataset in validate v5)

| stage | count |
|---|---|
| candidate repos | 26 (more-itertools dropped: a >60 s test makes whole-file runs too slow) |
| repos with tasks | 25 (all permissive: 15 MIT, 8 BSD-3-Clause, 2 BSD-2-Clause) |
| candidates (factory) | 965 (798 single + 167 pairs), ~120 min on 4 CPU cores |
| validated (verify_task gold resolved, empty not, <= 60 s) | **959** (dropped: 4 gold not resolved in tinydb (storage/ordering state), 2 empty resolved in sqlparse) |
| agent-view check | 96/96 (one commit, clean tree, hidden tests absent, gold applies) |
| published-dataset re-check | 100/100 (4 per repo, interleaved), agent view 31/31 |
| split | 815 train / 144 held-out = 81 tasks of 2 held-out repos (PyCQA/pycodestyle, alecthomas/voluptuous) + 63 tasks of 44 held-out functions in the other repos |
| dataset size | 18.1 MB (25 base snapshots, 959 patches, 7 extra wheels = 0.98 MB: hypothesis, more_itertools, railroad_diagrams, simplejson, sortedcontainers, wcwidth, whatever) |

Per repo (tag, commit, valid): sqlparse 0.6.0 2f40da99 45, boltons 26.2.0 4332b35a 47, mistune v3.3.4 69ec2b8a 25,
pyparsing 3.3.3 d90d38b2 47, toolz 1.1.0 568c2b83 47, tomlkit 0.15.1 1bd7e3bc 47, parse 1.22.2 529dc2e0 37,
cachetools v7.2.1 9976f1a8 36, inflection 0.5.1 b00d4d34 5, jmespath 1.1.0 6ff419a8 47, tabulate v0.10.0 3b4cd509 47,
xmltodict v1.0.4 8d7f1fdb 27, pyflakes 4.0.2 4756a379 47, markdown 3.11.0 0ffbf00c 47, semver 3.1.0 642d1bc8 10,
marshmallow 4.3.1 c7b559a1 47, tinydb v4.9.0 1e39ad3a 43, pydash v8.1.0 f46de366 47, tomli 2.4.1 c5f44690 47,
schedule 1.2.2 82a43db1 47, w3lib v2.5.0 537c5d46 23, pycodestyle 2.15.0 9647851c 34, jsonpointer v3.1.1 ada0cade 16,
funcy 2.1 04aafae9 47, voluptuous 0.16.0 87825d6d 47 (licences and full SHAs in `results/manifest.json`).

* Operators: remove_stmt 105, drop_except 94, swap_args 92, remove_early_exit 91, name_sibling 81, negate_cond 78,
  boolop_flip 74, cmp_flip 70, attr_sibling 59, const_off_by_one 55, binop_flip 52, augassign_flip 43,
  bound_off_by_one 41, default_value 41, return_bool_flip 41, index_flip 38, break_continue 37, kw_bool_flip 32.
* FAIL_TO_PASS per task: 1: 347, 2: 201, 3: 127, 4: 86, 5: 56, 6: 46, 7: 41, 8: 55.
* Styles: terse 34.2%, observed 46.0%, repro 19.8% (repro often falls back to observed when the test needs a
  fixture or helper). 27 statements were cleaned by `postprocess_statement` (unittest "Lists differ:" prefix)
  before upload; the generator now does this itself.
* Gold diffs are tiny: median 2 changed lines (p90 4). Statements: median 139 chars (p90 537). Failure types:
  AssertionError 620, TypeError 152, AttributeError 82, ValueError 61, KeyError 52, pytest.raises "DID NOT RAISE" 38.
* Verification time (concurrency 4, incl. sandbox setup): gold p50 12.8 s, p90 24.3 s, max 34.4 s.

### Caveats

* **Easier than the dev tasks in one way, harder in another.** Each bug is a 1–2 line mutation in a small
  library (dev tasks are real PRs, often multi-file, with new features). But the agent never sees the failing
  tests (they are hidden exactly like the competition's), and a third of the statements are terse titles, so
  localisation is the hard part, which is the skill we want (edit early on terse issues).
* Some mutations are "obviously wrong code" (e.g. `self.end - self.end`, `raise` replacing a handler) that an
  agent can spot by reading; others (off-by-one, swapped args, dropped statements) need reproduction.
* Titles derived from test names can read awkwardly ("Make error key doesnt exist gives wrong result").
* In the emulation the sandbox venv has no pip unless the shim is used; kernels that do not use
  `make_manager` will editable-install into the host (it overwrote the host's mistune in factory run v2).
* `inflection` (5) and `semver` (10) are small; `pycodestyle` and `voluptuous` are held out entirely.
