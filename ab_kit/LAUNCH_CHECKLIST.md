# A/B launch checklist (prepared 2026-10-09; run on or after the Kaggle GPU quota reset 2026-10-10)

Arms (control first): edit-early-v6-locate, edit-early-v7-decide-nt, edit-early-v7-lean-nt. 43 tasks (40 solvable + 3 FREE).
Kernel to push: `kernels/gemma4-lb-eval-audited` ONLY. Never push `kernels/gemma4-lb-eval`. Never `NvidiaL4X4`.

## Gate 0: facts to verify on launch day (do not trust this file's dates)
- [ ] Kaggle GPU quota reset happened and remaining quota >= 24 h (read it in the Kaggle UI: Settings > Account > GPU quota; the CLI cannot show it).
- [ ] `python -m kaggle datasets status <kaggle-owner>/gemma4-lb-eval-ckpt` is `ready`; files: run_config.json (303 B, launch_id lbeval-20261007-043225) + only results_edit-early-v4 / results_triage-first.
- [ ] `python -m kaggle datasets status <kaggle-owner>/gemma4-submissions` is `ready` and lists edit-early-v6-locate, edit-early-v7-decide-nt, edit-early-v7-lean-nt (the v7 arms were versioned 2026-10-09).
- [x] mock kernel `<kaggle-owner>/gemma4-lb-eval-mockfinal` COMPLETE, `MOCK CHECKS PASS` (23 checks, 0 failed, 2 info), final script incl. the context_overflow fix, 2026-10-09.
- [ ] `python -m pyflakes kernels/gemma4-lb-eval-audited/gemma4-lb-eval.py` shows nothing.
- [x] CPU preflight PASS: decide-nt (kernel v20, 26 PASS/2 SKIP) and lean-nt (kernel v21, 26 PASS/2 SKIP), read 2026-10-09.
- [ ] CONFIG in the audited script: MODE real, ARMS (3, control first), ONLY_TASKS = SOLVABLE + FREE (43), LAUNCH_ID lbeval-20261007-043225,
      SAVE_TRACES True, ARM_ORDER perm6, ATOMIC_TASKS/CALL_LOG/PREFLIGHT/PREFLIGHT_BUDGET_TEST/HEALTH_CHECK True, MOCK_FAULT None, TEST_EXPECT_OVERRIDE None,
      MAX_SESSION_MIN 265 (3 sessions, ~22.7 h quota) or 345 (2 sessions, ~21.1 h).
- [ ] kernel-metadata: id <kaggle-owner>/gemma4-lb-eval, enable_gpu true, enable_internet false.

## Launch
```
cd <repo root>
python -m kaggle kernels push -p kernels/gemma4-lb-eval-audited --accelerator NvidiaL4
python -m kaggle kernels status <kaggle-owner>/gemma4-lb-eval
```
Early read at ~35-40 min: status/session_report must show no ABORT/GUARD events:
`python -m kaggle kernels output <kaggle-owner>/gemma4-lb-eval -p <fresh dir> -o --file-pattern "^(status|session_report)\.json$"`
A COMPLETE far short of the session length means a guard fired (intended): fix the cause before relaunching.

## After each session (in this order)
1. `python kernels/gemma4-lb-eval-audited/merge_ckpt.py --with-traces`; wait for the ckpt dataset to be `ready`.
2. Extract traces on Kaggle (CPU) before the next push: see research/top10/ab/README.md (build_kernel.py -> push -> output metrics_s<k>).
3. `python research/top10/ab/analyze_ab.py --results datasets/gemma4-lb-eval-ckpt --trace-metrics research/top10/ab/data/metrics_s* --preflight unknown --out-json ... --out-md ...` (final decision with `--preflight pass`).
4. Push the next session only if the report shows no unexplained ABORT, no request_mismatch/config_error rows, <= 2 infra rows per arm.

## Stop / no-go triggers
Unexplained ABORT; request_mismatch or config_error rows; vLLM start failure, < 4 GPUs, self-test fail; first 3 runs of an arm degenerate;
> 2 infra rows per arm; quota < ~24 h before the push. Pre-registered analysis and Amendment 1: PREREGISTRATION.md.
