# Paired A/B analysis kit (edit-early-v7 token economy)

Pre-registered analysis for the 43-task paired run `edit-early-v6-locate` (control) vs `edit-early-v7-decide-nt` vs
`edit-early-v7-lean-nt` on `kernels/gemma4-lb-eval` (or the audited copy). Design and decision rules: `PREREGISTRATION.md`
(written before any data exists; do not edit after the first session).

| file | role |
|---|---|
| `PREREGISTRATION.md` | exact definitions, stats, stop rules, gates, decision table, power, design problems found |
| `analyze_ab.py` | CLI: results jsonl + traces -> markdown report + JSON (works on partial data, deterministic) |
| `trace_metrics.py` | ATIF-v1.7 trace -> per-run metrics (stdlib only; verified on `training/tests/fixtures/trace_*`) |
| `synth.py`, `test_analyze_ab.py` | synthetic generator + 36 pytest tests (`python -m pytest -q`, about 2.5 min) |
| `oc_sim.py`, `oc_results.json` | operating characteristics of the decision rule on synthetic worlds |
| `kernel_extract/` | single-file Kaggle CPU kernel that turns the lb-eval kernel's trace output into a small `trace_metrics_*.jsonl` |

Needs `numpy` and `scipy` (both present locally and on Kaggle).

## Data volume and the trace problem (read first)

* `results_<arm>.jsonl` and the summaries are small (about 150 KB per arm); `merge_ckpt.py` already downloads only those and
  accumulates them in `datasets/gemma4-lb-eval-ckpt/`. Analyse that directory.
* ATIF traces are about 150 KB each (about 20 MB for 129 runs, more with long runs). **Each lb-eval kernel version has its own
  output; a new push replaces what `kaggle kernels output` returns. So the traces of session k must be extracted BEFORE session
  k+1 is pushed.** The default route does the extraction on Kaggle (CPU, about 2 min) and downloads only a few hundred KB.
* The row format has no token counts, so tokens, first-edit and mechanism metrics exist only if traces are processed. Without
  traces the report still gives resolved, timeouts, agent minutes, calls; the gate then fails on `trace_coverage`.

## After each Kaggle session k (exact commands, repo root)

```bash
# 0. wait for the session to finish
python -m kaggle kernels status <kaggle-owner>/gemma4-lb-eval          # (or the audited kernel id)

# 1. small result files -> checkpoint dataset (existing flow; downloads results_*.jsonl + summaries only)
python kernels/gemma4-lb-eval/merge_ckpt.py            # use the merge_ckpt.py of the kernel copy that was pushed

# 2. traces -> small metrics file (default route, nothing large touches the laptop)
python research/top10/ab/kernel_extract/build_kernel.py
python -m kaggle kernels push -p research/top10/ab/kernel_extract           # CPU kernel, no --accelerator
python -m kaggle kernels status <kaggle-owner>/gemma4-ab-extract         # wait until complete (about 2 min)
mkdir -p research/top10/ab/data/metrics_s<k>
python -m kaggle kernels output <kaggle-owner>/gemma4-ab-extract -p research/top10/ab/data/metrics_s<k> -o
#    -> trace_metrics_<timestamp>.jsonl (< 1 MB). Do this BEFORE pushing the next lb-eval session.

# 3. analysis (partial data is fine; chronological order of --trace-metrics matters only for retried pairs)
python research/top10/ab/analyze_ab.py \
    --results datasets/gemma4-lb-eval-ckpt \
    --trace-metrics research/top10/ab/data/metrics_s* \
    --preflight unknown \
    --out-json research/top10/ab/out/report_s<k>.json --out-md research/top10/ab/out/report_s<k>.md
```

Final analysis (all 43 pairs finished, CPU preflight done): same command with `--preflight pass` (or `fail`).
Optional filters: `--tasks tasks43.txt` (one instance_id per line; default is every non-dead row), `--launch-ids id1,id2`
(only rows of the A/B launches; use it if the checkpoint dataset ever held earlier rows of the same arm names),
`--arm lean=<other-name>` to rename an arm, `--roles control,decide` to drop an arm that was stopped, `--seed`, `--n-boot`.

Optional route when the log of the lb-eval session says the trace folder is small (< 40 MB): download the traces directly
instead of step 2, and pass the folder or tarball to `--traces` (several allowed, chronological):

```bash
python -m kaggle kernels output <kaggle-owner>/gemma4-lb-eval -p research/top10/ab/data/traces_s<k> -o \
    --file-pattern "^(traces/.*[.]json|.*traces.*[.]tar[.]gz)$"
du -sh research/top10/ab/data/traces_s<k>          # delete and use the Kaggle route if > 40 MB
python research/top10/ab/analyze_ab.py --results datasets/gemma4-lb-eval-ckpt --traces research/top10/ab/data/traces_s* ...
```

Trace files must live in a directory named after the arm (`traces/<arm>/<task>.json`, as the kernel writes them) or in a
tarball with such member names; members are read in memory and never extracted to disk.

## Reading the output

`ab_report.md` first line is the decision (`CONTINUE`, `STOP <arm>`, `HOLD`, `SUBMIT lean-nt`, `SUBMIT decide-nt`,
`SUBMIT neither`). Before all 43 pairs are done the decision can only be `CONTINUE` or `STOP <arm>` (stop rules for harm,
config errors and futility apply from 30 complete pairs; success never stops the run early). The JSON carries every number
(`contrasts`, `mechanism`, `order`, `config_errors`, `stop_rules`, `gates`, `accounting`, `warnings`).
Power statement without data: `python research/top10/ab/analyze_ab.py --power research/teacher-gen/C-behaviour/paired_results.json`.

## Tests

```bash
cd research/top10/ab && python -m pytest -q          # 36 tests, about 2.5 min
python oc_sim.py --n-sim 100                         # regenerates oc_results.json (about 15 min)
```
