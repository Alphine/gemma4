# Gemma 4 developer-agent: time and resource analyses (release package)

**STATUS: STAGED, NOT PUBLISHED.** This tree was prepared for attachment to the paper-track Writeup. Do not make it
public before the paper is submitted (see RELEASE_DECISIONS.md in the author's repository, which is not part of this
package). Items marked EMBARGO in the licence table embody environment findings.

## Purpose
Everything needed to re-derive the numbers of the Writeup from files, without Kaggle Competition Data: analysis scripts
with an evidence map, the pre-registration and its analysis kit (36 tests), mock-mode evidence runs, scrubbed
checkpoint result rows, the synthetic task factory (code; the tasks themselves ship as a separate dataset), the
evaluator emulation (code only), and two application-track studies.

## Structure
| directory | files | size |
|---|---|---|
| `ab_kit/` | 35 | 0.3 MB |
| `agent_arms/` | 24 | 0.3 MB |
| `anatomy/` | 17 | 88.5 KB |
| `app_track/` | 59 | 2.4 MB |
| `calc/` | 1 | 3.0 KB |
| `data/` | 23 | 0.4 MB |
| `evaluator_emulation/` | 4 | 0.2 MB |
| `figures/` | 16 | 1.0 MB |
| `fourhands/` | 9 | 0.2 MB |
| `kernels/` | 2 | 0.2 MB |
| `locate_body/` | 14 | 5.3 MB |
| `lora/` | 8 | 0.3 MB |
| `strategy/` | 16 | 76.4 KB |
| `synth/` | 25 | 0.2 MB |
| `teacher/` | 17 | 0.7 MB |
| `time_cap/` | 6 | 0.1 MB |
| `token_economy/` | 10 | 53.0 KB |
| `tools/` | 1 | 21.7 KB |

Top level: `LICENSE` (BSD-3-Clause), `NOTICE`, `EVIDENCE.md` (claim id -> file), `EVIDENCE_MAP.md` (claim wording and
derivations, original repository paths), `smoke_test.py`, `tools/scan_competition_leaks.py`, `manifest.json` (sha256).

| directory | content |
|---|---|
| `ab_kit/` | pre-registration (+Amendment 1), `analyze_ab.py`, `trace_metrics.py`, `synth.py` generator, `test_analyze_ab.py` (36 tests), operating characteristics, `mock_evidence/` (5 mock sessions), `session1/` report |
| `data/` | `ckpt_scrubbed/` result rows of the lb-eval runs, `paper_ckpt_scrubbed/` graph-arm rows, submission scores, gold-control aggregates (counts only) |
| `anatomy/` | time-anatomy scripts a1-a7 and text outputs (task ids shuffled, reasoning excerpts removed) |
| `strategy/`, `time_cap/`, `token_economy/`, `fourhands/`, `locate_body/` | simulations and models behind the modelled claims |
| `app_track/` | study A (pre-registered, null), Pallas bug-injection factory (19 seeds, 296 verified mutants), Pallas mutant suite run in CPU interpret mode and natively on TPU v5e (`C_tpu/README.md`) |
| `synth/`, `kernels/` | synthetic task factory and its two Kaggle CPU kernels |
| `evaluator_emulation/` | grader-faithful local scorer and gold-control kernel (task sets externalised) |
| `agent_arms/` | control / decide / lean prompts, sampling configs and the swe-tools skill |
| `teacher/`, `lora/` | behaviour-profile scoring on synthetic runs, LoRA A/B summaries |
| `figures/` | 12 figures, captions, make scripts |
| `calc/` | recomputation of session-1 numbers from the scrubbed rows |

## Task identifiers
Instance ids of Competition dev tasks are replaced by shuffled pseudonyms `T001`... (same task, same pseudonym in all
files; the map is not published). Per-task text (patches, test output, error text, reasoning excerpts) is removed.
Only numeric and categorical fields remain. Synthetic task ids (`boltons_s0021`) are our own and kept.

## Reproducing (run from this directory)
```
python smoke_test.py            # all checks, about 15 minutes (study A dominates); --quick: about 1 minute, skips the kit tests and study A
python calc/ab_session1.py      # B11 (13 triples, resolved 6/7/6, timeouts 8/7/7) and B13 bootstrap
python ab_kit/analyze_ab.py --results data/ckpt_scrubbed --launch-ids lbeval-20261007-043225 --preflight unknown
python -m pytest -q ab_kit/test_analyze_ab.py       # pre-registered analysis kit, 36 tests (1 real-fixture test skipped)
python -m pytest -q synth/tests                      # task factory
python app_track/A_difficulty/analyze.py app_track/A_difficulty/results/features_tasks.jsonl OUT_DIR
```
Needs Python 3.11+, numpy, scipy, pandas, scikit-learn, pytest, matplotlib. `jax` (0.11.1) only for executing the Pallas
mutants: `python app_track/C_pallas/run_factory.py SEEDS OUT 3 0`; the 51-variant suite runs with `python app_track/C_tpu/pallas_tpu_suite.py` (CPU interpret only unless a TPU backend is present; see `app_track/C_tpu/README.md`). Kaggle kernels (`kernels/`, `evaluator_emulation/`)
need the competition data mount and wheelhouse, which are not part of this package.

Which paper numbers can be recomputed here: B1-B14 (A/B session 1: rows and kit; the registered gate needs traces, which
are not shipped, so B5/B13 gate values are reported from the Writeup), V1-V7 (strategy simulations; V4 is third-party),
N1-N5, N12 (summaries and models), N6-N11, G1 (behaviour tables and scripts on synthetic runs), the graph-position study
(all of it), the Pallas study (all of it). A1-A12 (time anatomy) need per-call traces of dev runs: scripts and printed
outputs are shipped, inputs are withheld. E1-E12 and the figures 1-9 on dev tasks are derived from Competition-derived
inputs; counts are in `data/gold_control_summary_aggregates.json`. See EVIDENCE.md for every claim id.

## Licence table
| component | licence | note |
|---|---|---|
| all code (`*.py`, kernels) | BSD-3-Clause (LICENSE) | OSI-approved, permits commercial use |
| documents, pre-registration, evidence map, figures | CC-BY-4.0 | |
| `data/*` rows, tables, results JSON | CC-BY-4.0 | numeric fields only; Gemma 4 output is Apache-2.0 |
| synthetic tasks (separate dataset) | per-repository MIT / BSD-3 / BSD-2 | notices in NOTICE and in each snapshot |
| Pallas seeds and mutants | own seeds and mutation code BSD-3-Clause; seeds taken from jax stay Apache-2.0 | jax library files not redistributed |
| `evaluator_emulation/` | BSD-3-Clause | code only: needs the competition harness wheels at run time; EMBARGO until the paper is submitted |

## Third-party notices
See NOTICE: 25 synthetic-task repositories (15 MIT, 8 BSD-3, 2 BSD-2) with tag and commit, and jax (Apache-2.0).

## Not included, and why
* Competition tasks, snapshots, gold and test patches, official graphs and embeddings, wheelhouse wheels and diffs,
  HARNESS_README: Competition Data (rule 2.4.b), not redistributable to non-participants.
* Per-task outputs on dev tasks (test output, agent patches, traces, per-call tables): derived from Competition Data.
* Trained adapters and Claude-authored teacher templates: licence question; the Writeup states they are withheld.
* Third-party forum posts, leaderboard snapshots and notebooks: not ours to redistribute; cited by topic id or URL.
* Unreleased strategy and environment-repair material.
