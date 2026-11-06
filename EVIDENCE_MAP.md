# Evidence map for paper/draft_3000.md

Confidence: **measured** = computed from raw rows/files; **modelled** = output of a fitted/simulated model; **assumed** = rests on an assumption stated in the text; **third-party** = statement by others (forum/notebook), not verified by us. Paths are relative to the repo root. Anatomy dir = `research/top10/anatomy/`.

## Abstract and section 3 (setting, environment control)

| id | claim as written | source file(s) | how derived | confidence |
|---|---|---|---|---|
| E1 | one scoring run executes about 120 tasks sequentially in 12 h; about 60 per public/private split | `research/top10/time-cap/staff_statements.json` (topic 744258, 2026-09-29T14:04, Ryan Holbrook) | quoted | third-party |
| E2 | cap 5.5 min / 60 tool calls | `research/submissions_history.csv` (edit-early rows), `research/top10/ab/session1/session_report.json` budget | read | measured |
| E3 | decode roughly 35 tokens/s | `research/top10/anatomy/a1_wall.txt` line 1 (0.0287 s/token -> 34.8 tok/s) | 1/0.0287 | measured |
| E4 | emulation details (python:3.13-slim, wheels pre-seeded, no host site-packages, sitecustomize neutralised, pip offline, container code path); CPU mock mode | `kernels/gemma4-lb-eval/README.md`, `research/gold_control/env_probe.json`, draft.md section 3.1-3.2 | read | measured (design) |
| E5 | 129 dev tasks; 40 solvable; fastapi 1/67 (63 typing_inspection); requests 0/13, 9 shadowed, 3 empty-patch passes; rich 39/48 | `research/gold_control/gold_control_summary.json` (by_repo, gold_failure_reasons, n_empty_resolved) | read | measured |
| E6 | control run committed 2026-10-04, wheelhouse up to v28; not re-run after v29 | `git log -- research/gold_control` (614f5a7, 2026-10-04); `research/top10/wheelhouse-0211/DIFF.md` version table | date + absence of re-run (we know of no re-run in repo) | measured; "not re-run" is an absence claim: author please confirm |
| E7 | leaky sandboxes: 10/67 fastapi gold pass, 31/67 with Starlette-1.x compat edit | `research/starlette-compat/FINDINGS.md` Headline (a=10, b=31) | read | measured |
| E8 | staff: hidden tasks validate 100% with gold (2026-09-29); private repositories (2026-10-01); no training-task repo in test set (2026-10-09) | `research/top10/time-cap/staff_statements.json` (744370, 744951), `research/top10/wheelhouse-0211/staff_statements_since_10-06.txt` (747248, 2026-10-09T13:30) | quoted | third-party |
| E9 | wheelhouse v29 (2026-10-08) added patch stripping and tool-error handling; v30 (2026-10-09) behaviour-neutral refactor | `research/top10/wheelhouse-0211/DIFF.md` | read | measured (diff of wheels) |
| E10 | stripping leaves all 129 gold patches unchanged | `research/top10/wheelhouse-0211/strip_check_gold_patches.json` (n=129, n_changed=0, n_emptied=0) | counted | measured |
| E11 | projections 0.101 (triage-first) and 0.116 (edit-early-v4), both scored 0.08 | `research/lb-eval/lb_eval_summary.json` (0.1008), `research/lb-eval-v4/lb_eval_summary.json` (0.1163), `research/submissions_history.csv` (56823681 0.08, 56894471 0.08) | read | measured (projection); LB single draw |
| E12 | interpretation that the dev-mix assumption was wrong because test repos differ | E8 | inference from staff statement | assumed (inference) |

## Section 4 (time anatomy; 40 runs)

| id | claim | source | derivation | confidence |
|---|---|---|---|---|
| A1 | 40 runs, 3 configurations, 14 tasks, 13 with all three, concurrency 1, 5.5 min cap | `anatomy/a5_arms.txt` per-task matrix (14 tasks); `research/top10/ab/session1/session_report.json` | counted | measured |
| A2 | latency = 0.163 + 0.0287 s per completion token, R2=0.996, 1,009 calls | `anatomy/a1_wall.txt` line 1 | regression output | measured |
| A3 | Table 1: wall 251.8; reasoning decode 151.2 (60.0%); tool-arg decode 46.2 (18.3%); compaction 22.4 (8.9%); killed in flight 18.0 (7.1%); prefill 6.1 + fixed 4.1 = 10.2 (4.1%); tool 1.8 (0.7%); other 1.3 content + 0.6 gap = 1.9 (0.8%) | `anatomy/a1_wall.txt` row "ALL" | divided by wall; sums to 251.7 (rounding) | measured |
| A4 | LLM 99% of wall | A3: (209.0 primary + 22.4 + 18.0)/251.8 = 99.0% | | measured |
| A5 | 45 of 1,009 calls >1,000 reasoning tokens (4.5%) took 30% of LLM seconds | `anatomy/a3_reasoning.txt` ALL row: 32+7+6 = 45; sec_gt1000 2508.6 / sec_total 8360.0 | ratio | measured |
| A6 | 7 calls reached the output cap | `anatomy/a3_reasoning.txt` n_len = 7 (ALL); finish counts 3/4/0 | read | measured |
| A7 | 61 invalid-argument calls cost 17.5 s/run (7.0% wall); 25 of 66 edit_file calls invalid (38%) | `anatomy/a4_sinks.txt` section A | read | measured |
| A8 | first edit in gold file 29 of 30 edited runs | `anatomy/a2_preedit.txt` ("edited runs 32 with gold info 30 gold file hit 29") | read | measured |
| A9 | gold lines visible by call 3.1 mean, first edit at call 11.3 (n=26) | `anatomy/a6_visibility.txt` last lines (k_visible mean 3.1; fe mean 11.3; 26 runs) | read | measured |
| A10 | 17.6% pre-edit calls redundant (105/597) | `anatomy/a2_preedit.txt` | read; classifier is ours (rule-based) | measured (heuristic) |
| A11 | cap thinking at 1,024: 25.7 s/run (10%); 512: 51.3 s (20%), trajectory fixed | `anatomy/a4_sinks.txt` section D | read | modelled (upper bound) |
| A12 | cap 5.5 -> 7 min: resolve rate 28% -> 30%, overlapping intervals; 86 runs | `research/top10/time-cap/results.json` P_resolve_solvable K60_a0.0_C5.5 (0.281 [0.175, 0.413]) and C7.0 (0.30 [0.175, 0.4375]); n_runs 86 | read | modelled |

## Section 5 (pre-registered A/B, session 1)

| id | claim | source | derivation | confidence |
|---|---|---|---|---|
| B1 | pre-registered 2026-10-09 before data; arms control/decide/lean; 43 tasks; Amendment 1 moved gate to censored tokens-to-first-edit and resolved margin -1 -> -2 (filed before data) | `research/top10/ab/PREREGISTRATION.md` (header, section 1, Amendment 1) | read | measured |
| B2 | decide prompt rules (3 short sentences, read at most 2 ranges, edit by call 5); lean = decide + thinking budget 1024 + max_output 2560 | `diff` of `datasets/gemma4-submissions/edit-early-v6-locate/prompts/system.md` vs `edit-early-v7-decide-nt`; `edit-early-v7-lean-nt/configs/sampling.yaml` | diff | measured |
| B3 | MDE 12.3% tokens at 43 pairs | `PREREGISTRATION.md` section 7 | read | modelled |
| B4 | session aborted by health guard after 184 min, 13 complete triples, 40 runs; decision CONTINUE | `research/top10/ab/session1/session_report.json` (minutes 183.9, n_rows 40, guard_events vllm_dead); `analyze_ab.py` output (rerun, "Decision: CONTINUE (13/43 complete pairs)") | rerun of `research/top10/ab/analyze_ab.py --launch-ids lbeval-20261007-043225` | measured |
| B5 | token gate not evaluable (no traces extracted) | same rerun: "trace coverage 0%" | | measured |
| B6 | Table 2 reads+searches before first edit 9.6 / 9.5 / 9.8 (n=14/13/13) | `session_report.json` mechanism.m2_mean_read_plus_search_before_edit 9.571 / 9.538 / 9.769 | read | measured |
| B7 | Table 2 mean reasoning tokens/call 194 / 236 / 202; max 4,093 / 4,093 / 1,042 | `anatomy/a3_reasoning.txt` per-arm rows (mean 194.4, 235.8, 201.5; max 4093, 4093, 1042 from lean section) | read | measured |
| B8 | output-cap hits 3 / 4 / 0 | `session_report.json` m1_n_calls_reaching_max_output | read | measured |
| B9 | mean s/call 10.5 / 10.7 / 8.5; lean 19% lower than control | `session_report.json` mean_call_latency_s 10.521 / 10.728 / 8.52 | 8.52/10.521-1 = -19.0% | measured |
| B10 | reasoning grew 21% under brevity rule | 235.8/194.4 - 1 | | measured |
| B11 | 13 triples: resolved 6/7/6; timeouts 8/7/7; mean agent min 4.41/4.18/3.90; lean vs control 2 each way | `paper/calc/ab_session1.py` (output reproduced in section "Calc outputs" below); `analyze_ab.py` rerun table | recomputed from `datasets/gemma4-lb-eval-ckpt/results_*.jsonl` | measured |
| B12 | lean shorter in 6 of 7 time-orderable pairs, sign test p=0.125 | `analyze_ab.py` rerun, "Censoring-aware time" table, lean_vs_control | read | measured |
| B13 | exploratory censored tokens-to-first-edit: lean -17.7% [-31.5, -5.8], decide -4.9% [-27.8, +17.1] | `paper/calc/ab_session1.py` (bootstrap 10,000, seed 20261010, from row fields ct_to_first_edit / ct_atif) | own recomputation, not the pre-registered kit (no traces) | measured, exploratory |
| B14 | budget forwarded in adk-submission 0.2.13 + vLLM 0.19.1 | `research/top10/token-economy/knob_evidence.txt`; `datasets/gemma4-submissions/edit-early-v7-lean-nt/configs/sampling.yaml` header comment; `research/top10/wheelhouse-0211/DIFF.md` | read | measured |

## Section 6 (negative results)

| id | claim | source | derivation | confidence |
|---|---|---|---|---|
| N1 | 53 trajectories -> 90 examples, r32, 2 epochs, 42 steps, $4.34 | `research/lora-pilot1/SUMMARY.md` | read | measured |
| N2 | train loss 0.119 -> 0.046; val 0.0905 -> 0.0909 (7 examples) | same | read | measured |
| N3 | 70/144 vs 72/144, +1.4 pts, CI [-4.2, +6.9], McNemar p=0.80, 9 vs 7, timeouts 41% vs 40% | `research/eval-lora/lora-ab1/summary.md` | read | measured |
| N4 | first 48 pairs +6.2 pts (p=0.375) | `research/lora-pilot1/SUMMARY.md` | read | measured |
| N5 | effects above ~+7 pts excluded | N3 CI upper bound 6.9; `PREREGISTRATION.md` MDE 6.7% tokens at 144 | read | modelled |
| N6 | profile: 22 of 52 SFT trajectories fit (strict) | `research/teacher-gen/C-behaviour/score_results.json` pilot1_kept53_sft_set fit_strict 22, n 52; spec reference n=48 from `spec.json` | read | measured |
| N7 | profile was built after the null result | file mtimes: `research/eval-lora/lora-ab1/summary.md` 2026-10-08 03:10; `research/teacher-gen/C-behaviour/score.py` 2026-10-08 07:33; git 69ee771 (03:11) vs f9c42c9 (10:02) | timestamp comparison | measured |
| N8 | oracle: 243 valid trajectories = 29.8% of 815; ~6 calls vs 19-20 | `research/teacher-gen/A-oracle/DESIGN.md` section 7 | read | measured |
| N9 | 87% edits confident without discriminating evidence; 0% exec before first edit | `research/teacher-gen/gate-oracle/gate_results.json` G5 (0.872, exec_evidence_before_first_edit 0.0) | read; heuristic defined in `gate_oracle.py` | measured (heuristic) |
| N10 | covered tasks: 28/33 (85%) at 5.5-min cap vs 25/66 (38%) elsewhere | `research/teacher-gen/gate-oracle/strata_results.json` | read | measured |
| N11 | hint levels 1-3 leak 0 of 815 gold lines; B not run | `research/teacher-gen/B-hindsight/hint_validation_all_train.json`; DESIGN.md verdict | read | measured |
| N12 | parallel attempts -5.0 to -0.5 (shared cap), -2.2 to +1.9 (cap lifted), base assumptions m=0.10, kappa=0.29 | `research/top10/fourhands/gain_table.md` (blocks 1-4; min/max recomputed) | min/max over rows K=2-4 | modelled |
| N13 | mock-validated plumbing only | `research/top10/fourhands/preflight_fh_checks.json`, `preflight_mock_transcripts.json` | existence | measured |
| N14 | locate body: all gold lines visible 11-13% dev (CI 6-19) | `research/top10/locate-body/tables.md` DEV all tasks, r1-60 11.2 [5.6, 16.8], r12-60 12.8 [7.2, 19.2] | read | measured |
| N15 | 0.48 calls/run saved (0.35-0.62), ~1-2 s, n=153 synthetic runs | `research/top10/locate-body/trace_summary.md` | read | modelled (trace replay) |
| N16 | dev gold functions median 77 lines vs 20 synthetic | `research/top10/locate-body/tables.md` (DEV median 77; TRAIN median 20) | read | measured |

## Section 7 (noise)

| id | claim | source | derivation | confidence |
|---|---|---|---|---|
| V1 | six configs scored 4,5,5,7,5,5 tasks; mean 5.17, SD 0.98, binomial SD 2.17, p=0.04 | `research/top10/strategy/s1_noise.json` lb_new_harness_6; `s2_posterior.json` configs.raw | read | assumed (k = floor(score x 58), 58 tasks) |
| V2 | beta-binomial dispersion 0.29 (CI 0.16-0.51), 144 pairs | `s1_noise.json` lora_ab1_144.kappa | read | modelled |
| V3 | true score 5.2 tasks (3.8-6.5) = 0.089; run SD 1.1 (0.5-2.1) | `s2_posterior.json` six_ours (mu0_mean 5.17, mu0_ci, sig_mean 1.107, sig_ci) | 5.17/58 | modelled |
| V4 | identical archive 0.12 / 0.10 / 0.06 | `research/top10/intel/nb/dmitriigluzdov_gemma-4-measure-before-you-tune/*.txt` (Figure 9 text) | read | third-party |
| V5 | LB power 0.43/0.82 (SD 1.1, 1 vs 3 slots), 0.29/0.59 (SD 1.5), +2.3 tasks one-sided alpha 0.05 | `research/top10/strategy/s5_out.txt` (lb rows); `s5_voi.py` header | read | modelled |
| V6 | dev paired reruns +5.1 tasks: power 0.64 (1 rerun), 0.96 (3); +2.5: 0.25, 0.50; alpha 0.05 | `s5_out.txt` dev rows | read | modelled |
| V7 | best of N=10 reaches 12/58: <0.5% at 0.12, 4% at 0.14, 35% at 0.16 | `research/top10/strategy/s3_out.txt` scenario A mode same (N=10 column: 0.001-0.003, 0.043, 0.350) | read | modelled |

## Calc outputs (paper/calc/ab_session1.py, run 2026-10-10)

```
complete triples 13 ; rows per arm control 14, decide 13, lean 13
control resolved 6 timeouts 8 mean_agent_min 4.41 ; decide 7 / 7 / 4.18 ; lean 6 / 7 / 3.90
decide vs control only-treated 1 only-ref 0 ; lean vs control 2 / 2 ; lean vs decide 1 / 2
cens ct_to_first_edit ratio-1: decide -0.049 CI [-0.278, 0.171]; lean -0.177 CI [-0.315, -0.058]
```


## Corrections from independent verification (2026-10-10; see paper/VERIFICATION.md; applies to paper/draft_final.md)

| id | new claim | source | change |
|---|---|---|---|
| B6 | Table 2 col 2 = reads+searches+locate before first edit attempt, 13 matched tasks: 11.2 / 10.6 / 10.7 (medians 8/9/10; never-edited 4/3/1 count all calls) | `datasets/gemma4-lb-eval-ckpt/results_*.jsonl` (pre_edit_read+search+locate); `paper/make_figures_v2.py` | replaces 9.6/9.5/9.8 (harness M2 excludes locate, all runs) |
| B9 | pooled mean call latency 8.49 / 9.58 / 8.49 s (primary only 7.97 / 9.15 / 7.87); per-run-mean 10.52 / 10.73 / 8.52 is the weighting artefact | `scratchpad/runs.json` (session-1 trace build via `research/top10/anatomy/build_runs.py`) | the 19% per-call claim removed |
| B13 | registered gate (first successful source edit, kit on traces/): lean -6.3% [-23.2,+23.4], order-adj -5.6%; decide +1.9% [-21.1,+29.3]; FAIL. Attempt-based (row field): -17.7%, -4.9% | `research/top10/ab/analyze_ab.py --traces <extracted traces/>`; `paper/calc/ab_session1.py` | registered result primary; B5 ("not evaluable") withdrawn |
| B2 | decide also removes the step-0 triage; control had a 5-sentence brevity rule | diff of `datasets/gemma4-submissions/edit-early-v6-locate/prompts/system.md` vs `edit-early-v7-decide-nt` | disclosure added |
| A2 | latency = 0.40 s + 0.0287 s/token (completion tokens only), R2 0.996; 0.163 s with prompt tokens | runs.json primary calls | intercept corrected |
| A12 | 28% to 30-32% (alpha 0 / 0.5 / 1) | `research/top10/time-cap/results.json` | range added |
| V1 | seven configs 4,5,5,7,5,5,6; mean 5.29; SD 0.95; binomial SD 2.19; p=0.02 | `research/submissions_history.csv` (v5 = 0.10, k=6) | six -> seven |
| V3 | posterior 5.3 tasks (4.2-6.4) = 0.091; run SD 1.05 (0.5-2.1) | `research/top10/strategy/s2_posterior.py` re-run with v5 (scratch) | updated |
| V2 | kappa 0.29 = Beta concentration, 144 base/adapter pairs (synthetic) | `s1_noise.json` | label |
| V7 | N=10: model A <0.5/4/35%, model B 4/23/67% | `research/top10/strategy/s3_hit.json` | model B added |
| N15 | 0.8-1.7 s of ~197 s per synthetic run | `research/top10/locate-body/trace_summary.md` | denominator |
| G1 | second distribution: 286 synthetic runs, minutes vs completion tokens r=0.943 (Spearman 0.853, boot CI 0.93-0.96), vs calls r=0.635; Modal per-call fit 0.44 + 0.0221 s/tok, R2 0.953 (4,060 calls, pilot1 harvest) | `research/teacher-gen/C-behaviour/per_run_table.jsonl`, `time_results.json` | new |
| E3 | "4x L4" is our evaluator, not the grader | `kernels/gemma4-lb-eval/README.md` | scope |

## Addendum 2026-10-10: graph-position negative result (section 6, abstract, contribution 3)
| claim | value as written | source | derivation | confidence |
|---|---|---|---|---|
| pre-registered, plan hashed before extraction | sha256 recorded before kernel push | research/app-track/A-difficulty/PREREG.md, PREREG.sha256 | written by the study agent before step 1 (its report); hash f2a37c36...95fe852 | measured (process) |
| AUC 0.921 -> 0.923, +0.002 [-0.012, +0.016]; kill +0.03 | pooled leave-one-repo-out logistic regression, 959 tasks, 25 repos | research/app-track/A-difficulty/results/results.json, tables.md | cluster bootstrap over repos then tasks, 2,000 resamples | measured |
| gradient boosting +0.008 | [-0.005, +0.023] | same | same | measured |
| mutated function named in statement: hit@5 87% vs 27% | 155+ named vs 664 not named | same, exploratory | locate v6 function-level hit@5 | measured (exploratory) |
| neutralising the name cost 49 points | 86.6% -> 37.7% (268 tasks) | same, counterfactual | rename in the query, re-run localiser | measured (exploratory) |
| structure mattered only at one hop; distance undefined for 41% of tasks | d_sym loss by hop; d_test undefined 41% | same | AST call graph resolves 32% direct + 13% by name | measured (exploratory) |
| NOT VERIFIED INDEPENDENTLY | these rows come from the study agent's report and result files; recompute before submission | | | pending |
| removed: 'Ceiling' bullet merged into the implication (1,024-token cap saves at most 10% of wall) | 25.7 s/run | research/top10/anatomy (token-cap simulation) | held-trajectory simulation | modelled |

## Addendum 2026-10-10 (gold-control re-run under wheelhouse v30)
| claim | value | source | derivation | confidence |
|---|---|---|---|---|
| Re-run of gold control under v30 gives identical 40 solvable tasks and failure reasons | n_solvable 40, gold_resolved 43, empty_resolved 3; fastapi 1/67, requests 0/13, rich 39/48 | kernel <kaggle-owner>/gemma4-gold-control-v30 (31.7 min, COMPLETE); outputs gold_control_summary.json, solvable_ids.json compared with research/gold_control/ (2026-10-04) | same script, new slug, same 129 tasks; solvable id sets equal (only_new, only_old empty) | measured |
