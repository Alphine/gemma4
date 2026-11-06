# Evidence pointers: claim id -> files in this package

Claim ids and wording are those of EVIDENCE_MAP.md (kept verbatim in this package, with the original repository paths).
This table resolves every path cited there to its location in the release, or states why it is not shipped.

| claim id | cited source | in this package | note |
|---|---|---|---|
| E1 | `research/top10/time-cap/staff_statements.json` | withheld (C44) | third-party text; the Writeup cites topics instead |
| E2 | `research/submissions_history.csv` | released | data/submissions_history.csv |
| E2 | `research/top10/ab/session1/session_report.json` | released | ab_kit/session1/session_report.json |
| E3 | `research/top10/anatomy/a1_wall.txt` | released | anatomy/a1_wall.txt |
| E4 | `kernels/gemma4-lb-eval/README.md` | released | evaluator_emulation/README_lb_eval.md |
| E4 | `research/gold_control/env_probe.json` | withheld (C93) | describes the grader image; part of embargoed environment findings; ids |
| E5 | `research/gold_control/gold_control_summary.json` | released | data/gold_control_summary_aggregates.json |
| E6 | `git log -- research/gold_control` | not in package | regenerate from the listed script or take the value from the Writeup |
| E6 | `research/top10/wheelhouse-0211/DIFF.md` | withheld (C97) | verbatim wheel source (harness_text hits 217) and per-task gold-patch file lists |
| E7 | `research/starlette-compat/FINDINGS.md` | withheld (C98) | contains gold-derived edits and per-task outcomes; environment finding |
| E8 | `research/top10/time-cap/staff_statements.json` | withheld (C44) | third-party text; the Writeup cites topics instead |
| E8 | `research/top10/wheelhouse-0211/staff_statements_since_10-06.txt` | withheld (C97) | verbatim wheel source (harness_text hits 217) and per-task gold-patch file lists |
| E9 | `research/top10/wheelhouse-0211/DIFF.md` | withheld (C97) | verbatim wheel source (harness_text hits 217) and per-task gold-patch file lists |
| E10 | `research/top10/wheelhouse-0211/strip_check_gold_patches.json` | withheld (C97) | verbatim wheel source (harness_text hits 217) and per-task gold-patch file lists |
| E11 | `research/lb-eval/lb_eval_summary.json` | released | data/lb_eval_summary_triage_first.json |
| E11 | `research/lb-eval-v4/lb_eval_summary.json` | released | data/lb_eval_summary_v4.json |
| E11 | `research/submissions_history.csv` | released | data/submissions_history.csv |
| E12 | (none: quoted / inferred) | - | - |
| A1 | `anatomy/a5_arms.txt` | released | anatomy/a5_arms.txt |
| A1 | `research/top10/ab/session1/session_report.json` | released | ab_kit/session1/session_report.json |
| A2 | `anatomy/a1_wall.txt` | released | anatomy/a1_wall.txt |
| A3 | `anatomy/a1_wall.txt` | released | anatomy/a1_wall.txt |
| A4 | (none: quoted / inferred) | - | - |
| A5 | `anatomy/a3_reasoning.txt` | released | anatomy/a3_reasoning.txt |
| A6 | `anatomy/a3_reasoning.txt` | released | anatomy/a3_reasoning.txt |
| A7 | `anatomy/a4_sinks.txt` | released | anatomy/a4_sinks.txt |
| A8 | `anatomy/a2_preedit.txt` | released | anatomy/a2_preedit.txt |
| A9 | `anatomy/a6_visibility.txt` | released | anatomy/a6_visibility.txt |
| A10 | `anatomy/a2_preedit.txt` | released | anatomy/a2_preedit.txt |
| A11 | `anatomy/a4_sinks.txt` | released | anatomy/a4_sinks.txt |
| A12 | `research/top10/time-cap/results.json` | released | time_cap/results.json |
| B1 | `research/top10/ab/PREREGISTRATION.md` | released | ab_kit/PREREGISTRATION.md |
| B2 | `datasets/gemma4-submissions/edit-early-v6-locate/prompts/system.md` | released | agent_arms/edit-early-v6-locate/prompts/system.md |
| B2 | `edit-early-v7-lean-nt/configs/sampling.yaml` | released | agent_arms/edit-early-v7-lean-nt/configs/sampling.yaml |
| B3 | `PREREGISTRATION.md` | released | ab_kit/PREREGISTRATION.md |
| B4 | `research/top10/ab/session1/session_report.json` | released | ab_kit/session1/session_report.json |
| B4 | `analyze_ab.py` | released | ab_kit/analyze_ab.py |
| B5 | (none: quoted / inferred) | - | - |
| B6 | `session_report.json` | released | ab_kit/session1/session_report.json |
| B7 | `anatomy/a3_reasoning.txt` | released | anatomy/a3_reasoning.txt |
| B8 | `session_report.json` | released | ab_kit/session1/session_report.json |
| B9 | `session_report.json` | released | ab_kit/session1/session_report.json |
| B10 | (none: quoted / inferred) | - | - |
| B11 | `paper/calc/ab_session1.py` | released | calc/ab_session1.py |
| B11 | `analyze_ab.py` | released | ab_kit/analyze_ab.py |
| B12 | `analyze_ab.py` | released | ab_kit/analyze_ab.py |
| B13 | `paper/calc/ab_session1.py` | released | calc/ab_session1.py |
| B14 | `research/top10/token-economy/knob_evidence.txt` | withheld (C46) | verbatim wheelhouse source lines (scanner: 6 harness_text hits) |
| B14 | `datasets/gemma4-submissions/edit-early-v7-lean-nt/configs/sampling.yaml` | released | agent_arms/edit-early-v7-lean-nt/configs/sampling.yaml |
| B14 | `research/top10/wheelhouse-0211/DIFF.md` | withheld (C97) | verbatim wheel source (harness_text hits 217) and per-task gold-patch file lists |
| N1 | `research/lora-pilot1/SUMMARY.md` | released | lora/pilot1/SUMMARY.md |
| N2 | (none: quoted / inferred) | - | - |
| N3 | `research/eval-lora/lora-ab1/summary.md` | released | lora/eval/lora-ab1/summary.md |
| N4 | `research/lora-pilot1/SUMMARY.md` | released | lora/pilot1/SUMMARY.md |
| N5 | `PREREGISTRATION.md` | released | ab_kit/PREREGISTRATION.md |
| N6 | `research/teacher-gen/C-behaviour/score_results.json` | released | teacher/C_behaviour/score_results.json |
| N6 | `spec.json` | released | teacher/C_behaviour/spec.json |
| N7 | `research/eval-lora/lora-ab1/summary.md` | released | lora/eval/lora-ab1/summary.md |
| N7 | `research/teacher-gen/C-behaviour/score.py` | released | teacher/C_behaviour/score.py |
| N8 | `research/teacher-gen/A-oracle/DESIGN.md` | withheld (C100) | licence question for Claude-authored teacher text: do not release as training data |
| N9 | `research/teacher-gen/gate-oracle/gate_results.json` | withheld (C102) | oracle_scored_rows.json embeds Claude-authored teacher text; whole directory held back pending licence decision |
| N10 | `research/teacher-gen/gate-oracle/strata_results.json` | withheld (C102) | oracle_scored_rows.json embeds Claude-authored teacher text; whole directory held back pending licence decision |
| N11 | `research/teacher-gen/B-hindsight/hint_validation_all_train.json` | released | teacher/B_hindsight/hint_validation_all_train.json |
| N12 | `research/top10/fourhands/gain_table.md` | released | fourhands/gain_table.md |
| N13 | `research/top10/fourhands/preflight_fh_checks.json` | not in package | regenerate from the listed script or take the value from the Writeup |
| N13 | `preflight_mock_transcripts.json` | not in package | regenerate from the listed script or take the value from the Writeup |
| N14 | `research/top10/locate-body/tables.md` | released | locate_body/tables.md |
| N15 | `research/top10/locate-body/trace_summary.md` | released | locate_body/trace_summary.md |
| N16 | `research/top10/locate-body/tables.md` | released | locate_body/tables.md |
| V1 | `research/top10/strategy/s1_noise.json` | released | strategy/s1_noise.json |
| V1 | `s2_posterior.json` | released | strategy/s2_posterior.json |
| V2 | `s1_noise.json` | released | strategy/s1_noise.json |
| V3 | `s2_posterior.json` | released | strategy/s2_posterior.json |
| V4 | `research/top10/intel/nb/dmitriigluzdov_gemma-4-measure-before-you-tune/*.txt` | withheld (C113) | third-party code/text, licences unknown |
| V5 | `research/top10/strategy/s5_out.txt` | released | strategy/s5_out.txt |
| V5 | `s5_voi.py` | released | strategy/s5_voi.py |
| V6 | `s5_out.txt` | released | strategy/s5_out.txt |
| V7 | `research/top10/strategy/s3_out.txt` | released | strategy/s3_out.txt |
| B6 | `datasets/gemma4-lb-eval-ckpt/results_*.jsonl` | released | 5 files, e.g. data/ckpt_scrubbed/results_edit-early-v4.jsonl |
| B6 | `paper/make_figures_v2.py` | released | figures/make_figures_v2.py |
| B9 | `scratchpad/runs.json` | not in package | regenerate from the listed script or take the value from the Writeup |
| B9 | `research/top10/anatomy/build_runs.py` | released | anatomy/build_runs.py |
| B13 | `research/top10/ab/analyze_ab.py` | released | ab_kit/analyze_ab.py |
| B13 | `paper/calc/ab_session1.py` | released | calc/ab_session1.py |
| B2 | `datasets/gemma4-submissions/edit-early-v6-locate/prompts/system.md` | released | agent_arms/edit-early-v6-locate/prompts/system.md |
| A2 | (none: quoted / inferred) | - | - |
| A12 | `research/top10/time-cap/results.json` | released | time_cap/results.json |
| V1 | `research/submissions_history.csv` | released | data/submissions_history.csv |
| V3 | `research/top10/strategy/s2_posterior.py` | released | strategy/s2_posterior.py |
| V2 | `s1_noise.json` | released | strategy/s1_noise.json |
| V7 | `research/top10/strategy/s3_hit.json` | released | strategy/s3_hit.json |
| N15 | `research/top10/locate-body/trace_summary.md` | released | locate_body/trace_summary.md |
| G1 | `research/teacher-gen/C-behaviour/per_run_table.jsonl` | released | teacher/C_behaviour/per_run_table.jsonl |
| G1 | `time_results.json` | released | teacher/C_behaviour/time_results.json |
| E3 | `kernels/gemma4-lb-eval/README.md` | released | evaluator_emulation/README_lb_eval.md |
| O1 | (none: quoted / inferred) | - | - |
| O2 | (none: quoted / inferred) | - | - |
| O3 | (none: quoted / inferred) | - | - |
| O4 | (none: quoted / inferred) | - | - |
| O5 | (none: quoted / inferred) | - | - |
| O6 | (none: quoted / inferred) | - | - |
| O7 | (none: quoted / inferred) | - | - |
| O8 | (none: quoted / inferred) | - | - |
| O9 | (none: quoted / inferred) | - | - |
| O10 | (none: quoted / inferred) | - | - |
| O11 | (none: quoted / inferred) | - | - |
| O12 | (none: quoted / inferred) | - | - |
| O13 | (none: quoted / inferred) | - | - |
| O14 | (none: quoted / inferred) | - | - |
| O15 | (none: quoted / inferred) | - | - |
| O16 | (none: quoted / inferred) | - | - |
| O17 | (none: quoted / inferred) | - | - |
| O18 | (none: quoted / inferred) | - | - |
| O19 | (none: quoted / inferred) | - | - |
| O20 | (none: quoted / inferred) | - | - |
| O21 | (none: quoted / inferred) | - | - |
| O22 | (none: quoted / inferred) | - | - |
| O23 | (none: quoted / inferred) | - | - |
| O24 | (none: quoted / inferred) | - | - |
| O25 | (none: quoted / inferred) | - | - |
| O26 | (none: quoted / inferred) | - | - |
| O27 | (none: quoted / inferred) | - | - |
| O28 | (none: quoted / inferred) | - | - |
| O29 | (none: quoted / inferred) | - | - |
| O30 | (none: quoted / inferred) | - | - |
| O31 | (none: quoted / inferred) | - | - |

Claims parsed: 105. Cited source paths resolved to released files: 71. Not in package: 4.

Reading guide: a path marked 'withheld' means the number is reported in the Writeup text and computed from
Competition-derived inputs that cannot be redistributed; the script that computes it is released and runs on
any user's own re-download of the competition data (see README.md, section 'Reproducing').
