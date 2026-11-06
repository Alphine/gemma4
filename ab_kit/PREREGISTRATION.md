# Pre-registration: paired A/B of edit-early-v7 token economy (written 2026-10-09, before any A/B data)

Frozen together with `analyze_ab.py` and `trace_metrics.py` in this folder. The analysis code was tested on synthetic data and
on the two real ATIF fixtures only. After the first A/B result row exists, changes to this file or to the decision logic must be
listed in a dated "Amendments" section at the end; they may not alter a decision already computed.

## 1. Design

* Unit: task (43 dev tasks = 40 gold-control-solvable + 3 FREE; dead tasks excluded, hidden tasks have none per staff). Every task
  is run once under each arm, paired.
* Arms (roles): `control` = edit-early-v6-locate (thinking_budget 4096, max_output_tokens 4096), `decide` =
  edit-early-v7-decide-nt (prompt change, no triage; sampling identical to control), `lean` = edit-early-v7-lean-nt (decide's
  prompt + thinking_budget 1024 + max_output_tokens 2560). eval_config identical (cap 5.5 min, 60 tool calls, 120 turns).
* Contrasts: `decide_vs_control` and `lean_vs_decide` (primary, attribution), `lean_vs_control` (overall; the gate for lean-nt).
  "x_vs_y" always means treated x minus reference y; a negative change is an improvement for tokens, minutes, calls.
* Arm order: the audit copy rotates arm order per task (6 permutations). The analysis infers each arm's position per task from a
  row field (`arm_pos`, `arm_position`, `order_pos`, `order` or `arm_order`; int, or list/comma string of arm names) or else from the
  trace timestamps (first agent step `extra.timestamp - extra.elapsed_s`, ranked within the task).

## 2. Exact definitions

**Run selection.** Rows come from `results_*.jsonl`; `run_mode != real` rows (mock) and `set == dead` rows are ignored (or
`--tasks`). For each (arm, task) the last row with `infra_error == false` is the run (more than one is reported as a warning).
If only infra rows exist: 1 infra row = `pending` (the kernel retries; up to `MAX_ATTEMPTS` = 2), 2 or more = `exhausted`. An
exhausted pair is **missing**, never imputed as a failure. A task is *complete* if all arms in the analysis have a run; every
paired statistic uses complete tasks only ("complete case over all arms in the analysis"; arms with no rows at all are dropped
with a warning, `--roles` drops an arm explicitly). Infra errors are the kernel's `INFRA_MARKERS` (connection errors, container
setup, harness crash); model-caused exceptions stay in the data as real, usually failed, runs.

**Tokens.** Per run: the sum over agent steps of `step.metrics.completion_tokens` (fallback `tool_calls[0].extra.usage.completion_tokens`).
This is the vLLM completion count and **includes thought tokens**; thought tokens are not reported separately in ATIF. One agent
step = one LLM call. Steps without token metrics are counted in `n_calls_missing_tokens`; a run with more than 10 percent of calls
missing tokens has NaN tokens (pair dropped from token metrics). The step sum is cross-checked against
`final_metrics.total_completion_tokens` (mismatch is a warning; step sum is used). LLM calls that the harness makes outside the agent
loop (event-compaction summaries) may not appear as steps; if they do not, they are missing equally in all arms. The audit
kernel's HTTP call log is the ground truth and should be compared once.

**Time.** `agent_min` from the row. A run is *timed out* if `agent_error`/`error` contains "session timeout" or `agent_min >=
cap - 0.01` (cap = `budget.max_time_minutes`); its `agent_min` is set to exactly the cap (the real data shows exact ties at the
cap). Estimated LLM seconds = 1.065 x LLM calls + 0.02968 x completion tokens (Kaggle concurrency-1 fit).

**Counted tool calls.** Tool calls excluding `submit_patch` and `get_status`. **Call kinds** (copied from
`research/teacher-gen/C-behaviour/score.py`, refined): `read` = `read_file`; `shell_read` = `run_command` whose command contains
cat / sed -n / head / tail / nl / grep / rg / less / awk; `locate` = `run_skill_script` of `locate.py`; `edit` = successful
`edit_file`, `write_file`, `edit_lines.py`, `sed -i`, `git apply`, `patch -p`, `perl -pi`, or a `cat >`/`tee` redirect to a `.py`
path, **excluding scratch paths** (any `tests`/`test` directory, `conftest.py`, `*_test.py`, basenames starting repro, reproduce,
debug, scratch, tmp, temp, test_, check_, verify, demo, try_, poc, issue). `write_file` of a root-level file that no earlier
`read_file` touched counts as scratch. A call *failed* if its observation JSON has a non-ok `status`, an `error`/`error_code`
key, `stdout` starting with `NOT WRITTEN`, or a non-zero `exit_code`.

**First source edit.** The first `edit` call that did not fail. `calls_to_first_edit` = its 1-based index among counted tool
calls (the locate call counts as call 1). `tokens_to_first_edit` / `sec_to_first_edit` = cumulative completion tokens / `elapsed_s`
at the step containing it. A run with no such edit has no value (NaN; the paired metric uses pairs with an edit in both arms) and a
sensitivity version `calls_to_first_edit_cens` assigns it `max_tool_calls`. Edits made through other mechanisms (python
heredocs) are not detected; the share of runs with `patch_chars > 0` but no detected edit is visible as `no_edit_runs` vs empty patches.

**Reads before first edit.** Count of `read` + `shell_read` calls before the first source edit, counted only after the first
`locate` call if one exists (the "between locate and first edit" window), else from the start.

**Config-attributable errors.** *Hard*: `thinking_token_budget` ValueError in the row error or in an agent step, or a row flagged
`config_error`/`error_class == config` by the audit kernel. *Soft*: the harness phrase "Tool call arguments were truncated", or
`MAX_TOKENS` / `finish_reason` length in a step's extra/metrics/non-stdout observation text (never in the task prompt or in tool
stdout, so repo code that contains the word does not count). A treated arm's attributable count = tasks with a hard error +
tasks with a soft marker where the **paired control run has none**. Control markers are reported, never counted against control.

**Errors/failures.** `empty_patch` = `patch_chars == 0`. `timeout` as above. `resolved` is the kernel's `resolved` (non-empty patch
and tests pass); it is a guard, not a target.

## 3. Statistics

All random draws are seeded from (seed 20261010, CRC32 of the metric label); 10,000 bootstrap resamples; percentile 95% CIs.

* Continuous metrics (tokens, agent_min, est_llm_s, llm_calls, tool_calls, tokens_per_call, calls_to_first_edit, reads_before_first_edit,
  tokens_to_first_edit, sec_to_first_edit, tokens_noTO = tokens on pairs where neither run timed out): paired bootstrap over
  tasks for the ratio of means `mean(x)/mean(y) - 1` (undefined, reported null, if mean(y)=0), and for the median of paired
  differences; two-sided Wilcoxon signed-rank (zero differences dropped, exact or normal approximation by scipy `method=auto`; p = 1
  if all differences are zero); sign-flip permutation test of the mean difference (exact for up to 16 non-zero differences, else 20,000
  seeded flips).
* Time under the cap (right-censored at 5.5 min, about half of the runs): three views: (a) restricted mean of agent_min (valid
  paired estimand at the common cap) with bootstrap and Wilcoxon; (b) **orderable-pair sign test** (both censored = tie with no
  information; one censored = the uncensored one is shorter; none = sign of difference), exact binomial p; (c) exact McNemar on
  the timeout indicator.
* Binary outcomes (resolved, timeout, empty patch): exact McNemar (binomial on discordant pairs), paired rate difference with
  bootstrap CI.
* **Multiplicity.** Confirmatory family = tokens in 3 contrasts, adjusted with Holm (sign-flip p; Wilcoxon p adjusted likewise and shown).
  Unadjusted values are always printed. Secondary metrics are descriptive and unadjusted and may not be used to claim an effect.
  The two submission gates each use the unadjusted two-sided 95% bootstrap CI of their single contrast (lean_vs_control,
  decide_vs_control); the fallback order (lean first) is a fixed hierarchy. Simulation (below) shows the probability of a false
  SUBMIT under the null is far below 5 percent, so no further correction is applied.
* **Order effect.** For each contrast and for tokens and agent_min: OLS of the per-task paired difference on the
  arm-position difference (slope = effect per position step, intercept = effect at equal positions), bootstrap CI over tasks. If the
  slope CI excludes 0 the order effect is flagged and **the gate additionally requires the order-adjusted (intercept) estimate to pass**.
  If positions do not vary across tasks (fixed order) the effect is reported as non-identifiable/confounded.

## 4. Mechanism checks

* **M1 (lean only; per-call thought limit).** Reasoning tokens are not in ATIF, so the pre-registered "max reasoning tokens <= budget + ~10"
  is evaluated on reasoning tokens only if a trace/call log provides them (`reasoning_tokens`, `completion_tokens_details.reasoning_tokens`).
  Otherwise the proxy applies: pass iff (i) the share of lean calls with completion_tokens >= max_output - 8 is <= 1 percent,
  (ii) max completion_tokens <= max_output + 8, (iii) lean p99 of per-call completion tokens < control p99. M1 does not apply to decide.
* **M2 (per treated arm vs control; pairs with an edit in both arms, >= 5 pairs).** Pass iff **both** the median paired difference of
  `calls_to_first_edit` is <= -2 **and** the ratio of means of reads-before-first-edit is <= -25 percent.

## 5. Stop rules (applied after every session to the complete pairs at that time)

* Stop an arm if resolved(control) - resolved(arm) > 4 with >= 30 complete pairs.
* Stop an arm if its attributable config errors >= 3.
* Futility with >= 30 pairs: lean if M1 fails **and** M2 fails; decide (M1 not applicable) if M2 fails.
* No early stop for success. A stopped arm is removed with `--roles`; the remaining arms continue on their complete pairs.

## 6. Submission gate and decision table

An arm is eligible only if all of: (1) mechanism: lean = M1 **or** M2 passes, decide = M2 passes; (2) tokens: lean: change in
`lean_vs_control` <= -8 percent **and** 95% CI upper bound < 0; decide: change in `decide_vs_control` <= -5 percent **and** CI upper
bound < +3 percent (and, if an order effect is flagged, the order-adjusted estimate passes too); (3) resolved(arm) >= resolved(control) - 1;
(4) zero attributable config errors; (5) timeouts(arm) <= timeouts(control); (6) trace coverage >= 95 percent of complete pairs
and token data non-degenerate (an arm with all-zero tokens fails); (7) CPU preflight recorded as `pass`.

### Amendment 1 (2026-10-09, made BEFORE any A/B data exists; supersedes the text of gate items (2) and (3) above)

Reason: total completion tokens are time-censored (about half of real runs hit the 5.5 min cap, and a leaner arm fits more calls into
the same time, so a true 30 percent per-call cut shows as only about -25 percent total tokens at 45 percent timeouts), and the
`resolved >= control - 1` guard rejected harmless arms in 31 to 37 percent of simulated null experiments.

* Item (2) now uses the gate metric `tokens_to_first_edit_cens`: completion tokens spent until the first successful source edit; a run that
  never edits is right-censored at its total completion tokens. The thresholds are unchanged (lean: <= -8 percent and CI upper bound < 0;
  decide: <= -5 percent and CI upper bound < +3 percent; order-adjusted estimate on the same metric). The thresholds were NOT re-tuned
  and the power of this metric was NOT computed; treat a failed gate as "not shown", not "shown worse". Total tokens stay reported and keep the
  Holm-adjusted tests, but they no longer gate.
* Item (3) is now `resolved(arm) >= resolved(control) - 2`. The stop rule (deficit > 4 after >= 30 pairs) is unchanged.
* The timeout guard (5) is unchanged.
* Code: `PRE["gate_metric"]`, `PRE["resolved_margin"]`, `run_metrics()["tokens_to_first_edit_cens"]`, `gate()`. The pre-amendment file is
  kept as `analyze_ab.pre_amend.py` for the audit trail.

Decision is made only when every planned task has finished for every arm (or is `exhausted` by infra errors), at least 41 of 43
pairs are complete, and none is partial. Before that the output is `CONTINUE` or `STOP <arm>`.

| situation (all planned pairs finished) | output |
|---|---|
| lean-nt eligible | `SUBMIT lean-nt` |
| lean-nt not eligible, decide-nt eligible | `SUBMIT decide-nt` |
| an arm fails only because preflight is not `pass` | `HOLD (...)` |
| neither eligible | `SUBMIT neither` = keep the current submission (edit-early-v6-locate); it means "not shown to be better", not "shown worse" |
| a stop rule fired for an arm | that arm is not eligible; `STOP <arm>` is shown while the run is still incomplete |

## 7. Power statement (computed from `research/teacher-gen/C-behaviour/paired_results.json`, 144 pairs of base vs LoRA adapter)

Per-run completion tokens: mean 6,731, SD 2,916, between-arm correlation 0.779, SD of paired difference 1,918 (relative SE of the
ratio of means at n = 43: 4.4 percent). Minimum detectable effect at 80 percent power, two-sided 5 percent, 43 pairs: **12.3 percent**
(6.7 percent at 144 pairs). Analytic probability that the token gate fires, by true token change (Monte Carlo of the exact
bootstrap rule agrees within 0.10 in `test_power_statement_matches_monte_carlo`):

| true change | lean gate (<= -8%, CI hi < 0), corr .78 | corr .60 | corr .40 | decide gate (<= -5%, CI hi < +3%), corr .78 | corr .40 |
|---|---|---|---|---|---|
| 0 % | 0.02 | 0.02 | 0.02 | 0.10 | 0.06 |
| -5 % | 0.21 | 0.13 | 0.10 | 0.44 | 0.20 |
| -8 % | 0.44 | 0.27 | 0.20 | 0.71 | 0.33 |
| -10 % | 0.62 | 0.39 | 0.28 | 0.84 | 0.44 |
| -12 % | 0.78 | 0.53 | 0.38 | 0.93 | 0.55 |
| -15 % | 0.93 | 0.72 | 0.55 | 0.98 | 0.70 |
| -20 % | 1.00 | 0.92 | 0.79 | 1.00 | 0.89 |

These assume the correlation observed for two arms with the same prompt. A prompt/budget change lowers the correlation, so the
columns for .6 and .4 are the realistic ones. The lean gate needs an observed fall of about 9 to 14 percent (the CI condition binds, not
the 8 percent threshold), the decide gate about 6 to 11 percent.

## 8. Problems found in the design as specified, and what was done

1. **The primary metric is time-censored.** In real data about half the runs hit the 5.5 min cap (24/43 for edit-early-v4). A run that
   hits the cap generates roughly (cap - overhead) / latency tokens whatever the arm, so a pure per-call token cut makes the agent
   fit more calls into the same time and **total tokens fall much less than tokens per call** (executable demonstration:
   `test_time_cap_censors_total_tokens_but_not_tokens_per_call`; magnitudes in section 9). Total tokens per run stays the pre-registered
   primary and the gate metric, but the report always adds tokens per call, LLM calls, tokens/seconds to first edit, tokens on
   non-timeout pairs and the censoring-aware time tests, and the decision text cites them. **Recommendation (not applied, needs the
   owner's decision before data): gate lean on tokens-to-first-edit or on timeouts/resolved rather than on total tokens.**
2. **M1 is not measurable from ATIF** (no reasoning-token field). Replaced by the proxy in section 4; switches to the real
   quantity automatically when traces carry it.
3. **"M1 or M2" is nearly vacuous for lean** (M1 holds by construction when the config is applied). Kept as specified; the token
   and guard conditions carry the weight. For decide only M2 can apply, so "both M1 and M2 fail" (futility) is defined as "M2 fails".
4. **"M2: median falls >= 2" and "reads fall >= 25 percent" were not combined explicitly.** Interpreted as AND on pairs with an
   edit in both arms; "median" as the median of paired differences; "locate" as the first `locate.py` call.
5. **Gate comparator** of "paired tokens fall" was not stated: lean and decide are each compared with control.
6. **Guards have poor operating characteristics.** `resolved >= control - 1` and `timeouts <= control` are strict point-count rules; with
   about 14 resolved and 12 discordant pairs they reject an arm with no true difference in 31 to 37 percent (resolved) and about 38 percent (timeouts, when ~45% of runs time out) of experiments
   (section 9). Kept as pre-registered (they are conservative safety rules); expect a real win to be lost to them often.
7. **Config-error rule cannot be absolute** because control (budget 4096 = cap 4096) also truncates. Defined as hard errors in any
   treated run plus soft markers in treated runs where the paired control run has none.
8. **Census of what the stop rule can see:** soft markers require traces; without traces only hard errors in row text count.
9. **Missing pairs / retries were unspecified.** Defined in section 2 (complete-case; exhausted infra pairs excluded, not imputed;
   eligibility needs >= 41 complete pairs and nothing partial).
10. **Traces of earlier sessions are lost** when the next lb-eval version is pushed (kernel output is per version). Extract after every session.
11. **Ckpt reuse hazard:** the kernel skips any (arm, task) already in the checkpoint dataset, so old rows of an arm with the same name
    would silently become "this" arm's data. The checkpoint currently holds only edit-early-v4 and triage-first rows (no control rows);
    keep it that way or pass `--launch-ids`.
12. **Fixed arm order would confound order with treatment**; the 3-arm alternation of the original kernel only swaps control and lean
    (decide always second). The audit copy's six-permutation rotation fixes this; the analysis refuses to claim an order estimate if
    positions do not vary and flags a linear order effect otherwise (nonlinear carry-over is not modelled).
13. **Power** (section 7): with 43 pairs the lean gate (8 percent) sits below the minimum detectable effect, so "SUBMIT neither" is the
    most likely outcome unless the true effect is >= about 12 percent; a failed gate is not evidence against the arm.

## 9. Operating characteristics of the decision rule (synthetic worlds, `oc_sim.py`, 100 simulated experiments each)

| scenario | mean lean token change | token gate | M1 or M2 | resolved guard | timeout guard | all gates (preflight pass) | SUBMIT lean | SUBMIT decide |
|---|---|---|---|---|---|---|---|---|
| no true effect (lean per-call clip only), no cap | -4% | 0.06 | 0.08 | 0.63 | 1.00 | 0.01 | 0.01 | 0.00 |
| no true effect, realistic cap (control timeouts 45%) | +1% | 0.02 | 0.08 | 0.69 | 0.62 | 0.00 | 0.00 | 0.00 |
| lean: per-call tokens -30%, first edit 3 calls earlier; no cap | -42% | 1.00 | 1.00 | 0.51 | 1.00 | 0.51 | 0.51 | 0.05 |
| same, realistic cap | -25% | 1.00 | 1.00 | 0.60 | 1.00 | 0.60 | 0.60 | 0.03 |
| lean: per-call tokens -15%, first edit 2 earlier; realistic cap | -11% | 0.80 | 0.62 | 0.66 | 0.99 | 0.31 | 0.31 | 0.00 |

Each cell is the share of 100 simulated 43-pair experiments (3 arms, rotated order, n_boot 400) in which the component passes. Reading:

* False SUBMIT under no effect: 0 to 1 percent (the 120-run test `test_gate_passes_on_clear_win_and_false_submit_rate_under_null` asserts <= 5 percent).
* **The two safety guards are the dominant loss.** With no true difference the resolved guard (>= control - 1) fails in 31 to 37 percent of experiments and the
  timeout guard (<= control) in 38 percent when about 45 percent of runs time out. Even a 25 to 42 percent real token cut is submitted only 51 to 60 percent of the time;
  a 15 percent per-call cut only 31 percent. These guards are pre-registered as specified; the numbers are here so that a "SUBMIT neither" is read correctly.
* A time cap attenuates the total-token effect: a 30 percent per-call cut with the first edit 3 calls earlier shows -42 percent total tokens without a cap and -25 percent when
  about 45 percent of runs hit the cap (in the extreme case of a 45 s cap, tests show total tokens fall only about 8 to 19 percent for a 50 percent per-call cut while calls rise 30 to 50 percent).
* The decide gate rarely fires here because the simulated decide arm differs from control only in first-edit position (no token cut); its 3 to 5 percent submissions
  are token-gate passes of the permissive +3 percent upper bound combined with M2.

The simulator is a model of the mechanisms, not of the real agents: use it for the rule's behaviour, not as a prediction of the A/B result.

