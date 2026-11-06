# PREREG: Does the pre-fix code graph predict localisation and repair difficulty beyond cheap lexical features? (Candidate A)

Written 2026-10-10, BEFORE any model was fitted and BEFORE any per-task graph feature was computed or looked at.
The sha256 of this file is stored in `PREREG.sha256` (sidecar, same folder) before the extraction kernel is pushed.
Any change after that hash is a deviation and must be listed under "Deviations" in REPORT.md.

What the authors had already seen when writing this: aggregate localisation tables of locate v6 (`research/locate-improve/tables_synth.md`:
func@5 46.5% on the 815 train tasks, 37.5% on the 144 held-out), the aggregate agent resolve rates (ab1 base 70/144, adapter 72/144,
pilot1 62/99), and the IDEATION.md kill criterion. No per-task rank, no per-task graph quantity and no relationship between any
graph quantity and any outcome has been seen.

## 1. Question and design in one paragraph

On 959 exactly-labelled injected-fault tasks (25 permissive repos), do code-graph features of the faulty function (computed on the
pre-fix start state; the failing test and the issue statement are the only task inputs they use) add predictive power for (i) whether
a standard localiser ranks the faulty function in its top-k and (ii) whether Gemma 4 31B resolved the task, beyond lexical/style
features, under leave-one-repo-out validation? And does localisation difficulty increase monotonically with call-graph distance after
adjusting for lexical features?

Scope statement (stated up front because it limits the claim): every feature, graph or lexical, is conditioned on the fault LOCATION
(the mutated function is the label and is also the node the features are computed for). The predictor is therefore a
*location-conditioned difficulty estimator* (what a benchmark builder or an auditor who knows where the bug is could compute without
the patch contents), not a bug locator. Features never use the gold patch text, the operator-applied code diff, or the mutation
details beyond the function identity and the operator label (operator is listed explicitly as a lexical/syntactic covariate because
the factory records it and baselines should be able to use it). Distance is an observational dose, not a randomised one: the factory
chose mutation sites by operator round-robin among covered sites, not by graph position.

## 2. Units, data, exclusions

* Unit: task = `instance_id` (959). Dataset `<kaggle-owner>/gemma4-synth-tasks` (private; tasks.jsonl, meta.jsonl, snapshots).
  Repositories: 25 (15 MIT, 8 BSD-3, 2 BSD-2). Clusters for inference: repository (25) and, for agent targets with several runs per
  task, task.
* Start state S(task) = base snapshot of the repo + `snapshots/patches/<iid>.patch` (bug applied, failing tests removed): exactly what
  the agent sees. Base snapshot B(repo) = pristine tree with all tests; used ONLY to read the failing test bodies.
* Mutated function(s): `meta.funcs` (`file::qualname`); `kind` = single (794) or pair (165). Pair tasks keep both functions.
* Exclusions: none by design. A task whose mutated function is not found in the localiser's index of S gets rank = missing (counted as
  a miss for hit@k, as in `bench.py`) and graph features of "not found" (flag `gold_missing`); the number is reported. If more than 5%
  of tasks are gold_missing the report says so and repeats the primary analysis without them as a sensitivity.
* Split flag (train/heldout of `splits.json`) is NOT a model feature. v6 weights were fitted on the train split (in-sample localiser
  performance there); reported as a stratified sensitivity (heldout-only n=144, train-only n=815).

## 3. Targets

Localiser query = first 60 words of the problem statement (`bench.query_of`; what the agent passes to locate.py), run on S.

| id | target | definition | n |
|---|---|---|---|
| **T1 (primary)** | v6 func@5 | locate v6 final config (`SWE_LOCATE_OFF=""`, file `edit-early-v6-locate/skills/swe-tools/scripts/locate.py`), de-duplicated definition ranking (`dedupe_top(final, 40)`), hit iff the best rank over the task's mutated functions (bench.py `func_rank_any`) is <= 5 | 959 |
| T2 | v6 func@1 | same, rank <= 1 | 959 |
| T3 | BM25 func@5 | locate's own BM25F-lite alone (`bm25_scores` over the definition chunks, terms from `extract_terms`), de-duplicated by (path, class, name), best gold rank <= 5 | 959 |
| T4 | BM25 func@1 | same, rank <= 1 | 959 |
| T5 | v6 file@3 | some gold file in the top-3 distinct files of the v6 definition ranking | 959 |
| **A1 (agent primary)** | resolved, base arm | `research/teacher-gen/C-behaviour/per_run_table.jsonl` rows `src=ab1, arm=base` (budget 5.5 min, 60 calls, Gemma 4 31B, held-out split), `resolved` true/false, one run per task | 144 |
| A2 | resolved, adapter arm | rows `src=ab1, arm=adapter` | 144 |
| A3 | resolved, pilot1 | rows `src=pilot1, arm=harvest` (budget 10 min, train tasks) | 99 |
| A4 | resolved, pooled | A1+A2+A3 stacked, covariate `run_set` (3 levels), cluster bootstrap resampling TASKS (a task appears in 1 to 2 rows) and repos | 387 rows, 243 tasks |

Timeouts count as not resolved (as recorded). No run is dropped or re-scored.

## 4. Features (one table; all computed on S unless "B")

Call graph G (the primary graph, built by our own `ast`-based builder in the kernel, justified in section 4.1):
nodes = every `FunctionDef`/`AsyncFunctionDef` (any nesting depth) in the python files of S, test files included (test nodes flagged),
identified by (path, qualified name); a `ClassDef` is not a target node but a call `C(...)` resolves to `C.__init__`
(or `C.__call__` if there is no `__init__`) of the same class. Directed edge caller -> callee for every `ast.Call` found anywhere in
the caller body, `await` expressions and nested lambdas/comprehensions included (async handled; nested defs belong to their own node, not
to the enclosing one). Resolution order per call: (1) bare name -> enclosing-scope nested def -> module-level def of the same file ->
`from x import y [as z]` / `import x [as y]` to a repo module -> class; (2) `self.m` / `cls.m` -> method of the enclosing class or of its
repo-resolved base classes (by name, depth-first); (3) `mod.f` with `mod` an imported repo module alias; (4) `Class.m`; (5) otherwise
`obj.m(...)`: name fallback, edge to every non-test def whose bare name is `m` if there are at most 3 such defs (else no edge).
Relative imports are resolved against the file path; star imports are ignored. Decorators, callbacks passed as arguments and
dynamic dispatch beyond (5) are not modelled (limitation).
`dist(u, v)` = length of the shortest directed path (number of edges, 0 if u = v); `dist_und` ignores direction. Unreachable and
undefined distances are encoded as value 7 plus indicator `*_na` (both enter the models); distances above 6 are clipped to 6.

For tasks with two mutated functions: distance features take the MINIMUM over the mutated functions; node-level numeric features
(fan-in, ...) take the MEAN of the log1p values; flags take OR. (Rule fixed here, independent of outcomes.)

| group | name | exact definition |
|---|---|---|
| **L0 lexical/style** | `stmt_chars`, `stmt_words` | log1p of characters / whitespace words of the full problem statement |
| | `style_obs`, `style_repro` | one-hot of `meta.style` (reference: terse); `terse` = style == terse |
| | `name_in_stmt` | 1 iff the bare name of a mutated function (case-insensitive, whole identifier token) occurs as an identifier token in the full statement |
| | `name_overlap` | share of the mutated function's name sub-tokens (`locate.name_parts`, length >= 2, not in locate STOP/WEAK) that appear among the statement's sub-tokens |
| | `sym_eq_name` | 1 iff the last component of `meta.symbol` (the package symbol named by the failing test's failure view) equals the bare mutated name |
| | `bm25_rel`, `bm25_margin` | with s_g = best BM25F-lite score among mutated-function definitions, s_t = best score overall, s_o = best score among non-mutated definitions: `bm25_rel` = s_g / s_t (0 if s_t = 0), `bm25_margin` = (s_g - s_o) / s_t |
| **L1 = L0 + operator/size** | `op_<name>` (18) | multi-hot of `meta.ops` (18 operators of `synth/mutators.py`) |
| | `pair`, `n_gold` | kind == pair; number of mutated functions |
| | `func_len`, `file_len` | log1p of lines of the mutated function / of its file (mean over mutated) |
| | `repo_n_defs`, `repo_n_files` | log of the number of indexed definitions / python source files of the repository (repository descriptor usable on an unseen repo; repository identity itself cannot be a feature under leave-one-repo-out) |
| | `public`, `has_doc` | `meta.public`, `meta.has_doc` |
| **G graph (Gs structure + Gt test evidence)** | `d_test` (Gt) | min over the task's failing test functions (FAIL_TO_PASS node ids -> test def in B, parameter ids stripped, matched also to the same def in S's test files) of dist(test node, mutated node) in G; 1 = the test calls it directly |
| | `d_sym` (Gt) | candidates = non-test defs whose bare name equals the last component of `meta.symbol` (if it has a dotted qualifier, only those whose qualname ends with it); <= 8 candidates else NA; min of dist(candidate, mutated) |
| | `d_sym_und` | same with undirected distance |
| | `d_stmt` | candidates = non-test defs whose bare name equals an identifier token of the full statement (stop-listed with locate STOP/WEAK, Python keywords and length < 3; skipping names with more than 4 homonym defs, locate's own rule); min over candidates of dist(candidate, mutated); also `n_stmt_cand` (log1p) |
| | `fan_in`, `fan_out` (Gs) | log1p number of distinct non-test callers / callees of the mutated node |
| | `fan_in_test` (Gt) | log1p number of distinct test functions that call the mutated node directly |
| | `anc`, `desc` (Gs) | log1p number of transitive callers (non-test) / transitive callees of the mutated node |
| | `pagerank` (Gs) | PageRank (damping 0.85, 60 iterations, dangling mass uniform) on the non-test subgraph, value x N, log |
| | `n_tests_static` (Gt) | log1p number of test functions with a directed path of length <= 3 to the mutated node |
| | `n_cover_dyn` (Gt) | log1p number of PASSING tests that covered a mutated line at baseline (factory screen log `n_cover`, joined on repo slug, file, qualname, line); `n_cover_na` when the join fails |
| | `n_f2p` (Gt) | log of the number of failing tests (FAIL_TO_PASS) |
| | `module_depth` (Gs) | number of path separators of the mutated file; `is_method`, `nest_depth` = qualname dot count |

Not used in any set: gold patch size/hunks (`gold_lines_changed`, post-fix knowledge) and the mutation `details`. `fail_exc_types` and
`fail_dur_s` are observable pre-fix but are kept out of the pre-specified sets; adding them is exploratory analysis E4 only.

Feature sets: **L0**, **L1**, **G**, **Full = L1 + G** (also **L1+Gs** and **L1+Gt** for decomposition). `bm25_rel`/`bm25_margin` are
removed from every set for targets T3 and T4 (they are near-deterministic functions of the BM25 rank; the sets are then called
L0\margin, L1\margin, Full\margin). Missing numeric values are imputed with the training-fold median and flagged.

### 4.1 Why our own AST graph (and not the REP graph or locate's name graph)

* The REP builder (`datasets/gemma4-paper-code/build_graph.py`) emits the organisers' schema and is tuned to reproduce node ids of the
  competition graphs; it needs no schema here and has not been validated on the 25 permissive repos. Its node set is the same
  (defs, async included, directed calls), so using it would not change which statements are testable.
* locate v6's name-based call map (`callee_ids`) is the very mechanism the v6 score uses (`gcall`), so graph features computed on it
  would partly re-express the localiser's own score for T1. The primary graph is therefore built independently from `ast` with import and
  `self.` resolution (above). The name-based graph is used as a robustness check (sensitivity S3) for the distance features.
* Graph choice is fixed here and is not tuned on outcomes.

## 5. Models, validation, statistic

* Models: (M1, primary) L2-regularised logistic regression on standardised features (sklearn `LogisticRegression`, lbfgs, `C` chosen
  per outer fold by inner 5-fold GroupKFold over the training repos on log loss from {0.03, 0.1, 0.3, 1, 3}); (M2, secondary)
  sklearn `GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, min_samples_leaf=20,
  random_state=0)`, no tuning.
* Validation: leave-one-repository-out over the 25 repos for T1..T5. Out-of-fold predictions are pooled; **pooled AUC** is the statistic.
  For A1/A2/A3 the folds are leave-one-repo-out over the repos present in that run set; for A4 repos again, rows of one task stay together.
* Uncertainty: two-stage cluster bootstrap (resample repos with replacement, then tasks within the drawn repos with replacement),
  2000 resamples, seed 20261010, percentile 95% CI; the same resamples score every feature set, so gains are paired.
* Primary decision statistic: **gain = AUC(M1, Full) - AUC(M1, L1) on T1**, with its 95% CI.
* **KILL CRITERION (fixed now)**: the study is declared a NULL (negative result written up, "graph position matters beyond lexical/style"
  claim abandoned) if the gain on T1 is < 0.03 (point estimate < 0.03) OR the CI upper bound is < 0.03. It is declared
  SUPPORTED only if the point estimate >= 0.03 AND the CI lower bound > 0. Anything else (point >= 0.03, lower bound <= 0) is
  "INCONCLUSIVE, leaning alive". There is no other route to a positive claim: the choice of target, model, feature set or threshold is
  not revisited after seeing results.
* Interpretation guard (declared now, does not alter the rule above): if T1 is SUPPORTED but the same gain on T3 (BM25-only, margin
  removed from both sets) is < 0.03, the verdict is "alive only through the localiser's own graph propagation (mechanical), not through
  structure carrying information beyond lexical retrieval". If M2 gains >= 0.03 with lower bound > 0 while M1 does not, it is reported
  as "non-linear signal, exploratory" and the verdict remains NULL by the primary rule.
* Sample size / precision: T1 has 959 tasks in 25 clusters (expected positive rate about 40%); the cluster bootstrap CI of an AUC gain
  is expected to be about +-0.02 to +-0.04. A1 has 144 tasks (~49% resolved): only gains of roughly 0.10 or more are detectable;
  agent results are descriptive and cannot kill or support the study alone.

## 6. Secondary and exploratory analyses (labelled)

Secondary (pre-specified, family S = {gain Full vs L1 for T2, T3, T4, T5, A1}; M1; Holm correction at family-wise 0.05 using the
bootstrap distribution of each gain (one-sided p = share of resamples with gain <= 0, doubled for two-sided); report adjusted p and CI):

* S-a: all five gains above. S-b: the same with M2. S-c: L1 vs L0 gain (T1) to show how much operator/size alone adds.
* S1 stratified: T1 gain within style (terse/observed/repro), within heldout/train, within single/pair.
* S2 decomposition: L1+Gs and L1+Gt gains; G alone vs L1 alone AUC.
* S3 graph robustness: recompute `d_test, d_sym, d_stmt, fan_in, fan_out` on locate's name-based call map and repeat T1 gain.

Exploratory (no multiplicity control; reported with BH-adjusted q over the exploratory list, flagged "exploratory"):

* E1 dose-response: hit rate of T1/T3 by `d_test`, `d_sym`, `d_stmt` bins (0, 1, 2, 3, 4+, unreachable/NA). Matched version:
  coarsened exact matching strata = repository x style x `name_in_stmt` x tertile of `stmt_words` (strata with only one distance bin
  dropped); fixed-effects logistic regression of the hit on strata dummies + ordinal distance (reachable bins) giving log-odds per hop,
  95% CI by cluster bootstrap over repos; g-computation (standardised) hit rate per bin from a logistic model on L1 + distance bins;
  monotonic if the slope is negative with CI excluding 0 and no adjacent-bin reversal larger than its bootstrap SE.
* E2 identifier-neutralising counterfactual: re-run v6 and BM25 on S with (C1) every occurrence of the bare mutated function name in the
  60-word query replaced by the neutral token `zzobj`; (C2) the same for the last component of `meta.symbol`; (C12) both. Measured:
  hit@5 loss (original - counterfactual) overall, among tasks whose query contained the name (`cf_changed`), and by `d_test` / `d_sym`
  bin; paired bootstrap CI over tasks. Test bodies are not editable (they are hidden from the agent), so the counterfactual is on
  the statement/query only; it removes the exact identifier but not its sub-tokens appearing elsewhere.
* E3 A2, A3, A4 gains; partial dependence of M2 on `d_test`, `fan_in`, `n_tests_static`.
* E4 add `fail_exc_types`-derived flags and `fail_dur_s` to L1 and repeat T1.

## 7. Leakage and confound checks (done and reported, whatever the outcome)

1. Label leakage: no feature reads the gold patch text or `details`; features are computed on S (bug applied) and B tests only.
   The mutation can change at most the callee set of the mutated function (e.g. `remove_stmt` dropping a call): reported by comparing
   fan_out on S with fan_out on B (pristine) for all tasks; the distance features towards the mutated node are unaffected by its own
   outgoing edges.
2. Issue symbol observable pre-fix: `meta.symbol` is produced by `statements.py` from the failing test's failure view (traceback
   frame or test body names), module part stripped; reported: share of tasks where `meta.symbol` occurs verbatim in the statement
   (repro/observed) versus only in the failing test (terse).
3. Tautology: BM25 margin features are near-deterministic for BM25 targets (handled by removal, T3/T4); for T1 they remain in L0/L1 as
   the conservative baseline (a stronger baseline can only make the gain smaller).
4. Repository confounding: leave-one-repo-out, repository identity not a feature, repository descriptors only; per-repo AUC
   distribution reported.
5. Same-task multiple arms: A4 clusters by task; A1..A3 are single-run-per-task.
6. Selection by the factory: sites were required to be covered by 1..250 passing tests and screened to fail 1..8 tests, so
   `n_cover_dyn` and `n_f2p` are range-restricted; mutated functions appear at most twice (clusters nest in repo).
7. In-sample localiser: v6 weights were fitted on the 815 train tasks; hit rates are optimistic there. Stratified sensitivity S1.

## 8. Outputs promised

`REPORT.md` (verdict first: NULL / SUPPORTED / INCONCLUSIVE per the rule), tables of AUC with CIs per feature set and target, gains
table, dose-response table/figure, counterfactual table, leakage checks, per-task feature/outcome table without source text
(`features_tasks.jsonl`), extraction kernel source and analysis scripts. A negative result is written up with the same completeness.
