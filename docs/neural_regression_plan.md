# Inter-Regional Neural Regression Implementation Plan

**Status:** Active phased implementation plan. The user authorized implementation in a later chat;
the live snapshot and dated package records below define the completed and currently authorized
scope. Experimental-data and batch runs still require their separately documented gates.

**Scientific authority:** `docs/spec_neural_regression_v3.md`.

## Live handoff snapshot

**Snapshot date:** 2026-10-08 14:55 EDT.

**Current phase:** WP1-WP8 are complete and GREEN. The standard linear workflow now includes
validated contracts, Pynapple count preparation, deterministic block CV, direct-unit OLS,
training-only fold-local regional PCA, bidirectional PC OLS, explicit unavailable PC ranks, and
complete-fold summaries, immutable run persistence, single-session/batch command boundaries,
standard saved-result PNGs, and a metadata-driven read-only webapp view. Its seeded full synthetic
integration, bounded performance gate, corrected CT026 inspection, and user acceptance are recorded
below. WP9 unit-count Poisson CV, matched OLS/Poisson MSE, saved figures/reporting, and read-only
webapp presentation are code-complete and GREEN on seeded synthetic data. The designated CT026
Poisson run `30989197` was OOM-killed after approximately 2 hours 44 minutes with a 32-GiB
allocation. Reduced execution telemetry is implemented and GREEN. The separately approved bounded
benchmark, Slurm job `30991145`, reproduced the OOM within the first Poisson cell after OLS completed;
the approved minimal change from Statsmodels' default least-squares backend to its QR backend within
the same unpenalized IRLS estimator is implemented and locally GREEN, but the separately approved
repeat bounded benchmark, Slurm job `30991248`, was also OOM-killed. QR slowed rather than bounded
the memory growth. No replacement, scale test, or full retry is authorized. WP9 scientific
acceptance and WP10 remain blocked on an inspected completed output.

**Repository state at this snapshot:**

- branch: `refactor`;
- HEAD before this handoff record: `54fc080`;
- plan-owned files: `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`; only the plan is currently modified;
- the worktree also contains many unrelated pre-existing untracked files/directories; and
- none of those unrelated entries belongs to this plan or may be staged, changed, removed, or
  absorbed into a later package.

**Completed planning evidence:**

- `src/neural_analysis`, `docs/SoftwareDesign.md`, and
  `docs/spec_neural_regression_updated.md` were audited;
- codebase/specification inconsistencies were identified and resolved with the user;
- v3 records the approved scientific and first-pass data-validity decisions;
- the current plan contains module responsibilities, explicit array/result contracts, a phased
  tests-first inventory, performance considerations, and reproducible run outputs; and
- the implementation-readiness correction froze the JSON configuration, scientific/CV eligibility,
  count/PCA determinism, result/status schema, numerical tolerances, atomic run identity,
  read-only webapp boundary, and advisory batch-memory policy;
- the second readiness review removed staged-schema and runtime-dependency contradictions, defined
  paired MSE comparison/aggregation, moved run identity out of the pure result, and removed a
  redundant package-gate layer; and
- the third review clarified scientific/condition/CV eligibility, canonical row identity,
  descriptive PCA scope, and minimal Poisson diagnostics; added runtime-version provenance; and
  replaced the run-state protocol with a same-parent incomplete-directory/final-rename boundary;
- the empirical binning review found no default-window overlap in either available CT026 session,
  so the frozen count contract retains the existing Pynapple tensor convention without a custom
  NumPy counting path or cross-fold overlap machinery; and
- existing repository plans were inspected for their Sol/Terra, interruption, and authoritative
  handoff patterns before this revision.

**Next exact action:** prepare and obtain approval for a new tests-first fitting plan that actually
bounds per-target optimizer memory while preserving the unpenalized Poisson/log scientific model.
Do not implement it or launch another run, scale test, full retry, or WP10 without approval.

### Authority order

When resuming, use this order:

1. `docs/spec_neural_regression_v3.md` owns scientific definitions, defaults, metrics, and
   interpretation limits.
2. This plan owns project architecture, dependency direction, file ownership, test inventory,
   work-package order, Sol/Terra assignments, acceptance gates, and live execution state.
3. `AGENTS.md` and `docs/SoftwareDesign.md` govern approval, TDD, commits, readability, data
   contracts, dependencies, filesystem discipline, and scientific-analysis workflow.
4. The top live snapshot and dated package records in this document own current authorization and
   progress. A chat summary or worker report is not a substitute.
5. Git history, the inspected worktree, and independently reproduced command output are evidence.
   Agent agreement alone is not.

If these authorities conflict, stop before editing and obtain a user decision. Do not silently
reinterpret the scientific specification to fit an implementation convenience.

### Work-package state

| Package | Scope | State at snapshot | Next gate |
|---|---|---|---|
| WP0 | Specification, architecture, orchestration, and handoff plan | Contract freeze complete; documentation only | User review; no implementation in this chat |
| WP1 | Configuration and result contracts | Complete | WP2 may begin |
| WP2 | Counts, masks, folds, windows, and histories | Complete | WP3 may begin |
| WP3 | Direct-unit OLS fitting, scoring, and aggregation | Complete | WP4 may begin |
| WP4 | Fold-local regional PCA and PC OLS | Complete | WP5 may begin after authorization |
| WP5 | Saved results, run identity, session runner, and batch runner | Complete | WP6 complete |
| WP6 | Standard-regression plotting, metadata webapp, and documentation | Complete | WP7 may begin after authorization |
| WP7 | Standard-regression synthetic integration and bounded performance check | Complete | WP8 may begin only after its separate approval |
| WP8 | One-session standard-regression scientific inspection | Complete and user-accepted | WP9 may proceed |
| WP9 | Unit Poisson CV, MSE comparison, plotting, and integration | QR benchmark also OOM-killed; fitting approach requires replanning | Approve a genuinely memory-bounded optimizer before another run |
| WP10 | Linear and Poisson descriptive Granger analyses | Not authorized | WP9 GREEN and inspected Poisson output |
| WP11 | Final synthetic integration, documentation, and one-session full inspection | Not authorized | WP10 GREEN; real-session command separately approved |

### Resume checklist

A new or returning Sol supervisor must:

1. read the complete v3 specification, this complete plan, `AGENTS.md`,
   `docs/SoftwareDesign.md`, and the most recent dated package record;
2. record current model/effort availability, branch, HEAD, `git status --short`, staged diff, and
   unstaged diff without changing them;
3. identify package-owned files and preserve all user/pre-existing changes; if edit ownership is
   uncertain, stop for user direction;
4. confirm the latest explicit user authorization, especially whether source/test edits, real-data
   inspection, batch execution, or external actions are allowed;
5. reproduce the last claimed RED or GREEN command before treating it as a completed gate;
6. verify every required preceding test-only and implementation commit from Git history;
7. resume at the first incomplete gate in the state table, not at the beginning of a completed
   package and not at a later convenient package;
8. update the live snapshot before assigning a worker or editing a package-owned file; and
9. keep this plan authoritative by appending a dated gate record before moving to another package.

An interrupted worker or reviewer report is never a completed gate. The supervisor first audits
the bounded diff and reproduces the last claimed command. Unknown edits are preserved and escalated;
they are never reset, overwritten, or silently included.

### Live update protocol

Update this snapshot and append a dated record after every:

- user authorization or scope decision;
- documentation baseline or architecture change;
- tests-only RED commit;
- implementation GREEN commit;
- independent review gate;
- interruption, worker replacement, or ownership discrepancy;
- synthetic benchmark or separately authorized real-session run; and
- package completion or blocked decision.

Each record must contain:

- timestamp, package, state, and authorization;
- lead Sol, Terra worker, and independent reviewer model/effort;
- starting and ending HEAD plus worktree ownership state;
- exact files inspected and changed;
- exact RED, GREEN, regression, or benchmark commands and outcomes;
- test-only and implementation commit IDs, plus any separate corrective documentation commit;
- scientific/configuration decisions and unresolved risks;
- real-data, filesystem, or external actions taken;
- output/run paths and measured performance when authorized; and
- one exact next action with its required authorization.

Do not rewrite older evidence to make later work appear cleaner. Append a dated correction when a
prior record is wrong.

Use this template:

```text
#### YYYY-MM-DD HH:MM TZ - WPx gate
- State:
- Authorization:
- Sol / Terra / reviewer:
- Start HEAD / end HEAD:
- Worktree and owned files:
- RED command and result:
- GREEN/regression commands and results:
- Commits:
- Real-data, filesystem, or external actions:
- Findings and unresolved risks:
- Exact next action and authorization:
```

### Live update record

#### 2026-10-06 14:52 EDT - WP0 architecture and handoff revision

- State: documentation revision active; implementation not started.
- Authorization: documentation-only; the user explicitly prohibited implementation in this chat.
- Sol / Terra / reviewer: primary planning agent only; no Terra worker or independent reviewer.
- Start HEAD / end HEAD: `6131d8e` / `6131d8e`.
- Worktree and owned files: only `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md` are owned; unrelated pre-existing untracked entries were left
  untouched.
- RED command and result: not applicable; no tests were created or run.
- GREEN/regression commands and results: not applicable; documentation structure/ASCII checks only.
- Commits: none.
- Real-data, filesystem, or external actions: no experimental data, benchmark, network, or external
  action; only the two documentation files were edited.
- Findings and unresolved risks: project architecture and Sol/Terra/handoff detail required this
  revision; plot theme remains a later user-facing choice, with light mode planned by default.
- Exact next action and authorization: return the revised plan for user review; implementation
  requires a later explicit request in a new or resumed chat.

#### 2026-10-06 16:36 EDT - WP0 implementation-readiness correction

- State: documentation contracts frozen; WP0 awaits user review and remains implementation-
  inactive.
- Authorization: documentation changes only; no source, tests, commits, or analysis runs.
- Sol / Terra / reviewer: primary documentation agent only; no worker or independent reviewer.
- Start HEAD / end HEAD: `6131d8e` / `6131d8e`.
- Worktree and owned files: changed only `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`; both are tracked plan-owned files. This corrects the earlier
  14:52 record's mistaken `untracked` label; `git ls-files --stage` confirms both were already
  tracked. The unrelated modified `docs/task_variable_implementation_plan.md` and existing untracked
  entries were not changed.
- RED command and result: not applicable; no tests were written or run.
- GREEN/regression commands and results: `git diff --check -- docs/neural_regression_plan.md
  docs/spec_neural_regression_v3.md` passed; `LC_ALL=C rg -n '[^ -~]'` over both files returned no
  matches; Markdown-fence counts were even (50 plan, 30 spec); targeted `rg` contradiction searches
  returned no stale compute-in-webapp, dual-result-tree, or weak file-identity language. No production
  verification is claimed.
- Commits: none.
- Real-data, filesystem, or external actions: no experimental-data, network, benchmark, batch, or
  external action. Only the two authorized Markdown files were edited; a local `uv run` inspection
  read installed library versions/signatures using a temporary cache under `/tmp`.
- Findings and unresolved risks: froze the exact configuration and defaults, unfiltered fold
  universe, local positional count tensor, deterministic PCA settings, saved-table/status/reason
  contracts, numerical tolerances, immutable atomic run state/fingerprint/rerun behavior,
  read-only completed-run webapp, and explicit batch memory estimate/cap. The deliberately strict
  direct-unit rank policy and implicit coverage assumption remain scientific limitations, not
  implementation gaps. Light-mode plotting remains the documented default.
- Exact next action and authorization: user reviews the corrected plan/spec. A later explicit
  implementation request authorizes only a fresh WP1 preflight; real-session and batch execution
  remain separately unauthorized.

#### 2026-10-06 17:13 EDT - WP0 correctness and conciseness revision

- State: documentation revision complete; implementation remains inactive.
- Authorization: documentation changes only in response to the user's readiness review; no source,
  tests, commits, analysis runs, or benchmarks.
- Sol / Terra / reviewer: primary documentation agent only; no worker or independent reviewer.
- Start HEAD / end HEAD: `787aadb` / `787aadb`.
- Worktree and owned files: changed only `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`; preserved the unrelated modified
  `docs/task_variable_implementation_plan.md` and pre-existing untracked entries.
- RED command and result: not applicable; no tests were written or run.
- GREEN/regression commands and results: `git diff --check -- docs/neural_regression_plan.md
  docs/spec_neural_regression_v3.md` passed; `LC_ALL=C rg -n '[^ -~]'` over both files found no
  non-ASCII text; Markdown-fence counts remained even (50 plan, 30 spec); stale-contract searches
  found no old scientific-fingerprint, automatic-memory-cap, combined-mask, absent-future-table, or
  Granger-runtime-dependency language. No production verification is claimed.
- Commits: none.
- Real-data, filesystem, or external actions: only the two authorized Markdown files were edited;
  no experimental data, network, benchmark, batch, or external action.
- Findings and unresolved risks: scientific and CV masks are now distinct; block IDs normalize
  integral numeric values; Granger stages are runtime-independent from CV; raw Poisson deviance is
  fold-only; paired MSE has an exact derived contract; run provenance is outside the pure result;
  dry-run avoids full-file hashing; memory estimates are advisory; future-stage tables are
  consistently present but empty. The first pass still assumes complete loaded data coverage.
- Exact next action and authorization: user reviews the revised documents. A later explicit
  implementation request authorizes a fresh WP1 preflight only.

#### 2026-10-06 17:53 EDT - WP0 third readiness review

- State: documentation revision complete; implementation remains inactive.
- Authorization: review and documentation corrections only; no source, tests, commits, experimental-
  data runs, or benchmarks.
- Sol / Terra / reviewer: primary documentation agent only; no worker or independent reviewer.
- Start HEAD / end HEAD: `787aadb` / `787aadb`.
- Worktree and owned files: changed only `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`. The user confirmed that concurrent changes to
  `docs/task_variable_implementation_plan.md` and `docs/task_variable_spec_v5.md` belong to another
  chat; they and all untracked entries were left untouched.
- RED command and result: not applicable; no tests were written or run.
- GREEN/regression commands and results: documentation-only checks passed:
  `git diff --check -- docs/neural_regression_plan.md docs/spec_neural_regression_v3.md`;
  `LC_ALL=C rg -n '[^ -~]'` returned no matches; Markdown fence counts were even (52 plan, 30
  spec); and stale current-contract searches returned no old run-state, scientific-base-mask,
  generating-functions, or next-valid-row language. A read-only
  `UV_CACHE_DIR=/tmp/context-inference-uv-cache uv run python -c ...` API probe confirmed
  statsmodels 0.15.0, Boolean `GLMResults.converged`, integer `fit_history["iteration"]`, and the
  expected convergence/perfect-separation warning/error classes. No production verification is
  claimed.
- Commits: none.
- Real-data, filesystem, or external actions: only the two authorized Markdown files were edited;
  no experimental data, network, batch, benchmark, or external action.
- Findings and unresolved risks: scientific eligibility, named-condition membership, and CV
  eligibility are now separate; PCA pool scope and whole-window bin identity are explicit;
  train/test row-set hashes make OLS/Poisson pairing auditable; Poisson diagnostics are limited to
  convergence, iterations, status, and reason; runtime versions enter run identity; and one hidden
  incomplete directory plus atomic final rename replaces the prior state/per-file transaction
  machinery. A final post-fix pass found no further correctness, completeness, or proportionality
  issue. The deliberate implicit-complete-coverage assumption remains clearly deferred work.
- Exact next action and authorization: user reviews the revised documents. A later explicit
  implementation request authorizes a fresh WP1 preflight only.

#### 2026-10-08 00:03 EDT - WP0 trial-local binning correction

- State: documentation correction complete; implementation remains inactive.
- Authorization: the user requested correction of the documentation after reviewing the identified
  Pynapple multi-epoch behavior; no source, tests, commits, experimental-data runs, or benchmarks.
- Sol / Terra / reviewer: primary documentation agent only; no worker or independent reviewer.
- Start HEAD / end HEAD: `9169c34` / `9169c34`.
- Worktree and owned files: changed only `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`; unrelated pre-existing untracked entries were left untouched.
- RED command and result: not applicable; no tests were written or run.
- GREEN/regression commands and results: documentation-only `git diff --check` and ASCII checks
  passed. No production verification is claimed.
- Commits: none.
- Real-data, filesystem, or external actions: only the two authorized Markdown files were edited;
  no experimental data, network, batch, benchmark, or external action.
- Findings and unresolved risks: a bulk Pynapple `IntervalSet` can merge overlapping trial windows,
  adjust touching endpoints, and sort epochs by time. The corrected contract uses per-trial NumPy
  `searchsorted` counting, retains ascending trial-row identity, counts a recorded spike once in
  each applicable overlapping trial-local window, and adds explicit overlap/order/boundary tests.
  The documents now also disclose that unpurged windows on opposite sides of a block boundary may
  reuse absolute-time spikes across CV folds. The deliberate implicit-complete-coverage assumption
  remains unchanged.
- Exact next action and authorization: user reviews the corrected documents. An explicit later
  implementation request authorizes a fresh WP1 preflight only.

#### 2026-10-08 00:11 EDT - WP0 Pynapple binning restoration

- State: documentation correction complete; implementation remains inactive.
- Authorization: after review of the practical overlap risk, the user explicitly directed the plan
  to retain Pynapple rather than introduce custom spike counting.
- Sol / Terra / reviewer: primary documentation agent only; no worker or independent reviewer.
- Start HEAD / end HEAD: `9169c34` / `9169c34`.
- Worktree and owned files: changed only `docs/neural_regression_plan.md`; restored
  `docs/spec_neural_regression_v3.md` to its tracked Pynapple contract. Unrelated pre-existing
  untracked entries were left untouched.
- RED command and result: not applicable; no tests were written or run.
- GREEN/regression commands and results: documentation-only structure, diff, and ASCII checks; no
  production verification is claimed.
- Commits: none.
- Real-data, filesystem, or external actions: read-only timing checks inspected the two available
  CT026 trial tables; no spike data were loaded and no outputs were written.
- Findings and unresolved risks: the two sessions had zero overlapping eligible windows at the
  default two- and four-second durations. Their minimum eligible alignment separations were
  4.020457 seconds and 4.006259 seconds for choice time, and 4.020491 seconds and 4.006413 seconds
  for start time. The custom NumPy counting path and overlap-specific warnings/tests were therefore
  removed, and the existing Pynapple tensor convention remains the implementation contract.
- Exact next action and authorization: user reviews the corrected documents. An explicit later
  implementation request authorizes a fresh WP1 preflight only.

#### 2026-10-08 00:17 EDT - WP1 implementation authorization and preflight

- State: WP1 active; preflight complete and tests-only RED work is next.
- Authorization: the user explicitly approved implementation after the Pynapple restoration. This
  activates WP1 configuration and result contracts only; real-data and batch execution remain
  unauthorized.
- Sol / Terra / reviewer: primary implementation agent; no worker or independent reviewer.
- Start HEAD / end HEAD: `9169c34` / `9169c34` at preflight.
- Worktree and owned files: the existing plan edit is owned by this task. WP1 owns the new
  `src/neural_analysis/interregional/` package files and the two planned interregional contract test
  files. Unrelated pre-existing untracked entries remain untouched.
- RED command and result: pending tests-only implementation.
- GREEN/regression commands and results: pending.
- Commits: pending tests-only and implementation commits.
- Real-data, filesystem, or external actions: none.
- Findings and unresolved risks: preflight found no additional implementation blocker. WP1 has no
  spike loading, fitting, plotting, persistence, or experimental-data access.
- Exact next action and authorization: write and run the WP1 configuration/record tests, confirm
  their expected import/API failures, and commit the tests before source implementation.

#### 2026-10-08 00:24 EDT - WP1 GREEN and WP2 handoff

- State: WP1 complete; WP2 preparation tests are authorized and next.
- Authorization: implementation remains within the user's approved plan. No real-data or batch
  execution is authorized.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `9169c34` / `a1b937b`.
- Worktree and owned files: added only the interregional configuration/record package and its two
  test files. The plan remains this task's only uncommitted tracked file; unrelated entries were
  untouched.
- RED command and result: the two new test modules initially failed collection because the package
  did not exist. Two focused contract additions then failed for NumPy integer/provenance behavior,
  and two input-type additions failed before their corresponding fixes.
- GREEN/regression commands and results: 30 focused interregional tests passed. A combined 121-test
  run covering those tests plus existing spike-behavior package, population package/PCA, and
  session-metadata tests passed with four existing Pynapple warnings. Two separately run quickstart
  tests fail on the pre-existing tracked `task_decoding_config.json` because that test globs every
  JSON example as session metadata; WP1 did not change that file or test.
- Commits: tests `7287bca`, `b7a95de`, and `691f31d`; test formatting `e8fb4db`;
  implementation `a1b937b`.
- Real-data, filesystem, or external actions: no experimental data, network, batch, or external
  action. Test files used pytest temporary directories only.
- Findings and unresolved risks: no WP1 contract issue remains. The quickstart glob failure is an
  unrelated pre-existing test/repository mismatch and was not changed.
- Exact next action and authorization: write WP2 preparation tests against the frozen contracts,
  confirm RED, and commit them before implementing the Pynapple preparation module.

#### 2026-10-08 00:28 EDT - WP2 GREEN and WP3 handoff

- State: WP2 complete; WP3 direct-unit OLS tests are authorized and next.
- Authorization: implementation remains within the approved plan; no real-data or batch execution.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `a1b937b` / `c76a6c7`.
- Worktree and owned files: added `preparation.py` and its test file only; the plan remains the only
  uncommitted tracked file owned by this task.
- RED command and result: preparation tests initially failed import because the module did not
  exist. After implementation, four failures identified three test-fixture errors: float-edge
  construction, zero-duration one-spike Pynapple support, and pandas index alignment.
- GREEN/regression commands and results: all nine preparation tests passed after the documented
  fixture corrections. A combined 94-test run covering all interregional tests plus existing
  population-PCA and spike-behavior package tests passed with four existing Pynapple warnings.
- Commits: tests `5fc6d16`; documented fixture correction `8de4b94`; implementation `c76a6c7`.
- Real-data, filesystem, or external actions: no experimental data, network, batch, or external
  action. All neural inputs were synthetic and hand checked.
- Findings and unresolved risks: the count implementation uses the existing Pynapple aligned tensor
  builder, preserves positional trial identity, and passes exact half-open-bin counts. No custom
  spike counter or overlap machinery was introduced.
- Exact next action and authorization: write and commit WP3 OLS and direct-unit pipeline tests,
  confirm RED, then implement only the direct-unit OLS path.

#### 2026-10-08 00:33 EDT - WP3 GREEN and direct-unit milestone

- State: WP3 and the initial direct-unit OLS milestone are complete; WP4 is ready but not started.
- Authorization: implementation remained within the approved plan; no real-data or batch execution.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `c76a6c7` / `1fd0fa6`.
- Worktree and owned files: added `linear.py`, `pipeline.py`, bounded package exports, and their two
  test files. The plan is the only remaining tracked edit before this handoff record is committed.
- RED command and result: linear and pipeline tests each initially failed import. Focused follow-up
  tests failed before fixes for full-only rank availability and metric-specific MSE aggregation.
- GREEN/regression commands and results: all 48 interregional tests passed. A combined 139-test run
  covering all interregional work plus existing spike-behavior package, population package/PCA,
  and session-metadata tests passed with four existing Pynapple warnings.
- Commits: OLS tests `7e889fb`; tolerance correction `7f40f2a`; pipeline tests `d14d08f`;
  nested-status test `2ca10de`; metric-specific-summary test `ea9795e`; implementation `1fd0fa6`.
- Real-data, filesystem, or external actions: no experimental data, network, batch, or external
  action. The integration input was generated with recorded NumPy seed 17.
- Findings and unresolved risks: direct-unit OLS is complete for the in-memory path. Strict rank and
  residual-degree rules are explicit; unavailable full models do not erase valid restricted-fit
  diagnostics; constant held-out targets retain MSE while R-squared remains unavailable. No new
  issue was found in the completed scope.
- Exact next action and authorization: begin WP4 with committed fold-local PCA leakage tests before
  any PCA implementation or pipeline extension.

#### 2026-10-08 00:47 EDT - WP4 GREEN and fold-local PC milestone

- State: WP4 and the in-memory unit/PC linear milestone are complete; WP5 is ready but not started.
- Authorization: the user explicitly requested a push followed by the next implementation; work
  remained within WP4 and used no experimental data or batch execution.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `5b3aefd` / `b7e57f8`.
- Worktree and owned files: added `pca.py`, extended the regional PCA record and linear pipeline,
  and added/extended PCA and pipeline tests. Unrelated pre-existing untracked entries remained
  untouched.
- RED command and result: the core PCA tests initially failed collection because `pca.py` did not
  exist. PC integration tests then failed because no PC rows or injectable fold transforms existed.
  The provenance/reuse test finally failed because `fit_cross_validation_pcas` did not exist.
- GREEN/regression commands and results: 59 focused interregional tests passed. A combined 150-test
  run covering all interregional work plus existing spike-behavior package, population package/PCA,
  and session-metadata tests passed with four existing Pynapple warnings.
- Commits: core PCA tests `ca19b41`; core implementation `8365994`; PC-pipeline tests `9b0f9f1`;
  PCA-provenance tests `1e67e28`; PC OLS/provenance implementation `b7e57f8`.
- External actions: pushed the completed WP1-WP3 history through `5b3aefd` from local `refactor` to
  `origin/refactor` at the user's request. WP4 commits have not yet been pushed. No experimental
  data or batch command was run.
- Findings and unresolved risks: no new issue was found in WP4. Fold transforms are trained from
  the unique requested-condition union, shared across directions/conditions/windows, and kept
  distinct from descriptive transforms. Requested unavailable ranks remain explicit. The separate
  quickstart JSON-glob mismatch recorded under WP1 remains unrelated and unchanged.
- Exact next action and authorization: do not begin WP5 until its scope is confirmed; then commit
  persistence/run-boundary tests in RED before implementation.

#### 2026-10-08 00:56 EDT - WP5 GREEN and immutable run boundary

- State: WP5 is complete; WP6 plotting and saved-only webapp work is ready but not started.
- Authorization: the user explicitly requested a push followed by the next implementation; work
  remained within WP5 and used only synthetic pytest data.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `408de4c` / `9bab21b`.
- Worktree and owned files: added `persistence.py`, `run_session.py`, `run_batch.py`, persistence and
  script tests, and bounded package exports. Unrelated pre-existing untracked entries remained
  untouched.
- RED command and result: persistence tests initially failed module import. Runner tests then failed
  module import. Follow-up delivery tests failed for absent exact stage progress and batch byte
  estimates before those fields were implemented.
- GREEN/regression commands and results: all 70 WP1-WP5 focused tests passed before the final
  delivery assertions; the final affected suite passed 162 tests covering interregional work plus
  existing spike-behavior, population package/PCA, and session-metadata tests, with four existing
  Pynapple warnings.
- Commits: persistence tests `f3815db`; persistence implementation `a754637`; runner tests
  `57c930a`; real composition test `447a516`; runner implementation `3de0662`; progress/estimate
  tests `3567a1d`; progress/estimate implementation `9bab21b`.
- External actions: pushed completed WP4 history through `408de4c` to `origin/refactor` at the
  user's request. WP5 commits have not yet been pushed. No experimental-data or batch run was
  performed.
- Findings and unresolved risks: no new scientific issue was found. The production composition
  path was exercised with generated metadata-v2 files and generated spike/trial inputs, including
  Pynapple preparation, OLS, atomic finalization, and trusted reload. The quickstart JSON-glob
  mismatch recorded under WP1 remains unrelated and unchanged.
- Exact next action and authorization: do not begin WP6 until its scope is confirmed; then commit
  plotting and saved-only webapp tests in RED before implementation.

#### 2026-10-08 01:06 EDT - WP6 GREEN saved-result presentation

- State: WP6 is complete; WP7 synthetic integration and bounded performance work is ready but has
  not started.
- Authorization: the user explicitly requested a push followed by resumed implementation. This
  authorized WP6 source, tests, and documentation; no experimental-data or batch run was implied.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `ef2ee0c` / `1a58ff3` before this handoff record.
- Worktree and owned files: added plotting and saved-only webapp tests, `plotting.py`,
  `interregional_views.py`, standard-figure generation in `run_session.py`, router integration,
  package/user documentation, and an example scientific configuration. Unrelated pre-existing
  untracked entries remained untouched.
- RED command and result: plotting and webapp tests initially failed collection because their
  modules did not exist. A follow-up composition test failed because completed runs did not yet log
  a figures stage or contain PNG output.
- GREEN/regression commands and results: 32 focused plotting, webapp, runner, and existing-webapp
  tests passed. The affected suite passed 187 tests covering WP1-WP6 plus existing spike-behavior,
  population package/PCA, webapp, and session-metadata behavior, with four existing Pynapple
  warnings and no new warnings.
- Commits: plotting/webapp tests `dce1743`; plotting implementation `dccf07c`; saved-figure test
  `c8d47a9`; webapp and saved-figure implementation `a23370d`; workflow documentation `1a58ff3`.
- Real-data, filesystem, or external actions: pushed completed WP5 history through `ef2ee0c` from
  local `refactor` to `origin/refactor` at the user's request. WP6 commits have not been pushed. All
  WP6 tests used generated or fixture data; no experimental-data or batch run was performed.
- Findings and unresolved risks: no new scientific or implementation issue was found. Plotting and
  the webapp consume validated saved tables only; the UI has no fit, resume, or recompute path.
  Completed CLI runs now save opaque light-mode PNGs before atomic finalization. The quickstart
  JSON-glob mismatch recorded under WP1 remains unrelated and unchanged.
- Exact next action and authorization: do not begin WP7 until its scope is confirmed; then run the
  synthetic integration and bounded performance check without experimental data.

#### 2026-10-08 01:13 EDT - WP7 GREEN synthetic integration and bounded performance

- State: WP7 is complete; WP8 one-session scientific inspection is ready but remains separately
  unauthorized.
- Authorization: the user explicitly requested a push followed by WP7. This covered synthetic
  integration tests and bounded performance evidence only, not experimental data.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `dc4ad4e` / `673bc3f` before this handoff record.
- Worktree and owned files: updated the authoritative plan and added WP7 cases to
  `test_interregional_pipeline.py` and `test_interregional_scripts.py`. No production source file
  required a change. Unrelated pre-existing untracked entries remained untouched.
- RED command and result: the first full-path run rejected a duplicated metadata-v2
  `unit_channels` fixture value. After correcting that fixture contract, the second run correctly
  reported `no_train_rows` for `stay` because the fixture marked every trial rewarded; the fixture
  was corrected to make the explicitly requested stay condition eligible. Neither failure exposed
  a production defect, and no production fix was made.
- GREEN/regression commands and results: both new WP7 tests passed. The affected suite passed 189
  tests covering WP1-WP7 plus existing spike-behavior, population package/PCA, webapp, and session-
  metadata behavior, with four existing Pynapple warnings and no new warnings.
- Commits: WP7 authorization handoff `a472bb4`; synthetic integration tests `673bc3f`; no
  implementation commit was needed.
- Real-data, filesystem, or external actions: pushed completed WP6 history through `dc4ad4e` from
  local `refactor` to `origin/refactor` at the user's request. The WP7 tests used only seeded data in
  pytest temporary directories. No experimental-data or batch run was performed.
- Bounded performance evidence: GNU `/usr/bin/time -v` measured the complete pytest process, so its
  452,644-KiB maximum resident set includes Python, pytest, Pynapple, NumPy, plotting, and two
  immutable production-path executions; it is not an isolated estimator measurement or upper
  bound. The two executions took 3.013673 seconds inside the test (4.76-second measured command
  wall time). The seed-91 fixture contained 25 trials, 40 whole-window bins, two units per region,
  two conditions, three windows, and three requested PCs per region. It produced 300 fold-score
  rows, 18 PNGs, and 1,493,247 bytes in the first finalized run directory.
- Findings and unresolved risks: identical inputs/configuration produced exactly equal typed tables
  across immutable reruns; fold/trial/bin hashes survived persistence; unavailable third PC ranks
  remained explicit while unit rows completed; and Poisson/Granger fields remained inactive. No
  new scientific or implementation issue was found. The measurement is deliberately bounded
  evidence for this small fixture, not a real-session capacity claim.
- Exact next action and authorization: obtain explicit approval of the exact WP8 metadata session,
  configuration, command, and output root before touching experimental data.

#### 2026-10-08 01:26 EDT - WP8 CT026 run complete; user review pending

- State: the approved WP8 command and technical inspection are complete. Scientific acceptance by
  the user is pending, and WP9 remains unauthorized.
- Authorization: the user approved session `CT026_2026-08-03_111938`, ProbeA as PFC, ProbeB as HPC,
  metadata-quality good channels with good/MUA units, documented 100-ms unit/PC OLS defaults, a
  session-local configuration, the existing `analysis_runs` root, and `new` only after a successful
  dry-run.
- Sol / Terra / reviewer: primary command runner and evidence reviewer; no worker.
- Start HEAD / end HEAD: `fef9202` / `b6ca810` before this handoff record.
- Worktree and owned files: committed only the plan authorization update. Created
  `interregional_regression_config.json` in the approved external session root and one immutable
  finalized analysis directory. Unrelated repository entries remained untouched.
- Commands and results: the exact single-session `dry-run` passed with session ID
  `CT026_2026-08-03_111938` and 274,440,585 input bytes. The exact `new` command completed with
  fingerprint `7b4d1a0fdbe1d16b8316b37f52fcabbe94b62e8ef5568ae62c7526b96fdc0985` and no captured
  warnings. Result reload through the trusted validator passed.
- Commit: exact run authorization `b6ca810`; no source or test commit.
- Real-data, filesystem, or external actions: pushed WP7 through `fef9202` before WP8. The finalized
  run is `analysis_runs/interregional_regression_20261008T052234859222Z_7b4d1a0fdbe1` beneath the
  CT026 session. It contains 49 files, 42 PNG figures, and approximately 38 MiB total output.
- Runtime and memory: input validation took 0.032806 seconds, hashing 0.677029 seconds, combined
  preparation/OLS 158.753958 seconds, persistence 0.431800 seconds, figures 14.276614 seconds, and
  the logged post-directory-creation total was 173.465737 seconds. GNU `/usr/bin/time -v` measured
  2:55.86 command wall time and 1,178,500-KiB maximum resident set size with no swaps.
- Scientific inspection: resolved populations contain 160 ProbeA/PFC units from 315 selected good
  inside-brain channels and 309 ProbeB/HPC units from 383 selected channels. Five folds contain 130
  trials each; 646 of 650 trials are scientifically/CV eligible, with the same four rows carrying
  invalid-alignment and invalid-reward-status reasons. All 1,800 PC fold rows completed with ten
  components per region/fold. Unit rows contain 36,852 successful, 5,161 fit-unavailable, and 197
  metric-unavailable rows; the saved reasons are 4,050 restricted-rank failures, 1,109 full-rank
  failures, 197 constant test targets, and two constant training targets. These strict direct-unit
  failures are expected and explicit, not silent fallback. PC median incremental held-out R-squared
  is consistently larger for PFC-to-HPC than HPC-to-PFC in this run, while direct-unit population
  medians are negative; these are predictive summaries without causal or significance claims.
- Presentation findings: saved tables, hashes, PCA records, and figures validate, but the two
  direction-specific count annotations overlap at the top of every increment plot. The absolute-
  score plots also lack a self-contained direction/color legend. Separately, `summary.md` records
  the hidden pre-finalization `.incomplete` path and shows a null run output root even though the
  copied `config.json` records the approved final root. These are concrete presentation/provenance
  defects; the immutable result tables were not altered.
- Exact next action and authorization: the user reviews the finalized output and decides whether to
  approve a tests-first bounded presentation correction. Do not begin WP9 before that review and
  the existing WP8 scientific gate are satisfied.

#### 2026-10-08 01:36 EDT - WP8 bounded correction GREEN and immutable rerun

- State: the user-approved presentation/provenance correction and fresh immutable CT026 rerun are
  complete. WP8 scientific acceptance by the user is pending; WP9 remains unauthorized.
- Authorization: the user approved the recommended tests-first correction and fresh immutable
  rerun. The original run was retained unchanged.
- Sol / Terra / reviewer: primary implementation agent and evidence reviewer; no worker.
- Start HEAD / end HEAD: `79c5cd9` / `22cc4ed` before this handoff record.
- Worktree and owned files: updated plotting, persistence final-path derivation, session-summary
  composition, and their plotting/runner tests. No scientific preparation, fit, score, aggregation,
  or result-schema code changed. Unrelated pre-existing entries remained untouched.
- RED command and result: the first focused run failed all three tests because increment counts were
  separate overlapping annotations, absolute plots had no legend, and summaries recorded the
  incomplete working path. A dense-target test then reproduced condition-cell spillover (minimum
  x=-2.78 for a cell bounded at -0.60). A follow-up test caught an implementation-time single-target
  centering regression before the implementation commit.
- GREEN/regression commands and results: six focused plotting/runner/persistence tests passed. The
  affected suite passed 191 tests with the same four existing Pynapple warnings and no new warning.
- Commits: correction authorization `5bd2a8a`; initial defect tests `1e284d9`; dense-jitter test
  `67cdfa0`; single-target centering test `9109437`; implementation `22cc4ed`.
- Real-data, filesystem, or external actions: the unchanged approved configuration passed dry-run
  again with 274,440,585 input bytes. A new immutable run finalized at
  `analysis_runs/interregional_regression_20261008T053259462655Z_983f52e6b4af` with fingerprint
  `983f52e6b4af8f5279f5623c64ca3dc56eebb1beeaf8b277d3baab28883399df`. The original
  `7b4d1a0fdbe1` run remains unchanged.
- Runtime and memory: corrected-run validation took 0.036471 seconds, hashing 0.144654 seconds,
  combined preparation/OLS 169.750858 seconds, persistence 0.415218 seconds, figures 15.224394
  seconds, and the logged total 185.394685 seconds. GNU `/usr/bin/time -v` measured 3:07.27 command
  wall time, 1,181,148-KiB maximum resident set size, and no swaps.
- Findings and unresolved risks: trusted reload passed, the corrected summary records the exact
  final directory and approved output root, all 42 figures were regenerated, and visual inspection
  confirmed bounded direction-cell jitter, readable contributing/unavailable counts, and an
  absolute-score direction legend. Every typed scientific result table and the bin-edge array are
  exactly equal to the original run; only code identity and presentation/provenance artifacts
  changed. No new issue was found. Predictive/coverage limitations and strict unit-rank
  unavailability remain as previously reported.
- Exact next action and authorization: present the corrected run to the user. Begin WP9 only after
  explicit scientific acceptance and separate WP9 authorization.

#### 2026-10-08 09:44 EDT - WP8 accepted and WP9 activated

- State: WP8 is complete and accepted; WP9 core tests are next.
- Authorization: after reviewing the interpretation of absolute versus incremental CV R-squared,
  the user requested a push and continued implementation. This accepts the corrected standard-
  regression output and authorizes WP9; it does not authorize another experimental-data run.
- Sol / Terra / reviewer: primary implementation agent; no worker.
- Start HEAD / end HEAD: `04ab655` / `04ab655` before this activation record.
- Worktree and owned files: no tracked modification before this record. Four known, noninterfering
  task-decoding commits from another task advanced the shared branch after the WP8 handoff; the user
  confirmed they are expected. They are preserved unchanged.
- API verification: installed statsmodels is 0.15.0. Source inspection confirmed `GLM(endog, exog,
  family, missing)`, `GLM.fit(method="IRLS", maxiter=100, tol=1e-8, scale=None,
  cov_type="nonrobust", full_output=True, disp=False)`, `Poisson(Log())`, Boolean `converged`, and
  integer `fit_history["iteration"]`. The installed source emits `PerfectSeparationWarning` and
  exposes the frozen statsmodels warning/exception classes.
- RED/GREEN commands and commits: pending WP9 tests-only work.
- Real-data, filesystem, or external actions: `git push origin refactor` reported everything up to
  date at `04ab655`. No experimental data were opened and no analysis output was created.
- Findings and unresolved risks: no new blocker was found. WP9 remains target-wise because
  convergence/status are target-specific; histories and designs will be reused within each fold.
  No within-session parallel fitting will be added before measured profiling.
- Exact next action and authorization: write and commit `test_interregional_poisson.py` in RED,
  then implement only the core target-wise fitting, scoring, and derived matched-MSE layer.

#### 2026-10-08 09:57 EDT - WP9 code GREEN and synthetic integration complete

- State: all WP9 source, tests, persistence integration, plotting, reporting, and saved-only webapp
  work is GREEN. The required designated-session Poisson inspection is not yet authorized, so WP9
  scientific acceptance and WP10 remain pending.
- Authorization: the user requested a push and continued implementation after accepting WP8. This
  authorized WP9 code and seeded synthetic testing, but the activation record explicitly excluded
  another experimental-data run.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `7a7db4b` / `f826d52` before this handoff record.
- Architecture: added target-wise unpenalized statsmodels Poisson GLMs with explicit log link and
  IRLS settings, direct deviance/null-deviance scoring, exact derived OLS/Poisson MSE pairing,
  shared OLS/Poisson fold-history construction, typed pipeline/session integration, separate
  deviance and exploratory count-MSE figures, Markdown reporting, and read-only webapp display.
  PCs remain OLS-only and raw deviance remains fold-only.
- RED evidence: the core suite first failed import because `poisson.py` did not exist; five pipeline
  cases then failed because the Poisson CV entry point did not exist; the production-path synthetic
  test failed the expected fold-key-grid validation while the runner still saved OLS only; plotting
  tests failed the hard-coded `delta_r2` path and absent MSE figure; the saved-view test failed the
  absent MSE display import; and the report test failed because the summary still omitted Poisson.
- GREEN evidence: all 113 inter-regional tests passed. The broader affected suite passed 275 tests
  covering inter-regional analysis, spike-behavior/Pynapple, population package/PCA, session
  metadata, and webapp packaging, with six existing Pynapple warnings. Ruff passed on every edited
  source/test group and `git diff --check` passed.
- Commits: activation `7a7db4b`; core tests `a4377fe`; core implementation `aa5f4ba`; pipeline tests
  `752acf0`; pipeline implementation `24248cb`; session test `a25d4e1`; session integration
  `cf9d8e8`; plotting tests `a42a317`; plotting implementation `9dbbad3`; webapp test `bf6c2af`;
  webapp implementation `fa650d2`; report test `0ad1714`; report implementation `187bc45`; and
  strengthened raw-prediction checks `f826d52`.
- Synthetic performance evidence: the production-path WP9 pytest completed in 2.85 seconds wall
  time with 411,792 KiB maximum resident set for the complete Python/pytest/numerical/plotting
  process. This is bounded fixture evidence, not an isolated estimator benchmark or a real-session
  capacity claim.
- Real-data, filesystem, or external actions: no experimental data or batch was opened or run. The
  integration test used seed 91 and pytest temporary directories. Unrelated pre-existing untracked
  files and the user-confirmed task-decoding commits were left untouched.
- Findings and unresolved risks: the seeded public runner produced complete matched unit OLS and
  Poisson folds with identical train/test fingerprints, converged fits, immutable reload, Poisson
  figures, and a correctly caveated exploratory MSE report. No new implementation issue was found.
  Real-session convergence and runtime remain intentionally unmeasured until separately approved.
- Exact next action and authorization: ask the user to approve the exact CT026 Poisson config,
  command, and output root. Do not begin WP10 until that run is inspected and accepted.

#### 2026-10-08 10:19 EDT - WP9 CT026 inspection replanned as one-shot Slurm execution

- State: the WP9 code remains GREEN. The designated CT026 Poisson computation is authorized and
  will run as one ordinary Slurm job; scientific inspection still follows only after a finalized
  immutable result directory exists.
- Authorization: the user first approved the CT026 Poisson configuration and local execution, then
  directed that a long run be submitted for offline cluster completion and later inspection. The
  user explicitly rejected resumability overengineering and approved the reduced one-shot Slurm
  plan. This authorizes the wrapper, cluster-only config copy, push, dry-run, and one `sbatch`
  submission. It does not authorize WP10.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `40b2ab3` / `b05d22e` before this handoff update.
- Approved scientific scope: session `CT026_2026-08-03_111938`; ProbeA/PFC and ProbeB/HPC; 160 PFC
  and 309 HPC units; all six configured conditions; before, after, and whole windows; units plus ten
  fold-local PCs per region; `ols_cv` plus `poisson_cv`; and the session's existing `analysis_runs`
  output root. The cluster config changes only the metadata and output-root path prefixes from the
  local mount to `/gs/gsfs0/users/mchin1/contextProjectData/...`.
- Measured local evidence: the approved command passed dry-run for 274,440,585 input bytes. The
  real command was interrupted cleanly after 6:21.13 wall seconds because it remained in
  statsmodels Poisson fitting; `/usr/bin/time -v` recorded 8,105.68 user-CPU seconds, 34.04 system-
  CPU seconds, 2,135% CPU, and 10,000,472-KiB peak RSS. The 88-KiB incomplete directory
  `.interregional_regression_20261008T140701151108Z.incomplete` is preserved as interruption
  evidence and is not a completed result.
- Architecture and performance decision: add only
  `src/shell_scripts/interregional_regression_slurm.sh`, forwarding exact arguments to the existing
  `src.neural_analysis.interregional.run_session` CLI through the repository's frozen offline `uv`
  environment. Request one task, eight CPUs, 32 GiB, and 72 hours on `unlimited`; expose all eight
  threads to OMP, MKL, and OpenBLAS because the measured single-process numerical path used threaded
  linear algebra. Require an exact tracked-clean repository root. Do not add sharding,
  checkpoints, a resume protocol, a Python scheduler layer, or within-session process workers.
- Tests written before implementation: Bash syntax and exact Slurm directives; required tracked
  dependencies; missing, malformed, or non-eight CPU metadata; missing, invalid, non-root, or
  tracked-dirty submission checkout; exact argument forwarding including spaces; frozen offline
  `uv`; eight-thread environment; stdout/stderr and exit propagation; and direct TERM propagation
  through `exec`.
- RED/GREEN evidence and commits: the 13 launcher cases all failed because the approved wrapper was
  absent, then passed after implementation. The tests-only commit is `7a2ab03`; the wrapper commit
  is `b05d22e`. The broader affected suite passed 305 tests covering inter-regional analysis,
  spike-behavior/Pynapple, population PCA, session metadata, webapp packaging, and both shell
  wrappers, with six existing Pynapple warnings. Ruff and `git diff --check` passed.
- Cluster execution receipt: pushed `49f63fd` to `origin/refactor`, fast-forwarded the tracked-clean
  cluster checkout to exact commit `49f63fde644e36b15484aab8fc9727235bfde01e`, and copied the
  separately named cluster config without overwriting another file. A semantic comparison proved
  that only the metadata and output-root prefixes differ from the approved local config. The
  cluster dry-run returned planned session `CT026_2026-08-03_111938` and 274,440,585 input bytes.
  `sbatch --parsable` returned job `30989197`; `squeue` showed it running on `cpu-743` with eight
  CPUs. The startup log at
  `/gs/gsfs0/users/mchin1/logs/interregional_regression_30989197.log` records the exact commit,
  tracked-clean status, OMP/MKL/OpenBLAS limits of eight, `uv 0.12.17`, and the intended exact
  `new --config` arguments.
- Failure policy: a failed or preempted job leaves its ordinary incomplete attempt for diagnosis;
  a retry is a fresh immutable `new` run. This is intentionally simpler than resumable execution.
- Exact next action and authorization: after job `30989197` leaves the queue, inspect `sacct`, the
  scheduler log, and the finalized immutable result with the standard loader. If the job fails or
  is preempted, diagnose the ordinary incomplete attempt before requesting a fresh submission.

#### 2026-10-08 14:08 EDT - WP9 OOM confirmed; reduced telemetry activated

- State: the scientific implementation remains unchanged and GREEN, but the designated full CT026
  run did not complete. Resource instrumentation is now the active WP9 support package.
- Authorization: after receiving the OOM handoff, the user approved a simplified instrumentation
  plan and emphasized that the logs must contain enough information to diagnose the failure. This
  authorizes tests and implementation, their commits and push, but not a real-data benchmark or a
  full retry; benchmark execution remains a separate gate.
- Sol / Terra / reviewer: primary implementation agent and self-review; no worker.
- Start HEAD / end HEAD: `f53cc47` / `9fe3b25` before this handoff update. The intervening commits
  after the Slurm wrapper are known task-decoding changes from the other task and are preserved
  unchanged. Tracked state was clean before this record; known unrelated untracked files remain out
  of scope.
- Failure evidence: job `30989197` used exact commit
  `49f63fde644e36b15484aab8fc9727235bfde01e`, eight CPUs, and 32 GiB. Its scheduler log ends with
  one `oom_kill` event. File timestamps bound execution from approximately 10:27 to 13:11 EDT,
  about 2 hours 44 minutes. Slurm accounting exposes no usable RSS row. The hidden incomplete run
  `.interregional_regression_20261008T142744818446Z.incomplete` contains only the existing config,
  manifest, script copies, figures directory, and a run log whose simultaneous preparation/OLS/
  Poisson starts do not localize the failure. No finalized result exists.
- Interpretation: the job certainly exceeded 32 GiB, but current evidence cannot distinguish a
  steady retained-allocation increase from one high-memory cell. The earlier 10-GiB local peak and
  later cluster OOM justify trajectory measurement before increasing the allocation. The planned
  84,420 restricted-plus-full fits are an upper bound, not an exact realized count, because rank,
  constant-target, and fit-availability checks are data dependent.
- Reduced architecture: add a standard-library `resource_usage.py` monitor that appends complete
  JSONL records every 30 seconds and at progress boundaries, recording UTC and monotonic time,
  sequence, PID, process CPU and peak RSS, and cgroup-v2 current/peak/limit/OOM counters. Write an
  atomic summary only on normal completion. Add optional callbacks, defaulting to `None`, for actual
  preparation/PCA/OLS/Poisson/result-assembly stages, every analysis-cell boundary, and the first,
  every 25th, and final Poisson target. Progress samples include direction, representation,
  condition, window, fold, row counts, relevant shapes/bytes, completed/total targets, and
  unavailable counts. Do not change scientific configuration, fitting, results, or persistence
  schemas; do not add cgroup-v1 support or an inaccurate "exact" dry-run fit estimate.
- Tests written before implementation: Linux RSS byte normalization; cgroup-v2 parsing including
  missing, malformed, and unlimited fields; stable increasing JSONL sequences and nonnegative
  finite values; monitor shutdown on normal/exception paths; killed-subprocess trace durability;
  atomic completion summary; actual stage order; correct analysis-cell identity/shapes/bytes;
  Poisson first/every-25/final cadence; telemetry-on/off scientific equality; failure trace
  retention; and compatibility with completed runs lacking telemetry.
- Performance boundary: one 30-second heartbeat plus coarse progress events should remain a few
  megabytes and must not add per-fit I/O. No dependency is added. A bounded production-path CT026
  benchmark using only condition `all`, window `whole`, units, OLS, and Poisson will be proposed
  after the instrumented test suite is GREEN.
- RED/GREEN evidence and commits: `44b13fd` committed the tests before production code. The new
  telemetry module initially failed import; the existing pipeline rejected the callback and cadence
  contracts; and session tests lacked durable trace/summary artifacts. Implementation commit
  `9fe3b25` made all 32 focused tests GREEN. The broader affected suite passed 315 tests spanning
  inter-regional analysis, spike-behavior/Pynapple, population PCA, metadata, webapp packaging, and
  shell launchers, with the same six existing Pynapple warnings. Ruff and `git diff --check` passed.
- Diagnostic sufficiency: the trace records the resolved cgroup path and interval at monitor start.
  Every heartbeat and progress event samples effective cgroup limit, current/peak bytes, high/max/
  OOM/OOM-kill counters, process peak RSS, and process user/system CPU. A pre-allocation cell record
  identifies model family, representation, direction, condition, window, fold, cell index, and
  target total; a post-allocation record adds every response/design shape and byte count. Poisson
  records then delimit completed target ranges at 1, every 25, and final target. A killed subprocess
  retained complete parseable JSONL records in the test gate. This is sufficient to distinguish a
  matrix-construction jump, a long individual fit, and progressive retention across targets/cells.
- Exact next action and authorization: ask for approval of a distinct immutable CT026 benchmark
  config containing only condition `all`, window `whole`, representation `units`, and analyses
  `ols_cv` plus `poisson_cv`. Do not run experimental data before that approval.

#### 2026-10-08 14:39 EDT - WP9 bounded CT026 memory benchmark reproduced OOM

- State: the separately approved diagnostic benchmark ran as Slurm job `30991145` on `cpu-755`
  with eight CPUs and 32 GiB, then reproduced the OOM in approximately 3 minutes 30 seconds. This
  receipt does not authorize a scale test, full retry, fitting-code change, or WP10.
- Authorization and scope: the user approved one CT026 benchmark retaining the designated
  session's units, folds, binning, lags, and model settings while restricting execution to condition
  `all`, window `whole`, representation `units`, and analyses `ols_cv` plus `poisson_cv`. Its distinct
  output root is
  `/gs/gsfs0/users/mchin1/contextProjectData/CT026/CT026_20260803_latent_inference/analysis_runs/interregional_memory_benchmarks`.
- Reproducibility evidence: the cluster checkout was tracked-clean apart from expected untracked
  Python cache files and was fast-forwarded to exact commit
  `13f7f832e02ac51631df9a0abf4fbe9c155d115e`. A programmatic comparison confirmed that the
  benchmark configuration differs from the designated full configuration only in
  `prediction_windows`, `filters.conditions`, `representations`, and `run.output_root`. The cluster
  dry-run returned one planned session, `CT026_2026-08-03_111938`, with input size 274,440,585 bytes.
- Submission evidence: scheduler log
  `/gs/gsfs0/users/mchin1/logs/interregional_regression_30991145.log` records node `cpu-755`, start
  time `2026-10-08T18:34:45Z`, the exact commit, tracked-clean status, eight-thread BLAS limits,
  `uv 0.12.17`, and the benchmark configuration path. The same log records one Slurm `oom_kill`
  event at `2026-10-08T14:38:15.881`; Slurm accounting had not yet published a job row when checked.
- Telemetry evidence: the retained incomplete immutable attempt
  `.interregional_regression_20261008T183535173307Z.incomplete` contains a parseable
  35,322-byte `resource_trace.jsonl`; no completion summary or finalized result exists. It records
  preparation and all ten OLS cells completing with process peak RSS about 1.04 GiB, followed by
  Poisson cell 1 (`HPC_to_PFC`, fold 0). That cell's arrays are modest: the largest recorded array
  is the 20,163-by-470 full training design at 75,812,880 bytes. The first of 160 targets completed,
  then 30-second heartbeats recorded process peak RSS of 1.65, 9.15, 16.99, 23.83, and 30.03 GiB;
  the last complete record preceded Slurm's OOM report by about ten seconds. Because target progress
  is intentionally coarse, the failure is localized to completed-target range 1 through 24, not an
  exact target or restricted/full side. The node does not expose a resolvable cgroup-v2 directory to
  the process, so the cgroup fields are explicitly `null`.
- Interpretation: reducing conditions, windows, and representations does not bound peak memory,
  because the failure occurs inside the first Poisson cell rather than after accumulating cells.
  OLS and matrix construction are not the immediate problem; peak RSS grows during target-wise
  Statsmodels IRLS execution and reaches the 32-GiB allocation. Peak RSS is a high-water mark, so
  the trace alone does not distinguish retained allocations from increasingly large transient fit
  allocations, but that distinction is not needed before replacing or bounding the fitting path.
- Exact next action and authorization: prepare a short tests-first plan for a memory-bounded
  unpenalized Poisson fitting path, preserving the frozen scientific and result contracts. Do not
  implement it or submit another data run without separate approval.

#### 2026-10-08 14:44 EDT - WP9 QR-backed IRLS correction activated

- State and authorization: the user approved the proposed minimal fitting correction. This
  authorizes tests, the source/specification change, local verification, commits, and push. It does
  not authorize another CT026 job, a scale test, a full retry, or WP10.
- Architecture: retain `fit_poisson_target(...)`, Statsmodels `GLM`, the explicit intercept,
  Poisson/log family, unpenalized IRLS, convergence handling, and all pipeline/result interfaces.
  Add only the explicit Statsmodels `wls_method="qr"` fit keyword so each IRLS weighted
  least-squares step uses QR instead of the default `lstsq` path and final pseudoinverse path.
- Dependency/API evidence: no dependency changes. Installed Statsmodels 0.15.0 source confirms
  that `GLM.fit(...)` forwards extra keywords to `_fit_irls(...)`, which accepts `wls_method`; its
  minimal WLS implementation supports `pinv`, `qr`, and `lstsq`. A seeded five-coefficient check
  found default and QR parameters agreeing within `1.4e-16` with the same five iterations.
- Tests to write before implementation: extend the mocked estimator-call contract to require
  `wls_method="qr"`; add a seeded full-rank multivariable comparison requiring QR-backed public-fit
  parameters, predictions, convergence, and iteration count to match the existing default IRLS
  reference within explicit numerical tolerance. Existing warning, exception, validation, pipeline,
  persistence, and telemetry tests remain unchanged.
- Performance boundary: this is a targeted backend selection, not a custom solver, dependency,
  parallelization change, or broad refactor. Local tests can establish correctness but cannot prove
  the 32-GiB cluster outcome; after GREEN and push, the same bounded CT026 benchmark requires a new
  explicit execution approval.
- Exact next action: commit the tests before production code, demonstrate the missing QR keyword as
  RED, then implement and run focused plus affected regression suites.

#### 2026-10-08 14:46 EDT - WP9 QR-backed IRLS correction GREEN

- State: the approved memory correction is implemented and locally GREEN. No experimental data was
  read and no cluster job was submitted.
- TDD evidence: tests-only commit `c52bfd0` added the explicit QR-call contract and seeded
  multivariable numerical-equivalence test. The focused test file then had exactly one failure: the
  production call omitted `wls_method="qr"`; its other 25 tests passed. Adding that one keyword made
  all 26 focused Poisson tests pass.
- Implementation: `fit_poisson_target(...)` now passes `wls_method="qr"` to the already frozen
  Statsmodels IRLS call. Public interfaces, estimator family/link, regularization, iteration and
  tolerance settings, validation, diagnostics, result schemas, pipeline control flow, and
  dependencies are unchanged. The v3 specification and implementation plan now make the backend
  explicit.
- Regression evidence: all 138 inter-regional tests passed. The broader affected suite passed 303
  tests across inter-regional analysis, Pynapple spike behavior, population PCA, session metadata,
  webapp packaging, and import packaging, with the same six existing Pynapple warnings. Ruff and
  `git diff --check` passed.
- Bounded synthetic performance: the existing seeded end-to-end WP9 runner passed in 2.39 seconds
  pytest time and 3.23 seconds measured wall time, with 432,124 KiB maximum resident set. This is a
  correctness and fixture-capacity check, not evidence that CT026 will fit within 32 GiB.
- Exact next action and authorization: after commit and push, request approval for the same bounded
  CT026 benchmark configuration and 32-GiB allocation using the QR-backed commit. Do not submit it
  or any larger run without that separate approval.

#### 2026-10-08 14:50 EDT - WP9 QR-backed bounded benchmark submitted

- State and authorization: the user separately approved repeating the same bounded CT026 benchmark.
  Exactly one job, Slurm `30991248`, was submitted with eight CPUs and 32 GiB. This does not
  authorize a replacement, scale test, full retry, or WP10.
- Reproducibility: the cluster checkout was tracked-clean apart from expected untracked Python cache
  directories and was fast-forwarded to exact QR-backed commit
  `88066433f4b3896c98b5b69dfb6b38133b0762ac`. The existing benchmark configuration was not changed:
  session `CT026_2026-08-03_111938`, condition `all`, window `whole`, representation `units`, and
  analyses `ols_cv` plus `poisson_cv`, with the distinct benchmark output root and original unit,
  fold, bin, lag, and model settings. The repeated dry-run again planned one session from
  274,440,585 input bytes.
- Submission evidence: scheduler log
  `/gs/gsfs0/users/mchin1/logs/interregional_regression_30991248.log` records node `cpu-755`, UTC
  start `2026-10-08T18:49:16Z`, the exact commit, tracked-clean status, eight-thread numerical
  limits, `uv 0.12.17`, and the exact benchmark configuration argument. The job entered `RUNNING`
  with the requested allocation.
- Telemetry startup: immutable attempt
  `.interregional_regression_20261008T185005743728Z.incomplete` contains a parseable resource trace.
  Its first inspected records show ordered OLS progress with approximately 1.04 GiB process peak
  RSS. Cgroup fields remain unavailable on this node as documented for the preceding benchmark.
- Exact next action and authorization: inspect job `30991248` after it leaves the queue, including
  the scheduler log, trace, and saved artifacts. Do not submit another job automatically.

#### 2026-10-08 14:55 EDT - WP9 QR-backed bounded benchmark OOM

- State: Slurm job `30991248` was OOM-killed at `2026-10-08T14:55:02.402` after approximately
  5 minutes 46 seconds. No finalized result or completion summary exists; the ordinary incomplete
  attempt and its 37,496-byte trace remain available for diagnosis.
- Final evidence: all ten OLS cells completed near 1.04 GiB process peak RSS. Poisson cell 1 then
  began for `HPC_to_PFC`, fold 0, and recorded completion of target 1 of 160. Subsequent 30-second
  heartbeats recorded process peak RSS of 4.88, 8.65, 12.22, 16.15, 19.83, 23.14, 26.32, and
  29.52 GiB before Slurm reported one `oom_kill`. Slurm `MaxRSS` reached approximately 31.37 GiB
  while the job was still in that first Poisson cell. No target-25 checkpoint was reached.
- Interpretation: QR approximately doubled the time to OOM and reduced the early memory-growth
  rate, but it did not bound memory under the same 32-GiB allocation. The evidence rejects the QR
  backend as a sufficient production fix. The trace still localizes failure to completed-target
  range 1 through 24 and cannot determine whether one difficult target or retained allocations
  across several targets dominate; do not claim a more precise cause from these records.
- Exact next action and authorization: replan the target-wise optimizer around a method with bounded
  working memory and validate numerical equivalence to the same unpenalized Poisson/log objective.
  Do not implement or submit another job without separate approval.

## Plan objective and status

This plan implements `docs/spec_neural_regression_v3.md`. It was approved and is now the
authoritative phased implementation and handoff log. Package-specific authorization and data-run
gates remain binding even though WP1-WP8 and WP9 code implementation are complete.

The implementation order is deliberately fixed:

1. Shared preparation and cross-validated OLS for direct units.
2. Training-only regional PCA and cross-validated OLS for PCs.
3. Standard-regression result summaries, plots, and metadata-driven web-app integration.
4. Cross-validated Poisson regression for units and held-out OLS/Poisson MSE comparison.
5. Descriptive in-sample linear and Poisson Granger-style analyses.

This order produces an inspectable standard-regression workflow before adding Poisson-specific or
Granger-specific complexity.

## Confirmed decisions

The plan treats the following as settled requirements:

- Cross-validation uses exactly five deterministic groups from `trial_df["cur_block"]`; there is
  no inferred-block or random-split fallback.
- The user explicitly assigns metadata populations to PFC and HPC. Region roles are never inferred
  from probe names.
- The scientific `all` condition means valid experimenter-reward status intersected with valid
  alignment, choice/context filters, and user exclusions. CV additionally requires a nonmissing
  block; descriptive Granger does not.
- Zero-based trial-table row positions are the internal trial identity. Original DataFrame index
  labels are retained only as provenance.
- The configured whole interval supplies the shared PCA fitting interval. Before and after exactly
  partition it.
- Successfully loaded spike data are assumed to cover requested windows. Zero spikes mean observed
  silence; no coverage mask or acquisition-boundary exclusion is implemented in this pass.
- The first UI integration supports metadata-driven loading only.
- The first standard-regression delivery includes both unit OLS and PC OLS.
- The computational core returns explicit in-memory records and tables. Versioned persistence,
  run summaries, logs, and session/batch runners are isolated at the execution boundary.

## Design priorities

The design follows `docs/SoftwareDesign.md`:

- Keep numerical functions deterministic and explicit about shapes, units, and axis meanings.
- Prefer small module-level functions over a behavior-heavy analysis class.
- Pass arrays, masks, identifiers, and small immutable configuration records rather than a large
  session object through the computational core.
- Separate loading/UI concerns from scientific computation.
- Add abstraction only where a concept is already shared: prepared activity, fold assignment,
  history matrices, regional PCA, fit records, and score aggregation.
- Treat unavailable results as expected scientific output with explicit reasons, not as conditions
  to repair by changing the requested model.
- Preserve the current public behavior of existing PCA, condition, and plotting modules.

No existing PCA API will be changed to satisfy the regression-specific zero-variance policy.

## Project architecture

### Architecture goals

The architecture must make the scientific calculation readable from top to bottom without forcing
a reviewer through Streamlit, filesystem, or estimator-framework machinery. It therefore uses
one-way dependencies:

```text
Existing metadata/loaders/condition definitions
                    |
                    v
Configuration and prepared scientific data contracts
                    |
                    v
Preparation -> regional PCA -> linear / Poisson / Granger numerics
                    |
                    v
Pure session pipeline and tidy in-memory results
                    |
                    v
Persistence/runners and plotting
                    |
                    v
Read-only webapp adapter over completed runs
```

Dependency rules:

1. Scientific numerical modules never import Streamlit, filesystem runners, or plotting code.
2. Plotting consumes completed result tables and never prepares data or fits a model.
3. Persistence serializes validated records but never recalculates a metric.
4. The webapp discovers and renders completed saved runs only. It does not load raw spikes, build
   computation configurations, invoke runners, construct design matrices, or call estimators.
5. `run_session.py` is the composition root: it joins existing loaders, the pure pipeline,
   persistence, plots, logging, and summaries.
6. `run_batch.py` delegates one independent session at a time to the single-session entry point;
   it contains no scientific branching.
7. Existing `spike_behavior.trials`, session metadata, and spike-loading contracts remain the
   authorities for their current responsibilities. The new package wraps or calls them rather than
   copying their logic.
8. Cross-package imports point into the new package only through documented public records and
   entry points. The initial implementation does not add a plugin registry, base estimator class,
   factory hierarchy, Protocol layer, or generic analysis framework.

The package is deliberately somewhat finer-grained than one large regression module because OLS,
Poisson, and Granger have different validity and scoring contracts. Within each module, prefer a
short sequence of explicit functions over classes with hidden state.

### Planned package structure

Create a focused package:

```text
src/neural_analysis/interregional/
    __init__.py
    configuration.py
    records.py
    preparation.py
    pca.py
    linear.py
    poisson.py
    granger.py
    pipeline.py
    plotting.py
    persistence.py
    run_session.py
    run_batch.py
    README.md
```

Add a thin web-app module:

```text
src/neural_analysis/webapp/interregional_views.py
```

Add tests alongside the existing neural-analysis tests:

```text
src/tests/neural_analysis/test_interregional_configuration.py
src/tests/neural_analysis/test_interregional_records.py
src/tests/neural_analysis/test_interregional_preparation.py
src/tests/neural_analysis/test_interregional_pca.py
src/tests/neural_analysis/test_interregional_linear.py
src/tests/neural_analysis/test_interregional_poisson.py
src/tests/neural_analysis/test_interregional_granger.py
src/tests/neural_analysis/test_interregional_pipeline.py
src/tests/neural_analysis/test_interregional_plotting.py
src/tests/neural_analysis/test_interregional_persistence.py
src/tests/neural_analysis/test_interregional_scripts.py
src/tests/neural_analysis/test_interregional_webapp.py
```

The test files are separated by scientific responsibility, not by implementation phase. They may
be introduced incrementally as each phase begins.

Add one portable configuration example:

```text
docs/examples/neural_analysis/interregional_regression_config.json
```

Update the existing user-facing documentation only after interfaces stabilize:

```text
src/neural_analysis/README.md
```

### Layer ownership and dependency direction

| Layer | Files | Owns | Must not own |
|---|---|---|---|
| Existing inputs | `session_metadata.py`, spike loaders, `spike_behavior/trials.py`, LFP filter semantics | Metadata resolution, aligned spike sources, current condition definitions | Regression models, new saved schema |
| Contracts | `configuration.py`, `records.py` | Immutable settings, shapes/units, status/reason codes, result-table schemas | Loading, fitting, plotting, writes |
| Scientific preparation | `preparation.py`, `pca.py` | Counts, masks, fold identity, histories, leakage-safe regional transforms | Streamlit, disk paths, figure styling |
| Numerical engines | `linear.py`, `poisson.py`, `granger.py` | Fit validity, coefficients, predictions, model-specific metrics | Trial-condition interpretation, file I/O, UI state |
| Orchestration | `pipeline.py` | Ordered condition/window/fold/direction execution and local failure isolation | Raw path discovery, plotting, serialization format |
| Run boundary | `persistence.py`, `run_session.py`, `run_batch.py` | Versioned outputs, skip/rerun behavior, logs, summaries, session-level parallelism | New scientific formulas |
| Presentation | `plotting.py`, `webapp/interregional_views.py`, small `app.py`/`session_inputs.py` routes | Figures and read-only saved-result selectors/status tables | Raw-data loading, run actions, estimators, or design matrices |

### End-to-end data flow

The ordinary single-session flow is:

1. Resolve one metadata session and two explicit, disjoint PFC/HPC population selections.
2. Validate configuration and trial-table columns without loading large arrays in dry-run mode.
3. Build the scientific eligibility and requested-condition masks. Build the complete-session fold assignment
   only when a CV stage is requested.
4. Load aligned spikes once per selected probe/population.
5. Build aligned regional count tensors over the configured whole interval for scientifically eligible
   trial rows, in ascending zero-based row order.
6. When PC CV is requested, fit regional PCA once per fold using only fold-training trials. When PC
   Granger is requested, fit the separate descriptive basis over all scientifically eligible rows
   in the union of requested conditions.
7. For each direction, condition, and window, build history matrices once and reuse them for every
   target sharing the design.
8. Fit and score matched restricted/full models, preserving target and row identity.
9. Aggregate fold rows into complete target summaries, then target summaries into population
   medians/IQRs.
10. Return `InterregionalResults` without writing or plotting.
11. At the run boundary, persist the validated result, render figures, and write the run log and
    scientific summary.
12. The webapp discovers only atomically completed run directories, loads the same persisted
    tables, and displays them without starting or resuming computation.

Stage ordering applies inside this flow: standard OLS outputs become stable first, Poisson tables
are added only afterward, and descriptive Granger tables are added last.

### Public API boundary

Keep `interregional/__init__.py` small. Export only:

- the top-level configuration and population-selection records;
- `InterregionalResults` and documented status/reason values;
- `prepare_interregional_session(...)`;
- `run_linear_cross_validation(...)`;
- `run_poisson_cross_validation(...)` after WP9;
- `run_descriptive_granger(...)` after WP10; and
- validated result load/save entry points.

All count, fold, PCA, fit, and scoring helpers remain module-private or module-level imports for
tests unless a real second caller establishes a stable public need.

### Existing-code change map

| Existing path | Planned change | Constraint |
|---|---|---|
| `src/neural_analysis/webapp/session_inputs.py` | Add one display/view option and metadata-only availability routing if required | No change to existing view semantics |
| `src/neural_analysis/webapp/app.py` | Add one small early route to `interregional_views.py` | No regression science or large new control block in `app.py` |
| `src/neural_analysis/README.md` | Add setup, dry-run/new commands, output locations, and webapp inspection instructions | User-facing summary only; do not duplicate the complete spec |
| `src/neural_analysis/spike_behavior/trials.py` | Reuse only | No new condition ontology or behavior change |
| `src/neural_analysis/population/pca.py` | Reuse conventions only | Do not alter its existing zero-variance behavior/API |
| `src/neural_analysis/session_metadata.py` | Reuse explicit probe/population metadata | No anatomy inferred from names; avoid schema change in first pass |

Any implementation discovery requiring another existing file or a public-interface change is a
stop-and-replan condition. The Sol supervisor records the evidence here before expanding scope.

### `configuration.py`

Define small frozen dataclasses and side-effect-free validation functions. The exact constants are:

```text
CONFIG_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"
ANALYSIS_VERSION = "interregional-regression-v1"
COVERAGE_ASSUMPTION_VERSION = "implicit-complete-v1"
N_CV_FOLDS = 5
CV_GROUP_COLUMN = "cur_block"
ALLOWED_BIN_SIZES_S = (0.5, 0.1, 0.05, 0.02)
CANONICAL_CONDITIONS = (
    "all", "correct_rewarded", "incorrect", "omission", "switch", "stay",
    "omission_switch", "omission_stay", "incorrect_switch", "incorrect_stay",
)
DEFAULT_CONDITIONS = ("all", "correct_rewarded", "incorrect", "omission", "switch", "stay")
```

`rewarded` remains an internal legacy alias returned by an existing helper; it is not accepted as a
saved regression condition. Requested conditions must be unique and are normalized to canonical
order.

The exact frozen records are:

- `RegionalPopulationConfig`: `role` (`"PFC"` or `"HPC"`), nonempty `probe_id`,
  `channel_source="metadata_quality"` (`"metadata_quality"` or `"explicit"`), sorted unique zero-
  based `selected_channels`, `require_inside_brain=true`, nonempty sorted unique
  `channel_quality_labels=("good",)`, `unit_quality_column="group"` (`"group"` or `"KSLabel"`), and
  nonempty sorted unique `unit_quality_labels=("good", "mua")`. For `metadata_quality`,
  `selected_channels` is empty and the resolver selects the stated quality labels/inside-brain rows,
  intersected with metadata `unit_channels` when present. For `explicit`, `selected_channels` must be
  nonempty and channel-quality fields are retained as provenance but do not further filter channels.
  Unit-quality filtering always applies. This unresolved record contains no cluster IDs.
- `ResolvedRegionalPopulation`: all normalized selection fields plus ordered `selected_channels`,
  ordered `cluster_ids`, and equally sized ordered qualified `unit_ids`. Resolution validates that
  every unit belongs to the stated probe/channel/unit-quality selection and that the PFC/HPC
  qualified unit sets are nonempty and disjoint. Selected channels and integer cluster IDs are
  ascending; qualified unit IDs follow that cluster order.
- `AnalysisWindows`: `whole_start_s=-2.0`, `split_s=0.0`, and `whole_stop_s=2.0`. Before is
  `[whole_start_s, split_s)`, after is `[split_s, whole_stop_s)`, and whole is
  `[whole_start_s, whole_stop_s)`. All values are finite, `whole_start_s < split_s < whole_stop_s`,
  and `split_s` is exactly the selected alignment boundary at zero seconds.
- `TemporalConfig`: `bin_size_s=0.1`, `lag_bins=1`, and `order_bins=1`. The bin size must be one of
  `ALLOWED_BIN_SIZES_S`; lag and order are positive integers. A window whose available bin count is
  not greater than `lag_bins + order_bins - 1` is recorded unavailable for that window rather than
  invalidating unrelated windows.
- `PCAConfig`: `pfc_components=10` and `hpc_components=10`, both positive integers.
- `FilterConfig`: requested conditions, `choice="all"`, `context="all"`, and an empty tuple of
  `excluded_trial_rows`. Choice/context values are `"all"`, `"left"`, or `"right"`; exclusions are
  sorted unique nonnegative zero-based row positions. Session preparation rejects an exclusion that
  is greater than or equal to the trial-table row count.
- `InterregionalAnalysisConfig`: schema/analysis/coverage versions, `session_metadata_path`, PFC and
  HPC population configurations, `alignment="choice_time"` (`"choice_time"` or `"start_time"`),
  windows, `prediction_windows=("before", "after", "whole")`, temporal/PCA/filter records,
  `representations=("units",)`, and
  `analyses=("ols_cv",)`. Allowed representations are `"units"` and `"pcs"`; allowed analysis
  stages in canonical order are `"ols_cv"`, `"poisson_cv"`, `"linear_granger"`, and
  `"poisson_granger"`.
- `RunOptions`: the non-scientific `output_root` loaded from the JSON `run` section plus CLI-only
  `rerun` and optional batch-worker override. These fields are not part of
  `InterregionalAnalysisConfig` or the run fingerprint.

The analysis dependency validator requires units and `ols_cv` when `poisson_cv` is requested,
because its matched OLS/Poisson MSE comparison is required. Both Poisson stages require units.
Granger stages have no runtime dependency on CV stages: their later position describes
implementation order and canonical execution order when stages are combined, not a requirement to
rerun an earlier analysis. PCs never enter either Poisson stage. Prediction windows,
representations, and stages must be nonempty and unique and are normalized to their documented
canonical order.

Window/bin compatibility is tested with
`isclose(duration_s / bin_size_s, round(...), rtol=0, atol=1e-9)`. The resolved integer bin count is
the rounded quotient. Validation rejects incompatible window geometry, empty selections, duplicate
or overlapping unit identities after resolution, unsupported values, and invalid stage
dependencies. It does not load data or inspect Streamlit state.
`session_metadata_path` and a nonnull `output_root` must be normalized absolute paths; relative
paths are rejected so input resolution and handoffs do not depend on a caller's working directory.

The one portable JSON format maps directly to these records and has no undocumented keys:

```json
{
  "config_schema_version": "1",
  "analysis_version": "interregional-regression-v1",
  "coverage_assumption_version": "implicit-complete-v1",
  "session_metadata_path": "/absolute/path/to/session_metadata.json",
  "populations": {
    "PFC": {
      "probe_id": "probe_a", "channel_source": "metadata_quality", "selected_channels": [],
      "require_inside_brain": true, "channel_quality_labels": ["good"],
      "unit_quality_column": "group", "unit_quality_labels": ["good", "mua"]
    },
    "HPC": {
      "probe_id": "probe_b", "channel_source": "metadata_quality", "selected_channels": [],
      "require_inside_brain": true, "channel_quality_labels": ["good"],
      "unit_quality_column": "group", "unit_quality_labels": ["good", "mua"]
    }
  },
  "alignment": "choice_time",
  "windows": {"whole_start_s": -2.0, "split_s": 0.0, "whole_stop_s": 2.0},
  "prediction_windows": ["before", "after", "whole"],
  "temporal": {"bin_size_s": 0.1, "lag_bins": 1, "order_bins": 1},
  "filters": {
    "conditions": ["all", "correct_rewarded", "incorrect", "omission", "switch", "stay"],
    "choice": "all", "context": "all", "excluded_trial_rows": []
  },
  "pca": {"pfc_components": 10, "hpc_components": 10},
  "representations": ["units"],
  "analyses": ["ols_cv"],
  "run": {"output_root": null}
}
```

`output_root=null` resolves to `<session_root>/analysis_runs`. CLI `--output-root` overrides it.
`rerun` and batch workers are CLI invocation choices and are never written as scientific settings.
The loader rejects unknown keys at every level so a typo cannot silently change an analysis. The
run boundary rejects an output root inside the Git repository; analysis outputs must stay in the
session hierarchy or another explicit nonrepository path.

All public functions and dataclasses document input types, array shapes, axes, physical units, and
return contracts.

### `records.py`

Define simple data containers without fitting behavior:

- `RegionalCountTensor` for counts, trial rows, original index labels, bin edges, and qualified unit
  IDs.
- `FoldAssignment` for trial row, original index, `cur_block`, and test-fold ID.
- `HistoryMatrices` for responses, target histories, source histories, and row identities.
- `RegionalPCATransform` for retained/omitted units, training mean/standard deviation, components,
  and explained-variance metadata.
- `FitStatus` values and stable unavailability reason codes.
- `InterregionalResults`, containing a small set of tidy DataFrames rather than one deeply nested
  dictionary.

Every textual identifier/status/reason/JSON column uses pandas `string`; every flag uses `boolean`;
every potentially unavailable integer field uses nullable `Int64`; every metric uses nullable
`Float64`. Nonnullable `trial_row`, `fold_scores.fold_id`, and always-defined requested/count
columns use `int64`; nullable fold IDs in provenance/PCA tables use `Int64`. JSON-list/scalar
columns are canonical compact JSON strings, not Python object values.
Every table is sorted by its primary key before save. The exact result tables are:

1. `fold_assignments`, primary key `session_id, trial_row`: `original_index_repr`,
   `block_value_json`, `block_present`, `fold_id`, `status`, `reason`. Missing-block rows have
   nullable `fold_id` and `not_applicable/missing_block`; assigned rows are `ok` with empty reason.
2. `trial_membership`, primary key `session_id, trial_row, alignment, condition`: provenance plus
   `original_index_repr`, `reward_status_valid`, `alignment_valid`, `choice_match`,
   `context_match`, `user_included`, `block_present`, `condition_match`, `scientific_eligible`,
   `condition_included`, `cv_included`, and `scientific_exclusion_reasons_json`.
   `scientific_eligible` is the condition-independent mask; `condition_included` additionally
   requires `condition_match`; `cv_included` additionally requires `block_present`. A missing
   block's CV reason is recorded by `fold_assignments`.
3. `fold_scores`, primary key
   `session_id, direction, representation, model_family, condition, window, target_id, fold_id`:
   `evaluation_scope`, `target_rank`, `restricted_status`, `restricted_reason`, `full_status`,
   `full_reason`, paired `status`, paired `reason`, `n_train_trials`, `n_test_trials`,
   `n_train_rows`, `n_test_rows`, `train_row_set_sha256`, `test_row_set_sha256`,
   `restricted_feature_count`, `full_feature_count`,
   `restricted_rank`, `full_rank`, `restricted_df_resid`, `full_df_resid`, plus nullable
   `restricted_converged`, `full_converged`, `restricted_iterations`, `full_iterations`,
   `r2_restricted`, `r2_full`, `delta_r2`, `mse_restricted`, `mse_full`,
   `deviance_restricted`, `deviance_full`, `null_deviance`,
   `deviance_explained_restricted`, `deviance_explained_full`, and
   `delta_deviance_explained`. Metrics not belonging to the model family are null.
4. `target_summaries`, primary key
   `session_id, evaluation_scope, direction, representation, model_family, condition, window,
   target_id, metric_name`: `target_rank`, `status`, `reason`, `requested_folds`, `valid_folds`, and
   nullable `mean_value`. This is a CV-only table: `evaluation_scope` is always `held_out_cv` and
   `requested_folds` is five. It is long by metric so valid MSE is retained when R-squared or
   deviance explained is unavailable. A mean is present only when all five values for that metric
   are defined; otherwise status is `incomplete_folds`, reason is `incomplete_requested_folds`, and
   `mean_value` is null. Granger target values remain in `granger_scores` and feed population
   summaries directly.
5. `population_summaries`, primary key
   `session_id, evaluation_scope, direction, representation, model_family, condition, window,
   metric_name`: `status`, `reason`, `n_targets`, `q25`, `median`, and `q75`. With no complete
   contributing targets, quartiles are null and the row is `not_applicable/no_complete_targets`.
6. `pca_fits`, primary key `session_id, scope, fold_id, region`: `status`, `reason`, requested and
   actual component counts as `requested_components`, `actual_components`,
   `n_training_trials`, `n_training_observations`, `retained_unit_ids_json`, and
   `omitted_unit_ids_json`. The final two are canonical JSON arrays. `fold_id` is null only for
   `scope="descriptive"`.
7. `granger_scores`, primary key
   `session_id, direction, representation, model_family, condition, window, target_id`:
   `evaluation_scope="in_sample"`, `target_rank`, `restricted_status`, `restricted_reason`,
   `full_status`, `full_reason`, paired `status`, paired `reason`, `diagnostic`, `n_trials`, `n_rows`,
   `restricted_feature_count`, `full_feature_count`, `restricted_rank`, `full_rank`,
   `restricted_df_resid`, `full_df_resid`, nullable `restricted_converged`, `full_converged`,
   `restricted_iterations`, `full_iterations`,
   `sse_restricted`, `sse_full`, `linear_granger`, `llf_restricted`, `llf_full`,
   `deviance_restricted`, `deviance_full`, `likelihood_ratio`, and
   `mean_deviance_improvement`.

`direction` is exactly `"HPC_to_PFC"` or `"PFC_to_HPC"`; `representation` is `"units"` or
`"pcs"`; `model_family` is `"ols"` or `"poisson"`; windows are `"before"`, `"after"`, or
`"whole"`. Unit `target_id` is qualified; PC `target_id` is `PFC:PC01`, `HPC:PC01`, and so on, with
one-based display rank in nullable `target_rank`. CV fold/target/population rows use
`evaluation_scope="held_out_cv"`; Granger rows use `"in_sample"`. Fold IDs are zero-based 0-4.
CV `metric_name` is restricted to metrics that are comparable across folds. OLS uses
`r2_restricted`, `r2_full`, `delta_r2`, `mse_restricted`, and `mse_full`. Poisson uses
`deviance_explained_restricted`, `deviance_explained_full`,
`delta_deviance_explained`, `mse_restricted`, and `mse_full`. Raw restricted/full/null deviance
remains fold-level diagnostic evidence and is never aggregated. Granger population summaries use
`linear_granger`, `likelihood_ratio`, or `mean_deviance_improvement` as applicable.
Train/test row-set hashes are lowercase 64-character hexadecimal strings whenever the corresponding
row set was constructed, and null when no such rows exist.
Materialize the complete applicable requested key grid: five folds for every requested unit target
or requested PC rank, direction, representation, supported model family, condition, and prediction
window. Unavailable cells are rows with null metrics, never absent keys. Granger materializes the
analogous grid without fold. Poisson/PC combinations are inapplicable and are not grid members. This
invariant makes reruns, stage additions, and user inspection comparable.

The exact status vocabulary is `ok`, `metric_unavailable`, `fit_unavailable`,
`incomplete_folds`, and `not_applicable`. An `ok` row uses an empty reason string. Stable reason
codes are:

```text
missing_block, invalid_reward_status, invalid_alignment, choice_filter_mismatch,
context_filter_mismatch, user_excluded, no_eligible_trials,
no_train_trials, no_test_trials, no_train_rows, no_test_rows, history_exceeds_window,
no_units, pca_no_variable_units,
pca_insufficient_components, constant_training_target, rank_deficient_restricted,
rank_deficient_full, nonpositive_df_restricted, nonpositive_df_full, nonfinite_coefficients,
nonfinite_predictions, constant_test_target, zero_null_deviance,
poisson_nonconverged_restricted, poisson_nonconverged_full,
poisson_fit_error_restricted, poisson_fit_error_full, nonpositive_poisson_mean,
zero_granger_residual, nested_fit_inconsistency, incomplete_requested_folds,
no_complete_targets
```

Only reasons from this vocabulary may be persisted. Unexpected programming/data-contract failures
remain exceptions and are logged at the run boundary rather than converted to an invented reason.
Restricted/full status fields report each fit independently. Paired status is `fit_unavailable` if
either fit is unavailable (restricted reason takes precedence if both fail), otherwise
`metric_unavailable` when the primary paired metric is undefined, otherwise `ok`; the numeric
diagnostic columns still expose both fits. This precedence is presentation only and never discards
the model-specific status/reason fields. Restricted/full fit statuses use only `ok` or
`fit_unavailable`; summary-only `incomplete_folds` and `not_applicable` never appear there.
`granger_scores.diagnostic` is either empty or `nested_roundoff`; it records the one tolerated
near-zero nested-fit correction without mislabeling a valid row as unavailable.

For OLS rows, all convergence/iteration fields are null. For a Poisson fit that returns,
`*_converged` is true only when its Boolean convergence result is true and no convergence warning
was captured; `*_iterations` is the integer IRLS iteration count from
`fit_history["iteration"]`. A fit that raises before returning has both fields null and a model-
specific `poisson_fit_error_*` reason. These four fields, together with status and reason, are the
complete persisted convergence diagnostic; warning histories and estimator objects are not saved.

`trial_membership.scientific_exclusion_reasons_json` is a canonical JSON array in this fixed order:
`invalid_reward_status`, `invalid_alignment`, `choice_filter_mismatch`,
`context_filter_mismatch`, `user_excluded`. It is empty iff `scientific_eligible=true`.
`condition_included` additionally requires the named `condition_match`; `cv_included` additionally
requires `block_present`. The corresponding `fold_assignments` row uses `missing_block` when absent.
Run/loader errors such as
`insufficient_blocks`, `unsupported_saved_version`, and `corrupt_saved_result` are exceptions
recorded in the log and, for a failed run, `failure.json`; they are not result-row reasons.

Configuration and coverage-assumption metadata remain attached to the top-level result record.
Large per-bin design matrices and fitted estimator objects are not retained after scoring.

`InterregionalResults` has exactly: `schema_version=RESULT_SCHEMA_VERSION`, `analysis_version`,
`coverage_assumption_version`, canonical `configuration`, `session_id`, `resolved_populations`,
`whole_bin_edges_s`, `units_and_axes`, `randomness_used`, nullable `random_seed`, and the seven named
tables. `units_and_axes` is a plain canonical mapping with exactly these entries:

```text
count_tensor: axes [trial, time_bin, unit], value_unit spike_count_per_bin
whole_bin_edges_s: axis [time_bin_edge], unit seconds_relative_to_alignment
history_row_identity: axis [observation], fields [trial_row, target_bin_position]
```

`whole_bin_edges_s` is `float64` with shape `(n_whole_bins + 1,)` and the exact validated end
points from configuration.

The persisted run manifest records the generating entry point. Apart from the configured
`session_metadata_path`, expanded consumed-file paths, file hashes, runtime versions, and Git
identity do not enter this pure computational record. Tables for stages not yet implemented are
present with their frozen empty schemas, which keeps later additions backward compatible. Large
per-bin count/design arrays and fitted estimator objects are not retained.
Deterministic paths record `randomness_used=false` and `random_seed=null` rather than inventing a
seed.

### `preparation.py`

Responsibilities:

- Build aligned integer count tensors for both explicitly selected populations.
- Enforce identical trial-row and bin-edge axes between the regions.
- Produce the authoritative base trial mask and requested condition masks using existing trial
  definitions and LFP-style choice/context filters.
- Create the deterministic session-level five-fold `cur_block` assignment when CV is requested.
- Select a condition/window without losing original trial identity.
- Construct within-trial, within-window lag histories.
- Return shared full-comparison eligibility rows for restricted and full fits.

Proposed public functions:

```text
build_regional_count_tensor(...)
build_analysis_trial_masks(...)
build_block_fold_assignment(...)
select_window_bins(...)
build_history_matrices(...)
fingerprint_row_identities(...)
```

`build_regional_count_tensor` is a new regression-local implementation. It uses Pynapple's aligned
tensor construction with the existing population builder's event/window conventions, then
transposes explicitly to `(trial, time_bin, unit)`, verifies finite nonnegative integer-valued
counts within the `int64` range, and casts to `int64`. It obtains trial metadata with `trial_df.iloc[trial_rows]`, never
`.loc`, because `trial_rows` are positions. It must not round rates, extract or change a helper in
the existing PCA module, or change any existing binning API. Spikes at a bin's left edge are
included; the configured final right edge is excluded. Tests compare these boundaries directly.

The prepared regional tensor trial axis contains exactly the ascending trial rows where the
scientific eligibility mask is true, even when `all` is not itself a requested output condition. Named
condition masks are indexed onto this fixed axis. Invalid-alignment, filtered, and user-excluded
rows remain in provenance tables but are never passed to Pynapple. Missing-block rows remain in the
tensor for descriptive Granger but are excluded by the CV mask.

Construct whole-window edges as
`whole_start_s + arange(n_whole_bins + 1, dtype=float64) * bin_size_s`, then assign the first and
last values exactly to `whole_start_s` and `whole_stop_s`. Before/after selection uses integer edge
positions derived during validation, not repeated floating comparisons. Thus both regions and all
windows share byte-identical bin-edge arrays.

`build_block_fold_assignment` receives the complete session trial table, constructs one sample per
zero-based row with nonmissing `cur_block`, and calls `GroupKFold(n_splits=5, shuffle=False)` on that
unfiltered row universe. Split enumeration defines fold IDs 0 through 4. Missing-block rows retain
no fold and are excluded from the CV mask. At least five distinct nonmissing blocks are required
when any CV stage is requested. The mapping is built once before alignment, choice/context,
condition, or user-exclusion filtering and is reused by every requested CV stage. A Granger-only
run neither requires five blocks nor constructs folds.

Each nonmissing block value must be a string, non-Boolean integer, or finite float scalar. Convert
NumPy scalars to Python scalars and normalize integral finite floats to integers before canonical
JSON encoding, so `1` and `1.0` cannot split one logical block. Pass the encoded strings to
`GroupKFold` and persist them as `fold_assignments.block_value_json`. Strings remain distinct from
numbers; nonintegral finite floats remain floats. Reject arrays, mappings, Booleans, infinities, and
NaNs.

The scientific eligibility mask is the existing `valid` reward-status mask AND finite selected-
alignment time AND choice/context matches AND not user-excluded. The `all` condition equals this
mask. Every other named condition intersects it with the corresponding existing canonical condition
mask. The CV mask additionally requires block present; Granger uses condition eligibility without
that CV-only requirement. If a non-`all` choice/context filter is requested and its required
trial column is absent, preparation raises a configuration/input error instead of silently
producing an empty analysis. The analysis reuses the LFP column names and side encoding: choice
uses `action`, context uses `state_int`, `left` is numeric 1, and `right` is numeric 0 after numeric
coercion. It intentionally strengthens the existing helper's absent-column behavior to an error.
`all` does not require that filter's column. Nonfinite values fail a selected left/right filter.

`build_history_matrices` receives tensors already aligned on trial and bin axes. Its contracts are:

```text
target_activity: (trial, selected_window_bin, target_feature)
source_activity: (trial, selected_window_bin, source_feature)
trial_rows:       (trial,)

returns:
    responses:       (observation, target_feature)
    target_history:  (observation, order * target_feature)
    source_history:  (observation, order * source_feature)
    row_trial:       (observation,)
    row_target_bin:  (observation,), zero-based position on the whole-window bin axis
```

History-column ordering is most-recent lag first, then stable feature order. The response matrix is
kept multi-target so all targets sharing one design do not require repeated history construction.
Before/after selection preserves whole-window bin positions rather than renumbering selected bins.

`fingerprint_row_identities` makes saved row provenance compact and comparable. It rejects duplicate
observation identities, sorts the integer `(trial_row, target_bin_position)` pairs
lexicographically, encodes them as an ASCII JSON list of two-element lists with separators `,` and
`:`, and returns the SHA-256 of the UTF-8 bytes. Thus a fingerprint describes the row set rather
than incidental array order.
Restricted and full fits must still receive identical ordered row arrays in memory. Persist both
training and test row-set fingerprints for every fold; OLS/Poisson comparison requires both hashes
to match.

The first-pass coverage assumption is represented in configuration/result metadata. Preparation
does not infer missing coverage. It still rejects nonfinite arrays and mismatched axes.

### `pca.py`

Implement regression-specific regional PCA without modifying
`src/neural_analysis/population/pca.py`.

Proposed public functions:

```text
fit_regional_pca(training_activity, unit_ids, requested_components)
transform_regional_activity(activity, fitted_transform)
fit_fold_regional_pcas(...)
fit_descriptive_regional_pcas(...)
```

The fitted transform records the training mean and population standard deviation (`ddof=0`) before
omission, retained and omitted unit IDs, unwhitened component matrix, actual component count, and
observation count. Calculations use `float64`. Omit a unit when its training standard deviation is
zero or nonfinite. Fit scikit-learn PCA with
`n_components=min(requested_components, n_training_observations, n_retained_units)`,
`svd_solver="full"`, and `whiten=False`. This path uses no random number generator and records
`random_seed=null` plus `randomness_used=false`.

Fold PCA uses only training trials pooled over the configured whole interval and union of requested
conditions after scientific eligibility. The same fitted regional transforms are reused for both
directions and every requested condition/window in that fold. Descriptive PCA uses all rows in that
same scientifically eligible requested-condition union, has a distinct scope label, and cannot be
passed to CV pipeline functions.

Expected PC target ranks run from 1 through the requested count for the target region. When a
fold's actual PCA dimension is smaller, emit each missing rank's `fold_scores` row as
`fit_unavailable/pca_insufficient_components` rather than omitting the key. This makes incomplete-
fold summaries and saved-schema comparisons explicit.

Do not add PC sign matching, cross-fold component alignment, whitening, or a post-PCA scaler.
Downstream scores are sign-invariant; component loadings and scores retain scikit-learn's
deterministic full-SVD sign convention.

### `linear.py`

Implement unpenalized OLS and linear scoring as explicit NumPy operations.

Proposed public functions:

```text
fit_ols_targets(design, responses)
predict_ols_targets(design, coefficients)
score_ols_predictions(observed, restricted_predictions, full_predictions)
summarize_complete_cv_targets(fold_scores)
```

The design passed to fitting already includes an explicit intercept column. Validate rank and
positive residual degrees of freedom before fitting. Since all target responses share a design,
use one clear multi-right-hand-side least-squares solve where targets have the same eligible rows;
calculate target-specific constant-response and score status separately.

Do not use `sklearn.LinearRegression` merely to obtain functionality already explicit in
`numpy.linalg.lstsq`. Record the coefficient count, rank, and residual degrees of freedom.
Call `numpy.linalg.lstsq(design, responses, rcond=None)` explicitly after the shared rank/degree-
of-freedom validation; all fitting arrays are `float64`.

OLS score functions implement held-out R-squared and MSE from the specification without library
replacement values for constant targets. Negative finite R-squared and increments are preserved.
Population summaries use `numpy.quantile(complete_target_values, [0.25, 0.5, 0.75],
method="linear")`; `median` is that 0.5 quantile. They never pool fold rows before quantiling.

### `poisson.py`

This module is absent from the initial OLS delivery and is added only in the Poisson phase.

Proposed public functions:

```text
fit_poisson_target(design, count_response)
predict_poisson_mean(design, fitted_parameters)
poisson_deviance(observed_counts, expected_counts)
score_poisson_predictions(...)
compare_count_prediction_mse(...)
```

The audited environment currently provides statsmodels 0.15.0. Use
`statsmodels.api.GLM(y, design, family=sm.families.Poisson(link=sm.families.links.Log()),
missing="raise")` and `.fit(method="IRLS", maxiter=100, tol=1e-8, scale=None,
cov_type="nonrobust", full_output=True, disp=False, wls_method="qr")`. The design already contains
the one explicit intercept; do not call `add_constant`, `fit_regularized`, or pass
weights/exposure/offset. Require
`result.converged is True`, an integer `result.fit_history["iteration"]`, finite `result.params`, and
finite `result.llf`. Calculate test means as `exp(test_design @ params)` and require them to be
positive and finite. Reconfirm these signatures and result attributes against the installed source
immediately before WP9 in case the environment changed.

Validate shapes, values, rank, and degrees of freedom before entering the estimator call. Around
`.fit(...)` only, capture `statsmodels` convergence and perfect-separation warnings and catch known
data-dependent fitting failures (`PerfectSeparationError`, `FloatingPointError`,
`numpy.linalg.LinAlgError`, and `ValueError`). A convergence warning or returned
`converged=false` becomes `poisson_nonconverged_*`; perfect separation or a caught fitting exception
becomes `poisson_fit_error_*`. The run log records only the warning/exception class and concise
message with the affected result key. Other exceptions propagate as programming or API-contract
failures. These failures are local to the affected target/model; they do not abort independent
targets.

Fit one target at a time because convergence and constant-response validity are target-specific.
The module returns plain parameters/diagnostics needed for scoring, not statsmodels result objects in
the public result record.

The deviance function is independently implemented and tested against hand calculations, including
zero counts. This prevents estimator-specific pseudo-R-squared conventions from entering the
analysis.

`compare_count_prediction_mse` is a pure derived view over `fold_scores`. It inner-joins OLS and
Poisson unit rows on session, direction, condition, window, target, fold, and evaluation scope,
requires identical train- and test-row-set fingerprints, and defines
`mse_advantage_poisson = mse_ols - mse_poisson`. Positive values therefore favor Poisson on
held-out error. It returns a target mean only when all five fold differences exist. The full-model
comparison is primary; the restricted comparison remains available for inspection. No additional
saved table is introduced.

### `granger.py`

This module is added after both CV model families are complete.

Proposed public functions:

```text
compute_linear_granger(...)
compute_poisson_granger(...)
validate_nested_fit_improvement(...)
```

Reuse the same preparation and fit functions. Linear Granger calculates
`log(SSE_restricted / SSE_full)` on identical in-sample rows. Poisson Granger retains both
`2 * (llf_full - llf_restricted)` and `deviance_restricted - deviance_full`, verifies their numerical
agreement, and displays the deviance difference divided by row count.

CV result types must not be accepted by the Granger summary functions. PC Granger receives only the
separately labeled descriptive all-data PCA scores.

No significance tests or p-value fields are added.

### Numerical comparison policy

Use named constants shared by linear, Poisson, and Granger tests:

```text
BIN_GEOMETRY_ATOL = 1e-9
METRIC_RTOL = 1e-9
METRIC_ATOL = 1e-12
```

`numpy.linalg.matrix_rank` with its documented default SVD tolerance decides design rank; tests use
clearly full-rank or deficient matrices rather than near-threshold examples. Training-response
constancy is exact for integer unit counts and uses zero variance under the same `ddof=0`
calculation for floating PC responses. Held-out `SST` and Poisson null deviance are treated as zero
when `isclose(value, 0, rtol=METRIC_RTOL, atol=METRIC_ATOL)`.

For a quantity theoretically nonnegative under a nested in-sample fit, a negative value within
`METRIC_ATOL + METRIC_RTOL * max(abs(restricted_value), abs(full_value), 1.0)` is recorded as zero
with `diagnostic="nested_roundoff"`; a more negative value is `nested_fit_inconsistency`. Poisson likelihood-
ratio and deviance-difference equivalence uses the same scale-aware tolerance. Tests assert exact
hand calculations where feasible and use these constants only for floating comparisons; no
scientific score is rounded before persistence.

### `pipeline.py`

Assemble the focused components without hiding scientific choices.

Proposed public entry points:

```text
prepare_interregional_session(...)
run_linear_cross_validation(...)
run_poisson_cross_validation(...)
run_descriptive_granger(...)
```

`prepare_interregional_session` loads no paths. It receives already loaded aligned spikes, trial
columns, selected unit identities, and configuration, and produces count tensors, masks, fold
assignments, and shared metadata.

Each run function loops explicitly over:

```text
direction -> representation -> condition -> window -> fold -> target/model pair
```

PCA fitting occurs outside condition/window loops at fold scope. History matrices occur outside
target loops. Restricted/full design construction happens once per shared row set. Failures are
captured at the narrowest independent result level and do not abort unrelated configurations.

Keep OLS, Poisson, and Granger entry points separate. Do not add a general-purpose estimator
registry or mode flag that causes one lower-level function to perform unrelated model families.

### `plotting.py`

Plot only from result tables; never refit or recover missing results while plotting.

Proposed public functions:

```text
plot_cv_increment_summary(...)
plot_absolute_cv_scores(...)
plot_model_family_mse(...)
plot_granger_summary(...)
```

All plots use readable fonts, opaque white backgrounds, black text/axes, individual target points,
median/IQR overlays, stable target identity, and captions containing the scientific caveat and
coverage assumption. Captions summarize the plotted content, contributing targets, and scientific
interpretation. Export PNG by default. Light mode is the current plan default; dark mode is not
added unless selected during plan review.

Conditions are separated more widely than directions within a condition. Unlike metrics never
share a numerical axis. Unavailable and contributing counts are visible.

### `webapp/interregional_views.py`

Keep Streamlit-specific code out of the analysis package.

Responsibilities:

- Accept the selected metadata session, derive `<session_root>/analysis_runs` with the same session
  resolver as the CLI, and discover completed interregional run directories only there.
- Reject incomplete, failed, corrupt, or unsupported-version runs through the validated loader.
- Render a completed-run selector labeled with timestamp, analysis version, short fingerprint, PFC
  and HPC population descriptions, and completed stages.
- Render display-only selectors for the conditions, windows, representations, model families, and
  metrics actually present in that saved result.
- Display the saved scientific configuration read-only, including population roles, channels,
  qualities, alignment, filters, bin/lag/order, PCA counts, coverage version, input/code identity,
  warnings, unavailable reasons, tables, and figures.
- If no completed run exists, show the exact documented `dry-run` and `new` CLI commands; do not
  offer an in-app compute button.

This saved-only boundary is intentional: long scientific computation is explicit, logged, and
restartable from the command line, while Streamlit rerenders remain read-only. Computation settings
are edited in the validated JSON configuration, not in transient widget state. The UI states that
the metadata route is required and loaded spike coverage is assumed. A nondefault CLI output root
remains valid but is not auto-discovered by the first-pass webapp; its saved figures/tables are
inspected directly.

Integrate the view through a small branch in `src/neural_analysis/webapp/app.py`. Avoid placing the
new scientific pipeline directly in that already-large file.

### `persistence.py`, `run_session.py`, and `run_batch.py`

Keep file I/O and run orchestration outside fitting and plotting functions.

`persistence.py` provides explicit functions such as:

```text
run_fingerprint(config, resolved_inputs, content_hashes, code_identity, runtime_versions)
create_working_run_directory(output_root, timestamp)
finalize_run_directory(working_path, short_fingerprint)
save_interregional_result(result, path)
load_interregional_result(path)
write_run_summary(...)
write_run_log(...)
```

Use one immutable run directory as both the machine-readable and human-readable output. There is no
second version/fingerprint result tree and no mutable canonical result file:

```text
<session_root>/analysis_runs/
    .interregional_regression_<YYYYMMDD>T<HHMMSSffffff>Z.incomplete/
        ...work in progress...
    interregional_regression_<YYYYMMDD>T<HHMMSSffffff>Z_<fingerprint12>/
        config.json
        input_manifest.json
        result.pkl
        run.log
        summary.md
        run_session.py
        run_batch.py
        figures/
```

The timestamp is UTC with microseconds. After input hashing and reuse detection, a new computation
creates the hidden `.incomplete` directory exclusively in the same parent as its final path. On
success, close all files, validate every required artifact, then atomically rename that directory to
the final timestamp/fingerprint name. A collision is an error rather than permission to reuse or
overwrite a path. Final-run discovery ignores every `.incomplete` directory.

The pickle contains the pure `InterregionalResults` computational record only. Execution
provenance, paths, content hashes, runtime versions, generating entry point, and Git identity remain
in the manifest. It still contains only project-generated records and pandas/NumPy data; untrusted
pickle files must never be loaded. The copied scripts are the exact runner files used.
`summary.md` records the goal, sessions,
scripts, configuration, warnings, unavailable-result counts, output locations, and a scientific
summary. That summary names the compared direction/condition/window, contributing target count,
median/IQR of the primary metric, and whether the metric is held-out or in-sample; it does not turn
descriptive values into significance claims. It explicitly states the complete-coverage and causal
limitations.
`run.log` records the runtime environment, parameters, processed session IDs, warnings or errors,
and execution time.

The `run_fingerprint` is SHA-256 over canonical JSON containing the configuration/result schema,
analysis and coverage versions; all `InterregionalAnalysisConfig` fields except the location-only
`session_metadata_path`; session ID; resolved ordered unit IDs; logical role, byte size, and
streamed SHA-256 content hash for every consumed metadata, trial, aligned-spike, sorter, cluster,
and channel-quality input file; Git HEAD; and the exact computation runtime versions listed below.
Absolute paths are recorded in the manifest but excluded from the hash, so relocating identical
data does not change run identity. Timestamp, output root, rerun, worker count, and Streamlit
version are excluded. `config.json` and `input_manifest.json` retain expanded values rather than
only the hash. Real runs require a clean tracked worktree so Git HEAD identifies the code; dry-run
reports dirty tracked paths and refuses `new` until the user resolves them. `new` also rejects
untracked Python files under `src/neural_analysis`, because imported untracked code would not be
identified by Git HEAD. Other untracked files do not change code identity.

`input_manifest.json` has exact top-level keys `manifest_schema_version="1"`, `run_fingerprint`,
`session_id`, `git_head`, `entrypoint`, `runtime_versions`, `resolved_populations`, and `files`.
`entrypoint` is exactly
`src.neural_analysis.interregional.run_session.run_single_session`. `runtime_versions` has exactly
`python`, `numpy`, `pandas`, `scipy`, `pynapple`, `scikit_learn`, `statsmodels`, and `matplotlib`;
all enter the fingerprint because they can affect the persisted numerical or figure artifacts. Each
file entry has `logical_role`, normalized absolute `resolved_path`, `size_bytes`, and `sha256`.
Entries are sorted by `logical_role` then path before writing; fingerprint input is sorted by
logical role and content identity, excluding the path. `new` hashes large files in fixed-size chunks
without deserializing them, before reuse detection or fitting. Dry-run validates each path and
reports file sizes and runtime versions but does not compute full content hashes or claim a final
fingerprint.

If computation fails after the working directory exists, append the failure to `run.log`, write a
small `failure.json` containing timestamp, last entered stage, and exception class/message, and
leave the directory with its `.incomplete` suffix. An abrupt interruption may leave only partial
artifacts; this is still unambiguous because the suffix excludes it from reuse and display. The
first pass neither resumes nor automatically removes incomplete directories. A retry recomputes
from the beginning in a new timestamped working directory. This directory-level completion marker
is sufficient; do not add a status state machine or per-file durability protocol.

Run-stage names are exactly `input_validation`, `input_hashing`, `preparation`, `ols_cv`,
`poisson_cv`, `linear_granger`, `poisson_granger`, `persistence`, `figures`, and `summary`; skip
stages not requested. The CLI logs a start/end record and elapsed seconds for each entered stage.
For a new computation, the working-directory log begins by recording the already measured
validation/hash timings; a reuse-only invocation reports its preflight and matching path to the
console without mutating the prior run. This is the progress contract; there is no background-task
or Streamlit progress protocol.

`run_session.py` loads one metadata session, applies skip/rerun logic, runs the requested completed
analysis stages, saves the result, and writes figures/log/summary. `run_batch.py` reads a session
list, supports a dry-run mode, calls the same single-session function, and parallelizes across
sessions with `concurrent.futures.ProcessPoolExecutor`. Its proposed worker count is the available
CPU-core/session minimum or an explicit lower override. Each process receives one
configuration path and returns a small status/run-path record; results and arrays are never passed
between workers. Results remain session-separated.

For `new`, the runner computes input hashes, scans finalized directories below the output root, and
skips computation when one has the same fingerprint, printing that run path. `--rerun` always
creates a new immutable timestamped directory and recomputes even when the fingerprint matches; it
never overwrites or mutates the earlier run. The webapp is read-only and never creates a run
directory.

### Offline command and documentation surface

The first stable command surface is intentionally small:

```bash
uv run python -m src.neural_analysis.interregional.run_session dry-run \
  --config /path/to/session/interregional_regression_config.json

uv run python -m src.neural_analysis.interregional.run_session new \
  --config /path/to/session/interregional_regression_config.json

uv run python -m src.neural_analysis.interregional.run_session new \
  --config /path/to/session/interregional_regression_config.json --rerun

uv run python -m src.neural_analysis.interregional.run_batch dry-run \
  --config-list /path/to/interregional_regression_configs.txt

uv run python -m src.neural_analysis.interregional.run_batch new \
  --config-list /path/to/interregional_regression_configs.txt --workers 4
```

`dry-run` validates paths, metadata roles, trial columns, filters, CV block count when applicable,
selected units,
window/bin compatibility, requested feature counts, output paths, and a work/memory estimate without
deserializing full spike arrays, hashing every large input, or fitting models. `new` computes the
content hashes and reuses a completed matching run fingerprint by default. `--rerun` creates a new
complete immutable run with the same run fingerprint.

The batch config list contains one UTF-8 configuration path per nonblank, non-comment line. Batch
parallelism is across sessions only. Start from
`min(os.cpu_count() or 1, number_of_sessions, --workers when supplied)` as the proposed count and
report the advisory memory estimate and planned concurrency. After WP8 measures a representative
session peak, the user must explicitly approve the exact batch worker count; no batch is implied by
single-session approval, and the rough estimate never automatically certifies or rejects a run. A
per-session failure is logged without merging or deleting successful session results.

Do not add a notebook-only launcher, generic workflow engine, database, automatic cluster wrapper,
or a second scientific configuration format.

`src/neural_analysis/interregional/README.md` is the maintainer map. It documents:

- package dependency direction and each file's public entry points;
- trial/count/history/PCA/result axes and units;
- configuration, status/reason, persistence, and fingerprint contracts;
- test-file ownership for each module;
- how OLS, Poisson, and Granger stages remain separate; and
- the exact extension boundary for a future coverage mask.

`src/neural_analysis/README.md` is the scientist-facing quickstart. It documents:

- how to copy and edit the portable example configuration;
- metadata-only PFC/HPC selection requirements;
- exact dry-run, single-session, rerun, and batch commands;
- output locations and how to inspect a run in the existing webapp;
- the complete-coverage assumption and rank-unavailable behavior; and
- the predictive/noncausal interpretation boundary.

Neither README duplicates the complete specification or this execution log.

## Dependency plan

No new dependencies are proposed.

| Dependency | Intended use | Constraint |
|---|---|---|
| NumPy | count arrays, histories, OLS, metrics | Explicit `float64` fitting; counts remain integer until design conversion |
| pandas | trial columns and tidy result tables | Preserve row-position/index distinction |
| SciPy | numerical backend used by scientific dependencies | No new direct algorithm when NumPy/statsmodels already provides it |
| Pynapple/current spike helpers | aligned bin counts | Verify half-open bin behavior against existing tests/source |
| scikit-learn | `GroupKFold`, PCA | Five nonshuffled groups; full-SVD, `whiten=False`; no default scaler |
| statsmodels | Poisson GLM | Explicit Poisson/log/unpenalized settings and convergence checks |
| Matplotlib | figures | Plotting consumes records only |
| Streamlit | completed-run discovery and display | Read-only adapter; no raw loading or computation |
| Python standard library | paths, JSON metadata, pickle, logging, timestamps, process pool | No new serialization or workflow dependency |

Before first use in production code, inspect the installed API for `GroupKFold`, PCA, and
statsmodels GLM. Record important estimator assumptions in docstrings and tests. All Python commands
use `uv run`. The documentation audit observed NumPy 2.4.3, scikit-learn 1.8.0, statsmodels 0.15.0,
and Pynapple 0.11.0; implementation preflight records the versions again and stops if changed APIs
invalidate a frozen call contract.

## Implementation orchestration - Sol supervisor and Terra workers

This section defines future implementation ownership. It does not authorize implementation in the
current chat. The named model/effort combinations must be rechecked against the active runtime at
implementation start; never silently substitute a different model, effort, or agent topology.

### Fixed roles

**Lead Sol supervisor - `gpt-5.6-sol`, high reasoning**

- owns authorization, scientific/architectural decisions, package order, shared integration files,
  the live handoff, commits, and final package acceptance;
- audits relevant code and worktree state, freezes each package's allowlist/tests/contracts, and
  independently reproduces RED and GREEN rather than accepting worker summaries as evidence; and
- resolves review conflicts or asks the user when a decision changes science or scope.

**Terra package worker - `gpt-5.6-terra`, high by default, xhigh where assigned**

- receives one bounded task and exact file allowlist;
- writes tests and stops at RED, then implements only after Sol verifies and commits those tests;
- makes the smallest readable change, reports exact commands/results and risks, and stops at GREEN;
  and
- never commits, expands scope, changes tests merely to pass, runs unauthorized data/batch/external
  work, or spawns agents.

Use Terra xhigh for fold/history identity, numerical model validity, PCA leakage, persistence/run
identity, Poisson, Granger, and full synthetic integration. Use Terra high for straightforward
configuration records, plotting, UI adapters, documentation, and authorized command execution.

**Independent Sol gate reviewer - `gpt-5.6-sol`, high or xhigh as assigned**

- read-only reviews stable test designs and GREEN diffs for scientific drift, leakage, axes/units,
  row identity, estimator settings, saved-schema identity, failure handling, and missing tests; and
- is mandatory at WP2-WP5, WP7, and WP9-WP11 but never edits or replaces the lead's final gate.

An optional `gpt-5.6-terra` medium read-only scout may inspect an independent call site, installed
API, or bounded failure log. It supplies evidence only and performs no mutation.

No `max` or `ultra` assignment is planned. If a scientific or architectural discrepancy cannot be
resolved at the assigned level, stop and ask the user rather than escalating model effort or scope
silently.

### Shared-worktree and concurrency rules

- At most one Terra worker edits at once; tests and implementation are sequential. Reviewers inspect
  only stable diffs, and only Sol manages agents or commits.
- Sol records HEAD/status around assignments, stages only reviewed package paths while the worker is
  idle, and stops on unexpected or uncertain ownership. Workers report any new file need rather
  than expanding their allowlist.
- The package handoff update normally accompanies the implementation commit and identifies it as
  `this commit`; Git supplies the hash. Use a separate documentation commit only for a substantive
  later correction, not as routine package ceremony.

### Mandatory package sequence

For every implementation package:

1. Sol audits authorization, code/usages, worktree ownership, package contract, and allowlist.
2. Terra writes only the named tests, demonstrates RED, and stops.
3. Sol inspects and reproduces RED, obtains the required test-design review, and commits tests only.
4. The same Terra worker implements the committed contract and demonstrates focused GREEN.
5. Sol inspects the diff, runs focused and affected suites, and obtains the required stable-diff
   review; accepted fixes return to Terra.
6. Sol updates the live handoff, stages reviewed implementation/handoff files only, and creates the
   implementation commit after every gate passes.

A test change after the tests-only commit requires an explicit dated explanation that the
requirement changed or the test was genuinely wrong. It is never changed merely to accommodate an
implementation.

### Terra assignment contract

Every worker prompt states the package, model/effort, one objective, exact allowlist, relevant v3
data/scientific contracts, tests and commands, expected gate, prohibited actions, current ownership
facts, and required evidence return. It must be sufficient without relying on chat memory.

When explicit model overrides are available, spawn Terra with `model="gpt-5.6-terra"` and the effort
listed in the package map, and spawn the reviewer with `model="gpt-5.6-sol"`. Use a bounded context
fork rather than relying on full inherited history; the prompt carries the authoritative contract.
If an override is rejected or reports a different model/effort, the agent must not edit and Sol
must ask the user how to proceed.

Use a follow-up to move the same Terra worker from tests to implementation after the RED commit.

### Interruption and recovery contract

- An interrupted report is not a gate. Sol records and reproduces the last verified HEAD, diff, and
  command before assigning a replacement with the same role, effort, allowlist, and stop gate.
- A replacement lead completes the resume checklist and establishes state from Git and reproduced
  commands, not summaries. Restart an interrupted review only against a stable diff.
- Stop for the user when ownership is uncertain, evidence conflicts, a numerical/API contract is
  uncertain, or the required solution expands the approved scientific/interface scope.

### Work-package Sol/Terra map

| Package | Primary files/responsibility | Terra assignment | Independent Sol gate | Package completion evidence |
|---|---|---|---|---|
| WP0 | v3, this plan, architecture, handoff state | None; Sol documentation only | User approval | Approved documents and documentation-only baseline commit |
| WP1 | `configuration.py`, `records.py`, configuration/record tests | High, tests-only then implementation | High final contract review | Validated records/status schema; focused GREEN; separate commits |
| WP2 | `preparation.py`, preparation tests | Xhigh, tests-only then implementation | Xhigh test-design and final identity review | Count/mask/fold/history contracts GREEN; 19/39 examples verified |
| WP3 | `linear.py`, unit-only `pipeline.py`, linear/pipeline tests | Xhigh, tests-only then implementation | Xhigh test-design and numerical final review | Bidirectional direct-unit OLS synthetic results and complete fold evidence |
| WP4 | `pca.py`, PC pipeline extensions, PCA/pipeline tests | Xhigh, tests-only then implementation | Xhigh leakage test-design and final review | Held-out perturbation leakage tests GREEN; unit OLS unchanged |
| WP5 | `persistence.py`, `run_session.py`, `run_batch.py`, result/persistence/script tests | Xhigh, tests-only then implementation | Xhigh saved-identity/restart/provenance review | Round-trip, skip/rerun, logging, dry-run, session isolation GREEN |
| WP6 | `plotting.py`, `interregional_views.py`, bounded router edits, README/example, plot/UI/docs tests | High, serialized tests then implementation/documentation | High final UI-boundary review | Standard OLS figures/UI/docs GREEN; no science in UI |
| WP7 | Synthetic full standard-regression integration and performance evidence | Xhigh, integration tests then bounded fixes | Xhigh scientific/leakage/performance review | Deterministic unit/PC OLS output, saved run, plots, timing/memory record |
| WP8 | Exact user-designated standard-regression session | High command runner only; no source edits | High evidence review plus user inspection | Approved command/run path, logs, warnings, figures, user decision |
| WP9 | `poisson.py`, pipeline/persistence/plot/UI extensions, Poisson tests | Xhigh, tests-only then implementation | Xhigh numerical/API/test-design and final review | Unpenalized Poisson CV and matched MSE GREEN; OLS unchanged |
| WP10 | `granger.py`, descriptive PCA/pipeline/plot/UI extensions, Granger tests | Xhigh, tests-only then implementation | Xhigh scientific/formula/scope review | Separate descriptive results/figures GREEN; no inference fields |
| WP11 | Final synthetic integration, docs, affected neural suite, and separately approved full-session inspection | Xhigh tests/fixes; later high command runner only | Xhigh final code review; high run-evidence review; user scientific gate | Full regression suite, immutable run evidence, user inspection, final handoff |

WP8 and the real-session portion of WP11 are command-only packages requiring separate explicit user
approval of the exact session, configuration, command, and output root. Completion of a preceding
code package is not authorization to run experimental data.

## Test-driven workflow

Every implementation work package follows the repository's required RED-GREEN-REFACTOR sequence:

1. Add only the phase's tests.
2. Run the targeted tests with `uv run pytest ...` and confirm the expected failures.
3. Commit the failing tests before adding production implementation.
4. Implement the smallest clear solution that satisfies the approved contracts.
5. Run the targeted tests until green.
6. Run directly related existing neural-analysis tests.
7. Refactor only while all relevant tests remain green.
8. Commit implementation separately from the preceding test-only commit.

Tests must not be weakened to accommodate implementation behavior. If a test is genuinely wrong,
document the requirement mismatch before modifying it.

Synthetic random fixtures use explicit seeds and record their axes/units. Most contract tests use
small hand-constructed arrays so expected values are independently calculable.

## Phase 1 (WP1-WP3): contracts, preparation, and direct-unit OLS

### Deliverable

A pure-Python path that accepts two prepared regional spike populations plus a trial table and
returns fold-, target-, and population-level OLS CV results in both directions for direct units. No
PCA, Poisson, Granger, plotting, or Streamlit code is included yet.

### Tests written first

`test_interregional_configuration.py`:

1. Load the exact documented JSON into the frozen records and round-trip canonical JSON.
2. Accept the documented defaults and reject unknown keys at every JSON level.
3. Reject unsupported bin sizes, nonpositive lag/order, and nonintegral window/bin geometry under
   the named tolerance.
4. Reject bounds that do not exactly partition whole at zero.
5. Keep unresolved selection fields separate from resolved qualified unit identities and reject
   overlapping resolved PFC/HPC identities.
6. Reject an empty condition request, noncanonical/duplicate conditions, and the legacy `rewarded`
   alias; preserve canonical condition order.
7. Validate analysis-stage dependencies, reject Poisson when `units` is absent, and keep Poisson
   rows unit-only when units and PCs are both requested; allow either Granger stage without its CV
   counterpart.
8. Preserve schema, analysis, and coverage versions while excluding run options from run-identity
   input.

`test_interregional_records.py`:

1. Construct all seven empty tables with the exact frozen columns and dtypes.
2. Reject missing/extra columns, wrong dtypes, duplicate primary keys, unsorted keys, and unknown
   status/reason/diagnostic values.
3. Accept the complete applicable unit/PC key grid and require explicit unavailable rows rather
   than missing keys.
4. Preserve valid metric-specific target summaries when another metric is incomplete.
5. Validate canonical JSON scalar/list columns, the fixed scientific-exclusion order, and distinct
   scientific-eligibility, condition-membership, and CV-membership fields.
6. Validate exact `InterregionalResults` versions, metadata fields, whole-window edges, the frozen
   axes/units mapping, randomness fields, and empty future-stage schemas without importing fitting,
   persistence, or run-provenance code.

`test_interregional_preparation.py`:

1. Count spikes correctly in hand-checked half-open bins, including spikes exactly on interior and
   terminal edges.
2. Return `(trial, bin, unit)` integer counts with qualified unit order preserved.
3. Produce matching trial/bin axes for PFC and HPC and reject mismatches.
4. Treat an empty bin as observed zero under the complete-coverage assumption.
5. Define scientific `all` as valid alignment plus experimenter-reward validity, choice/context
   filters, and user exclusions; define CV eligibility by additionally requiring `cur_block`.
6. Intersect named conditions with the scientific eligibility mask without changing existing condition
   definitions.
7. Use zero-based row positions for masks while preserving nontrivial original DataFrame indexes as
   provenance.
8. Assign all trials in one `cur_block` to one test fold.
9. Produce exactly five deterministic folds and the same assignment on repeated calls.
10. Leave missing-block rows unassigned and excluded from CV with reason `missing_block`, reject
    fewer than five distinct nonmissing session blocks when CV is requested, and never fall back to
    random splitting.
11. Reuse one session fold mapping when condition masks select different trial subsets.
12. Build lag-1/order-1 histories with 19 rows for a two-second 100-ms window and 39 for a
    four-second window.
13. Build lag/order greater than one in documented most-recent-to-oldest feature order.
14. Never cross trial or selected-window boundaries.
15. Keep response, target history, source history, trial row, and target-bin identities aligned.
16. Derive restricted and full designs from one shared full-comparison row mask.
17. Build folds from the complete nonmissing-block row universe before alignment, condition,
    choice/context, or user-exclusion filters, using one sample per trial row.
18. Use positional trial-table indexing with a deliberately nontrivial original DataFrame index.
19. Raise a clear input error when a requested non-`all` choice/context filter lacks its required
    trial column.
20. Normalize integral numeric block labels so `1` and `1.0` remain in one fold while strings remain
    distinct from numbers.
21. Retain a scientifically eligible missing-block trial for Granger while excluding it from CV.
22. Make row-set fingerprints independent of pair order, sensitive to either identity coordinate,
    and invalid for duplicate `(trial_row, target_bin_position)` pairs.

`test_interregional_linear.py`:

1. Add exactly one explicit intercept column.
2. Recover hand-constructed full-rank OLS coefficients and predictions.
3. Fit multiple target responses against one design without changing target order.
4. Reject a rank-deficient design and report its rank/feature count.
5. Reject nonpositive residual degrees of freedom.
6. Mark a constant training target unavailable without invalidating other targets.
7. Match hand-calculated held-out SSE, SST, R-squared, MSE, and incremental R-squared.
8. Preserve negative finite absolute and incremental R-squared.
9. Mark R-squared unavailable for constant held-out responses while retaining defined MSE.
10. Require restricted/full scores to have identical ordered held-out row identities and persist
    canonical train/test row-set fingerprints.
11. Require all five paired folds for the primary target mean; retain incomplete fold rows without
    presenting a partial mean as complete.
12. Calculate population median and quartiles from target means, not from pooled fold values.

`test_interregional_pipeline.py` initially covers units only:

1. Run HPC-to-PFC and PFC-to-HPC on a seeded synthetic coupled-count session.
2. Detect stronger incremental prediction in the deliberately coupled synthetic direction without
   asserting a publication-style significance threshold.
3. Reuse the identical fold assignment and row identities across directions and restricted/full
   pairs, and preserve their canonical train/test fingerprints in fold rows.
4. Keep condition/window failures local while returning unrelated valid results.
5. Retain qualified target unit IDs and the complete configuration in result records.

### Implementation sequence

1. Add validated configuration and record dataclasses.
2. Add raw count-tensor preparation and axis validation.
3. Add authoritative masks and deterministic block folds.
4. Add window selection and history construction.
5. Add OLS fit, prediction, metrics, and availability handling.
6. Add unit-only bidirectional CV orchestration and tidy aggregation.
7. Run the new tests and the existing spike-behavior, population-PCA, session-metadata, and package
   import tests to check for regressions.

### Acceptance gate

- All Phase 1 tests pass.
- Existing public PCA and condition tests remain unchanged and pass.
- A synthetic session produces inspectable bidirectional unit OLS tables with exact fold/row
  provenance.
- No UI or plotting code is needed to inspect the tables.

## Phase 2 (WP4): training-only regional PCA and PC OLS

### Deliverable

Add fold-local regional PCA and PC-rank OLS using the same preparation, folds, scoring, and result
contracts. Unit OLS behavior remains unchanged.

### Tests written first

`test_interregional_pca.py`:

1. Fit PFC and HPC transforms separately.
2. Pool only training trials, the configured whole interval, and the union of requested conditions.
3. Include an overlapping-condition trial only once in the PCA fitting pool.
4. Omit and report zero-variance training units without changing the existing general PCA API.
5. Apply training means, standard deviations, and components to held-out data.
6. Demonstrate no leakage by changing held-out values and confirming the fitted transform is
   unchanged.
7. Use `whiten=False` and perform no post-PCA score scaling.
8. Retain at most the requested count and report the actual available component count.
9. Reuse the same target-region axes for restricted/full models and both directions in a fold.
10. Reuse fold transforms across requested conditions and windows.
11. Mark a component rank incomplete if it is missing in any requested fold.
12. Keep descriptive all-data PCA scope structurally distinct from fold PCA scope.
13. Use population standard deviation (`ddof=0`), `float64`, `svd_solver="full"`, and
    `whiten=False`, with no random state or randomized solver.
14. Set the actual component count to the minimum of request, training observations, and retained
    units.

`test_interregional_pipeline.py` gains PC cases:

1. Run bidirectional PC OLS without changing unit OLS results.
2. Label targets as fold-specific PC ranks in CV outputs.
3. Prevent an all-data descriptive PCA transform from entering CV.
4. Preserve the same folds and held-out rows used by direct-unit analyses where eligibility is the
   same.

### Implementation sequence

1. Add regional PCA transform records.
2. Add fit/transform functions with explicit zero-variance omission.
3. Add fold-level shared PCA construction outside condition/window loops.
4. Extend the OLS pipeline to PC-rank responses and histories.
5. Populate the already-frozen PCA scope and effective-dimension fields.
6. Run targeted tests plus all existing population PCA/decoding tests.

### Acceptance gate

- Unit and PC OLS both pass complete synthetic leakage tests.
- Existing PCA behavior and tests are untouched.
- Fold-specific rank summaries cannot be mistaken for one fixed all-session trajectory.

## Phase 3 (WP5-WP8): reproducible standard-regression delivery and inspection

### Deliverable

Expose the completed unit/PC OLS workflow in the metadata-driven application, render standard CV
summary/inspection figures, and add the versioned run boundary. This completes the reproducible
standard-regression milestone before Poisson work.

### Tests written first

`test_interregional_plotting.py`:

1. Plot individual target values plus median and IQR from supplied result tables.
2. Group conditions more widely than the two directions within a condition.
3. Keep stable plotted-point target IDs.
4. Label the primary axis Incremental CV R-squared.
5. Display contributing and unavailable target counts.
6. Produce absolute restricted/full inspection plots without refitting.
7. Include the prediction/causality caveat and complete-coverage assumption in captions.
8. Use a white opaque background and readable labels.

`test_interregional_webapp.py`:

1. Discover only finalized interregional run directories for the selected metadata session.
2. Ignore `.incomplete` directories and reject corrupt, fingerprint-mismatched, or unsupported-
   version finalized directories.
3. Label each saved run with timestamp, version, short fingerprint, two regional selections, and
   completed stages.
4. Offer display selectors only for conditions, windows, representations, model families, and
   metrics present in the loaded result.
5. Show population roles/selections, scientific configuration, input/code identity, coverage
   assumption, warnings, and unavailable reasons read-only.
6. Explain that legacy manual loading is unsupported and loaded coverage is assumed.
7. Display independent valid rows when another result row is unavailable.
8. Show documented CLI instructions when no completed result exists.
9. Import no raw-spike loader, pipeline run function, design helper, or estimator and expose no run,
   resume, or recompute action.
10. Prove an ordinary widget rerender creates or modifies no filesystem path.

`test_interregional_persistence.py`:

1. Produce the same SHA-256 run fingerprint for semantically identical canonical configurations and
   resolved input/code identities.
2. Change the fingerprint when a scientific parameter, version, selected unit order, file
   size/content hash, Git HEAD, or computation runtime version changes; ignore relocated absolute
   paths, timestamp, output root, rerun, and worker count.
3. Create one hidden `.incomplete` working directory beneath a temporary session data root, never
   the repository; reject repository-contained output roots and do not create a second result tree.
4. Round-trip every named table with exact columns, dtypes, primary-key order, configuration,
   coverage, units/axes, and no-randomness declaration; keep input/code identity in the manifest
   rather than `InterregionalResults`.
5. Reuse an identical validated finalized run by default; make `--rerun` create a different
   directory with the same fingerprint and never overwrite the first.
6. Atomically rename the same-parent working directory only after every required artifact validates;
   never discover it under the final name earlier.
7. Leave a caught failure as `.incomplete` with `run.log` and `failure.json`; ignore and never
   resume either caught-failure or abruptly partial working directories.
8. Retain but never reuse corrupt, fingerprint-mismatched, or unsupported-version finalized runs.
9. Copy the exact session/batch runner files and create config, manifest, result, log, summary, and
   figure directory.
10. Include goal, session, scripts, warnings, unavailable counts, scientific interpretation, and
    coverage/causal caveats in the Markdown summary.
11. Reject duplicate primary keys, wrong columns/dtypes, unknown status/reason values, and unsafe
    load targets outside a discovered project run directory.
12. Let dry-run validate paths and report sizes without hashing large files; require `new` to hash
    every consumed file before fitting.
13. Refuse `new` for dirty tracked files or untracked Python files under `src/neural_analysis` while
    ignoring unrelated untracked data/artifacts for code identity; record the exact manifest entry
    point and runtime-version keys.

`test_interregional_scripts.py`:

1. Run one synthetic metadata session through the public single-session composition root; the UI
   only loads its completed output.
2. Skip an existing matching result by default and honor an explicit rerun request.
3. Print planned sessions, stages, and output paths without computation in batch dry-run mode.
4. Keep independent session results separate and never combine unit columns.
5. Start batch workers at the lesser of CPU cores, session count, and an explicit lower override;
   report the advisory estimate in dry-run without treating it as an automatic cap or fit verdict.
6. Propagate a failed session as a logged per-session failure without discarding successful sessions.
7. Accept the portable example configuration through the production configuration loader.
8. Keep documented `--help`, dry-run, new, and rerun command syntax synchronized with the CLI.

`test_interregional_pipeline.py` gains the WP7 standard-regression integration cases:

1. Execute a seeded two-region synthetic session through configuration, counts, folds, unit OLS,
   fold PCA, PC OLS, aggregation, persistence, reload, and plotting.
2. Reproduce identical scientific tables from identical inputs and configuration.
3. Preserve target/fold/trial/bin identities after save/load.
4. Keep unavailable cells explicit while completing independent valid cells.
5. Confirm the frozen Poisson/Granger-capable tables exist but remain empty, and no Poisson or
   Granger compute/display action is present before its package.
6. Record stage timings, synthetic dimensions, peak-memory measurement method, and output size for
   the bounded performance gate without imposing a brittle wall-clock unit-test threshold.

### Implementation sequence

1. WP5 adds immutable versioned run-directory persistence plus single-session and batch runners.
2. WP5 adds run logging, summary generation, figure-output paths, skip/rerun behavior, and batch
   dry-run.
3. WP6 adds plotting functions over immutable result tables.
4. WP6 adds completed-run discovery and read-only result selectors in
   `interregional_views.py`, plus one small navigation branch in `webapp/app.py`.
5. WP6 adds fold-detail, target-summary, PCA-omission, and unavailability tables plus the package
   README, user quickstart, and portable example configuration.
6. WP7 runs the seeded full standard-regression synthetic integration and bounded timing/memory
   inspection; only bounded fixes are permitted after their own tests-first gate.
7. Sol runs plotting, persistence, script, web-app, session-metadata, channel-quality, package-
   import, and affected existing PCA tests.
8. WP8 is a separate command-only/user gate: after explicit approval of an exact metadata session,
   configuration, command, and output root, run and inspect one standard-regression session.

### Acceptance gate

- The CLI configuration resolves two explicit populations and never infers their anatomical roles;
  the UI faithfully displays those saved selections without recomputation.
- Unit and PC OLS results and diagnostics are inspectable without Poisson/Granger code.
- Figures visibly distinguish direction, condition, target identity, and unavailable counts.
- A completed standard-regression run is reproducible from its saved configuration, result record,
  copied scripts, log, and summary.
- The seeded full synthetic run passes a separate xhigh scientific/leakage review and records
  interpretable timing/memory evidence.
- The user reviews one designated session's output before Phase 4 begins.

## Phase 4 (WP9): Poisson CV and OLS/Poisson count-MSE comparison

### Deliverable

Add unpenalized Poisson GLMs for direct-unit counts, incremental CV deviance explained, and paired
held-out count-MSE comparison with OLS. PCs remain OLS-only.

### Tests written first

`test_interregional_poisson.py`:

1. Reject negative, fractional, or nonfinite count responses.
2. Verify the statsmodels model uses an explicit Poisson family, log link, intercept column, and
   unpenalized fitting path.
3. Recover finite expected means on a seeded synthetic Poisson dataset and retain only convergence
   Boolean plus integer IRLS iteration count.
4. Detect nonconvergence, convergence/perfect-separation warnings, known fitting exceptions,
   nonfinite parameters, and nonpositive/nonfinite predicted means with stable local reasons.
5. Match hand-calculated Poisson deviance, including zero-count terms.
6. Match hand-calculated null deviance and deviance explained.
7. Preserve negative finite deviance-explained values and increments.
8. Mark normalized scores unavailable when held-out null deviance is zero.
9. Require identical restricted/full test rows and null denominator.
10. Calculate OLS and Poisson MSE on identical unit-count responses without rounding or clipping.
11. Keep restricted/restricted and full/full model-family comparisons paired.
12. Reject Poisson for PC targets.
13. Build comparisons by an exact inner join on session, direction, condition, window, unit target,
    fold, and evaluation scope; reject differing train- or test-row-set fingerprints.
14. Mark a target comparison unavailable unless all five OLS/Poisson fold pairs are defined.
15. Match hand-calculated `mse_advantage_poisson = mse_ols - mse_poisson` and its five-fold target
    mean for full and restricted models.

`test_interregional_pipeline.py` gains Poisson cases:

1. Reuse exactly the OLS fold assignments, target IDs, histories, and test rows.
2. Continue other targets after one target fails convergence or raises a recognized data-dependent
   fit error, while allowing unexpected exceptions to reach the run boundary.
3. Require five valid paired folds for a primary Poisson target summary.
4. Retain only Poisson convergence/iteration diagnostics plus both model families' fold-level MSE.
5. Never aggregate raw restricted/full/null deviance into target or population summaries.

`test_interregional_plotting.py` gains:

1. Incremental CV deviance-explained plots with no R-squared labeling.
2. Separate held-out MSE comparison plots labeled exploratory.
3. No mixed OLS/Poisson primary metric axis.

### Implementation sequence

1. Verify the installed statsmodels GLM API and diagnostics.
2. Add target-wise Poisson fitting and plain diagnostic records.
3. Add independent deviance/null-deviance scoring.
4. Extend the CV pipeline only for direct units.
5. Add a pure derived model-family MSE comparison view and plots; do not add a redundant saved
   table.
6. Extend saved-result display selectors/labels for Poisson unit results; add no compute action.
7. Run all OLS tests unchanged, then the Poisson and UI suites.
8. Inspect runtime and convergence on the same designated session before Phase 5.

### Acceptance gate

- Poisson and OLS use identical scientific rows/folds for direct comparison.
- No estimator-default regularization or pseudo-R-squared enters the output.
- A failed target remains an explicit local result and does not abort the analysis.

## Phase 5 (WP10): descriptive linear and Poisson Granger-style analyses

### Deliverable

Add separate in-sample Granger-style tables and figures after all CV analyses are stable.

### Tests written first

`test_interregional_granger.py`:

1. Match a hand-calculated linear `log(SSE_restricted / SSE_full)` value.
2. Use identical rows and the same row-count denominator for restricted/full fits.
3. Mark zero residual variance and nonfinite ratios unavailable.
4. Treat a substantive negative nested OLS improvement as a consistency failure while tolerating
   documented floating-point roundoff near zero.
5. Match Poisson `2 * (llf_full - llf_restricted)`.
6. Match the equivalent restricted-minus-full deviance and verify numerical agreement.
7. Divide the displayed Poisson improvement by the eligible row count while retaining raw LR.
8. Use all eligible rows without folding or pooling CV residuals.
9. Use the separate descriptive all-data PCA basis for PC linear Granger.
10. Prevent fold-PCA transforms from entering descriptive PC Granger and descriptive transforms
    from entering CV.
11. Add no p-value, significance, or joint-multivariate regional fields.
12. Preserve lag-restricted history exactly when lag exceeds one.
13. Include a scientifically eligible missing-block trial and require no fold construction in a
    Granger-only run.
14. Allow linear or Poisson Granger to run without requesting its CV counterpart.

`test_interregional_plotting.py` gains:

1. Separate linear and Poisson Granger figures with correct metric labels.
2. Prominent in-sample/descriptive labeling and no significance stars.
3. Individual target values, median/IQR, target counts, and stable IDs.
4. No shared numerical axis with CV metrics.

`test_interregional_webapp.py` gains:

1. Granger display options appear only when present in a validated completed saved result.
2. Descriptive PCA and in-sample status are visible.
3. No inference/significance language or compute action appears.

### Implementation sequence

1. Add descriptive regional PCA fitting for PC Granger.
2. Add linear and Poisson Granger calculations over existing fit functions.
3. Add nested-fit consistency checks and explicit unavailable reasons.
4. Add separate Granger result table, aggregation, figure, and saved-result UI display.
5. Run every CV test unchanged to prove scope separation.
6. Inspect one designated session before any cross-session or publication use.

### Acceptance gate

- Granger results are visibly and structurally separate from held-out prediction results.
- Linear and Poisson formulas agree with independent hand calculations.
- No formal inference or implicit causal claim is introduced.

## Phase 6 (WP11): final integration, documentation, and scientific inspection

This is verification and cleanup, not a new analysis method.

1. Run all new interregional tests.
2. Run the full `src/tests/neural_analysis` suite.
3. Run package/import tests.
4. After separate approval of the exact command and output path, inspect one user-designated
   session at 100-ms, lag-1, order-1 defaults.
5. Confirm the 19/39 row-count examples in real prepared metadata.
6. Confirm fold/block/trial identities and selected regional units in displayed diagnostics.
7. Review unavailable rates, especially direct-unit rank failures, before interpreting scientific
   output.
8. Check figure readability and captions.
9. Update user-facing neural-analysis documentation with the metadata-only and complete-coverage
   limitations.
10. Record any desired coverage-mask or alternative persistence-format work as a separate follow-up
    specification.

The full test command is:

```bash
uv run pytest src/tests/neural_analysis
```

Targeted commands should be used during RED/GREEN cycles so failures remain attributable to the
current phase.

## Performance and memory considerations

Correctness and auditability take priority, but avoid obvious repeated work:

- Build each regional count tensor once per session/alignment/bin-size/population selection.
- Build the session fold assignment once.
- Build one history matrix per representation/direction/condition/window/fold row set, not per
  target.
- Fit fold PCA once per region/fold and reuse it across directions, conditions, and windows.
- Use one multi-target NumPy least-squares solve for targets sharing one valid OLS design.
- Keep Poisson target-wise because convergence/status are target-specific.
- Streamlit may cache only validated completed-result loading by immutable run path plus manifest
  identity; it never caches prepared tensors, transforms, designs, or fitted models.
- Do not retain large repeated design matrices in result records.
- In batch runs, parallelize across sessions rather than within a session. Default to the lesser of
  available CPU cores and session count, with an optional lower worker override. No batch run occurs
  until WP8 supplies a measured session peak and the user explicitly approves the exact worker
  count.

For dry-run session dimensions `T` trials, `B` whole-window bins, `N` total selected units,
`R = T * max_window(B_window - (lag + order - 1))` as the conservative maximum design rows,
`P = 1 + order * N` full-design columns, and
`Y = max(PFC units, HPC units, requested PFC PCs, requested HPC PCs)`, report:

```text
count_bytes = 8 * T * B * N
design_bytes = 8 * R * P
response_bytes = 8 * R * Y
estimated_session_peak_bytes = 256 MiB + 3 * count_bytes + 4 * (design_bytes + response_bytes)
```

This is a rough planning estimate, not a trustworthy upper bound: NumPy/LAPACK workspaces,
statsmodels allocations, Python/process overhead, and concurrent system use are not modeled.
Dry-run prints every term and the proposed worker count but does not automatically cap, certify, or
reject a run from this estimate. WP8 measures representative peak memory; the user then approves an
exact batch worker count. Any later automatic memory policy requires its own measured design and is
outside this first pass.

At 20-ms bins and large unit populations, direct full designs may be both wide and rank-invalid.
The implementation reports that outcome rather than allocating recovery searches. Before adding
parallel Poisson fitting, profile the designated session. Add parallelism only if measured runtime
justifies the extra complexity; this single-session first pass does not require it.

Avoid brittle wall-clock assertions in unit tests. Use a documented manual timing check on the
designated session after correctness is established.

## Scientific inspection checklist

Before accepting results from a real session, verify:

- PFC and HPC roles and qualified unit IDs are correct and disjoint.
- The intended region-specific channels/quality filters were applied.
- Trial row positions map to the expected original trial-table labels.
- `cur_block` has at least five groups and each group belongs to one fold.
- Condition counts match the centralized condition definitions and choice/context filters.
- Bin edges and window partitions match the displayed configuration.
- Default histories contribute 19/39 rows per eligible trial for two-/four-second windows.
- Restricted/full train and test row identities are identical.
- Fold PCA omitted-unit and retained-component counts are plausible.
- Direct-unit rank failures are visible rather than silently repaired.
- OLS negative held-out scores are retained.
- Poisson predicted means and convergence diagnostics are finite/valid.
- Granger outputs say in-sample/descriptive and contain no significance interpretation.
- Every figure states the complete-coverage assumption and causal caveat.

## Explicit non-goals for this implementation

- No neural coverage mask or acquisition-boundary inference.
- No legacy manual-input workflow.
- No database, remote object store, transactional workflow engine, or untrusted pickle import.
- No changes to existing PCA semantics.
- No regularization, feature selection, dimensionality matching, or fallback model.
- No automatic lag/order search.
- No PC axis matching across folds.
- No smoothing, recursive prediction, or cross-trial histories.
- No formal Granger inference or joint multivariate regional statistic.
- No temporal/confound correction or causal claim.
- No cross-session pooling of unit columns.

## Risks and deliberate trade-offs

### Direct-unit availability

The strict full-rank requirement means many unit models may be unavailable, especially at 20- or
50-ms resolution or under narrow conditions. This is an intended consequence of the agreed
unpenalized full-population model, not an implementation defect. The PC representation is an
explicit user choice, not an automatic fallback.

### Complete-coverage assumption

A zero count caused by an uncovered recording interval would be misinterpreted as silence. The UI,
result metadata, plan, and v3 specification state this limitation. Initial inspection should use
sessions believed to have continuous coverage of the requested windows.

### Fold-specific PCs

PC-rank aggregation is scientifically less intuitive than fixed session axes, but it prevents
held-out leakage. Plots and records must call these fold-specific rank summaries.

### Shared event structure

Grouped CV does not remove common event drive, movement, slow trends, or other temporal confounds.
The first-pass result remains predictive. Additional controls require a later scientific
specification rather than silent expansion of this pipeline.

### Offline runtime

Unpenalized Poisson fits over many targets and configurations may be slow. Computation therefore
runs through the logged CLI boundary, while the UI reads only complete outputs. The first response
to runtime pressure is fingerprint reuse, explicit CLI stage logging, and measured profiling.
Within-session parallel fitting is considered only after profiling because it adds failure and
memory complexity.

## Approval gate

Implementation should begin only after the user approves:

- the layered package/module boundaries and one-way dependency rules;
- the in-memory result-table schema and single immutable atomic run-directory design;
- the WP0-WP11 order and separate tests-only, implementation, review, and handoff commits;
- one Sol/high supervisor, one bounded Terra writer, independent Sol review at the listed gates,
  and the one-writer shared-worktree rule;
- this plan as the authoritative live progress/handoff record, updated at every package gate;
- metadata-only, read-only completed-run UI support;
- the complete-coverage assumption; and
- inspection of one designated session after the standard OLS milestone and before later stages;
  and
- light-mode PNG figures as the default, unless the user selects dark mode before WP6.

Approval of this plan permits no implementation in the current documentation-only chat. A later
explicit implementation request activates only WP1 after the resume/preflight checklist. It does
not authorize WP8/WP11 real-session commands, batch runs, or external actions; those remain separate
exact-command approvals.

Any change to response units, history boundaries, CV grouping, PCA scope, estimability policy, or
metric formulas requires a specification update before implementation. Any change to package
ownership, model/effort assignments, agent concurrency, saved schema, or implementation order
requires a dated plan revision before work proceeds.
