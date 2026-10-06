# Inter-Regional Neural Regression Implementation Plan

**Status:** Proposed documentation-only plan. This chat is planning-only and does not authorize
production code, tests, commits, experimental-data runs, benchmarks, or other implementation
actions. Implementation may begin only in a later chat after explicit user approval and a fresh
repository audit.

**Scientific authority:** `docs/spec_neural_regression_v3.md`.

## Live handoff snapshot

**Snapshot date:** 2026-10-06 14:52 EDT.

**Current phase:** WP0, documentation architecture and implementation planning. The v3 scientific
specification and this plan exist as new untracked documentation files. No implementation or test
work has begun, and the user has explicitly prohibited implementation in this chat.

**Repository state at this snapshot:**

- branch: `refactor`;
- HEAD before this documentation revision: `6131d8e`;
- plan-owned files: `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`;
- both plan-owned files are currently untracked;
- the worktree contains many unrelated pre-existing untracked files and directories; and
- none of those unrelated entries belongs to this plan or may be staged, changed, removed, or
  absorbed into a later package.

**Completed planning evidence:**

- `src/neural_analysis`, `docs/SoftwareDesign.md`, and
  `docs/spec_neural_regression_updated.md` were audited;
- codebase/specification inconsistencies were identified and resolved with the user;
- v3 records the approved scientific and first-pass data-validity decisions;
- the current plan contains module responsibilities, explicit array/result contracts, a phased
  tests-first inventory, performance considerations, and reproducible run outputs; and
- existing repository plans were inspected for their Sol/Terra, interruption, and authoritative
  handoff patterns before this revision.

**Next exact action:** finish this documentation revision and return it for user review. Do not
create tests or production files in this chat. A later implementation chat begins only after the
user approves the plan and explicitly requests implementation; the incoming Sol supervisor then
performs the resume checklist below and starts WP1, not source edits from WP2 or later.

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
| WP0 | Specification, architecture, orchestration, and handoff plan | Active; documentation only | User review; no implementation in this chat |
| WP1 | Configuration and result contracts | Not authorized | Explicit implementation request and fresh Sol preflight |
| WP2 | Counts, masks, folds, windows, and histories | Not authorized | WP1 GREEN and recorded handoff |
| WP3 | Direct-unit OLS fitting, scoring, and aggregation | Not authorized | WP2 GREEN and numerical test-design gate |
| WP4 | Fold-local regional PCA and PC OLS | Not authorized | WP3 GREEN and leakage test-design gate |
| WP5 | Saved results, run identity, session runner, and batch runner | Not authorized | WP4 GREEN and saved-schema freeze |
| WP6 | Standard-regression plotting, metadata webapp, and documentation | Not authorized | WP5 GREEN and stable standard-result schema |
| WP7 | Standard-regression synthetic integration and bounded performance check | Not authorized | WP1-WP6 focused gates GREEN |
| WP8 | One-session standard-regression scientific inspection | Not authorized | WP7 GREEN plus explicit approval of the exact session/command |
| WP9 | Unit Poisson CV, MSE comparison, plotting, and integration | Not authorized | WP8 user inspection/approval |
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
- test-only, implementation, and documentation commit IDs;
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

## Plan objective and status

This plan implements `docs/spec_neural_regression_v3.md`. It is a proposal for review, not
authorization to begin coding. No production or test code should be written until the plan is
approved. This document is also the authoritative progress and handoff log for future
implementation.

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
- The analysis-wide `all` condition means valid experimenter-reward status intersected with valid
  alignment, choice/context filters, user exclusions, and a nonmissing block for CV.
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
a reviewer through Streamlit, filesystem, or estimator-framework machinery. It therefore uses five
layers with one-way dependencies:

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
Persistence/runners and plotting/webapp adapters
```

Dependency rules:

1. Scientific numerical modules never import Streamlit, filesystem runners, or plotting code.
2. Plotting consumes completed result tables and never prepares data or fits a model.
3. Persistence serializes validated records but never recalculates a metric.
4. The webapp builds configuration, invokes the run boundary, and renders saved/in-memory results;
   it does not construct design matrices or call estimator libraries directly.
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
| Presentation | `plotting.py`, `webapp/interregional_views.py`, small `app.py`/`session_inputs.py` routes | Figures, controls, status tables, explicit run action | Direct estimator or design-matrix calls |

### End-to-end data flow

The ordinary single-session flow is:

1. Resolve one metadata session and two explicit, disjoint PFC/HPC population selections.
2. Validate configuration and trial-table columns without loading large arrays in dry-run mode.
3. Load aligned spikes once per selected probe/population.
4. Build aligned regional count tensors over the configured whole interval.
5. Build the authoritative trial masks and one session-level `cur_block` fold assignment.
6. For each fold, fit regional PCA once when PCs are requested, using only fold-training trials.
7. For each direction, condition, and window, build history matrices once and reuse them for every
   target sharing the design.
8. Fit and score matched restricted/full models, preserving target and row identity.
9. Aggregate fold rows into complete target summaries, then target summaries into population
   medians/IQRs.
10. Return `InterregionalResults` without writing or plotting.
11. At the run boundary, persist the validated result, render figures, and write the run log and
    scientific summary.
12. The webapp displays the same result tables and figures and never redefines a metric.

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

Define small frozen dataclasses and validation functions. Proposed records are:

- `AnalysisWindows`: whole, before, and after half-open bounds in seconds.
- `TemporalConfig`: bin size in seconds, positive lag in bins, and positive order in bins.
- `PCAConfig`: requested component counts for PFC and HPC.
- `FilterConfig`: requested conditions, choice filter, context filter, and excluded trial rows.
- `PopulationSelection`: anatomical role, probe ID, qualified unit IDs, and channel/unit-selection
  metadata.
- `InterregionalConfig`: alignment, the two regional selections, windows, temporal configuration,
  PCA configuration, exactly five folds, and the coverage-assumption version.

Validation is explicit and side-effect free. It rejects overlapping PFC/HPC unit identities,
unsupported bin sizes, nonpositive lag/order, an invalid window partition, an empty condition list,
and Poisson with a PC response. It does not load data or inspect Streamlit state.

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

The result tables have stable, documented columns:

1. `fold_assignments`
   - session, trial row, original index label, block, and test fold.
2. `fold_scores`
   - analysis keys, target ID/rank, fold, status/reason, trial and row counts, design dimensions,
     rank, residual degrees of freedom, restricted/full metrics, paired increment, and optional MSE.
3. `target_summaries`
   - analysis keys, target, completeness, number of valid/requested folds, and mean fold metrics.
4. `population_summaries`
   - analysis keys, metric, contributing targets, median, 25th percentile, and 75th percentile.
5. `pca_fits`
   - fold or descriptive scope, region, requested/actual components, retained/omitted unit IDs, and
     training-observation counts.
6. `granger_scores`, added only in the Granger phase.

Configuration and coverage-assumption metadata remain attached to the top-level result record.
Large per-bin design matrices and fitted estimator objects are not retained after scoring.

The result record also carries an analysis version, deterministic configuration fingerprint,
generating function names, units/axis conventions, and a random-seed field. Deterministic paths
record that no random seed was used rather than inventing one.

### `preparation.py`

Responsibilities:

- Build aligned integer count tensors for both explicitly selected populations.
- Enforce identical trial-row and bin-edge axes between the regions.
- Produce the authoritative base trial mask and requested condition masks using existing trial
  definitions and LFP-style choice/context filters.
- Create the deterministic session-level five-fold `cur_block` assignment.
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
```

`build_regional_count_tensor` should use the same Pynapple/event-alignment conventions as the
existing population tensor builder but return raw integer counts. It must not obtain counts by
rounding displayed firing rates. If a common lower-level bin-count operation can be extracted
without changing existing public behavior, use it; otherwise implement the small regression-
specific function locally and test boundary equivalence.

`build_block_fold_assignment` receives one block value per zero-based trial row and returns exactly
one test-fold ID per row with a nonmissing block. It is built once before condition filtering.

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
    row_target_bin:  (observation,)
```

History-column ordering is most-recent lag first, then stable feature order. The response matrix is
kept multi-target so all targets sharing one design do not require repeated history construction.

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

The fitted transform records the training mean and standard deviation before omission, retained and
omitted unit IDs, unwhitened component matrix, actual component count, and observation count.

Fold PCA uses only training trials pooled over the configured whole interval and union of requested
conditions. The same fitted regional transforms are reused for both directions and every requested
condition/window in that fold. Descriptive all-data PCA has a distinct scope label and cannot be
passed to CV pipeline functions.

Do not add PC sign matching, cross-fold component alignment, whitening, or a post-PCA scaler.

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

OLS score functions implement held-out R-squared and MSE from the specification without library
replacement values for constant targets. Negative finite R-squared and increments are preserved.

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

Use statsmodels GLM with the Poisson family and log link, explicitly confirming unpenalized fitting,
intercept handling, convergence attributes, and prediction semantics against installed statsmodels
source or official documentation immediately before implementation.

Fit one target at a time because convergence and constant-response validity are target-specific.
The module returns plain parameters/diagnostics needed for scoring, not statsmodels result objects in
the public result record.

The deviance function is independently implemented and tested against hand calculations, including
zero counts. This prevents estimator-specific pseudo-R-squared conventions from entering the
analysis.

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

- Render separate PFC and HPC metadata-population selectors with unique widget keys.
- Reuse existing unit-quality/channel-selection behavior independently for each role.
- Render regression-specific alignment, window, condition, choice/context, bin, lag/order, PCA, and
  model controls.
- Convert widget values into validated `InterregionalConfig` and `PopulationSelection` records.
- Invoke cached preparation/analysis entry points.
- Display configuration errors, unavailable-result reasons, tables, and figures.

The UI must state that the metadata route is required and loaded spike coverage is assumed. It must
not mutate the existing single-population state and then attempt to use it for both regions.

Integrate the view through a small branch in `src/neural_analysis/webapp/app.py`. Avoid placing the
new scientific pipeline directly in that already-large file.

### `persistence.py`, `run_session.py`, and `run_batch.py`

Keep file I/O and run orchestration outside fitting and plotting functions.

`persistence.py` provides explicit functions such as:

```text
configuration_fingerprint(config, analysis_version)
result_path(session_root, config, analysis_version)
save_interregional_result(result, path)
load_interregional_result(path)
create_run_directory(output_root, timestamp)
write_run_summary(...)
write_run_log(...)
```

Save the complete named-table result as a project-generated pickle under the original session data
hierarchy, for example:

```text
<session_root>/analysis/interregional_regression/
    <analysis_version>/<configuration_fingerprint>/result.pkl
```

The pickle contains only project-generated configuration records, metadata, and pandas/NumPy data.
Document that untrusted pickle files must never be loaded. A parameter or analysis-version change
produces a different path. An existing complete matching result is reused unless `rerun=True`.

Each execution also creates a human-readable directory outside the repository, normally under the
session data hierarchy:

```text
<session_root>/analysis_runs/
    interregional_regression_<YYYYMMDD>_<HHMMSS>/
        config.json
        result_location.txt
        run.log
        summary.md
        run_session.py
        run_batch.py
        figures/
```

The copied scripts are the exact runner files used. `summary.md` records the goal, sessions,
scripts, configuration, warnings, unavailable-result counts, output locations, and a scientific
summary. It explicitly states the complete-coverage and causal limitations.
`run.log` records the runtime environment, parameters, processed session IDs, warnings or errors,
and execution time.

`run_session.py` loads one metadata session, applies skip/rerun logic, runs the requested completed
analysis stages, saves the result, and writes figures/log/summary. `run_batch.py` reads a session
list, supports a dry-run mode, calls the same single-session function, and parallelizes across
sessions. Its default worker count is the available CPU-core count, with an explicit override for
RAM- or I/O-constrained runs. Results remain session-separated.

The web app invokes the same run-boundary functions only when the user presses an explicit run
action. Ordinary Streamlit rerenders do not create duplicate run directories.

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

`dry-run` validates paths, metadata roles, trial columns, filters, block count, selected units,
window/bin compatibility, requested feature counts, output paths, and a work/memory estimate without
loading full spike arrays or fitting models. `new` reuses a completed matching fingerprint by
default. `--rerun` creates a new timestamped human-readable run directory while leaving the
versioned intermediate result intact or creating a new result when its identity changes.

The batch config list contains one UTF-8 configuration path per nonblank, non-comment line. Batch
parallelism is across sessions only. A per-session failure is logged without merging or deleting
successful session results.

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
| Pynapple/current spike helpers | aligned bin counts | Verify half-open bin behavior against existing tests/source |
| scikit-learn | `GroupKFold`, PCA | Five nonshuffled groups; `whiten=False`; no default scaler |
| statsmodels | Poisson GLM | Explicit Poisson/log/unpenalized settings and convergence checks |
| Matplotlib | figures | Plotting consumes records only |
| Streamlit | controls/caching/display | Thin integration layer only |
| Python standard library | paths, JSON metadata, pickle, logging, timestamps, process pool | No new serialization or workflow dependency |

Before first use in production code, inspect the installed API for `GroupKFold`, PCA, and
statsmodels GLM. Record important estimator assumptions in docstrings and tests. All Python commands
use `uv run`.

## Implementation orchestration - Sol supervisor and Terra workers

This section defines future implementation ownership. It does not authorize implementation in the
current chat. The named model/effort combinations must be rechecked against the active runtime at
implementation start; never silently substitute a different model, effort, or agent topology.

### Fixed roles

**Lead Sol supervisor - `gpt-5.6-sol`, high reasoning**

- owns user communication, scientific/architectural decisions, authorization checks, package
  order, the live handoff snapshot, and final package acceptance;
- re-reads relevant source, callers, tests, specification, and worktree state before each package;
- freezes the exact task, file allowlist, data contracts, tests, expected RED, and stopping gate;
- independently inspects every worker diff and reproduces RED/GREEN commands;
- owns shared integration surfaces, including package exports, `webapp/app.py`,
  `webapp/session_inputs.py`, this plan, and final cross-package documentation, unless it assigns one
  of those files to a single bounded worker;
- stages and commits reviewed package files, keeping tests-only, implementation, and
  documentation/handoff commits separate;
- resolves worker/reviewer conflicts or asks the user when the issue is scientific or expands
  scope; and
- is the only role permitted to mark a work package complete.

The lead does not treat a worker summary as evidence and does not bypass the Terra tests-first
assignment merely to move faster.

**Terra package worker - `gpt-5.6-terra`, high by default, xhigh where assigned**

- receives exactly one bounded read-only, tests-only, implementation-only, documentation-only, or
  command-only assignment;
- edits only its explicit allowlist and stops if another file or public-interface change appears
  necessary;
- writes tests first, runs the named focused command, reports genuine expected RED, and stops;
- resumes implementation only after Sol has independently verified and committed those RED tests;
- makes the smallest readable in-scope implementation, runs focused tests to GREEN, and stops;
- does not commit, amend, reset, clean, run unauthorized experimental data, launch batch/external
  work, modify tests merely to pass, or spawn another agent; and
- returns inspected/changed files, command output, expected-versus-actual RED/GREEN, risks,
  questions, and confirmation that no out-of-scope action occurred.

Use Terra xhigh for fold/history identity, numerical model validity, PCA leakage, persistence/run
identity, Poisson, Granger, and full synthetic integration. Use Terra high for straightforward
configuration records, plotting, UI adapters, documentation, and authorized command execution.

**Independent Sol gate reviewer - `gpt-5.6-sol`, high or xhigh as assigned**

- reviews a stable tests-only design or stable GREEN diff read-only;
- checks scientific drift, leakage, array axes/units, matched-row identity, estimator settings,
  saved-schema identity, failure handling, and missing tests;
- does not edit, commit, spawn agents, broaden scope, or replace the lead's final gate; and
- is mandatory at the high-risk WP2-WP5, WP7, WP9-WP11 gates described below.

**Optional Terra scout - `gpt-5.6-terra`, medium, read-only**

- may inspect independent call sites, installed-library source, or a bounded failure log;
- returns file/line evidence and uncertainty, not design authority;
- performs no edits or generated-artifact/external mutation; and
- is used only when the evidence task is genuinely independent of the active writer's work.

No `max` or `ultra` assignment is planned. If a scientific or architectural discrepancy cannot be
resolved at the assigned level, stop and ask the user rather than escalating model effort or scope
silently.

### Shared-worktree and concurrency rules

- Use at most four active agents including the lead: one lead Sol, one write-enabled Terra package
  worker, one read-only Sol reviewer, and at most one optional read-only Terra scout.
- Only one agent may edit the shared worktree at a time. Tests-only and implementation stages are
  sequential.
- Read-only scouting may overlap stable independent work, but a reviewer never audits a changing
  diff.
- No two agents edit the same source, test, package initializer, shared fixture, README, webapp
  router, specification, or plan concurrently.
- The lead records HEAD and `git status --short` before and after every assignment. Unexpected or
  unowned changes stop the package immediately.
- The Terra allowlist names exact files or one narrow responsibility group. A newly discovered file
  need is reported to Sol; the worker does not expand its own scope.
- The lead commits only while the writer is idle, stages only reviewed package paths, inspects the
  staged diff, and leaves unrelated user changes unstaged.
- Subagents do not create subagents. The lead alone spawns, follows up, interrupts, or replaces
  workers and reviewers.
- Documentation/handoff updates are separate from tests-only and implementation commits so the TDD
  boundary remains auditable.

### Mandatory package sequence

For every implementation package:

1. Sol performs the resume/preflight audit, confirms authorization, re-reads relevant code and
   usages, and freezes the package contract and file allowlist.
2. If needed, a Terra scout performs a bounded read-only evidence task. Sol decides whether any
   finding changes the assignment.
3. Sol assigns the Terra writer a tests-only task with exact tests, command, expected RED, forbidden
   actions, and return format.
4. Terra adds tests only, runs the focused command, reports RED, and stops.
5. Sol inspects the diff, independently reproduces RED, obtains the required independent Sol
   test-design review, and commits tests only.
6. Sol follows up with the same Terra worker to authorize implementation of the committed contract.
7. Terra implements the smallest in-scope change, runs focused tests to GREEN, reports, and stops
   without committing.
8. Sol inspects the full diff and runs focused plus affected regression suites. The assigned Sol
   reviewer audits the stable GREEN diff.
9. Specific accepted review findings return to the same Terra worker. Sol repeats verification and
   creates the separate implementation commit only after every required gate is green.
10. Sol updates the live snapshot and appends the package record, including commits and exact next
    action, in a separate documentation commit before starting another package.

A test change after the tests-only commit requires an explicit dated explanation that the
requirement changed or the test was genuinely wrong. It is never changed merely to accommodate an
implementation.

### Terra assignment contract

Every worker prompt must be self-contained and include:

- package ID, unique task name, exact model and effort, and task mode;
- one objective and exact file allowlist;
- relevant v3 decisions, array shapes/units, public and saved-data contracts, and compatibility
  requirements;
- exact tests/commands, expected RED or GREEN, and stop gate;
- prohibited files/actions, including implementation during a tests-only task, commits, real-data
  runs, and scope expansion;
- current HEAD/worktree ownership facts required to avoid absorbing unrelated changes; and
- required return fields: files inspected/changed, concise findings/diff summary, exact commands
  and results, expected-versus-actual gate, risks/questions, and scope confirmation.

When explicit model overrides are available, spawn Terra with `model="gpt-5.6-terra"` and the effort
listed in the package map, and spawn the reviewer with `model="gpt-5.6-sol"`. Use a bounded context
fork rather than relying on full inherited history; the prompt carries the authoritative contract.
If an override is rejected or reports a different model/effort, the agent must not edit and Sol
must ask the user how to proceed.

Use follow-up tasks to move the same Terra writer from tests-only to implementation after the
external RED/commit gate. Do not create a replacement implementation agent merely to avoid a clean
handoff.

### Interruption and recovery contract

- An interrupted agent report is not a gate. Sol records the last independently verified HEAD,
  worktree, diff, command, and whether tests or source changes remain uncommitted.
- If a Terra worker becomes unavailable, Sol re-audits its bounded diff and reproduces its last
  claimed result. A replacement receives the same model, effort, allowlist, role, and stopping gate,
  plus the recovery record. It does not rely on the previous worker's summary.
- If an independent reviewer is interrupted, discard the partial review and restart the review with
  a fresh reviewer only after the diff is stable.
- If the lead chat is interrupted, the replacement lead must satisfy the Sol/high role, complete the
  resume checklist, inspect staged and unstaged changes, and establish the last completed gate from
  Git and commands before assigning work.
- If ownership cannot be established, stop without changing, staging, or committing the uncertain
  files and ask the user.
- Conflicting agent reports, irreproducible failures, uncertain library behavior, near-tolerance
  numerical discrepancies, or a required interface outside the approved plan are stop-and-escalate
  conditions.

### Work-package Sol/Terra map

| Package | Primary files/responsibility | Terra assignment | Independent Sol gate | Package completion evidence |
|---|---|---|---|---|
| WP0 | v3, this plan, architecture, handoff state | None; Sol documentation only | User approval | Approved documents and documentation-only baseline commit |
| WP1 | `configuration.py`, `records.py`, configuration tests | High, tests-only then implementation | High final contract review | Validated records/status schema; focused GREEN; separate commits |
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

### Phase-to-package mapping

| Scientific phase | Work packages | Exit condition |
|---|---|---|
| Standard OLS foundation | WP1-WP4 | Unit and PC OLS core contracts GREEN |
| Standard OLS delivery | WP5-WP7 | Reproducible saved results, figures, UI, docs, and synthetic integration GREEN |
| Standard scientific inspection | WP8 | User inspects one designated session and authorizes moving on |
| Poisson regression | WP9 | CV deviance/MSE behavior GREEN and inspected without OLS regression |
| Descriptive Granger | WP10 | Linear/Poisson in-sample outputs structurally separate and GREEN |
| Final integration and inspection | WP11 | Full suite and separately authorized one-session evidence accepted |

### Package gates and handoffs

#### WP0 - Documentation approval

- **Purpose:** freeze v3, the complete architecture, tests, agent ownership, package order, and live
  handoff format.
- **Owned files:** `docs/spec_neural_regression_v3.md` and this plan only.
- **Terra work:** none.
- **Gate:** user approves the documents and later explicitly requests implementation. A
  documentation-only baseline commit exists before WP1 tests are edited.
- **Handoff:** record approval, baseline commit, current HEAD/worktree inventory, and WP1 as the one
  next package. The existence of the commit does not itself authorize WP1.

#### WP1 - Configuration and result contracts

- **Purpose:** establish readable immutable settings, shape/unit records, stable status/reason codes,
  and table schemas without loading or fitting data.
- **Owned source/tests:** `configuration.py`, `records.py`, package initializer exports limited to
  these records, and `test_interregional_configuration.py`.
- **Terra:** high, tests-only then implementation.
- **Sol gate:** public names, validation policy, dataclass immutability, documentation contracts, and
  no premature model/persistence behavior.
- **Completion:** focused GREEN plus existing package-import tests; test-only and implementation
  commits recorded.
- **Handoff to WP2:** freeze exact field names, status/reason vocabulary, array-axis notation, and
  any config JSON representation used later.

#### WP2 - Activity, eligibility, folds, and histories

- **Purpose:** create the shared scientific rows used by every later model.
- **Owned source/tests:** `preparation.py` and `test_interregional_preparation.py`; existing condition,
  PCA, loading, and metadata modules are read-only dependencies.
- **Terra:** xhigh, tests-only then implementation.
- **Independent Sol:** xhigh review before the tests-only commit and after GREEN, focused on
  half-open bins, zero-based row identity, `cur_block`, shared masks, history ordering, and no
  cross-trial/window leakage.
- **Completion:** 19/39 examples, deterministic five-fold identity, condition semantics, qualified
  unit IDs, and regional tensor-axis equality are independently verified.
- **Handoff to WP3:** record exact count dtype, tensor shapes, history column order, fold-assignment
  table, and unavailable reasons. No model code is present.

#### WP3 - Direct-unit OLS

- **Purpose:** implement the first complete scientific model using WP2 rows.
- **Owned source/tests:** `linear.py`, the unit-only portion of `pipeline.py`,
  `test_interregional_linear.py`, and unit-only additions to `test_interregional_pipeline.py`.
- **Terra:** xhigh, tests-only then implementation.
- **Independent Sol:** xhigh numerical test-design/final review covering intercepts, rank, residual
  degrees of freedom, constants, matched restricted/full rows, R-squared, MSE, fold completeness,
  and population aggregation.
- **Completion:** seeded bidirectional direct-unit synthetic OLS is deterministic and every local
  unavailable target remains explicit.
- **Handoff to WP4:** freeze the unit OLS result columns and prove later PCA work cannot change them.

#### WP4 - Fold-local regional PCA and PC OLS

- **Purpose:** add the second standard representation without held-out leakage or changing existing
  population PCA behavior.
- **Owned source/tests:** `pca.py`, PC extensions to `pipeline.py`,
  `test_interregional_pca.py`, and bounded pipeline-test additions.
- **Terra:** xhigh, tests-only then implementation.
- **Independent Sol:** xhigh leakage test-design/final review, including held-out perturbation,
  training-condition union, zero-variance omission, regional separation, no whitening/rescaling,
  and fold-specific rank identity.
- **Completion:** unit OLS regression suite remains unchanged and GREEN; PC OLS is reproducible by
  rank with correct incomplete-fold behavior.
- **Handoff to WP5:** freeze standard in-memory result schemas, PCA metadata, and exact result version
  inputs before any serialization.

#### WP5 - Persistence and offline runners

- **Purpose:** make standard OLS outputs reproducible and runnable without embedding scientific
  logic in file or CLI code.
- **Owned source/tests:** `persistence.py`, `run_session.py`, `run_batch.py`,
  `test_interregional_persistence.py`, and `test_interregional_scripts.py` excluding later docs/UI
  cases.
- **Terra:** xhigh, tests-only then implementation.
- **Independent Sol:** xhigh saved-identity, skip/rerun, corruption, session isolation, logs,
  summaries, and worker-policy review.
- **Completion:** trusted-local result round-trip, configuration fingerprinting, dry-run, new/rerun,
  batch isolation, and timestamped artifacts are GREEN under temporary directories.
- **Handoff to WP6:** freeze file layout, result version, CLI arguments, and loader entry point. WP6
  may consume them but not redesign them silently.

#### WP6 - Standard plots, webapp, and documentation

- **Purpose:** expose stable unit/PC OLS results without placing science in presentation code.
- **Owned source/tests:** `plotting.py`, `webapp/interregional_views.py`, one serialized bounded edit
  to `webapp/session_inputs.py` and `webapp/app.py`, plotting/webapp tests, package README,
  scientist-facing README section, and example configuration.
- **Terra:** high. Plot/UI tests come first; implementation follows their commit. Documentation is a
  separate bounded task/commit after interfaces stabilize.
- **Independent Sol:** high final review for view routing, separate population state, metric labels,
  unavailable counts, output-only plotting, and no duplicated scientific logic.
- **Completion:** all documented help/dry-run commands work against temporary fixtures; standard
  plots and UI pass; existing views remain compatible.
- **Handoff to WP7:** record exact user-visible view name, controls, command examples, and output
  paths.

#### WP7 - Standard synthetic integration and bounded benchmark

- **Purpose:** prove all standard OLS layers compose and collect interpretable performance evidence
  before touching real data.
- **Owned tests/fixes:** end-to-end additions to `test_interregional_pipeline.py`; any source fix is
  limited to a separately identified owning package/file and receives a RED test first.
- **Terra:** xhigh integration-test author, then bounded fix work only after Sol gates each failure.
- **Independent Sol:** xhigh review of scientific signal timing, fold leakage, identity after
  round-trip, unavailable results, figures, stage timings, and memory measurement method.
- **Completion:** deterministic saved synthetic unit/PC OLS output and plots; focused and affected
  existing neural suites GREEN; measured synthetic timing/memory recorded without claiming real-
  session performance.
- **Handoff to WP8:** propose, but do not execute, one exact metadata-session configuration, dry-run
  command, output root, expected work count, and stop conditions.

#### WP8 - Standard one-session inspection

- **Purpose:** let the user inspect the first standard-regression scientific output before Poisson.
- **Authorization:** separate explicit approval of the exact session, config, command, output path,
  and whether computation may write results under that session.
- **Terra:** high command runner only; no source/test/config changes.
- **Sol/reviewer:** Sol validates inputs and interprets raw run evidence; high read-only reviewer
  checks commands, logs, row/fold/unit counts, runtime/memory, and figures.
- **Completion:** user reviews the run and explicitly authorizes proceeding to WP9. A poor result is
  evidence for scientific replanning, not permission to tune parameters silently.
- **Handoff to WP9:** record immutable run paths, commit/config identity, warnings, unavailable rates,
  runtime/memory, and user decision.

#### WP9 - Poisson regression

- **Purpose:** add unit-only unpenalized Poisson CV and matched OLS/Poisson count-MSE comparison after
  standard OLS is accepted.
- **Owned source/tests:** `poisson.py`; bounded pipeline/result/persistence/plot/UI extensions;
  `test_interregional_poisson.py`; and Poisson additions to integration, plot, and webapp tests.
- **Terra:** xhigh, tests-only then implementation.
- **Independent Sol:** xhigh test-design/final review of installed statsmodels API, unpenalized fit,
  convergence, positive means, deviance formula, same null denominator, same OLS rows/folds, and
  schema backward compatibility.
- **Completion:** every pre-existing OLS test remains GREEN; Poisson saved round-trip and synthetic
  integration pass; any designated-session Poisson inspection requires a new exact user approval.
- **Handoff to WP10:** freeze Poisson result fields and record convergence/unavailable evidence plus
  any separately authorized inspection result.

#### WP10 - Descriptive Granger analyses

- **Purpose:** add the final requested model family without mixing in-sample results with CV output
  or adding significance inference.
- **Owned source/tests:** `granger.py`; descriptive-PCA and bounded pipeline/result/persistence/plot/UI
  extensions; `test_interregional_granger.py`; and Granger integration tests.
- **Terra:** xhigh, tests-only then implementation.
- **Independent Sol:** xhigh scientific/formula/final review covering same-row nested fits,
  `log(SSE_R/SSE_F)`, likelihood/deviance equivalence, per-row normalization, lag restriction,
  descriptive PCA isolation, and absence of p-value/joint-causality fields.
- **Completion:** all OLS and Poisson CV tests remain GREEN; Granger tables/figures are separately
  labeled and saved.
- **Handoff to WP11:** freeze the complete schema and exact final integration matrix.

#### WP11 - Final integration, documentation, and scientific inspection

- **Purpose:** verify the complete staged system; this package adds no new scientific method.
- **Terra:** xhigh for final synthetic tests/bounded fixes, then high command runner only for an
  explicitly approved session.
- **Independent Sol:** xhigh final stable-diff review and high run-evidence review.
- **Completion:** complete interregional tests, affected full neural suite, package imports,
  documented commands, saved round-trip, figures, timing/memory, and worktree scope all pass.
  User separately inspects one complete session before any batch of sessions is considered.
- **Final handoff:** append commit IDs, exact commands, run paths, warnings, deferred coverage-mask
  work, user acceptance, and whether batch work remains unauthorized or is proposed separately.

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

1. Accept the documented defaults.
2. Reject unsupported bin sizes and nonpositive lag/order.
3. Reject before/after bounds that do not exactly partition whole.
4. Reject overlapping qualified PFC/HPC unit identities.
5. Reject an empty condition request.
6. Preserve the coverage-assumption version in the validated configuration.

`test_interregional_preparation.py`:

1. Count spikes correctly in hand-checked half-open bins, including spikes exactly on interior and
   terminal edges.
2. Return `(trial, bin, unit)` integer counts with qualified unit order preserved.
3. Produce matching trial/bin axes for PFC and HPC and reject mismatches.
4. Treat an empty bin as observed zero under the complete-coverage assumption.
5. Define `all` as valid alignment plus experimenter-reward validity, choice/context filters, user
   exclusions, and nonmissing `cur_block`.
6. Intersect named conditions with the authoritative base mask without changing existing condition
   definitions.
7. Use zero-based row positions for masks while preserving nontrivial original DataFrame indexes as
   provenance.
8. Assign all trials in one `cur_block` to one test fold.
9. Produce exactly five deterministic folds and the same assignment on repeated calls.
10. Reject missing block values for otherwise CV-eligible trials and fewer than five distinct
    eligible blocks; never fall back to random splitting.
11. Reuse one session fold mapping when condition masks select different trial subsets.
12. Build lag-1/order-1 histories with 19 rows for a two-second 100-ms window and 39 for a
    four-second window.
13. Build lag/order greater than one in documented most-recent-to-oldest feature order.
14. Never cross trial or selected-window boundaries.
15. Keep response, target history, source history, trial row, and target-bin identities aligned.
16. Derive restricted and full designs from one shared full-comparison row mask.

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
10. Require restricted/full scores to have identical held-out row identities.
11. Require all five paired folds for the primary target mean; retain incomplete fold rows without
    presenting a partial mean as complete.
12. Calculate population median and quartiles from target means, not from pooled fold values.

`test_interregional_pipeline.py` initially covers units only:

1. Run HPC-to-PFC and PFC-to-HPC on a seeded synthetic coupled-count session.
2. Detect stronger incremental prediction in the deliberately coupled synthetic direction without
   asserting a publication-style significance threshold.
3. Reuse the identical fold assignment and row identities across directions and restricted/full
   pairs.
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
5. Extend result schemas with PCA scope and effective-dimension metadata.
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

1. Build distinct PFC and HPC population selections from metadata controls.
2. Use unique widget keys and independent unit/channel-quality selections.
3. Reject overlapping selected unit IDs.
4. Expose the documented bin sizes, positive lag/order, conditions, choice/context filters, and
   partitioned windows.
5. Default to units, OLS, choice alignment, 100 ms, lag 1, and order 1.
6. Hide or disable Poisson and Granger actions before their phases are implemented.
7. Explain that legacy manual loading is unsupported and loaded coverage is assumed.
8. Convert controls into a validated configuration without performing numerical work in the view.
9. Display independent valid results when another configuration row is unavailable.
10. Create outputs only from an explicit run action, not an ordinary widget rerender.

`test_interregional_persistence.py`:

1. Produce the same configuration fingerprint for semantically identical configurations.
2. Change the fingerprint when a scientific parameter or analysis version changes.
3. Save intermediate results beneath a temporary session data root, never the repository.
4. Round-trip every named result table, configuration field, coverage version, units/axes metadata,
   and seed/no-randomness declaration.
5. Reuse an identical complete result unless rerun is explicitly requested.
6. Never overwrite a result from a different version or parameter set.
7. Reject incomplete/corrupt records with an actionable error.
8. Refuse to load a record with an unsupported analysis version without explicit migration logic.
9. Create a timestamped run directory containing config, result location, log, summary, copied
   session/batch scripts, and figure directory.
10. Include goal, session, scripts, warnings, unavailable counts, scientific interpretation, and
    coverage/causal caveats in the Markdown summary.

`test_interregional_scripts.py`:

1. Run one synthetic metadata session through the same public single-session function as the UI.
2. Skip an existing matching result by default and honor an explicit rerun request.
3. Print planned sessions, stages, and output paths without computation in batch dry-run mode.
4. Keep independent session results separate and never combine unit columns.
5. Default batch workers to available CPU cores and honor a lower explicit worker count.
6. Propagate a failed session as a logged per-session failure without discarding successful sessions.
7. Accept the portable example configuration through the production configuration loader.
8. Keep documented `--help`, dry-run, new, and rerun command syntax synchronized with the CLI.

`test_interregional_pipeline.py` gains the WP7 standard-regression integration cases:

1. Execute a seeded two-region synthetic session through configuration, counts, folds, unit OLS,
   fold PCA, PC OLS, aggregation, persistence, reload, and plotting.
2. Reproduce identical scientific tables from identical inputs and configuration.
3. Preserve target/fold/trial/bin identities after save/load.
4. Keep unavailable cells explicit while completing independent valid cells.
5. Confirm no Poisson or Granger table/action is present before its package.
6. Record stage timings, synthetic dimensions, peak-memory measurement method, and output size for
   the bounded performance gate without imposing a brittle wall-clock unit-test threshold.

### Implementation sequence

1. WP5 adds versioned result persistence plus single-session and batch runners.
2. WP5 adds run logging, summary generation, figure-output paths, skip/rerun behavior, and batch
   dry-run.
3. WP6 adds plotting functions over immutable result tables.
4. WP6 adds population-selection/control rendering in `interregional_views.py`, cached metadata
   loading adapters, and one small navigation branch in `webapp/app.py`.
5. WP6 adds fold-detail, target-summary, PCA-omission, and unavailability tables plus the package
   README, user quickstart, and portable example configuration.
6. WP7 runs the seeded full standard-regression synthetic integration and bounded timing/memory
   inspection; only bounded fixes are permitted after their own tests-first gate.
7. Sol runs plotting, persistence, script, web-app, session-metadata, channel-quality, package-
   import, and affected existing PCA tests.
8. WP8 is a separate command-only/user gate: after explicit approval of an exact metadata session,
   configuration, command, and output root, run and inspect one standard-regression session.

### Acceptance gate

- The UI loads two populations simultaneously and never infers their anatomical roles.
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
3. Recover finite expected means on a seeded synthetic Poisson dataset.
4. Detect nonconvergence, nonfinite parameters, and nonpositive/nonfinite predicted means.
5. Match hand-calculated Poisson deviance, including zero-count terms.
6. Match hand-calculated null deviance and deviance explained.
7. Preserve negative finite deviance-explained values and increments.
8. Mark normalized scores unavailable when held-out null deviance is zero.
9. Require identical restricted/full test rows and null denominator.
10. Calculate OLS and Poisson MSE on identical unit-count responses without rounding or clipping.
11. Keep restricted/restricted and full/full model-family comparisons paired.
12. Reject Poisson for PC targets.

`test_interregional_pipeline.py` gains Poisson cases:

1. Reuse exactly the OLS fold assignments, target IDs, histories, and test rows.
2. Continue other targets after one target fails convergence.
3. Require five valid paired folds for a primary Poisson target summary.
4. Retain Poisson fit diagnostics and both model families' fold-level MSE.

`test_interregional_plotting.py` gains:

1. Incremental CV deviance-explained plots with no R-squared labeling.
2. Separate held-out MSE comparison plots labeled exploratory.
3. No mixed OLS/Poisson primary metric axis.

### Implementation sequence

1. Verify the installed statsmodels GLM API and diagnostics.
2. Add target-wise Poisson fitting and plain diagnostic records.
3. Add independent deviance/null-deviance scoring.
4. Extend the CV pipeline only for direct units.
5. Add paired model-family MSE tables and plots.
6. Enable the Poisson UI option only for unit representation.
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

`test_interregional_plotting.py` gains:

1. Separate linear and Poisson Granger figures with correct metric labels.
2. Prominent in-sample/descriptive labeling and no significance stars.
3. Individual target values, median/IQR, target counts, and stable IDs.
4. No shared numerical axis with CV metrics.

`test_interregional_webapp.py` gains:

1. Granger actions become available only after the CV configuration is valid.
2. Descriptive PCA and in-sample status are visible.
3. No inference/significance language appears.

### Implementation sequence

1. Add descriptive regional PCA fitting for PC Granger.
2. Add linear and Poisson Granger calculations over existing fit functions.
3. Add nested-fit consistency checks and explicit unavailable reasons.
4. Add separate Granger result table, aggregation, figure, and UI action.
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
- Cache prepared tensors, fold assignments, and PCA transforms at the Streamlit adapter boundary
  with keys containing every scientific input that changes them.
- Do not cache opaque fitted model objects or use UI state as a scientific cache key substitute.
- Do not retain large repeated design matrices in result records.
- In batch runs, parallelize across sessions rather than within a session. Default to available CPU
  cores, but retain a worker override when memory or storage throughput is limiting.

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

### UI runtime

Unpenalized Poisson fits over many targets and configurations may be slow. The first response is
cache reuse and clear progress/status reporting. Parallel fitting is considered only after
profiling, because it adds failure-handling and memory complexity.

## Approval gate

Implementation should begin only after the user approves:

- the layered package/module boundaries and one-way dependency rules;
- the in-memory result-table schema and isolated versioned persistence/run-directory design;
- the WP0-WP11 order and separate tests-only, implementation, review, and handoff commits;
- one Sol/high supervisor, one bounded Terra writer, independent Sol review at the listed gates,
  and the one-writer shared-worktree rule;
- this plan as the authoritative live progress/handoff record, updated at every package gate;
- metadata-only UI support;
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
