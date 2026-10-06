# Inter-Regional Neural Regression Implementation Plan

**Status:** Proposed documentation-only plan. This chat is planning-only and does not authorize
production code, tests, commits, experimental-data runs, benchmarks, or other implementation
actions. Implementation may begin only in a later chat after explicit user approval and a fresh
repository audit.

**Scientific authority:** `docs/spec_neural_regression_v3.md`.

## Live handoff snapshot

**Snapshot date:** 2026-10-06 16:36 EDT.

**Current phase:** WP0 documentation contract freeze complete; awaiting user review. The v3
scientific specification and this plan are tracked documentation files with plan-owned working-tree
changes. No implementation or test work has begun, and the user has explicitly prohibited
implementation in this chat.

**Repository state at this snapshot:**

- branch: `refactor`;
- HEAD before this documentation revision: `6131d8e`;
- plan-owned files: `docs/neural_regression_plan.md` and
  `docs/spec_neural_regression_v3.md`, both currently tracked and modified;
- the worktree also contains an unrelated modified `docs/task_variable_implementation_plan.md` and
  many unrelated pre-existing untracked files/directories; and
- none of those unrelated entries belongs to this plan or may be staged, changed, removed, or
  absorbed into a later package.

**Completed planning evidence:**

- `src/neural_analysis`, `docs/SoftwareDesign.md`, and
  `docs/spec_neural_regression_updated.md` were audited;
- codebase/specification inconsistencies were identified and resolved with the user;
- v3 records the approved scientific and first-pass data-validity decisions;
- the current plan contains module responsibilities, explicit array/result contracts, a phased
  tests-first inventory, performance considerations, and reproducible run outputs; and
- the implementation-readiness correction froze the JSON configuration, condition/fold universe,
  count/PCA determinism, result/status schema, numerical tolerances, atomic run identity,
  read-only webapp boundary, and batch-memory rule; and
- existing repository plans were inspected for their Sol/Terra, interruption, and authoritative
  handoff patterns before this revision.

**Next exact action:** return the corrected plan/spec for user review. Do not create tests or
production files in this chat. A later implementation chat begins only after the user approves the
plan and explicitly requests implementation; the incoming Sol supervisor then performs the resume
checklist below and starts WP1, not source edits from WP2 or later.

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
3. Build the complete-session fold assignment and authoritative trial masks from the trial table.
4. Load aligned spikes once per selected probe/population.
5. Build aligned regional count tensors over the configured whole interval for the authoritative
   base-mask (`all`) trial rows, in ascending zero-based row order.
6. For each fold, fit regional PCA once when PCs are requested, using only fold-training trials.
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
- `RunOptions`: non-scientific execution fields `output_root`, `rerun`, and optional batch-worker
  override. These fields are not part of `InterregionalAnalysisConfig` or its scientific
  fingerprint.

The analysis dependency validator requires `ols_cv` before `linear_granger`, requires units and
`ols_cv` when `poisson_cv` is requested, and requires units and `poisson_cv` before
`poisson_granger`. Request order is normalized to canonical stage order. PCs never enter either
Poisson stage. Prediction windows, representations, and stages must be nonempty and unique and are
normalized to their documented canonical order.

Window/bin compatibility is tested with
`isclose(duration_s / bin_size_s, round(...), rtol=0, atol=1e-9)`. The resolved integer bin count is
the rounded quotient. Validation rejects incompatible window geometry, empty selections, duplicate
or overlapping unit identities after resolution, unsupported values, and invalid stage
dependencies. It does not load data or inspect Streamlit state.
`session_metadata_path` and a nonnull `output_root` must be normalized absolute paths; relative
paths are rejected so fingerprints and handoffs do not depend on a caller's working directory.

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
every potentially unavailable integer diagnostic uses nullable `Int64`; every metric uses nullable
`Float64`. Nonnullable `trial_row`, CV `fold_id`, and always-defined requested/count columns use
`int64`. JSON-list/scalar columns are canonical compact JSON strings, not Python object values.
Every table is sorted by its primary key before save. The exact result tables are:

1. `fold_assignments`, primary key `session_id, trial_row`: `original_index_repr`,
   `block_value_json`, `block_present`, `fold_id`, `status`, `reason`. Missing-block rows have
   nullable `fold_id` and `not_applicable/missing_block`; assigned rows are `ok` with empty reason.
2. `trial_membership`, primary key `session_id, trial_row, alignment, condition`: provenance plus
   `original_index_repr`, `reward_status_valid`, `alignment_valid`, `choice_match`,
   `context_match`, `user_included`, `block_present`, `condition_match`, `included`, and
   `exclusion_reasons_json`.
3. `fold_scores`, primary key
   `session_id, direction, representation, model_family, condition, window, target_id, fold_id`:
   `evaluation_scope`, `target_rank`, `restricted_status`, `restricted_reason`, `full_status`,
   `full_reason`, paired `status`, paired `reason`, `n_train_trials`, `n_test_trials`,
   `n_train_rows`, `n_test_rows`, `restricted_feature_count`, `full_feature_count`,
   `restricted_rank`, `full_rank`, `restricted_df_resid`, `full_df_resid`, plus nullable
   `r2_restricted`, `r2_full`, `delta_r2`, `mse_restricted`, `mse_full`,
   `deviance_restricted`, `deviance_full`, `null_deviance`,
   `deviance_explained_restricted`, `deviance_explained_full`, and
   `delta_deviance_explained`. Metrics not belonging to the model family are null.
4. `target_summaries`, primary key
   `session_id, evaluation_scope, direction, representation, model_family, condition, window,
   target_id, metric_name`: `target_rank`, `status`, `reason`, `requested_folds`, `valid_folds`, and
   nullable `mean_value`. This table is long by metric so valid MSE is retained when R-squared or
   deviance explained is unavailable. A mean is present only when all five values for that metric
   are defined; otherwise status is `incomplete_folds`, reason is `incomplete_requested_folds`, and
   `mean_value` is null.
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
   `restricted_df_resid`, `full_df_resid`, nullable
   `sse_restricted`, `sse_full`, `linear_granger`, `llf_restricted`, `llf_full`,
   `deviance_restricted`, `deviance_full`, `likelihood_ratio`, and
   `mean_deviance_improvement`.

`direction` is exactly `"HPC_to_PFC"` or `"PFC_to_HPC"`; `representation` is `"units"` or
`"pcs"`; `model_family` is `"ols"` or `"poisson"`; windows are `"before"`, `"after"`, or
`"whole"`. Unit `target_id` is qualified; PC `target_id` is `PFC:PC01`, `HPC:PC01`, and so on, with
one-based display rank in nullable `target_rank`. CV fold/target/population rows use
`evaluation_scope="held_out_cv"`; Granger rows use `"in_sample"`. Fold IDs are zero-based 0-4.
CV `metric_name` values are exactly the metric-column names from `fold_scores`; Granger population
summaries use `linear_granger`, `likelihood_ratio`, or `mean_deviance_improvement` as applicable.
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
context_filter_mismatch, user_excluded, condition_mismatch, no_eligible_trials,
no_train_trials, no_test_trials, no_train_rows, no_test_rows, history_exceeds_window,
no_units, pca_no_variable_units,
pca_insufficient_components, constant_training_target, rank_deficient_restricted,
rank_deficient_full, nonpositive_df_restricted, nonpositive_df_full, nonfinite_coefficients,
nonfinite_predictions, constant_test_target, zero_null_deviance,
poisson_nonconverged_restricted, poisson_nonconverged_full, nonpositive_poisson_mean,
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

`trial_membership.exclusion_reasons_json` is a canonical JSON array in this fixed order:
`invalid_reward_status`, `invalid_alignment`, `choice_filter_mismatch`,
`context_filter_mismatch`, `user_excluded`, `missing_block`, `condition_mismatch`. It is empty iff
`included=true`. Run/loader errors such as `insufficient_blocks`, `unsupported_saved_version`, and
`corrupt_saved_result` are exceptions recorded in `run_state.json`/the log, not result-row reasons.

Configuration and coverage-assumption metadata remain attached to the top-level result record.
Large per-bin design matrices and fitted estimator objects are not retained after scoring.

`InterregionalResults` has exactly: `schema_version=RESULT_SCHEMA_VERSION`, `analysis_version`,
`coverage_assumption_version`, canonical `configuration`, `scientific_fingerprint`, `session_id`,
`resolved_populations`, `bin_edges_s`, `units_and_axes`, `generating_functions`,
`randomness_used`, nullable `random_seed`, and the seven named tables. Tables for stages not yet
implemented are present with their frozen empty schemas, which keeps later additions backward
compatible. Large per-bin count/design arrays and fitted estimator objects are not retained.
Deterministic paths record `randomness_used=false` and `random_seed=null` rather than inventing a
seed.

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

`build_regional_count_tensor` is a new regression-local implementation. It uses Pynapple's aligned
tensor construction with the existing population builder's event/window conventions, then
transposes explicitly to `(trial, time_bin, unit)`, verifies finite nonnegative integer-valued
counts within the `int64` range, and casts to `int64`. It obtains trial metadata with `trial_df.iloc[trial_rows]`, never
`.loc`, because `trial_rows` are positions. It must not round rates, extract or change a helper in
the existing PCA module, or change any existing binning API. Spikes at a bin's left edge are
included; the configured final right edge is excluded. Tests compare these boundaries directly.

The prepared regional tensor trial axis contains exactly the ascending trial rows where the
authoritative base mask is true, even when `all` is not itself a requested output condition. Named
condition masks are then indexed onto this fixed base trial axis. Invalid-alignment, missing-block,
filtered, and user-excluded rows remain in provenance tables but are never passed to Pynapple.

Construct whole-window edges as
`whole_start_s + arange(n_whole_bins + 1, dtype=float64) * bin_size_s`, then assign the first and
last values exactly to `whole_start_s` and `whole_stop_s`. Before/after selection uses integer edge
positions derived during validation, not repeated floating comparisons. Thus both regions and all
windows share byte-identical bin-edge arrays.

`build_block_fold_assignment` receives the complete session trial table, constructs one sample per
zero-based row with nonmissing `cur_block`, and calls `GroupKFold(n_splits=5, shuffle=False)` on that
unfiltered row universe. Split enumeration defines fold IDs 0 through 4. Missing-block rows retain
no fold and are excluded from the base CV mask. At least five distinct nonmissing blocks are
required at session preparation time. The mapping is built once before alignment, choice/context,
condition, or user-exclusion filtering and is reused by every analysis stage.

Each nonmissing block value must be a string, non-Boolean integer, or finite float scalar. Convert
NumPy scalars to their Python scalar, encode each with canonical JSON, and pass that encoded string
to `GroupKFold`. This keeps numeric and string labels distinct, makes mixed scalar types sortable,
and gives `fold_assignments.block_value_json` its exact persisted value. Reject arrays, mappings,
Booleans, infinities, and NaNs as invalid block labels.

The authoritative base mask is existing `valid` reward-status mask AND finite selected-alignment
time AND choice/context matches AND not user-excluded AND block present. A named condition is this
base mask AND the corresponding existing canonical condition mask. If a non-`all` choice/context
filter is requested and its required trial column is absent, preparation raises a configuration/
input error instead of silently producing an empty analysis.
The existing LFP mapping is explicit: choice uses `action` and context uses `state_int`; for each,
`left` is numeric 1 and `right` is numeric 0 after numeric coercion. `all` does not require that
filter's column. Nonfinite values fail a selected left/right filter.

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

The fitted transform records the training mean and population standard deviation (`ddof=0`) before
omission, retained and omitted unit IDs, unwhitened component matrix, actual component count, and
observation count. Calculations use `float64`. Omit a unit when its training standard deviation is
zero or nonfinite. Fit scikit-learn PCA with
`n_components=min(requested_components, n_training_observations, n_retained_units)`,
`svd_solver="full"`, and `whiten=False`. This path uses no random number generator and records
`random_seed=null` plus `randomness_used=false`.

Fold PCA uses only training trials pooled over the configured whole interval and union of requested
conditions. The same fitted regional transforms are reused for both directions and every requested
condition/window in that fold. Descriptive all-data PCA has a distinct scope label and cannot be
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
cov_type="nonrobust", full_output=True, disp=False)`. The design already contains the one explicit
intercept; do not call `add_constant`, `fit_regularized`, or pass weights/exposure/offset. Require
`result.converged is True`, finite `result.params`, and finite `result.llf`. Calculate test means as
`exp(test_design @ params)` and require them to be positive and finite. Reconfirm these signatures
and result attributes against the installed source immediately before WP9 in case the environment
changed.

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
input_fingerprint(config, resolved_inputs, code_identity)
create_run_directory(output_root, timestamp, short_fingerprint)
save_interregional_result(result, path)
load_interregional_result(path)
write_run_summary(...)
write_run_log(...)
```

Use one immutable run directory as both the machine-readable and human-readable output. There is no
second version/fingerprint result tree and no mutable canonical result file:

```text
<session_root>/analysis_runs/
    interregional_regression_<YYYYMMDD>T<HHMMSSffffff>Z_<fingerprint12>/
        config.json
        input_manifest.json
        run_state.json
        result.pkl
        run.log
        summary.md
        run_session.py
        run_batch.py
        figures/
```

The timestamp is UTC with microseconds; directory creation is exclusive and a collision is an
error rather than permission to reuse or overwrite a path.

The pickle contains only project-generated configuration records, metadata, and pandas/NumPy data.
Untrusted pickle files must never be loaded. The copied scripts are the exact runner files used.
`summary.md` records the goal, sessions,
scripts, configuration, warnings, unavailable-result counts, output locations, and a scientific
summary. That summary names the compared direction/condition/window, contributing target count,
median/IQR of the primary metric, and whether the metric is held-out or in-sample; it does not turn
descriptive values into significance claims. It explicitly states the complete-coverage and causal
limitations.
`run.log` records the runtime environment, parameters, processed session IDs, warnings or errors,
and execution time.

The scientific fingerprint is SHA-256 over canonical JSON containing the configuration/result
schema, analysis, and
coverage versions; all `InterregionalAnalysisConfig` fields; session ID; resolved ordered unit IDs;
normalized absolute path, byte size, and streamed SHA-256 content hash for every consumed metadata,
trial, aligned-spike, sorter, cluster, and channel-quality input file; and Git HEAD for code identity. It
excludes timestamp, output root, rerun, worker count, and presentation-only choices. `config.json`
and `input_manifest.json` retain the complete expanded values rather than only the hash. Real runs
require a clean tracked worktree so Git HEAD fully identifies all reused code; dry-run reports dirty
tracked paths and refuses `new` until they are committed or otherwise resolved by the user.
Untracked files do not change code identity.

`input_manifest.json` has exact top-level keys `manifest_schema_version="1"`,
`scientific_fingerprint`, `session_id`, `git_head`, `resolved_populations`, and `files`. Each file
entry has `logical_role`, normalized absolute `resolved_path`, `size_bytes`, and `sha256`. Entries
are sorted by `logical_role` then path before hashing/writing. Hash large files in fixed-size chunks
without deserializing them; dry-run may therefore perform substantial sequential I/O while keeping
memory bounded. Missing or changed inputs between dry-run and `new` invalidate the fingerprint and
stop before fitting.

`run_state.json` uses exactly `initialized`, `running`, `failed`, or `complete`, with timestamps,
fingerprint, current/last stage, warnings, and any exception summary. Every JSON and pickle file is
flushed and `fsync`ed in a same-directory temporary file and installed with `os.replace`;
`run_state.json` is
changed to `complete` only after the result, manifest, log, summary, script copies, and figures are
present and validated. Only a `complete` run whose fingerprint and saved versions validate may be
reused or displayed. Interrupted/failed runs are retained as evidence but are never resumed in this
first pass; a new invocation recomputes from the beginning.

Run-stage names are exactly `input_validation`, `input_hashing`, `preparation`, `ols_cv`,
`poisson_cv`, `linear_granger`, `poisson_granger`, `persistence`, `figures`, and `summary`; skip
stages not requested. The CLI logs a start/end record and elapsed seconds for each entered stage.
This is the progress contract; there is no background-task or Streamlit progress protocol.

`run_session.py` loads one metadata session, applies skip/rerun logic, runs the requested completed
analysis stages, saves the result, and writes figures/log/summary. `run_batch.py` reads a session
list, supports a dry-run mode, calls the same single-session function, and parallelizes across
sessions with `concurrent.futures.ProcessPoolExecutor`. Its starting worker count is the available
CPU-core/session minimum, with the documented explicit and memory caps. Each process receives one
configuration path and returns a small status/run-path record; results and arrays are never passed
between workers. Results remain session-separated.

For `new`, the runner scans completed directories below the output root and skips computation when
one has the same fingerprint, printing that run path. `--rerun` always creates a new immutable
timestamped directory and recomputes even when the fingerprint matches; it never overwrites or
mutates the earlier run. The webapp is read-only and never creates a run directory.

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
deserializing full spike arrays or fitting models; it does stream every consumed file to compute its
content hash. `new` reuses a completed matching fingerprint by
default. `--rerun` creates a new complete immutable run with the same scientific fingerprint.

The batch config list contains one UTF-8 configuration path per nonblank, non-comment line. Batch
parallelism is across sessions only. Start from
`min(os.cpu_count() or 1, number_of_sessions, --workers when supplied)` and apply the exact memory
cap in the performance section. Before an actual batch, dry-run reports every estimate and the
planned concurrency. After WP8 measures a representative session peak, the user must explicitly
approve the exact batch worker count; no batch is implied by single-session approval. A per-session
failure is logged without merging or deleting successful session results.

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
  these records, `test_interregional_configuration.py`, and `test_interregional_records.py`.
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
- **Independent Sol:** high final review for view routing, completed-run validation, read-only
  behavior, metric labels, unavailable counts, output-only plotting, and no duplicated scientific
  logic.
- **Completion:** all documented help/dry-run commands work against temporary fixtures; standard
  plots and UI pass; existing views remain compatible.
- **Handoff to WP7:** record exact user-visible view name, saved-result selectors, command examples,
  and output paths.

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
   rows unit-only when units and PCs are both requested.
8. Preserve schema, analysis, and coverage versions while excluding run options from the scientific
   fingerprint input.

`test_interregional_records.py`:

1. Construct all seven empty tables with the exact frozen columns and dtypes.
2. Reject missing/extra columns, wrong dtypes, duplicate primary keys, unsorted keys, and unknown
   status/reason/diagnostic values.
3. Accept the complete applicable unit/PC key grid and require explicit unavailable rows rather
   than missing keys.
4. Preserve valid metric-specific target summaries when another metric is incomplete.
5. Validate canonical JSON scalar/list columns and the fixed membership-exclusion order.
6. Validate exact `InterregionalResults` versions, metadata fields, axes/units, randomness fields,
   and empty future-stage schemas without importing fitting or persistence code.

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
10. Leave missing-block rows unassigned/excluded with reason `missing_block`, reject fewer than five
    distinct nonmissing session blocks, and never fall back to random splitting.
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

1. Discover only completed interregional run directories for the selected metadata session.
2. Exclude initialized, running, failed, corrupt, fingerprint-mismatched, and unsupported-version
   directories.
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

1. Produce the same SHA-256 fingerprint for semantically identical canonical configurations and
   resolved input/code identities.
2. Change the fingerprint when a scientific parameter, version, selected unit order, small-file
   or large-file path/size/content hash, or Git HEAD changes; ignore timestamp, output root,
   rerun, and worker count.
3. Create one immutable timestamp/fingerprint run directory beneath a temporary session data root,
   never the repository; reject repository-contained output roots and do not create a second result
   tree.
4. Round-trip every named table with exact columns, dtypes, primary-key order, configuration,
   coverage, units/axes, input/code identity, and no-randomness declaration.
5. Reuse an identical validated `complete` run by default; make `--rerun` create a different
   directory with the same fingerprint and never overwrite the first.
6. Exercise `initialized -> running -> complete` and `running -> failed` transitions.
7. Install JSON/pickle artifacts atomically and never recognize a run as complete before every
   required artifact validates.
8. Retain but never reuse/resume incomplete, failed, corrupt, fingerprint-mismatched, or
   unsupported-version records.
9. Copy the exact session/batch runner files and create config, manifest, state, result, log,
   summary, and figure directory.
10. Include goal, session, scripts, warnings, unavailable counts, scientific interpretation, and
    coverage/causal caveats in the Markdown summary.
11. Reject duplicate primary keys, wrong columns/dtypes, unknown status/reason values, and unsafe
    load targets outside a discovered project run directory.

`test_interregional_scripts.py`:

1. Run one synthetic metadata session through the public single-session composition root; the UI
   only loads its completed output.
2. Skip an existing matching result by default and honor an explicit rerun request.
3. Print planned sessions, stages, and output paths without computation in batch dry-run mode.
4. Keep independent session results separate and never combine unit columns.
5. Start batch workers at the lesser of CPU cores, session count, and an explicit lower override;
   apply the documented 80%-RAM cap, report every estimate in dry-run, and reject even one session
   that cannot fit.
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
  available CPU cores and session count, then cap that value by the memory rule below. A worker
  override is an upper bound, not permission to exceed the memory cap. No batch run occurs until WP8
  supplies a measured session peak and the user explicitly approves the worker count.

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

This is a conservative planning estimate, not a measured peak. Determine the default memory budget
as 80% of physical RAM using standard-library `os.sysconf`; if unavailable, batch `new` requires an
explicit `--memory-budget-bytes`. The automatic worker cap is
`floor(memory_budget / max_session_estimate)`, with minimum one only when one session fits. Reject a
batch before computation if even one estimated session does not fit. Dry-run prints every term,
the CPU cap, memory cap, chosen workers, and that WP8's measured peak supersedes this estimate for
later approval.

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
