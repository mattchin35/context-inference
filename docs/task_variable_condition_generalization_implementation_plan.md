# Pooled-Window Task-Variable Decoding and Condition-Generalization Plan

**Status:** Documentation revision complete; awaiting user review and
approval. Implementation is not yet authorized. No code, tests, experimental
computation, or cluster work is authorized by this document.

**Scientific contract:**
`docs/task_variable_condition_generalization_spec_v1.md`.

**Reference contracts:** `docs/task_variable_spec_v5.md`,
`docs/task_variable_spec_v6.md`, `docs/task_variable_spec_v7.md`, and
`docs/task_variable_implementation_plan.md` describe the implemented
condition-specific per-time-bin decoder. Their scientific meaning and saved
results do not change.

## 1. Live implementation handoff

Update this section at every work-package boundary and before handing work to
another task. Do not replace prior commit or test evidence with only a prose
summary.

| Field | Current value |
| --- | --- |
| Snapshot date | 2026-10-10 |
| Branch | `refactor` |
| Repository HEAD at documentation audit | `775e182b4df8273ecc2a3df9e245fdf15b9fb931` |
| Last commit touching these documents | `108a585` (`new task variable plans for generalized encoding analysis`) |
| Current work package | PW-WP0: specification and implementation plan |
| Current phase | PW-WP0 revision and validation complete; awaiting user approval; no RED test commit exists |
| Completed work packages | None |
| Active implementation files | None |
| Latest validation | 2026-10-10: referenced paths present; code fences balanced; ASCII-only edits; contract scan clean; `git diff --check` passed; installed scikit-learn weighting APIs characterized |
| Open blockers | User review and approval of the revised scientific and implementation contracts |
| Exact next action | Review this document and the companion specification; after approval, plan and commit PW-WP1 RED tests |

The repository contains unrelated untracked files. They are outside this
analysis and must not be added, changed, or removed. Before each implementation
work package, record any new relevant worktree changes and coordinate rather
than overwriting concurrent work.

### 1.1 Work-package ledger

Add one row when a phase completes. A work package is complete only after its
focused tests, the full existing task-decoding suite, independent review, and
user approval all pass.

| Work package | RED commit | GREEN/refactor commit | Focused tests | Full regression | Review and approval | State |
| --- | --- | --- | --- | --- | --- | --- |
| PW-WP0 documentation | Not applicable | Pending | Documentation checks passed 2026-10-10 | Not applicable | User review pending | Awaiting approval |
| PW-WP1 configuration and folds | - | - | - | - | - | Not started |
| PW-WP2 scientific core | - | - | - | - | - | Not started |
| PW-WP3 result schema and checkpoints | - | - | - | - | - | Not started |
| PW-WP4 reporting | - | - | - | - | - | Not started |
| PW-WP5 single-session lifecycle and usage | - | - | - | - | - | Not started |
| PW-WP6 synthetic end-to-end gate | - | - | - | - | - | Not started |
| PW-WP7 CT026 benchmark | - | - | - | - | - | Not started |
| PW-WP8 optional descriptive comparison | - | - | - | - | - | Deferred |
| PW-WP9 cluster and batch execution | - | - | - | - | - | Deferred |
| PW-WP10 additional windows and experimental batch | - | - | - | - | - | Deferred |

### 1.2 Handoff update requirements

Every handoff must record:

- branch, HEAD, and relevant clean/dirty paths;
- the current work package and RED/GREEN/refactor/review phase;
- exact commits, test commands, pass/fail counts, and unresolved failures;
- files changed and any public interface change;
- decisions made, open decisions, blockers, and user approvals;
- synthetic fixture or experimental run directories, without committing data;
- measured elapsed time, CPU time, peak RSS, and output size when available;
- the exact next safe action and actions still requiring external approval.

### 1.3 PW-WP0 validation evidence

No Python tests were run because PW-WP0 changes documentation only. The
following read-only checks passed on 2026-10-10:

```text
git diff --check
rg scans for unresolved weighting, dense-prediction, comparison, module, and
status language
ASCII scan of both changed Markdown files
existence checks for every referenced existing document and reused module
even code-fence count in both changed Markdown files
installed scikit-learn source/API characterization for LogisticRegression and
ElasticNet sample-weight behavior
```

The installed scikit-learn version is 1.8.0; both estimator `fit` signatures
accept `sample_weight`. On a 120-row, six-feature synthetic fixture with NumPy
seed `20261010`, strict controls (`max_iter=10000`, `tol=1e-10`) gave maximum
repeated-versus-single coefficient/intercept deltas of `2.17e-10` for SAGA
logistic regression and `1.67e-16` for ElasticNet. Under the project's frozen
production controls, the coefficient/intercept deltas were
`2.39e-4`/`1.15e-4` for SAGA and at most `8.33e-17` for ElasticNet. Section 7
therefore separates exact weight-plumbing tests from explicitly toleranced
numerical characterization.

The documentation changes are uncommitted at this snapshot. Record the commit
identity here after the user approves and commits them.

## 2. Fixed revision-1 decisions

- The reference training condition is exactly `correct_rewarded`.
- Canonical windows are `pre_choice` [-0.5 s, 0 s), `post_choice`
  [0 s, +0.5 s), and `full` [-2 s, +2 s).
- `pre_choice` is the default and first experimental benchmark.
- A run uses one explicit nonempty subset of those windows. Runtime never
  changes the selection.
- Every configured condition has a native pooled decoder evaluated across the
  complete time axis.
- The correct-rewarded decoder is evaluated on every configured destination.
- One estimator is fit per window/condition/target/fold/region/representation,
  never per evaluation time or destination.
- Revision 1 uses fixed regularization only.
- Revision 1 introduces no new dependency.
- Every stacked row has sample weight `1 / n_training_bins`, so each trial has
  total weight one across any supported window. Every row from one trial
  remains in the same block-grouped fold.
- Trials, not stacked rows, are the independent sampling units.
- Revision 1 saves fold metrics and model/transform provenance, but not dense
  per-trial predictions.
- Revision 1 does not calculate a paired time-specific-minus-pooled metric.
  A later, read-only descriptive overlay may compare independently completed
  analyses without refitting either one.

## 3. Minimal architecture and reuse boundary

Keep the workflow in `src/neural_analysis/task_decoding` with a separate
analysis identity. Follow the current config/modeling/results/pipeline/plotting
separation, but do not add an independent split module, batch module, web-app
route, or generic framework before a concrete need is demonstrated.

### 3.1 New files required for the validated single-session workflow

| File | Purpose |
| --- | --- |
| `pooled_config.py` | Validate a small pooled-analysis overlay and resolve the existing base task-decoding configuration. |
| `pooled_decoding.py` | Assemble pooled features, fit one fixed model, evaluate it through time, and construct native/transfer/gap records. |
| `pooled_results.py` | Define, save, load, and strictly validate schema-1 checkpoints and final results. |
| `pooled_pipeline.py` | Plan, prepare, execute, resume, and inspect one session while reusing current lifecycle helpers where extraction is safe. |
| `pooled_plotting.py` | Load saved pooled results and create the required native, transfer, and gap figures. |
| `run_pooled_session.py` | Thin `dry-run`, `new`, `resume`, and `status` CLI; no scientific logic. |

Six focused modules are justified by separate data contracts. Do not create
classes that only wrap functions, introduce a shared plugin system, or refactor
the current decoder speculatively.

### 3.2 Existing code reused unchanged where possible

- `config.py`: `TaskDecodingConfig`, target/condition constants, frozen
  estimator controls, and `load_task_decoding_config`.
- `targets.py`: `validate_augmented_trials` and `build_target_table`.
- `conditions.py`: `build_condition_masks`.
- `activity.py`: coverage inspection, memory estimates, region loading,
  `build_session_rate_tensors`, and target-mask projection.
- `modeling.py`: `GroupedSplit`, `SplitPlan`, `make_outer_splits`,
  `RegionTransform`, `fit_region_transform`, `build_region_features`,
  `make_estimator`, `classification_metrics`, and `regression_r2`. PW-WP2
  first extracts a small fitted-estimator/evaluation seam from the existing
  `fit_and_score` implementation because pooled evaluation must reuse one fit
  many times. Existing `fit_and_score` delegates to that seam and retains its
  current public behavior and byte contracts.
- `results.py`: input-manifest and scientific-source fingerprint behavior,
  not its existing result schema.
- `resource_usage.py`: retain the current schema, validator, and tracker
  unchanged, while adding the separate pooled resource contract in Section
  6.6 and reusing its low-level measurement and atomic-write helpers.
- Current execution-guard, detached-launch, atomic-write, and status behavior
  may be extracted into small shared helpers only if focused regression tests
  first prove that the existing public commands and byte contracts are
  unchanged.

The current private `pipeline.py` functions are not a public reuse contract.
Copying a large private function or making the existing pipeline branch on an
analysis mode is disallowed. A small shared extraction is allowed only when it
removes literal duplication and preserves both callers clearly.

### 3.3 Explicitly deferred files and features

- No `pooled_decoding_splits.py`; use `modeling.make_outer_splits`.
- No `run_pooled_batch.py` or pooled Slurm code before PW-WP9.
- No web-app integration in revision 1.
- No automatic loader for current per-time-bin results before PW-WP8.
- No tuned regularization, train-time x test-time matrix, arbitrary windows,
  inferential statistics, or within-session multiprocessing.

PW-WP5 also adds the portable overlay
`docs/examples/neural_analysis/pooled_task_decoding_config.json`, whose
`base_config_path` names the sibling base example with all six conditions and
all 18 current targets explicit, and updates
`src/neural_analysis/task_decoding/README.md` with
the pooled single-session commands and result-location contract. These are
required usage deliverables, not new architecture.

## 4. Configuration contract

Avoid duplicating the established task-decoding parser. A pooled configuration
contains exactly:

```json
{
  "base_config_path": "task_decoding_config.json",
  "training_window_names": ["pre_choice"]
}
```

`base_config_path` is required and resolves relative to the pooled config.
The referenced base configuration supplies session/input paths, selected
conditions and targets, bin width, PCA counts, fixed estimator controls, fold
count, trusted bounds, and output root. It must have `alignment ==
"choice_time"` and `regularization_mode == "fixed"`.

Omitting `training_window_names` selects `pre_choice`. Reject unknown top-level
fields, missing/invalid base configurations, empty/duplicate/unknown windows,
and windows whose edges do not coincide with the base time-bin edges. Select a
bin only when its complete half-open `[edge_i, edge_{i+1})` interval is contained
in the named half-open window. Restore accepted windows to canonical order.

`PooledDecodingConfig` is a frozen dataclass with:

```text
base_config: TaskDecodingConfig
base_config_path: pathlib.Path
training_window_names: tuple[str, ...]
condition_names: tuple[str, ...]
```

`condition_names` is the base condition subset with `correct_rewarded` added
in canonical order when absent. The existing base default remains `all`; it is
not reinterpreted as all six conditions. The complete pooled example must list
all six explicitly in the base configuration.

Its public functions are:

```text
load_pooled_decoding_config(path: Path | str) -> PooledDecodingConfig
scientific_pooled_config_payload(config: PooledDecodingConfig) -> dict[str, object]
resolve_training_windows(config: PooledDecodingConfig,
                         time_bin_edges_s: np.ndarray) -> tuple[TrainingWindow, ...]
```

`TrainingWindow` records `name`, `(start_s, stop_s)`, and a one-dimensional
ascending integer `bin_indices` array. Bounds and time axes are seconds
relative to choice. The scientific payload embeds the canonical base
scientific payload, effective `condition_names`, fixed
`reference_condition`, and resolved window definitions; a change to any of
them changes the run fingerprint. The pooled overlay file and referenced base
config file are both included by byte identity in the input manifest.

## 5. Scientific core API and data flow

Use functions and small immutable records. Each production function must have
a docstring stating input types, shapes, axis order, units, returns, and
scientific unavailable states.

### 5.1 Public scientific functions

On the full chronological axis, construct
`master_eligible = baseline_eligible & target_eligible &
bilateral_full_window_coverage`. Preserve `target_eligible` separately. Assign
the one saved ineligibility reason with exact precedence
`baseline_ineligible`, then `target_ineligible`, then
`neural_window_unavailable`; an eligible row has reason index `-1`.
`trial_row_indices` contains exactly the baseline-valid rows with bilateral
coverage, so only target eligibility can exclude a row after projection to
the common tensor axis.

```text
make_master_outer_split(target_values: np.ndarray,
                        block_ids: np.ndarray,
                        master_eligible: np.ndarray,
                        *, target_family: str,
                        fold_count: int) -> modeling.SplitPlan
```

Inputs are one-dimensional arrays on the common bilateral neural-tensor row
axis `(M,)`, after the pipeline projects full-table values through
`SessionRateTensors.trial_row_indices`. The function calls the existing grouped
splitter on eligible tensor rows and projects its fold IDs back to length `M`
with `-1` for target-ineligible tensor rows. Every split index is a tensor-row
position. It never regenerates folds after condition filtering.

```text
stack_training_window(feature_tensor: np.ndarray,
                      trial_targets: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

`feature_tensor` has shape `(training_trial, training_time, feature)`;
`trial_targets` has shape `(training_trial,)`. Return features with shape
`(training_trial * training_time, feature)`, repeated targets with shape
`(training_trial * training_time,)`, and float64 sample weights of the same
length. Ordering is trial-major then time-major. Every weight is
`1 / training_time`, so weights sum to one within each trial. Empty axes,
nonfinite values, or misaligned shapes raise `ValueError`.

```text
fit_pooled_transforms(
    pfc_training_rates_hz: np.ndarray,
    hpc_training_rates_hz: np.ndarray,
    training_bin_indices: np.ndarray,
    *,
    pfc_pc_count: int,
    hpc_pc_count: int,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
) -> PooledTransforms
```

Both rate tensors are finite Hz arrays shaped `(training_trial,
evaluation_time, regional_unit)` on the same trial/time axes; selected bin
indices are distinct, ascending, and in range. The function fits exactly one
PFC and one HPC training-only transform for the current
window/condition/target/fold cell. `PooledTransforms` is frozen and records
both transform states, stable unit identities, requested/effective counts,
and declared compact availability codes. It is reused by all three region
configurations and both representations.

PW-WP2 first adds these minimal reusable seams to `modeling.py`:

```text
fit_estimator(estimator: object,
              train_features: np.ndarray,
              train_targets: np.ndarray,
              *, target_family: str,
              sample_weight: np.ndarray | None = None) -> FittedEstimator
evaluate_fitted_estimator(fitted: FittedEstimator,
                          held_out_features: np.ndarray,
                          held_out_targets: np.ndarray,
                          *, target_family: str) -> FitEvaluation
```

Fit inputs are finite `(training_row, feature)` float matrices and aligned
`(training_row,)` targets. A supplied weight vector is finite, nonnegative,
one-dimensional, aligned to training rows, and has positive sum; it is passed
unchanged to exactly one estimator `fit` call. When it is `None`, the helper
omits the keyword so legacy estimator call behavior is unchanged. Evaluation
inputs are a finite `(held_out_trial, feature)` matrix with the fitted feature
width and an aligned `(held_out_trial,)` target vector.

`modeling.py` owns closed `FitFailureKind` and `EvaluationFailureKind` string
enums. The exact fit members are `convergence_warning`,
`invalid_coefficient_or_intercept_shape`,
`nonfinite_coefficient_or_intercept`, `invalid_class_structure`, and
`ambiguous_positive_class`. The exact evaluation members are
`invalid_probability_shape`, `nonfinite_probability`,
`positive_score_alignment_invalid`,
`categorical_test_classes_invalid`, `invalid_numerical_prediction`, and
`invalid_regression_metric`. These fine-grained semantic states distinguish
every reviewed branch in current `fit_and_score`; they are not serialized
pooled reason codes.

`FittedEstimator` is a frozen runtime container with the sklearn estimator,
coefficients, intercept, class orientation, convergence status, and optional
`FitFailureKind`; unavailable fits have no estimator. `FitEvaluation` carries
metrics, the categorical positive-score vector/class when applicable, or one
`EvaluationFailureKind`. The evaluation helper predicts and scores without
fitting. Existing `fit_and_score` delegates to these two functions with
`sample_weight=None`; its adapter retains the current up-front validation of
both partitions and their shared feature width before any estimator call. It
maps each kind through this exhaustive legacy table:

| Failure member | Exact legacy `FitResult.reason` |
| --- | --- |
| `convergence_warning` | `convergence_warning` |
| `invalid_coefficient_or_intercept_shape` | `invalid coefficient or intercept shape` |
| `nonfinite_coefficient_or_intercept` | `non-finite coefficient or intercept` |
| `invalid_class_structure` | `categorical estimator must expose binary classes including positive class 1` |
| `invalid_probability_shape` | `invalid probability shape` |
| `nonfinite_probability` | `non-finite probability or score` |
| `ambiguous_positive_class` | `positive class is ambiguous` |
| `positive_score_alignment_invalid` | `positive_scores must be finite and align with target_values.` |
| `categorical_test_classes_invalid` | `Categorical metrics require held-out values from both classes.` |
| `invalid_numerical_prediction` | `non-finite prediction or score` |
| `invalid_regression_metric` | `constant, singleton, or non-finite held-out numerical target` |

The adapter also preserves the current invalid NaN fields and convergence
status. The pooled adapter maps the same kinds through the separate exhaustive
table in Section 6.4. Unknown enum values fail rather than falling through.
This avoids prose parsing and a dependency from generic `modeling.py` back to
pooled modules. Unexpected estimator or programming exceptions still
propagate.

```text
fit_pooled_model(
    transforms: PooledTransforms,
    pfc_training_rates_hz: np.ndarray,
    hpc_training_rates_hz: np.ndarray,
    training_targets: np.ndarray,
    training_bin_indices: np.ndarray,
    *,
    target_identifier: str,
    target_family: str,
    region_configuration: str,
    representation: str,
) -> FittedPooledModel
```

The function applies the already-fitted transforms, builds one requested
region/representation tensor, stacks it, passes the returned sample weights to
`modeling.fit_estimator`, and constructs a durable `PooledModelRecord` with
coefficients, intercept, independent-trial and stacked-row counts, feature
identities, estimator parameters, status, and compact reason. It never fits a
transform and never scores evaluation data.

```text
evaluate_pooled_model(
    model: FittedPooledModel,
    pfc_test_rates_hz: np.ndarray,
    hpc_test_rates_hz: np.ndarray,
    test_targets: np.ndarray,
    test_trial_indices: np.ndarray,
    evaluation_time_centers_s: np.ndarray,
) -> tuple[PooledEvaluationRecord, ...]
```

Held-out tensors have shape `(test_trial, evaluation_time, regional_unit)` in
Hz; targets and common tensor-row positions have shape `(test_trial,)`;
centers have shape `(evaluation_time,)` in choice-relative seconds. The fitted
transform and estimator are applied unchanged at each time. The return has one
immutable record per time and applicable metric, including test indices,
counts, status, reason, and score. It never refits, recenters, rescales, or
recalibrates.

```text
decode_pooled_window_target(
    *,
    training_window: TrainingWindow,
    target_identifier: str,
    target_family: str,
    target_values: np.ndarray,
    master_eligible: np.ndarray,
    block_ids: np.ndarray,
    condition_masks: Mapping[str, np.ndarray],
    outer_split: modeling.SplitPlan,
    pfc_rate_tensor_hz: np.ndarray,
    hpc_rate_tensor_hz: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    evaluation_time_centers_s: np.ndarray,
    pfc_pc_count: int,
    hpc_pc_count: int,
) -> PooledWindowTargetResult
```

The one-dimensional target, eligibility, block, and condition arrays have
shape `(M,)`, and both Hz tensors have shape
`(M, evaluation_time, regional_unit)` on exactly the same common tensor-row
axis. This function handles one window-target checkpoint cell, orchestrates
every native model, reuses each correct-rewarded model for all destination
evaluations, and computes fold-level native-minus-transfer gaps only from
scores on identical destination rows. `PooledWindowTargetResult` contains the
complete durable model, evaluation, gap, transform-status/count, and fold
records for that cell. Shape/identity violations raise; declared scientific
insufficiency remains in the returned records. The pipeline retains the
full-table `(N,)` audit arrays and `trial_row_indices (M,)` separately for
final assembly.

### 5.2 Required computation order

For each target:

1. Construct full-table target/eligibility/condition audit arrays, then project
   target values, valid masks, conditions, and block identities through
   `trial_row_indices` onto the common tensor-row axis.
2. Construct one target-specific master block-fold map on that tensor-row axis.
3. For each training window, condition, and fold, select outer-training tensor
   rows.
4. Fit one PFC and one HPC transform from those trials and selected window bins.
5. Reuse those frozen transforms to build all three region configurations and
   both representations.
6. Stack equal-count bins with `1 / n_training_bins` row weights and fit one
   estimator for each region/representation.
7. Evaluate the frozen native model at every time in its own held-out condition.
8. Reuse the correct-rewarded model on each held-out destination condition.
9. Score native and reference predictions on exactly the same destination
   tensor rows and compute native-minus-transfer gaps. The valid
   correct-rewarded self-gap is exactly zero.
10. Save the complete `PooledWindowTargetResult` checkpoint, then continue to
    the next window-target cell.
11. Preserve scientific unavailable cells; propagate programming, schema, and
    I/O errors.

`correct_rewarded -> correct_rewarded` is fit and stored once. Adding a
destination adds evaluation calls, not reference fits.

Before invalidities, the exact regional-transform fit count is
`W * C * T * F * 2`: two anatomical regions per
window/condition/target/fold, independent of region configuration,
representation, evaluation time, and destination.

## 6. Frozen result and checkpoint contract

Use pooled analysis version `pooled-task-variable-decoding-v1` and result
schema version `1`. Final output is `results.npz` plus the same small JSON
sidecar categories used by the existing workflow. NPZ loading always uses
`allow_pickle=False`.

### 6.1 Run-directory layout

```text
analysis_runs/
  pooled_task_variable_decoding_<UTC timestamp>/
    config.json
    input_manifest.json
    scientific_source.json
    execution.json
    run_state.json
    resume_command.txt
    status_command.txt
    run.log
    console.log                 # detached local runs only
    local_launch.json           # latest detached launch receipt only
    execution_guard.json        # present only while an invocation owns the run
    resource_usage.json
    results.npz                 # present only after atomic final publication
    summary.md                  # present only for a complete run
    run_pooled_session.py       # immutable launch-module snapshot
    pooled_pipeline.py          # immutable main-analysis snapshot
    checkpoints/
      <window>__<target>.npz
    figures/
      native__<window>__<target>__<region>__<representation>__<metric>.png
      transfer__<window>__<target>__<region>__<representation>__<metric>.png
      gap__<window>__<target>__<region>__<representation>__<metric>.png
```

`config.json` is the resolved pooled scientific payload and already embeds the
resolved base payload; no redundant base-config copy is written. A checkpoint
key is exactly `<training_window>::<target>`. Filename labels use the existing
strict filename-safe identifier rule. One checkpoint is complete only when it
holds all configured conditions, folds, regions, representations, evaluation
pairs, and times for that cell.

The source snapshots are copied from the exact fingerprinted repository files
at preparation and validated byte-for-byte on complete reentry. PW-WP9 defines
the batch run-directory additions only when the batch launcher exists; the
single-session schema contains no future placeholder.

### 6.2 Coordinate arrays

The result declares these axes and coordinate arrays:

| Axis | Coordinate | Meaning |
| --- | --- | --- |
| `window` (`W`) | `training_window_names` | Canonical selected windows |
| `condition` (`C`) | `condition_names` | Configured native/destination conditions; reference is always present |
| `target` (`T`) | `target_names` | Configured target order |
| `fold` (`F`) | `fold_ids` | Zero-based requested outer folds |
| `region` (`R=3`) | `region_names` | `PFC`, `HPC`, `PFC+HPC` |
| `source_region` (`A=2`) | `source_region_names` | Anatomical transform order `PFC`, `HPC` |
| `representation` (`P=2`) | `representation_names` | `pca`, `units` |
| `time` (`Q`) | `evaluation_time_centers_s` | Choice-relative seconds |
| `metric` (`K=3`) | `metric_names` | `balanced_accuracy`, `auc`, `r2` |
| `pair` (`E=2C-1`) | `evaluation_train_condition`, `evaluation_test_condition` | Unique reference-transfer and native pairs |
| `trial` (`N`) | existing safe trial identity arrays | Complete chronological trial table |
| `tensor_trial` (`M`) | `trial_row_indices` | Common bilateral neural-tensor rows mapped to `trial` |
| `feature` (`G`) | padded model-feature slot | `G=max(1, selected PFC units + selected HPC units, requested PFC PCs + requested HPC PCs)` |

UTF coordinate widths are derived before allocation and validated without
truncation. Selected source-unit counts are available from dry-run metadata,
so `G` and exact result bytes do not depend on fitting or large neural-array
loading. Fold-specific feature removal or reduced PCA counts are represented
by `model_feature_active`; they never resize `G`. Time and rate metadata state
seconds and Hz explicitly.

The NPZ also contains the one-dimensional coordinate arrays
`status_names`, `reason_code_names`, and `convergence_status_names` in the
orders declared below. Pair coordinates are the two length-`E` arrays
`evaluation_train_condition` and `evaluation_test_condition`. Pair order is
all `correct_rewarded -> destination` pairs in `condition_names` order,
followed by every non-reference `condition -> itself` pair in that same order.
The duplicate reference self-pair is omitted.

### 6.3 Required scientific arrays

Status arrays are `uint8`: `0=not_applicable`, `1=valid`, and `2=unavailable`.
Reason arrays are `int16` indices into `reason_code_names`; `-1` means no
reason. Scores, target values, coefficients, intercepts, and time coordinates
are `float64`; unavailable or padded float entries are NaN. Counts are
`int32`, masks are Boolean, fold maps are `int16`, and trial positions/row
indices are `int64`; integer padding is `-1` unless an array says otherwise.
Convergence arrays are `uint8` indices into `not_run`, `converged`, `warning`,
and `invalid` in that order. Categorical positive-class coordinates are
`int8`. String-coordinate Unicode widths are derived and validated before
allocation.

| Array | Shape | Contract |
| --- | --- | --- |
| `target_values` | `(T,N)` | Native target units; NaN only where target-ineligible |
| `target_eligible` | `(T,N)` | Target-specific eligibility |
| `master_eligible` | `(T,N)` | Baseline eligibility AND target eligibility AND bilateral full-window coverage |
| `eligibility_reason_index` | `(T,N)` | First failing reason in baseline, target, neural precedence; `-1` when eligible |
| `condition_masks` | `(C,N)` | Full-table canonical masks |
| `trial_row_indices` | `(M,)` | Full-table positions with baseline eligibility and bilateral full-window coverage |
| `outer_fold_ids` | `(T,M)` | Fold IDs or `-1` for target-ineligible tensor rows |
| `training_window_bin_mask` | `(W,Q)` | Exact selected training bins |
| `transform_status` | `(W,C,T,F,A)` | One shared regional-transform availability state |
| `transform_reason_index` | `(W,C,T,F,A)` | Strict compact transform reason code |
| `transform_input_unit_count` | `(W,C,T,F,A)` | Selected source units before training-only removal |
| `transform_usable_unit_count` | `(W,C,T,F,A)` | Nonconstant finite training units |
| `transform_component_limit` | `(W,C,T,F,A)` | Deterministic PCA component limit |
| `transform_requested_pc_count` | `(W,C,T,F,A)` | Configured regional PC count |
| `transform_effective_pc_count` | `(W,C,T,F,A)` | Fitted retained PC count |
| `model_status` | `(W,C,T,F,R,P)` | Model availability |
| `model_reason_index` | `(W,C,T,F,R,P)` | Strict compact invalidity code |
| `model_train_trial_count` | `(W,C,T,F,R,P)` | Independent training trials |
| `model_stacked_row_count` | `(W,C,T,F,R,P)` | Trials times selected bins |
| `model_requested_feature_count` | `(W,C,T,F,R,P)` | Width before training-only feature removal |
| `model_feature_count` | `(W,C,T,F,R,P)` | Effective fitted width |
| `model_convergence_status` | `(W,C,T,F,R,P)` | Index into declared convergence statuses |
| `model_intercept` | `(W,C,T,F,R,P)` | Fitted estimator intercept |
| `model_coefficients` | `(W,C,T,F,R,P,G)` | Coefficients on saved feature scale |
| `model_feature_active` | `(W,C,T,F,R,P,G)` | Features present in the fitted model |
| `evaluation_status` | `(W,E,T,F,R,P,Q,K)` | Fold-level metric status |
| `evaluation_reason_index` | `(W,E,T,F,R,P,Q,K)` | Strict compact reason code |
| `evaluation_scores` | `(W,E,T,F,R,P,Q,K)` | Fold-level held-out metrics |
| `evaluation_test_trial_count` | `(W,E,T,F)` | Destination test-trial count |
| `gap_status` | `(W,C,T,F,R,P,Q,K)` | Native-minus-transfer availability |
| `gap_reason_index` | `(W,C,T,F,R,P,Q,K)` | `source_score_unavailable` or no reason |
| `gap_scores` | `(W,C,T,F,R,P,Q,K)` | Native minus reference-transfer score |

The following non-score arrays are also required:

| Array | Shape | Contract |
| --- | --- | --- |
| `time_bin_edges_s` | `(Q+1,)` | Choice-relative seconds |
| `training_window_start_s`, `training_window_stop_s` | `(W,)` | Exact half-open bounds in seconds |
| `trial_positions` | `(N,)`, `int64` | Complete-table row positions |
| `trial_ids` | `(N,)`, `int64` | Stable complete-table trial IDs |
| `behavioral_block_payload` | `(B,)`, `uint8` | Packed exact behavioral-block identity bytes |
| `behavioral_block_offsets` | `(N+1,)`, `int64` | Byte offsets delimiting each trial's block identity |
| `target_family_names`, `target_unit_names` | `(T,)`, `(T,)` | Family and physical/dimensionless target units |
| `target_positive_class` | `(T,)` | `1` for categorical targets, `-1` for numerical targets |
| `coefficient_feature_ids` | `(R,P,G)` | Stable direct-unit or regional-PC identity per padded feature slot |
| `coefficient_feature_regions` | `(R,P,G)` | Anatomical source of each feature slot |
| `coefficient_feature_present` | `(R,P,G)` | Non-padding feature slots in the run-wide union |
| `unit_selection_rules_json` | `(2,)` | Canonical PFC/HPC source-selection provenance |
| `model_train_class_counts` | `(W,C,T,F,2)` | Class 0/1 counts, `-1` for numerical targets |
| `evaluation_test_class_counts` | `(W,E,T,F,2)` | Class 0/1 counts, `-1` for numerical targets |
| `target_estimator_class_names` | `(T,)` | Concrete fixed estimator class |
| `target_estimator_parameters_json` | `(T,)` | Canonical fixed-control JSON |

`B` is the packed behavioral-block payload byte count, not a scientific
coordinate or result axis.

Metadata records software versions, random seed, source fingerprint, run
fingerprint, input manifest identity, stable regional source-unit identities,
coefficient scales/units, the fixed `1 / n_training_bins` weighting policy,
and the exact axis list for every array. Fitted
scaler/PCA matrices are not saved: immutable inputs, configuration, folds,
source fingerprint, selection rules, and model records are the reproduction
contract, matching the current task decoder's level of persisted transform
provenance. The scientific source fingerprint includes every new pooled module
plus the reused `activity.py`, `conditions.py`, `config.py`, `modeling.py`,
`resource_usage.py`, `results.py`, and `targets.py` files and the current direct
runtime dependencies already covered by task decoding. The array names,
dtypes, and axes in this section are exact; implementations must not factor,
rename, or omit them because values happen to repeat across axes.

Do not save per-trial predictions in revision 1. AUC is computed in memory
from held-out positive-class probabilities. Reproduction is provided by fold
identities, coefficients, intercepts, estimator controls, source/config
fingerprints, and immutable inputs.

### 6.4 Compact unavailable vocabulary

Only the following codes may be serialized in revision 1:

```text
baseline_ineligible
target_ineligible
neural_window_unavailable
target_outer_unavailable
empty_training_subset
empty_test_subset
training_class_unavailable
test_class_unavailable
training_target_unavailable
test_target_unavailable
pfc_transform_unavailable
hpc_transform_unavailable
regional_transform_unavailable
no_features
fit_convergence_failure
outer_estimator_failure
invalid_probability
invalid_prediction
invalid_metric
source_score_unavailable
```

`pooled_decoding.POOLED_REASON_CODES` is the sole ordered vocabulary used by
producers and validators; `reason_code_names` serializes that exact order.

Pooled scientific functions choose these codes at the decision point or map a
closed modeling failure enum through the exact adapter below. They never infer
codes from substrings or arbitrary exception prose. The required producer
contract is:

| Decision point | Allowed compact code |
| --- | --- |
| Full-table baseline/target construction | `baseline_ineligible`, `target_ineligible` |
| Bilateral coverage projection | `neural_window_unavailable` |
| Reviewed unavailable `SplitPlan` | `target_outer_unavailable` |
| Condition/fold subset validation | `empty_training_subset`, `empty_test_subset`, `training_class_unavailable`, `test_class_unavailable`, `training_target_unavailable`, `test_target_unavailable` |
| Standalone regional transform | `pfc_transform_unavailable`, `hpc_transform_unavailable` |
| Combined regional feature construction | `regional_transform_unavailable`, `no_features` |
| Estimator fit validation | `fit_convergence_failure`, `outer_estimator_failure` |
| Prediction/metric validation | `invalid_probability`, `invalid_prediction`, `invalid_metric` |
| Paired gap construction | `source_score_unavailable` |

The exhaustive modeling-to-pooled adapter is:

| Modeling failure member | Pooled code |
| --- | --- |
| `convergence_warning` | `fit_convergence_failure` |
| `invalid_coefficient_or_intercept_shape`, `nonfinite_coefficient_or_intercept`, `invalid_class_structure`, `ambiguous_positive_class` | `outer_estimator_failure` |
| `invalid_probability_shape`, `nonfinite_probability` | `invalid_probability` |
| `positive_score_alignment_invalid`, `categorical_test_classes_invalid`, `invalid_regression_metric` | `invalid_metric` |
| `invalid_numerical_prediction` | `invalid_prediction` |

`make_master_outer_split` uses the existing `SplitPlan.is_available` state and
emits `target_outer_unavailable` without parsing its human-readable detail.
`fit_pooled_transforms` similarly uses reviewed transform availability and the
known anatomical context, not reason prose. The pooled estimator adapter owns
the table above; generic `modeling.py` does not import pooled code. The legacy
adapter separately maps every enum member to the exact current
human-readable `fit_and_score` reason and preserves its invalid coefficient,
intercept, metric, positive-score, and convergence outputs. Unknown enum
members, strings, states, exceptions, schema changes, and I/O errors fail the
run instead of being relabeled unavailable. The validator rejects undeclared
or over-width codes; every declared code must fit NumPy `U32` without
truncation.

### 6.5 Persistence and lifecycle APIs

```text
save_pooled_checkpoint(path: Path, *, checkpoint_key: str,
                       run_fingerprint: str,
                       result: PooledWindowTargetResult) -> None
load_pooled_checkpoint(path: Path, *, expected_checkpoint_key: str,
                       expected_run_fingerprint: str) -> PooledWindowTargetResult
save_pooled_run(run_directory: Path, *, arrays: Mapping[str, np.ndarray],
                metadata: Mapping[str, object]) -> Path
load_pooled_run(run_directory: Path, *,
                expected_run_fingerprint: str | None = None
                ) -> dict[str, object]
```

Savers validate before one atomic publication. Loaders use
`allow_pickle=False`, validate the complete schema, and return no fitted
scikit-learn objects.

```text
plan_pooled_decoding_session(config_path: Path | str) -> dict[str, object]
prepare_pooled_decoding_run(config_path: Path | str) -> pathlib.Path
run_prepared_pooled_decoding(run_directory: Path | str) -> None
inspect_pooled_decoding_status(run_directory: Path | str) -> dict[str, object]
```

Planning reads small metadata and reports exact dimensions/resources without
loading spikes or fitting. Preparation writes the immutable identity and
lifecycle sidecars. Execution is the only function that loads full neural
arrays or fits. Status reads only small sidecars and validates a completed
result without mutating it. `run_pooled_session.main(argv)` only parses CLI
arguments and delegates to these functions.

### 6.6 Pooled resource-usage contract

The current `RESOURCE_USAGE_SCHEMA_VERSION`, `ResourceUsageTracker`, and
`validate_resource_usage_payload` remain unchanged for existing task-decoding
runs. In the same `resource_usage.py` module, PW-WP5 adds a distinct
`POOLED_RESOURCE_USAGE_SCHEMA_VERSION = 1`, `PooledResourceUsageTracker`,
`validate_pooled_resource_envelope`, and
`validate_pooled_resource_usage_payload`. Both trackers may reuse the existing
`resource.getrusage`, RSS-normalization, output-size, and atomic-JSON helpers,
but neither validator accepts the other analysis's payload.

The pooled resource envelope has exactly these integer fields:

```text
full_trial_count
tensor_trial_count
pfc_unit_count
hpc_unit_count
time_bin_count
training_window_count
condition_count
evaluation_pair_count
target_count
outer_fold_count
coefficient_feature_capacity
categorical_fit_count
numerical_fit_count
transform_fit_count
evaluation_metric_cell_count
tensor_allocation_bytes
planned_major_array_bytes
```

Family fit counts may individually be zero; all dimensions and byte counts are
positive, and categorical plus numerical fit count is positive. The envelope
comes from dry run and is identical across every resume.

The pooled payload has exactly `schema_version`, `analysis_type` (fixed to
`"pooled_task_variable_decoding"`), `measurement_method`, `status`,
`peak_rss_bytes`, `wall_time_seconds`, `user_cpu_seconds`,
`system_cpu_seconds`, `cpu_efficiency`, `resource_envelope`,
`completed_checkpoint_keys`, `target_split_measurements`,
`checkpoint_measurements`, `stage_timing_seconds`, `operation_counts`, and
`output_size_bytes`. Measurement method remains `"resource.getrusage"`, and
statuses retain the current running/failed/interrupted/complete vocabulary. A
checkpoint key is the canonical
`<training_window>::<target>` identity. A complete payload contains every and
only the canonical `W * T` keys, in pipeline order; it never uses the current
schema's `C * T` completion rule.

`target_split_measurements` is keyed once per target because the master split
is shared across windows. Each value contains exactly `target_family`,
`timing_available`, nullable `total_seconds`, and `call_count`; the call count
is zero for a restored timing-unavailable split and a positive cumulative
count when deterministic split construction was repeated across resumes.
Each
`checkpoint_measurements` value contains exactly `training_window`,
`target_identifier`, `target_family`, `timing_available`, nullable
`total_seconds`, `requested_counts`, `valid_model_cell_count`,
`unavailable_model_cell_count`, `valid_evaluation_metric_cell_count`,
`unavailable_evaluation_metric_cell_count`, `timing_seconds`, and
`operation_counts`. `requested_counts` has exactly `estimator_fits`,
`transform_fits`, and `evaluation_metric_cells`. When timing is available, the
last two mappings have the same exact operation structure (zero-valued entries
remain present):

- `regional_transform`, keyed by `PFC` and `HPC`;
- `feature_construction`, `stacking`, `estimator`, `native_evaluation`, and
  `transfer_evaluation`, each keyed by the six canonical
  `<region>|<representation>` identities; and
- scalar `checkpoint_serialization`.

For one window-target checkpoint, the requested counts are `C*F*R*P`
estimator fits, `C*F*A` transform fits, and `E*F*R*P*Q*K_T` evaluation metric
cells, where `A=2`, `R=3`, `P=2`, and `K_T` is two for a categorical target or
one for a numerical target. The reference self-pair is counted only under
`native_evaluation`; `transfer_evaluation` contains the other `C-1` reference
transfer pairs, so no metric cell is counted twice.

Evaluation operation counts count produced metric cells, not Python function
calls; categorical evaluation can therefore add two cells while numerical
evaluation adds one. Timing still records elapsed work once, under the
applicable native or transfer operation.

The top-level `operation_counts` has exactly
`requested_estimator_fits`, `measured_estimator_fits`,
`requested_transform_fits`, `measured_transform_fits`,
`requested_evaluation_metric_cells`, `measured_evaluation_metric_cells`,
`valid_model_cells`, `unavailable_model_cells`,
`valid_evaluation_metric_cells`, and `unavailable_evaluation_metric_cells`.
On complete evidence, requested aggregates must equal the corresponding
envelope counts and all aggregates must equal the sums of checkpoint records.
Valid plus unavailable model cells must equal requested estimator fits; valid
plus unavailable evaluation cells must equal requested applicable metric
cells. Measured operation counts may be lower than requested counts when a
declared earlier scientific invalidity prevents downstream work.
Complete `stage_timing_seconds` has exactly `input_loading`, `final_assembly`,
and `reporting`; target splits and all checkpoint-local stages stay in their
dedicated records rather than being double-counted there.

Resume loads matching prior evidence. A restored checkpoint retains its prior
measurement values unchanged when the key, window, target, family, and run
envelope match. If no matching historical record exists, it is recorded with
`timing_available=false`, null total time, and empty measured timing/count
maps, while its requested and valid/unavailable counts are reconstructed from
the validated checkpoint. Timing is never fabricated. Partial, failed, and
interrupted snapshots may contain a strict prefix/subset of checkpoint keys;
`checkpoint_measurements` always has exactly the same keys as that completed
list. Complete evidence also has every target split record in target order.
Complete validation enforces the full `W * T` identity and aggregate rules.

## 7. Tests committed before production

Tests use pytest and are committed in a RED commit before the corresponding
production code. Do not weaken a test after RED unless the requirement changed
or the test is demonstrably wrong.

### 7.1 Proposed test files

| File | Scope |
| --- | --- |
| `src/tests/neural_analysis/task_decoding/test_pooled_config.py` | Overlay configuration, canonical windows, serialization, identity |
| `src/tests/neural_analysis/task_decoding/test_pooled_decoding.py` | Master folds, stacking, model fitting, reuse, evaluation, gaps, invalidity |
| `src/tests/neural_analysis/task_decoding/test_modeling.py` | Regression coverage for extracted fit/evaluate seams and unchanged `fit_and_score` behavior |
| `src/tests/neural_analysis/task_decoding/test_pooled_results.py` | Exact arrays, dtypes, axes, reason vocabulary, checkpoints, safe load/save |
| `src/tests/neural_analysis/task_decoding/test_pooled_pipeline.py` | Planning, prepare/run/resume/status, ownership, atomic completion |
| `src/tests/neural_analysis/task_decoding/test_pooled_plotting.py` | Saved-result-only figures and captions |
| `src/tests/neural_analysis/task_decoding/test_resource_usage.py` | Existing resource-schema regression and distinct pooled envelope/checkpoint accounting |
| `src/tests/neural_analysis/task_decoding/test_pooled_synthetic.py` | Seeded public-entry-point scientific cases |

### 7.2 Required tests

Configuration and folds:

1. Omitted windows select only `pre_choice`; subsets restore canonical order.
2. Unknown fields, invalid base mode/alignment, empty/duplicate/unknown or
   misaligned windows fail.
3. Serialization embeds the base payload, reference condition, exact bounds,
   bin indices, and analysis version.
4. Full-table values, masks, and blocks project through `trial_row_indices`
   without reordering; master eligibility is the exact baseline/target/
   bilateral intersection; rows failing multiple predicates retain the
   independent target mask and use the fixed reason precedence; one
   deterministic `(M,)` target fold map is shared by every condition, and
   behavioral blocks never cross train/test.
5. Condition-specific insufficiency is unavailable without fold regeneration,
   seed search, or trial-wise fallback.

Scientific core:

6. Stacking is exactly trial-major/time-major; repeated targets align; every
   row weight is `1 / n_training_bins`; each trial's weights sum to one; and
   the weight vector is passed to each estimator fit.
7. A spy estimator receives the exact row-weight vector. A seeded
   strict-control characterization uses `atol=1e-8` for repeated-versus-single
   coefficients/intercepts; a separate frozen-production-control smoke uses
   `atol=1e-3` for SAGA and `atol=1e-12` for ElasticNet. These are explicit
   characterization tolerances, not `COEFFICIENT_TOLERANCE` or a scientific
   equality claim.
8. Scaling/PCA use outer-training condition/window rows only and remain frozen
   for every time and destination. Exactly two regional transforms are fit per
   valid window/condition/target/fold cell, independent of region,
   representation, time, and destination counts.
9. PFC, HPC, and combined direct/PCA feature IDs, shapes, units, and coefficient
   positions are stable.
10. Estimator call count is one per model identity and is independent of time
   count and destination count.
11. Every native model is evaluated at every time, and each reference model is
    reused across destinations.
12. Transfer and native metrics use identical destination fold rows; gaps are
    exactly native minus transfer, including exact zero for valid
    correct-rewarded self-comparisons.
13. Categorical positive-class orientation and numerical native units match
    the current decoder.
14. Each reviewed scientific decision emits only its declared compact code;
    every `FitFailureKind` and `EvaluationFailureKind` member has the exact
    exhaustive mapping in Section 6.4; every code fits `U32`; no
    substring/prose inference exists; unknown states and unexpected exceptions
    propagate.
15. Extracted fit/evaluate seams preserve every existing `fit_and_score`
    output, including exact human-readable invalidity reasons, NaN fields,
    positive scores, metrics, class orientation, and convergence status.

Results:

16. Every required array has the exact declared axes, shape, dtype, units,
    padding, sentinel, and coordinate range, including `(B,) uint8` block
    payload and `(N+1,) int64` block offsets.
17. `target_values`/condition/eligibility arrays use full-table `N`,
    eligibility reasons follow the frozen precedence, `trial_row_indices` maps
    baseline-valid bilateral rows exactly, and `outer_fold_ids` uses `(T,M)`
    with `-1` only for target-ineligible common rows.
18. Model and pair identities are complete and unique; reference models are
    stored once.
19. Dense per-trial prediction arrays are absent.
20. A checkpoint contains exactly one window-target cell and cannot be reused
    across a different window, target, condition set, source, config, schema,
    or code fingerprint.

Reporting:

21. Figures identify window, train/test conditions, target, region,
    representation, metric, and time units; unavailable differs from
    below-chance performance. Every time-course marks the training interval,
    and captions distinguish held-out in-window evaluation from out-of-window
    temporal generalization (`full` has no out-of-window time).
22. Plotting reads saved results only and never loads spikes or fits models.

Lifecycle and usage:

23. Interrupted execution preserves complete checkpoints; exact resume skips
    them; final publication occurs only after validated results, summary, and
    required figures exist; complete reentry is read-only; launch snapshots
    match the fingerprinted source bytes.
24. Dry run loads no large spike arrays or fits; it computes exact `G` and
    major-array bytes from selected source-unit/requested-PC metadata.
25. The pooled envelope has exactly the fields in Section 6.6; complete
    evidence requires exactly `W * T` canonical checkpoint keys and reconciles
    requested/measured transform, estimator, evaluation, validity, and timing
    aggregates across multi-window execution and resume.
26. Restored pooled checkpoints preserve matching prior measurements; missing
    prior timing is explicit rather than fabricated; condition-by-target
    completion, mismatched identities, unknown fields, and inconsistent
    aggregates are rejected.
27. The existing resource schema, validator, tracker, and serialized payloads
    retain their current behavior and `C * T` completion rule.
28. Existing per-time-bin runs remain readable and are never discovered as
    pooled runs; log and summary record the goal, session, parameters, scripts,
    warnings/errors, and elapsed time.
29. The portable example resolves to all six conditions and all 18 targets;
    README commands name the pooled entry point and immutable run directory.

Scientific fixtures:

30. Seeded stable, rotating/sign-changing, stable-transfer, remapped,
    absent-code, delayed-code, constant-reference, overlapping-condition, and
    multi-window fixtures produce their specified qualitative outcomes.
31. The full-window fixture fits once per model identity, not once per time.
32. One-shot and interrupted/resumed synthetic runs produce byte-identical
    scientific arrays and canonical metadata plus complete validated results,
    summary, and required PNGs.

Every work package must also run all existing tests under
`src/tests/neural_analysis/task_decoding` to protect the current workflow.

## 8. Work packages and gates

### PW-WP0: documentation

- Deliverables: this plan, the companion specification, and necessary
  clarifications in existing reference documents.
- Validation: link/path checks, forbidden-open-decision search, terminology
  consistency, `git diff --check`, and human review.
- Gate: user approves the revised contract before any test or code change.

### PW-WP1: configuration and master folds

- RED: tests 1-5 in `test_pooled_config.py` and `test_pooled_decoding.py`.
- GREEN: `pooled_config.py` plus the minimal master-fold helper.
- Gate: focused tests, existing task-decoding tests, independent leakage/API
  review, user approval, and ledger update.

### PW-WP2: pooled scientific core

- RED: tests 6-15 and focused seeded unit fixtures, including regression tests
  for the existing `fit_and_score` API.
- GREEN: the minimal fit/evaluate extraction in `modeling.py` and
  `pooled_decoding.py`; reuse current transforms, estimators, and metrics
  without changing existing public behavior.
- Gate: exact estimator and transform call counts, verified weighting
  semantics, no evaluation refit/recalibration, scientific review, full
  regression, and ledger update.

### PW-WP3: result schema and checkpoints

- RED: tests 16-20 in `test_pooled_results.py`.
- GREEN: `pooled_results.py`, schema-1 validation, immutable atomic checkpoint
  save/load, and no prediction arrays.
- Gate: independent byte/axis/sentinel/reason-code review and ledger update.

### PW-WP4: reporting

- RED: tests 21-22 in `test_pooled_plotting.py` against validated saved-result
  fixtures.
- GREEN: `pooled_plotting.py` with saved-result-only required-PNG rendering.
  Lifecycle and summary integration remain in PW-WP5.
- Gate: visibly marked training intervals, complete captions, readable
  light-mode outputs, user inspection, full regression, and ledger update.

### PW-WP5: single-session lifecycle and usage documentation

- RED: tests 23-29 in `test_pooled_pipeline.py`, `test_resource_usage.py`, and
  focused configuration/documentation checks. Existing resource-schema
  regression tests are committed before adding the pooled tracker.
- GREEN: `pooled_pipeline.py` and thin `run_pooled_session.py` with dry-run,
  `new`, `resume`, local detached execution, and `status`; integrate validated
  pooled resource evidence and saved-result summary/PNG rendering before
  terminal completion; add
  `docs/examples/neural_analysis/pooled_task_decoding_config.json`, and the
  pooled usage section in `src/neural_analysis/task_decoding/README.md`.
- Gate: ownership/failure-preservation review, atomic complete publication,
  portable examples, full regression, user inspection, and ledger update.

### PW-WP6: synthetic end-to-end scientific gate

- RED: tests 30-32 through the real public entry point.
- GREEN: only fixes required by those contracts; no new feature scope.
- Gate: deterministic complete validated results, summary, and figures; no
  unexplained unavailable cells; one-shot/resume equality; independent
  scientific review; user inspection; and ledger update.

### PW-WP7: CT026 pre-choice benchmark

- Prepare an immutable all-target/all-condition pre-choice configuration.
- Run dry-run first, then one user-approved experimental session.
- Record predicted versus actual fits/evaluations/bytes, elapsed and CPU time,
  peak RSS, stage timings, invalidity, checkpoint bytes, and final output size.
- Gate: user inspects scientific results before any new feature, window, or
  multi-session execution is authorized.

### PW-WP8: optional descriptive comparison

- Enter only if the user wants a consolidated existing-versus-pooled figure
  after PW-WP7.
- Compare saved complete-fold means only after validating common session,
  target, condition, time, region, representation, and metric coordinates plus
  matching neural-source/unit-selection identity. Require and label both
  transform-training provenances. Label curves as independently
  cross-validated and descriptive.
- State that the existing per-time-bin analysis fits its scaler/PCA basis over
  the complete analysis window, while pre/post pooled models fit transforms in
  their named windows. Thus those overlays compare preprocessing as well as
  readout constraints; `full` is the closest transform-matched comparison.
- Do not calculate paired differences, modify either result, load spikes, or
  rerun either analysis.
- This work package is not required for a valid pooled analysis.

### PW-WP9: cluster and batch execution

- Enter only after PW-WP7 supplies a measured resource envelope and the user
  approves cluster implementation.
- Reuse or minimally generalize reviewed current Slurm/lifecycle behavior.
- Add dry-run and session-manifest tests before code. Default parallelism is
  across sessions, one session per worker/job-array element.
- A synthetic scheduler smoke precedes experimental submission. Submission and
  retrieval are separately approved external actions.

### PW-WP10: additional windows and experimental batch

- Run post-choice/full only with an explicit immutable configuration approved
  after PW-WP7.
- Verify adding windows does not change pre-choice scientific arrays.
- Batch only a reviewed session manifest inside the measured resource
  envelope; larger sessions require a new benchmark.

## 9. Performance policy

Before scientific invalidities, one window with all six conditions, 18
targets, five folds, three region configurations, and two representations has:

```text
1 * 6 * 18 * 5 * 3 * 2 = 3,240 model fits
```

All three windows have 9,720 fits. Destination transfer adds evaluation, not
reference fits. At 100 ms, pre/post use five stacked rows per trial and full
uses 40. These are dimensions, not runtime forecasts.

Transform fits number `W * C * T * F * 2` before invalidities because PFC and
HPC transforms are shared across regions and representations. Row weights keep
each training trial's total loss weight at one across five-bin and forty-bin
windows.

Dry run reports model/evaluation/transform counts, stacked shapes, selected
source-unit counts, exact `G`, and exact planned major-array bytes without
loading spikes. Runtime instrumentation separates
input loading, target split construction, transform fitting, stacking,
categorical/numerical fitting, native evaluation, transfer evaluation,
checkpoint serialization, final assembly, and reporting (including plotting).

Prioritize correctness and clarity. Do not add Numba, Cython, a new estimator,
within-session multiprocessing, sparse/on-disk schemas, or caching beyond the
declared reference-model reuse unless profiling after PW-WP7 identifies a real
bottleneck and the user approves a revised plan.

## 10. Implementation discipline

For every code work package:

1. Re-read the surrounding production files and all callers before editing.
2. Verify any newly used library API from installed source or official
   documentation; record the evidence in the handoff.
3. Write and commit focused failing tests.
4. Record the expected RED failure and obtain the required review/approval.
5. Implement the smallest change that satisfies the approved contract.
6. Run focused tests, the full existing task-decoding suite, and `git diff
   --check`.
7. Refactor only while tests remain green.
8. Record commits, commands, results, review, approval, and exact next action
   in Section 1.

Implementation approval for one work package does not authorize a later work
package, experimental data access, cluster synchronization/submission, or
multi-session execution.

## 11. Immediate next action

Complete PW-WP0 validation and user review. If approved, begin PW-WP1 by
writing the listed configuration and master-fold tests and committing the RED
state. Do not begin production implementation directly from documentation
approval.
