# Task-Variable Condition-Generalization Implementation Plan

**Status:** Documentation approved; implementation not yet authorized. No code,
tests, experimental data, or cluster jobs are authorized by this document.

**Scientific contract:**
`docs/task_variable_condition_generalization_spec_v1.md`.

**Reference implementation contracts:** `docs/task_variable_spec_v5.md`,
`docs/task_variable_spec_v6.md`, `docs/task_variable_spec_v7.md`, and
`docs/task_variable_implementation_plan.md` describe the existing
condition-specific per-time-bin decoder and remain unchanged in scientific
meaning.

## 1. Approved planning decisions

- The reference training condition is exactly `correct_rewarded`.
- Canonical training windows are:
  - `pre_choice`: [-0.5 s, 0 s), default;
  - `post_choice`: [0 s, +0.5 s); and
  - `full`: [-2 s, +2 s).
- A run selects any explicit nonempty subset or all three windows.
- Runtime never changes that selection automatically.
- Every configured condition receives a native condition decoder.
- The correct-rewarded decoder is evaluated on every configured destination.
- One decoder is fit per training window and fold, not per evaluation time.
- The frozen decoder is evaluated separately at every existing time bin.
- Revision 1 uses fixed regularization only.
- The first experimental benchmark uses only `pre_choice`.
- Running all windows is decided explicitly after measured performance is
  reviewed.

## 2. Architecture

Keep the workflow inside `src/neural_analysis/task_decoding`, but give it
separate entry points, configuration, result schema, and run identity.

Proposed modules:

| Module | Responsibility |
| --- | --- |
| `generalization_config.py` | Dedicated validated configuration, canonical windows, scientific serialization, and analysis identity. |
| `generalization_splits.py` | Target-specific master block folds shared by reference and native comparisons. |
| `condition_generalization.py` | Pooled-window transforms, weighted model fitting, frozen time-resolved evaluation, native/transfer pairing, and gaps. |
| `generalization_results.py` | NPZ schema, validation, checkpoints, manifests, atomic publication, and read-only loading. |
| `generalization_plotting.py` | Transfer/native time courses and generalization-gap figures. |
| `run_generalization_session.py` | Single-session prepare/new/resume/status lifecycle. |
| `run_generalization_batch.py` | Dry-run planning and session-level local/Slurm batching. |

Reuse existing modules where their public contracts match:

- `targets.py` for exact target construction;
- `conditions.py` for canonical masks;
- `activity.py` for bilateral rate tensors and stable unit identities;
- `modeling.py` for estimator construction, transform semantics, and metric
  definitions where reuse does not alter existing behavior;
- current provenance, execution-guard, resource-measurement, and Slurm
  patterns where they can be shared without coupling result schemas.

Do not route through `spike_behavior.decoding`. It is useful historical
evidence for the analysis question, but its row-wise random split and limited
target/window contracts are not suitable for the authoritative workflow.

Prefer small new public helpers over making the existing per-time-bin decoder
accept a large mode switch. Any extraction from an existing module must retain
the old public interface and pass its complete regression suite.

## 3. Core data flow

For each target:

1. construct the complete chronological target and eligibility arrays;
2. construct one target-specific master block-fold assignment before
   condition filtering;
3. select each training condition/window/fold training tensor;
4. fit training-window scaling and regional PCA from only those rows;
5. reshape transformed training features from
   `(trial, training_time, feature)` to `(trial * training_time, feature)`;
6. repeat trial targets across training bins and assign row weight
   `1 / n_training_bins`;
7. fit one fixed elastic-net estimator;
8. evaluate its unchanged transform and estimator at every evaluation time on
   held-out destination trials;
9. score the reference-transfer and destination-native predictions on the
   same destination fold rows; and
10. calculate fold-paired native-minus-transfer gaps.

The correct-rewarded native model is the reference model. Cache and reuse it
across destinations; never refit it for each transfer pair.

## 4. Configuration contract

The dedicated configuration should contain the current task-decoding path,
region, target, alignment, bin-width, PCA-count, outer-fold, trusted-bound,
and output settings plus:

```json
{
  "condition_names": [
    "all",
    "correct_rewarded",
    "omission",
    "incorrect",
    "switch",
    "stay"
  ],
  "training_window_names": ["pre_choice"]
}
```

`reference_condition` should be saved explicitly as `correct_rewarded` in the
scientific payload but not exposed as a revision-1 user choice. Omitting
`training_window_names` means `pre_choice`; omitting `condition_names` selects
all six canonical conditions. Revision 1 accepts only `choice_time` alignment.
The portable full example will select all conditions explicitly.

The loader must reject unknown fields, unknown/duplicate windows, window edges
that do not align to configured bin edges, an unsupported regularization mode,
output paths outside the session root, and all existing malformed input cases.

## 5. Result and checkpoint design

Use a new analysis version and result schema rather than extending schema 2 of
the current decoder.

Separate unique model identities from evaluation identities so reference
coefficients are not duplicated for every destination.

Model arrays/index tables carry:

```text
training_window, train_condition, target, fold,
region, representation, feature
```

Evaluation arrays/index tables carry:

```text
training_window, train_condition, test_condition, target, fold,
region, representation, evaluation_time, metric
```

Gap arrays carry:

```text
training_window, destination_condition, target, fold,
region, representation, evaluation_time, metric
```

Checkpoint at the `training_window::target` boundary. A checkpoint is complete
only when it contains every required model and evaluation pair for that cell.
Use filename-safe canonical labels and validate the run fingerprint, window,
target, condition set, result schema, source identity, and payload before
reuse.

Dense per-trial out-of-fold prediction storage remains a deliberate decision
gate. Before freezing the schema, calculate its exact bytes for the synthetic
and CT026 envelopes. If retained, predictions use explicit full-trial and
evaluation-time axes with NaN only as a documented ineligible sentinel. If
not retained, save fold scores, counts, fold membership, model parameters, and
enough provenance to reproduce them.

## 6. Dependencies and API verification

Do not introduce a new dependency for revision 1. Use the Python standard
library plus existing NumPy, pandas, scikit-learn, Matplotlib, Pynapple, and
pytest dependencies.

Before production code, inspect the installed scikit-learn source or official
documentation for:

- `LogisticRegression.fit(..., sample_weight=...)`;
- `ElasticNet.fit(..., sample_weight=...)`;
- normalization behavior of sample weights, especially its interaction with
  regularization;
- `GroupKFold` and `StratifiedGroupKFold` behavior; and
- metric behavior for one-class and constant-target inputs.

Pin the verified behavior in focused tests. Do not assume that classifier and
regressor sample-weight semantics are identical.

All local Python commands use `uv run`.

## 7. Tests written and committed before production

Testing follows strict RED, commit, review, GREEN, refactor sequencing. Each
work package below starts with focused failing tests committed separately from
production code.

### 7.1 Configuration and window tests

1. Omitted windows select exactly `pre_choice`.
2. Each canonical window selects the exact expected bin indices at every
   supported bin width.
3. Multiple windows restore canonical order.
4. Duplicate, unknown, empty, and misaligned windows fail.
5. Scientific serialization records bounds, bin indices, and fixed reference.
6. Selecting all windows does not mutate the existing task-decoding config.

### 7.2 Fold and leakage tests

1. Every master-eligible row receives exactly one deterministic outer fold.
2. Behavioral blocks never cross train/test partitions.
3. The same target fold map is reused for every condition and pair.
4. Overlapping correct-rewarded/switch/stay membership cannot leak a row.
5. Every time row from one trial remains in the same fold.
6. Insufficient condition-specific class or target support is unavailable;
   folds are not silently regenerated.
7. Three-fold use occurs only when configured, never as an automatic fallback.

### 7.3 Pooled-window feature tests

1. Trial-major/time-major reshape order is exact and documented.
2. Repeated targets align with stacked feature rows.
3. Every trial's stacked weights sum to one.
4. Adding training bins does not increase total weight per trial.
5. Scaling and PCA use only training-condition, training-block, selected-window
   rows.
6. Evaluation conditions and times never refit or alter the transform.
7. PFC, HPC, and concatenated features retain stable identities and shapes.
8. PCA component limits use training trials times selected training bins.

### 7.4 Model and evaluation tests

1. Exactly one estimator is fit per model identity, independent of evaluation
   time and destination count.
2. The correct-rewarded model is reused for every reference transfer.
3. Each destination receives one native model.
4. The same fitted estimator is evaluated at every time bin.
5. Categorical probabilities retain positive-class orientation.
6. Numerical predictions retain native target units.
7. Transfer and native metrics use identical destination held-out rows.
8. Gaps equal native minus transfer cell by cell.
9. A constant reference target leaves transfer unavailable while an eligible
   native destination remains valid.
10. Convergence and nonfinite artifacts use reviewed unavailable reasons;
    unexpected exceptions propagate.

### 7.5 Result-schema and lifecycle tests

1. Every array has declared axes, dtype, units, and sentinel behavior.
2. Model and evaluation identities are complete and duplicate-free.
3. Reference coefficients are stored once, not once per destination.
4. Multi-window checkpoints cannot be misapplied to another window.
5. Interrupted runs preserve completed checkpoints and exact resume state.
6. Final publication is atomic and complete-run reentry is read-only.
7. Source/config/input changes invalidate resume.
8. Old task-decoding runs remain readable and are never discovered as
   condition-generalization runs.
9. Dry run loads no large spike arrays and fits no estimators.
10. Local detached and Slurm ownership follow the existing no-double-execution
    contract.

### 7.6 Plotting and interpretation tests

1. Each figure identifies training window, train/test conditions, target,
   region, representation, and metric.
2. Transfer and native curves use the same time axis and destination rows.
3. Gap figures use native minus transfer and show a zero reference.
4. Outcome and switch/stay contrasts are not presented as mutually exclusive.
5. Unavailable cells are visually distinct from below-chance scores.
6. Captions state that one pooled-window decoder is evaluated through time.
7. Plotting loads saved results only and never invokes fitting.

### 7.7 Seeded synthetic scientific tests

1. Stable-code fixture: transfer and native decoding are both strong, with a
   small gap.
2. Remapped-code fixture: native decoding is strong and transfer is weak.
3. Absent-code fixture: both are weak without a false remapping conclusion.
4. Delayed-code fixture: one reference decoder produces a shifted destination
   time-course peak.
5. Multi-window fixture: pre-, post-, and full-window models are distinct and
   selecting extra windows does not change pre-choice outputs.
6. End-to-end fixture: loading, tensors, models, checkpoints, persistence,
   validation, summary, figures, resume, and complete reentry all execute.

## 8. Work packages and approval gates

### CG-WP0: specification and implementation plan

- Deliverables: these two documentation files plus explicit clarification in
  the existing task-variable documents.
- Gate: user approves the scientific and execution contracts before tests.
- Current state: documentation drafting authorized; no implementation.

### CG-WP1: configuration, windows, and master folds

- RED: Section 7.1 and 7.2 tests.
- GREEN: `generalization_config.py`, `generalization_splits.py`, and minimal
  shared helpers.
- Gate: exact window bins, canonical ordering, deterministic shared folds, and
  leakage review.

### CG-WP2: pooled-window transforms and weighted models

- RED: Section 7.3 plus sample-weight API characterization.
- GREEN: pooled feature assembly and one-model fitting primitives.
- Gate: trial weights, transform boundaries, stable identities, and no
  evaluation-time refitting.

### CG-WP3: transfer, native evaluation, and gaps

- RED: Section 7.4 tests.
- GREEN: reference reuse, native fits, time-resolved evaluation, metrics, and
  paired gaps.
- Gate: exact estimator call counts and synthetic stable/remapped behavior.

### CG-WP4: schema, checkpoints, and persistence

- RED: scientific arrays, validation, sentinel, checkpoint, and compatibility
  tests from Section 7.5.
- GREEN: `generalization_results.py` and immutable atomic publication.
- Gate: independent result validation and exact dense-prediction byte decision.

### CG-WP5: single-session lifecycle

- RED: prepare/new/resume/status, execution ownership, failure preservation,
  and complete-reentry tests.
- GREEN: `run_generalization_session.py` and orchestration.
- Gate: seeded one-shot and interrupted/resumed runs.

### CG-WP6: reporting and saved-result loading

- RED: Section 7.6 tests.
- GREEN: `generalization_plotting.py`, summaries, default PNGs, and optional
  integration into the existing read-only app through a separate routed view.
- Gate: plotting never loads raw spikes or fits a model.

### CG-WP7: batch planning and Slurm execution

- RED: dry-run dimensions, resource profiles, per-session isolation, array
  manifest, and failure-locality tests.
- GREEN: `run_generalization_batch.py` plus reviewed shell/Slurm support.
- Gate: exact local/cluster dry-run agreement and synthetic scheduler smoke.

### CG-WP8: synthetic end-to-end scientific gate

- Run every Section 7.7 fixture through the real public entry point.
- Require exact deterministic rerun, complete schema validation, all expected
  figures, and no unexplained invalid cells.
- Gate: user approves synthetic outputs before experimental data.

### CG-WP9: CT026 pre-choice benchmark

- Use all 18 targets, all six conditions, both representations, three region
  configurations, five outer folds, 100 ms bins, fixed regularization, and only
  `pre_choice` training.
- Run dry-run first; compare predicted dimensions with the actual result.
- Record wall time, CPU time, peak RSS, fit counts, evaluation counts,
  invalidity, operation timings, checkpoint bytes, and final output bytes.
- Gate: user inspects scientific plots and approves any broader window run.

### CG-WP10: post-choice/full-window characterization

- If approved after CG-WP9, run `post_choice` and `full` as separately named
  immutable configurations or one explicitly configured multi-window run.
- Verify that adding windows leaves pre-choice results byte-for-byte unchanged
  where the schema permits direct comparison.
- Decide explicitly whether routine session configs select pre-choice only,
  pre/post, or all three. Do not encode a runtime-triggered policy.

### CG-WP11: experimental batch

- Prepare a reviewed session manifest and resource envelope.
- Default parallelism is one session per worker/job-array element.
- Admit only sessions within the measured no-larger envelope or stop for a new
  benchmark.
- Experimental submission and result retrieval remain separately approved
  external actions.

## 9. Performance plan

The current six-condition CT026 task-decoding run requested 129,600
time-bin-specific fixed-mode cells and spent approximately 48,070 seconds in
modeling. The generalization workflow replaces time-bin-specific training with
pooled-window training:

| Selected windows | Unique fixed-mode models before invalidity |
| --- | ---: |
| `pre_choice` | 3,240 |
| `post_choice` | 3,240 |
| `full` | 3,240 |
| all three | 9,720 |

These are fit counts, not runtime forecasts. At 100 ms, half-second models use
five stacked rows per trial and full-window models use 40. Direct-unit solver
cost may scale with both row count and feature count. Benchmark rather than
claiming a 40-fold speedup.

Instrumentation must separate:

- transform fitting;
- stacked feature construction;
- categorical/numerical estimator fitting;
- reference evaluation across destinations/times;
- native evaluation;
- checkpoint serialization; and
- final assembly/plotting.

Dry run reports exact planned dimensions and array bytes. Resource decisions
use measured peak RSS and elapsed/CPU time. Do not optimize with Numba, Cython,
within-session multiprocessing, or a different estimator without profiling,
scientific review, and user approval.

## 10. Documentation deliverables during implementation

After the contracts are approved and implementation begins, maintain:

- a dated live handoff in this plan;
- RED/GREEN commit identities for every work package;
- exact focused/full test commands and outcomes;
- API-verification evidence;
- synthetic fixture parameters and expected effects;
- dry-run and measured fit/evaluation counts;
- benchmark resource evidence;
- output inspection notes; and
- explicit remaining approval gates.

The existing task-variable specifications should receive no new
condition-generalization requirements. They remain references for the current
per-time-bin workflow; all new behavior belongs here.

## 11. Next action after documentation approval

Review the completed documentation diff. If the scientific contract is
approved, authorize CG-WP1 planning and tests separately. The first code change
must be a RED test commit; implementation must not begin from documentation
approval alone.
