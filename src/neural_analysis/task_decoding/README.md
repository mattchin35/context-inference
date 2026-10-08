# Task-variable decoding package

This package owns the frozen configuration, target construction, neural
activity tensors, grouped decoding, durable execution, saved results, and
plots for the offline task-variable analysis. Dependencies point outward to
the existing behavior utilities, neural-session metadata, spike loading, and
population PCA modules; those modules do not depend on this package. The
integrated webapp imports only the saved-result loader and plotting layer, not
the fitting pipeline.

The scientific requirements are the revision-5 base contract plus its active
revision-6 convergence amendment in `docs/task_variable_spec_v5.md` and
`docs/task_variable_spec_v6.md`. Work-package history remains in
`docs/task_variable_implementation_plan.md`. This file is the maintainer map,
not a duplicate specification.

## File and entry-point map

| File | Responsibility and public entry points |
| --- | --- |
| `__init__.py` | Package marker; it deliberately re-exports no broad convenience API. |
| `config.py` | Frozen constants and `TargetDefinition`, `RegionConfig`, `TaskDecodingConfig`; `load_task_decoding_config` validates portable JSON and `scientific_config_payload` freezes scientific identity. |
| `targets.py` | `validate_augmented_trials` and `build_target_table` validate the full chronological table and construct encoded targets plus masks. |
| `activity.py` | `ProbeCoverage`, `RegionActivity`, `SessionRateTensors`, and `ActivityDryRunReport`; probe inspection/loading, common tensor construction, target-mask projection, byte estimates, memory-budget selection, and dry-run inspection. |
| `modeling.py` | Grouped split, transform, fit, fold, and target result records; public split/feature/estimator/scoring/aggregation functions and `decode_target`. |
| `results.py` | Input/source/run fingerprints, source-cleanliness validation, target checkpoints, and validated atomic `save_task_decoding_run` / `load_task_decoding_run`. |
| `plotting.py` | `plot_decoding_heatmap`, `save_default_decoding_figures`, `summarize_unit_coefficients`, and `plot_unit_coefficients` operating on validated saved results. |
| `resource_usage.py` | `ResourceUsageTracker`, RSS normalization, shared resource-envelope validation, and atomic resumable `resource_usage.json` evidence. |
| `pipeline.py` | `plan_task_decoding_session`, `prepare_task_decoding_run`, `run_prepared_task_decoding`, and `inspect_task_decoding_status`; all launch paths share these preparation/execution seams. |
| `run_session.py` | Standard-library CLI `main` for public `dry-run`, `new`, `resume`, and `status` modes; numerical imports occur only after one-thread environment variables are set. |
| `run_batch.py` | Standard-library CLI `main` plus `read_config_list`; bounded multi-session `dry-run` and foreground `new`, with evidence-gated cross-session workers. |
| `slurm.py` | Standard-library one-shot `submit-new`, exact-directory `submit-resume`, and read-only `status` operations; it owns the fixed resource receipt and `sacct` parsing but no scientific computation. |
| `../psth_webapp.py` and `../webapp/task_decoding_views.py` | Existing Streamlit entrypoint and early-routed **Task-variable decoding results** view. Discovery and rendering use completed saved runs only. |
| `../../../docs/examples/neural_analysis/task_decoding_config.json` | Portable, explicit configuration template. Its repository path is `docs/examples/neural_analysis/task_decoding_config.json`. |
| `../../shell_scripts/task_variable_decoding_slurm.sh` | Thin single-session Slurm wrapper with fixed one-CPU, 3-GiB, five-hour resources, tracked-clean checkout checks, offline locked execution, and direct TERM propagation. Its repository path is `src/shell_scripts/task_variable_decoding_slurm.sh`. |

## Data contracts

### Inputs and target table

The augmented input is a chronological `pandas.DataFrame` with shape
`(n_trials, n_columns)`. Every run requires `cur_trial`, `cur_block`, `action`,
`experimenter_reward_given`, and its configured alignment column. Alignment is
`choice_time` or `start_time` in UTC Unix seconds. Selected targets add their
source columns:

| Target group | Required source columns |
| --- | --- |
| Current state/action/correctness | `state_int`, `action`, `correct` as selected |
| Previous/next action and switch/stay | `action`; previous rewarded also uses `reward` |
| Counts and indices | `consecutive_omissions`, `consecutive_rewards`, `cur_trial`, `cur_trial_in_block`, `rewards_in_block` as selected |
| Model-derived numerical targets | `Qlearning_rel_value`, `FQlearning_rel_value`, `HMM_rel_value_logodds`, `HMM_rel_value_logodds_decay`, `relative_doubt_index` as selected |

`build_target_table` returns a range-indexed table with shape
`(n_trials, 5 + 2 * n_targets)`. It contains original trial/block identity,
baseline validity and reason, and one encoded float value plus Boolean valid
mask per target. Categorical values are encoded class 0/1. Numerical values
retain their native source units and are not standardized as targets.

Manual-reward, no-choice, and missing-alignment rows remain in this complete
table with explicit invalidity. Chronological previous/next derivations never
bridge an invalid adjacent choice.

### Activity and result axes

Each `RegionActivity` contains one region's selected unit IDs and aligned spike
times in UTC seconds. `build_session_rate_tensors` creates PFC and HPC float64
rates with shape `(common_neural_tensor_row, time, unit)` in Hz. Time-bin edges
and centers are event-relative seconds over the frozen `[-2, 2]` second window.
PFC and HPC share the same trial-row axis, but retain separate unit axes.

The common tensor contains only bilateral rows whose complete alignment window
is covered. `trial_row_indices` maps every tensor row back to `row_position` in
the complete target table. `project_target_mask_to_tensor_rows` intersects a
target's scientific eligibility with that mapping. Missing alignment rows are
not represented as artificial all-zero firing rates: zero-filling would make
missing coverage look like measured silence and could bias both PCA and the
decoder.

The final `results.npz` contains primitive, non-object arrays and scalar JSON
metadata. Central axes are:

- scores: `(target, region, representation, metric, time, fold)`;
- fit status/counts: `(target, region, representation, time, fold)`;
- encoded values and eligibility: `(target, full_table_row)`;
- outer fold IDs: `(target, common_neural_tensor_row)`;
- coefficients: `(target, region, representation, time, fold, feature)`; and
- tuned candidate audit, when applicable: target, outer fold, region,
  representation, time, candidate, and inner fold.

Regions are `PFC`, `HPC`, and `PFC+HPC`; representations are `pca` and
`units`. Metrics are balanced accuracy and AUC for categorical targets and R2
for numerical targets. Score units are fractions or coefficient of
determination. Coefficient/intercept units, seconds, Hz, and elapsed seconds
are stored explicitly in `meta`.

## Configuration and saved schema

`load_task_decoding_config` is the sole JSON loader. The required path/region
objects identify metadata, augmented trials, feature parameters, distinct PFC
and HPC probes, channel labels, inside-brain policy, cluster groups, and
optional zero-based channel IDs. User controls select alignment, supported bin
width, regional PC counts, target subset, `fixed` or `tuned` regularization,
legal fold counts, optional trusted UTC bounds, and the contained output root.
Unknown fields and invalid cross-field combinations are rejected.

`scientific_config_payload` adds the analysis version and code-owned constants:
the window, seed, coefficient tolerance, estimator controls, and tuning grid.
It excludes execution-only output paths. The run fingerprint binds this
scientific payload to input byte identities, the exact feature-parameter file,
the scoped scientific source fingerprint, and session identity.

`results.py` owns result schema version 1. `save_task_decoding_run` validates
all names, dtypes, shapes, axes, units, target/fold coherence, transform reuse,
tuning audit, and identity before atomic publication. `load_task_decoding_run`
uses `allow_pickle=False`, repeats validation, and returns a mapping containing
`arrays`, `meta`, `scientific_config`, `input_manifest`, and the run
fingerprint. Callers never treat arbitrary NPZ content as trusted.

## Scientific validity and leakage boundaries

Target eligibility is constructed on the complete chronological table, then
projected to the common bilateral neural tensor. Outer and inner folds group by
behavioral block. Fold planners reject impossible class coverage, empty folds,
or invalid grouped partitions; unavailable targets/cells carry compact reason
codes rather than fabricated scores.

Scaling and PCA are fit on training rows only. PFC and HPC transforms are fit
separately, and combined features concatenate those training-derived regional
features. Held-out rows are transformed with the corresponding training fold
state. Tuned mode performs grouped inner selection strictly inside an outer
training fold and refits the selected candidate before outer scoring.

After each target finishes, `pipeline.py` serializes primitive target arrays
through `save_target_checkpoint`. A checkpoint is reused only when its full
run fingerprint, target identity, schema, and arrays validate. Resume therefore
skips completed targets without accepting stale scientific work.

## Existing code deliberately reused

- `session_metadata.py` resolves the canonical session and two probe records.
- `spike_behavior/loading.py` supplies the established aligned spike loader.
- `population/pca.py` supplies the existing regional PCA calculation.
- `behavior_analysis/project_utils.py` supplies canonical missing/no-choice
  semantics used by target validation.
- NumPy, pandas, scikit-learn, and Matplotlib provide arrays/tables,
  elastic-net models, metrics, and light-mode saved figures.

Do not duplicate these functions inside this package. If one of their public
contracts is insufficient, adapt its use here rather than modifying an
external dependency.

## Execution, ownership, and publication

Configuration paths are portable and session-relative. Preparation resolves
them once, verifies containment, writes canonical scientific/input/source
sidecars, snapshots `run_session.py`, `run_batch.py`, and `pipeline.py`, and
saves exact resume/status commands before neural loading begins. Persistent
execution revalidates live inputs and scoped source identity against those
sidecars.

Foreground, detached local, and scheduled Slurm launches call the same
`run_prepared_task_decoding` function. A detached parent creates one child in a
new process session and atomically publishes `local_launch.json`; the child
waits briefly for its matching receipt before claiming the execution guard.
The Slurm path prepares the same immutable sidecars, marks submission pending
before one `sbatch` call, and writes only `slurm_submission.json` after
acceptance so a fast job's running state cannot regress. The wrapper executes
the private prepared-run seam with `uv run --frozen --no-sync --offline` and
translates scheduler TERM into the existing interruption boundary. The guard
enforces a single writer. Normal interruption records `interrupted`, releases
ownership, and leaves valid checkpoints for resume. CLI commands return only
bounded status/acceptance or after foreground computation; scientific results
are returned through the immutable run directory, not as a large command
value.

State and JSON sidecars use sibling-temporary replacement. Target checkpoints,
final `results.npz`, summary, and PNGs are validated before
`final_results_published` becomes true. Completed-run discovery requires both
`lifecycle == "complete"` and that publication flag. An identical completed
fingerprint is skipped; `--rerun` allocates a new directory rather than
overwriting it.

Dry-run resource provenance includes trial/unit/time/fold/fit dimensions,
per-target common-row/class/block/grouped-fold diagnostics, exact tensor
allocation bytes, runtime/platform/thread identity, and source sizes. Every
execution atomically updates `resource_usage.json` after target checkpoints
and at terminal completion/failure/interruption. The evidence records
normalized peak RSS bytes, cumulative wall/user/system seconds across resume,
per-target split/transform/feature/estimator timings, fit counts, and output
bytes. Batch execution remains one worker unless an explicit compatible
completed measurement is supplied; evidence is never inferred from an
arbitrary run.

Final reporting renders the applicable heatmaps before publishing the
terminal resource snapshot. `summary.md` therefore receives the complete
post-figure wall/CPU/RSS evidence rather than the earlier running checkpoint.
If figure or summary publication fails, terminal failure evidence replaces
the provisional resource status and an explicit resume publishes only missing
immutable artifacts.

The WP11 cluster path is deliberately single-session. The fixed request is one
CPU, `3G`, and `05:00:00`; login-side tensor memory and active cgroup memory are
both guarded at 50 percent before large allocation. Status combines durable
state with one `sacct` query and never polls. If accounting fails or has no
exact root-job row, status still returns the durable pipeline state with
`scheduler: null` and an explicit `scheduler_error`; it does not infer a live
scheduler state or hide completed results. Exact-commit/offline checks and
non-destructive staged transfer are documented in the scientist README. WP13
separately owns any array wrapper; it must reuse this preparation/execution
path rather than duplicate scientific logic.

## Focused test ownership

| Implementation | Focused tests |
| --- | --- |
| `config.py` | `task_decoding/test_config.py` |
| `targets.py` | `task_decoding/test_targets.py` |
| `activity.py` | `task_decoding/test_activity.py` |
| `modeling.py` | `task_decoding/test_modeling.py` |
| `resource_usage.py` | `task_decoding/test_resource_usage.py` |
| `results.py` | `task_decoding/test_results.py` |
| `pipeline.py`, `run_session.py` | `task_decoding/test_pipeline.py` |
| `run_batch.py` | `task_decoding/test_run_batch.py` |
| `slurm.py`, `task_variable_decoding_slurm.sh` | `task_decoding/test_slurm.py` |
| `plotting.py` | `task_decoding/test_plotting.py` |
| saved-result webapp | `test_webapp_task_decoding.py`, `test_webapp_task_decoding_rendering.py`, and routing tests |
| READMEs, example, and command surface | `test_task_decoding_documentation.py` |

`test_synthetic_integration.py` is reserved for the later WP9 end-to-end gate.
Run focused tests before the full task-decoding and affected webapp suites.

## Adding a future target

1. Add one `TargetDefinition` in canonical order in `config.py`, including its
   family, source column, derivation, display label, and positive label when
   categorical.
2. Add only the target-specific encoding rule needed in `targets.py`; reuse the
   common baseline and chronological validity rules.
3. Extend target/config tests first for required columns, values, masks,
   metadata, missingness, malformed values, and canonical ordering.
4. If the target uses an existing family, leave activity, split, estimator,
   pipeline, plotting, and webapp code unchanged. Those modules consume target
   metadata generically.
5. Extend saved-schema fixtures and the synthetic integration test, then update
   the portable example and user documentation.

Do not add a generic estimator registry or target plugin mechanism for one new
variable. A small explicit definition and encoding branch is easier to audit.

## Non-goals

This package does not replace or absorb the older exploratory PCA decoder. It
does not pool sessions or units, support multiple probes per region, compute
permutation significance or decoding onset, standardize numerical targets,
tune bins/PC counts/class weights/thresholds, perform live fitting in
Streamlit, scan arbitrary filesystem paths, or manage a general workflow
engine. Scheduler wrappers and batch arrays remain separately gated rather
than being hidden inside the local pipeline.
