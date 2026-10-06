# Task-Variable Decoding Implementation Plan

**Status:** Proposed documentation-only plan; planning only. This document does not authorize
implementation, test creation, data mutation, or decoding runs. Implementation
may begin only after the user separately approves the plan and requests code
changes.

**Scientific contract:** `docs/task_variable_spec_v5.md`.

## Live handoff snapshot

**Snapshot date:** 2026-10-05

**Current phase:** WP0 documentation revision. The revision-5 specification
and this implementation plan exist, but the user has authorized documentation
work only. No production code, tests, experimental data, decoding output,
benchmark, local long run, or cluster action is authorized.

**Repository state at this snapshot:**

- branch: `refactor`;
- HEAD before this documentation overlay: `e9e2ed7`;
- `docs/task_variable_spec_v5.md` and this plan were staged by the user after
  their initial creation;
- `docs/task_variable_spec_v4.md` has a pre-existing title-indentation
  modification that this work does not own; and
- all unrelated dirty/untracked files remain outside this plan's ownership.

**Completed planning evidence:**

- `src/neural_analysis`, `docs/SoftwareDesign.md`, and revision 4 were audited;
- revision 5 records the resolved scientific/data contracts;
- CT026 2026-08-03 input presence and small-table/alignment metadata were
  inspected read-only; and
- the implementation architecture, tests-first sequence, saved-run contract,
  integrated webapp view, and benchmark strategy are drafted below.

**Next exact action:** finish and review this documentation revision. A later
implementation begins only after the user explicitly approves the plan and
requests implementation. At that point the Sol supervisor starts WP1 with a
fresh worktree/HEAD audit; it must not infer implementation authority from the
existence or staging of these documents.

### Authority order

When resuming, use this order:

1. `docs/task_variable_spec_v5.md` owns scientific definitions and defaults.
2. This document owns implementation order, file ownership, tests, agent
   assignments, documentation deliverables, and live status.
3. `AGENTS.md` and `docs/SoftwareDesign.md` govern TDD, readability, data
   contracts, dependencies, and user approval.
4. The top live snapshot and package records in this document own execution
   state. Historical chat summaries are not a substitute.
5. Git history and independently reproduced commands are evidence; an agent
   summary alone is not.

### Work-package state

| Package | State at snapshot | Next gate |
| --- | --- | --- |
| WP0 documentation approval | Active; documentation only | User review and explicit implementation request |
| WP1 behavioral feature | Not authorized | Sol freezes scope and assigns tests-only Terra task |
| WP2 configuration and targets | Not authorized | WP1 GREEN and recorded handoff |
| WP3 activity loading and coverage | Not authorized | WP2 GREEN and recorded handoff |
| WP4 grouped modeling | Not authorized | WP3 GREEN and independent numerical test-design review |
| WP5 results and session pipeline | Not authorized | WP4 GREEN and saved-schema freeze |
| WP6 batch runner | Not authorized | WP5 GREEN |
| WP7 plotting and webapp | Not authorized | WP5 saved loader stable; WP4 metrics stable |
| WP8 documentation and examples | Not authorized | CLI/webapp interfaces stable through WP7 |
| WP9 synthetic integration | Not authorized | WP1-WP8 focused gates GREEN |
| WP10 CT026 preflight/benchmark | Not authorized | WP9 GREEN plus explicit real-session benchmark approval |
| WP11 conditional cluster wrapper | Conditional and not authorized | Benchmark shows need; user approves wrapper/cluster work |
| WP12 CT026 scientific inspection | Not authorized | User accepts benchmark and authorizes exact run |

### Resume checklist

A new or returning Sol supervisor must:

1. read revision 5, this complete plan, `AGENTS.md`,
   `docs/SoftwareDesign.md`, and the most recent package record;
2. record current branch, HEAD, `git status --short`, staged diff, and unstaged
   diff without modifying either;
3. distinguish user/pre-existing changes from package-owned changes and stop
   for user direction if ownership is uncertain;
4. reproduce the last claimed RED/GREEN command before accepting it as a gate;
5. verify the current user authorization, especially for implementation,
   CT026 computation, or Slurm use;
6. resume at the first incomplete gate, not at the beginning of a completed
   package; and
7. update this snapshot before delegating or editing.

If a Terra worker or Sol reviewer is interrupted, its partial report is not a
completed gate. The supervisor re-audits the bounded diff and reproduces the
last claimed command before resuming the same worker or assigning a replacement.
Unknown edits are preserved and escalated; they are never reset or absorbed
into a package.

### Live update record

After every tests-only commit, implementation commit, documentation gate,
benchmark, interruption, or user approval, the Sol supervisor updates this
section and adds a concise package record containing:

- date/time, package and gate;
- supervisor, Terra worker, and independent reviewer model/effort;
- branch, starting/ending HEAD, and worktree ownership;
- exact files changed;
- exact commands and RED/GREEN outcomes;
- commit IDs, if any;
- scientific/configuration decisions and unresolved risks;
- real-data or external actions taken (normally none);
- output/run paths and measured performance, when authorized; and
- the one exact next action and authorization required.

Do not rewrite earlier package evidence to make a later run look cleaner.
Correct errors with a dated correction note.

Use this template for each appended record:

```text
#### YYYY-MM-DD HH:MM - WPx gate
- State:
- Authorization:
- Sol / Terra / reviewer:
- Start HEAD / end HEAD:
- Owned files:
- RED command and result:
- GREEN/regression commands and results:
- Commits:
- Real-data or external actions:
- Findings and unresolved risks:
- Exact next action:
```

## 1. Objective

Implement a readable, single-session task-variable decoding pipeline that:

1. consumes the existing augmented behavioral table and aligned spikes;
2. constructs deterministic categorical and numerical targets;
3. compares PFC, HPC, and PFC + HPC using regional PCA or direct units;
4. evaluates elastic-net decoders with grouped, leakage-safe CV;
5. saves complete offline results and provenance;
6. displays saved results in one read-only view inside the existing Streamlit
   webapp; and
7. reports realistic runtime and memory expectations before routine use.

The implementation should be direct scientific Python. It should use small
functions, explicit arrays and tables, frozen configuration dataclasses, and
docstrings that state types, shapes, axes, units, and return values. It should
not introduce a framework, plugin system, estimator hierarchy, or generalized
workflow engine.

## 2. Design constraints

### 2.1 Simplicity and readability

- Prefer module-level functions to behavior-heavy classes.
- Use dataclasses only for configuration and compact data records.
- Pass narrow arrays/series to computational helpers when dataframe-level row
  alignment is not their responsibility.
- Keep transformations visible: trial selection, rate binning, scaling, PCA,
  fitting, scoring, and aggregation should be separately testable steps.
- Reuse existing loaders and rate-tensor code when their contracts match.
- Avoid generic `utils.py` or `helpers.py` modules.

### 2.2 Scope boundaries

The first implementation will not:

- modify the neural-session metadata schema;
- refactor or replace the existing exploratory PCA decoder;
- support multiple probes per brain region;
- pool units across sessions;
- compute permutation significance or decoding onset;
- standardize numerical targets;
- tune PC counts, bin widths, class weights, or thresholds;
- run decoding inside Streamlit;
- add a second viewer application;
- add a new dependency; or
- perform major performance optimization before profiling.

### 2.3 Required workflow

Every implementation package follows Red-Green-Refactor:

1. write focused tests;
2. run them and record the expected failure;
3. commit the tests before implementation;
4. implement the smallest coherent package;
5. run focused and affected regression tests; and
6. refactor only while tests remain green.

No production analysis is run until unit tests and a synthetic end-to-end
pipeline are green. The CT026 run begins with a read-only preflight and bounded
benchmark, not a full tuned analysis.

## 3. Audited codebase baseline

### 3.1 Reusable code

| Existing code | Planned use |
| --- | --- |
| `src/neural_analysis/session_metadata.py` | Load and resolve authoritative session/probe paths. Do not duplicate metadata parsing. |
| `spike_behavior.loading.load_sorter_metadata` | Load spike-cluster assignments and curated cluster metadata. |
| `spike_behavior.loading.load_aligned_spikes` | Load aligned UTC spike timestamps. Extend validation outside this function rather than changing its public contract unnecessarily. |
| `spike_behavior.loading.validate_aligned_spike_inputs` | Confirm spike/cluster one-to-one length. |
| `spike_behavior.loading.build_spike_tsgroup` | Construct Pynapple unit spike series. |
| `spike_behavior.loading.select_units_by_channels` | Retain curated `good`/`mua` clusters on selected channels. |
| `population.pca.build_trial_unit_rate_tensor` | Produce trial x time x unit unsmoothed firing rates in Hz. |
| `behavior_analysis.session_analysis.make_augmented_trial_df` | Add the general `rewards_in_block` feature at the existing augmentation boundary. |
| `webapp.session_inputs` | Own the new view name and availability entry. |
| `webapp.app._start_metadata_webapp` | Existing early view-selection boundary. Route saved decoding results before population controls and raw spike loading. |

The Pynapple-backed population rate-tensor function is the primary binning
path. The NumPy implementation remains useful as a test/reference path; the new
pipeline should not fork another spike-binning implementation.

### 3.2 Code not reused as the decoding core

`src/neural_analysis/population/decoding.py` implements a different,
exploratory/limited analysis: choice-only windows, ungrouped stratified CV,
accuracy/permutation scoring, and one-probe PCA controls. Its model-fitting
functions do not meet revision-5 targets, grouping, metrics, regional, or
saved-result contracts.

Do not stretch those functions with many flags. Leave their public behavior
unchanged and build the revision-5 pipeline in a focused package. Shared rate
binning is reused at the lower boundary.

### 3.3 Resolved inconsistencies from revision 4

| Revision-4 ambiguity or mismatch | Resolution in revision 5 and this plan |
| --- | --- |
| Metadata may point to raw trials | Configuration explicitly names augmented CSV and feature-parameter JSON. |
| Missing number-of-rewards-in-block target | Add `rewards_in_block` to general behavior augmentation. |
| HMM columns say `logodds` | Source code returns tanh-transformed signed belief; keep column names but correct scientific labels. |
| State parser supports dark state 2 | Binary current-state decoding accepts only states 0 and 1. |
| Region identity could be inferred from probe names | Require explicit PFC/HPC probe and channel-selection configuration. |
| Cluster IDs can collide across probes | Persist `probe_id:cluster_id` identities. |
| Trial and alignment eligibility were incomplete | Define baseline, target-specific, matched-trial, and full-window coverage masks. |
| PC count could be silently reduced | Record requested/effective count and visible cap status. |
| Solver warnings could be ignored | Treat nonconvergence as an invalid candidate/fold. |
| Separate result viewer was proposed | Use one integrated, read-only existing-webapp view. |
| Runtime was unknown | Add bounded synthetic and CT026 benchmark gates. |

## 4. Proposed package structure

### 4.1 Behavior change

`src/behavior_analysis/session_analysis.py`

- Add a small, independently tested function that computes
  `rewards_in_block` from `cur_block`, `action`,
  `reward`, and normalized experimenter-reward flags.
- Call it from `make_augmented_trial_df`.
- Preserve every existing column and return shape; this is one additive public
  column.

The helper should iterate once in chronological row order. A dataframe groupby
expression is not preferred if it obscures the entering-trial update order or
manual/no-choice behavior.

### 4.2 New analysis package

Create `src/neural_analysis/task_decoding/` with:

| Module | Responsibility | Principal public interface |
| --- | --- | --- |
| `config.py` | Frozen user settings, region definitions, defaults, JSON serialization, and lightweight value validation. | `TaskDecodingConfig`, `RegionConfig`, `load_task_decoding_config(...)` |
| `targets.py` | Augmented-table validation, chronological shifted targets, source mappings, and target-specific eligibility. | `build_target_table(...)`, `validate_augmented_trials(...)` |
| `activity.py` | Probe loading, channel/unit selection, trusted coverage, matched trial windows, rate tensors, stable feature identities. | `load_region_activity(...)`, `build_session_rate_tensors(...)` |
| `modeling.py` | Grouped splits, fold-local standardization/PCA, fixed/tuned elastic-net fits, metrics, coefficients, and validity. | `make_outer_splits(...)`, `decode_target(...)` |
| `results.py` | In-memory result record, NPZ/config/log save-load contract, completion state, and compatible-run discovery. | `save_task_decoding_run(...)`, `load_task_decoding_run(...)` |
| `plotting.py` | Light-mode heatmaps, coefficient summaries, captions, and PNG export from saved results. | `plot_decoding_heatmap(...)`, `plot_unit_coefficients(...)` |
| `pipeline.py` | Assemble loading, targets, activity, decoding, checkpoints, logging, and reporting for one session. | `run_task_decoding_session(...)`, `plan_task_decoding_session(...)` |
| `run_session.py` | Thin single-session offline CLI with explicit `dry-run`, `new`, and `resume` modes. | `main(...)` |
| `run_batch.py` | Thin session-list offline CLI with `dry-run` and `new` modes; session-level parallelism only. | `main(...)` |
| `README.md` | File-by-file ownership, dependency direction, public entry points, array/result contracts, and developer extension notes. | Documentation only |

No inheritance or Protocol is needed initially: there is one loader path and
one model family per target family. Add an abstraction only if a second real
implementation creates a concrete need.

### 4.3 Existing webapp integration

Add `src/neural_analysis/webapp/task_decoding_views.py`.

This module:

- discovers completed decoding-run directories below the selected session;
- loads only saved result/config files;
- renders selectors, heatmaps, fold details, and coefficients; and
- contains no estimator or raw-spike loading call.

Add one `PLOT_VIEW_TASK_DECODING` constant and option in
`webapp/session_inputs.py`. In `webapp/app.py`, route this
view immediately after the metadata session and view are selected, before
`_metadata_population_controls` and
`load_metadata_viewer_data_cached`.

The initial result view is metadata-session-only. The legacy manual-path route
does not have a reliable session root/run directory contract and should show a
short message directing the user to launch with `neural_session.json`
rather than adding another path browser.

For a metadata session, keep the view selectable even when no completed run
exists or raw behavior/spike sources are currently unavailable. The view itself
then explains that no saved run is available. This requires a small explicit
availability case rather than letting the existing generic spike-readiness
rule classify it as a live-computation view.

### 4.4 Required user and developer documentation

Documentation is a required implementation deliverable, not cleanup after the
scientific code is complete.

Update `src/neural_analysis/README.md` with a short
**Task-variable decoding** section that explains:

- required metadata, augmented-trial, feature-parameter, and configuration
  files;
- how to copy and edit the example configuration;
- the exact local `dry-run`, `new`, and `resume`
  commands;
- how to perform a batch dry run and batch launch;
- where run directories, checkpoints, logs, NPZ results, summaries, and PNGs
  are written;
- how to open the saved results in the existing webapp;
- the difference between fixed and tuned mode, including the large tuned-mode
  work-count warning;
- how matching completed runs are skipped and how an explicit rerun creates a
  new immutable directory; and
- the cluster path only if WP11 is activated.

Create `src/neural_analysis/task_decoding/README.md` with:

- one concise paragraph describing the package and its dependency direction;
- a table describing every Python file and its public entry points;
- input dataframe columns and rate/result array shapes, axes, and units;
- configuration and result-schema summaries;
- which existing modules are deliberately reused;
- how target eligibility, fold validity, leakage prevention, and checkpoints
  work;
- a map from package files to their focused tests;
- a short recipe for adding a future target without changing unrelated
  modules; and
- explicit non-goals so a future maintainer does not merge this work into the
  older exploratory decoder.

Add `docs/examples/neural_analysis/task_decoding_config.json` as a
portable template. It uses obvious placeholder paths and probe IDs, contains
all required fields, and is accepted by the same config loader used for real
runs. Do not put CT026 absolute paths in the reusable example.

The top-level README is for scientists running the analysis. The in-package
README is for maintainers reviewing or extending it. Neither should duplicate
the complete scientific specification.

### 4.5 Straightforward offline command surface

The planned local workflow is:

```bash
cp docs/examples/neural_analysis/task_decoding_config.json \
  /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_session dry-run \
  --config /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_session new \
  --config /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_batch dry-run \
  --config-list /path/to/task_decoding_configs.txt

uv run python -m src.neural_analysis.task_decoding.run_batch new \
  --config-list /path/to/task_decoding_configs.txt --workers 4
```

Every successful `new` launch prints and saves its exact resume
command:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session resume \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp>
```

`dry-run` validates metadata, configuration, augmented columns,
feature parameters, output paths, and planned work count without loading spike
arrays or fitting models. `new` creates the run directory and resume
record before large-array loading. `resume` accepts only the exact run
directory and saved configuration; it does not reconstruct settings from the
current command line.

The batch config list is a UTF-8 text file with one configuration path per
nonblank, non-comment line. Batch resume is deliberately per-session through
the printed single-session resume commands; do not add a second batch
checkpoint format.

The CLI returns nonzero on invalid configuration or failed execution and
prints a short actionable error plus the run directory/resume command when one
exists. Do not require a notebook or Streamlit to start offline computation.

### 4.6 Conditional cluster wrapper

Do not build cluster support preemptively. WP10 first measures the default
fixed-mode workload locally. If the measured/projected runtime or memory makes
local use impractical, the user decides whether to activate WP11.

If activated, add one thin Slurm wrapper:

`src/shell_scripts/task_variable_decoding_slurm.sh`

It forwards the same Python CLI and supports only:

```bash
sbatch src/shell_scripts/task_variable_decoding_slurm.sh new \
  --config /path/to/session/task_decoding_config.json

sbatch src/shell_scripts/task_variable_decoding_slurm.sh resume \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp>
```

The wrapper:

- contains no scientific defaults or duplicate configuration parsing;
- invokes the same `run_session.py` entry point used locally;
- keeps CPU, memory, wall-time, environment activation, repository commit, and
  Slurm log location visible;
- never submits another job automatically;
- never selects the latest run implicitly;
- preserves exact target checkpoints and resume semantics;
- requires a successful local dry run before submission; and
- receives focused forwarding/failure tests plus README instructions.

If the benchmark instead identifies a need for within-session parallel model
fitting, stop and revise this plan. A scheduler wrapper provides unattended
wall time and memory; it does not itself justify a new parallel algorithm.
No `sbatch` command is run without separate explicit user approval.

## 5. Configuration contract

`TaskDecodingConfig` should contain only scientific/run settings that
must be explicit:

- analysis version;
- session metadata path;
- augmented-trial path;
- trial-feature-parameter path;
- PFC and HPC `RegionConfig` records;
- alignment;
- window and bin width;
- requested PFC/HPC PC counts;
- target names;
- fixed or tuned regularization mode;
- outer and inner fold counts;
- fixed settings and optional tuning grid;
- random seed;
- coefficient nonzero tolerance;
- output root;
- worker count for batch session parallelism; and
- optional trusted UTC bounds per manually aligned probe.

`RegionConfig` should contain:

- canonical display region, exactly `PFC` or `HPC`;
- metadata probe ID;
- channel selection mode;
- channel labels;
- `inside_brain` requirement;
- cluster groups; and
- optional explicit channel IDs when manual selection is intentionally used.

Avoid a dictionary of arbitrary settings. Typed fields make the scientific
choices discoverable and testable.

Defaults match revision 5: choice alignment, [-2, 2] s, 100 ms, 10 PCs per
region, fixed regularization, five outer folds, three inactive inner folds,
seed 0, and `1e-8` coefficient tolerance.

Configuration validation checks values and cross-field consistency without
opening large arrays. Input/path validation is a separate pipeline stage.

Paths in a JSON configuration may be relative; resolve them against the
configuration file's parent directory. Save their resolved absolute forms in
the run manifest. The reusable example therefore works when copied into a
session root and edited there instead of requiring machine-specific absolute
paths.

## 6. Data flow and array contracts

```
neural_session.json + explicit augmented table/config
        |
        +--> validated target table and target-specific masks
        |
        +--> explicit PFC/HPC probe and unit selection
        |
        +--> trusted alignment coverage intersection
        |
        +--> PFC and HPC rate tensors
             shape: (trial, time_bin, unit), units: Hz
        |
        +--> target/fold-local training transforms
             pooled unit z-score
             optional separate regional PCA
        |
        +--> one elastic-net decoder per target/time/region/representation/fold
        |
        +--> fold results, coefficients, failures, timings, provenance
        |
        +--> versioned NPZ + config + log + summary + light-mode PNGs
        |
        +--> read-only existing-webapp view
```

### 6.1 Trial table

`build_target_table(...)` returns a dataframe with one row per
original trial and:

- `trial_id` copied from `cur_trial`;
- `block_id` copied from `cur_block`;
- one normalized numeric column per target;
- target-valid boolean columns; and
- baseline validity/reason fields.

Shifted targets are built before baseline filtering. The function must not
modify its input dataframe.

### 6.2 Rate tensors

PFC and HPC tensors use:

- axis 0: the same ordered original trial rows;
- axis 1: common event-relative time-bin centers;
- axis 2: region-specific stable units; and
- values: float firing rates in Hz.

Do not build separate rate tensors for targets. Build each regional tensor once
per alignment/bin-width run, then index it with target masks.

The initial implementation may hold both regional tensors in memory after a
preflight estimate. If the estimate exceeds a documented safe fraction of
available RAM, fail before allocation and request a smaller configuration;
do not add disk-backed arrays or streaming in the first implementation.

### 6.3 Feature preprocessing

Within one target and outer fold:

- determine constant/unavailable units from the outer training observations;
- pool training trials and time bins for regional mean/scale;
- reuse that transform across all time-bin decoders in the fold;
- fit one PCA per selected region and fold when PCA is requested;
- reuse PFC/HPC transforms for standalone and combined results; and
- apply unchanged transforms to the outer test rows.

Tuned mode repeats these steps within each inner training split. Do not reuse
outer-training transforms inside inner validation.

### 6.4 Fits and scores

One fit record is identified by:

```
target
alignment
bin_width
time_bin
region_configuration
representation
outer_fold
regularization_mode
```

It records status/reason, train/test counts, class counts when applicable,
requested/effective feature counts, estimator parameters, convergence status,
metrics, intercept, and coefficients with stable feature identities.

Classification balanced accuracy and AUC come from one set of held-out
predictions. A metric toggle never creates another fit.

## 7. Split and model implementation details

### 7.1 Outer splits

Build one split assignment per target from its matched eligible rows:

- categorical: `StratifiedGroupKFold`;
- numerical: `GroupKFold`;
- groups: `cur_block`;
- deterministic, no shuffle; and
- validate all categorical train/test class sets after splitting.

Persist an array shaped `(n_targets, n_original_trials)` containing
outer-fold IDs and `-1` for ineligible rows. This makes matching and
reproducibility directly inspectable.

### 7.2 Fixed mode

For each target/fold:

1. fit training-only regional transform(s);
2. transform train/test rate tensors;
3. fit each time-bin decoder at the fixed settings;
4. catch convergence warnings as invalid fits;
5. compute held-out metric(s); and
6. retain coefficients and fold diagnostics.

### 7.3 Tuned mode

For each outer fold and time-bin/region/representation cell:

1. create three grouped inner folds from only the outer training rows;
2. evaluate the 15 accepted candidates on all inner folds;
3. invalidate a candidate if any required inner fit/metric is invalid;
4. choose highest mean inner balanced accuracy or $R^2$;
5. resolve exact ties by declared candidate order;
6. refit preprocessing and estimator on all outer training rows; and
7. evaluate once on the outer test rows.

Keep tuned mode mechanically explicit rather than hiding it inside a generic
search object: the pipeline must fit PCA inside each inner training split and
retain failure reasons and selected settings.

### 7.4 Complete-fold aggregation

Aggregate after all individual folds are stored. A summary cell is available
only if every requested fold is valid and finite. Unavailable cells preserve
surviving fold records but have no primary mean.

## 8. Saved-run layout and schema

Each execution creates an immutable session-local directory:

```
<session_root>/analysis_runs/
    task_variable_decoding_<YYYY-MM-DDTHH-MM-SSZ>/
        config.json
        input_manifest.json
        run_state.json
        resume_command.txt
        results.npz
        run.log
        summary.md
        run_session.py
        run_batch.py
        checkpoints/
        figures/
```

The script files are snapshots of the two launch modules used for the run.
Human-readable plots and report material remain in the run directory. No output
is written into the Git repository.

### 8.1 Identity and rerun behavior

A stable run fingerprint is computed from:

- analysis version;
- normalized scientific configuration;
- session ID;
- augmented-table and feature-parameter identities;
- resolved alignment/sorter/quality identities; and
- explicit region/unit-selection rules.

The default single-session runner searches for a completed matching
fingerprint and reports/skips it. `--rerun` creates a new timestamped
directory rather than overwriting. A changed version/input/configuration
always creates a different fingerprint.

Use small JSON manifests and ordinary NPZ files. No database, lock service, or
content-addressed object store is needed.

### 8.2 Launch, interruption, and resume state

`new` writes the run directory, immutable saved configuration,
input/code identity, initial `run_state.json`, and exact
`resume_command.txt` before loading large spike arrays. Run state uses
a small explicit lifecycle such as initialized, preflight complete, running,
interrupted, failed, and complete, with completed target names and the last
error/warning.

Update the state file atomically at package-defined boundaries and flush the
log before returning a failure/interruption code. A resumed run trusts only its
saved configuration/fingerprint and valid target checkpoints. It must not use
new command-line scientific overrides or infer the latest run directory.

The summary and live handoff record distinguish an interrupted resumable run
from a completed result. A missing final `results.npz` is never
presented as complete.

### 8.3 Checkpoint boundary

The restart boundary is one completed target. Write a target checkpoint only
after all requested region/representation/time/fold cells for that target have
finished or been recorded unavailable. Resume only checkpoints whose complete
fingerprint matches.

This limits lost work without adding per-fit transaction machinery. The final
`results.npz` is written only after combining target checkpoints.

### 8.4 NPZ arrays

The exact schema is frozen by round-trip tests before implementation. It should
use named arrays and include at least:

- target, region, representation, metric, time, fold, and feature labels;
- time-bin edges/centers in seconds;
- dense fold score arrays with NaN for non-applicable/invalid metrics;
- fit status and compact reason-code arrays;
- requested/effective feature counts;
- train/test and class counts;
- target eligibility masks and outer-fold assignment;
- direct-unit and PC coefficients grouped by region configuration;
- selected fixed/tuned parameters;
- stable unit/feature identities;
- stage timings; and
- a dictionary named `meta` containing version, seed, units, axes,
  paths, parameters, provenance, and warning text.

The loader validates the `meta` schema/version before exposing result
arrays. The NPZ is a trusted local analysis artifact, matching the project's
existing metadata convention.

If dense coefficient arrays would waste unreasonable space after real
preflight, use parallel long-form primitive arrays inside the same NPZ. Do not
switch to a new storage dependency.

## 9. Plotting and webapp plan

### 9.1 Offline figures

Generate from saved results:

1. categorical heatmap for balanced accuracy;
2. categorical heatmap for ROC AUC;
3. numerical $R^2$ heatmap; and
4. optional unit-coefficient figure(s) explicitly selected in the run config.

Plots use opaque white backgrounds, black text/axes, readable fonts, complete
labels, and captions. Unavailable cells use a mask/color distinct from chance
or zero.

The pipeline does not generate every combination as a PNG by default. Saved
arrays support interactive inspection; default report figures should remain a
small, declared set.

### 9.2 Integrated read-only view

The new existing-webapp view provides:

- saved-run selector, completion/fingerprint/config summary;
- region and representation selectors;
- alignment/bin-width information from the run, not recomputation controls;
- categorical target heatmap with metric toggle;
- numerical heatmap;
- fold score/count/failure table for a selected cell;
- coefficient plot/table for a selected direct-unit cell; and
- links or displayed paths to run summary and PNGs.

It does not expose a Compute button. If no compatible completed run exists, it
shows the expected session-local location and CLI invocation pattern.

## Implementation orchestration - Sol supervisor and Terra workers

These assignments organize future user-approved implementation. They do not
authorize source edits, tests, CT026 computation, or Slurm actions by
themselves.

The structure follows official OpenAI multi-agent guidance: delegate concrete,
bounded work with clear expected results; use parallel agents for independent
work; and avoid concurrent agents contending over shared mutable files.
Reference:
`https://developers.openai.com/api/docs/guides/responses-multi-agent`.

### Roles

**Lead Sol supervisor - `gpt-5.6-sol`, high reasoning**

- owns user communication, requirements, scientific interpretation, package
  order, authorization checks, and the live handoff;
- reads the relevant source/callers before each assignment;
- freezes the exact file allowlist, contracts, tests, and stop gate;
- independently reproduces RED and GREEN rather than accepting a worker
  summary;
- stages and commits only reviewed package files;
- owns shared integration files unless it explicitly assigns one writer;
- requests user decisions for scientific ambiguity or scope expansion; and
- is the only role that marks a package complete.

**Terra package worker - `gpt-5.6-terra`, high by default and xhigh
where assigned**

- receives one bounded tests-only or implementation-only task;
- edits only its explicit allowlist;
- writes tests first and stops after reporting genuine RED;
- resumes implementation only after Sol has verified and committed the tests;
- runs the focused commands named in the assignment;
- makes no commits, real-data runs, Slurm submissions, or nested agent
  assignments; and
- returns changed files, commands/results, unresolved risks, and confirmation
  that no out-of-scope action occurred.

**Independent Sol gate reviewer - `gpt-5.6-sol`, high or xhigh as
assigned**

- reviews a stable tests-only design or stable GREEN diff read-only;
- checks scientific drift, leakage, array axes/units, validity policy,
  checkpoint identity, failure handling, and missing tests;
- does not edit, commit, or broaden scope; and
- is mandatory for WP4, WP5, WP9, WP10 interpretation, and conditional WP11.

**Optional Terra scout - `gpt-5.6-terra`, medium, read-only**

- may inspect independent call sites, installed APIs, or failure logs;
- returns file/line evidence and uncertainties;
- does not edit or establish a gate; and
- is used only when the evidence-gathering task is genuinely independent.

Recheck model and effort availability at implementation start. Do not silently
substitute another model or effort. If the named configuration is unavailable,
stop and ask the user to revise the agent plan.

### Shared-worktree and concurrency rules

- Use at most one write-enabled Terra worker at a time.
- Sol may run independent read-only review/scouting concurrently, within the
  available agent limit, only while the inspected diff is stable.
- No two agents edit the same source, test, README, fixture, package
  initializer, webapp router, or plan document concurrently.
- Sol records HEAD and `git status --short` before and after every
  assignment.
- Unexpected or unowned changes stop the package immediately.
- The worker's allowlist must name exact files or a narrow responsibility
  group. Needing another file is a stop-and-replan condition.
- The Sol supervisor performs commits only while the Terra writer is idle.
- Subagents do not spawn subagents. The supervisor alone assigns, follows up,
  interrupts, or replaces workers.

### Mandatory package sequence

For each implementation package:

1. Sol re-reads the package's existing files, callers, tests, plan scope, and
   current worktree.
2. Sol sends a self-contained tests-only assignment to the Terra worker with
   model/effort, file allowlist, required tests, expected RED, forbidden
   actions, and return format.
3. Terra writes tests, runs the focused command, reports RED, and stops.
4. Sol inspects the test diff, reproduces RED, obtains the assigned independent
   test-design review when required, and commits tests only.
5. Sol follows up with the same Terra worker to authorize the bounded
   implementation.
6. Terra implements the smallest in-scope change, runs focused tests to GREEN,
   reports, and stops without committing.
7. Sol reviews the diff and runs focused plus affected regression suites. The
   independent Sol reviewer performs the assigned stable-diff gate.
8. Accepted findings return to the same Terra worker. Sol reruns verification,
   commits implementation separately, and updates the live handoff/package
   record.

Documentation-only updates are separate commits from tests and implementation.
The README/example package may characterize and test already-stable interfaces,
but must not silently redesign them.

### Worker assignment format

Every Terra prompt must include:

- package ID, task name, model/effort, and read-only/tests-only/implementation
  status;
- one objective and exact file allowlist;
- relevant revision-5 decisions and public/saved-data contracts;
- exact tests/commands and the expected stop gate;
- prohibited files, implementation, real-data, cluster, and commit actions;
- current HEAD/worktree facts needed for safe ownership; and
- return fields: files inspected/changed, summary, commands/results,
  expected-versus-actual RED/GREEN, risks/questions, and scope confirmation.

An interrupted worker report never counts as a gate. Sol records the last
independently verified HEAD, diff, command, and commit state, then re-audits
before resuming the same worker or assigning a replacement with the same model,
effort, allowlist, and stop gate.

### Package-specific Sol/Terra map

| Package | Sol supervisor responsibility | Terra worker assignment | Independent Sol gate |
| --- | --- | --- | --- |
| WP0 | Own specification/plan, reconcile user decisions, maintain live snapshot | None; documentation remains with supervisor | User review is the gate |
| WP1 | Freeze additive behavior contract and existing-output compatibility | High: tests-only then `rewards_in_block` implementation in bounded behavior files | Sol supervisor review |
| WP2 | Freeze config/target public contracts and positive-class mappings | High: tests-only then `config.py`/`targets.py` | High final review |
| WP3 | Freeze probe/region, unit-ID, coverage, shape, and Hz contracts | High: tests-only then `activity.py` using existing loaders/binning | High final review |
| WP4 | Own leakage/CV/model validity decisions and installed sklearn verification | Xhigh: tests-only then fixed modeling; separate follow-up for tuned modeling | Xhigh test-design and final numerical review |
| WP5 | Freeze NPZ/meta/fingerprint/checkpoint/CLI state machine | Xhigh: tests-only then results/pipeline/session CLI | Xhigh test-design and final restart/provenance review |
| WP6 | Freeze session-isolation, memory, dry-run, and worker-count policy | High: tests-only then batch CLI | High final review |
| WP7 | Own shared webapp router and saved-loader boundary | High: plotting/view tests then implementation; Sol or one assigned worker alone edits shared router | High final read-only/UI-boundary review |
| WP8 | Freeze exact commands and stable public file descriptions | High: README/example/`--help` documentation task only after interfaces stabilize | Sol runs every documented dry-run/help command |
| WP9 | Own end-to-end scientific assertions and full focused regression gate | Xhigh: synthetic integration tests and bounded fixes only | Xhigh scientific/leakage review |
| WP10 | Authorize exact CT026 read-only/preflight commands and interpret timing/memory | High command runner: execute only approved benchmark commands; no source edits | Xhigh independent benchmark interpretation |
| WP11 | Decide whether cluster support is needed; freeze wrapper resources/forwarding | High: wrapper tests/documentation, then thin shell implementation | Xhigh safety/resume/Slurm review; actual submission still user-gated |
| WP12 | Freeze exact real-session config and stop conditions; present output to user | High command runner only; no source edits or parameter changes | High evidence review plus user scientific inspection |

## 10. Tests to write before implementation

### 10.1 Behavioral feature tests

Add to `src/tests/behavior_analysis/test_session_analysis.py`:

1. `rewards_in_block` is recorded before the current trial update.
2. Counts reset at a block change.
3. Positive numeric rewards increment; zero rewards do not.
4. Numeric-string actions/rewards are handled consistently.
5. Manual rewards and no-choice rows neither increment nor reset.
6. A malformed reward on a valid animal choice raises a clear error.
7. `make_augmented_trial_df` preserves existing columns and adds the
   new column.

### 10.2 Config and input tests

Create `src/tests/neural_analysis/task_decoding/test_config.py` and
`test_targets.py`:

1. Defaults exactly match revision 5.
2. Region mappings require distinct configured probes and recognized regions.
3. Invalid alignment, bin size, folds, PC counts, bounds, and grids fail
   clearly.
4. Missing augmented columns are reported together.
5. Empty/duplicate `cur_trial` values fail.
6. Feature-parameter JSON must be an object.
7. HMM display labels describe signed belief, while source columns remain
   unchanged.

### 10.3 Target and eligibility tests

1. Previous/next targets are constructed before filtering.
2. Manual/no-choice adjacent rows are not bridged.
3. Switch/stay derivations match explicit action sequences.
4. Dark state 2 is excluded only from binary current-state eligibility.
5. Target-specific missingness does not leak into unrelated masks.
6. Baseline manual/no-choice/current-alignment exclusions are correct.
7. PFC/HPC/combined representations receive identical target trial rows.
8. Class labels and positive-class mappings are persisted.

### 10.4 Activity and unit-selection tests

Create `test_activity.py`:

1. Explicit PFC/HPC probe mapping is used instead of probe-name inference.
2. Channel selection combines channel-quality label, inside-brain status, and
   manual channel restriction correctly.
3. Cluster selection accepts `good`/`mua` and rejects
   noise.
4. Stable unit IDs include probe IDs and preserve axis order.
5. Spike and cluster length mismatch fails.
6. Standard alignment coverage uses finite `irig_utc_unix` bounds.
7. Manual alignment without explicit trusted bounds fails.
8. Full event window, not only the alignment timestamp, must be covered.
9. Both regional tensors share trial/time axes and retain Hz units.
10. Existing Pynapple and NumPy reference binning agree on a small fixture.

### 10.5 Split, preprocessing, and model tests

Create `test_modeling.py`:

1. Blocks never cross train/test folds.
2. Categorical folds contain both classes or return a declared unavailable
   result.
3. Numerical folds do not discretize targets.
4. Outer assignments are reused across time/region/representation.
5. Scaling statistics come only from training rows.
6. PCA directions and effective rank come only from training rows.
7. PFC + HPC PC features are concatenated separate regional projections.
8. Direct combined features preserve regional/stable-unit order.
9. Requested PC count is capped visibly and deterministically.
10. Fixed mode performs outer evaluation without inner fits.
11. Tuned mode never passes outer-test rows into inner splitting,
    preprocessing, or selection.
12. Candidate tie resolution follows declared order.
13. Balanced accuracy uses threshold 0.5 and AUC uses the same fit's scores.
14. $R^2$ uses native targets and returns unavailable for constant/singleton
    test targets.
15. Negative $R^2$ and below-chance categorical scores remain valid.
16. A convergence warning invalidates the fit; a converged all-zero solution
    remains valid.
17. Coefficient signs, scales, selection frequencies, and `1e-8`
    threshold are correct.
18. Primary means require all requested valid outer folds.
19. Results are deterministic for the configured seed.

Include a direct installed-API test for the supported scikit-learn 1.8
logistic configuration so implementation does not depend on the deprecated
`penalty` argument.

### 10.6 Result I/O and restart tests

Create `test_results.py` and `test_pipeline.py`:

1. NPZ round trip preserves arrays, labels, axes, units, and metadata.
2. The named `meta` dictionary preserves required provenance.
3. Schema/version mismatch fails clearly.
4. Fingerprints change with scientific input/config/version changes.
5. An identical completed run is skipped by default.
6. Explicit rerun creates a new path without overwrite.
7. `new` writes saved config, run state, and exact resume command
   before an injected large-array/long-running stage.
8. Matching target checkpoints resume; mismatched ones are ignored/rejected.
9. Interrupted/failed states return nonzero, preserve logs/checkpoints, and do
   not publish a complete result.
10. A failed target does not abort unrelated targets.
11. Log and summary contain required session, parameter, warning, and timing
   information.
12. Dry-run loads metadata/small tables and reports planned work without loading
    spike arrays or fitting models.

### 10.7 Plot and webapp tests

Create `test_plotting.py` and extend
`src/tests/neural_analysis/test_webapp_package.py`:

1. Heatmaps preserve negative and below-reference values.
2. Unavailable cells are visually/data-wise distinct.
3. Reference values, labels, captions, trial counts, and fold coverage appear.
4. Saved PNGs have opaque light backgrounds.
5. Coefficient plots distinguish zero, unavailable, and excluded features.
6. The new view has one canonical module owner.
7. Metadata availability keeps the result view selectable without raw-spike
   readiness and presents the no-saved-run state cleanly.
8. The app routes the decoding-results view before population controls and raw
   spike loading.
9. The view loads a saved fixture and never invokes computation.
10. Metric/selector changes only re-render saved data.

### 10.8 Documentation and command tests

Create `test_task_decoding_documentation.py`:

1. The reusable example configuration loads and validates without CT026 or
   another machine-specific absolute path.
2. Every public Python file in `task_decoding` is described in the
   in-package README.
3. The top-level neural README contains local dry-run, new-run, resume, batch,
   output-location, and webapp instructions.
4. `run_session --help` and `run_batch --help` exit
   successfully and show the documented modes.
5. The documented single-session dry-run works against a small temporary
   metadata/config fixture without loading spike arrays.
6. The exact resume command is written to run state before an injected
   long-running stage.
7. README result paths and CLI mode names match implementation constants rather
   than describing obsolete paths.

Documentation tests should validate stable commands and file ownership, not
word-for-word prose.

### 10.9 Synthetic integration tests

Create `test_synthetic_integration.py`:

1. A seeded two-region synthetic session completes end to end.
2. Injected time-local signal produces its strongest decoding in the expected
   interval without requiring an exact score.
3. PFC-only, HPC-only, and combined axes are correct.
4. Both target families and both representations complete.
5. A deliberately impossible grouped categorical target is unavailable while
   other targets finish.
6. Saved results reload and reproduce plotted summaries.
7. A held-out-only offset does not influence training scaling/PCA, serving as
   an end-to-end leakage regression.

### 10.10 Conditional cluster-wrapper tests

Write these only if WP11 is activated:

1. The wrapper forwards `new` and `resume` arguments to the
   same Python CLI without changing scientific settings.
2. Unknown modes and missing required paths return nonzero before launch.
3. Resource/log directives are explicit and match the documented benchmark
   decision.
4. Resume requires one exact run directory and never searches for a latest
   run.
5. A mocked submission/launcher failure is returned rather than hidden.
6. The top-level README includes local dry-run, `sbatch new`,
   `sbatch resume`, log location, environment, and tracked-commit
   requirements.
7. No test invokes a real scheduler or reads scientific arrays.

## 11. Work packages and gates

### WP0: Documentation approval

- Sol finalizes revision 5, this plan, the live snapshot, documentation
  deliverables, agent map, and interruption rules.
- No Terra worker, tests, source edits, or data runs.
- Gate: user accepts the documents and separately requests implementation.
- Handoff: record the documentation commit/HEAD and set WP1 as the single next
  package.

### WP1: General behavioral feature

- Terra high writes RED tests for `rewards_in_block` and stops.
- Sol verifies/commits tests, then the same Terra worker implements the additive
  augmented-table feature.
- Sol runs focused and affected behavior tests.
- Gate: existing augmented columns/output remain compatible.
- Handoff: record test/implementation commits and the exact normal command that
  will later regenerate an augmented table; do not run it on CT026 here.

### WP2: Configuration and targets

- Terra high writes RED config, validation, target, and eligibility tests.
- After Sol's tests-only commit, Terra implements `config.py` and
  `targets.py`.
- Gate: all target definitions and masks are inspectable without neural data.
- Handoff: freeze the config fields/example schema needed by later CLI and
  documentation work.

### WP3: Activity loading and coverage

- Terra high writes RED activity/unit/coverage tests.
- After Sol's tests-only commit, Terra implements explicit region loading,
  stable IDs, trusted bounds, matched trial windows, and regional tensors using
  existing loaders/binning.
- Gate: shapes, axes, Hz units, coverage, and unit identities are verified.
- Handoff: record representative synthetic tensor shapes and memory-estimate
  inputs; no CT026 arrays are loaded.

### WP4: Grouped modeling

- Terra xhigh writes RED split/preprocessing/model tests.
- An independent Sol xhigh reviewer audits leakage and validity coverage before
  the tests-only commit.
- Terra implements fixed mode first. Sol verifies it before a separate
  follow-up authorizes optional tuned mode.
- Gate: leakage tests, invalid-fold policy, coefficient contract, and
  deterministic results are green; independent Sol xhigh review finds no
  unresolved scientific/numerical issue.
- Handoff: record exact installed scikit-learn API/version and measured
  synthetic fit counts.

### WP5: Results and single-session pipeline

- Terra xhigh writes RED NPZ, fingerprint, checkpoint, dry-run, interruption,
  and exact-resume tests.
- Independent Sol xhigh review approves the saved schema/state-machine tests
  before the tests-only commit.
- Terra implements results, pipeline, run directory, logging, summary,
  `dry-run`/`new`/`resume`, and exact resume
  command persistence.
- Gate: synthetic small target runs resume and round-trip.
- Handoff: freeze CLI modes, output paths, schema version, and resume behavior
  before batch, webapp, or README work.

### WP6: Batch runner

- Terra high writes RED dry-run, skip/rerun, and session-isolation tests.
- After Sol's tests-only commit, Terra implements the session-list runner with
  parallelism across sessions only.
- Default worker count is available CPU count, reduced explicitly when the
  rate-tensor memory estimate predicts unsafe aggregate use.
- Gate: no within-session parallelism and no shared mutable session state.
- Handoff: record batch input format and exact documented commands.

### WP7: Plotting and integrated webapp view

- Terra high writes RED plotting/routing/read-only tests.
- Sol freezes shared `session_inputs.py`/`app.py` ownership
  and commits tests.
- Terra implements light-mode plotting and the saved-result view; only Sol or
  that one assigned worker edits shared webapp routing files.
- Gate: selecting the view never loads raw spikes or calls fitting code.
- Handoff: freeze user-visible view names, selectors, and no-result messages.

### WP8: User and maintainer documentation

- Terra high updates `src/neural_analysis/README.md`, creates
  `src/neural_analysis/task_decoding/README.md`, and creates the
  portable example configuration after WP1-WP7 interfaces are stable.
- Terra adds only the stable documentation/command tests listed in Section
  10.8; documentation changes are committed separately.
- Sol runs every documented `--help` and dry-run command against a
  temporary fixture and checks every Python file is described.
- Gate: a scientist can configure, dry-run, start, resume, locate outputs, and
  open results without reading implementation source; a maintainer can identify
  every file's role from the package README.
- Handoff: record exact commands and note whether cluster documentation remains
  intentionally absent pending WP10.

### WP9: Synthetic full pipeline

- Terra xhigh writes RED end-to-end synthetic tests and, after the tests-only
  gate, makes only bounded integration fixes.
- Sol runs all focused task-decoding tests and affected neural/behavior
  regressions.
- Independent Sol xhigh review audits scientific signal timing, leakage,
  failures, saved outputs, and documentation consistency.
- Gate: deterministic saved output and plots from both target families,
  representations, and regions.
- Handoff: record test counts/runtime, output fixture identity, and the exact
  proposed CT026 preflight commands without executing them.

### WP10: CT026 2026-08-03 preflight and benchmark

- This package requires separate explicit user approval for the exact CT026
  commands.
- Terra high acts as a command runner only: read-only validation, dry-run, and
  the approved bounded representative fixed-mode benchmark. It makes no source
  edits or parameter changes.
- Sol confirms selected channels/units, eligible trials, block/class coverage,
  tensor memory, exact model-fit count, stage timings, and peak memory.
- An independent Sol xhigh reviewer checks raw evidence and the local/full/tuned
  runtime projection.
- Gate: user reviews the benchmark and authorizes any longer single-session
  scientific run.
- Handoff: update the live snapshot with exact run directory, commands, logs,
  results, resource measurements, and recommendation.

### WP11: Conditional cluster wrapper

- Skip this package entirely if WP10 supports practical local use.
- If WP10 shows cluster use is warranted, stop for user approval of the wrapper
  and later separate approval of any submission.
- Terra high writes mocked forwarding/failure/resume tests, then implements the
  thin Slurm wrapper and cluster README section after the tests-only gate.
- Independent Sol xhigh review checks exact-run resume, resource visibility,
  environment/commit identity, no automatic submission, and no duplicated
  scientific settings.
- Gate: local dry-run plus mocked wrapper tests pass and documentation contains
  exact `sbatch new`/`resume` commands. No real job is part
  of this gate.
- Handoff: record wrapper commit and proposed resource request; actual
  submission remains unperformed until explicitly approved.

### WP12: One-session scientific inspection

- This package requires explicit approval of the exact fixed-mode CT026
  configuration and local or Slurm command.
- Terra high acts as command runner only; Sol monitors only as requested and
  does not alter parameters mid-run.
- User inspects saved heatmaps, fold coverage, warnings, and coefficients.
- Only after user approval should a batch of additional sessions be considered.
- Handoff: record immutable run paths, configuration, commit, commands, timing,
  warnings/errors, and user acceptance. Do not treat completion as permission
  for other sessions.

## 12. Performance plan

### 12.1 Reuse before optimization

The first implementation should:

- load each probe once;
- build one regional tensor per alignment/bin-width run;
- build one outer split assignment per target;
- fit one regional fold transform and reuse it across time bins;
- reuse separate PFC/HPC PCA projections in the combined analysis;
- compute balanced accuracy and AUC from one classifier fit; and
- checkpoint at target boundaries.

Do not add Numba, Cython, multiprocessing within a session, memory mapping, or
GPU code without a measured bottleneck and user approval.

### 12.2 Work-count warning

The default target set has 8 categorical and 10 numerical targets. At 100 ms
over four seconds there are 40 time bins.

Fixed mode requests:

```
18 targets x 40 bins x 3 region configurations
x 2 representations x 5 outer folds = 21,600 outer fits
```

Tuned mode evaluates 15 candidates in 3 inner folds and then refits once for
each outer fold:

```
21,600 x (15 x 3 + 1) = 993,600 estimator fits
```

This is the principal runtime risk. Tuned mode must remain optional and should
not be presented as an ordinary interactive choice. Before any full tuned run,
measure a representative subset and report the projected duration.

### 12.3 Benchmark stages

Record with `perf_counter` and a documented memory measurement:

1. configuration/metadata/table validation;
2. sorter/alignment loading and unit selection;
3. PFC and HPC rate-tensor construction;
4. target/split construction;
5. fold preprocessing/PCA;
6. estimator fitting/scoring by target family and representation;
7. result serialization;
8. plotting/report generation; and
9. total wall time and peak resident memory.

Report number of trials, blocks, units, time bins, requested/valid folds, model
fits, invalid cells, and output size so timing is interpretable.

### 12.4 Benchmark tiers

1. **Synthetic microbenchmark:** catches pathological overhead and produces a
   stable regression fixture; it is not a production-time estimate.
2. **CT026 bounded benchmark:** choice alignment, 100 ms, fixed mode, one
   representative categorical target and one numerical target, both
   representations and all regions. Include exact loading/binning once.
3. **Projection:** extrapolate model time using observed fit counts and separate
   fixed costs. Clearly label it as an estimate.
4. **Full default benchmark:** run only if the projection is acceptable and the
   user authorizes it. Compare measured vs projected time.
5. **Tuned estimate:** representative subset only unless separately authorized.
6. **Local/cluster decision:** Sol reports whether ordinary local use is
   practical. The user decides whether to skip WP11 or authorize the thin
   wrapper. There is no automatic duration threshold or cluster submission.

## 13. CT026 2026-08-03 validation fixture

Planned session root:

```
/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260803_latent_inference
```

Planned inputs:

- metadata: `neural_session.json`;
- augmented trials:
  `processed/CT026_2026-08-03_111938_augmented_trials.csv`;
- feature parameters: `processed/trial_feature_params.json`;
- PFC: `ProbeA`;
- HPC: `ProbeB`.

Read-only audit observations:

- 650 augmented rows;
- 646 valid animal choices;
- 83 behavioral blocks;
- both requested alignment columns are present;
- both probes have 3,780 finite IRIG values with common bounds
  1785770350.0 to 1785774129.0 UTC seconds;
- ProbeA has 132 good, 72 MUA, and 49 noise clusters before channel filters;
- ProbeB has 229 good, 81 MUA, and 119 noise clusters before channel filters;
- the augmented table contains all revision-5 source columns except the new
  `rewards_in_block` column; and
- metadata currently points to the raw trial CSV, which is why the augmented
  path is explicit in the decoding configuration.

After WP1 is approved and implemented, create the missing column by rerunning
the normal behavioral augmentation/feature-save path. Do not patch the
experimental CSV ad hoc.

These counts are preflight facts, not authorization to mutate the table or run
decoding.

## 14. Dependencies and API verification

Use existing project dependencies only:

- Python standard library;
- NumPy;
- pandas;
- scikit-learn;
- Pynapple;
- Matplotlib;
- Streamlit; and
- pytest.

All local Python commands use `uv run`. Before implementation, verify
the exact installed APIs from package source or official documentation. The
audit found scikit-learn 1.8.0 with:

- `ElasticNet(alpha=..., l1_ratio=...)`;
- `LogisticRegression(solver="saga", C=..., l1_ratio=...)`;
- deterministic `GroupKFold(..., shuffle=False)`; and
- deterministic `StratifiedGroupKFold(..., shuffle=False)`.

Pin behavior in tests rather than relying on memory of older scikit-learn
interfaces.

## 15. Risks and deliberate trade-offs

| Risk | Planned response |
| --- | --- |
| Full fixed run has many fits | Benchmark first, reuse fold transforms, checkpoint targets, report work count. |
| Tuned run is roughly 46 times the fixed estimator-fit count | Keep optional; benchmark/project before authorization. |
| Grouped folds may lack class coverage | Validate actual splits and mark unavailable; never fall back silently. |
| PCA count exceeds fold rank | Cap visibly and persist requested/effective counts. |
| Manual alignment lacks trustworthy coverage | Require explicit bounds and fail early. |
| Constant units differ by fold | Mark fold-level unavailable features separately from elastic-net zeros. |
| Metadata points to raw trials | Require explicit augmented and parameter paths in config. |
| Results UI could accidentally recompute | Route early and test that the view imports/uses result loading only. |
| Result schema becomes elaborate | Use one documented NPZ plus small JSON manifests; avoid databases/new dependencies. |
| Dense coefficients become large | Measure first; use primitive long-form NPZ arrays only if justified. |
| Offline use is difficult or tribal knowledge | Provide a portable config, three-mode session CLI, exact resume command, top-level quickstart, and package file map. |
| Long execution is interrupted | Create run state/resume command before large work and checkpoint at target boundaries. |
| Cluster support duplicates science or becomes another launcher | Add only a conditional thin wrapper around the same CLI after benchmark/user approval. |
| Agent handoff loses authorization or TDD state | Maintain the live snapshot/package records and require the Sol resume checklist. |
| Parallel agents contend over shared files | One Terra writer at a time; parallelism is read-only or on frozen disjoint files. |

## 16. Approval checklist

Before implementation, confirm:

- revision 5 is the scientific authority;
- `rewards_in_block` is added to general augmented trials;
- current-state target excludes dark state 2;
- PFC/HPC are explicitly mapped to probes in config;
- manual alignment without explicit trusted bounds fails;
- outputs are offline, immutable timestamped runs;
- the session CLI uses documented `dry-run`, `new`, and
  exact-directory `resume` modes;
- the only UI is an integrated read-only existing-webapp view;
- top-level user documentation, an in-package file map, and a portable example
  config are required implementation deliverables;
- the plan remains the live handoff and is updated at every package gate;
- one Sol supervisor owns gates/commits and bounded Terra workers follow the
  tests-only then implementation sequence;
- fixed mode is the first/default production path;
- tuned mode remains optional and benchmark-gated;
- cluster wrapping is conditional on WP10 evidence and separately authorized;
- initial plots are light mode; and
- CT026 2026-08-03 is the first real-session benchmark, not an automatically
  authorized full analysis.

Approval of this document should be followed by a separate implementation
request. Until then, no code, tests, augmented data, or neural results should be
changed.
