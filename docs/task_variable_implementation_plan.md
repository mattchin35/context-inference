# Task-Variable Decoding Implementation Plan

**Status:** Ready for user approval; planning only. This document does not
authorize implementation, test creation, data mutation, or decoding runs.
Implementation may begin only after the user separately approves the plan and
requests code changes.

**Scientific contract:** `docs/task_variable_spec_v5.md`.

## Live handoff snapshot

**Snapshot date:** 2026-10-06

**Current phase:** WP0 readiness review complete; awaiting user approval. The
revision-5 specification and this implementation plan exist, but the user has
authorized documentation work only. No production code, tests, experimental
data, decoding output, benchmark, local long run, or cluster action is
authorized.

**Repository state at this snapshot:**

- branch: `refactor`;
- HEAD and `origin/refactor`: `740aceac8b63d01e9529bb7c8842b8afe29221db`
  (`updated specs`);
- this plan contains the uncommitted WP0 documentation revision awaiting user
  approval/commit;
- concurrently modified neural-regression documents are owned by another user
  task and were not edited or incorporated into this readiness pass; and
- all other pre-existing dirty/untracked files remain outside this plan's
  ownership.

**Completed planning evidence:**

- `src/neural_analysis`, `docs/SoftwareDesign.md`, and revision 4 were audited;
- revision 5 records the resolved scientific/data contracts;
- CT026 2026-08-03 input presence and small-table/alignment metadata were
  inspected read-only;
- the implementation architecture, tests-first sequence, saved-run contract,
  integrated webapp view, and benchmark strategy are drafted below; and
- the offline contract now includes unattended local launch, one-shot
  read-only status inspection, and an unattended single-session Slurm path. It
  explicitly does not require active Codex monitoring or polling; and
- cluster planning uses `src/shell_scripts/hpc_ppc.sh` as the site
  execution reference, manual non-destructive `rsync` for data/results, and
  benchmark-derived task-decoding resources rather than copied PPC values; and
- a second correctness pass resolved tuned-transform reuse, error boundaries,
  portable path/source identity, comparable benchmark threading, and
  measurement-based memory/resource decisions.

**Next exact action:** the user reviews this ready plan. Before a later
implementation begins, the accepted WP0 documents must be committed and their
HEAD recorded; implementation then requires a separate explicit user request.
At that point the Sol supervisor starts WP1 with a fresh worktree/HEAD audit;
it must not infer implementation authority from the existence or staging of
these documents.

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
| WP0 documentation approval | Ready for user review; documentation only | User approval, documentation commit/HEAD, and explicit implementation request |
| WP1 behavioral feature | Not authorized | Sol freezes scope and assigns tests-only Terra task |
| WP2 configuration and targets | Not authorized | WP1 GREEN and recorded handoff |
| WP3 activity loading and coverage | Not authorized | WP2 GREEN and recorded handoff |
| WP4 grouped modeling | Not authorized | WP3 GREEN and independent numerical test-design review |
| WP5 results and session pipeline | Not authorized | WP4 GREEN and saved-schema freeze |
| WP6 batch runner | Not authorized | WP5 GREEN |
| WP7 plotting and webapp | Not authorized | WP5 saved loader stable; WP4 metrics stable |
| WP8 documentation and examples | Not authorized | CLI/webapp interfaces stable through WP7 |
| WP9 synthetic integration | Not authorized | WP1-WP8 focused gates GREEN |
| WP9A CT026 augmented-table preparation | Not authorized | WP9 GREEN plus explicit approval of the exact behavior-processing command |
| WP10 CT026 preflight/benchmark | Not authorized | WP9A validation passes plus explicit real-session benchmark approval |
| WP11 single-session cluster path | Not authorized | WP10 evidence and user choice justify cluster convenience/cost; user approves wrapper/transfer work |
| WP12 CT026 scientific inspection | Not authorized | User accepts benchmark and authorizes exact run |
| WP13 cluster batch-array follow-up | Deferred end-job and not authorized | WP12 accepted; resource profile finalized; user approves batch work |

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

#### 2026-10-06 - WP0 readiness review

- State: implementation plan ready for user review; WP0 remains open until the
  accepted documentation is committed and the user separately requests
  implementation.
- Authorization: documentation changes only.
- Sol / Terra / reviewer: primary Codex review; no Terra worker or independent
  implementation reviewer was used.
- Start HEAD / end HEAD: `740aceac8b63d01e9529bb7c8842b8afe29221db` /
  `740aceac8b63d01e9529bb7c8842b8afe29221db`.
- Owned files: `docs/task_variable_implementation_plan.md` only. Concurrent
  neural-regression documentation changes belong to another user task and were
  excluded.
- RED command and result: not applicable; no tests or implementation were
  authorized.
- GREEN/regression commands and results: documentation-only checks passed:
  `git diff --check -- docs/task_variable_implementation_plan.md`; Markdown
  fence count was even; revision-5 specification remained unchanged.
- Commits: none; the WP0 plan revision is an uncommitted working-tree change.
- Real-data or external actions: none.
- Findings and unresolved risks: the readiness findings were resolved by
  freezing CT026 augmented-table preparation, neural tensor row mapping,
  scientific/execution and file/code identity, shared prepared-run execution,
  single-writer resume behavior, atomic publication, inner grouped CV tests,
  memory/resource thresholds, and non-destructive transfer scope. No known
  implementation-contract blocker remains.
- Exact next action: user reviews the plan; if accepted, commit the WP0
  documentation and record its HEAD before any separately authorized WP1 work.

#### 2026-10-06 - WP0 second readiness correction

- State: the specification and plan were corrected after a second
  correctness/completeness/conciseness audit; WP0 remains documentation only.
- Authorization: documentation changes only.
- Sol / Terra / reviewer: primary Codex review; no worker was used.
- Start HEAD / end HEAD: `740aceac8b63d01e9529bb7c8842b8afe29221db` /
  `740aceac8b63d01e9529bb7c8842b8afe29221db`.
- Owned files: `docs/task_variable_spec_v5.md` and this plan only.
- RED command and result: not applicable; no implementation was authorized.
- GREEN/regression commands and results: `git diff --check --
  docs/task_variable_spec_v5.md docs/task_variable_implementation_plan.md`
  passed; both Markdown fence counts were even; stale superseded terms were
  searched explicitly.
- Commits: none.
- Real-data or external actions: none.
- Findings and unresolved risks: corrected per-fold transform reuse in tuned
  CV, declared-versus-unexpected failure handling, single-thread benchmark
  identity, session-root path containment, scoped scientific-source identity,
  fixed estimator defaults, and measured rather than guessed resource gates.
- Exact next action: user reviews the corrected documents, then separately
  approves implementation if satisfied.

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
pipeline are green. After the separately approved WP9A behavior-table
preparation, CT026 neural work begins with a read-only preflight and bounded
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
- Preserve row count, row order, index, and every existing column; append one
  public column.

The helper should iterate once in chronological row order. A dataframe groupby
expression is not preferred if it obscures the entering-trial update order or
manual/no-choice behavior.

### 4.2 New analysis package

Create `src/neural_analysis/task_decoding/` with:

| Module | Responsibility | Principal public interface |
| --- | --- | --- |
| `__init__.py` | Mark the focused package boundary without broad re-exports. | Package marker only |
| `config.py` | Frozen user settings, analysis version, region definitions, defaults, scientific/execution separation, JSON serialization, and lightweight value validation. | `ANALYSIS_VERSION`, `TaskDecodingConfig`, `RegionConfig`, `load_task_decoding_config(...)`, `scientific_config_payload(...)` |
| `targets.py` | Augmented-table validation, chronological shifted targets, source mappings, and target-specific eligibility. | `build_target_table(...)`, `validate_augmented_trials(...)` |
| `activity.py` | Probe loading, channel/unit selection, trusted coverage, matched trial windows, rate tensors, stable feature identities. | `load_region_activity(...)`, `build_session_rate_tensors(...)` |
| `modeling.py` | Grouped splits, fold-local standardization/PCA, fixed/tuned elastic-net fits, metrics, coefficients, and validity. | `make_outer_splits(...)`, `decode_target(...)` |
| `results.py` | Input and scientific-source manifests, in-memory result record, NPZ/config/log save-load contract, completion state, and compatible-run discovery. | `build_input_manifest(...)`, `scientific_source_fingerprint(...)`, `save_task_decoding_run(...)`, `load_task_decoding_run(...)` |
| `plotting.py` | Light-mode heatmaps, coefficient summaries, captions, and PNG export from saved results. | `plot_decoding_heatmap(...)`, `plot_unit_coefficients(...)` |
| `pipeline.py` | Assemble preparation, loading, targets, activity, decoding, checkpoints, logging, and reporting for one session. | `plan_task_decoding_session(...)`, `prepare_task_decoding_run(...)`, `run_prepared_task_decoding(...)` |
| `run_session.py` | Standard-library-only single-session entrypoint that sets numerical thread limits before lazy pipeline imports; provides `dry-run`, `new`, `resume`, and read-only `status`, with optional unattended detachment. | `main(...)` |
| `run_batch.py` | Standard-library-only session-list entrypoint with the same early thread limits and lazy imports; provides `dry-run` and `new` with session-level parallelism only. | `main(...)` |
| `README.md` | File-by-file ownership, dependency direction, public entry points, array/result contracts, and developer extension notes. | Documentation only |

No inheritance or Protocol is needed initially: there is one loader path and
one model family per target family. Add an abstraction only if a second real
implementation creates a concrete need.

Every initial run computes all six region/representation combinations: PFC,
HPC, and PFC + HPC crossed with PCA and direct units. The webapp selectors only
choose which saved result to display; they do not control computation.

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
- how the normal behavior-processing path creates the augmented table and how
  to regenerate it when lightweight validation reports a missing required
  column such as `rewards_in_block`, without hand-editing the CSV;
- how to copy and edit the example configuration;
- the exact local `dry-run`, foreground and detached `new`, detached
  `resume`, and read-only `status` commands;
- how to start an unattended run, close the terminal/Codex task, and inspect
  its durable state and logs once later without polling;
- how to perform a batch dry run and batch launch;
- where run directories, checkpoints, logs, NPZ results, summaries, and PNGs
  are written;
- how to open the saved results in the existing webapp;
- the difference between fixed and tuned mode, including the large tuned-mode
  work-count warning;
- the bounded two-target benchmark recipe, recorded timing/memory fields, and
  how to use its projection before choosing a full local or Slurm run;
- how matching completed runs are skipped and how an explicit rerun creates a
  new immutable directory;
- the single-session Slurm submit/resume/status path, exact-commit/offline
  environment gates, safe input/result rsync, and finalized resource block
  when WP11 is activated; and
- the array batch command only after the separately gated WP13 follow-up.

Create `src/neural_analysis/task_decoding/README.md` with:

- one concise paragraph describing the package and its dependency direction;
- a table describing every Python file and its public entry points;
- input dataframe columns and rate/result array shapes, axes, and units;
- configuration and result-schema summaries;
- which existing modules are deliberately reused;
- how target eligibility, fold validity, leakage prevention, and checkpoints
  work;
- how the common neural-eligible tensor rows map back to the complete target
  table and why invalid alignment rows are not zero-filled;
- how portable relative paths, cluster execution paths, termination, returned
  results, single-writer guards, atomic publication, and resource provenance
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

The single-session CLI is the primary offline interface. A normal local
workflow is shown below. `dry-run` may be used during development, but a
persistent `new` or `resume` requires the scoped scientific source files in
Section 8.1 to be tracked and clean.

```bash
cp docs/examples/neural_analysis/task_decoding_config.json \
  /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_session dry-run \
  --config /path/to/session/task_decoding_config.json

uv run python -m src.neural_analysis.task_decoding.run_session new \
  --config /path/to/session/task_decoding_config.json --detach

uv run python -m src.neural_analysis.task_decoding.run_session status \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp>
```

Omit `--detach` when an intentionally small run should remain in the
foreground. A detached `new` launch must do only bounded setup in the calling
process: validate small inputs, create the immutable run directory, write the
saved configuration/manifests/state and exact follow-up commands, start one
detached child with its console streams redirected into the run directory, and
return. It prints the run directory, local PID, log paths, and exact status and
resume commands. It does not poll the child.

The matching unattended resume command is:

```bash
uv run python -m src.neural_analysis.task_decoding.run_session resume \
  --run-directory /path/to/session/analysis_runs/task_variable_decoding_<timestamp> \
  --detach
```

The entry modules set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1` before lazily importing NumPy/scikit-learn or the
pipeline. They record these values in execution provenance. The Slurm wrapper
sets the same values explicitly, so the initial local benchmark and one-CPU
cluster projection have the same threading identity.

The detached child uses the same foreground pipeline, saved configuration,
and target checkpoints as an ordinary invocation. Use the Python standard
library process launcher with a new process session and explicit file handles;
do not require `nohup`, a terminal multiplexer, a notebook, Streamlit, or a
resident Codex task.

Preparation and execution have one internal contract shared by all launch
paths:

- `prepare_task_decoding_run(config_path, rerun, execution_mode) -> Path`
  performs bounded validation, creates the immutable run directory, and saves
  the configuration, manifests, initial state, and exact follow-up commands;
- `run_prepared_task_decoding(run_directory) -> None` loads only that saved
  configuration, claims the run's single-writer guard, and executes or resumes
  the foreground scientific pipeline; and
- local foreground `new`, the detached child, and a Slurm compute job all call
  these same functions rather than reconstructing scientific settings.

`run_session.py` may expose private `_prepare` and `_execute-prepared`
subcommands for the shell wrapper and detached child. They are tested internal
interfaces, not additional scientist-facing workflows and are not advertised
in the user quickstart.

`status` is a one-shot, read-only inspection. It reports the saved lifecycle,
current stage, completed/total targets, start/update/end times, last warning or
error, result completeness, local PID or Slurm job identity when present, and
the tail location of durable logs. It neither loads spike arrays nor fits,
resumes, kills, or continuously watches anything. PID liveness is advisory:
after an abrupt process or machine failure, saved checkpoints and files are
authoritative, and the user must explicitly run the exact resume command.
Default status reads only small state/manifest files. An explicit
`--verify-results` option may additionally validate the saved NPZ schema
for a completed or returned run; it still never accesses source spikes or
recomputes results.

This is also the Codex operating contract. Codex may start an authorized
detached run, report the path and commands, and end its task. The user can ask
Codex later for one fresh status/log inspection. No recurring wait, background
agent, or token-consuming polling is part of the computation plan.

The separate batch interface remains:

```bash
uv run python -m src.neural_analysis.task_decoding.run_batch dry-run \
  --config-list /path/to/task_decoding_configs.txt

uv run python -m src.neural_analysis.task_decoding.run_batch new \
  --config-list /path/to/task_decoding_configs.txt --workers 4
```

`dry-run` validates metadata, configuration, selected-target columns, feature
parameters, output paths, small `cluster_info.tsv`/channel metadata, and
planned work count without calling `load_sorter_metadata`, loading
`spike_clusters.npy` or aligned spike arrays, or fitting models. `new` creates
the run directory and resume
record before large-array loading. `resume` accepts only the exact run
directory and saved configuration; it does not reconstruct settings from the
current command line.

The batch config list is a UTF-8 text file with one configuration path per
nonblank, non-comment line. Batch resume is deliberately per-session through
the printed single-session resume commands; do not add a second batch
checkpoint format. Do not add detached batch supervision during the first
implementation: prove the simpler single-session unattended path on CT026
before deciding whether batch detachment is actually needed.

For a foreground run, the CLI returns nonzero on invalid configuration or
failed scientific execution. A detached `new` or `resume` exit code reports
only whether preparation and child launch succeeded; later scientific success
or failure is read from `status`, durable state, and logs. Likewise, a Slurm
submission exit code reports only preparation and scheduler acceptance. Every
mode prints a short actionable error plus the run directory/resume command when
one exists. Do not require a notebook or Streamlit to start offline
computation.

### 4.6 Routine single-session cluster path

WP10 first measures the default fixed-mode workload locally. Cluster use does
not require proof that local execution is impossible: the user may choose WP11
because the projected run is expensive, occupies the workstation, or is more
convenient to leave on the cluster. There is no automatic duration threshold.

Add one thin, self-submitting Slurm wrapper:

`src/shell_scripts/task_variable_decoding_slurm.sh`

Run these commands on the cluster login node from the tracked-clean repository
root, after the session/configuration has been transferred:

```bash
bash src/shell_scripts/task_variable_decoding_slurm.sh submit-new \
  --config /cluster/session/task_decoding_config.json

bash src/shell_scripts/task_variable_decoding_slurm.sh submit-resume \
  --run-directory /cluster/session/analysis_runs/task_variable_decoding_<timestamp>

bash src/shell_scripts/task_variable_decoding_slurm.sh status \
  --run-directory /cluster/session/analysis_runs/task_variable_decoding_<timestamp>
```

`src/shell_scripts/hpc_ppc.sh` is the site-specific reference, not a
script to modify or call. Reuse its reviewed operational pattern:

- partition `unlimited`, one task, a descriptive job name, five-minute
  `TERM` notice, the existing private Slurm log directory, and existing mail
  settings;
- resolution of `SLURM_SUBMIT_DIR` to the exact Git root, rejection of
  tracked-dirty or wrong-root checkouts, and logging of the actual commit;
- `uv run --frozen --no-sync --offline` so compute jobs never install,
  upgrade, or download dependencies;
- explicit `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
  `OPENBLAS_NUM_THREADS` limits; and
- `exec` of the Python process so scheduler signals and exit codes propagate.

Do not copy PPC's eight CPUs, 32 GB, or 72-hour request without evidence.
Those values serve only as a known high-resource reference. Task decoding has
no initial within-session process pool, so the expected starting CPU request
is one; request more only after a measured and separately approved parallel or
threaded path exists. After WP10, Sol proposes the exact memory and wall-time
values, the user approves them, and the tests freeze all top-of-script Slurm
fields before wrapper implementation.

The self-submission layer contains no scientific defaults or duplicate
configuration parsing. `submit-new` runs the lightweight cluster dry run,
calls the runner's private `_prepare` interface to create one immutable cluster
run directory, submits that exact directory to `_execute-prepared`, and
returns. `submit-resume` submits the same `_execute-prepared` interface for one
existing run directory. Both private modes call the public Python preparation
and execution functions above. Preparation is an implementation detail rather
than a fifth scientist-facing Python workflow.

On successful submission, write `slurm_submission.json` with the job ID,
submission time, requested partition/tasks/CPUs/memory/time, code commit,
scheduler log path, and exact status/resume commands. The wrapper prints the
same receipt and returns immediately. It never selects a latest run, submits a
replacement job, or retries automatically.

Cluster `status` reads small run files and performs at most one
`sacct` query. Preserve at least `State`, `Elapsed`,
`TotalCPU`, `AllocCPUS`, `MaxRSS`, `ReqMem`,
`Timelimit`, and `ExitCode` when available. It presents scheduler
and pipeline state side by side rather than guessing which is authoritative or
polling for a transition.

The Python pipeline handles `SIGTERM`/`SIGINT` by stopping at the
safest available boundary, flushing logs and state, and preserving completed
target checkpoints. A hard timeout or out-of-memory kill may prevent that
handler from running; later status must expose that discrepancy without
declaring completion. Resume remains an explicit separately authorized
submission against the exact cluster run directory.

If benchmarking identifies a need for within-session parallel fitting, stop
and revise this plan. A scheduler provides unattended wall time and memory; it
does not itself justify a second numerical implementation. No `sbatch`
command is run without explicit approval. Once submitted, neither the user nor
Codex remains connected or polls the job.

### 4.7 Manual rsync and portable run contract

The workstation session is the authoritative long-term copy. The cluster holds
an execution copy with the same directory structure below the session root;
absolute workstation and cluster prefixes may differ. Configuration paths are
therefore saved in portable session-relative form. Resolved absolute paths are
execution provenance only and must not enter the scientific fingerprint or be
required to view returned saved results.

Input transfer remains an explicit, simple operation documented with concrete
site paths, following this pattern:

```bash
rsync -a --info=progress2 --exclude='/analysis_runs/' /local/session/ \
  user@cluster:/cluster/session/
```

The workstation copy is authoritative, so updating its corresponding cluster
input files is intentional. Excluding `analysis_runs/` is mandatory: input
synchronization must never overwrite cluster run state, checkpoints, logs, or
results. First run the same command with `-n` added when the destination has
not been inspected recently. Do not use broad `--delete`, and do not submit
while local preprocessing files are changing. After transfer, the cluster `dry-run` revalidates
the configuration, required augmented columns, source identities, probe files,
output path, and planned work count before `sbatch`. The Git repository is
handled separately: the cluster uses a tracked-clean checkout of the exact
pushed commit rather than an rsynced dirty source tree, and its uv environment
is prepared explicitly on the login node before offline jobs run.

Only a terminally complete run directory is returned. Transfer it first to a
hidden, uniquely named directory on the same local filesystem:

```bash
rsync -a --info=progress2 \
  user@cluster:/cluster/session/analysis_runs/<run_id>/ \
  /local/session/analysis_runs/.incoming-<run_id>/
```

Run the ordinary read-only `status`/result loader against the incoming
copy using `status --verify-results`. It must validate the complete
state and loadable result schema without requiring cluster source paths or
spike arrays. Only then rename the incoming directory atomically to its final
`<run_id>` name; the final target must not already exist. Interrupted
transfers remain hidden and can be safely repeated; they are never presented
by the webapp as completed runs. Input synchronization and result return are
separate commands, and returning results never overwrites preprocessing inputs
or an existing immutable run.

The top-level neural README owns the exact site commands and the distinction
between local and cluster paths. The in-package README owns portability,
fingerprint, termination, and resume semantics. No automatic SSH, rsync,
remote deployment, or general workflow manager belongs in the first version.

### 4.8 Deferred cluster batch-array follow-up

Cluster batch submission is WP13, deliberately the last implementation job.
It starts only after the single-session CT026 result is accepted and the
single-session Slurm resource profile is finalized from measured cluster
usage. It is not part of WP11 or the first scientific cluster run.

The intended extension is mechanically narrow:

- one Slurm array element per session configuration, using the same
  single-session wrapper and Python runner;
- one immutable mapping/receipt from array index to configuration, run
  directory, job ID, and log;
- one explicit array concurrency cap;
- independent run state, checkpoints, result files, and exact resume commands
  for every session;
- no invocation of the local `run_batch` process inside one large
  allocation; Slurm owns cross-session parallelism;
- no within-session parallelism or shared mutable batch state;
- one-shot aggregate status only; and
- manual resubmission of an exact failed session, never automatic array-wide
  retry.

WP13 begins with mocked array tests and a two-session synthetic smoke test. A
real multi-session array remains separately authorized. A later session may
reuse a finalized single-session Slurm profile only when a benchmark-derived
projection is available and both `max(1.5 * projected_peak_RSS,
projected_peak_RSS + 2 GiB) <= approved_memory` and `2 * projected_upper_wall
time <= approved_wall_time`. Otherwise stop for a new resource decision.
Trial, unit, tensor-byte, time-bin, and fit-count ratios to CT026 are reported
as context, not treated as fixed scaling laws or arbitrary admission cutoffs.

## 5. Configuration contract

`TaskDecodingConfig` contains explicit scientific settings plus two execution
settings. Keep one typed configuration object for a simple user experience,
but classify its fields explicitly so execution-host choices do not alter
scientific identity.

Scientific fields are:

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
- optional tuning grid;
- random seed;
- coefficient nonzero tolerance;
- optional trusted UTC bounds per manually aligned probe.

Execution-only fields are:

- output root; and
- worker count for batch session parallelism.

The command-line `--workers` value, when present, is an explicit execution-only
override of the configured batch worker count. It never changes a session's
scientific fingerprint or saved scientific configuration. Resolved absolute
paths, local/detached/Slurm mode, PIDs/job IDs, Slurm resources, log paths, and
plot display selections are also execution provenance rather than scientific
settings.

`scientific_config_payload(config) -> dict[str, object]` is the single owner of
this separation. It returns a JSON-serializable, deterministically ordered
mapping of the scientific fields with portable paths plus the code-owned
analysis version and frozen estimator/PCA controls. Fingerprinting, matching,
tests, and result provenance use this function rather than maintaining parallel
include/exclude lists.

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
seed 0, and `1e-8` coefficient tolerance. The exact LogisticRegression,
ElasticNet, and PCA controls listed in revision 5 are code constants recorded
in the scientific payload; the initial JSON does not expose extra knobs for
them. Changing one requires an analysis-version bump.

Configuration validation checks values and cross-field consistency without
opening large arrays. Input/path validation is a separate pipeline stage.
Augmented-table validation always requires shared identity/alignment/baseline
columns plus only the source columns needed by the selected targets. The
default full target list therefore requires the complete revision-5 set. One
validation error reports every missing required column together.

Resolve relative JSON paths against the configuration file's parent directory.
The resolved `neural_session.json` parent is the canonical `session_root`.
Require the metadata file, augmented-trial CSV, feature-parameter JSON, every
explicit neural input, and output root to be inside that root; reject required
data or output paths that escape it. The configuration file itself may live
elsewhere, although the documented workflow places it in the session root.
Save execution-host absolute paths only as provenance. Scientific
configuration and input identities use canonical paths relative to
`session_root`, so copying the same tree under a different workstation or
cluster prefix preserves identity.

Use one small, documented file-identity policy:

- every identity, including `neural_session.json`, records the canonical
  session-relative path, file size, and
  whole-second modification time;
- small structured inputs (JSON, CSV, TSV, and text files no larger than 64
  MiB) additionally record a streaming SHA-256 digest;
- large binary spike/alignment arrays use path, size, and modification time so
  lightweight dry-run does not scan their entire contents; `rsync -a` is
  required to preserve these fields between workstation and cluster; and
- a directory identity is the sorted list of the explicit files the loader
  will open, never a hash of an unspecified directory tree.

This policy protects against ordinary replacement or preprocessing changes; it
is not intended as an adversarial integrity system. Save the identity method
and values in `input_manifest.json` and use exactly the same records locally
and on the cluster.

`build_input_manifest(...) -> dict[str, object]` is the single owner of this
policy and accepts the explicit resolved files plus their portable paths; the
CLI, fingerprinting code, cluster preflight, and tests must not recreate file
identity rules independently.

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
- one canonical numeric-encoding column per target; numerical targets retain
  their stored native values and units without normalization or
  standardization;
- target-valid boolean columns; and
- baseline validity/reason fields.

Shifted targets are built before baseline filtering. The function must not
modify its input dataframe.

### 6.2 Rate tensors

PFC and HPC tensors use:

- axis 0: the same ordered common neural-eligible trial subset for both
  regions, accompanied by `trial_row_indices` mapping each tensor row to the
  original target-table row and stable `cur_trial` identity;
- axis 1: common event-relative time-bin centers;
- axis 2: region-specific stable units; and
- values: float firing rates in Hz.

Do not build separate rate tensors for targets. Build each regional tensor once
per alignment/bin-width run over rows with finite alignment and a complete
window on every configured probe, then project each target's eligibility mask
through `trial_row_indices`. Rows excluded from the tensor are never represented
as zero-rate observations. Preserve full-original-row eligibility and outer-fold
arrays separately in results so trial matching remains inspectable.

The initial implementation may hold both regional tensors in memory. Dry run
reports their exact float64 allocation as
`n_tensor_trials * n_time_bins * (n_pfc_units + n_hpc_units) * 8`, and reports
source file sizes separately as I/O/provenance facts. Compressed or on-disk
file sizes are not RAM estimates. The only pre-benchmark hard stop is when the
exact tensor allocation alone exceeds 50% of Linux `MemAvailable` (injectable
in tests), or available memory cannot be determined. Otherwise do not claim
that dry run predicts peak RSS. The bounded CT026 benchmark's measured peak
RSS is authoritative for later local concurrency and Slurm sizing; do not add
disk-backed arrays or streaming without measured need and a revised plan.

### 6.3 Feature preprocessing

Within one target and outer fold:

- determine constant/unavailable units from the outer training observations;
- pool training trials and time bins for regional mean/scale;
- reuse that transform across all time-bin decoders in the fold;
- fit one PCA per selected region and fold when PCA is requested;
- reuse PFC/HPC transforms for standalone and combined results; and
- apply unchanged transforms to the outer test rows.

Tuned mode repeats these steps within each inner training split. Do not reuse
outer-training transforms inside inner validation. Within one target and outer
fold, fit each outer regional transform once and reuse it across all time bins,
representations, and standalone/combined region results. Within each inner
fold, fit each regional transform once on that inner-training subset and reuse
it across every time bin and all 15 candidates. Candidate selection is
time-bin/region/representation-specific, but transform fitting is not.

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

For each target and outer fold:

1. create three grouped inner folds from only the outer-training rows, using
   deterministic `StratifiedGroupKFold(shuffle=False)` for categorical targets
   and deterministic `GroupKFold(shuffle=False)` for numerical targets;
2. for each inner fold, fit the PFC/HPC training transforms once and cache the
   transformed train/validation tensors for reuse across all time bins,
   representations, region configurations, and 15 candidates;
3. for each time-bin/region/representation cell, evaluate all candidates on
   those transformed inner folds, invalidating a candidate if any required
   inner fit/metric is invalid;
4. select the highest mean inner balanced accuracy or $R^2$, resolving exact
   ties by declared candidate order; and
5. fit the PFC/HPC transforms once on all outer-training rows, reuse them for
   every outer cell, refit each selected estimator, and evaluate once on the
   outer-test rows.

Keep tuned mode mechanically explicit rather than hiding it inside a generic
search object: the pipeline must fit PCA inside each inner training split and
retain failure reasons and selected settings. Validate that blocks never cross
inner train/validation partitions and that both partitions contain both classes
for categorical targets. If the requested valid inner folds cannot be formed,
the tuned cell is unavailable rather than falling back to trial-wise splitting
or a different fold count.

### 7.4 Complete-fold aggregation

Aggregate after all individual folds are stored. A summary cell is available
only if every requested fold is valid and finite. Unavailable cells preserve
surviving fold records but have no primary mean.

Only declared scientific invalidities become unavailable cells and allow
unrelated work to continue: invalid grouped folds/class coverage, a constant
target, no usable features, a convergence warning, or a non-finite fit/score.
Unexpected programming, schema, or I/O exceptions fail the run, flush state
and logs, preserve already-published target checkpoints, and return nonzero.
They are never caught by a broad exception handler and relabeled as missing
scientific results.

## 8. Saved-run layout and schema

Each execution creates an immutable session-local directory:

```
<session_root>/analysis_runs/
    task_variable_decoding_<YYYY-MM-DDTHH-MM-SSZ>/
        config.json
        input_manifest.json
        run_state.json
        resume_command.txt
        status_command.txt
        execution.json
        execution_guard.json  # present only while an execution owns the run
        results.npz
        run.log
        console.log
        summary.md
        run_session.py
        run_batch.py
        slurm_submission.json  # only when submitted through WP11
        checkpoints/
        figures/
```

The script files are snapshots of the two launch modules used for the run.
Human-readable plots and report material remain in the run directory. No output
is written into the Git repository.

### 8.1 Identity and rerun behavior

A stable run fingerprint is computed from:

- analysis version and a scoped scientific-source fingerprint;
- normalized scientific configuration with portable relative paths, not
  workstation or cluster root prefixes;
- session ID;
- `neural_session.json`, augmented-table, and feature-parameter identities;
- portable alignment/sorter/quality identities using the Section 5 policy; and
- explicit region/unit-selection rules.

The normalized scientific configuration contains only the scientific fields
listed in Section 5. Output roots, batch worker counts, resolved absolute path
prefixes, foreground/detached/Slurm mode, scheduler resources, logs, and display
choices are excluded. `scientific_source_fingerprint(...)` hashes file paths
and contents for every `.py` file in `src/neural_analysis/task_decoding/`, the
direct runtime dependencies
`src/neural_analysis/session_metadata.py`,
`src/neural_analysis/spike_behavior/loading.py`, and
`src/neural_analysis/population/pca.py`, plus `pyproject.toml` and `uv.lock`.
The initial list is explicit and reviewed whenever a new direct dependency is
introduced. A local persistent `new`/`resume` requires only these relevant
tracked files to be clean and rejects relevant untracked source; unrelated
documentation or source changes do not block it. The full Git HEAD and dirty
summary are recorded as execution provenance, not scientific identity.
`dry-run` may report relevant source dirtiness but performs no run preparation.
The cluster wrapper retains the stronger whole-checkout tracked-clean,
exact-pushed-commit gate for operational reproducibility.

Resume compares the saved scoped source fingerprint, analysis version,
scientific configuration, and input manifest. It does not require the same Git
HEAD when unrelated repository files changed. Any scientific-code change must
also bump `ANALYSIS_VERSION`; tests enforce that the version and scoped source
identity are both saved, while review/commit discipline enforces the bump.

The default single-session runner searches for a completed matching
fingerprint and reports/skips it. `--rerun` creates a new timestamped
directory rather than overwriting. A changed version, scoped source, input, or
scientific configuration always creates a different fingerprint.

Use small JSON manifests and ordinary NPZ files. No database, lock service, or
content-addressed object store is needed.

### 8.2 Launch, interruption, and resume state

`new` writes the run directory, immutable saved configuration,
input/code identity, initial `run_state.json`, and exact
`resume_command.txt` and `status_command.txt` before loading large
spike arrays. `execution.json` records foreground/detached/Slurm mode
and the local PID or scheduler identity as advisory execution metadata. Run
state uses a small explicit lifecycle such as initialized, submitted,
preflight complete, running, interrupted, failed, and complete.

Only one process may execute a run directory. Before entering the foreground
pipeline, atomically create a small `execution_guard.json` containing mode,
host, PID, start time, and Slurm job ID when applicable. `resume` refuses when
the recorded same-host PID is alive or the recorded Slurm job is pending or
running. A dead same-host PID or terminal scheduler state is a stale guard that
may be replaced only after this bounded liveness check is recorded. An
unresolvable foreign-host guard stops with an actionable message rather than
guessing. Remove the guard on orderly exit; retain enough execution history in
`execution.json` for diagnosis. This is a single atomic file, not a lock
service or heartbeat system.

The state file includes started, last-updated, and completed timestamps;
current stage; completed and total target names; the last error/warning; and
whether final results were published. Update it at stage transitions and
completed-target boundaries. Do not add a heartbeat daemon or a monitoring
database. A long target can legitimately leave the timestamp unchanged, so
`status` reports facts rather than declaring a run stale from elapsed time
alone.

Update the state file atomically at package-defined boundaries and flush the
log before returning a failure/interruption code. A resumed run trusts only its
saved configuration/fingerprint and valid target checkpoints. It must not use
new command-line scientific overrides or infer the latest run directory.
Before claiming the execution guard, it also verifies that the current scoped
scientific-source fingerprint and analysis version equal the saved values; a
difference requires a new run rather than resuming old checkpoints with changed
scientific code.

The summary and live handoff record distinguish an interrupted resumable run
from a completed result. A missing final `results.npz` is never
presented as complete. `console.log` captures detached-process or
scheduler standard streams, while `run.log` is the pipeline log; both
are flushed at stage/target boundaries so later inspection does not depend on
the original terminal or Codex task.

### 8.3 Checkpoint boundary

The restart boundary is one completed target. Write a target checkpoint only
after all requested region/representation/time/fold cells for that target have
finished or been recorded unavailable. Resume only checkpoints whose complete
fingerprint matches.

This limits lost work without adding per-fit transaction machinery. The final
`results.npz` is written only after combining target checkpoints. Write every
checkpoint, final NPZ, summary, and required default PNG to a uniquely named
temporary file in its destination directory, flush/close it, and publish it
with `os.replace`. Set `final_results_published=true` and lifecycle
`complete` only after `results.npz`, `summary.md`, and the required default
heatmaps have all been published and validated. Optional coefficient figures
are not part of the completion gate. A failure during final reporting remains
resumable from target checkpoints and must never expose a false complete run.

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

`summary.md` records the analysis goal, analysis name, UTC date/time, included
session, the snapshotted main and batch scripts, scientific configuration,
eligibility/fold coverage, warnings and unavailable results, stage and total
timings, and a concise scientific description of the saved outputs. It does
not claim a scientific conclusion that the saved metrics do not support.

If dense coefficient arrays would waste unreasonable space after real
preflight, use parallel long-form primitive arrays inside the same NPZ. Do not
switch to a new storage dependency.

## 9. Plotting and webapp plan

### 9.1 Offline figures

Generate from saved results:

1. categorical heatmap for balanced accuracy;
2. categorical heatmap for ROC AUC;
3. numerical $R^2$ heatmap; and
4. optional unit-coefficient figure(s) generated on demand from a selected
   saved direct-unit result.

Plots use opaque white backgrounds, black text/axes, readable fonts, complete
labels, and captions. Unavailable cells use a mask/color distinct from chance
or zero.

The pipeline does not generate every combination as a PNG by default. Saved
arrays support interactive inspection; default report figures should remain a
small, declared set. On-demand coefficient display/exports are presentation
choices, not scientific configuration fields or run-completion requirements.

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

### Roles

**Lead Sol supervisor - `gpt-5.6-sol`, high reasoning**

- owns authorization, scientific contracts, package order, shared files,
  commits, and the live handoff;
- gives each worker an exact allowlist, tests, commands, and stop gate; and
- independently reproduces RED/GREEN and marks gates complete.

**Terra package worker - `gpt-5.6-terra`, high by default and xhigh
where assigned**

- receives one bounded tests-only or implementation-only task and edits only
  its allowlist;
- stops after genuine RED, then implements only after Sol verifies and commits
  the tests; and
- makes no commits, real-data/Slurm actions, or nested assignments and returns
  exact files, commands/results, and unresolved risks.

**Independent Sol gate reviewer - `gpt-5.6-sol`, high or xhigh as
assigned**

- reviews a stable tests-only design or stable GREEN diff read-only;
- checks scientific drift, leakage, array axes/units, validity policy,
  checkpoint identity, failure handling, and missing tests;
- does not edit, commit, or broaden scope; and
- is mandatory for WP4, WP5, WP9, WP10 interpretation, WP11, and WP13.

Recheck model and effort availability at implementation start. Do not silently
substitute another model or effort. If the named configuration is unavailable,
stop and ask the user to revise the agent plan.

### Shared-worktree and concurrency rules

- Use at most one write-enabled Terra worker at a time.
- Sol may run independent read-only review concurrently only while the diff is
  stable.
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

Every Terra assignment repeats its package ID/model/effort, objective, exact
allowlist, contract, commands, expected RED/GREEN stop, prohibited actions,
current HEAD/ownership, and required return evidence. An interrupted report is
not a gate; Sol re-audits the diff and reruns the last command before resuming.
The detailed Sol/Terra breakdown and independent-review gates live once, in the
WP0-WP13 descriptions below.

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
4. Missing augmented columns are reported together; a selected subset requires
   shared baseline/alignment columns plus only its target sources, while the
   default target set requires the complete revision-5 set.
5. Empty/duplicate `cur_trial` values fail.
6. Feature-parameter JSON must be an object.
7. HMM display labels describe signed belief, while source columns remain
   unchanged.
8. Paths resolve from the config parent to the canonical metadata-parent
   session root; required inputs and output root outside it are rejected, while
   moving the whole tree preserves portable identities.
9. Scientific fingerprint payload includes every scientific field but excludes
   output root, worker count, execution mode, scheduler resources, resolved root
   prefix, and display choices.
10. Frozen LogisticRegression, ElasticNet, and PCA controls match revision 5,
    are recorded, and are not extra initial JSON knobs.

### 10.3 Target and eligibility tests

1. Previous/next targets are constructed before filtering.
2. Manual/no-choice adjacent rows are not bridged.
3. Switch/stay derivations match explicit action sequences.
4. Dark state 2 is excluded only from binary current-state eligibility.
5. Target-specific missingness does not leak into unrelated masks.
6. Baseline manual/no-choice/current-alignment exclusions are correct.
7. PFC/HPC/combined representations receive identical target trial rows.
8. Class labels and positive-class mappings are persisted.
9. Numerical target values equal their source values exactly after numeric
   coercion; no normalization or standardization is applied.

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
11. Invalid or uncovered original rows are absent from tensors rather than
    represented as zeros, and `trial_row_indices` maps every tensor row back to
    the full target table and `cur_trial` identity.
12. Projecting different target masks through `trial_row_indices` preserves
    matched PFC/HPC/combined trial order.
13. Dry run reports exact tensor allocation bytes and source file sizes
    separately; only tensor bytes participate in the pre-benchmark 50%-of-
    `MemAvailable` guard.

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
12. Inner folds use the declared grouped splitter, never split a block, and
    validate categorical train/validation class coverage.
13. Invalid requested inner folds make the tuned cell unavailable without
    changing fold count or falling back to trial-wise splitting.
14. Instrumented transform fit counts are invariant to time-bin and candidate
    count: one PFC/HPC transform per outer fold and one per inner fold, reused
    across representations and standalone/combined results.
15. Candidate tie resolution follows declared order.
16. Balanced accuracy uses threshold 0.5 and AUC uses the same fit's scores.
17. $R^2$ uses native targets and returns unavailable for constant/singleton
    test targets.
18. Negative $R^2$ and below-chance categorical scores remain valid.
19. A convergence warning invalidates the fit; a converged all-zero solution
    remains valid.
20. Coefficient signs, scales, selection frequencies, and `1e-8`
    threshold are correct.
21. Primary means require all requested valid outer folds.
22. Results are deterministic for the configured seed.

Include a direct installed-API test for the supported scikit-learn 1.8
logistic configuration so implementation does not depend on the deprecated
`penalty` argument.

### 10.6 Result I/O and restart tests

Create `test_results.py` and `test_pipeline.py`:

1. NPZ round trip preserves arrays, labels, axes, units, and metadata.
2. The named `meta` dictionary preserves required provenance.
3. Schema/version mismatch fails clearly.
4. The input manifest includes `neural_session.json`; small structured inputs
   use SHA-256 identities and large binaries use portable path/size/mtime
   without reading their contents.
5. Scoped scientific-source identity changes when a relevant package/direct
   dependency or environment lock changes, but not for an unrelated document;
   relevant dirty/untracked source blocks persistent local preparation.
6. Fingerprints change with scientific input/config/analysis-version/scoped
   source changes, remain stable across execution-only/unrelated-repository
   changes and root-prefix moves, and use the documented input identities.
7. An identical completed run is skipped by default.
8. Explicit rerun creates a new path without overwrite.
9. `new` writes saved config, run state, and exact resume/status
   commands before an injected large-array/long-running stage.
10. Matching target checkpoints resume; mismatched checkpoints or a different
   scoped source fingerprint/analysis version are rejected.
11. Interrupted/failed foreground states return nonzero, preserve
   logs/checkpoints, and do not publish a complete result; detached/submission
   exit codes report launch acceptance rather than later scientific outcome.
12. Each declared scientific invalidity is recorded unavailable without
    aborting unrelated cells, time bins, or targets; an injected unexpected
    programming, schema, or I/O exception fails the run, preserves completed
    checkpoints, returns nonzero, and is not masked as unavailable.
13. Log and summary contain the analysis goal/name/date, session, snapshotted
    scripts, parameters, eligibility/folds, warnings, scientific output
    description, and timing information.
14. Dry-run loads metadata/small tables and directly inspects small cluster and
    channel metadata without calling `load_sorter_metadata`, loading
    `spike_clusters.npy`/aligned spikes, or fitting models.
15. Detached `new` and `resume` return after bounded setup, redirect
    child output, save execution identity, and run the same foreground pipeline
    without a shell-dependent scientific path.
16. One-shot `status` reads only small saved files, reports progress and
    result completeness, and never loads spikes, fits, resumes, or polls.
17. A simulated abrupt child death leaves checkpoints resumable and is
    reported without automatic restart or a false `complete` state.
18. A Linux integration smoke test confirms that a detached synthetic child
    survives the launcher process and completes with no open terminal. Before
    relying on Codex as the launcher, repeat that smoke through the actual
    Codex command environment because a host may clean up child processes.
19. `status --verify-results` additionally validates a completed NPZ
    through the saved-result loader without opening source data or mutating the
    run.
20. Foreground, detached, and mocked Slurm paths use the same prepare/execute
    functions and the private CLI bridge does not re-parse scientific settings.
21. A second execution refuses a live same-host PID or pending/running Slurm
    job; a verified stale guard can be replaced without a lock service.
22. Checkpoints, NPZ, summary, PNGs, and state are atomically published, and
    `complete` is written last only after required artifacts validate.
23. Session and batch entrypoints set and record one-thread OpenMP/MKL/OpenBLAS
    values before lazily importing numerical modules; local benchmark and
    mocked Slurm execution identities match.

### 10.7 Batch-runner tests

Create `test_run_batch.py`:

1. The config list ignores blank/comment lines and preserves declared session
   order.
2. Dry-run reports every session, planned fits, exact tensor bytes, and selected
   worker count without loading spike arrays or fitting.
3. Before a measured benchmark profile exists, batch execution is capped at one
   worker. With projected peak-RSS values, default/explicit workers are capped
   so their summed projected peaks remain within 50% of injected
   `MemAvailable`.
4. One session whose exact tensor allocation fails the single-session guard is rejected rather
   than launched with one forced worker.
5. Sessions use independent run directories/state and invoke the same
   single-session preparation/execution path without within-session
   parallelism.

### 10.8 Plot and webapp tests

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

### 10.9 Documentation and command tests

Create `test_task_decoding_documentation.py`:

1. The reusable example configuration loads and validates without CT026 or
   another machine-specific absolute path.
2. Every Python file in `task_decoding`, including `__init__.py`, is described
   in the in-package README.
3. The top-level neural README contains local dry-run, foreground/detached
   new-run, detached resume, one-shot status, batch, output-location, and
   webapp instructions.
4. `run_session --help` and `run_batch --help` exit
   successfully and show the documented modes.
5. The documented single-session dry-run works against a small temporary
   metadata/config fixture without loading spike arrays.
6. Exact resume and status commands are written before an injected long-running
   stage.
7. README result paths and CLI mode names match implementation constants rather
   than describing obsolete paths.
8. The quickstart explicitly permits the terminal or Codex task to close after
   detached launch and documents later one-shot state/log inspection.
9. The quickstart explains how to regenerate an incomplete augmented table
   through normal behavior processing and explicitly rejects hand-editing it.

Documentation tests should validate stable commands and file ownership, not
word-for-word prose.

### 10.10 Synthetic integration tests

Create `test_synthetic_integration.py`:

1. A seeded two-region synthetic session completes end to end.
2. Injected time-local signal produces its strongest decoding in the expected
   interval without requiring an exact score.
3. All six region/representation combinations are computed, and PFC-only,
   HPC-only, and combined axes are correct.
4. Both target families and both representations complete.
5. A deliberately impossible grouped categorical target is unavailable while
   other targets finish.
6. Saved results reload and reproduce plotted summaries.
7. A held-out-only offset does not influence training scaling/PCA, serving as
   an end-to-end leakage regression.

### 10.11 Single-session cluster-path tests

Write these only after WP10 evidence and explicit WP11 approval:

1. `submit-new` runs a lightweight cluster dry run, prepares one exact
   cluster run directory through the Python runner, and submits that directory
   to the same foreground pipeline without changing scientific settings.
2. `submit-resume` forwards one exact saved cluster run directory
   without re-parsing or overriding its scientific configuration.
3. Unknown modes, missing paths, wrong repository roots, and tracked-dirty
   checkouts return nonzero before launch.
4. The top-of-script partition/task/CPU/memory/time/signal/log/mail fields match
   the user-approved WP10 resource record. Tests do not assume PPC's eight
   CPUs, 32 GB, or 72 hours.
5. The job uses the exact submitted commit, explicit thread limits, and
   `uv run --frozen --no-sync --offline`; arguments, exit status, and
   scheduler signals reach the Python process.
6. A simulated `SIGTERM` leaves no false complete result, flushes
   state/logs, and preserves only valid completed-target checkpoints.
7. Resume requires one exact run directory and never searches for a latest
   run.
8. A mocked submission failure is returned, recorded in the initialized run,
   and never triggers automatic resubmission.
9. Cluster `status` combines saved pipeline state with at most one
   mocked `sacct` lookup, reports the required accounting fields, and
   never polls or mutates the run.
10. Moving the same session/config tree between local and cluster root prefixes
    preserves the scientific fingerprint while recording both resolved
    execution paths.
11. A completed returned fixture loads from `.incoming-<run_id>`
    without cluster source access; an incomplete/corrupt fixture fails before
    final promotion.
12. Documentation contains non-destructive input/result `rsync`
    commands, excludes `analysis_runs/` from input synchronization, prohibits
    broad `--delete`, uses an incoming directory, and explains
    exact-commit/offline-environment preparation.
13. No test invokes a real scheduler, SSH, rsync, network, or scientific data.

### 10.12 Deferred cluster batch-array tests

Write these only in WP13, after WP12 acceptance:

1. A config list produces one immutable array-index-to-session/run mapping.
2. Each array element invokes the proven single-session cluster path with its
   exact prepared run directory and no shared mutable scientific state.
3. The documented concurrency cap is forwarded exactly.
4. One element's failure does not alter another element's state or outputs.
5. Aggregate status performs one bounded scheduler query and read-only
   per-session status checks; it never polls.
6. Resume instructions name individual failed run directories and never
   resubmit the whole array automatically.
7. A dry run rejects sessions lacking the measured projection/admission checks
   required by Section 4.8 and reports trial/unit/tensor/fit-count context.
8. A two-session synthetic array smoke passes with a fake scheduler; no test
   submits a real job or touches experimental data.

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

- Terra high writes RED config, scientific/execution payload, validation,
  target, and eligibility tests.
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
- Gate: shapes, axes, Hz units, coverage, original-row mapping, exact tensor
  allocation guard,
  and unit identities are verified; invalid alignment rows cannot appear as
  zero-filled neural observations.
- Handoff: record representative synthetic tensor shapes and memory-estimate
  inputs; no CT026 arrays are loaded.

### WP4: Grouped modeling

- Terra xhigh writes RED split/preprocessing/model tests.
- An independent Sol xhigh reviewer audits leakage and validity coverage before
  the tests-only commit.
- Terra implements fixed mode first. Sol verifies it before a separate
  follow-up authorizes optional tuned mode.
- Gate: leakage and transform fit-count tests, frozen estimator/PCA controls,
  invalid-fold/error-boundary policy, coefficient contract, and
  deterministic results are green; independent Sol xhigh review finds no
  unresolved scientific/numerical issue.
- Handoff: record exact installed scikit-learn API/version and measured
  synthetic fit counts.

### WP5: Results and single-session pipeline

- Terra xhigh writes RED NPZ, input/scoped-source fingerprint, checkpoint,
  failure-boundary, atomic-publication,
  prepare/execute, single-writer, dry-run, interruption, exact-resume,
  detached-launch, and read-only-status tests.
- Independent Sol xhigh review approves the saved schema/state-machine tests
  before the tests-only commit.
- Terra implements results, pipeline, run directory, logging, summary,
  `dry-run`/`new`/`resume`/`status`, optional
  local detachment, internal prepared-run execution, single-writer protection,
  and exact resume/status command persistence.
- Gate: synthetic small target runs resume and round-trip; a detached fixture
  returns immediately, completes without its launcher, and is inspectable once
  later without computation or polling.
- Handoff: freeze CLI modes, output paths, schema version, detachment behavior,
  and resume/status semantics before batch, webapp, or README work.

### WP6: Batch runner

- Terra high writes RED dry-run, skip/rerun, and session-isolation tests.
- After Sol's tests-only commit, Terra implements the session-list runner with
  parallelism across sessions only.
- Before benchmark-derived projected peak RSS is available, cap batch execution
  at one worker. Afterwards start from available CPUs/requested workers and
  reduce until summed projected peaks are at most 50% of
  `MemAvailable`. A session whose exact tensor allocation fails Section 6.2
  still stops before allocation. Print the profile source, worker count, and
  calculation in dry-run and the run log.
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
  10.9; documentation changes are committed separately.
- Sol runs every documented `--help` and dry-run command against a
  temporary fixture, exercises detached start plus one later status check, and
  checks every Python file is described.
- Gate: a scientist can configure, dry-run, start and leave a run, inspect its
  state/log once later, resume it, locate outputs, and open results without
  reading implementation source; a maintainer can identify every file's role
  from the package README.
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

### WP9A: CT026 augmented-table preparation

- This package requires separate explicit user approval for the exact normal
  behavior-processing command recorded by WP1. It authorizes preparation of
  the augmented table only, not neural loading, benchmarking, or decoding.
- Sol first records the source raw/augmented paths, current augmented-table
  identity, expected output path, and whether the normal command overwrites the
  existing CSV. If it overwrites, make one timestamped backup beside the input
  before running so the change is recoverable.
- Terra high acts as a command runner only: run the approved existing behavior
  augmentation/feature-save path that now produces `rewards_in_block`; never
  patch the experimental CSV by hand. Record the exact command, exit status,
  output identity, and log, then stop.
- Sol performs read-only post-write validation: all revision-5 required columns
  exist, row count and ordered `cur_trial` identities match the pre-run table,
  feature-parameter JSON remains present/valid, and the new column passes the
  WP1 semantic checks on the produced rows.
- Gate: the regenerated CT026 table passes validation and the user is shown the
  backup/output identities. A mismatch stops before WP10 and does not trigger
  an ad hoc repair.
- Handoff: record the approved command, backup path when used, old/new file
  identities, validation output, and the exact proposed read-only WP10 dry-run.

### WP10: CT026 2026-08-03 preflight and benchmark

- This package requires separate explicit user approval for the exact CT026
  commands.
- WP9A must already have produced and validated the complete augmented table;
  WP10 remains read-only with respect to behavior inputs.
- Before real data, repeat the detached synthetic smoke through the intended
  launcher. If the Codex execution host cleans up detached children, use the
  same documented command from the user's ordinary terminal; do not keep a
  Codex task alive as a workaround.
- Terra high acts as a command runner only: read-only validation, dry-run, and
  the approved bounded representative fixed-mode benchmark. It starts the
  benchmark detached if it is not expected to finish promptly, records the run
  directory and follow-up commands, and stops; it makes no source edits,
  parameter changes, or monitoring loop.
- Sol confirms selected channels/units, eligible trials, block/class coverage,
  tensor memory, exact model-fit count, one-thread execution identity, stage
  timings, and peak memory.
- An independent Sol xhigh reviewer checks raw evidence and the local/full/tuned
  runtime projection.
- Gate: after the process has finished, one explicit later task reads
  `status`, logs, and saved results once. The user reviews the evidence
  and chooses a detached full local fixed run or activation of WP11. Cluster
  preference may be based on cost/convenience even when local execution would
  technically succeed. There is no automatic threshold and no implicit long
  run.
- Handoff: update the live snapshot with exact config/run directory, launch and
  status commands, logs, fit counts, stage timings, peak RSS, output size,
  full fixed/tuned projections, CPU efficiency, and the proposed Slurm CPU,
  memory, and wall-time request with its safety factors.

### WP11: Single-session cluster execution and transfer

- Start only when the user chooses cluster execution after WP10 and explicitly
  approves wrapper/transfer implementation. Technical local feasibility does
  not preclude this choice.
- Sol records the benchmark-derived resource proposal. Use one CPU unless
  measured evidence justifies more. Proposed memory is the larger of 1.5 times
  projected peak RSS or projected peak plus 2 GiB, rounded upward; proposed
  wall time is twice the conservative projected full-run duration, rounded
  upward to an hour. These are first-job safety rules, not permanent defaults,
  and the user approves the exact final directives before tests are written.
- Terra high writes the Section 10.11 mocked wrapper, portability, signal,
  transfer-validation, and documentation tests and stops at RED. After Sol's
  tests-only gate, the same worker implements the thin wrapper and bounded
  Python state/termination support.
- Independent Sol xhigh review checks portable fingerprints, exact-run resume,
  resource visibility, repository/environment identity, one-shot scheduler
  accounting, input/result transfer safety, no automatic resubmission, and no
  duplicated scientific settings.
- Gate: local and cluster dry-run fixtures plus mocked wrapper/transfer tests
  pass; exact one-command `submit-new`, `submit-resume`, and
  `status` instructions and manual rsync round trip are documented. No
  experimental transfer or real job is part of this gate.
- Handoff: record wrapper/resource directives, transfer roots, exact pushed
  commit, environment gate commands, and proposed synthetic Slurm smoke. A real
  transfer/submission remains unperformed until explicitly approved.

### WP12: One-session scientific inspection

- This package requires explicit approval of the exact fixed-mode CT026
  configuration and local command or the exact input rsync plus Slurm command.
- Terra high acts as command runner only. It launches the detached local run or
  Slurm job, records the receipt/run directory, and stops. Sol does not monitor
  or alter parameters mid-run.
- In a later user-requested task, Terra or Sol performs one read-only
  status/log/result inspection. If the run is interrupted, resumption is a new
  explicit action using the saved exact command; it is never automatic.
- For a cluster run, first execute the separately approved input rsync and
  cluster dry run. After completion, inspect `sacct` once, record actual
  `Elapsed`, `TotalCPU`, and `MaxRSS`, then separately approve the
  exact result-return rsync into a hidden local incoming directory. Validate
  before atomic promotion.
- Compare requested versus actual resources and freeze the normal
  single-session resource profile for later sessions. Do not reduce safety
  margins or generalize beyond the observed workload without recording the
  decision.
- User inspects saved heatmaps, fold coverage, warnings, and coefficients.
- Only after user approval should a batch of additional sessions be considered.
- Handoff: record immutable run paths, configuration, commit, launch and status
  commands, transfer commands, requested versus measured timing/CPU/memory,
  finalized resource profile, warnings/errors, and user acceptance. Do not
  treat completion as permission for other sessions.

### WP13: Deferred cluster batch array

- This is the final package. It starts only after WP12 scientific acceptance,
  finalized single-session cluster resources, completed documentation, and
  separate user approval.
- Terra high writes the Section 10.12 mocked array and two-session synthetic
  tests, stops at RED, and later implements only the narrow submission/status
  extension after Sol commits the tests.
- The array mapping is immutable, concurrency is explicit, every element uses
  the proven single-session runner, and failed sessions resume individually.
- Independent Sol xhigh review checks isolation, resource-profile admission,
  argument quoting, one-shot status, and the absence of array-wide automatic
  retries.
- Gate: focused tests and a fake-scheduler two-session smoke pass; user and
  maintainer documentation includes exact batch commands and recovery. No real
  array is part of the implementation gate.
- Handoff: present the exact proposed real config list, concurrency cap,
  transfer plan, and resource estimate for separate authorization.

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

Record wall time with `time.perf_counter`. On the planned Linux hosts,
record and normalize peak RSS with the standard-library
`resource.getrusage` API; if WP11 is activated, also preserve Slurm's
reported `MaxRSS` for comparison. No profiling dependency is needed for
this first benchmark. Also record process user/system CPU time so CPU
utilization can distinguish a long single-core workload from a genuinely
multi-core one. Run the local projection benchmark with
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`, matching the initial one-CPU Slurm job, and record
those values. Record:

1. configuration/metadata/table validation;
2. sorter/alignment loading and unit selection;
3. PFC and HPC rate-tensor construction;
4. target/split construction;
5. fold preprocessing/PCA;
6. estimator fitting/scoring by target family and representation;
7. result serialization;
8. plotting/report generation; and
9. total wall time, user/system CPU time, and peak resident memory.

Report number of trials, blocks, units, time bins, requested/valid folds, model
fits, invalid cells, and output size so timing is interpretable.

Persist these measurements in structured result metadata as well as
`summary.md`; do not rely on a terminal transcript. For detached work,
flush the stage timing and peak-RSS-so-far at every target checkpoint. The
measurement approach must be the same in foreground, detached local, and Slurm
execution so the comparison is meaningful. A Slurm result additionally records
requested and actual scheduler resources from the one-shot accounting query.

### 12.4 Benchmark tiers

1. **Synthetic microbenchmark:** catches pathological overhead and produces a
   stable regression fixture; it is not a production-time estimate.
2. **CT026 bounded benchmark:** choice alignment, 100 ms, fixed mode, one
   categorical target (`Current action`) and one numerical target
   (`Relative doubt`), both representations and all regions. This is
   2,400 requested outer fits and
   includes exact loading/binning once, so it exercises the real session path
   rather than only timing estimators in isolation. Launch it detached if it is
   not expected to finish promptly. The dry run must first confirm that both
   targets have sufficient eligible rows and valid grouped folds; if either
   fails, stop and choose a replacement explicitly rather than silently
   changing the benchmark.
3. **One-shot benchmark inspection:** after launch, the initiating user or
   Codex task ends. In a later task, run `status` once and read the
   completed log/result metadata. If it is still running, report that fact and
   stop; do not begin a watch loop.
4. **Projection:** separate fixed loading/tensor/serialization costs from model
   costs. Project the 8 categorical and 10 numerical targets from their own
   observed per-fit/per-target timings, scale saved-result size by target count,
   and carry forward measured peak tensor memory. Report both the estimate and
   its assumptions and a conservative range because target eligibility and fit
   convergence can change cost; do not multiply total bounded wall time by
   nine.
5. **Full default benchmark/run:** only after the user authorizes the exact
   local or cluster path. A local run uses `new --detach`; a cluster run
   uses the approved rsync, dry-run, and `submit-new` sequence. End the
   initiating task and inspect once later. Compare measured versus projected
   wall time, CPU time, peak RSS, fit throughput, and output size.
6. **Tuned estimate:** representative subset only unless separately authorized.
7. **Local/cluster decision:** Sol reports whether ordinary local use is
   practical and proposes cluster resources. The user may still choose the
   cluster to avoid occupying the workstation. There is no automatic duration
   threshold, transfer, or submission.

### 12.5 Unattended local-to-cluster decision path

Follow WP9A-WP12: validate/regenerate the table when needed, dry-run, launch the
bounded local benchmark and leave it unattended, inspect its durable state
once later, then choose an explicitly authorized detached-local or Slurm run.
The Slurm branch uses the Section 4.7 transfer and Section 4.6 wrapper; returned
results are validated in the hidden incoming directory before promotion.
Both paths use the same prepared run, checkpoints, logs, and result schema, and
neither polls, retries, or resumes automatically.

### 12.6 Slurm resource selection and finalization

`hpc_ppc.sh` proves the site accepts partition `unlimited`, one
task, explicit CPU/memory/time requests, five-minute termination notice, the
private log directory, and existing mail settings. Its PPC-specific values are
not task-decoding defaults.

After the bounded local benchmark, Sol produces a resource table containing:

- requested fit count and measured fit throughput by target family;
- full-run conservative wall-time range;
- measured and projected peak RSS, including full tensor and scaled result
  storage;
- user plus system CPU time divided by wall time;
- proposed tasks/CPUs, memory, wall time, partition, signal notice, and log
  path; and
- the safety-factor calculation and benchmark session/config identity.

For the first task-decoding cluster job, request one task and one CPU unless
measurement supports a revised parallel plan. Propose memory as the larger of
1.5 times projected peak RSS or projected peak plus 2 GiB, rounded upward.
Propose wall time as twice the upper-bound full-run projection, rounded upward
to a whole hour. If these requests exceed site limits, stop for a new plan;
do not reduce them silently.

After the first completed cluster run, compare the request with Slurm
`Elapsed`, `TotalCPU`, `AllocCPUS`, `MaxRSS`,
`ReqMem`, and `Timelimit`. Record and user-approve the resulting
normal single-session resource block. A later session can reuse it only when
its benchmark-derived projection satisfies both the memory and wall-time
admission inequalities in Section 4.8. Otherwise stop for another explicit
estimate. Report trials, units, tensor bytes, time bins, and fit-count ratios
to CT026 as explanatory context. WP13 uses the approved per-session profile for
each element rather than creating a larger shared allocation.

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

WP1 implements the general feature but does not touch CT026. In separately
approved WP9A, create the missing column by rerunning the exact normal
behavioral augmentation/feature-save path recorded by WP1, with the recoverable
backup and post-write checks defined there. Do not patch the experimental CSV
ad hoc.

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
| Fixed/tuned fit cost is large | Reuse fold transforms, benchmark fixed mode, project tuned mode, and checkpoint targets. |
| Leakage or invalid grouped folds distort results | Fit preprocessing only within each training subset; test fold membership, transform call counts, and declared invalidity behavior. |
| Dry-run memory guesses are misleading | Guard only exact tensor allocation; use measured CT026 peak RSS for concurrency and Slurm sizing. |
| Scientific identity changes across hosts or unrelated edits | Use session-relative input identities and scoped scientific-source hashes; keep host paths/full Git state as provenance. |
| A long run is interrupted or resumed twice | Publish atomically, checkpoint targets, keep durable state/logs, and enforce one bounded execution guard without a service. |
| Unexpected defects are hidden as unavailable science | Limit unavailable results to enumerated invalidities; fail and preserve checkpoints on unexpected exceptions. |
| The saved-results UI recomputes | Route before raw-data loading and test a saved-loader-only view. |
| Cluster execution diverges or transfer exposes partial state | Reuse the prepared Python run, require exact clean pushed code, keep rsync explicit, and validate hidden incoming results before promotion. |
| Handoff or concurrent editing loses TDD state | Keep dated package records, one Terra writer, separate RED/GREEN commits, and Sol reproduction of every gate. |

## 16. Approval checklist

Before implementation, confirm:

- revision 5 is the scientific authority and WP0 remains documentation only
  until a separate implementation request;
- `rewards_in_block` is an additive general augmented-table column with the
  frozen entering-trial semantics;
- target eligibility, explicit probe/coverage rules, common tensor-row mapping,
  native numerical units, and all six region/representation outputs match the
  specification;
- fold-local transform reuse and declared-versus-unexpected error boundaries
  are covered by tests before modeling code;
- session-root containment, portable input identities, scoped source identity,
  and analysis-version rules are frozen;
- dry-run guards exact tensor bytes only, the local benchmark uses recorded
  one-thread limits, and measured peak RSS/runtime drive local and Slurm sizing;
- one prepared-run path provides immutable outputs, atomic target checkpoints,
  detached launch, exact resume, and one-shot status without active monitoring;
- the only UI is an integrated saved-results view, with scientist quickstart,
  maintainer file map, and portable example configuration;
- Sol/Terra work follows the documented RED-commit-GREEN sequence and updates
  this live handoff at every gate; and
- CT026 table preparation, benchmark/full run, cluster transfer/submission, and
  final batch-array follow-up each retain their separate user gates.

Approval of this document should be followed by a separate implementation
request. Until then, no code, tests, augmented data, or neural results should be
changed.
