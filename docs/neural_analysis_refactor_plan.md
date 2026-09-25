# Neural Analysis Organization Refactor Plan

## Status and authority

**Current status:** U1-U4 and R0 are complete. R1 was authorized on
2026-09-25, implemented tests-first, and verified locally; user review is
pending. Each R2 scientific-domain slice requires separate approval.

This plan supersedes the former NR2-NR18 package roadmap and the former broad
layered target architecture. Historical execution details, reviews, paths,
hashes, and external-action gates remain in
`docs/neural_analysis_refactor_execution_log.md` and Git history. They are not
repeated here.

`docs/neural_analysis_refactor.md` defines the approved design and package
boundaries. `docs/SoftwareDesign.md` governs readability and implementation
style. `docs/Tasks_neural.md` remains authoritative for established scientific
methods and completed-result evidence.

This is an organization-only plan. It authorizes no source edit, real-data
computation, cache mutation, report generation, external copy, Slurm action,
or deletion.

## 1. Completed usability work

| Milestone | Status | Outcome |
| --- | --- | --- |
| U1 - session metadata | Implemented | Compact version-2 metadata loads, validates, resolves contained probe records, and reports action availability for the supported Open Ephys and SpikeGLX layouts. Version 1 is intentionally rejected. |
| U2 - metadata-driven webapp | Implemented | The existing webapp accepts one metadata path and derives its session, probe/site sources, cache, and existing manual/channel-quality unit controls. One implicit analysis population is selected per probe; there is no separate metadata population selector. |
| U3 - metadata-driven computation | Implemented | The existing Spike-phase/PPC dry run, local launcher, and Slurm wrapper accept the same resolved metadata while preserving explicit scientific and execution choices. |
| U4 - guide and examples | Implemented and accepted | The quickstart and three example metadata files document metadata creation, validation, webapp launch, local/Slurm handoff, resume, recovery, and rerender. The user accepted the current version-2 workflow on 2026-09-25. |

The initial implementation sequence is recorded by the tests-first and
implementation commits from `75397e9` through `5e131fa`. The compact version-2
replacement and restoration of the established webapp channel controls are
recorded by `860de5e` through `e43cdcc`. This plan does not reopen those current
interfaces except to simplify internal adapters or move implementations without
changing behavior.

### U4 acceptance record

The user accepted the version-2 quickstart and examples on 2026-09-25. That
acceptance satisfies the U4 prerequisite only. It does not authorize R0, source
movement, a real run, external copy, Slurm submission, or deletion.

## 2. Refactor objective

Reorganize the whole `src/neural_analysis` package so a scientist can locate
and understand existing responsibilities without learning the history of the
webapp or LFP-summary implementation.

The refactor prioritizes:

1. scientific domain names over generic architecture terms;
2. shallow packages over deeply nested layers;
3. direct functions and frozen data records over behavior-heavy classes;
4. explicit arrays, tables, shapes, axes, and units;
5. one implementation behind temporary compatibility imports; and
6. small, independently reviewable moves.

The refactor does not prioritize uniform file size, architectural symmetry, or
completion of every provisional target module.

## 3. Absolute scope boundary

No readability work may add or broaden:

- metadata fields or schemas;
- source formats, acquisition families, or path-placement rules;
- scientific calculations, defaults, conditions, statistics, or seeds;
- plots, reports, webapp controls, or user actions;
- CLI commands, launcher modes, recovery actions, or schedulers;
- cache, checkpoint, manifest, snapshot, or receipt behavior;
- parallelism, worker policy, or resource policy;
- support for currently unsupported layouts or failure cases; or
- generalized frameworks for analyses, artifacts, workflows, execution,
  plotting, reports, compatibility, or plugins.

If a move requires new behavior, leave the code where it is and plan the new
behavior separately.

Compact metadata schema version 2 is the refactor baseline. Version 1 is
intentionally rejected by the current loader and is not a saved-result or
compatibility surface. No work package adds a version-1 reader, migration
command, or additional metadata field.

### Preserved scientific baseline

Every work package preserves:

- corrected Open Ephys gain, offset, voltage-unit, and sample-rate semantics;
- current source-value semantics and source fingerprints;
- immutable historical artifacts and explicit legacy interpretation;
- current cache, checkpoint, manifest, and configuration identities;
- the approved Power, Synchrony, and Spike-phase/PPC definitions;
- current seeds, trial filters, windows, frequency bands, and unit-selection
  rules; and
- existing saved-result and resume compatibility.

The detailed contracts remain in `docs/Tasks_neural.md` and the execution log.

## 4. Approved architecture

The organization is shallow and domain-oriented:

- root session metadata and stable entry modules;
- `synchronization`;
- `lfp`;
- `spike_behavior`;
- `spike_lfp`;
- `population`;
- workflow-specific `lfp_summary`; and
- `webapp`.

Workflow-specific cache, launcher, report, snapshot, relocation, and profiling
code stays under `lfp_summary`. Plotting stays with the scientific domain whose
result it displays. The complete responsibility map and dependency rules are in
`docs/neural_analysis_refactor.md`.

Every new subpackage has a minimal `__init__.py`, normally only a package
docstring. Do not add wildcard imports or broad convenience re-exports.
Canonical imports identify the owning module unless a narrow re-export is an
explicitly approved public API.

`src/external_tools`, `src/irig_tools`, behavior-preprocessing packages, and
installed dependencies are outside the organization refactor. They may be read
and called through their existing APIs but are not modified by these work
packages.

## 5. Dependencies and tooling

No new dependency is planned.

Use only the existing project dependencies where they are already appropriate:

- standard-library paths, dataclasses, JSON, hashing, process control, and file
  operations;
- NumPy, SciPy, Pandas, and Pynapple for current scientific data operations;
- scikit-learn for current decoding and PCA workflows;
- Matplotlib for current plotting; and
- Streamlit inside the webapp only.

Use `uv` for dependency management and `uv run` for Python commands. Use
`pytest` for tests. Do not add formatting, static-analysis, schema, CLI,
workflow, storage, or plugin dependencies as part of this refactor.

When a work package creates a new use of a library API, verify that API from
the installed package source or official documentation before implementation.
A mechanical move of an already tested call does not require revalidating the
library API.

## 6. Tests-first and commit policy

Every source work package follows this sequence:

1. inspect the surrounding modules and every repository caller;
2. identify the exact public and saved-data contracts in scope;
3. write the new-location, compatibility, and missing characterization tests;
4. run the focused tests and record meaningful RED caused by the absent new
   organization;
5. commit tests before implementation;
6. perform one responsibility-preserving move or split;
7. run focused tests and directly affected integration tests;
8. review the diff for behavior changes, duplicated implementation, and
   undocumented data transformations;
9. run the complete neural suite at work-package completion; and
10. update documentation in the same accepted work package when user-facing
    imports or navigation changed.

Do not copy the full test tree merely to mirror new package paths. Existing
feature tests remain in place unless their filenames become misleading.

Compatibility modules are implementation, not duplicate ownership. Each old
module path forwards to the new canonical implementation and contains no
scientific logic.

R0 classifies compatibility before movement:

| Surface | Migration policy |
| --- | --- |
| Documented command | Preserve permanently. |
| Current root script/function entry point | Preserve its inventoried module dispatch and callable behavior; do not infer a CLI contract. |
| Documented/public Python import | Forward through migration; removal requires a fresh audit and explicit approval. |
| Session metadata JSON | Preserve compact schema version 2 exactly; do not add version-1 migration or compatibility. |
| Repository-private import or helper | Update the repository caller; do not imply an external compatibility promise. |
| Serialized module-qualified class | Preserve the old loading path or add a tested compatibility reader. |
| Multiprocessing worker | Keep it as an importable top-level canonical callable and test process spawning. |
| Test-only monkeypatch seam | Update tests to patch the canonical owner unless the seam is intentionally public. |

A compatibility module guarantees only its inventoried public surface. It does
not have to reproduce arbitrary private globals, private helper locations, or
module-level monkeypatch behavior.

## 7. Documentation and data contracts

Every function or method moved, created, or materially edited during the
refactor, including private helpers, must explicitly document:

- input types;
- array or table shapes and axis meanings;
- physical units;
- missing-value conventions;
- return types, shapes, and units; and
- important errors or preconditions.

A short docstring is sufficient when a helper has no arrays or physical units.
Moving a function does not authorize changing its contract to make the
docstring easier to write.

Array shape, axis order, channel indexing, sample rate, time units, voltage
units, stable IDs, and metadata must remain unchanged. Any reshape, transpose,
resample, or unit conversion already performed by the code remains explicit in
the owning function's contract.

## 8. Performance policy

Clarity has priority unless an existing path is performance-sensitive.

- An ordinary import move requires correctness tests, not a benchmark.
- Preserve the webapp's current loading behavior. Metadata validation and
  metadata/summary-source resolution do not load numerical arrays, but the
  default Unit raster/PSTH view loads the selected population's sorter and
  aligned-spike arrays on its initial render. Do not introduce additional
  eager loading or make the existing default view lazy without separate
  approval.
- Scientific moves must not introduce new full-array copies or eager loading.
- Existing vectorized NumPy/SciPy/Pynapple operations remain unchanged.
- Worker counts, memory maps, chunk/block sizes, checkpoint representation, and
  parallelization strategy remain unchanged.
- Before moving or splitting PPC kernel/runtime code, run the existing
  synthetic or injected representative workload with the same environment and
  worker settings before and after the move. Record runtime and peak memory and
  investigate any material regression. Do not create a new benchmark
  framework.
- Referencing an existing real-data profile is allowed. Running a CT026 or
  other real-session profile requires separate explicit authorization.
- No optimization, Numba, Cython, new storage format, or parallel redesign is
  in scope.

## Execution orchestration - Sol and Terra

The default implementation team uses one `gpt-5.6-sol` root orchestrator and
up to three `gpt-5.6-terra` workers. These assignments organize already
approved work; they grant no approval to start a work package or cross an
authorization gate.

### Sol orchestrator responsibilities

Sol is the sole integrator. For each approved work package or slice, Sol:

1. confirms the authorization and freezes the exact scope, allowed files,
   contracts, tests, and stop conditions;
2. assigns bounded read-only, tests-only, or implementation-only Terra tasks;
3. reconciles worker findings and decides any interface or ownership question;
4. reviews meaningful RED evidence and commits the tests before authorizing an
   implementation task;
5. reviews all worker diffs for scope and behavior changes before integration;
6. owns shared entry points, package initializers, compatibility forwards,
   shared fixtures, and final documentation unless it explicitly assigns one
   of them to a single worker; and
7. runs affected integration tests, the required complete suite, performance
   checks, and final acceptance review.

Sol does not delegate user decisions about public compatibility, deletion,
scientific meaning, or external actions.

### Terra worker task contract

Each Terra task states:

- one concrete objective and work-package identifier;
- exact files or responsibility group in scope;
- whether the task is read-only, tests-only, or implementation-only;
- contracts and behavior that must remain unchanged;
- commands to run and the expected RED or GREEN result;
- files and actions that are forbidden; and
- the required handoff: inspected and changed files, findings, test output,
  unresolved questions, performance evidence when applicable, and any scope
  pressure.

A tests-only worker does not edit production code. An implementation worker
starts only after Sol has accepted and committed the corresponding RED tests.
A worker stops rather than widening its scope, resolving an authorization
question, or changing a public or scientific contract.

### Parallelism and file ownership

Parallel Terra work is limited to independent read-only audits or disjoint
files whose interfaces Sol has already frozen. Agents do not concurrently edit
the same source module, root compatibility module, package `__init__.py`, test
fixture, or documentation file. When a source such as
`unit_spike_plotting.py`, `psth_webapp.py`, or an LFP-summary runtime is shared
across destinations, its mutations are serialized through one owner. Sol may
run up to three read-only audits concurrently, but ordered tests-first and
implementation steps remain sequential.

### Work-package task map

| Work package | Sol task | Terra worker tasks |
| --- | --- | --- |
| R0 | Freeze the baseline; reconcile classifications; write and obtain review of the inventory. | `R0-T1`: scientific-domain modules; `R0-T2`: LFP-summary, saved artifacts, process workers, and profiling; `R0-T3`: metadata, webapp, commands, compatibility, and deletion candidates. These are parallel read-only audits that return inventory rows and import edges. |
| R1 | Freeze synchronization ownership and integrate the root entry/forwards. | Sequential `R1-tests` and `R1-move` tasks for the approved synchronization files. |
| R2 | Approve and integrate one scientific slice at a time. | For each of R2A-R2D, one tests-only task followed by one implementation task. Different slices may be audited in parallel but do not mutate the shared tree concurrently. |
| R3 | Own the shared plotting source and compatibility surface. | Domain-specific plotting inventories may run in parallel; tests and moves are serialized by destination because `unit_spike_plotting.py` is shared. |
| R4 | Own saved-artifact compatibility and integrate R4A-R4D in order. | One tests-only/implementation pair per slice. R4C stays with one worker context at a time because planning, workers, checkpoints, and reduction share invariants. |
| R5 | Freeze `app.py`, session-input, loading-cache, and root-entry contracts; integrate the app. | After those interfaces are frozen, disjoint unit, LFP, Spike-LFP, population, and summary-view tasks may run independently. Only Sol or one assigned integration worker edits shared app/input/loading/root files. |
| R6 | Freeze root command behavior and integrate launcher, relocation, and profile paths. | R6A-R6C receive separate bounded tests/implementation tasks. Read-only audits may overlap; mutations to shared `lfp_summary` package surfaces are serialized. |
| R7 | Reconcile fresh caller audits, present deletion decisions to the user, and own final acceptance. | Caller audits may run in parallel. Approved wrapper or unused-code deletion is performed by one bounded worker task only after the specific deletion approval. |

This map is intentionally small. It does not create permanent agent roles,
extra review layers, or a task for every helper function.

## 9. Work package R0 - Baseline and supported-surface inventory

### Purpose

Record the exact baseline needed for safe movement. R0 changes no production
code.

### Inventory

For every current module, record:

- repository importers and command invocations;
- directly associated tests;
- documented user-facing functions and commands;
- saved formats, filenames, and public records it owns;
- whether it is production, exploratory, infrastructure, profiling, or an
  unused candidate; and
- whether an external-use decision is required.

For a module that will be split, also inventory its top-level functions and
classes by responsibility. Record individual symbols when they are public,
serialized, used as multiprocessing workers, imported privately across
modules, test monkeypatch seams, or apparently unused. Do not create a catalog
of every ordinary private helper in a cohesive module.

For the compact metadata path, explicitly classify:

- `ChannelGroupMetadata`, `PopulationMetadata`, and the derived
  `channel_groups` and `populations` properties;
- duplicate identity and display-label properties on resolved records;
- always-absent compatibility fields such as `session_date`,
  `lfp_summary_snapshot_directory`, and `treadmill_file`;
- aliases around the single version-2 alignment and cache fields; and
- `MetadataPopulationInputs` and `metadata_population_inputs()`, which have
  test callers but no production caller at the planning baseline.

The inventory distinguishes useful semantic aliases from version-1 scaffolding.
A passing test alone does not make a test-only adapter public.

For the current LFP configuration records, explicitly classify
`AnalysisWindowConfig`, `FrequencyBandConfig`, and `PowerAnalysisConfig` for
repository and possible external imports, module-qualified serialization,
configuration JSON, and fingerprint identity. R0 must confirm whether aliases
from `lfp_summary_models.py` are sufficient before R2A changes their canonical
module identity.

Classify `sync_ephys.py` as a current root script/function entry point. Record
its callable signatures plus the workflow selected by its `__main__` block.
Its hardcoded session, recording, and sorter paths are replaceable local
examples, not strict compatibility contracts. It has no argument parser or
documented CLI contract.

R0 also records the current internal dependency graph for
`src/neural_analysis` with a small repository or standard-library AST audit.
The inventory identifies current cycles and back-imports, including the
LFP-summary runtime/PPC-runtime dependency and the current reusable-LFP to
`lfp_summary_models` to LFP-loading ownership inversion, and compares every
edge with the approved dependency direction. Do not add a graph-analysis
dependency.

The inventory uses the module disposition in the design document as its
starting point, not as proof that a move is safe.

The reviewed R0 deliverable is
`docs/neural_analysis_refactor_inventory.md`. It contains one row per current
module plus grouped or individual symbol entries where required, including:

- support category and current responsibility;
- documented commands and public imports;
- serialized or process-bound identities;
- primary callers and tests;
- approved destination; and
- current internal import edges, known cycles or back-imports, and their
  intended resolution owner; and
- external-use or deletion gates.

### Baseline verification

Before R1 tests are written:

- record the exact Git commit and tracked worktree state;
- run the complete `src/tests/neural_analysis` suite;
- run the complete repository suite if shared non-neural modules are imported
  by the first move;
- save and review the baseline internal import graph and all edges that violate
  the approved dependency direction;
- record existing warnings rather than repairing them implicitly; and
- identify any user-owned worktree changes that must remain untouched.

### Completion

R0 is complete when every current neural module has a support classification
and every mixed-responsibility module has an adequate callable classification,
the dependency baseline and inventory document are reviewed, and the first R1
edit/caller/test list is approved.

## 10. Work package R1 - Synchronization organization

### Scope and architecture

Move the relatively self-contained synchronization responsibilities first:

- `spikeglx_sync_io.py` digital-line reading;
- reusable mapping/alignment behavior from `ephys_sync_utils.py`;
- `manual_session_synchronization.py`;
- the existing `sync_ephys.py` root script and callable workflows.

Create only the `synchronization` modules needed by those current
responsibilities. Keep `sync_ephys.py` as a thin root script/function entry
point.

### Tests written and committed first

- new canonical imports do not yet exist and therefore produce the expected
  RED;
- old and new functions map identical samples/timestamps;
- time zones, bounds, extrapolation warnings, and missingness are unchanged;
- written synchronization NPZ/note content is equivalent;
- Open Ephys and SpikeGLX current paths retain their existing behavior;
- the callable workflow signatures, return values, writes, and dependency
  calls are unchanged;
- tests do not freeze hardcoded session, recording, or sorter example paths;
- direct module execution selects the same current hardcoded workflow when its
  real-data dependencies are stubbed; and
- old import paths forward to the new implementations.

### Implementation constraints

- Do not add another synchronization method or source format.
- Do not add an argument parser or turn the hardcoded root script into a new
  configurable CLI; that would be a separate feature.
- Do not redesign IRIG code outside `src/neural_analysis`.
- Do not combine synchronization cleanup with a change in time semantics.
- Split source-specific I/O from reusable mapping only where the current code
  already exposes that boundary.
- Do not create `synchronization/treadmill.py`. Current repository evidence
  gives `analog_treadmill_decode.py` no production synchronization caller;
  retain it unchanged for the R0/R7 external-use and deletion audit.

### Performance and acceptance

No benchmark is required. The move must not add a second read of large source
files or materialize arrays that are currently streamed or selected. Focused
synchronization tests and the complete neural suite must pass.

### Implementation result

R1 tests were committed first in `b7696aa`, with six expected RED failures for
the absent canonical package and 59 passing focused tests. The implementation
was committed in `6588fc5`. It adds only `synchronization/alignment.py`,
`synchronization/spikeglx.py`, `synchronization/manual.py`, and a minimal
initializer; the three old modules are compatibility aliases, while
`sync_ephys.py` remains the root script/function entry.

The final focused synchronization and LFP-loading run passed 76 tests. The
complete neural suite passed 1,676 tests with the 22 previously inventoried
warnings. A complete repository run was attempted but could not collect the
unrelated behavior-analysis suite because `autograd` is absent from the local
environment. No dependency was added, and no experimental data, cache,
report, external system, or scheduler was accessed.

## 11. Work package R2 - Scientific-domain organization

R2 is divided into independently approved slices. Do not migrate all domains in
one commit.

R2 moves loading, calculation, and existing persistence responsibilities.
Plotting that is currently mixed into these modules remains behind its current
path until R3, so each plotting responsibility moves only once.

### R2A - LFP

Move or split:

- the `AnalysisWindowConfig`, `FrequencyBandConfig`, and
  `PowerAnalysisConfig` records currently in `lfp_summary_models.py`;
- `lfp_loading.py`;
- `lfp_power_summary.py`;
- `lfp_spectrogram.py`;
- `lfp_phase_clustering.py`; and
- `lfp_synchrony_summary.py`.

Move those three records first to the small `lfp/config.py` module so reusable
LFP calculations do not import the LFP-summary workflow. Preserve the current
`lfp_summary_models.py` names as aliases to the same class objects; do not copy
or redefine the records. If R0 finds a module-qualified serialized or public
contract that aliases do not preserve, stop and define the smallest explicit
compatibility mechanism before movement. Keep LFP-summary-wide validation in
the workflow models; do not create a general configuration framework.

Loading, reusable numerical calculations, and result persistence are separated
only where currently mixed. Open Ephys and SpikeGLX loading may remain together
in one clearly sectioned module; separate adapter subpackages are not required.
Plotting movement is deferred to R3.

Tests written first:

- canonical `lfp` imports produce RED before the move;
- canonical LFP configuration imports produce RED, while compatibility tests
  require the old names to resolve to the same class objects after the move;
- configuration defaults, equality, JSON, fingerprints, and any inventoried
  serialized representation are exact;
- current loader values, sample indices, sample rates, voltage units, and time
  coordinates are exact;
- Power, spectrogram, phase, relative-phase, PLV, and Synchrony arrays are
  exact or retain their existing reviewed tolerance;
- saved result behavior is unchanged;
- no module under `lfp` imports `lfp_summary`; and
- old module imports forward correctly.

### R2B - Spike behavior

Split current responsibilities from:

- `spike_behavior_pynapple.py`;
- `unit_spike_loading.py`; and
- `psth_behavior.py`.

The target responsibilities are loading, trial classification, binning,
decoding, PSTH computation, and existing table publication. Plotting movement
is deferred to R3. Do not introduce a mutable general-purpose session object
or merge metadata models with numerical data.

Tests written first:

- canonical `spike_behavior` imports produce RED;
- sorter, aligned-spike, channel-quality, and behavior loading is equivalent;
- trial masks, selected units, bin edges, spike/lick counts, decoder inputs,
  seeded results, and saved tables are unchanged;
- PSTH arrays and plot inputs are unchanged; and
- old import paths remain compatible.

### R2C - Spike-LFP

Move numerical responsibilities from:

- `spike_lfp_hilbert_phase.py`;
- `spike_lfp_phase_locking.py`;
- `spike_lfp_summary.py`; and
- `lfp_summary_ppc_kernel.py`.

Tests written first:

- canonical `spike_lfp` imports produce RED;
- Hilbert phases, wavelet samples, phase bins, firing rates, PPC values,
  permutation summaries, FDR flags, and exemplar selections are unchanged;
- deterministic schedules and seeds are exact;
- kernel allocation estimates and output ownership/read-only contracts remain
  unchanged; and
- old import paths forward correctly.

Run the existing synthetic/injected representative PPC kernel workload before
and after this slice. Do not load CT026 or another real session without a
separate authorization.

### R2D - Population and cross-session analysis

Move or split:

- `population_pca.py`;
- `population_pca_decoding.py`;
- `population_pca_switch_trajectories.py`; and
- `plot_cross_session_analysis.py`.

Move its cross-session selection, aggregation, naming, and publication
responsibilities in R2D. Leave its plotting functions behind the current path
until R3.

Tests written first:

- canonical `population` imports produce RED;
- trial/unit tensors, PCA inputs/results, decoded targets, folds, seeded
  results, switch classifications, and trajectory arrays are unchanged;
- cross-session date selection, aggregation, filenames, table columns/order,
  missingness, and overwrite behavior remain unchanged; and
- old imports remain compatible.

### R2 completion

Each slice receives focused and affected integration tests plus the complete
neural suite. R2 is complete only when all four accepted slices are green and
their package boundaries do not introduce cycles back to workflows or UI.

## 12. Work package R3 - Plotting organization

### Scope and architecture

Split plotting by the scientific result being inspected. The main source is
`unit_spike_plotting.py`, supplemented by plotting currently embedded in LFP,
Spike-LFP, spike-behavior, population, and cross-session modules.

Plots move beside their scientific domain:

- unit/trial/PSTH plots to `spike_behavior`;
- LFP power/spectrogram/phase plots to `lfp`;
- Spike-LFP plots to `spike_lfp`;
- PCA/decoding/trajectory/cross-session plots to `population`; and
- production summary plots remain in `lfp_summary`.

Small numerical helpers used only to prepare one figure may stay with that
plot. Reusable scientific calculations move to the relevant analysis module.

### Tests written and committed first

- new plotting imports produce RED;
- returned figure/axes structures are unchanged;
- plotted numerical data, panel counts, axis labels, titles, legends,
  annotations, and captions are unchanged;
- white-background and readability contracts remain unchanged;
- filenames and save behavior remain unchanged; and
- old plotting imports forward correctly.

Use existing structural/content assertions and rendering smoke tests. Do not
add brittle pixel-for-pixel snapshots when the current contract is structural.

### Acceptance

No plot, style, control, statistic, caption, or report content changes. Focused
plot tests, affected webapp/report tests, and the complete neural suite pass.

## 13. Work package R4 - LFP-summary workflow organization

R4 has the highest scientific and saved-artifact risk. It begins only after R2
and R3 establish stable numerical and plotting locations. Completing it before
the webapp decomposition prevents the summary view from being moved twice.

### R4A - Stable workflow foundations

Move without semantic changes:

- `lfp_summary_models.py`;
- `lfp_summary_session.py`;
- `lfp_summary_preparation.py`;
- `lfp_summary_pipeline.py`;
- `lfp_summary_payloads.py`;
- `lfp_summary_io.py`; and
- `lfp_summary_work_cache.py`.

`lfp_summary_session.py` should consume version-2 probe records directly. Its
internal configuration path must not reconstruct explicit population or
channel-group metadata that the compact schema removed. Preserve any
inventoried public request signature, but treat its selected population ID as
the current probe ID and use the probe's optional `unit_channels` directly.

Tests written first:

- canonical `lfp_summary` imports produce RED;
- canonical configuration JSON and component fingerprints are exact;
- source-value semantics and source fingerprints are exact;
- payload schemas, dtypes, shapes, axes, and read-only ownership are exact;
- manifest/component status and transactional publication are unchanged;
- work-cache identity, checkpoint validation, cleanup targets, and resume
  behavior are unchanged;
- metadata-derived configurations are equivalent;
- version-2 probe selection, optional `unit_channels`, inferred source
  sidecars/children, and existing quality-selection rules are unchanged;
- no version-1 metadata reader or population/channel-group model is restored;
  and
- legacy saved configurations and snapshots remain readable.

### R4B - Component runtime split

Split `lfp_summary_runtime.py` into:

- the smallest shared preparation/runtime code;
- Power runtime;
- Synchrony runtime; and
- Spike-phase runtime.

Move functions by current component responsibility. Do not redesign
`PipelineDependencies`, component payloads, or public computation entry points.
Break the current runtime/PPC-runtime cross-dependency by relocating only the
already shared validation or preparation function to its natural owner, not by
adding a new interface framework.

Tests written first:

- new component runtime imports produce RED;
- prepared trial, Power, phase, and spike records are exact;
- component payload arrays and metadata are exact;
- progress events and cleanup targets are unchanged;
- composed and component-specific dependency factories behave identically;
- synthetic compute/cache/reload/plot integration is unchanged; and
- old runtime imports forward correctly.

### R4C - PPC planning and execution

Split `lfp_summary_ppc_runtime.py` only at its established planning/allocation
versus execution/checkpoint boundary. Keep serial and parallel execution paths
together unless the focused review identifies a smaller current boundary that
improves readability without expanding the interface.

Tests written first:

- new planning/execution imports produce RED;
- job/component plans, schedule shapes, fingerprints, and allocation estimates
  are exact;
- serial and parallel summaries are exact;
- checkpoint schemas, resume selection, staged results, locks, and cleanup are
  unchanged;
- seeds, unit blocks, worker bounds, progress, and error behavior are exact;
- top-level worker callables remain importable and pickle-safe under a real
  process-spawn smoke test; and
- old PPC runtime imports forward correctly.

Run the same existing synthetic/injected representative PPC workload before
and after the split and record runtime and peak memory. A CT026 or other
real-session profile requires separate authorization.

### R4D - Validation, reports, snapshots, and plotting

Move:

- `lfp_power_validation.py`;
- `lfp_synchrony_validation.py`;
- `lfp_spike_phase_validation.py`;
- `lfp_summary_plotting.py`; and
- non-Streamlit snapshot inspection from `lfp_summary_webapp.py`.

Do not create a generic report package. Split report construction from
validation only when the current module has a clear internal boundary and the
split shortens the scientific path a reviewer follows.

Tests written first:

- canonical imports produce RED;
- report JSON, markdown content, filenames, figure selection, and publication
  validation are unchanged;
- cached result selection and provenance are unchanged;
- snapshot validation, component/population selection, and failure messages
  are unchanged;
- plotting structures and counts remain unchanged; and
- historical approved artifacts remain readable and immutable.

### R4 completion

Run focused tests after each slice, the full LFP-summary/Power/Synchrony/Spike/
PPC/launcher/webapp integration set after R4D, and the complete neural suite at
R4 completion.

## 14. Work package R5 - Webapp decomposition

### Scope and architecture

Make `psth_webapp.py` a thin documented entry point over direct domain view
modules after the LFP-summary and snapshot seams are stable. The intended
responsibilities are:

- `webapp/app.py`: top-level view selection and composition;
- `webapp/session_inputs.py`: direct version-2 probe/site inputs and the
  separately preserved legacy manual inputs, without a parallel metadata view
  model;
- `webapp/data_loading.py`: existing Streamlit-cached loading seams;
- domain view modules for unit, LFP, Spike-LFP, and population views; and
- `webapp/summary_view.py`: existing LFP-summary controls and rendering over
  the R4 snapshot and workflow interfaces.

This is not a router, page framework, registry, or state architecture. A small
explicit branch in `app.py` is acceptable and preferred when easiest to read.

The metadata branch passes the selected probe, sites, alignment path, and
channel restriction directly to its owning view. It must not route the first
two metadata probes through legacy PFC/HPC variable names. The current manual
legacy branch remains separate. Do not move `MetadataPopulationInputs` or
`metadata_population_inputs()` into the new package if R0 confirms that they
remain compatibility-only/test-only; propose their deletion through the
cleanup gate instead.

### Tests written and committed first

- canonical webapp imports produce RED;
- every current view remains selectable under the same conditions;
- controls pass the same values to the same scientific functions;
- metadata and legacy launch paths remain supported;
- the metadata route retains its current `Probe / region`, manual-channel,
  channel-quality-label, and inside-brain controls;
- an optional probe `unit_channels` restriction intersects the same quality-
  approved channels;
- probe IDs remain distinct from probe-owned site names and saved-channel
  indices;
- internal version-2 adapters use one session identity rather than modeling
  independent subject and session fields;
- existing visible labels and captions remain unchanged unless a separate UI
  change is approved;
- cached and live provenance remain distinct;
- missing inputs disable the same affected views with the same meaning;
- metadata validation and metadata/summary-source resolution load no large
  numerical arrays, while the default Unit raster/PSTH render retains its
  current selected-population spike loading;
- unrelated rerenders add no new eager numerical loading compared with the
  baseline;
- expensive existing actions occur only after the same explicit user action;
- Streamlit cache inputs and invalidation semantics remain equivalent; a
  one-time in-memory cache miss caused by deployment of moved code is allowed;
- the root `psth_webapp.py` command and parser behavior are unchanged; and
- inventoried public helper paths forward where required, while repository-
  private callers migrate to the canonical owner.

### Performance and acceptance

Run the existing noninteractive startup smoke before and after the move. No new
cache layer, background worker, eager preprocessing, or page capability is
added. Focused webapp tests, directly used domain tests, summary integration
tests, and the complete neural suite pass.

## 15. Work package R6 - Launcher, relocation, and profiling

### R6A - Spike-phase launcher

Keep `lfp_spike_phase_launcher.py` as the documented root command. Move only
established internal responsibility groups under `lfp_summary`:

- request parsing and top-level dispatch remain at the entry point;
- retained-work and source preflight;
- saved launcher-state and report-recovery lifecycle; and
- execution/resource measurement.

The exact split is frozen only after a caller and helper-group review. Do not
create generic execution, run-state, filesystem-safety, or resource packages.

Tests written first:

- new internal imports produce RED;
- `new`, dry run, final intent, resume, recover-report, and rerender-report
  commands parse identically;
- metadata and legacy session inputs build equivalent requests;
- preflight, retained work, locking, signal handling, and cleanup ordering are
  unchanged;
- state transitions and persisted JSON/log/summary files are equivalent;
- failure preserves the same recovery inputs;
- source/configuration/report identity checks are unchanged;
- resource measurements retain their current meanings; and
- the root command remains unchanged.

### R6B - Cache relocation

Move `lfp_summary_cache_relocation.py` under `lfp_summary` while preserving its
inventoried public import and command forwards through migration.

Tests written first:

- canonical import produces RED;
- path validation, source/destination identity, copy behavior, rollback,
  checksums, receipt content, and failure cleanup are unchanged; and
- old imports/commands remain compatible.

No external copy is run as part of the refactor.

### R6C - Profiling and CT026 bindings

Move these modules initially to like-named modules under `lfp_summary`:

- `lfp_summary_ppc_profile.py`;
- `lfp_summary_ct026_profile_adapter.py`;
- `lfp_summary_ct026_profile_locks.py`; and
- `lfp_summary_ct026_profile_runner.py`.

Keep CT026 explicit in names and behavior. Do not consolidate the adapter,
locks, and runner unless a separate R6 review identifies actual duplication
and a clearer result.

Tests written first:

- canonical profile imports produce RED;
- representative job selection, slice identity, locks, recovery, measurements,
  and profile documents are unchanged;
- CT026 bindings remain isolated from generic defaults; and
- old imports forward correctly.

### R6 completion

Run focused launcher, relocation, profiling, shell-wrapper, LFP-summary, and
complete neural suites. No real-data run, cache mutation, external copy, or
Slurm action is authorized.

## 16. Work package R7 - Compatibility cleanup and unused-code deletion

### R7A - Compatibility audit

After all repository callers use canonical imports:

- inventory every root forwarding module;
- search source, tests, shell scripts, notebooks, and documentation;
- identify likely external imports with the user;
- retain the two documented root command modules and the current
  `sync_ephys.py` root script; and
- propose each other wrapper removal separately.

No wrapper is removed merely because repository tests no longer import it.

### R7B - Unused candidates

The planning audit found no repository consumer for:

- `behavior_pynap.py`;
- `spike_behavior_analysis.py`;
- `spike_behavior_binning.py`;
- `modified_sinc_smoother.py`; and
- `plot_single_session_analysis.py`.

Audit `analog_treadmill_decode.py` separately. `volts2speed` has a repository
test but no production caller, while `analyze_treadmill_signal`,
`wavelet_hard_threshold`, and `robust_spectral_subtraction` have no repository
callers. This evidence is not yet enough to delete the module because external
use has not been ruled out.

Immediately before deletion:

1. repeat exact module-name, filename, distinctive-symbol, command, notebook,
   dynamic-import, and package-entry-point searches;
2. verify that any useful-looking responsibility is either obsolete or already
   covered by a retained tested implementation;
3. obtain user confirmation that no script outside the repository depends on
   the module;
4. add and commit a removal test that is RED while the approved obsolete
   modules remain present; and
5. obtain explicit deletion approval.

Delete approved unused modules rather than moving them to `legacy`. Git history
retains their source. Do not delete the active and heavily used
`spike_behavior_pynapple.py`; its similar name is not evidence that it is a
candidate.

### R7C - Final documentation

- Update `src/neural_analysis/README.md` only for canonical Python imports or
  navigation that an end user needs.
- Update the design and this plan to reflect the realized, not merely proposed,
  tree.
- Add a concise package guide only where multiple interacting modules require
  one; do not require a README in every directory.
- Record removed compatibility paths and approved deletions.
- Keep execution history in the execution log rather than expanding the active
  plan again.

### R7 acceptance

Run the complete neural and repository suites. Review the final tree from the
perspective of a scientist locating each supported calculation, plot, webapp
view, and production workflow. The user explicitly approves completion even if
some large cohesive modules remain.

## 17. Behavior-preserving simplification candidates

Simplification is allowed only when it reduces code or control flow while
preserving an inventoried contract. It is performed inside the work package
that owns the affected responsibility and follows the same tests-first rules as
movement.

Current candidates are:

1. Delete modules and top-level callables proven unused through R0/R7. This
   includes auditing the unreferenced signal-analysis functions in
   `analog_treadmill_decode.py`, not only the five current module candidates.
2. Retire version-1-shaped metadata adapters after R0 classifies them. Likely
   candidates are the derived population/channel-group records and properties,
   duplicate identity/display-label aliases, always-absent fields, and the
   test-only `MetadataPopulationInputs` path. Preserve a semantic alias only
   when it still makes an active caller clearer.
3. Replace the metadata webapp's first/second-probe-to-PFC/HPC bridge with
   direct version-2 probe/site inputs during R5. This is an internal routing
   simplification; it does not retire the separately supported manual legacy
   route.
4. Review the identical region-name normalization currently present in
   `spike_behavior_pynapple.py` and `plot_cross_session_analysis.py`, while
   preserving the cross-session filename builder's distinct `None` behavior.
   Prefer the small duplication over a generic naming module or an otherwise
   unnecessary cross-domain dependency; consolidate only if R2 establishes a
   natural existing owner for both callers.
5. Consolidate overlapping sorter, aligned-spike, channel-quality, and unit-
   selection loading after R2B establishes one tested owner. Do not introduce
   an all-purpose session object.
6. Remove pass-through wrappers that add no validation, caching boundary,
   compatibility value, or meaningful scientific name.
7. Compare repeated validation/report helpers such as planned-run paths, plot
   context, JSON writing, directory/component size, peak memory, and source
   identifiers. Extract only functions with identical data, filesystem, and
   failure contracts; keep merely similar report logic separate.
8. Remove the `lfp_summary_ppc_runtime` back-import of
   `lfp_summary_runtime` by moving the already shared validation/preparation
   function during R4B.
9. Remove compatibility modules after the R7 audit instead of retaining a
   permanent duplicate navigation surface.
10. Propose retirement of the legacy webapp path-entry branch only after the
   user confirms that metadata-driven launch fully replaces that existing
   capability. Because that removes supported behavior, it is a separate
   approval gate rather than an automatic R5 cleanup.
11. Remove an older PPC executor, CT-specific adapter, profile path, or legacy
   CLI only when the inventory proves it has no reproduction, resume,
   profiling, test, or external caller.

Do not create a generic utility, atomic-writer, report, artifact, or execution
module solely to reduce repeated lines. Similar safety-sensitive functions may
have intentionally different contracts. A proposed extraction must name at
least two current callers and demonstrate that one direct implementation is
easier to review.

## 18. Verification matrix

| Change type | Minimum focused verification |
| --- | --- |
| Pure import move | New-location RED test, old/new import compatibility, owning module tests |
| Numerical function move | Exact deterministic results or existing reviewed tolerance; shapes, axes, units, missingness |
| Loader move | Same paths read, same records/arrays, same lazy/eager behavior |
| Plot move | Same plotted data, structure, labels, captions, filenames, rendering smoke |
| Webapp move | Same controls/actions and current view-specific loading behavior, metadata/legacy routing, cached/live provenance |
| Cache/artifact move | Same schemas, identities, publication, validation, and recovery |
| PPC move | Numerical equivalence, seeds/schedules/checkpoints, serial/parallel behavior, runtime and peak memory |
| CLI/launcher move | Same parsing, state transitions, failure/recovery behavior, command path |
| Root script move | Same `__main__` dispatch, callable signatures, return values, and stubbed dependency calls; hardcoded local example paths are not frozen; no parser added |
| Deletion | Fresh no-caller audit, external-use confirmation, RED removal test, explicit approval |

After each accepted work package:

- focused tests pass;
- directly affected integration tests pass;
- the complete neural suite passes;
- affected internal import edges follow the approved dependency direction and
  introduce no new cycle;
- no unrelated user files changed; and
- no unauthorized real-data or external-state action occurred.

Run the complete repository suite after any change to shared non-neural code
and at final R7 acceptance.

## 19. Stop and replan conditions

Stop the active work package and return for user direction if:

- a move requires a public behavior change;
- an existing saved artifact cannot be read through the proposed organization;
- exact or reviewed-tolerance numerical equivalence fails;
- a proposed abstraction has only one caller and a direct function is clearer;
- the change begins supporting a new source layout or edge case;
- the work requires a new dependency;
- the diff expands into another scientific domain not listed in the approved
  slice;
- a candidate for deletion has a repository or possible external consumer; or
- a material PPC runtime or memory regression appears.

## 20. Authorization gates

U4 acceptance was satisfied on 2026-09-25 and is no longer an open gate.
The following approvals are separate:

1. start of the R0 inventory;
2. start of R1 implementation;
3. start of each later R work package or slice after review of its exact edit
   and tests-first list;
4. deletion of each compatibility or unused module;
5. any real-session numerical run or cache/report mutation;
6. any external copy; and
7. any Slurm submission, monitoring, resume, cancellation, or result
   inspection.

Approval of this document grants none of those later actions.
