# Neural Analysis Refactor Implementation Plan

## Status and authority

**Current status:** NR0 is complete. The corrected CT026 Power and Synchrony
artifacts and the final presentation-only Synchrony report are approved. The
corrected Power/Synchrony cache was transferred to the cluster, its metadata-
only launcher dry run passed, and the user authorized the reviewed
100-shuffle ProbeB preview. Slurm accepted that submission as job `30766437`.
The job is intentionally unmonitored; status or result inspection requires a
new user request. The execution history, exact paths, hashes, reviews, and
authorization boundaries are recorded in
`docs/neural_analysis_refactor_execution_log.md`.

This plan is authoritative for future work. It supersedes the former mandatory
NR2-NR18 package topology and wave sequence. Git history retains that design,
but it is not an implementation requirement.

Future work has two strictly ordered goals:

1. **End-user usability for new sessions.** A user points one session metadata
   file at the correct behavior, synchronization, LFP, spike, and optional
   cache sources for each probe. The same metadata launches the webapp and
   supplies an easy entry point for the existing large cluster computations.
   A concise guide documents the principal commands and Python functions.
2. **Scientific code readability, only after usability is accepted.** Later
   work may reorganize existing capabilities so a scientist can understand the
   code more easily. It may split or move existing code, clarify names and
   contracts, and retain compatibility forwards. It must add no functionality.

The second goal is explicitly deferred until the user has exercised and
approved the usability workflow. Supporting more complex cases of any kind is
forbidden in both goals unless the user replaces this plan. That prohibition
includes new acquisition families, metadata-placement modes, execution
backends, artifact formats, analysis methods, UI capabilities, recovery
systems, plugin systems, generalized frameworks, and hypothetical concurrency
or hostile-filesystem defenses.

`docs/neural_analysis_refactor.md` remains authoritative for scientific
intent, `docs/Tasks_neural.md` remains authoritative for the established
analysis methods, and `docs/SoftwareDesign.md` governs implementation style.
If they conflict with this simplified scope, stop for user direction rather
than expanding the implementation.

Real-data computation, cache mutation, external copying, Slurm submission,
and deletion remain separate authorization gates. Documentation edits and Git
pushes grant none of those authorities.

### Live implementation handoff

- Branch `refactor` is pushed through documentation commit `19befe2`.
- Cluster code remains pinned at the reviewed tracked-clean commit `5cc1385`.
- Slurm job `30766437` was submitted once and is not being monitored.
- The next allowed action concerning that job is a user-requested, bounded
  status or result inspection.
- No usability implementation starts until this rewritten plan is reviewed
  and explicitly approved.
- The user's existing `docs/SoftwareDesign.md` modification is unrelated and
  remains untouched and unstaged by this work.

## 1. Objectives

### 1.1 Immediate objective: simple use with new sessions

Create one small metadata-driven path through the capabilities that already
exist. A user must be able to:

1. create or edit `<session_home>/neural_session.json`;
2. identify the mouse/session and the behavior inputs;
3. identify, for each probe, its acquisition family, LFP source, required
   synchronization source, spike sorter/aligned-spike sources, optional
   channel-quality source, sites, and populations;
4. validate those paths and relationships without loading large arrays;
5. launch the existing webapp from that metadata and use the existing bounded
   interactive and cached-result views without editing Python constants;
6. dry-run and hand off the existing large LFP-summary computation locally or
   through the existing Slurm wrapper from the same resolved metadata; and
7. find short instructions for the supported commands, outputs, recovery
   steps, and important Python functions.

The metadata describes identity and sources. Scientific presets, frequency
definitions, seeds, shuffle methods, cache schemas, and execution algorithms
remain in reviewed code.

### 1.2 Deferred objective: readability-only reorganization

After the usability objective is complete and user-approved, existing code may
be reorganized for scientific readability. The only valid reasons are to make
an existing calculation, source boundary, plot, or workflow easier for a
scientist to locate and understand. Existing behavior remains the reference.

The deferred work must not add features, broaden supported cases, redesign the
webapp, invent generic infrastructure, or require completion of an idealized
package tree. Large modules may remain large when splitting them would not
materially improve end-user scientific readability.

### 1.3 End-user documentation objective

`src/neural_analysis/README.md` becomes the single concise operational guide.
It answers which metadata file to edit, how to validate it, how to launch the
webapp, how to start a dry run or cluster computation, how to identify outputs,
and which small set of Python functions are intended for direct scientific use.
Internal architecture documentation must not compete with that guide.

## 2. Current baseline and constraints

### 2.1 Relevant current architecture

The production LFP-summary path is currently distributed across these
responsibilities:

- `lfp_loading.py` reads SpikeGLX and derived Open Ephys LFP data.
- `lfp_summary_models.py` owns immutable analysis configuration and component
  fingerprints.
- `lfp_summary_preparation.py` loads trial-aligned native-rate traces.
- `lfp_summary_runtime.py` prepares Power, Synchrony, and Spike-phase inputs and
  contains a separate continuous block loader for phase transforms.
- `lfp_summary_io.py`, `lfp_summary_work_cache.py`, and
  `lfp_summary_pipeline.py` own final component state, work caches, and
  publication.
- CT026 builders and reports remain in the validation and profile modules.
- `psth_webapp.py` and `lfp_summary_webapp.py` mix UI composition with source or
  artifact routing.

Both Open Ephys numerical routes converge on
`read_open_ephys_lfp_channel_window`: trial loading reaches it through
`load_open_ephys_trial_lfp_trace`, while continuous phase preparation calls it
from `_production_phase_block_loader_factory`. The scaling correction should
therefore be made at this lowest shared Open Ephys value boundary.

`src/tests/neural_analysis/test_get_brain_channels.py` is retained in the full
neural-suite gate but is not assigned a migration owner: despite its location,
it tests `src/external_tools/get_brain_channels.py`, which is outside this
`src/neural_analysis` refactor. If a new source adapter begins to consume that
tool, the active package must first amend its allowlist and dependency contract.

### 2.2 Verified focused baseline

Before this draft, the following read-only baseline passed:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_open_ephys_behavior_integration.py \
  src/tests/neural_analysis/test_lfp_loading.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_preparation.py

99 passed
```

The implementation phase must record the exact commit and rerun the applicable
focused and full-suite baselines because the repository may advance after this
draft.

### 2.3 Non-negotiable implementation rules

- Follow RED-GREEN-REFACTOR and commit focused tests before implementation.
- Preserve existing scientific values, shapes, axis order, sampling rates,
  units, channel indices, time coordinates, seeds, and output meanings.
- Preserve public call signatures or add a thin compatibility-preserving
  extension that is covered by existing behavior tests.
- Run project Python commands with `uv run`.
- Add no runtime dependency for this work.
- Never overwrite approved caches, reports, run directories, or historical
  evidence.
- Do not combine code movement with numerical or scientific changes.
- Complete and obtain user approval for the usability milestones before any
  readability-only reorganization.
- Do not implement support for more complex cases. A need for a new source
  layout, acquisition family, metadata-placement rule, analysis, output type,
  UI mode, execution backend, or recovery mode stops the milestone for user
  direction.
- `src/neural_analysis/README.md` is the end-user landing page. Add a package
  README only when a real package needs internal explanation; do not create
  READMEs to satisfy structural symmetry.

## 2.4 Design guidance for the simplified work

### Direct user paths before architecture

- Begin with the three user actions: describe a session, open the webapp, and
  start a large computation.
- Prefer plain functions and frozen data records. Do not create a mutable
  all-purpose session object.
- Keep metadata, path resolution, and action validation together until their
  size or use demonstrates a need to split them.
- Reuse the existing webapp, analysis functions, cache code, launcher, and
  Slurm wrapper. A new layer is justified only when it removes duplicated
  session-specific path selection from at least two of those real callers.
- Use direct dispatch for the small known set of components. No analysis
  registry, plugin discovery, base-analysis hierarchy, or workflow engine.
- Use a Protocol only when two current implementations genuinely require the
  same behavior and plain callables would be less clear.

### Scientific readability and reproducibility

- Public scientific functions document input types, shapes/axes, physical
  units, missingness, return values, and important errors.
- Metadata and adapter functions document whether each path names a file or a
  directory and whether it is required or optional.
- Randomness remains explicit and recorded. Scientific defaults remain in
  versioned code rather than user metadata.
- Preserve established NumPy, SciPy, Pandas, Pynapple, Matplotlib, and current
  project implementations. Verify a library API only when this work first
  creates a new call site.
- Optimize only a measured regression in an existing performance-sensitive
  path. Metadata and UI glue do not receive speculative optimization.

### Explicit complexity prohibitions

Do not add a generic artifact framework, execution framework, transaction or
receipt protocol, generalized registry, dynamic plugin system, database,
service, schema-migration framework, concurrency defense, or hostile-writer
protection. Do not refactor code merely to match a diagram or create symmetric
packages. Ordinary validation, temporary-file replacement where already used,
clear errors, and the current trusted single-user filesystem model are enough.

Before adding a new abstraction, answer all of these questions:

1. Which current end-user action requires it?
2. Which two current callers share it?
3. Why is a direct function inadequate?
4. Which realistic failure does it handle?
5. Is the added code easier for a scientist to understand?

If the answers are not concrete, do not add the abstraction.

## 2.5 Ordered milestones

The former NR2-NR18 dependency waves are superseded. Future work is sequential
and outcome-driven:

| Milestone | Required outcome |
| --- | --- |
| U1 - session metadata | CT026, CT014, and a differently named synthetic session can describe and validate behavior and per-probe sources without code edits. |
| U2 - metadata-driven webapp | The existing webapp launches from one metadata path and uses those resolved sources for existing bounded interactive and cached views. |
| U3 - metadata-driven computation entry | The existing Spike-phase/PPC dry run, local entry, and Slurm handoff resolve the same metadata and scientific configuration. |
| U4 - documentation and usability approval | A user can follow the concise guide without reading implementation modules; the user approves the workflow. |
| R - readability-only reorganization | After U4 approval, selected existing modules may be moved or split with no new functionality. |

No readability milestone may start early because a future module boundary would
be convenient. Usability code may remain in existing modules when that is the
simplest implementation.

## 2.6 Implementation and review policy

- Use one active writer at a time. A Terra/high worker is the default for
  straightforward metadata, CLI, and UI changes.
- The lead owns requirements, user communication, worktree safety, commits,
  documentation, and authorization boundaries.
- Use one independent Sol/high review when a milestone changes scientific
  source selection, provenance, cache identity, serialization, or execution.
- Use Sol/xhigh only for an actual numerical-method change or unexpected
  real-data scientific discrepancy. Neither is planned by this rewrite.
- Do not require a separate reviewer, model assignment table, or documentation
  commit for every small extraction.
- Keep one editing agent and preserve unrelated user changes.

Milestone ownership is deliberately small:

| Milestone | Terra/high writer | Sol/high review | Lead responsibility |
| --- | --- | --- | --- |
| U1 | Focused RED tests, metadata records, JSON/CLI, and resolution | Schema meaning, source binding, missingness, and the no-array boundary | Freeze the field/consumer contract, protect scope, commit, and update the guide |
| U2 | Focused webapp RED tests and minimal replacement of manual source wiring | Source routing, saved-versus-live provenance, and startup non-computation | Freeze the supported view list and conduct the usability check |
| U3 | Focused adapter/launcher/wrapper RED tests and minimal metadata integration | Configuration equivalence, source fingerprints, saved-run behavior, and dry-run nonmutation | Preserve execution and authorization boundaries and approve user-facing commands |
| U4 | Examples, README, and command smoke tests | README-only novice workflow review and final cross-milestone consistency review | Conduct user acceptance and decide whether readability phase R may start |

One Terra/high writer may carry U1 through U3 for continuity. Do not create a
new writer or reviewer for every file. Sol/xhigh is reserved for an actual
numerical-method change or unexpected real-data scientific discrepancy; none
is planned for U1-U4.

Every implementation handoff contains only:

- the milestone goal and explicit non-goals;
- the baseline commit and known unrelated user-owned changes;
- the exact editable file list and current functions/callers in scope;
- the metadata field-to-consumer mapping relevant to that milestone;
- the tests to add, the focused RED command, and the expected RED reason;
- the focused GREEN and milestone-completion commands;
- the array/I/O performance boundary and forbidden real-data or external
  actions; and
- the final diff, test summary, and remaining user gate.

Reviews address ordinary end-user failures, scientific source/configuration
identity, saved provenance, and destructive mistakes. They must not add
hostile-writer, concurrency, generalized transaction, hypothetical-layout, or
framework requirements.

For each implementation milestone:

1. inspect the current callers and write a short file/function plan;
2. write focused tests and demonstrate meaningful RED;
3. commit the tests;
4. implement the smallest behavior needed for the milestone;
5. run focused and directly affected tests;
6. review the stable diff at the level warranted by its scientific risk;
7. run the complete neural suite once at milestone completion; and
8. update the end-user guide and execution log when behavior or commands
   changed.

Run the complete repository suite at final usability acceptance and final
readability acceptance, or earlier only when a change affects shared code
outside neural analysis. Commit grouping after the tests-first commit follows
the clarity of the change; separate implementation and documentation commits
are not mandatory.

## 2.7 Dependencies and tooling

No new runtime dependency is planned. The implementation uses:

- standard-library dataclasses, JSON, hashing, paths, temporary files,
  subprocess/process control, and atomic replacement;
- NumPy for explicit arrays, memory mapping, and numerical contracts;
- Pandas for source/trial/metadata tables;
- SciPy and Pynapple for the already approved signal/statistical operations;
- scikit-learn for the existing approved PCA/decoding operations;
- Matplotlib for pure visualization; and
- Streamlit only inside the webapp package.

Use `pytest` for every test. Use `uv` for dependency management and `uv run` for
all Python commands. Do not introduce Pydantic, Marshmallow, Click/Typer, Zarr,
HDF5, Dask, a workflow engine, or a plugin framework merely to implement this
plan. A proposed new dependency requires demonstrated substantial benefit,
official/package-source API verification, focused tests, and explicit user
approval before its lockfile change.

Formatting or static-analysis tools are not added implicitly. If the existing
project later adopts one, integrate it as a separately reviewed tooling package
rather than mixing broad mechanical rewrites into a scientific migration.

## 2.8 Minimal integration contract

Exact metadata field names are frozen by the U1 tests. The following user and
data flow is binding.

### Metadata boundary

- The canonical file is `<session_home>/neural_session.json`.
- It records session identity and session-level behavior sources.
- It contains a list of probes. Each probe records its stable probe ID,
  acquisition family, LFP source, required synchronization source, spike
  sorter/aligned-spike sources, optional channel-quality source, sites, and
  population references used by existing capabilities.
- Field names make file-versus-directory expectations explicit.
- Hardware/probe identity remains distinct from anatomical labels.
- Paths are relative to the session root and must remain contained within it.
- Metadata contains no raw arrays and no analysis presets.
- Schema version 1 supports only the inspected Open Ephys and SpikeGLX layouts
  already used by CT026 and CT014. Unknown layouts fail clearly; no migration,
  external-root mapping, or layout inference is added.

The U1 tests freeze exact field spellings. Before those tests are written, the
tests-only handoff must map each field to its current consumer using this
minimal content contract:

| Metadata value | Kind | Needed by | Existing consumer |
| --- | --- | --- | --- |
| subject and session identity | scalar labels | every action | current display and `LFPSummaryConfig` identity |
| behavior session directory and explicit trial/event files used by current code | directory and files | behavior views and applicable summary components | current behavior/trial loaders |
| optional existing treadmill source | file | treadmill-dependent existing views | current treadmill loader |
| stable probe ID and acquisition family | scalar labels | probe selection and LFP loading | current Open Ephys or SpikeGLX loader selection |
| LFP source and its authoritative preprocessing/metadata sidecar | files | LFP views and summary components | current Open Ephys `lfp_preprocessing.json` or SpikeGLX same-stem metadata reader |
| synchronization source | file | aligned LFP views and summary components | current alignment loader |
| sorter output | directory | spike views and Spike phase | current sorter metadata loader |
| aligned spikes | file | spike views and Spike phase | current aligned-spike loader |
| optional channel-quality source | file | quality-selected populations | current channel-quality loader |
| site IDs, display labels, probe references, and saved-channel indices | records | LFP controls and configuration | current `LFPSiteConfig` construction |
| site-pair references | ID pairs | existing synchrony views | current site-pair configuration |
| optional approved cache or snapshot directory | directory | cached views | current cache/snapshot inspection |

The metadata records source locations and stable labels. It does not store
derived unit IDs, selected-unit results, scientific thresholds, quality-filter
rules, bands, seeds, or execution policy. Current code-owned selection rules
remain code owned; explicit run choices remain command inputs.

### Resolution boundary

One small metadata module loads, structurally validates, resolves paths, and
reports action-specific missing inputs. One narrow LFP-summary adapter uses the
resolved result to call the current configuration builders. The webapp and
launcher must not maintain separate mouse-specific path logic.

Conceptual public functions are:

```text
load_session_metadata(path) -> SessionMetadata
resolve_session_metadata(metadata, metadata_path) -> ResolvedSession
validate_session_for_action(session, action) -> ValidationResult
resolve_probe_sources(session, probe_id) -> ResolvedProbeSources
build_lfp_summary_config(session, request) -> LFPSummaryConfig
```

Tests-only review may refine names, but it may not expand responsibility.

### Webapp boundary

- The existing webapp accepts one explicit metadata path.
- It derives subject/session, probe, site, pair, population, source, and cache
  controls from resolved metadata.
- Existing bounded interactive capabilities may run only after explicit user
  action and continue using their current numerical functions.
- Spike-phase/PPC production computation remains outside Streamlit. The app
  retains its current explicitly triggered bounded Power/Synchrony actions and
  may show the exact Spike-phase/PPC dry-run or launcher command.
- Cached views use the saved cache/snapshot provenance and never relabel saved
  arrays with new live metadata.
- Startup and unrelated rerenders do not perform expensive computation.

### Computation boundary

- The existing Spike-phase/PPC launcher and `hpc_ppc.sh` wrapper gain a metadata
  input; they are not replaced by a generic execution system.
- Dry run, local execution, and Slurm handoff resolve the same metadata into the
  same scientific configuration.
- Explicit user choices remain explicit: probe/population, shuffle count,
  worker count, preview/final intent, cache directory, and output/run root when
  needed.
- Resume uses the saved run directory and saved resolved configuration.
- Existing cache, checkpoint, progress, resource, signal, and report behavior
  remains unchanged.
- Power and Synchrony retain their current bounded local/webapp entry points.
  U3 does not add Power, Synchrony, or Compute All to the cluster launcher. A
  future request for those cluster modes requires a separate user-approved
  usability milestone.

### Supported flow

```text
neural_session.json
  -> load / validate / resolve session and per-probe sources
  -> existing bounded webapp views
  -> existing cache inspection
  -> existing Spike-phase/PPC dry run / local launcher / Slurm wrapper
```

This is a user-input and routing change, not a new scientific workflow.

## 2.9 End-user quickstart documentation contract

`src/neural_analysis/README.md` is a short operational guide for a lab user
returning after months away. Its first screen answers: which file do I edit,
how do I open the webapp, how do I start a large computation, and where do I
look afterward?

It contains these sections:

1. **Workflow at a glance.** Create/edit metadata, validate, launch the
   webapp, and dry-run or start the existing LFP-summary computation.
2. **Create the session metadata.** Show the canonical path, the editable
   creator values, skeleton generation, and the exact validation command.
3. **Metadata fields.** A compact table states the type, file/directory
   expectation, required/optional status, and meaning of session behavior and
   every per-probe LFP/synchronization/spike/quality field. Include generated
   Open Ephys and SpikeGLX examples rather than an untested hand-written schema.
4. **Launch the webapp.** Show one metadata-driven command, expected selectors,
   live versus cached behavior, and unavailable-input messages.
5. **Run a large computation.** Show exact metadata-driven dry-run, local, and
   Slurm commands for the existing Spike-phase/PPC launcher. Explain preview
   versus final intent and identify the run directory. Describe Power and
   Synchrony only through their already-supported bounded local/webapp paths;
   do not imply that they have a cluster launcher.
6. **Know when a run finished.** Identify state, log, manifest, component,
   report, failure, and resume locations without implying that file existence
   alone proves compatibility.
7. **Important Python functions.** Document only the small supported set for
   metadata load/validation, probe-source resolution, LFP-summary planning,
   cache inspection, and the existing scientific entry points users actually
   call. Each entry states types, shapes/axes, units where relevant, returns,
   and a short example.
8. **Troubleshooting.** Cover wrong roots, missing behavior, LFP, sync, sorter,
   aligned-spike, or quality inputs; stale caches; incomplete runs; and resume.
9. **Safety and provenance.** Explain immutable approved outputs, code-owned
   scientific presets, and the difference between live metadata and saved
   cache provenance.

Documentation rules:

- Use generic placeholders, not CT014/CT026 paths as universal defaults.
- Commands come from real parser contracts and use `uv run`.
- Every pasted command has a parser or synthetic smoke test.
- Generated examples share the tested metadata fixture/creator path.
- Keep internals, architecture, profiling, and historical evidence out of the
  quickstart.
- Update the guide in the same milestone that changes a user-visible command,
  path, output, or control.

## 3. Approved version and compatibility policy

The user approved this policy before implementation. Tests still freeze exact
field spellings and compatibility fixtures before source edits.

### 3.1 Do not use a global storage-schema bump for a value-semantics change

The correction changes the interpretation of Open Ephys source values, not the
shape or serialization format of every final component. Bumping the top-level
manifest `schema_version` alone would unnecessarily invalidate SpikeGLX
results, make the existing cache directory fail at manifest loading, and blur
the distinction between artifact schema and scientific semantics.

### 3.2 Add an explicit source-value semantics version

Add a small, code-owned version identifier for each acquisition adapter that
affects numerical values. The initial identifiers should distinguish at least:

- legacy Open Ephys stored-value behavior; and
- corrected Open Ephys metadata-scaled physical-value behavior.

The corrected identifier must participate in the scientific identity of every
component that consumes an Open Ephys site. It must also be saved in resolved
configuration/component provenance. SpikeGLX identity remains unchanged by
NR0; U1/U3 add its already-required authoritative metadata identity when the
metadata-driven path is implemented.
The exact field name is chosen during the tests-only package, but it should
describe source-value semantics rather than reuse `schema_version`.

Recommended ownership:

- the Open Ephys adapter defines its semantics version;
- `component_fingerprint` includes the versions required by the configured
  sites;
- prepared-phase and PPC work identities inherit the corrected source identity;
- the final manifest component entry records the resolved adapter versions.

An old component assessed under corrected live configuration must become
`stale`, not `compatible` and not an unreadable/corrupt artifact. Cache-only
inspection of the old immutable snapshot must continue through its saved
configuration and compatibility reader.

Do not make a new required field in the legacy canonical
`LFPSummaryConfig` JSON the only representation of this version. Persist the
resolved adapter-version mapping as component provenance and include that
mapping in the component fingerprint input. A receipt-validated historical
snapshot whose component entry predates the mapping is interpreted as the
recorded legacy Open Ephys semantics for inspection only; absence is never
accepted as current live-computation identity. This preserves deserialization
of old saved configurations while labeling their arrays honestly. If an
optional configuration field is also introduced for internal convenience,
`lfp_summary_config_from_json` must use dataclass defaults for absent optional
fields and the public legacy snapshot test must cover that path.

### 3.3 Fingerprint the authoritative preprocessing metadata

`fingerprint_source_files` currently includes the Open Ephys `lfp.dat` and
aligned sync file, but not the sibling `lfp_preprocessing.json`. The correction
must include that JSON as a required source sidecar for Open Ephys components.
Changing its scaling, unit, channel count/order, dtype, layout, or sampling
metadata must stale affected components and prepared work caches.

The source fingerprint for this small authoritative JSON must include a
streamed SHA-256 content digest in addition to its resolved path, size, and
mtime. Size/mtime alone cannot guarantee invalidation for a same-length edit or
a copy that preserves timestamps. This does not change the existing
large-binary policy: production-sized LFP binaries retain the reviewed
size/mtime convention unless a separate package approves content hashing.

### 3.4 Close the other source-identity gaps in the usability path

NR0 remains limited to the Open Ephys correction. The codebase audit found two
separate omissions that U1/U3 must close while routing the existing supported
sources rather than silently folding them into NR0:

- SpikeGLX resolution includes each LFP binary's authoritative same-stem `*.lf.meta`
  file in the source identity, including a SHA-256 content digest because the
  metadata file is small and numerically authoritative. A binary path or
  directory stat is not a substitute for the metadata that defines
  saved-channel order, sample rate, channel type, and voltage conversion.
- Metadata/configuration resolution carries the exact consumed spike-population sources into resolved
  provenance and cache identity: `spike_times.npy`, `spike_clusters.npy`,
  `cluster_info.tsv`, the aligned-spike NPZ, and the channel-quality file when
  that file participates in population selection. Fingerprinting only the
  sorter directory is insufficient because editing a contained file need not
  change the directory entry itself.

These are cache-integrity migrations, not numerical-method changes. Their
tests-only gates must define old/new compatibility behavior before source edits:
new artifacts use the complete source identity, legacy snapshots remain
inspectable, and deterministic numerical arrays remain equal when the source
contents are unchanged.

## 4. Work package NR0 - Open Ephys scaling correction

### 4.1 Scope

NR0 is a minimal, isolated scientific correction. It does not introduce the new
session schema, move modules, redesign loaders, change analysis defaults, or
implement WP13 absolute-amplitude thresholds.

The exact NR0 allowlist is:

- `src/neural_analysis/lfp_loading.py`
- `src/neural_analysis/lfp_summary_models.py`
- `src/neural_analysis/lfp_summary_preparation.py`
- `src/neural_analysis/lfp_summary_pipeline.py`
- `src/neural_analysis/lfp_summary_plotting.py`
- `src/neural_analysis/lfp_summary_runtime.py`
- `src/neural_analysis/lfp_summary_webapp.py`
- `src/neural_analysis/psth_webapp.py`
- `src/tests/neural_analysis/test_open_ephys_behavior_integration.py`
- `src/tests/neural_analysis/test_lfp_loading.py`
- `src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`
- `src/tests/neural_analysis/test_lfp_summary_models.py`
- `src/tests/neural_analysis/test_lfp_summary_io.py`
- `src/tests/neural_analysis/test_lfp_summary_pipeline.py`
- `src/tests/neural_analysis/test_lfp_summary_plotting.py`
- `src/tests/neural_analysis/test_lfp_summary_preparation.py`
- `src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py`
- `src/tests/neural_analysis/test_lfp_summary_runtime.py`
- `src/tests/neural_analysis/test_lfp_summary_webapp.py`
- `src/tests/neural_analysis/test_lfp_summary_work_cache.py`
- `src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py`
- `src/tests/neural_analysis/test_psth_webapp.py`
- `docs/neural_analysis_refactor_execution_log.md` by the lead only, after the
  tests/implementation gates

If implementation requires broader files, stop and amend the reviewed plan
before editing them.

`lfp_summary_plotting.py` was added during the tests-only gate on 2026-09-23.
Section 4.3 item 30 already requires corrected physical-unit labels from plot
builders, and the existing implementation hard-codes ambiguous source/filtered
trace labels. Omitting the owning module made the approved requirement
impossible to implement inside the original source allowlist. This amendment is
limited to those required label changes; it does not authorize numerical plot
changes.

### 4.2 Loader contract

`lfp_preprocessing.json` is authoritative for the derived Open Ephys binary.
The loader must validate the exact observed CT026 representation of:

- sample rate in Hz;
- saved-channel count and order;
- output-binary name, segment count, and sample count;
- dtype and time-major channel-interleaved layout;
- the `lfp_binary_scaling` object and its documented affine conversion from
  stored values to physical values; and
- the physical voltage unit for every saved channel.

The inspected CT026 representation is not a scalar or bare channel vector.
`lfp_binary_scaling` is an object containing `data_units`,
`has_scaleable_traces`, `channel_ids`, `gain_to_uV_by_channel`,
`offset_to_uV_by_channel`, `physical_unit_by_channel`,
`export_scale_factor`, and the explicit conversion
`trace_uV = trace_value * gain_to_uV + offset_to_uV`. Support this observed
documented representation, not speculative variants.

The loader must require `output_binary` to match the selected binary basename,
require the declared segment count to agree with the single observed sample-
count entry, require the nested `channel_ids` to exactly match
`channel_ids_in_binary_order`; require the gain, offset, and unit vectors to
have length `num_channels`; select all three values by zero-based saved-channel
index; and validate the observed data-unit, scalable-trace, export-factor, and
conversion declarations. The initial adapter supports only the observed
canonical `dtype == "float32"`; integer, float64, explicitly opposite-endian,
or otherwise unsupported storage declarations fail closed rather than being
accepted merely because NumPy can construct a dtype. It also supports the
observed `export_scale_factor == 1.0` and fails closed on other values until
their semantics are reviewed. Gains must be finite and strictly positive. Offsets
must be finite and may be zero, positive, or negative. Units must be explicit
and supported; the initial supported physical unit is `uV`. Missing, malformed,
nonfinite, nonpositive-gain, unsupported, or dimensionally inconsistent
metadata fails before returning data. The loader must not infer a factor,
offset, or unit from a filename.

`read_open_ephys_lfp_channel_window` returns a one-dimensional float array in
the declared physical voltage unit and the existing sample rate in Hz. After
selecting and copying only the requested one-channel window, the reader applies
the selected affine conversion exactly once. It should use in-place multiply
and add operations on that float result so it does not allocate another
window-sized array, and it must not materialize or scale the complete binary.

The trial-window and continuous-block callers retain their current time grids,
shapes, filtering order, and public signatures. Their documentation is updated
to say physical microvolts for the supported CT026 metadata rather than
ambiguous "derived units."

For the current pre-session-schema `LFPSiteConfig` boundary, production Open
Ephys adapters must compare the configured `sample_rate_hz` and `voltage_unit`
with the authoritative normalized sidecar values before numerical loading.
Mismatch fails closed with the site identity and both values; the adapter must
not silently replace a conflicting configured value and then save the original
configuration as provenance. Injected synthetic loader seams may continue to
exercise other explicit units. After U1 and U3, resolved production
configurations obtain these fields from adapter metadata, so users do not
transcribe them into session JSON.

The shared Open Ephys loader is also used by the existing exploratory Streamlit
application. Every `st.cache_data` path that can consume Open Ephys values must
receive a hashable cache token containing the resolved LFP and aligned-sync
path/size/mtime identities, the adapter value-semantics version, and the
SHA-256-bearing `lfp_preprocessing.json` fingerprint. The reviewed large-binary
size/mtime policy remains unchanged. That token is part of the Streamlit cache
key and is propagated through nested spectrogram, phase, phase-locking, and
Hilbert calls. Corrected Open Ephys plots and saved
exploratory results label raw/filtered values as `uV`, power as relative to
`uV^2` where applicable, and record the semantics version and sidecar digest.
SpikeGLX labels, cache identity, and values remain unchanged. A
compatibility-preserving optional keyword-only cache-token extension is
authorized for existing directly tested webapp helper functions; callers that
omit it must derive the token before an Open Ephys cached computation rather
than use an identity-free cache entry.

### 4.3 Tests written and committed first

The tests-only commit must include the following focused contracts.

#### Metadata validation tests

1. The observed CT026-style metadata normalizes per-channel gain, offset, and
   physical unit without changing sample rate, sample count, dtype, layout, or
   channel axes.
2. Missing `lfp_binary_scaling` fails closed.
3. Missing or unsupported `data_units`, `has_scaleable_traces`,
   `export_scale_factor`, or conversion declarations fail closed with the exact
   metadata field in the error.
4. Channel-ID disagreement; gain, offset, or unit vectors with the wrong
   length; and a channel-count mismatch fail closed.
5. NaN, infinity, zero, or a negative gain fails. Nonfinite offsets fail, while
   finite zero, positive, and negative offsets remain valid.
6. A missing, mixed, or unsupported physical unit fails rather than being
   labeled `uV` downstream.
7. Integer, float64, explicitly opposite-endian, and other unsupported dtype
   declarations fail; the observed canonical `float32` declaration remains
   accepted.
8. Output-binary name and segment/sample-count disagreements fail closed;
   existing file-size and saved-channel validation remains unchanged.
9. A production Open Ephys site's configured voltage unit or sample rate that
   disagrees with authoritative metadata fails before numerical loading or
   cache mutation; the error identifies the site and both values.

#### Value-loading tests

10. A synthetic multi-channel `lfp.dat` applies the selected channel's exact
   gain and nonzero offset and returns the existing one-dimensional float shape
   and sample rate.
11. A channel-dependent synthetic fixture proves that the reader does not use a
   neighboring channel's gain, offset, or unit.
12. The high-level Open Ephys trial loader applies the affine conversion
    exactly once, before optional filtering, and preserves the requested
    event-relative time grid.
13. The LFP-summary trial preparation adapter returns physical values without
    changing interpolation, validity, RMS, or peak-to-peak axes.
14. The continuous phase-block loader receives physical values on the same
    absolute time grid.
15. The exploratory Open Ephys trace, spectrogram, continuous-phase,
    phase-locking, and Hilbert paths receive physical values, retain their
    existing numerical axes, and expose truthful `uV`/`uV^2` labels and saved
    provenance. SpikeGLX exploratory behavior is unchanged.
16. Every exploratory Streamlit cache that consumes Open Ephys data changes
    identity when the adapter semantics version or same-length sidecar content
    changes, even if binary path/mtime and sidecar size/mtime are restored.
17. SpikeGLX reading and gain correction are unchanged.

#### Cache identity tests

18. Open Ephys source fingerprints include `lfp_preprocessing.json` and its
    SHA-256 digest; a same-length content edit with restored size/mtime still
    changes the source fingerprint.
19. Corrected Open Ephys semantics change Power, Synchrony, and Spike-phase
    component fingerprints relative to the recorded legacy semantics.
20. SpikeGLX component fingerprints do not change because of the Open Ephys
    correction.
21. A legacy complete Open Ephys component is classified as stale under the
    corrected live configuration while its receipt-validated immutable
    snapshot remains readable and explicitly identified as legacy-unscaled by
    the public cache-only inspection path.
22. The saved configuration from that legacy snapshot deserializes through the
    cache-only webapp path, and launcher/resume parsing rejects mixed semantics
    without turning the legacy snapshot into a malformed artifact.
23. Prepared-phase and PPC work caches reject legacy source-value semantics and
    cannot resume mixed-semantics checkpoints.
24. The manifest records the resolved source-value semantics version for each
    completed affected component.

#### Numerical impact tests

For positive gain `c` and zero offset, deterministic synthetic inputs must
demonstrate:

25. source, filtered, RMS, and peak-to-peak amplitudes change by `c`;
26. linear PSD, reference PSD, and linear band power change by `c^2`;
27. session- and presession-normalized dB power remains equal within a stated
    tight tolerance and without an added epsilon;
28. Hilbert/Morlet phase, ITPC, ISPC, PLV, PPC, spike counts, null schedules,
    and stable identities remain equal, using exact equality where stable and a
    documented tight tolerance only for floating phase transforms;
29. cached source and band-filtered traces change scale while phase arrays keep
    their documented radians/dimensionless units;
30. plot builders retain scientifically applicable unit labels and render the
    corrected arrays without reopening raw data: physical traces use `uV`;
    the generic linear-PSD cache schema remains
    `source-voltage-unit^2/Hz` and resolves to `uV^2/Hz` together with cached
    per-site voltage units; and reference-normalized or log-power plots remain
    labeled in dB. A normalized dB axis must not be mislabeled as linear
    `uV^2/Hz`.

A separate affine-conversion test uses a nonzero offset and asserts the exact
physical trace. Scale-invariance claims are not applied to nonzero-offset data.

Tests must assert values and units, not merely that functions return without
error.

### 4.4 Required RED evidence

Run the focused tests after the tests-only commit and before any source edit.
The failure record must show failures caused by absent scaling, absent metadata
validation, or unchanged compatibility identity. Pre-existing unrelated
failures do not satisfy RED.

The initial focused command should cover:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_open_ephys_behavior_integration.py \
  src/tests/neural_analysis/test_lfp_loading.py \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_pipeline.py \
  src/tests/neural_analysis/test_lfp_summary_plotting.py \
  src/tests/neural_analysis/test_lfp_summary_preparation.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_lfp_summary_work_cache.py \
  src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py \
  src/tests/neural_analysis/test_psth_webapp.py
```

Narrow `-k` expressions may be used to make the first RED output readable, but
the complete files must pass before NR0 is considered green.

### 4.5 Implementation sequence

1. Normalize and validate the authoritative scaling and unit metadata.
2. Apply the selected conversion in the shared Open Ephys channel-window
   reader and update its data contract.
3. Add preprocessing metadata to source fingerprints.
4. Add the source-value semantics version to scientific and work-cache
   identities and persisted provenance.
5. Make production LFP-summary configuration agree exactly with the
   authoritative Open Ephys sample-rate/unit metadata before computation.
6. Propagate the same semantics/sidecar identity through exploratory Streamlit
   caches and correct their physical-unit labels and saved provenance.
7. Prove through the compatibility tests that the existing generic reader can
   inspect old snapshots while corrected live assessment marks them stale. No
   `lfp_summary_io.py` or `lfp_spike_phase_launcher.py` source edit is expected
   in NR0; if an adjustment is genuinely required, stop and amend the exact
   source allowlist before editing.
8. Refactor only within the touched functions after all focused tests are
   green.

Implementation and tests remain separate commits. Do not modify tests merely
to accept implementation output unless the reviewed requirement was wrong; in
that case stop and amend the plan first.

### 4.6 Verification ladder

Run verification in this order:

1. focused scaling and metadata tests;
2. complete affected test files listed in the RED command;
3. all LFP-summary, Power, Synchrony, Spike-phase, cache, plotting, webapp, and
   launcher tests;
4. complete `src/tests/neural_analysis` suite;
5. complete repository test suite if the neural suite is green.

Record exact commands, test counts, warnings, commit IDs, and elapsed times in
the execution record. A skipped or unavailable test must be explained; it is
not silently counted as verification.

### 4.7 Performance constraints

- Read only the requested memmap slice and allocate only its one-dimensional
  float result.
- Apply the affine conversion with in-place vectorized NumPy multiply/add
  operations on the requested one-channel float result only.
- Do not read or scale the full Open Ephys binary.
- Do not add a second copy of production-sized phase or PPC arrays.
- Metadata parsing must occur no more often than it does in the current path;
  deduplicating existing repeated metadata reads is optional and should be a
  separate behavior-preserving micro-refactor if pursued.
- The exploratory Open Ephys source/cache token is computed once per selected
  source at the workflow/view boundary and reused by nested cached calls; do not
  rehash the small sidecar independently for every trial or transform.
- Compare focused loader time and peak allocation before and after the change.
  A material regression requires investigation before CT026 execution.

## 5. Work package NR1 - CT026 correction impact and baseline approval

NR1 begins only after NR0 code and automated tests are green and reviewed. It is
a data-analysis validation package, not part of the unit-test implementation.

The Terra command/evidence runner may write only the new user-approved
versioned CT026 cache/report/run locations and temporary profiling locations
declared by the lead before execution. In the repository, only the lead edits
`docs/neural_analysis_refactor_execution_log.md` and the authoritative handoff
in `docs/Tasks_neural.md`. NR1 has no source or unit-test edit authority.

### 5.1 Safety and provenance

- Use the user-designated CT026 Open Ephys session from the design document.
- Never modify or overwrite the approved legacy cache, copied snapshot, report,
  receipt, or cluster run.
- Create a new timestamped analysis-run directory containing the exact scripts
  used, a Markdown summary, configuration, log, source identifiers, commit ID,
  adapter semantics version, and parameters.
- Write corrected intermediate/final artifacts to a new versioned cache
  location until user approval.
- Seed every stochastic operation with the already approved seed and record it.
- Start with a dry run that resolves paths, metadata, output locations,
  components, expected cache states, and resource bounds without numerical
  computation or cache mutation.

### 5.2 Ordered CT026 checks

1. Inspect both probe preprocessing files and record the exact scaling-object
   shape, gain and offset ranges, physical units, channel count/order, dtype,
   layout, conversion declaration, and sample rate.
2. Load small representative windows from each configured site and verify raw
   stored values, corrected physical values, standard deviations, and the
   expected affine conversion without producing a scientific cache.
3. Run corrected Power only. Compare all axes, trial selections, validity,
   references, traces, amplitudes, linear PSDs, band powers, normalized dB
   arrays, reports, and plots against the legacy result.
4. After Power review, run corrected Synchrony. Compare phase-derived arrays,
   bootstrap schedules/results, validity masks, source/filtered traces,
   exemplars, reports, and plots.
5. After Synchrony review, run the bounded 100-shuffle ProbeB Spike-phase
   preview. Compare unit/trial/site identities, phase, PPC, null results,
   reliability/significance, exemplar selection, traces, reports, runtime, and
   memory.
6. Do not run the corrected 1,000-shuffle final Spike-phase computation until
   the preview comparison is reviewed and the user explicitly authorizes it.
7. If authorized, run and inspect the final corrected Spike-phase artifact,
   copy it locally using the existing receipt/checksum workflow, and retain the
   producing path separately from the copied-local path.

### 5.3 Expected comparisons, not automatic acceptance criteria

The following are hypotheses to test:

- because the inspected CT026 offsets are zero, amplitude arrays scale by the
  recorded per-site gain;
- because those offsets are zero, linear power arrays scale by the gain squared;
- normalized dB arrays remain numerically equivalent;
- phase-derived statistics and categorical selections remain equivalent;
- plot y-ranges change for amplitude-bearing panels while scientific labels and
  captions become truthful.

Any unexpected change in trial validity, phase validity, selected units,
reliable/significant cells, null schedules, exemplars, or normalized power
requires investigation. Do not loosen tolerances or relabel the change as
expected without identifying its numerical cause.

### 5.4 NR1 approval gate

NR1 completes only when:

- corrected cache and report artifacts validate independently;
- old artifacts remain intact and inspectable;
- the comparison report explains every changed and invariant quantity;
- performance and memory remain acceptable;
- plots receive explicit user visual approval; and
- the user designates the corrected artifacts as the structural refactor's
  regression baseline.

No structural module move begins before this gate.

### 5.5 Synchrony band-summary interpretation gate

The flagged `PFC_theta_whole_itpc_band_summary.png` is not a median/IQR plot.
For each displayed condition, the current analysis:

1. selects trials passing the shared filter, objective-valid, user-exclusion,
   condition-membership, and PFC-validity gates;
2. represents each retained trial at every frequency/time point by its complex
   unit phase vector;
3. computes the observed ITPC map as the magnitude of the vector sum divided
   by the number of numerically valid trials at each point;
4. averages the finite ITPC-map values over the inclusive 6, 8, and 10 Hz
   theta samples and the half-open `[-2, 2)` second whole epoch to obtain the
   filled-circle point estimate;
5. performs 1,000 deterministic, with-replacement resamples of the selected
   trial positions, recomputes the nonlinear ITPC map for every resample, and
   averages the same theta/whole points; and
6. draws a vertical line from the 2.5th to the 97.5th percentile of those
   1,000 bootstrap scalar values.

The nine x positions are overlapping behavioral trial sets rather than nine
independent groups. Their selected-trial counts are respectively 135, 40, 249,
68, 218, 11, 29, 57, and 189 in the saved order. The y axis is dimensionless
ITPC in `[0, 1]`; the plot is descriptive and contains no between-condition
hypothesis test or multiplicity correction. The current caption calls the
lines `95% bootstrap CI`, while it does not identify the circle as the
plug-in estimate or show the bootstrap median.

The geometry is possible because ITPC is a nonnegative magnitude after vector
averaging. Resampling with replacement repeats some trial directions and
reduces effective directional diversity. Especially near weak phase locking
or with small trial counts, this shifts the bootstrap magnitude distribution
upward. A percentile interval from that shifted distribution is not required
to contain the original plug-in estimate. The effect occurs in the legacy and
corrected artifacts, so it is not an Open Ephys affine-scaling regression.
Nevertheless, systematic noncoverage in all nine displayed conditions is a
scientific communication problem and remains a user-approval blocker.

The user selected a presentation-only revision on 2026-09-23. Apply it to both
ITPC and ISPC band summaries. ITPC uses one site's phase vectors; ISPC uses the
pair's relative-phase vectors. After that input distinction, both call the same
`bootstrap_phase_clustering_bands` calculation: observed vector magnitude,
finite time-frequency averaging, seeded with-replacement trial resampling, and
the same percentile operation. Consistent presentation is therefore required.

#### Frozen presentation contract

- Preserve the observed ITPC/ISPC computation, selected trials, valid masks,
  seed, 1,000 resamples, time-frequency averaging, and existing 2.5th/97.5th
  percentile endpoints exactly. This package changes no estimand, interval
  method, or condition definition.
- Derive the 25th percentile, median, and 75th percentile directly from each
  finite `(bootstrap,)` scalar series while that series exists in
  `bootstrap_phase_clustering_bands`. Use one explicit
  `numpy.percentile(..., method="linear")` operation for
  `(2.5, 25, 50, 75, 97.5)`. Never infer an interior quantile from the saved
  endpoints, recenter the distribution, or substitute the bootstrap median for
  the observed estimate.
- Persist three additional dimensionless quantile arrays per metric:
  `itpc_bootstrap_q25`, `itpc_bootstrap_median`, `itpc_bootstrap_q75`, and the
  corresponding `ispc_bootstrap_q25`, `ispc_bootstrap_median`, and
  `ispc_bootstrap_q75` arrays. Also persist the routine's exact
  `selected_trial_count` as integer `itpc_band_trial_count` and
  `ispc_band_trial_count` arrays rather than reconstructing a broader count from
  condition/site or pair validity in presentation code. Retain
  `itpc_ci_low/high` and `ispc_ci_low/high` as compatibility array names for the
  unchanged 2.5th/97.5th endpoints, but do not expose confidence-interval
  terminology in the revised figures. Do not persist all bootstrap draws.
- Render one horizontal row per condition, preserving the current condition
  order from top to bottom. Put the dimensionless ITPC or ISPC metric on the x
  axis and label each condition with its exact trial count as
  `<condition> (n=<count>)`.
- Draw the observed plug-in estimate as a prominent filled circle. Draw the
  bootstrap resampling distribution at a small nonoverlapping vertical offset
  within the same condition row: an unnotched box from Q25 to Q75, a visible
  median line, and capped whiskers at the unchanged 2.5th/97.5th percentiles.
  Do not draw Tukey-derived whiskers, fliers, individual draws, or notches.
- Include a compact legend with exactly two semantic entries: `Observed
  estimate` and `Bootstrap resampling distribution`. Captions may identify the
  number of deterministic trial resamples and instability threshold, but the
  plot, caption, title, axes, and legend must not use `confidence interval`,
  `CI`, `null`, `significant`, or `significance` language.
- Retain the existing fewer-than-ten contributing-trials instability rule and
  identify affected conditions descriptively without inferential language.
  These single-session, overlapping condition summaries remain descriptive;
  no between-condition test or multiplicity claim is added.

#### Cache and implementation architecture

The 1,000 bootstrap scalar draws were transient during the approved
calculation and are not present in `synchrony.npz`; only the observed estimates
and outer endpoints were saved. The new median and quartiles therefore cannot
be recovered truthfully from the approved cache. Do not interpolate them from
the endpoints or recompute them in report/webapp code.

- Extend `PhaseBandBootstrapSummary` in `lfp_synchrony_summary.py` with Q25,
  median, and Q75 arrays on `(epoch, band)` axes, computed beside the unchanged
  endpoints from the same finite bootstrap values.
- Extend `build_synchrony_payload` and `SYNCHRONY_ARRAY_SCHEMA` with the six
  small arrays above. Add a code-owned Synchrony payload-contract version to
  the Synchrony component fingerprint only. This must classify the old
  Synchrony result as stale for active recomputation without invalidating the
  approved Power component or pretending that the scientific phase settings
  changed. Generic receipt/snapshot inspection of the old artifact remains
  available.
- Change `plot_phase_band_summary` to receive the observed values, five actual
  bootstrap quantiles, exact saved band/epoch trial counts, bootstrap count,
  and labels under non-inferential parameter names. Use the same public
  renderer for ITPC and ISPC.
- Update both cache-only report and webapp callers to require and pass the
  saved quantiles. Neither presentation path may open raw LFP, reconstruct
  phase tensors, generate bootstrap draws, or otherwise compute missing data.
- Do not overwrite the approved cache or report. After code verification and
  separate real-data authorization, create a new timestamped corrected cache,
  copy the approved `power.npz` bytes unchanged with truthful manifest
  provenance, and recompute only Synchrony with the unchanged scientific
  configuration plus the new payload contract. Render a new timestamped report
  for user approval. The prior cache, failed attempt, successful retry, report,
  comparison, and packaging evidence remain immutable.

#### Tests written first

Commit RED tests before any source edit. They must prove:

1. Q25/median/Q75 and the existing endpoints equal an independent explicit-
   linear percentile calculation on the actual saved-in-memory bootstrap scalar
   series for seeded uniform and concentrated phase fixtures, including NaN
   handling and fewer-than-ten-trial instability;
2. the observed ITPC/ISPC estimates, selected counts, full bootstrap scalar
   series, seeds, and 2.5th/97.5th endpoints are unchanged by adding the three
   interior summaries;
3. the six new quantile arrays and two exact trial-count arrays have exact
   `(condition, site-or-pair, epoch, band)` axes, dimensionless or trial units,
   safe float/integer dtypes, and survive payload write/load validation;
4. the component-specific payload version makes the old Synchrony cache stale
   while leaving the same Power component compatible and receipt-only legacy
   inspection readable;
5. horizontal plotting preserves top-to-bottom condition order, places the
   filled observed circle separately from the box, uses Q25/Q75 box edges, the
   actual median, and capped 2.5th/97.5th whiskers, and permits the observed
   point to lie outside those whiskers;
6. the plot has no notches, fliers, or individual bootstrap points; displays
   `<condition> (n=<count>)`; contains the two required legend entries; and
   contains none of the prohibited inferential words, case-insensitively;
7. ITPC and ISPC report figures both receive their metric-specific saved
   quantiles and the webapp renders each cache-only without a numerical loader;
8. report filenames, non-band Synchrony views, PLV views, trial identities,
   and exclusions remain unchanged; and
9. the seeded synthetic cache/write/reload/report integration exercises both
   ITPC and ISPC horizontal summaries and remains deterministic.

The initial source allowlist is
`src/neural_analysis/lfp_synchrony_summary.py`,
`src/neural_analysis/lfp_summary_runtime.py`,
`src/neural_analysis/lfp_summary_payloads.py`,
`src/neural_analysis/lfp_summary_models.py`,
`src/neural_analysis/lfp_summary_plotting.py`,
`src/neural_analysis/lfp_synchrony_validation.py`, and
`src/neural_analysis/lfp_summary_webapp.py`. The matching initial test
allowlist is `test_lfp_synchrony_runtime.py`, `test_lfp_summary_runtime.py`,
`test_lfp_summary_payloads.py`, `test_lfp_summary_models.py`,
`test_lfp_summary_io.py`, `test_lfp_summary_plotting.py`,
`test_lfp_synchrony_validation.py`, `test_lfp_summary_webapp.py`, and
`test_lfp_summary_synthetic_integration.py` beneath
`src/tests/neural_analysis`. Amend the allowlists before touching any other
file. Keep tests and implementation in separate commits.

#### Sol/Terra work division for NR1P

NR1P is a source package, not an NR1 evidence run. It follows the complete
mandatory sequence in Section 2.6 with no overlapping writer/reviewer edits:

1. The lead Sol/high records HEAD/status/diffs, re-reads every allowed source,
   test, and direct caller, verifies the installed NumPy percentile and
   Matplotlib boxplot APIs from package source, and prepares a bounded-context
   prompt containing the frozen presentation contract. The optional
   Terra/medium scout, if used, is read-only and may inspect only call sites and
   installed-library APIs; it does not write tests or source.
2. One Terra/xhigh writer receives only the test allowlist above. It writes all
   NR1P tests, runs the focused suite, records genuine failures caused by the
   absent quantile/count/schema/plot contracts, and stops. It may not edit
   source or documentation, touch CT026, stage/commit, push, or spawn agents.
3. The lead audits and independently reproduces RED. A fresh Sol/xhigh
   read-only test-design reviewer checks scientific invariance, actual-draw
   quantiles, axes/units, payload versioning, Power compatibility, horizontal
   artist semantics, report/webapp cache-only behavior, and allowlist purity.
   Any finding returns to the same Terra writer. Only after approval does the
   lead commit the tests-only diff.
4. The lead follows up with that same Terra/xhigh writer and authorizes only the
   source allowlist above. The writer implements the smallest change, does not
   alter the committed tests, runs focused GREEN, and stops without committing,
   real-data access, artifact writes, push, or cluster action.
5. The lead audits the diff, reproduces focused GREEN, and runs every affected
   test file plus the complete `src/tests/neural_analysis` suite. A new fresh
   Sol/xhigh read-only implementation reviewer audits numerical invariance,
   schema/fingerprint migration, public interface callers, plotting semantics,
   performance, and the stable complete diff. Findings return to the same
   Terra writer until the reviewer approves with no unresolved P0-P3 issue.
6. The lead alone commits implementation and then a documentation/evidence
   update. NR1P stops at code/test completion; it has no authority to create a
   real-data cache or report.

The NR1P focused RED/GREEN command is:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_synchrony_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_payloads.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_plotting.py \
  src/tests/neural_analysis/test_lfp_synchrony_validation.py \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py
```

#### Sol/Terra work division for NR1V

NR1V begins only after NR1P implementation is committed and reviewed on an
exact tracked-clean local checkout, and only after the user separately
authorizes the exact new local cache/report/run paths. NR1V does not require or
authorize a Git push; pushing is deferred to the separately gated NR1E cluster
prerequisite.

- One Terra/high command/evidence runner receives write authority only for
  those exact new external analysis paths. It may copy the approved Power bytes
  with a truthful receipt/manifest transition, execute production Synchrony
  once, render the new report, and write reproducibility evidence. It may read
  the immutable prior cache/report for comparison. It may not edit repository
  files, stage/commit/push, overwrite any prior artifact, run Spike-phase, use
  the cluster, loosen a tolerance, or rerun after an invariant failure without
  new lead/user direction.
- The runner stops after one terminal success or first invariant/error stop and
  returns exact commands, hashes, counts, timings, peak memory, comparisons,
  inventory, and preservation evidence. An interrupted or failed run remains
  immutable and is not silently repaired or replaced.
- A fresh Sol/xhigh read-only reviewer independently audits the stable artifact
  package: unchanged Power bytes, new Synchrony manifest identity, exact old
  numerical arrays/endpoints, new quantile/count ordering, performance,
  provenance, filenames, and systematic ITPC/ISPC figure semantics. It does not
  edit evidence or trigger computation. Findings requiring new computation or
  a tolerance decision return to the lead and user.
- The lead records the review and presents the figures. Only explicit user
  visual approval designates the NR1V Synchrony component as the Section 5.6
  transfer source.

#### NR1V presentation-correction checkpoint

The single authorized NR1V scientific invocation completed successfully and
its cache passed numerical, schema, provenance, preservation, and performance
review. Do not recompute it. The first immutable report is rejected and
preserved because rendered observed-circle footprints overlap bootstrap boxes
in 12 of 18 ISPC summaries. Tests-first commits `88d7826` and `389e4fd` close
that defect by validating display-coordinate footprints and moving a 0.20-row
box to a +0.30-row offset; all scientific values and meanings are unchanged.

The user authorized exactly one cache-only report rerender at
`<session>/analysis_runs/ct026_nr1v_synchrony_presentation_rerender_2026-09-23T23-30-02Z`.
After fresh static approval, runner SHA-256
`678b2b110ca897cc1d696159b78cf53269a26d374a5289f31006f057315aa273`
called the production cache-only renderer once at clean HEAD `f8ecbf0`. The
render completed in 74.826 seconds with 685,510,656-byte peak RSS; no retry,
raw LFP/sync/spike load, scientific recomputation, cache mutation, PPC/work
write, cluster action, or prior-report overwrite occurred. The exact accepted
cache hashes remained manifest `7967c14b...`, Power `164114d6...`, and
Synchrony `d9673f21...` before and after.

The sole new report leaf contains exactly 257 direct files: 252 PNGs and the
five required metadata files. All 216 non-band PNGs are byte-identical to both
preserved reports, and exactly all 36 ITPC/ISPC band summaries changed from the
rejected geometry. Fresh Sol/xhigh review inspected every band summary and
found no P0-P3 issue. An independent pixel audit found zero marker/box overlap
across all 324 condition rows, with a minimum seven clear pixel rows. This
artifact received explicit user visual approval on 2026-09-23. It is now the
sole Section 5.6 Synchrony transfer source. That approval does not authorize a
Git push, cluster checkout update, transfer, dry run, or Slurm submission.

#### Dependencies, performance, and real-data gate

Introduce no dependency. Reuse NumPy's installed percentile API and
Matplotlib's low-level boxplot artists; verify those APIs from installed source
before first project use. The six added float arrays plus two integer count
arrays total fewer than 10 KiB for the current nine-condition, three-site,
three-pair, three-epoch, two-band CT026 shape before NPZ compression. Bootstrap
draws already exist one summary at a time and remain transient; do not add a
second production phase tensor or retain all draws. Plotting loads only saved
scalar summaries.

The source change requires ordinary focused and complete-neural test gates. A
subsequent real-data run requires separate authorization because it reads CT026
and writes a new cache/report. Its expected Synchrony resource envelope is the
reviewed prior run (about 729 seconds and 4.29 GB peak RSS); material deviation
requires investigation. Acceptance requires exact equality of all identity,
validity, phase, estimate, bootstrap endpoint, exemplar, and nonpresentation
arrays; ordered endpoint/Q25/median/Q75 consistency; exact agreement between
saved counts and instability flags; systematic figure inspection; and explicit
user visual approval. Direct quantile-from-draw equality is established in the
tests while the transient draws are available rather than by retaining or
recomputing those production draws. Only the newly approved revised Synchrony
artifact may advance to the Section 5.6 cluster copy.

### 5.6 Corrected-cache cluster preview prerequisites

The eventual 100-shuffle ProbeB preview will reuse the approved corrected Power
bytes and the user-approved Section 5.5 revised Synchrony bytes rather
than recompute either component on the cluster. The currently approved
pre-presentation Synchrony file is retained as evidence but is not the cluster
preview prerequisite.
The NR1C-A and NR1C-B source packages and NR1V real-data presentation gate are
complete and approved. This section still does not approve transfer, push,
cluster checkout mutation, Slurm submission, or the preview itself.

#### Architecture and ownership

- Add a required `--cache-directory PATH` argument to the launcher's `new`
  mode. Apply it by immutably replacing only `LFPSummaryConfig.output_directory`
  after the existing CT026 builder returns. Persist the resolved value in the
  configuration, identity, preflight, state, summary, and resume contract.
  `resume`, report recovery, and report rerender continue to recover the exact
  path from saved state and must not accept a replacement path.
- Validate the new-run target before creating numerical work. It must be a
  nonsymlink direct child of the selected session's `processed` directory,
  must not be the protected `processed/lfp_summary_cache`, and must contain
  compatible complete Power and Synchrony components but no Spike-phase
  component. Reject missing/stale/failed/running prerequisites, path aliases,
  unexpected cache members, and a pre-existing matching PPC work root.
  Interrupted work is continued only through `resume`.
- Keep `src/shell_scripts/hpc_ppc.sh` as the resource boundary. It already
  forwards launcher arguments without reinterpretation. Freeze that behavior
  in tests and require the submitted command to include the explicit corrected
  cache, `--probe ProbeB`, `--shuffles 100`, and `--workers 8`, without
  `--final-run`.
- Add a narrow cache-relocation command in
  `src/neural_analysis/lfp_summary_cache_relocation.py` and a pure manifest
  rebinding helper in `src/neural_analysis/lfp_summary_io.py`. The relocation
  command owns path validation, source-equivalence evidence, byte-preserving
  component copying, destination-manifest construction, atomic publication,
  and an external ASCII JSON receipt. It does no Power, Synchrony, phase, or
  Spike-phase numerical computation.
- The transfer copies `power.npz` and `synchrony.npz` byte for byte. The local
  manifest is retained only as source provenance and must not be installed
  unchanged: component fingerprints and source records include resolved
  absolute local paths and therefore classify an unchanged copied manifest as
  stale on the cluster. Build a destination manifest against the active
  cluster configuration while preserving the original producing manifest and
  its SHA-256 in the relocation receipt. Preserve each component's original
  completion time, generator, array schema, units, and scientific metadata;
  replace only the destination-bound top-level configuration, component
  configuration fingerprints/snapshots, and source fingerprints. Do not claim
  that the copied components were computed on the cluster.
- Before rebinding, prove source equivalence. Compare canonical configuration
  fields relevant to Power and Synchrony after applying the single declared
  local-session-root to cluster-session-root mapping and excluding only output
  location. Unit-population and PPC execution fields are not component inputs
  and must not be represented as having produced the copied components. Stream
  SHA-256 once for every unique Power/Synchrony source on both hosts, including
  both LFP binaries, aligned-sync NPZs, the trial table, and preprocessing
  sidecars; compare exact byte size and digest. Sidecar semantics must remain
  `open_ephys_affine_uV_v1`. A mismatch aborts without publishing a destination.
- Stage under an absent sibling directory on the cluster filesystem. Require
  the final destination to be absent, validate both NPZ schemas using the
  original manifest, copy and hash the bytes, create the cluster-bound
  manifest, validate both components through the public compatibility API,
  assert Spike-phase is missing, write the relocation receipt outside the
  cache, and atomically rename the staging directory into place. The published
  cache contains only `manifest.json`, `power.npz`, and `synchrony.npz`.
- Use a new timestamped cluster analysis-run directory for transfer/submission
  receipts. Record the local and cluster roots, original and rebound manifest
  hashes, component hashes, all source hashes, exact Git commit, command,
  hostname, UTC times, and final public component states. Preserve the local
  corrected cache and the protected legacy cluster cache unchanged.

#### Tests written first

Commit the following RED tests before any source edit:

1. launcher parsing requires an explicit cache target for `new`, preserves it
   through dry-run/state/resume, and rejects any resume-time replacement;
2. launcher preflight rejects the protected legacy cache, symlinks, paths
   outside the session's direct `processed` children, missing or stale
   Power/Synchrony, an existing Spike component, unexpected cache members, and
   pre-existing work; no numerical loader or run directory is reached;
3. the Slurm wrapper forwards the cache path and exact preview arguments
   unchanged and continues to enforce eight CPUs/workers;
4. a fake local/cluster-root fixture proves that a literal manifest copy is
   stale under the cluster configuration;
5. relocation rejects any non-path scientific configuration difference, any
   source size/digest/semantics mismatch, an existing or symlink destination,
   and a source cache containing Spike/work/unexpected members;
6. relocation keeps both NPZ SHA-256 values byte-identical, preserves original
   producer identity in its receipt, changes only destination-bound manifest
   identity, and yields public `compatible` states for Power/Synchrony and
   `missing` for Spike-phase;
7. injected interruption before atomic publication leaves the final
   destination absent and the source/legacy caches unchanged; and
8. a launcher dry run against the relocated cache writes evidence only and
   performs no phase, spike, PPC, cache, work, report, or cleanup mutation.

The initial source allowlist is
`src/neural_analysis/lfp_spike_phase_launcher.py`,
`src/neural_analysis/lfp_summary_io.py`, and the new
`src/neural_analysis/lfp_summary_cache_relocation.py`. The initial test
allowlist is `src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`,
`src/tests/neural_analysis/test_lfp_summary_io.py`, and a new
`src/tests/neural_analysis/test_lfp_summary_cache_relocation.py`. A shell source
edit is not expected; if its tests reveal one is necessary, amend this
allowlist before editing. Keep tests and implementation in separate commits.

#### Sol/Terra work division for NR1C-A, NR1C-B, and NR1E

Cluster preparation is split into two sequential source packages so no worker
receives both atomic-relocation and launcher-orchestration authority at once.

**NR1C-A - cache relocation.** One Terra/xhigh writer receives source authority
only for `src/neural_analysis/lfp_summary_io.py` and the new
`src/neural_analysis/lfp_summary_cache_relocation.py`, and test authority only
for `src/tests/neural_analysis/test_lfp_summary_io.py` and the new
`src/tests/neural_analysis/test_lfp_summary_cache_relocation.py`. It first
writes tests and stops at genuine RED. The lead reproduces RED; a fresh
Sol/xhigh read-only reviewer performs the mandatory test-design gate for source
equivalence, streamed hashing, producer/destination provenance, path
containment, failure atomicity, and byte preservation. After the lead commits
tests, the same Terra writer implements source only and stops at focused GREEN.
The lead runs affected/full-neural gates, and a second fresh Sol/xhigh
read-only reviewer audits the stable implementation before the lead commits.
No role in NR1C-A may access CT026, transfer files, push, or use the cluster.

The NR1C-A focused command is:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_cache_relocation.py
```

**NR1C-A result - complete and approved.** Tests were committed before source
implementation as `e940671` and `72028ad`. Three later tests-first hardening
commits freeze rollback ownership under concurrent replacement (`6e243c3`), a
coherent source-manifest mapping/digest snapshot (`f10b7bd`), and the atomic
publication window (`b994495`). The reviewed source implementation is
`e1d99c3`.

The package adds a pure destination-manifest rebinding helper, bounded ZIP/NPY
header validation without numerical-array materialization, backward-compatible
header-only public component status, and an executable relocation module. The
relocation command validates exact root/path/configuration/source equivalence,
streams each source/component digest through the shared hash seam, copies only
Power and Synchrony bytes, prepares an external ASCII receipt before cache
publication, and atomically publishes the three-member cache. The producer
manifest mapping and SHA-256 come from one bounded byte snapshot. Staging
ownership is captured before rename and checked immediately after publication
and immediately before receipt commit; rollback removes only the unchanged
cache published by the current invocation and preserves any concurrently
changed destination. No normal success path rehashes component bytes for those
ownership checks.

Final verification was 107 focused tests and 1,503 complete neural-analysis
tests with 22 known warnings; `py_compile` and `git diff --check` passed. A
fresh Sol/xhigh implementation reviewer reported no P0-P3 findings. Work stayed
within the NR1C-A source/test allowlists and did not access CT026, transfer data,
touch a cluster, push Git, or run scientific computation.

**NR1C-B - launcher target and preflight.** Only after NR1C-A is committed, one
Terra/xhigh writer receives source authority only for
`src/neural_analysis/lfp_spike_phase_launcher.py` and test authority only for
`src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`. The worker may
read, but not edit, `src/shell_scripts/hpc_ppc.sh` to test unchanged argument
forwarding. It follows the same tests-only RED, lead reproduction, fresh
Sol/xhigh mandatory test-design review, lead test commit, same-writer
implementation, GREEN, complete-neural gate, and second fresh Sol/xhigh final
review sequence. If shell source or another file must change, the worker stops
and the lead amends the plan before any edit. No CT026, push, transfer, cache,
or cluster action is allowed.

The NR1C-B focused command is:

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py
```

**NR1C-B result - complete and approved.** The reviewed RED contract is
`0d056fa`; `4646b5d` repairs two dangling-symlink fixtures and isolates an
older retained-work recovery subcase without weakening the new safety
requirements. The reviewed source implementation is `c844adc`.

`new` now requires `--cache-directory` and immutably replaces only the builder
configuration's `output_directory`. Before trial metadata, numerical work, or
run-directory creation, it requires an existing nonsymlink direct child of the
selected session's `processed` directory, rejects the protected legacy cache
and every alias/nested/outside target, enforces the exact three-member cache
inventory, validates Power and Synchrony as compatible and Spike phase as
missing through the public header-only status API, and rejects the derived
pre-existing PPC work root. The exact resolved target is persisted in canonical
configuration, identity, paths, preflight, state, summary, resume, report
recovery, and rerender contracts; non-new modes accept no replacement target.
The existing Slurm wrapper remains unchanged and was dynamically verified to
forward the corrected cache, ProbeB, 100 shuffles, and eight workers without
`--final-run`.

Final verification was 50 focused tests and 1,530 complete neural-analysis
tests with 22 known warnings; `py_compile` and `git diff --check` passed. Fresh
Sol/xhigh test-design and implementation reviewers reported no P0-P3 findings.
Work stayed within the one NR1C-B source/test pair and did not access CT026,
transfer data, touch a cluster, push Git, or run scientific computation.

**NR1C-C - retained shared-work-root correction.** The first authorized NR1E
cluster preflight safely updated the tracked-clean cluster checkout to pushed
commit `c859afe`, then stopped before evidence-directory creation because
`processed/lfp_summary_work` already exists. Read-only review proved this is a
valid historical post-success state: the root contains one complete retained
prepared-phase representation (about 1.15 GB), an empty real `ppc/` directory,
and no PPC run, lock, symlink, or unexpected member. The prior successful
1,000-shuffle run removed its exact PPC run directory during cleanup. The
current launcher incorrectly rejects the shared root's mere existence, so any
successful session run permanently blocks a later `new` run. This contradicts
the intended NR1C-B contract, which protects active or retained PPC work rather
than prohibiting identity-safe prepared-phase retention.

Do not delete, archive, move, or numerically open the retained cluster work.
Keep the shared `processed/lfp_summary_work` path and every CLI, saved-state,
resume, report, cleanup, and runtime fingerprint interface unchanged. Correct
only metadata-only launcher preflight classification:

- an absent work root remains valid;
- an existing root must be a real nonsymlink directory with only optional
  `prepared_phase/` and `ppc/` children;
- `ppc/` must be absent or a real nonsymlink empty directory; any child blocks
  `new` conservatively and preserves resume-only semantics;
- `prepared_phase/` may contain multiple fingerprint-named, complete,
  lock-free, nonsymlink representation directories with exactly the expected
  regular files and mutually consistent small `metadata.json` and
  `complete.json` identity records; and
- preflight must not open `phase.npy`, `valid.npy`, `axes.npz`, or any other
  numerical array payload. Runtime identity checks remain solely responsible
  for deciding whether a completed representation is reusable. Legacy and
  `open_ephys_affine_uV_v1` source/value identities remain distinct, so the
  retained legacy representation cannot be reused as corrected work.

Tests must be committed before source and must prove:

1. a historical post-success root with complete prepared phase plus an empty
   `ppc/` container permits launcher dry-run preflight and remains byte- and
   inventory-identical;
2. multiple complete prepared representations are permitted;
3. any complete, incomplete, malformed, or locked child under `ppc/` rejects
   before trial loading or run-directory creation;
4. root/container/representation symlinks, active locks, incomplete
   structures, malformed or inconsistent identity JSON, unexpected members,
   and non-fingerprint representation names reject;
5. preflight never calls `numpy.load`, the component-array loader, or opens the
   retained `.npy`/`.npz` payloads;
6. existing resume and cleanup behavior for an exact retained PPC run remains
   unchanged; and
7. corrected-versus-legacy prepared-phase and PPC identity tests remain green.

The source/test allowlist is exactly
`src/neural_analysis/lfp_spike_phase_launcher.py` and
`src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`, plus this plan
and execution log for lead-owned status. One Terra/xhigh writer follows the
same tests-only RED, fresh Sol/xhigh test-design review, test commit,
source-only GREEN, complete-neural gate, and second fresh Sol/xhigh
implementation-review sequence as NR1C-B. No new dependency is allowed.
Inspection is linear in the small metadata inventory and may read only bounded
JSON/stat records. No role may access numerical CT026 work arrays, mutate the
cluster, transfer artifacts, push, or submit Slurm during NR1C-C. After it is
approved and committed, a new Git push and exact cluster checkout update are
separate user gates before NR1E preflight restarts in a new evidence directory.

**NR1C-C result - complete and approved.** Tests-first commits `337480a`,
`2522b45`, `ca3863b`, `84c138d`, and `6b2f937` cover safe retained layouts,
authentic prepared-metadata schema/fingerprint validation, bounded identity
records, every unsafe/special-node layout, and deterministic leaf/container/
hierarchy replacement races. Implementation commit `4a9554a` replaces the
blanket root-existence rejection with descriptor-anchored, metadata-only
classification. It uses no-follow/nonblocking opens, bounded 64-KiB identity
reads, authentic writer-schema and fingerprint checks, stable inventories,
and repeated anchored identity checks through the final hierarchy decision.
Numerical `.npy`/`.npz` payloads are never opened. Complete prepared work plus
an absent or empty PPC container is permitted, while every PPC child and every
unsafe, incomplete, malformed, locked, replaced, or changing structure fails
closed before trial loading or run creation. Public CLI, work paths, resume,
report, cleanup, and runtime scientific identities are unchanged.

Final verification was 135 launcher tests and 1,616 complete neural-analysis
tests with 22 known warnings. `py_compile`, Ruff, and `git diff --check` pass.
A fresh Sol/xhigh implementation reviewer reported no P0-P3 findings after
adversarial path-replacement, late-child, descriptor-leak, and schema probes.
No cluster path or numerical CT026 work was accessed or modified during this
package. The NR1C-C commits were pushed through `5cc1385`, and the separately
authorized cluster checkout update reached that exact tracked-clean commit.
Cache transfer, launcher dry run, and Slurm submission remain separately
user-gated.

#### NR1E second-preflight correction

The separately authorized second NR1E evidence preflight used a fresh immutable
directory and stopped at its first invariant failure before focused tests. The
production launcher population seam calls
`load_sorter_metadata(sorter)[1]`. That loader unconditionally materializes the
selected ProbeB `spike_clusters.npy` before returning `cluster_info.tsv`, even
though the population builder discards the categorical assignment array and
uses only the cluster and channel metadata tables. The evidence runner's
blanket `numpy.load` prohibition therefore rejected the production call before
NumPy opened the 22,715,988-byte file. Its access time still predates the
attempt. This is an evidence-runner contract mismatch, not a scientific
computation failure or a need to alter the selected population.

Preserve the failed evidence directory unchanged as rejected incident evidence:

```text
/gs/gsfs0/home/mchin1/contextProjectData/CT026/
  CT026_20260801_latent_inference/analysis_runs/
  ct026_nr1e_cluster_checkout_preflight_2026-09-24T04-07-04Z
```

It contains only the two runner scripts and three partial text logs; it has no
terminal JSON, inventory, relocation, launcher, cache, work, report, or Slurm
artifact. The cluster checkout remains exact tracked-clean `5cc1385`, the
corrected destination remains absent, and current metadata inventories show the
protected legacy cache plus the same one complete retained prepared-phase
representation and empty `ppc/` container. Because the failed attempt did not
reach its after-snapshot, byte preservation is inferred from the stopped code
path and current inode/size/mtime records rather than claimed as a completed
hash proof.

The smallest faithful correction is evidence-only; no source package or new
Git push is required. In one newly authorized, newly timestamped evidence
directory, the runner may invoke the unchanged production population seam with
an exact-path, exactly-once `numpy.load` allowance for only the selected ProbeB
categorical `spike_clusters.npy`. It must reject every other `numpy.load`,
pre-validate that one owner-controlled exact resolved path with `lstat` as a
regular nonsymlink file, then open it with
`O_RDONLY | O_NOFOLLOW | O_NOATIME | O_CLOEXEC`. An immediate `fstat` must
match the pre-open device, inode, regular-file type, and size before the binary
handle is passed to NumPy with only the exact production arguments. Post-load
`fstat` must still match device, inode, type, mode/permissions, size, mtime, and
ctime. The descriptor must then close on every success and failure path, and a
post-close final no-follow path status must match the descriptor identity and
all of those non-atime fields.
`O_NOATIME` remains mandatory and there is no ordinary-open fallback, but the
NFSv3 server is permitted to leave `st_atime_ns` unchanged or advance it
monotonically for this exact authorized read; no other status field may change.
Record `findmnt` filesystem type/options plus all pre-path, initial-fd,
post-load-fd, and post-close-final-path atime values. Record the resolved path,
exact-one call, size,
dtype/shape, stat records, wall time, and peak RSS, and do not inspect, retain,
or serialize array values. Never restore the old access time because that
would itself mutate source metadata. The allowance does not extend to aligned
spikes, LFP, component NPZs, prepared-phase arrays, PPC work, or scientific
kernels. This preserves the production binding that the launcher will actually
use and makes explicit the same bounded categorical-metadata read used by the
previously approved local NR1 dry run. A cluster-info-only source refactor
would avoid this discarded read, but it would introduce another tests/source/
push/checkout cycle without changing configuration or execution; defer that
cleanup from NR1E.

The corrected preflight order is binding:

1. verify checkout, environment, wrapper, corrected-destination absence, and
   run the focused and complete neural test gates;
2. invoke the guarded production population/configuration path exactly once;
3. run descriptor-anchored retained-work classification without numerical
   member access, source/path checks, legacy/work before-after preservation,
   and resource-bound checks; and
4. record final Git status plus complete immutable evidence hashes, inventory,
   command order, and terminal status.

A fresh Sol/xhigh read-only reviewer must approve that stable evidence. The
failed directory may not be overwritten or reused. The fresh preflight rerun
requires explicit user authorization and does not authorize transfer, launcher
dry run, or Slurm submission.

The first guarded attempt is also rejected incident evidence and must remain
unchanged:

```text
/gs/gsfs0/home/mchin1/contextProjectData/CT026/
  CT026_20260801_latent_inference/analysis_runs/
  ct026_nr1e_cluster_checkout_preflight_guarded_2026-09-24T04-48-54Z
```

It contains exactly eight direct files and no final guarded JSON, terminal
status, evidence hash manifest, or inventory. Phase 1 completed successfully:
107 relocation tests, 135 launcher tests, 15 wrapper tests, and 1,425 complete
tracked neural tests with 12 known warnings. The one production NumPy read
then completed materializing the categorical assignment array from the
22,715,988-byte `.npy` file, but the runner stopped before `.astype(int)`,
population/configuration construction, or any later check. Array dtype, shape,
and in-memory byte size were not persisted. Only `st_atime_ns` advanced, from
1789950270373073347 to 1790226094047832332. Device, inode, regular mode and
permissions, size, mtime, and ctime remained exact. The source is on NFSv3
mounted `relatime`; Linux does not guarantee that `O_NOATIME` suppresses a
server-managed NFS access-time update. This was the exact authorized read, not
a content or source-identity mutation. Do not restore the prior access time.

The current corrected destination remains absent, the checkout remains exact
tracked-clean `5cc1385`, and every one of the 14 legacy/work status records in
the durable before-snapshot still matches the current path without opening any
payload. The next attempt must use another fresh timestamped evidence directory
and the same binding order, exact-one guarded read, and no-scientific-compute
boundary. It must accept either unchanged atime or a monotonic NFS-only atime
advance while requiring all content-identity and modification fields to remain
exact. That third evidence attempt required and received explicit user
authorization but failed as recorded below; it did not authorize transfer,
launcher dry run, or Slurm submission.

#### NR1E held-runner gate after the third incident

The authorized NFS-aware attempt used a third fresh directory and completed
all phase-1 gates, then stopped before the permitted read because the evidence
script called nonexistent `dependencies.load_population` rather than the
actual `LauncherDependencies.load_active_population` field. The guarded NumPy
seam was never called, so ProbeB atime and every other file status remained
unchanged from the prior incident. Preserve this directory unchanged:

```text
/gs/gsfs0/home/mchin1/contextProjectData/CT026/
  CT026_20260801_latent_inference/analysis_runs/
  ct026_nr1e_cluster_checkout_preflight_nfs_aware_2026-09-24T05-20-12Z
```

It contains exactly seven direct files and no guarded-population JSON, terminal
status, hash manifest, or final inventory. The run passed 107 relocation, 135
launcher, 15 wrapper, and 1,425 complete tracked neural tests with 12 known
warnings. Checkout/environment/mount/wrapper/destination checks passed, all 14
protected legacy/work status records still match the saved before-snapshot,
and the pre-existing untracked generated-file name set is unchanged. No array,
cache, work, report, launcher dry run, or Slurm action occurred outside the
synthetic/test data used by pytest; no CT026 experimental/source array or
protected numerical payload was opened.

A one-line retry is prohibited. Static review of that correction found that the
runner would still omit necessary final evidence: protected after-inventory
comparison, destination recheck, exact sorter/aligned-spike/trial/LFP/sync
source-path validation, resource bounds and peak RSS covering the entire
population/configuration/classifier call, final HEAD/origin/tracked and
untracked-name checks, failure evidence after descriptor closure, immutable
hash/inventory records, and terminal status.

At that point the plan required two separately authorized mutation gates with
an intervening read-only static review:

1. **Staging authorization.** Create one newly timestamped absent evidence
   directory and write every executable/helper byte for the attempt: phase-1
   driver, snapshot/inventory helper, guarded runner, finalizer, hash/inventory
   logic, and exact ordered invocation command. Record every held file's
   SHA-256 and stop, without executing tests, production imports, population
   loading, NumPy, finalization, transfer, launcher, or Slurm.
2. **Held-script static review.** A fresh Sol/xhigh reviewer verifies those
   exact remote bytes without mutation: no-bytecode compile and AST inspection;
   `dataclasses.fields` and `inspect.signature(...).bind(...)` checks against
   the exact `5cc1385` checkout; production field/path names; the guarded
   pre-path -> initial-fd -> post-load-fd -> close -> post-close-final-path
   lifecycle; every failure branch; before/after preservation; resources;
   final Git state; hashes/inventory; and terminal status. The reviewer freezes
   the approved SHA-256 values. Any subsequent byte change invalidates review.
   Every compile, AST, import, and runtime-introspection command must use
   `PYTHONDONTWRITEBYTECODE=1` plus the frozen/offline/no-sync environment so
   review cannot create or refresh Python caches. Introspection may instantiate
   the production dependency factory but must not call a data seam or any held
   script's `main`.
3. **Invocation authorization.** Only after the lead presents static approval
   and the exact held digests may the user separately authorize one execution.
   Remote staging must rehash the held bytes immediately before execution and
   require exact equality. Execution then follows the existing tests-first,
   exact-one guarded categorical read, no-scientific-compute, complete evidence
   contract and stops without transfer, launcher dry run, or Slurm.

This held-runner requirement was attempted once and later superseded by the
simplified completion plan below. The two earlier rejected directories and
this third rejected directory remain immutable incident evidence.

**First held bundle - staged and rejected.** The user authorized staging only.
The runner created this fresh directory and wrote the bundle without executing
any held script, test, import, compile, NumPy call, production factory, data
seam, transfer, launcher, or Slurm command:

```text
/gs/gsfs0/home/mchin1/contextProjectData/CT026/
  CT026_20260801_latent_inference/analysis_runs/
  ct026_nr1e_held_runner_2026-09-24T06-07-05Z
```

Its frozen `held_files.sha256` digest is
`43d4fc248c57b4cf16401d768ae97d9daf6d2d9447a896f7f3c6cf0a1f72f813`;
the staging-manifest TSV digest is
`9050ba44dd5f9fb9566bffdb26983c30bcd712dffa3785587510a8db8e8c6de1`.
Fresh static review rehashed the unchanged bytes, passed shell syntax,
no-bytecode Python AST/compile, dataclass-field, and signature-binding checks,
and confirmed no guarded read occurred. It rejected invocation for six P1
fail-closed defects:

- the wrapper trusts a mutable digest manifest rather than an externally
  approved literal digest;
- initial/final checkout state is logged but not required to equal clean
  `5cc1385` and the preserved untracked-name baseline;
- shell `set -e` bypasses final preservation/failure evidence on any gate
  error;
- guarded-read failures can claim descriptor closure falsely and lose the
  initial/post/post-close status records;
- success is written before evidence hashing/inventory completion; and
- exact CT026 paths, 273-unit/383-channel population identity, stable-unit
  digest, seed, 8/25/64 blocks, checkpoint/prepared-cache settings, 2-GiB/
  12-GiB bounds, resource sufficiency, and full-call RSS are not enforced.

P2 gaps are the absence of a durable protected-after comparison receipt,
incomplete command/failure chronology, and a self-inaccurate staging inventory.
Do not invoke or modify this bundle. The original requirement for another
held-runner bundle and a separate staging review is superseded by the
simplified plan below. No invocation, transfer, launcher dry run, or Slurm
action followed this rejected bundle.

#### Simplified NR1E completion plan - superseding the held-runner design

The user rejected further held-runner development after local-only drafts
v7-v12 repeatedly expanded into wrappers, sealers, commit receipts,
transactional publication, and defenses against hypothetical concurrent file
replacement. Those drafts were never executed, copied to the cluster, or added
to the repository. Preserve them only as local rejected design artifacts; they
are not inputs to any later run.

This project is scientific end-user software operated by one user and trusted
collaborators. Apply the `SoftwareDesign.md` guidance on KISS, YAGNI, and
avoiding overengineering. Handle realistic failures: wrong or missing paths,
malformed metadata, symlinks, insufficient resources, interrupted commands,
ordinary write failures, and inconsistent outputs. Do not build adversarial
race defenses, a transaction protocol, or a generalized execution framework
for a one-use preflight script. If review begins expanding beyond this threat
model, stop and simplify the design rather than preserving complexity already
written.

The remaining pre-refactor work has five bounded stages:

1. **Simple evidence-only preflight.** Create one fresh timestamped evidence
   directory containing one readable Python script, its command, log, result
   JSON, and a SHA-256 inventory. Verify tracked-clean `5cc1385`, expected
   paths, resource availability, destination absence, ordinary legacy/work
   structure, and the exact CT026 production population/configuration. Call
   `load_active_population` once. Permit exactly one production read of ProbeB
   `spike_clusters.npy`; on NFS, allow its atime to remain unchanged or advance
   monotonically while device, inode, regular type, permissions, size, mtime,
   and ctime remain exact. Run the metadata-only retained-work classifier and
   compare protected path inventories before/after. Do not read LFP, sync,
   retained phase, component NPZ, or spike-time arrays. Do not create a cache,
   report, work product, or Slurm job.
2. **Two-host cache transfer and publication.** Record the approved local
   source cache's exact three-file inventory and hashes, transfer its manifest,
   `power.npz`, and `synchrony.npz` into a fresh cluster staging directory, and
   verify remote hashes. A small session-specific cluster script constructs the
   destination CT026 configuration, verifies the documented local-to-cluster
   path mapping from the source manifest, uses the existing pure manifest-
   rebinding helper, validates Power/Synchrony through the public header-only
   status API, requires Spike-phase missing, then atomically renames staging to
   the absent final cache and writes one external receipt. This deliberately
   replaces direct use of the general relocation CLI: that CLI requires the
   local and cluster session trees to be visible to one process, which is not
   true across these two hosts. Do not introduce mounts, fake source trees, or
   another repository package to force that interface.
3. **Metadata-only launcher dry run.** Run `new` with the corrected cache,
   ProbeB, 100 shuffles, eight workers, and `--dry-run`, without `--final-run`.
   Verify the exact cache identity, compatible Power/Synchrony, missing Spike,
   accepted retained prepared-phase state, resource estimates, and absence of
   scientific/cache/work/report mutation.
4. **Review and submit once.** Perform one bounded read-only review of the
   transfer receipt, cache hashes, dry-run configuration, legacy/work
   preservation, and exact `sbatch` command. After separate user authorization,
   submit exactly once, record only the returned job ID and durable command,
   then stop. Do not poll Slurm, logs, launcher state, checkpoints, or results.
5. **User-requested result inspection and NR1 closure.** When the user later
   requests inspection, read the completed state once. If interrupted, present
   the saved resume command and wait for direction. If successful, validate the
   Spike-phase component and report, obtain user visual approval, update this
   plan and the execution log, commit/push the documentation, and only then
   begin U1 after its separate plan/implementation approval. Never escalate
   automatically to 1,000 shuffles.

No repository source implementation or new unit test is planned for these
session-specific operations. Existing committed tests already cover manifest
rebinding, header-only component validation, retained-work classification,
launcher dry-run behavior, and Slurm argument forwarding. The cluster has
already passed the focused and complete tracked neural suites repeatedly at
the unchanged `5cc1385`; do not rerun them solely to produce another evidence
copy. Any newly discovered production defect becomes a separate, small
tests-first correction rather than being patched into an external script.

#### Sol/Terra work division for simplified NR1E

Use one worker at a time. These are operational assignments, not repository
code packages, so they do not use a RED/test-design/source-implementation
sequence.

**Lead Sol/high - authority and handoff.** The lead owns this plan, the
execution log, exact path/commit constants, user authorization boundaries,
documentation commits/pushes, and final user-facing decisions. The lead does
not expand a worker's mutation scope implicitly. If a production source defect
appears, the lead stops NR1E and creates a separate tests-first correction
package.

**Terra/high - preflight author and runner.** One Terra/high worker may author
exactly one session-specific Python preflight script outside the repository.
Its write allowlist is one newly declared evidence directory. Before execution
it returns the script, command, paths, and expected outputs to the lead and
stops. After explicit preflight authorization, the same worker executes it
once and returns the result, log, and inventory. It may perform the one
documented categorical `spike_clusters.npy` read but no other experimental
array read and no cache, work, report, transfer, or Slurm mutation.

**Sol/high - preflight review.** One read-only Sol/high reviewer checks the
script against a short practical checklist: correct production API names;
exact CT026 paths and constants; one allowed categorical read; forbidden-array
and mutation boundaries; before/after preservation; understandable failure
messages; and complete ordinary evidence files. It rejects only concrete
scientific, data-loss, path, authorization, or likely operational failures. It
must not require defenses against a hostile concurrent writer, transactional
receipts, wrapper/sealer processes, or a generalized recovery framework. After
execution it performs one read-only result review against the same checklist.

**Terra/high - transfer and dry-run runner.** After separate authorization,
one Terra/high worker owns only the declared local transfer record, remote
transfer run/staging/final cache paths, external receipt, and launcher dry-run
directory. It writes one simple session-specific publication script, streams
the three cache members with `rsync`, verifies hashes, rebinds and validates the
manifest, publishes the absent final cache, and runs the metadata-only launcher
dry run. It stops before `sbatch`. It may not edit the repository, legacy
cache, retained work, scientific source files, or any earlier evidence run.

**Sol/high - transfer/dry-run review.** One read-only Sol/high reviewer checks
source/destination hashes, destination manifest/configuration identity, public
component states, receipt, legacy/work preservation, dry-run contents, and the
literal proposed `sbatch` vector. The review is bounded to those artifacts and
does not reopen component arrays or introduce new machinery.

**Terra/high - submission only.** After explicit submission authorization, the
same transfer runner may execute exactly the reviewed `sbatch` command, record
the returned job ID and command, and stop. It must not poll or inspect any job
state. A later user-requested result check is a new read-only Sol/high task.

Every NR1E worker prompt must include the realistic single-user threat model,
its exact path allowlist, the allowed reads/writes, and its stop condition.
Workers stop on an actual path/configuration/hash/status mismatch, insufficient
resources, an unexpected ordinary filesystem error, or a need to mutate
outside the allowlist. They do not stop merely because additional defensive
machinery could be imagined.

**NR1E - external cluster evidence.** NR1E begins only after NR1V, NR1C-A, and
NR1C-B are approved and committed, NR1C-C is complete and pushed, and the
exact clean pushed cluster checkout is verified. These prerequisites are now
complete at `5cc1385`. Keep the cluster checkout pinned there for NR1E; later
unrelated local commits do not require another checkout update. One simple
evidence script and one simple transfer/publication script are sufficient. A
reviewer checks realistic scientific, path, and preservation invariants, not
hypothetical hostile concurrency.

Each external mutation remains separately user-gated: preflight invocation;
cache transfer/publication plus launcher dry run; and Slurm submission. A
documentation commit or Git push does not grant any of those authorities.

The lead remains the sole user-facing authority throughout NR1E and records
every authorization boundary in the execution log. A later user-requested
status/result inspection is a new bounded read-only assignment, not a
continuation monitor and not authority to resume, repair, or submit.

#### Dependencies and performance

Introduce no package dependency. Reuse the standard library, NumPy, `uv`, SSH,
`rsync`, Slurm, existing canonical configuration/fingerprint functions, the
pure manifest-rebinding helper, and the public header-only component validator.
Hash files in fixed-size chunks and reuse each digest. The transfer is about
232 MB and is I/O-bound; do not load component NPZ arrays. The preflight may
materialize the one 22,715,988-byte categorical `spike_clusters.npy` array
required by the production population seam, but no LFP, sync, phase, or
spike-time array. Record ordinary wall time and peak RSS without building a
profiling framework.

#### Execution and no-monitoring handoff

Use the already tested, tracked-clean `5cc1385` checkout. Run the simple
preflight once; after its review and separate authorization, transfer/publish
the cache and run the metadata-only launcher dry run; then present the exact
`sbatch src/shell_scripts/hpc_ppc.sh new ...` command for separate approval.
Do not rerun the complete test suite unless the checkout or relevant source
changes.

If submission is approved, capture only the returned Slurm job id and durable
submission command, then stop. Codex must not poll `squeue`, `sacct`, logs, or
launcher state. Slurm writes the persistent job log and the launcher writes its
timestamped state, progress, checkpoints, exact resume command, component, and
report. The user may later request a one-time status or result inspection. A
timeout/preemption is never converted into a new run; the saved explicit
`resume` command is submitted only after separate user direction. The preview
cannot automatically escalate to 1,000 shuffles or `--final-run`.

## 6. Ordered future roadmap

The completed NR0/NR1 scientific correction remains the regression baseline.
Future implementation is limited to the usability milestones U1-U4. The
readability phase R is deferred until the user approves U4. There is no target
package topology and no requirement to move every current function.

## 7. U1 - Session metadata and source resolution

### 7.1 Purpose

Provide one small, editable metadata document that points existing code to the
correct sources for a new session. It is a user input format, not a workflow
engine or scientific configuration language.

### 7.2 Initial production scope

Prefer a flat, direct implementation:

- new `src/neural_analysis/session_metadata.py` for data records, JSON I/O,
  structural validation, contained path resolution, and action validation;
- new `src/neural_analysis/cli/create_session_metadata.py` for editable creator,
  skeleton generation, and validation;
- new focused tests in
  `src/tests/neural_analysis/test_session_metadata.py` and
  `src/tests/neural_analysis/cli/test_create_session_metadata.py`;
- the initial metadata and validation sections of
  `src/neural_analysis/README.md`.

Do not split the module unless the tests and current callers demonstrate two
independent responsibilities that are difficult to understand together.

### 7.3 Metadata content

The initial schema contains only values required by existing capabilities:

- schema version;
- subject/mouse identity, session identity, and optional date/label fields
  already displayed or saved by current code;
- session-level behavior trial/event sources and optional existing treadmill
  source;
- a list of probes, each with stable probe ID and the inspected acquisition
  family;
- for each probe, the exact existing LFP data source and authoritative metadata
  sidecar or directory expected by its current loader;
- for each probe, the exact synchronization source used by current alignment;
- for each probe, the existing spike sorter directory, aligned-spike source,
  and optional channel-quality source used by current population selection;
- existing site/channel/anatomical labels and site-pair references;
- stable population labels and probe references needed to request the current
  code-owned population selection; and
- optional references to approved caches or snapshots already supported by the
  webapp.

File and directory fields are separate and explicitly named. Metadata does not
accept arbitrary bags of paths or infer a source by searching a directory.

Scientific bands, window definitions, seeds, shuffle procedures, numerical
thresholds, memory limits, cache schemas, and figure settings remain outside
metadata.

Validation has three direct levels:

1. structural validation checks JSON types, IDs, references, and the one schema
   version without touching the filesystem;
2. filesystem validation resolves session-root-relative paths and checks their
   declared file/directory kinds without opening scientific arrays; and
3. action validation reports which existing webapp or computation actions are
   available and which required inputs are missing.

Open Ephys versus SpikeGLX comes from the explicit acquisition-family field,
not a filename guess. Authoritative sample rate, units, and source-value
semantics come from the current acquisition metadata readers, not duplicated
user-entered values.

### 7.4 User-facing contract

The initial public Python boundary is the five small functions in Section 2.8.
The tests-only package may improve their names only to avoid a real collision;
it may not introduce managers, registries, base classes, or a second metadata
representation.

The CLI has two ordinary operations:

```text
uv run python -m src.neural_analysis.cli.create_session_metadata create \
  --session-root SESSION_ROOT [--output METADATA_PATH]

uv run python -m src.neural_analysis.cli.create_session_metadata validate \
  --metadata METADATA_PATH [--action webapp|power|synchrony|spike-phase]
```

`create` writes one human-editable incomplete skeleton and never searches the
session tree. The user fills explicit paths and records. `validate` prints a
short availability summary and returns nonzero for malformed metadata or for
missing inputs required by the requested action. With no `--action`, it reports
all four action categories without requiring every optional capability.

### 7.5 Tests written first

- deterministic JSON round trip and stable output ordering;
- exactly one schema version and clear rejection of unknown versions;
- canonical filename and session-root-relative path resolution;
- ordinary absolute-path, traversal, and symlink escape rejection;
- explicit file-versus-directory and required-versus-optional validation;
- one and multiple probes with different source paths;
- inspected Open Ephys/CT026 and SpikeGLX/CT014 layouts;
- a differently named synthetic mouse/session/probe with no CT-specific branch;
- duplicate IDs and broken probe/site/pair/population references;
- missing, `null`, and empty values remain distinct;
- incomplete metadata can be displayed while action validation explains which
  current view or computation is unavailable;
- creator output and skeleton output validate;
- metadata validation loads no scientific array and mutates no cache, work, or
  report path.

Do not test unrequested metadata locations, mixed new acquisition layouts,
automatic migrations, network sources, dynamic discovery, or hostile concurrent
replacement.

Focused RED/GREEN command:

```bash
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_session_metadata.py \
  src/tests/neural_analysis/cli/test_create_session_metadata.py
```

### 7.6 Performance and completion

Metadata validation reads JSON and small authoritative text metadata only when
filesystem/action validation requires it. It does not read LFP, sync, spike-
time, prepared-phase, or component arrays.

U1 is complete when CT026, CT014, and the synthetic differently named session
can be created, loaded, resolved, and action-validated without editing shared
code.

## 8. U2 - Metadata-driven existing webapp

### 8.1 Purpose

Make the existing webapp easy to launch for a new mouse/session by replacing
hardcoded or manually repeated source selection with the U1 metadata. Preserve
current scientific and presentation behavior.

### 8.2 Implementation scope

Modify `psth_webapp.main`, its existing LFP-summary bridge, and
`lfp_summary_webapp.render_lfp_summary_view` only as required. Add one narrow
`lfp_summary_session.py` adapter in U2 because both the webapp and the later U3
launcher need the same resolved-session-to-`LFPSummaryConfig` construction.
The adapter contains direct functions, not a workflow registry, factory tree,
or source framework. Do not create a new application shell, router, state
framework, data-access framework, or view package hierarchy.

The initial U2 edit set is:

- `src/neural_analysis/lfp_summary_session.py`;
- `src/neural_analysis/psth_webapp.py`;
- `src/neural_analysis/lfp_summary_webapp.py`;
- `src/tests/neural_analysis/test_lfp_summary_session.py`;
- `src/tests/neural_analysis/test_psth_webapp.py`;
- `src/tests/neural_analysis/test_lfp_summary_webapp.py`; and
- the webapp section of `src/neural_analysis/README.md`.

Broader source edits require a concrete missing current caller and a plan
amendment; a future preferred package shape is not sufficient.

The documented launch shape is:

```bash
uv run streamlit run src/neural_analysis/psth_webapp.py -- \
  --session-metadata /path/to/session/neural_session.json
```

The app loads and resolves that metadata once, then passes resolved records to
the current loader and view functions. Existing directly called helper
signatures remain compatible during U2. The metadata path replaces hardcoded
browser roots, session IDs, probe paths, fixed site/channel definitions, and
CT-specific region defaults in the metadata-driven route; it does not add a
second application.

The webapp accepts one explicit metadata path and derives the existing:

- subject/session display;
- probe selection;
- LFP site and site-pair selection;
- unit population and channel selection;
- behavior/trial source selection;
- bounded live-view source inputs; and
- approved cache/snapshot choices.

Existing bounded interactive analysis may run after the same explicit user
actions used today. Spike-phase/PPC production computation remains outside
Streamlit. The webapp may display the exact launcher command for that work.

The supported current views are explicitly:

- Unit raster/PSTH;
- Trial spikes/licks/choices;
- Population PCA decoding;
- Population PCA switch trajectories;
- LFP phase clustering;
- Single-trial relative phase;
- Spike-LFP phase locking;
- Single-trial spike-LFP phase; and
- Cached LFP summary, including its existing cached and explicitly triggered
  bounded Power/Synchrony routes.

Metadata may make a view unavailable when its current required source is
missing. U2 does not add a view or extend an existing view to a new analysis.

### 8.3 Tests written first

- the webapp entry point requires or explicitly receives a metadata path;
- CT026, CT014, and the synthetic differently named fixture produce controls
  from metadata without subject-specific branches;
- selecting a probe supplies that probe's behavior/LFP/sync/spike/quality
  sources to the existing loader seams;
- hardware identity and anatomy remain separate labels;
- missing optional inputs disable only affected existing views with a useful
  explanation;
- cached and live modes remain distinct;
- cached views use saved cache/snapshot provenance rather than relabeling with
  live metadata;
- startup and unrelated rerenders perform no expensive computation;
- bounded live operations occur only after explicit user action and reuse the
  existing scientific functions;
- existing public webapp entry points remain compatible; and
- a synthetic noninteractive startup smoke reaches the metadata-derived UI
  without raw-data computation.

Focused RED/GREEN command:

```bash
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_session_metadata.py \
  src/tests/neural_analysis/test_lfp_summary_session.py \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_psth_webapp.py
```

### 8.4 Performance and completion

Startup may read metadata and small manifests/tables but not full LFP, phase,
spike-time, or component arrays. A selected cached figure may load the existing
component data it already requires. No new Streamlit cache layer or background
execution is added.

U2 is complete when the same documented launch command opens the existing
supported webapp capabilities for all three metadata fixtures without editing
Python paths or dataset constants.

## 9. U3 - Metadata-driven computation entry

### 9.1 Purpose

Give the existing large Spike-phase/PPC computation one easy, metadata-driven
entry point while retaining its proven launcher, cache, checkpoint, report,
and Slurm behavior.

### 9.2 Initial production scope

Reuse the narrow `lfp_summary_session.py` adapter introduced in U2. Extend
`lfp_spike_phase_launcher.py` and `src/shell_scripts/hpc_ppc.sh` only as needed
to accept an explicit metadata path. Preserve direct-path commands during the
transition. Do not introduce another configuration builder, workflow registry,
or execution package.

The initial U3 edit set is:

- `src/neural_analysis/lfp_summary_session.py`, only for launcher-facing
  request construction not already covered by U2;
- `src/neural_analysis/lfp_summary_models.py`, only for the missing current
  source identities listed below;
- `src/neural_analysis/lfp_spike_phase_launcher.py`;
- `src/shell_scripts/hpc_ppc.sh`;
- `src/tests/neural_analysis/test_lfp_summary_session.py`;
- `src/tests/neural_analysis/test_lfp_summary_models.py`;
- `src/tests/neural_analysis/test_lfp_summary_io.py`;
- `src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py`;
- `src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`;
- `src/tests/neural_analysis/test_hpc_ppc_shell.py`; and
- the computation section of `src/neural_analysis/README.md`.

For `new`, `--session-metadata` and the legacy `--session-path` are mutually
exclusive. `--cache-directory` remains explicit, so session metadata does not
become an output-placement policy. Resume, report recovery, and report rerender
continue to accept only their saved run directory and never reread live session
metadata.

The metadata-driven command shape is:

```text
new --session-metadata PATH --probe ID --shuffles 100|1000 --workers N \
    --cache-directory PATH [--analysis-root PATH] [--dry-run|--final-run]
```

The user continues to state existing run choices explicitly:

- probe/population;
- shuffle count;
- worker count;
- dry run versus execution;
- preview versus final intent where already supported; and
- output/run root where the current command requires it.

The metadata supplies session identity and sources. It does not silently choose
scientific or execution settings.

U3 does not add Power, Synchrony, or Compute All modes to this launcher. Their
existing Python pipeline functions and bounded webapp actions remain available,
but adding new local/Slurm execution modes would be new execution capability
and requires a separate user-approved plan.

### 9.3 Tests written first

- CT026 metadata resolves to the approved configuration, source semantics,
  population, and component fingerprints;
- CT014 metadata resolves its existing SpikeGLX paths and authoritative
  metadata without CT026 assumptions;
- the synthetic differently named fixture passes through labels unchanged;
- webapp handoff and launcher dry run resolve the same session/probe/source
  identities;
- Spike phase requires only its current inputs and reports missing behavior,
  LFP, sync, sorter, aligned-spike, or quality sources clearly;
- action validation still proves that missing Spike inputs do not disable the
  existing Power or Synchrony webapp paths;
- dry run performs no numerical array load and no cache/work/report mutation;
- one bounded synthetic integration reaches the existing Spike-phase pipeline;
- existing direct-path CLI, resume, recovery, checkpoint, and Slurm forwarding
  tests remain green;
- resume continues to use saved run configuration rather than changed live
  metadata; and
- documentation commands parse and reach no-submit smoke paths.

The same milestone adds only the source-identity records already required for
correct reuse with new sessions: the authoritative same-stem SpikeGLX metadata
file plus the consumed `spike_times.npy`, `spike_clusters.npy`,
`cluster_info.tsv`, aligned-spike file, and channel-quality file when used.
Implement this by extending the current component-scoped source fingerprint
function directly; do not add a source registry or fingerprint framework.

Focused RED/GREEN command:

```bash
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_session.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py \
  src/tests/neural_analysis/test_hpc_ppc_shell.py
```

No new scheduler, execution backend, run-state model, lock behavior, recovery
mode, resource policy, component schema, or cache publication behavior is in
scope.

### 9.4 Performance and completion

Metadata resolution adds no large-array copy. Existing memory maps, prepared-
phase sharing, worker bounds, checkpoint behavior, and scientific kernels are
unchanged. Run no real session computation without a separate user gate.

U3 is complete when the metadata-driven Spike-phase/PPC dry run matches the
existing approved configuration, a synthetic run reaches the current pipeline,
and the existing Slurm wrapper can receive the same metadata-driven request.

## 10. U4 - Documentation and usability acceptance

### 10.1 Deliverables

Finalize `src/neural_analysis/README.md` as the single end-user guide described
in Section 2.9. Keep technical details in function docstrings or one linked
developer note only when needed.

Provide tested, generated examples under
`docs/examples/neural_analysis/` for:

- an Open Ephys session shaped like the existing CT026 inputs;
- a SpikeGLX session shaped like the existing CT014 inputs; and
- a differently named synthetic session demonstrating that names are data, not
  branches.

Document the small supported Python API, not every internal function.

The final README contains copyable commands for metadata creation and
validation, Streamlit launch, Spike-phase/PPC dry run and local execution, the
current Slurm handoff, resume, report recovery, and report rerender. Owning
parser tests exercise those commands; do not build a documentation-test
framework.

U4 edits only `src/neural_analysis/README.md`, the three files under
`docs/examples/neural_analysis/`, and the smallest owning parser/smoke tests
needed to exercise copied commands. If U4 uncovers a source defect, stop and
plan that localized fix rather than hiding it in documentation work.

### 10.2 Usability review

A reviewer who starts only from the README and an example session layout must
be able to:

1. generate or edit metadata;
2. identify every behavior and per-probe path that must be supplied;
3. validate the session;
4. launch the webapp and locate existing live/cached views;
5. produce a metadata-only computation dry run;
6. identify the supported local/Slurm handoff;
7. find run state, logs, caches, reports, and resume instructions; and
8. locate the important Python functions and understand their inputs, outputs,
   shapes, axes, and units.

Every copied command is exercised by a parser or synthetic smoke test. The
review must not require reading source code or architecture documents.

The U4 reviewer performs one novice walkthrough for an Open Ephys-shaped
example and one for a SpikeGLX-shaped example, plus the differently named
synthetic fixture. The walkthrough may parse commands and run metadata-only
smokes, but it performs no real numerical computation, cache publication,
external copy, or Slurm submission.

### 10.3 Usability completion gate

U4 requires explicit user approval. Until then, do not start readability-only
reorganization. A real-session run or cluster submission remains separately
authorized even when its command is documented.

## 11. R - Deferred readability-only reorganization

### 11.1 Entry condition

This phase starts only after U4 user approval. It reorganizes existing
capabilities for end-user scientific readability and simpler code where that
improvement is concrete. It does not complete a preselected architecture.

### 11.2 Absolute no-new-functionality rule

Readability work must not add or broaden:

- metadata fields or layouts;
- source formats or acquisition families;
- supported path-placement modes;
- analysis methods, conditions, statistics, plots, or report outputs;
- webapp pages, controls, live operations, or caches;
- CLI commands, modes, recovery actions, or execution backends;
- artifact formats, schemas, states, writers, or migration behavior;
- resource policies, parallelism, checkpoints, or scheduler support; or
- compatibility promises for previously unsupported cases.

Handling a more complex case of any kind is forbidden. When existing code
cannot be moved cleanly without broadening behavior, leave it where it is.

### 11.3 Allowed changes

- move a pure existing calculation into a clearly named module;
- move existing Matplotlib construction away from loading or computation;
- split a very large file when the resulting responsibilities are immediately
  clearer to a scientist;
- replace duplicated current-session path selection with the already approved
  metadata resolver;
- clarify scientific names, docstrings, shapes, axes, units, and comments;
- retain old imports as thin forwards; and
- remove a known-unused wrapper only after an actual caller audit and explicit
  user approval.

Candidate large modules are a backlog, not mandatory packages. Work on one
current pain point at a time. There is no requirement to move every analysis,
plot, report, artifact function, launcher function, or compatibility module.

### 11.4 Tests and review

- write compatibility/equivalence tests before movement;
- require exact old/new results where deterministic and the existing reviewed
  tolerance where floats are not bitwise stable;
- add no mirrored test tree solely to reproduce old tests under a new path;
- preserve all existing public entry points until deletion is separately
  approved;
- run focused and affected tests for each move and the complete neural suite at
  each accepted readability milestone; and
- benchmark only when a moved function is already performance-sensitive.

Stop and replan if a change begins designing a generic framework, introduces
more than the minimal modules needed for the current split, or requires new
user-visible behavior.

## 12. Explicitly deferred and forbidden backlog

The following former roadmap items are not future requirements:

- a generic analysis registry;
- a generic artifact, tabular-publication, snapshot, or legacy-result layer;
- a generic execution/launcher/Slurm package;
- a new Streamlit shell, router, state system, or view hierarchy;
- a full source-adapter package tree created before a current caller needs it;
- mirrored pure-analysis packages for every existing scientific module;
- a compatibility/profiling package split;
- repository-wide AST enforcement of dataset-specific literals;
- automatic schema migration or external metadata-root mapping;
- dynamic plugins or discovery;
- exhaustive retirement of wrappers or historical modules; and
- support for hypothetical concurrency, hostile writers, network sources, or
  uninspected acquisition layouts.

Existing implementations of current cache safety, checkpointing, launcher
state, and source validation remain in use. This section forbids building new
generalized versions; it does not remove current proven safety behavior.

## 13. Testing, performance, and acceptance

### 13.1 After each usability milestone

- a focused tests-first commit records meaningful RED;
- focused and directly affected tests pass after implementation;
- scientific source/configuration changes receive one independent review;
- the complete neural suite passes once at milestone completion;
- public commands and README snippets match tested parser behavior;
- no unauthorized real-data, cache, report, cluster, or external-state mutation
  occurred; and
- unrelated user files remain untouched.

### 13.2 Final usability acceptance

- CT026, CT014, and the synthetic differently named session validate from
  metadata;
- behavior and per-probe LFP/sync/spike/quality sources are selected from that
  metadata with no dataset-specific code edit;
- the existing webapp opens the supported existing views from metadata;
- the existing Spike-phase/PPC dry run, local entry, and Slurm handoff resolve
  the same metadata and scientific configuration;
- approved historical and corrected artifacts remain readable and unchanged;
- the concise README is sufficient for the end-user workflow; and
- the complete neural and repository suites pass.

### 13.3 Final readability acceptance

- each accepted move makes an existing scientific responsibility easier to
  find and understand;
- numerical behavior and user-visible capabilities are unchanged;
- compatibility forwards cover known current callers;
- no generalized framework or complex-case support was introduced;
- the complete neural and repository suites pass; and
- the user approves stopping even if some large legacy modules remain.

## 14. Approved decisions and authorization gates

The following earlier scientific decisions remain binding:

- corrected Open Ephys values use authoritative per-channel gain, offset, and
  unit metadata;
- source-value semantics and small authoritative sidecar content hashes keep
  legacy and corrected cache identities distinct;
- historical artifacts remain immutable and explicitly legacy;
- current configurations must agree with authoritative source units and sample
  rates before computation;
- absent historical semantics remain legacy only inside reviewed snapshot
  inspection;
- metadata is canonical at `<session_home>/neural_session.json`, with contained
  relative paths in schema version 1;
- the editable metadata creator and incomplete skeleton are both supported;
- only the observed Open Ephys and SpikeGLX layouts are supported initially;
  and
- user-supplied output names continue to use current validated safe naming
  where that capability already exists.

Former decisions that mandated generic workflow, tabular-artifact,
compatibility-package, exploratory-workflow, registry, or execution
architecture are superseded. Existing already-implemented behavior remains;
the rewrite does not authorize removing it.

The following remain separate explicit authorization gates:

- implementation start for U1 after review of this rewritten plan;
- every real-session numerical run or cache mutation;
- every external copy or Slurm submission;
- inspection, resume, or escalation of Slurm job `30766437`;
- the start of readability phase R after U4 user approval; and
- deletion of any compatibility wrapper, legacy module, cache, report, or
  evidence directory.

Approval of documentation or one milestone grants none of the later gates.
