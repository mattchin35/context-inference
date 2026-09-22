# Neural Analysis Refactor Design Goals

## Status and purpose

This document collects design goals for a broader neural-analysis refactor. It
is requirements intake, not an implementation plan. It does not authorize code
changes, choose a final metadata schema, define work packages, or supersede the
scientific contracts and completed-result evidence in `docs/Tasks_neural.md`.

The immediate motivation is to make the existing LFP summary analyses usable
for additional recordings, sessions, and mice without embedding CT026 paths,
probe meanings, site names, or channel selections in Python source. The same
configuration should eventually support local validation, cluster execution,
cache production, and webapp inspection.

## Current design pressure

The numerical configuration and pipeline are substantially reusable, but the
operator-facing bindings remain specific to CT026. In particular, current
production paths assume ProbeA/ProbeB directory names, PFC/HPC site meanings,
fixed saved-channel indices, a fixed trial-table convention, and a particular
sorter and synchronization layout.

Editable path text boxes alone do not solve this problem. A different file can
have a different acquisition format, sample rate, voltage unit, channel layout,
anatomical annotation, synchronization source, or unit population. Allowing a
path to change while silently retaining CT026 metadata would create plausible
but scientifically mislabeled output.

## Initial user direction

The first design direction is intentionally simple:

- Use a small JSON metadata file to select the session data home, session ID,
  and probe-specific sorter, aligned-spike, and LFP paths.
- Direct the webapp launcher to one explicit metadata file rather than entering
  subject-specific paths in fixed PFC/HPC controls.
- Retain dropdowns that select the active plot, cached component, population,
  and computation; those are useful runtime choices rather than session
  identity.
- Provide a simple entry point for running cache-producing analyses on the
  cluster from the same metadata.
- Reorganize the codebase substantially while retaining the exact approved
  computations. Webapp visualization, LFP-LFP analyses, Spike-LFP analyses,
  regression, and dimensionality analyses should have clear package homes.

JSON is the current preferred representation because it is simple, portable,
and already compatible with saved configuration metadata. The exact JSON schema
is not frozen by this document. Although the first recordings have two probes,
the schema should use a keyed or ordered probe collection instead of baking in
exactly `probe_1` and `probe_2` fields. The first UI may still present two probe
entries when the selected metadata contains two probes.

Further design direction:

- Each session should explicitly declare one acquisition family, such as Open
  Ephys or SpikeGLX. The system should not infer this scientific I/O contract
  from a filename. Mixed-acquisition sessions are outside the initial design;
  they must fail clearly rather than silently choosing a loader.
- Each session should normally have its own JSON metadata file. Pointing a
  restarted webapp at that file should restore the session inputs and known
  result locations quickly without re-entering paths.
- The metadata may contain optional references to approved copied/local cache
  snapshots, including the approved 100-shuffle preview and 1,000-shuffle final
  Spike results when they exist. Producing cluster paths and transfer checksums
  may remain in the snapshot manifest and receipt instead of being duplicated
  in the routine session JSON.
- Missing results are valid metadata state. A new session, a one-probe session,
  or a session whose cache has not yet been computed must still have a useful
  metadata description.
- A separate simple Python entry point should create a new session metadata
  file. Its first design should expose clearly labeled path and metadata
  variables near the top of the Python file for the user to edit, with its
  `main` function validating those values and writing the JSON. It may record
  incomplete or unavailable values explicitly rather than inventing paths,
  probes, sites, or results.
- Channel-to-region groupings may be added to the metadata model later, but a
  general channel-grouping system is not a priority for the first refactor.

Cache references in session metadata are navigation hints, not proof that a
scientific result is valid. The webapp and runner must still validate the
referenced manifest, configuration identity, completion state, and transfer
receipt where applicable. A path existing on disk must never be treated as
sufficient compatibility evidence.

## Design goals

### 1. Remove subject- and recording-specific assumptions from shared code

- Shared numerical, launcher, cache, report, and webapp code should not contain
  CT026 session IDs, CT026 directory names, or fixed PFC/HPC channels.
- Mouse/subject identity and session identity should be metadata, not control
  flow or module naming.
- A new mouse or session should normally require new metadata, not a new Python
  adapter.
- Dataset-specific compatibility adapters may remain during migration, but
  they should be visibly isolated and should not define generic defaults.

### 2. Establish one authoritative session-analysis description

- One explicit, human-reviewable metadata document should describe the inputs
  needed to validate and run an analysis.
- Local execution, Slurm launch, cache fingerprinting, reports, and the webapp
  should consume the same description rather than maintain parallel settings.
- The description should be serializable, versioned, and saved with every
  cache/report so a result can be reconstructed without relying on current UI
  state.
- Paths must be explicit. Automatic discovery may propose values, but it must
  never silently select a recording, sorter, synchronization file, trial table,
  channel, site, or population.
- Optional values should have an unambiguous incomplete representation. Missing
  must remain distinguishable from an empty string, an invalid path, and a
  completed result containing an empty scientific population.

### 3. Separate hardware identity from anatomical meaning

- A probe identifier such as `ProbeA` or `Probe1` should identify a recording
  source. It should not imply `PFC`, `HPC`, `V1`, or another anatomical region.
- The webapp should use generic, metadata-derived path labels such as
  `ProbeA sorter`, `ProbeA aligned spikes`, and `ProbeA LFP`, rather than fixed
  labels such as `PFC sorter` or `HPC/V1 LFP`.
- An LFP site should separately identify its probe, saved-channel index, label,
  and anatomical annotation. For example, a site may be displayed as
  `ProbeA / PFC / channel 5` while retaining a stable machine-readable site ID.
- Region should normally be site- or channel-level metadata, not only
  probe-level metadata. A single probe may cross multiple regions, so a
  probe-wide `region: PFC` value must not prevent more specific per-site or
  per-channel annotations.
- Probe-level region metadata may be allowed as an explicit default only when
  the entire probe is known to share that annotation.

### 4. Represent arbitrary probe and site topology

- The description should support one or more probes without assuming exactly
  ProbeA and ProbeB.
- Each probe should be able to declare its LFP source, synchronization source,
  sorter, aligned-spike file, and relevant metadata files. The initial schema
  declares acquisition format once for the session.
- Each site should declare a stable ID, display label, probe ID, saved-channel
  index, voltage unit, sample rate, and optional anatomical annotations.
- Synchrony pairs should reference stable site IDs explicitly. They should not
  be generated from anatomical names or tuple position.
- The current one-probe-per-Spike-run rule should remain explicit unless a
  separate scientific decision authorizes combined populations.

### 5. Make unit-population selection explicit and auditable

- Population definitions should identify the selected probe, sorter, aligned
  spikes, channel-quality rules, unit-quality rules, and stable unit IDs.
- Metadata column requirements and accepted categorical values should be
  validated before numerical work begins.
- The system should report how many channels and units each rule retains and
  should fail closed when required metadata is missing or ambiguous.
- Anatomical site selection and unit-population selection are related but not
  interchangeable. A site channel used for LFP does not by itself define the
  spike population.

### 6. Provide one simple computation entry point

- An operator should be able to select a metadata document, choose Power,
  Synchrony, Spike phase, or all approved components, and request a dry run or
  execution.
- The same entry point should support local and Slurm execution without changing
  scientific parameters.
- Execution controls such as worker count, memory limits, checkpointing, and
  partition/time requests should remain separate from scientific settings such
  as sites, bands, windows, filters, and shuffle count.
- Dry run should resolve paths, validate metadata, show planned components and
  outputs, estimate relevant resource bounds when available, and perform no
  numerical computation or cache mutation.
- Resume and report-recovery operations should consume the saved run metadata;
  they should not accept a different session or silently rebuild configuration.
- Metadata creation should be a separate entry point from numerical execution.
  Creating or editing a session description must not launch an analysis.

### 7. Keep local and cluster layouts portable

- Metadata should distinguish logical dataset-relative paths from environment-
  specific roots where practical.
- Moving between the workstation and cluster should require an explicit root
  mapping or environment profile, not editing every source path or Python file.
- Resolved absolute paths must be recorded at execution time for provenance.
- A path mapping must never change probe, site, channel, unit, or session
  identity; it changes location only.
- The workflow should continue to support copying completed snapshots locally
  instead of requiring a persistent cluster service or SSH tunnel.
- When both cluster and copied-local result paths are recorded, their roles and
  relationship must be explicit. The local path should not masquerade as the
  producing cluster path, and copied bytes should retain a verifiable identity.

### 8. Generate the webapp from metadata

- Probe path controls, site selectors, site-pair selectors, population choices,
  and captions should be created from the active metadata rather than fixed
  PFC/HPC labels.
- Hardware identity and anatomical annotation should both remain visible where
  scientifically useful.
- Snapshot inspection should use the configuration saved with the snapshot.
  It must not relabel a saved result using a newly selected local metadata file.
- Live configuration should clearly show the intended cache directory and
  resolved inputs before offering any compute or launcher action.
- Expensive analysis must remain outside ordinary Streamlit rerenders. The app
  may validate metadata and render cached results, while long work is handed to
  the generic computation entry point.
- Webapp startup should accept one explicit session metadata path and initialize
  available probes, sites, source paths, and cache choices from it. Missing
  optional fields should produce actionable unavailable states rather than
  preventing unrelated cached views from opening.
- A small Python launcher should accept `--session-metadata PATH` and start the
  webapp with that explicit file. Short how-to documentation must show the
  exact command and the expected startup behavior.

### 9. Preserve reproducibility and cache integrity

- Scientific cache identity should include every input and parameter that can
  change numerical results, including site/channel definitions and population
  rules.
- Execution-only settings should remain distinguishable from scientific cache
  identity when they do not change results.
- Every result should retain schema version, analysis version, random seed,
  source identities, resolved configuration, units, and axis conventions.
- Existing atomic component writes, manifest-last publication, immutable report
  directories, checkpoint recovery, and copied-snapshot receipts should be
  preserved.
- A metadata change must never make an incompatible existing component appear
  current.

### 10. Fail early and explain configuration errors

- Validation should occur before expensive source loading or cluster
  submission.
- Errors should identify the exact probe, site, population, path, column, unit,
  or channel that is invalid.
- Duplicate stable IDs, missing files, unsupported acquisition formats,
  inconsistent sample rates, invalid channel indices, unresolved site pairs,
  and population/probe mismatches should fail closed.
- The system should not infer voltage units, sample rates, anatomical regions,
  or synchronization semantics from filenames alone.

### 11. Preserve completed scientific behavior during migration

- The approved CT026 Power, Synchrony, and ProbeB Spike-phase results remain the
  regression reference for the refactor.
- Generalization must not change numerical definitions, default bands, windows,
  trial filters, PPC inference, reliability rules, or plotting semantics unless
  a separate scientific change is documented and approved.
- A generalized CT026 description should reproduce compatible cache identities
  and outputs where the effective scientific configuration is unchanged.
- Existing completed snapshots and reports should remain inspectable even after
  the live configuration interface changes.

### 12. Organize the package by responsibility and scientific domain

- Webapp composition and visualization controls should live in a dedicated
  package rather than one large multipurpose module.
- Scientific computation should be grouped into recognizable domains such as
  LFP-LFP, Spike-LFP, population/dimensionality, regression/decoding, and basic
  spike/behavior analyses.
- Shared configuration, validation, cache I/O, plotting primitives, source
  loading, synchronization, execution, and reporting should have separate
  package boundaries where they are genuinely shared.
- File placement should communicate whether code is a pure numerical method, a
  source-format adapter, a workflow/orchestrator, a report writer, or a user
  interface.
- Dataset-specific code such as CT026 builders and profiling fixtures should be
  isolated from generic analysis packages.
- Reorganization must not be combined casually with numerical changes. Import
  moves and responsibility splits should retain behavior through regression
  tests and temporary compatibility imports where needed.

### 13. Separate reusable kernels from loading and orchestration

- Pure numerical functions should accept explicit arrays and metadata contracts
  rather than discover files or construct session paths.
- Source loaders should translate acquisition-specific files into documented
  internal structures without choosing scientific analyses.
- Workflows should connect validated metadata, loaders, numerical functions,
  caches, reports, and progress reporting without duplicating calculations.
- Plotting should consume cached result structures and should not trigger source
  loading or computation.
- Local execution, Slurm submission, resume, recovery, and rerender should be
  orchestration concerns layered above the scientific kernels.

### 14. Make analyses discoverable without central UI branching

- Each supported analysis family should expose a small, consistent description
  of its required inputs, configuration, runnable components, result artifacts,
  and available cached views.
- The launcher and webapp should use those descriptions to populate appropriate
  choices rather than accumulating large chains of analysis-specific branches.
- This does not require a dynamic plugin framework. A simple explicit registry
  may be preferable if it keeps behavior reviewable and type contracts clear.
- Adding a new analysis should not require editing an unrelated monolithic
  webapp function in many locations.

### 15. Standardize result and workflow behavior across analysis families

- Analyses should share conventions for validated configuration, dry run,
  progress events, logs, cache status, atomic publication, report directories,
  warnings, and failure states where those concepts apply.
- Result artifacts should distinguish numerical caches from human-readable
  reports and temporary execution/checkpoint state.
- Cache readers and plotters should have explicit versioned contracts rather
  than depending on incidental filenames or module globals.
- Cross-session analyses should consume documented single-session artifacts
  instead of reopening raw sources when cached sufficient statistics exist.
- Common conventions must not force unrelated analyses into one oversized base
  class or erase meaningful differences in their data shapes and units.

## Proposed target package organization

The target should organize code first by responsibility and then, for
scientific code and figures, by analysis domain. Moving today's files into new
directories without splitting their responsibilities would preserve the main
problem, so the proposal below describes ownership rather than a mechanical
rename map.

```text
src/neural_analysis/
  session/
    models.py
    json_io.py
    validation.py
    paths.py

  sources/
    contracts.py
    open_ephys/
      lfp.py
      synchronization.py
    spikeglx/
      lfp.py
      synchronization.py
    spikes/
      kilosort.py
      aligned.py
      channel_quality.py
    behavior/
      trials.py
      events.py
      treadmill.py

  synchronization/
    alignment.py
    irig.py
    manual.py

  analyses/
    lfp_lfp/
      power.py
      spectrogram.py
      wavelet_phase.py
      relative_phase.py
      synchrony.py
    spike_lfp/
      phase_sampling.py
      phase_locking.py
      ppc.py
      ppc_kernel.py
    spike_behavior/
      binning.py
      psth.py
      decoding.py
    population/
      pca.py
      decoding.py
      switch_trajectories.py
    cross_session/
      decoding.py

  artifacts/
    manifests.py
    component_cache.py
    work_cache.py
    snapshots.py

  workflows/
    lfp_summary/
      models.py
      preparation.py
      pipeline.py
      power.py
      synchrony.py
      spike_phase.py

  execution/
    launcher.py
    run_state.py
    resources.py
    slurm.py

  visualization/
    common.py
    lfp_lfp/
      power.py
      phase.py
      synchrony.py
    spike_lfp/
      phase.py
      ppc.py
    units/
      raster.py
      psth.py
    population/
      pca.py
      decoding.py
    cross_session/
      decoding.py

  reports/
    common.py
    power.py
    synchrony.py
    spike_phase.py

  webapp/
    app.py
    routes.py
    state.py
    controls.py
    data_access.py
    views/
      cached_summary.py
      unit_activity.py
      trial_activity.py
      lfp_lfp.py
      spike_lfp.py
      population.py

  cli/
    create_session_metadata.py
    run_cache.py
    launch_webapp.py

  compatibility/
    ct026.py

  profiling/
    ppc.py
    ct026.py
```

The exact filenames are provisional. The important boundaries are:

- `session` owns the versioned session description, JSON translation,
  action-specific validation, and relative-path resolution. It performs no
  scientific computation.
- `sources` translates native Open Ephys, SpikeGLX, Kilosort, aligned-spike,
  and behavioral files into documented internal data contracts. It does not
  select analyses or render figures.
- `synchronization` owns time-coordinate transformations and alignment logic
  shared across acquisition adapters. Acquisition-specific digital I/O remains
  in `sources`.
- `analyses` contains numerical methods over explicit arrays and tables. These
  modules do not read session paths, write caches, launch processes, or import
  Streamlit.
- `artifacts` owns reusable manifest, cache, checkpoint, snapshot, and receipt
  mechanics. It must not know CT026 paths or implement scientific formulas.
- `workflows` joins validated session metadata, source adapters, numerical
  analyses, artifacts, and progress reporting for a scientific pipeline.
- `execution` owns generic restartable launcher state, process/resource
  measurement, and Slurm submission rather than embedding those concerns in
  numerical or analysis-specific modules.
- `visualization` contains reusable Matplotlib figure construction. It accepts
  explicit results and plot context, performs no source discovery, and has no
  Streamlit dependency.
- `reports` selects figures, captions, summaries, and immutable report outputs.
  It may call `visualization` but must not recalculate numerical results.
- `webapp` owns Streamlit state, controls, routing, cached artifact selection,
  and view composition. A view calls `visualization` or reads a validated
  artifact; it does not become a second implementation of an analysis.
- `cli` contains thin user entry points. Argument parsing should delegate
  immediately to `session`, `workflows`, or the webapp launcher.
- `compatibility` temporarily isolates CT026 builders and old import paths.
  Dataset-specific defaults must not leak back into generic packages.
- `profiling` contains developer performance harnesses and representative
  fixtures, not production numerical definitions.

### Dependency direction

`session`, `sources`, `analyses`, and `artifacts` form the foundational layer.
Source adapters and numerical analyses are peers: workflows pass adapter output
into numerical functions, but an analysis must not import an Open Ephys or
SpikeGLX reader. `workflows` may depend on all four foundational packages, and
`execution` invokes workflows through narrow run contracts.

`visualization` consumes documented analysis-result or artifact contracts and
is used by both `reports` and `webapp`. `reports` may depend on `artifacts` and
`visualization`. `webapp` may depend on session metadata, validated artifacts,
visualization, and narrow workflow command interfaces. Finally, `cli` delegates
to session creation, workflows, or the webapp launcher.

Dependency arrows must not point back upward: foundational modules must not
import workflows, reports, Streamlit views, or command-line modules. A generic
`utils.py` package should be avoided; genuinely shared code should be named for
its responsibility and kept near the lowest layer that owns it.

### Current-module disposition

The largest current modules need responsibility splits rather than single-file
moves:

| Current area | Proposed ownership |
| --- | --- |
| `psth_webapp.py` | `webapp/app.py`, domain view modules, `webapp/data_access.py`, and thin calls into analysis/visualization packages |
| `lfp_summary_webapp.py` | cached-summary view state in `webapp/views/cached_summary.py`; snapshot validation in `artifacts/snapshots.py`; command construction in workflow/CLI code |
| `unit_spike_plotting.py` | pure figures split across the `visualization/units`, `visualization/lfp_lfp`, `visualization/spike_lfp`, and `visualization/population` domain packages |
| `lfp_summary_runtime.py` | component-specific Power, Synchrony, and Spike workflow modules plus shared preparation |
| `lfp_summary_ppc_runtime.py` | restartable execution under `workflows/lfp_summary/spike_phase.py`; numerical operations remain under `analyses/spike_lfp` |
| `lfp_spike_phase_launcher.py` | generic launcher lifecycle, persisted run state, resource measurement, and a thin CLI entry point |
| validation/report modules | action validation stays with workflows; report construction moves to `reports`; pure figures move to `visualization` |
| `lfp_loading.py` and sync modules | acquisition adapters in `sources` plus shared time alignment in `synchronization` |
| PCA modules | `analyses/population`; their figures move to `visualization/population.py` |
| spike/behavior modules | source-table loading under `sources`; numerical binning/decoding/PSTH under `analyses/spike_behavior` |
| CT026 profile modules | temporary `profiling/ct026.py` and `compatibility/ct026.py`, with no generic production defaults |

### Migration constraints

- Reorganization should proceed as a series of behavior-preserving moves and
  splits, not a repository-wide rename followed by simultaneous rewrites.
- The intended migration sequence is: session metadata and acquisition
  adapters; generic cache/webapp entry points; webapp and visualization
  decomposition; scientific analysis package moves; launcher/runtime
  decomposition; then legacy cleanup and final documentation consolidation.
- Existing public imports and `python -m` commands should have temporary thin
  compatibility modules while callers and tests migrate.
- Tests should mirror the target packages and compare approved CT026 arrays,
  manifests, reports, and figures before obsolete imports are removed.
- Pure analysis and visualization splits should occur before workflow and
  launcher decomposition, so orchestration can target stable lower-level
  interfaces.
- Performance-sensitive PPC kernels should move only after characterization
  tests capture numerical equality, deterministic seeds, checkpoint identity,
  and representative runtime/memory behavior.
- A module should normally have one primary reason to change. File-size targets
  may guide review, but should not cause arbitrary splits of cohesive numerical
  code.

Before work packages are frozen, the repository inventory must classify every
current neural-analysis module and inspect representative Open Ephys and
SpikeGLX session layouts. The inventory should include repository imports,
tests, documented commands, notebooks where searchable, likely external entry
points, source metadata files, sorter layouts, aligned-spike files,
synchronization inputs, channel-quality files, and augmented trial tables.

### Organization decisions

- Use one Streamlit application shell for user readability. `webapp/app.py`
  should load the session, show its validation/availability state, and delegate
  through a small explicit route registry. Each scientific view remains in its
  own domain-named module. This preserves one obvious launch command without
  recreating one enormous application file.
- Cached production summaries and interactive exploratory views may coexist in
  that shell, but their routes must label the distinction clearly. A cached
  result view must not silently perform exploratory computation.
- Reusable Matplotlib visualization and human-readable report generation remain
  separate. Reports may compose approved visualization functions with captions
  and provenance, while webapp views may compose the same functions with
  interactive controls.
- Tests should mirror the target package boundaries. Cross-package CT026
  equivalence tests remain in an integration suite rather than being assigned
  to one implementation module.
- Legacy import wrappers are temporary. They should be removed after repository
  callers, tests, documented commands, and required resume paths have migrated,
  with the removal milestone recorded in the eventual implementation plan.

### Classify existing modules before moving them

Classification is a migration decision, not a scientific reclassification. It
asks what support obligation each current module has and therefore whether it
belongs in the new architecture:

1. **Supported production:** used by an approved cache/report workflow and
   protected by numerical or integration tests. Move or split it while keeping
   behavior and compatibility wrappers.
2. **Supported exploratory:** intentionally exposed through the current webapp
   or a documented analysis workflow and covered by meaningful tests. Move it
   into the appropriate analysis/view packages, but do not silently promote its
   outputs to approved production artifacts.
3. **Shared infrastructure:** source loading, synchronization, artifact I/O, or
   plotting infrastructure with active callers. Move it to the corresponding
   responsibility package.
4. **Dataset-specific compatibility or profiling:** still needed to reproduce,
   inspect, or benchmark CT026 behavior but unsuitable as a generic default.
   Isolate it under `compatibility` or `profiling`.
5. **Unreferenced or superseded candidate:** no known repository callers,
   tests, documented command, or unique behavior. Do not move it into the clean
   package automatically. First check notebooks and external usage, then either
   preserve it temporarily with a clear legacy label or propose deletion for
   explicit approval.

The current scan suggests that `behavior_pynap.py`,
`spike_behavior_analysis.py`, `spike_behavior_binning.py`,
`modified_sinc_smoother.py`, and the empty
`plot_single_session_analysis.py` need category-5 review. This is not a deletion
decision. In contrast, `analog_treadmill_decode.py` is exercised by manual
synchronization tests and should initially be treated as supporting
infrastructure even if its final ownership changes.

### Documentation contract for new packages

Every newly created subfolder must contain a short README. Documentation is a
two-stage responsibility rather than a final cleanup-only task:

1. When the folder is introduced, its README records the package purpose,
   dependency boundary, public entry points, and a one-line responsibility for
   every file initially placed there.
2. After migrations and responsibility splits stabilize, a documentation pass
   verifies the inventory and expands computationally intensive entries with
   the algorithmic stages, major array shapes and units, determinism/seed
   behavior, cache/checkpoint strategy, parallelization, and expected runtime
   or memory characteristics where known.

READMEs should explain module-level ownership and computational flow without
duplicating function contracts. Function inputs, outputs, shapes, axes, and
physical units remain authoritative in Python docstrings. A README change is
part of the same migration that adds, removes, or substantially changes a file
in its folder; the final cleanup verifies completeness rather than reconstructing
all documentation from scratch.

## Conceptual metadata relationships

The following is illustrative vocabulary, not a frozen file schema:

```text
subject
  session
    trial table
    probes
      probe identity and source files
      sorter and aligned spikes
      channel/anatomy metadata
    sites
      stable site identity
      probe reference
      saved-channel index
      region and display label
      units and sampling metadata
    site pairs
      explicit references to two site identities
    populations
      probe reference
      quality-selection rules
    scientific analysis settings
    execution profile reference
```

This separation is important: the probe describes the physical/data source;
the site describes the analyzed LFP location; the population describes selected
units; and the execution profile describes how the work runs.

## Additional refactor opportunities

The current module inventory suggests several supporting improvements that fit
the goals above:

### Decompose oversized integration modules

`psth_webapp.py`, `lfp_summary_ppc_runtime.py`, `unit_spike_plotting.py`,
`lfp_summary_runtime.py`, `lfp_summary_webapp.py`, and the launcher/validation
modules currently contain multiple responsibilities. They are candidates for
separation by route, scientific method, I/O boundary, execution concern, or
report concern. Line count alone is not a reason to split a file; the split
should follow stable data contracts and reduce unrelated reasons to change.

### Introduce acquisition adapters behind one source contract

Open Ephys-derived and SpikeGLX loading, synchronization, sample-rate metadata,
and voltage-unit handling should meet one validated source interface. Analysis
code should not infer the acquisition family repeatedly from filenames. The
adapter boundary should preserve native units and explicit timing conventions.

### Centralize metadata validation and schema migration

JSON decoding, schema validation, default handling, path resolution, and legacy
schema migration should have one owner. The webapp and command-line tools should
not implement separate interpretations of the same metadata. Migration must be
versioned and conservative: absent execution-only fields may be defaultable,
while absent scientific or provenance fields should fail closed.

Validation should support at least two distinct questions: whether a file is a
well-formed session description, and whether it is complete enough for a
specific requested action. A partial one-probe description may be valid for
inspection or one-probe analysis even though it cannot support a two-site
Synchrony request.

### Separate environment-specific locations from scientific metadata

The logical session description should remain portable by storing source paths
relative to the session metadata location. Because the current local and
cluster session trees share the same internal layout, the initial design should
not add a root-mapping system. Any resolved execution configuration should
still record exact absolute paths for provenance. External data roots can be
designed later if a real session requires them.

### Consolidate launcher and execution behavior

Local commands and Slurm scripts should call the same Python entry point with
the same validated metadata and scientific options. Shell scripts should supply
cluster resources and environment setup, not reconstruct scientific settings.
Launcher output, dry-run summaries, resume receipts, and failure messages should
follow one convention across analysis families.

### Mirror package boundaries in tests

Tests should be organized around the same public contracts as the refactored
packages: metadata/configuration, source adapters, pure computation, cache I/O,
workflows, reports, and webapp views. CT026 end-to-end regression tests should
remain as scientific equivalence gates, while smaller synthetic tests protect
generic behavior for arbitrary probe and site names.

### Define a compatibility and deprecation boundary

Existing import paths, saved snapshots, run directories, and cluster resume
commands may outlive the reorganization. The refactor should identify which
interfaces need temporary forwarding wrappers, which artifacts require readers,
and which CT026-only helpers can be retired after equivalence is demonstrated.
Compatibility code should be isolated and time-bounded rather than becoming the
new permanent architecture.

## Desired operator experience

Without prescribing implementation order, the eventual workflow should make
the following actions straightforward:

1. Create or select a session-analysis description for a new recording.
2. Validate all paths, site channels, metadata columns, units, and pair
   references without computing.
3. Review the resolved configuration in human-readable form.
4. Launch one component locally or submit the same component through Slurm.
5. Resume an interrupted run using only its saved run directory.
6. Copy a completed snapshot locally and verify its receipt.
7. Open the webapp, select the description or snapshot, and see metadata-derived
   probe/site/population labels.
8. Add a new mouse or session without editing shared analysis code.

## Non-goals for the design-intake phase

- Do not freeze the exact JSON schema, field names, or migration policy yet.
- Do not introduce a database or service merely to replace the intended simple
  file-based session metadata.
- Do not freeze exact field names, command-line syntax, class boundaries, or
  module ownership yet.
- Do not redesign the approved numerical methods as part of generalization.
- Do not implement automatic anatomical registration or infer regions from
  channel depth.
- Do not combine ProbeA and ProbeB spike populations without a separate
  scientific requirement.
- Do not start WP13 amplitude-threshold implementation implicitly.
- Do not make Streamlit the sole configuration store or the process responsible
  for long-running computation.

## Settled design decisions

- The canonical session metadata file normally lives at
  `<session_home>/neural_session.json`. An explicit external metadata path may
  be used when the session directory is read-only or otherwise unsuitable.
- Session metadata is explicitly maintained by the user. Analysis and copy
  commands must not silently rewrite it. Manual JSON editing is acceptable for
  the initial design.
- Paths beneath the session home should be session-relative. Local and cluster
  roots are deployment concerns and must not be encoded as scientific identity.
  In the initial design, relative paths resolve from the directory containing
  `neural_session.json`, so matching workstation and cluster session layouts do
  not require a separate root-mapping system.
- Initial source paths are required to remain within the session tree. Support
  for external data roots, absolute source paths, or root remapping is deferred
  until a real session requires it.
- Session JSON contains session identity, acquisition family, session-relative
  source paths, probes, sites, populations, and approved cache references.
  Versioned code presets own analysis defaults such as bands, windows, filters,
  shuffle defaults, and plot defaults. Every saved run records the fully
  resolved combination so defaults never become invisible provenance.
- Cache references must be fully descriptive: analysis, probe, population, and
  shuffle tier where applicable. There is one user-approved result per such
  identity rather than an in-file history of candidate runs.
- To keep the common file simple, the session JSON should normally point only
  to the approved local snapshot. The snapshot manifest and copy receipt remain
  authoritative for the producing cluster path, resolved configuration,
  checksum, and transfer provenance. The session schema may permit optional
  cluster fields later, but they are not required for routine use.
- Optional unavailable values use JSON `null`. A probe that does not exist is
  omitted rather than represented by a dummy or all-null probe entry.
- Probe identifiers are arbitrary stable strings. Names such as `ProbeA`,
  `ProbeB`, and `Probe1` are all valid and carry no anatomical meaning.
- Each probe records one selected LFP source and one selected sorter generation
  in the initial schema. Retaining alternative source histories is out of scope.
- Each LFP site records a stable site ID, display label, probe ID, saved-channel
  index, and optional region.
- Sample rate and voltage units are extracted by the acquisition adapter from
  the designated reference file. The metadata creator must not require the user
  to look up or transcribe them. Extracted values should still be saved with
  validated configurations and results for provenance.
- The metadata creator is a straightforward Python file with clearly labeled,
  hardcoded variables for the user to edit and a `main` function that validates
  them and writes the JSON. It is not an interactive wizard and does not launch
  numerical work.
- Acquisition family is session-wide in the initial schema. It explicitly
  selects Open Ephys or SpikeGLX. Supporting mixed-acquisition sessions would
  require a later schema decision.
- The authoritative Open Ephys and SpikeGLX metadata files and their exact
  extraction rules must be chosen from representative real session layouts
  before acquisition-adapter implementation begins.
- The webapp is launched through a short Python wrapper accepting
  `--session-metadata PATH`. A concise how-to document must include the exact
  invocation.
- A valid but incomplete metadata file opens normally in the webapp. Only the
  views and actions whose required inputs are unavailable should be disabled.
- Metadata creation always validates schema and internal relationships. Checking
  whether non-null paths currently exist is an optional validation mode, so a
  session description may be prepared before every referenced file is copied.
  When the user explicitly requests the filesystem check, any missing required
  non-null path fails that check rather than producing only a warning.
- Power and Synchrony have independent approved cache references. Shuffle tiers
  apply only to analyses, such as Spike phase, whose scientific configuration
  actually includes shuffle count.
- Existing CT026 snapshots remain readable through compatibility readers and do
  not need to be rewritten into the new session schema. New runs use new
  metadata/artifact contracts, while existing resume operations retain their
  saved run configuration. Removing a compatibility reader requires explicit
  approval after its consumers are retired.
- Behavior-preserving migration requires exact equality for deterministic
  arrays, identities, seeds, manifests, and selection logic. Tight numerical
  tolerances are acceptable only where exact equality is genuinely unstable;
  reports receive structural/content tests, figures receive rendering smoke
  tests, and performance-sensitive PPC paths retain representative runtime and
  peak-memory regression checks.

Manual metadata maintenance is the simplest initial workflow, but it has one
predictable failure mode: a mistyped or stale snapshot path. A possible later
quality-of-life addition is an explicit `register-cache` command that validates
one copied snapshot and prints or applies the exact JSON change. It should be an
opt-in edit, not an automatic side effect of computation or transfer.

## Open design questions

### Required pre-planning investigations

1. Which acquisition reference file is authoritative for extracting sample
   rate, voltage units, and any required raw-value scaling for Open Ephys and
   SpikeGLX sessions?
2. How consistent are sorter, aligned-spike, synchronization, channel-quality,
   and augmented-trial-table layouts across mice and recording generations?

### Explicitly deferred design

3. When channel-to-region grouping becomes a priority, will its authoritative
   source be manually selected channel ranges, channel-quality metadata, or a
   separate anatomical registration artifact?

## Design decisions to record later

As decisions are made, this document should record the rationale for:

- metadata granularity and ownership;
- stable identifiers and display labels;
- storage format and schema-version policy;
- relative-path resolution and portability rules;
- validation and dry-run behavior;
- local versus Slurm execution profiles;
- migration and backward-compatibility policy;
- UI configuration and snapshot-selection behavior; and
- the boundary between shared generic code and dataset-specific adapters.

Only after those decisions and open questions are sufficiently resolved should
this design intake be converted into an implementation plan with tests,
packages, ownership, sequencing, and performance gates.
