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

- Each probe/LFP source should explicitly declare its acquisition family, such
  as Open Ephys or SpikeGLX. The system should not infer this scientific I/O
  contract from a filename.
- Each session should normally have its own JSON metadata file. Pointing a
  restarted webapp at that file should restore the session inputs and known
  result locations quickly without re-entering paths.
- The metadata may contain optional references to copied/local cache snapshots
  and cluster cache or run locations, including the approved 100-shuffle
  preview and 1,000-shuffle final results when they exist.
- Missing results are valid metadata state. A new session, a one-probe session,
  or a session whose cache has not yet been computed must still have a useful
  metadata description.
- A separate simple Python entry point should create a new session metadata
  file. It may record incomplete or unavailable values explicitly rather than
  inventing paths, probes, sites, or results.
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
  sorter, aligned-spike file, acquisition format, and relevant metadata files.
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

The logical session description should be portable, while local and cluster
root mappings are deployment concerns. This avoids duplicating nearly identical
JSON files solely because `/home/...` and `/gs/...` prefixes differ. Any
resolved execution configuration should still record exact absolute paths.

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

## Open design questions

### Metadata ownership and location

1. Where should each session JSON live, and what filename convention should it
   use? Should it reside inside the session directory, in a separate metadata
   index, or support both with one canonical location?
2. Is the JSON primarily user-authored configuration, or may launch/copy tools
   update its cache references automatically? Automatically mutating the same
   file on the workstation and cluster risks divergent copies; a generated run
   registry or explicit metadata-update command may be safer.
3. Should unavailable values be represented as explicit JSON `null`, omitted
   keys, or typed status objects? The choice affects validation, human editing,
   and backward-compatible schema migration.
4. Which values belong in every session JSON, and which should come from a
   versioned project-level defaults file? Hidden code defaults should not regain
   control through the back door.

### Probe, source, and site identity

5. Should probe identifiers be arbitrary stable strings such as `ProbeA` and
   `Probe1`, or should the project enforce one naming convention?
6. Can one probe have more than one LFP source or sorter generation, and if so,
   does the metadata select one active source or retain named alternatives?
7. Are sample rate and voltage unit always available from acquisition metadata,
   or must the JSON record explicitly confirmed values?
8. Before channel-region grouping is implemented, what minimum site information
   must be entered for LFP analyses: stable site ID, display label, probe,
   channel, and optional region?
9. How consistent are sorter, aligned-spike, synchronization, channel-quality,
   and augmented-trial-table layouts across mice and recording generations?

### Cache and run references

10. Are the 100- and 1,000-shuffle references stored per probe/population? A
    ProbeA preview and ProbeB preview cannot safely share one undifferentiated
    `preview_cache` property.
11. Should metadata point to the immutable copied `cache_snapshot`, the original
    cluster cache, the timestamped launcher run directory, or all three as
    separately typed references?
12. May there be multiple valid 100- or 1,000-shuffle runs for one session and
    probe? If so, should metadata retain a history, a user-selected active run,
    or only one canonical approved result?
13. Power and Synchrony are not defined by shuffle count, even when their files
    are copied beside a 1,000-shuffle Spike result. Should they have independent
    cache references or be referenced through a complete summary snapshot?
14. After a user copies a result from the cluster, should a tool verify the
    receipt and update the local cache reference, or should metadata editing and
    transfer remain separate explicit actions?

### Entrypoints and webapp behavior

15. Should the metadata-creation entry point be interactive prompts, command-
    line flags, a template generator followed by manual editing, or a combination
    with `--non-interactive` support for reproducibility?
16. May the creator inspect a session directory and propose paths, provided the
    user explicitly confirms them, or should the first version accept only
    entered values?
17. How should the webapp receive its metadata path: a command-line argument
    after `streamlit run`, an environment variable, or a small startup chooser?
18. When the JSON is valid but incomplete, should the webapp open every available
    feature and disable only unsupported actions, or require the user to choose
    an explicitly partial mode?

### Portability, compatibility, and future scope

19. Should local/cluster root mappings live in user-specific environment files,
    command-line arguments, or another explicit deployment description?
20. How should legacy CT026 snapshots whose saved schema predates newer
    execution-only fields be represented after migration?
21. Which pieces of metadata may be proposed by a discovery tool, and which must
    always receive explicit human confirmation?
22. When channel-to-region grouping becomes a priority, will its authoritative
    source be manually selected channel ranges, channel-quality metadata, or a
    separate anatomical registration artifact?

## Design decisions to record later

As decisions are made, this document should record the rationale for:

- metadata granularity and ownership;
- stable identifiers and display labels;
- storage format and schema-version policy;
- path-root mapping and portability rules;
- validation and dry-run behavior;
- local versus Slurm execution profiles;
- migration and backward-compatibility policy;
- UI configuration and snapshot-selection behavior; and
- the boundary between shared generic code and dataset-specific adapters.

Only after those decisions and open questions are sufficiently resolved should
this design intake be converted into an implementation plan with tests,
packages, ownership, sequencing, and performance gates.
