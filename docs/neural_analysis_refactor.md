# Neural Analysis Organization Refactor Design

## Status and purpose

U1-U4 are implemented. The repository now has one compact version-2 session-
metadata path for the existing webapp and Spike-phase/PPC launcher, plus a
concise operational guide and example metadata. The user explicitly accepted
the current version-2 U4 workflow on 2026-09-25.

This document defines the approved design for the next phase: reorganizing the
whole `src/neural_analysis` package for scientific readability. It describes
ownership and dependency boundaries, not an instruction to begin coding.
R0 and every source work package remain separately gated; acceptance of this
design does not authorize implementation.

The refactor changes organization only. It adds no scientific method, source
format, metadata field, command, plot, report, cache behavior, recovery mode,
or supported edge case.

Detailed implementation history remains in
`docs/neural_analysis_refactor_execution_log.md`. Scientific definitions and
approved result evidence remain in `docs/Tasks_neural.md`.
`docs/SoftwareDesign.md` governs implementation style.

## Current system

### Implemented usability path

The supported new-session path is:

```text
neural_session.json
  -> load, validate, and resolve session sources
  -> launch the existing metadata-driven webapp
  -> inspect existing live or cached views
  -> dry-run or launch the existing Spike-phase/PPC computation
```

The principal user-facing files are:

- `src/neural_analysis/README.md`;
- `src/neural_analysis/session_metadata.py`;
- `src/neural_analysis/session_metadata_cli.py`;
- `src/neural_analysis/psth_webapp.py`;
- `src/neural_analysis/lfp_summary_session.py`;
- `src/neural_analysis/lfp_spike_phase_launcher.py`; and
- `src/shell_scripts/hpc_ppc.sh`.

Session metadata contains one session name, one acquisition family, and
probe-owned source paths, site names, saved-channel indices, and optional unit-
channel restrictions. Probe dictionary keys and site names are their display
labels; the file does not duplicate separate label, population, or channel-
group records. Scientific settings remain in versioned code and explicit run
arguments. Resume, report recovery, and report rerender use saved run
configuration rather than rereading live session metadata.

### Preserved scientific baseline

Organization work must preserve the scientific baseline established before
U1-U4:

- corrected Open Ephys gain, offset, voltage-unit, and sample-rate semantics;
- current source-value semantics and source fingerprints;
- immutable historical artifacts and explicit legacy interpretation;
- current cache, checkpoint, manifest, and configuration identities;
- the approved Power, Synchrony, and Spike-phase/PPC definitions;
- current seeds, trial filters, windows, frequency bands, and unit-selection
  rules; and
- existing saved-result and resume compatibility.

The detailed definitions and evidence remain in `docs/Tasks_neural.md` and
`docs/neural_analysis_refactor_execution_log.md`. Moving code does not reopen
them.

### Organization pressure

The package has grown through successful feature work, but several modules now
mix responsibilities that a scientist would reasonably expect to find
separately. At the planning baseline:

| Module | Lines | Main mixed responsibilities |
| --- | ---: | --- |
| `psth_webapp.py` | 6,750 | session inputs, cached loading, computations, controls, and all interactive views |
| `lfp_summary_ppc_runtime.py` | 6,103 | PPC planning, allocation, execution, checkpointing, parallel workers, and reduction |
| `lfp_spike_phase_launcher.py` | 3,717 | parsing, preflight, state, execution, recovery, persistence, and resource measurement |
| `unit_spike_plotting.py` | 3,082 | unit, trial, LFP, Spike-LFP, and population plots plus computations |
| `lfp_summary_runtime.py` | 2,958 | Power, Synchrony, and Spike-phase preparation and runtime behavior |
| `lfp_summary_webapp.py` | 2,948 | snapshot validation, artifact selection, plotting, commands, and Streamlit rendering |
| `spike_behavior_pynapple.py` | 2,405 | loading, trial masks, binning, decoding, publication, plotting, and legacy CLI behavior |

File size alone is not a reason to split a module. These modules are candidates
because their contents have different scientific or operational reasons to
change.

## Design goals

### 1. Make scientific responsibilities easy to locate

A scientist looking for an LFP power calculation, spike-behavior decoder,
population PCA method, synchronization transform, or plot should be able to
find it from the package tree without first understanding the webapp or LFP-
summary execution machinery.

### 2. Keep the package shallow and domain-oriented

Organize reusable science by recognizable domains. Keep workflow-specific
cache, execution, report, and profiling code with the workflow that owns it.
Do not create global architecture layers merely because similar nouns appear in
different workflows.

### 3. Separate calculations, loading, plotting, and user interfaces

- Scientific functions accept explicit arrays, tables, and small
  configurations.
- Loading functions translate existing files into documented structures.
- Plotting functions accept already prepared results.
- Webapp functions collect controls and compose existing operations.
- Command entry points parse arguments and delegate to existing workflows.

This separation is applied only where current code already demonstrates more
than one responsibility. It is not a requirement to create a matching module
for every conceptual layer.

### 4. Preserve behavior and provenance

The current calculations, shapes, axes, physical units, seeds, source
identities, fingerprints, cache formats, checkpoint formats, report content,
commands, and supported UI actions are the reference. Reorganization must not
silently improve, generalize, or reinterpret them.

### 5. Preserve recognizable user entry points

Existing root entry modules remain easy to find:

- `psth_webapp.py` for the Streamlit application;
- `lfp_spike_phase_launcher.py` for Spike-phase/PPC execution; and
- `sync_ephys.py` for its current hardcoded synchronization script and callable
  workflows.

These may become thin forwarding modules. Documented command invocations remain
unchanged, and `sync_ephys.py` retains its current direct module dispatch and
callable behavior. This refactor does not invent a synchronization CLI.

### 6. Delete only code proven unused

Unused prototypes and scratch modules should be deleted rather than moved into
a permanent `legacy` package. Repository search is necessary but not
sufficient: deletion also requires a fresh audit, explicit confirmation that
no external script depends on the module, tests for the intentional removal,
and separate approval.

## Approved target organization

The target is deliberately shallower than the former proposed architecture.
Exact filenames may be refined during the owning work package, but the domain
boundaries are approved.

```text
src/neural_analysis/
  README.md
  __init__.py

  session_metadata.py
  session_metadata_cli.py

  psth_webapp.py
  lfp_spike_phase_launcher.py
  sync_ephys.py

  synchronization/
    __init__.py
    alignment.py
    manual.py
    spikeglx.py

  lfp/
    __init__.py
    config.py
    loading.py
    power.py
    spectrogram.py
    phase.py
    synchrony.py
    plotting.py

  spike_behavior/
    __init__.py
    loading.py
    trials.py
    binning.py
    decoding.py
    psth.py
    plotting.py

  spike_lfp/
    __init__.py
    hilbert.py
    phase_locking.py
    ppc.py
    ppc_kernel.py
    plotting.py

  population/
    __init__.py
    pca.py
    decoding.py
    switch_trajectories.py
    cross_session.py
    plotting.py

  lfp_summary/
    __init__.py
    models.py
    session.py
    preparation.py
    pipeline.py
    payloads.py
    cache.py
    work_cache.py
    power_runtime.py
    synchrony_runtime.py
    spike_phase_runtime.py
    ppc_planning.py
    ppc_execution.py
    plotting.py
    power_validation.py
    synchrony_validation.py
    spike_phase_validation.py
    snapshot.py
    cache_relocation.py
    ppc_profile.py
    ct026_profile_adapter.py
    ct026_profile_locks.py
    ct026_profile_runner.py

  webapp/
    __init__.py
    app.py
    session_inputs.py
    data_loading.py
    unit_views.py
    lfp_views.py
    spike_lfp_views.py
    population_views.py
    summary_view.py
```

This tree is a responsibility map, not a mandate to create every file before it
has content. A work package creates only the modules needed for its current
split. Empty scaffolding and symmetry-only modules are forbidden.

The R6 launcher-internal filenames are intentionally omitted until its caller
and helper-group review identifies the smallest readable split. That review
may also retain cohesive launcher code in the root entry point.

Each created package has a minimal `__init__.py`, normally containing only a
package docstring. Do not add wildcard imports or broad convenience re-exports.
Canonical imports should identify the module that owns a function. A narrow
re-export requires an intentionally approved public API.

## Package responsibilities

### Session metadata

`session_metadata.py` remains the direct user-facing metadata API. Its compact
version-2 loading, structural validation, contained path resolution, probe
lookup, and action-availability responsibilities are cohesive. Version 1 was
explicitly replaced and rejected before the organization refactor; it is not a
compatibility target. The module should not be converted into a multi-file
schema framework merely to match the rest of the tree.

Some version-1-shaped records and properties currently derive populations,
channel groups, duplicate labels, and absent fields for existing internal
callers. R0 classifies those surfaces before R4/R5 migrate internal callers to
the direct version-2 probe/site model. Test-only or repository-private adapters
should be removed through the approved cleanup gate rather than defining the
target architecture.

### Synchronization

`synchronization` owns timestamp decoding, sample/time transformations, manual
alignment, and acquisition-specific digital synchronization input. It does not
load scientific signals for analysis or choose downstream analyses.

### LFP

`lfp` owns the small immutable configuration records directly consumed by LFP
calculations, LFP source loading, and reusable Power, spectrogram, phase, and
Synchrony calculations. In particular, `AnalysisWindowConfig`,
`FrequencyBandConfig`, and `PowerAnalysisConfig` belong in `lfp/config.py`.
Numerical functions operate on explicit inputs. The package does not know about
Streamlit, launchers, cache publication, or CT026 run directories.

### Spike behavior

`spike_behavior` owns sorter/aligned-spike loading used by the existing
behavior analyses, trial classification, spike/lick binning, behavioral
decoding, PSTH calculation, and the corresponding reusable plots. Loading,
calculation, plotting, and publication remain separate modules rather than one
all-purpose session class.

### Spike-LFP

`spike_lfp` owns Hilbert and wavelet phase sampling, phase-locking calculations,
PPC statistics, and performance-sensitive PPC kernels. It contains numerical
methods, not session discovery, checkpoint policy, or cluster execution.

### Population

`population` owns PCA, decoding, switch trajectories, cross-session summaries,
and their plots. Cross-session loading/publication stays explicit and does not
become a generic artifact framework.

### LFP summary

`lfp_summary` owns the existing production summary workflow: workflow-specific
session and execution configuration, preparation, component assembly, caches,
checkpoint work, Power/Synchrony/Spike-phase runtime composition, validation
reports, snapshot inspection, cache relocation, launcher internals, and
profiling. It imports the three shared LFP configuration records from `lfp`
rather than defining competing records. These concerns remain workflow-specific
because no second current workflow justifies generic artifact or execution
infrastructure.

### Webapp

`webapp` owns Streamlit caching, controls, rendering, and composition of the
existing views. `app.py` is a direct application coordinator, not a route or
plugin framework. Domain view modules call the established scientific and
plotting functions; they do not reimplement analyses. The metadata route uses
the direct version-2 probe/site model, while the existing manual route remains
an explicitly separate compatibility path.

## Dependency direction

```text
session metadata
       |
       v
synchronization and scientific domains
       |
       v
LFP-summary workflow
       |
       +----------------+
       v                v
    webapp          root entry modules
```

The following rules make that diagram concrete:

- Session metadata imports no analysis, UI, cache, or launcher code.
- Scientific-domain modules may use NumPy, SciPy, Pandas, Pynapple,
  scikit-learn, and narrowly named functions from another scientific domain
  when the dependency is real.
- Scientific-domain modules never import Streamlit, command modules, or the
  LFP-summary workflow.
- `lfp.config` owns `AnalysisWindowConfig`, `FrequencyBandConfig`, and
  `PowerAnalysisConfig`. LFP calculations import them from that lower-level
  owner; LFP-summary models may expose compatibility aliases to the same class
  objects but do not redefine them.
- `lfp_summary` may depend on session metadata, synchronization, and scientific
  domains.
- `webapp` may depend on session metadata, scientific domains, and
  `lfp_summary`.
- Root entry modules may assemble concrete dependencies and delegate inward.
- No foundational package imports the webapp or a root entry module.

Protocols are introduced only when two current implementations already share
one behavior and a plain callable would be less readable. No registry, base-
analysis hierarchy, dependency-injection container, or plugin discovery is
planned.

## Current-module disposition

| Current modules | Approved direction |
| --- | --- |
| `session_metadata.py`, `session_metadata_cli.py` | Remain top-level and direct. |
| `ephys_sync_utils.py`, `manual_session_synchronization.py`, `spikeglx_sync_io.py` | Move or split under `synchronization` at existing source/alignment boundaries. |
| `analog_treadmill_decode.py` | Keep in place pending R0 callable-level review. Current repository evidence shows one test-only conversion function and three signal-analysis functions with no callers; do not create a synchronization module merely to house them. |
| `lfp_loading.py`, `lfp_power_summary.py`, `lfp_spectrogram.py`, `lfp_phase_clustering.py`, `lfp_synchrony_summary.py` | Move under `lfp`; move the three LFP-facing configuration records currently in `lfp_summary_models.py` to `lfp/config.py`; separate plotting or persistence only where currently mixed. |
| `spike_behavior_pynapple.py`, `unit_spike_loading.py`, `psth_behavior.py` | Split under `spike_behavior` by loading, trials, binning, decoding, PSTH, plotting, and publication. |
| `spike_lfp_hilbert_phase.py`, `spike_lfp_phase_locking.py`, `spike_lfp_summary.py`, `lfp_summary_ppc_kernel.py` | Move numerical responsibilities under `spike_lfp`. |
| `population_pca.py`, `population_pca_decoding.py`, `population_pca_switch_trajectories.py`, `plot_cross_session_analysis.py` | Move or split under `population`. |
| `unit_spike_plotting.py` | Split among the scientific domains represented by its current plots. |
| `lfp_summary_models.py`, `lfp_summary_session.py`, `lfp_summary_preparation.py`, `lfp_summary_pipeline.py`, `lfp_summary_payloads.py`, `lfp_summary_io.py`, `lfp_summary_work_cache.py` | Move workflow-specific responsibilities under `lfp_summary` without changing contracts. `lfp_summary_models.py` retains compatibility aliases for the three records moved to `lfp/config.py`. |
| `lfp_summary_runtime.py` | Split into Power, Synchrony, Spike-phase, and the smallest genuinely shared preparation/runtime code. |
| `lfp_summary_ppc_runtime.py` | Separate planning/allocation from execution/checkpoint behavior; keep parallel worker details with execution unless a measured readability problem remains. |
| `lfp_power_validation.py`, `lfp_synchrony_validation.py`, `lfp_spike_phase_validation.py` | Move under `lfp_summary`; split report construction only when it makes the existing flow clearer. |
| `lfp_summary_plotting.py` | Remain workflow-specific plotting under `lfp_summary`. |
| `lfp_summary_webapp.py` | Move snapshot inspection to `lfp_summary/snapshot.py` and UI composition to `webapp/summary_view.py`. |
| `psth_webapp.py` | Become the thin documented entry point over domain view modules in `webapp`. |
| `lfp_spike_phase_launcher.py` | Remain the documented entry point; move only established preflight/state/execution groups under `lfp_summary`. |
| `sync_ephys.py` | Remain the current root script/function entry point over synchronization modules. Preserve its callable signatures and direct module dispatch; its hardcoded session, recording, and sorter paths are replaceable local examples rather than strict compatibility contracts. Do not add a parser. |
| `lfp_summary_cache_relocation.py` | Move under `lfp_summary`; do not generalize it. |
| `lfp_summary_ppc_profile.py`, `lfp_summary_ct026_profile_adapter.py`, `lfp_summary_ct026_profile_locks.py`, `lfp_summary_ct026_profile_runner.py` | Move initially to like-named modules under `lfp_summary`; remain visibly profile- and dataset-specific. Consolidation requires a later concrete readability case. |

## Compatibility policy

Compatibility is based on an explicit R0 inventory rather than an unlimited
promise that every current module attribute remains patchable forever.

| Surface | Policy |
| --- | --- |
| Documented command | Preserve permanently. |
| Current root script/function entry point | Preserve its inventoried module dispatch and callable behavior; do not infer a CLI contract. |
| Documented/public Python import | Preserve with a forwarding module through migration; removal requires a fresh audit and explicit approval. |
| Session metadata JSON | Preserve compact schema version 2 exactly. Version 1 is intentionally rejected and receives no migration layer. |
| Repository-private import or helper | Migrate the repository caller to the canonical owner; no external compatibility promise is inferred. |
| Serialized module-qualified class | Preserve the old loading path or provide a tested compatibility reader. |
| Multiprocessing worker | Keep it as an importable top-level callable in the canonical module and test process spawning. |
| Test-only monkeypatch seam | Patch the canonical owner after migration unless the seam is intentionally public. |

A forwarding module preserves only the inventoried public names and
signatures. It contains no duplicated implementation and need not reproduce
arbitrary private globals, module-level monkeypatch behavior, or private helper
locations. All three documented command modules remain permanent entry paths:
`session_metadata_cli.py` stays direct at the root, while `psth_webapp.py` and
`lfp_spike_phase_launcher.py` become thin entries over moved implementations.
The current `sync_ephys.py` root script remains subject to its separately
inventoried script/function contract. Any other compatibility removal occurs
only in the final cleanup phase after a fresh audit and explicit approval.

## Unused-code policy and current candidates

A repository-wide planning audit found no consumer, command, prior
documentation reference outside these planning documents, notebook reference,
dynamic import, or package entry point for:

- `behavior_pynap.py`;
- `spike_behavior_analysis.py`;
- `spike_behavior_binning.py`;
- `modified_sinc_smoother.py`; and
- the empty `plot_single_session_analysis.py`.

This is not deletion authorization. Immediately before deletion, the cleanup
work package must:

1. repeat the repository-wide audit at the implementation commit;
2. compare any apparently duplicated behavior with the retained tested path;
3. ask the user to confirm that no script outside this repository imports or
   invokes the candidate;
4. add and commit a tests-first removal contract that is RED while the
   candidate modules still exist; and
5. obtain explicit deletion approval.

Git history is the archive. Do not create a permanent `legacy` package for
code confirmed unused.

R0 also audits top-level functions and classes in modules that will be split.
An active module can contain an unused callable. Such a callable follows the
same fresh-search, external-use confirmation, tests-first removal, and explicit
approval gates as an unused module.

`analog_treadmill_decode.py` is a separate audit candidate rather than an
already proven unused module: `volts2speed` currently has a direct repository
test but no production caller, while its other three top-level analysis
functions have no repository caller. R0 must decide whether any externally used
surface remains before proposing callable or module deletion.

## Documentation policy

`src/neural_analysis/README.md` remains the single operational guide for lab
users. The organization refactor should not turn it into an architecture
manual.

Every function or method moved, created, or materially edited during the
refactor, including private helpers, documents input types, shapes and axes,
physical units, missingness, return values, and important preconditions. A
short docstring is sufficient when a helper has no arrays or physical units.
A package README is added only when a package has enough interacting modules
to need a short navigation guide; `lfp_summary` and `webapp` are likely
candidates. There is no requirement to add a README to every directory.

## Behavior-preserving simplification opportunities

Reorganization may expose code that can be made shorter or more direct without
changing behavior. These are candidates, not blanket authorization:

1. Delete modules and callables proven unused through the approved cleanup
   gate.
2. Retire version-1-shaped metadata adapters after R0 classifies their callers.
   Current candidates include the derived `ChannelGroupMetadata` and
   `PopulationMetadata` records, derived population/channel-group collections,
   duplicate identity/label properties, always-absent compatibility fields,
   and the test-only `MetadataPopulationInputs` path. Do not remove a semantic
   path alias when it still makes an active caller clearer.
3. Remove temporary import wrappers after repository and external-use audits.
4. Consolidate genuinely identical spike/behavior loading, channel-selection,
   and trial-mask functions behind one existing tested implementation. The two
   small region-name normalization functions are textually identical, but do
   not add cross-domain coupling or a generic naming module merely to remove
   those few lines. Consolidate them only if R2 reveals a natural existing
   owner used by both callers.
5. Replace repeated pass-through functions with a direct call when the wrapper
   adds no validation, naming value, caching boundary, or compatibility value.
6. Extract exact duplicate LFP-summary report-writing or size/accounting code
   only when its filesystem and failure contracts are the same. Similar-looking
   atomic writers with different safety requirements remain separate.
7. Remove the current LFP-summary runtime/PPC-runtime back-import by moving the
   already shared validation or preparation function to its natural owner.
8. Let the metadata webapp path pass version-2 probe and site records directly
   to its views instead of translating the first two probes through legacy PFC/
   HPC variable names. Keep the manual legacy path isolated until its separate
   retirement gate is approved.
9. Retire superseded legacy webapp path-entry branches only after the user
   confirms that session metadata fully replaces that existing capability.
10. Remove an older PPC executor, profile path, CT-specific adapter, or legacy
   CLI only when the R0/R7 audit proves it has no required reproduction,
   resume, profiling, or external caller.

Simplification occurs inside the work package that owns the responsibility and
uses the same RED-GREEN compatibility tests as movement. It must reduce code or
control flow; creating a generic helper used once is not simplification.

## Non-goals

The organization refactor does not add or broaden:

- compact metadata schema version 2, its fields, path-placement rules, or
  acquisition families;
- analyses, statistical tests, scientific defaults, trial conditions, or
  population definitions;
- plots, reports, webapp views, controls, or live operations;
- CLI commands, launcher modes, recovery paths, schedulers, or execution
  backends;
- cache, checkpoint, manifest, snapshot, receipt, or report schemas;
- automatic source discovery, schema migration, root remapping, or mixed-
  acquisition support;
- concurrency defenses or hostile-filesystem behavior beyond the current
  reviewed implementation; or
- generic artifact, workflow, execution, reporting, visualization, registry,
  plugin, or service frameworks.

The refactor covers `src/neural_analysis` and the stable commands that invoke
it. `src/external_tools`, `src/irig_tools`, behavior-preprocessing packages,
and installed dependencies are dependencies, not refactor targets.

The refactor also does not require every large module to become small. A large
cohesive numerical kernel may be easier to verify than several files joined by
private interfaces.

## Acceptance principles

An organizational change is accepted only when:

- a scientist can find the responsibility more easily from the package tree;
- the moved code has one clear primary reason to change;
- public contracts document types, shapes, axes, units, and returns;
- existing deterministic results and identities are exact;
- existing tolerated numerical comparisons remain within their reviewed
  tolerance;
- current commands, UI behavior, cache behavior, and saved artifacts remain
  compatible;
- no unsupported case or new abstraction was added; and
- the focused, affected, and required full test suites pass.

The refactor is complete when the user judges the package sufficiently easy to
navigate. Completion does not require realizing every provisional filename in
the target tree.
