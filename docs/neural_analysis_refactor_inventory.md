# Neural analysis refactor inventory

## Status and scope

This is the reviewed-deliverable draft for work package R0. It records the
baseline at commit `8d82051331f107d899696aac661faf1d89b87480` before any source
movement. R0 made no production-code changes, loaded no experimental data,
mutated no cache or report, performed no external copy, and submitted no Slurm
work.

The inventory classifies the current repository surface. It is not deletion
authorization and does not turn every test seam or unprefixed name into a
permanent public API.

## Frozen baseline

- Baseline commit: `8d82051331f107d899696aac661faf1d89b87480`
  (`update refactor plan`).
- Tracked worktree state before R0: clean.
- Pre-existing untracked state: 4,077 porcelain paths. The SHA-256 of the
  exact `git status --porcelain=v1 -uall` output was
  `497d2ca2d6443cb79e8d33dcbd549b3ea3341cc50aaa0336bf3dd5899dd9babd`.
  Those paths are user-owned and outside the refactor.
- Baseline command: `uv run pytest src/tests/neural_analysis`.
- Result: 1,667 passed, 2 failed, 22 warnings in 171.86 seconds.
- Existing failures:
  - `test_ct026_open_ephys_defaults.py::test_open_ephys_sync_workflow_defaults_to_both_ct026_probes`
  - `test_open_ephys_sync_utils.py::test_main_open_ephys_workflow_syncs_probe_a_and_probe_b_with_expected_paths`
- Both failures have the same cause. Current `sync_ephys.py` selects CT026
  session `20260803`, recording `2026-08-03_11-19-09`, and sorter directory
  `Kilosort4.1.3_2026-09-11_115727`; the tests expect CT026 session `20260801`,
  recording `2026-08-01_13-08-22`, and sorter directory `kilosort4`.
  Commit `44ade2d` changed the source defaults without changing these tests.
  The user clarified that both sets are session-specific examples and neither
  is an authoritative compatibility contract. The strict path assertions are
  obsolete and should be replaced by behavior-focused R1 tests.
- Recorded warnings include multiprocessing/fork deprecations, all-NaN
  plotting/NumPy warnings, a duplicate ZIP member warning, and Pynapple
  empty-duration/divide-by-zero warnings. They are not repaired implicitly.

R1 should make the suite green by correcting these obsolete strict-example
tests before moving implementation. The current source examples remain
unchanged unless a separate practical need is established.

## Compatibility classifications

### Documented commands and root entries

The accepted quickstart documents three command modules, not two:

1. `python -m src.neural_analysis.session_metadata_cli ...`
2. `uv run streamlit run src/neural_analysis/psth_webapp.py -- ...`
3. `python -m src.neural_analysis.lfp_spike_phase_launcher ...`

All three are permanent documented entry paths. The design phrase "the two
documented command modules" should say that the webapp and launcher are the
two documented commands being decomposed; `session_metadata_cli` is the third
documented command and remains direct at the package root.

`sync_ephys.py` is a current root script/function entry, not a documented CLI.
Its four callable workflows, defaults, return behavior, writes, and current
`__main__` dispatch are preserved without inventing an argument parser.
`manual_session_synchronization.py`, `spike_behavior_pynapple.py`,
`psth_behavior.py`, and `plot_cross_session_analysis.py` also have hardcoded,
undocumented direct-module execution. Their old dispatch must remain during
migration pending an external-use decision; no new CLI contract is inferred.

The root README still refers to the long-removed `preprocess_daq.py`. That
stale reference is not evidence of a supported current command.

### Python imports and serialized identities

No `__all__`, package entry point, or guide documenting scientific module
imports as public APIs was found. Repository callers and tests define the
known supported surface. Old paths for active modules should forward during
migration; later removal still requires the planned fresh audit and approval.

No audited scientific or workflow dataclass is persisted with a
module-qualified Python class identity. Some exploratory/scientific NPZ files
store an object-array `meta` dictionary, but not pickled class instances.
LFP-summary configuration and fingerprints use canonical JSON primitives.
Consequently, moving `AnalysisWindowConfig`, `FrequencyBandConfig`, and
`PowerAnalysisConfig` to `lfp/config.py` and aliasing those exact class objects
from `lfp_summary_models.py` is sufficient for repository-observed equality,
JSON, and fingerprint contracts. Possible external Python pickles remain an
external-use gate because aliases do not preserve the new classes'
`__module__` value.

### Process-bound identities

The spawn workers `_initialize_grouped_parallel_worker`,
`_compute_grouped_parallel_block`, and legacy `_compute_parallel_block` must
remain top-level importable callables in the canonical PPC execution module.
The process-bound records `_PhaseWorkDescriptor`, `_GroupedParallelBlockTask`,
`_GroupedParallelBlockResult`, `_ParallelBlockTask`, and
`_ParallelBlockResult` move with them. Existing tests assert worker module and
qualified names, the spawn context, initializer identity, and initializer
arguments. A forwarding old module may expose public entry functions, but the
workers themselves need the new canonical identity.

The CT026 profile adapter deliberately uses Linux `fork` with a nested child
closure. Its `_profile_child_worker` is a test seam, not a spawn-identity
contract.

## Module inventory

"Tests" names the primary direct or integration coverage rather than every
test that imports a transitive dependency. "Forward" means one implementation
at the canonical owner plus a narrow old-path compatibility module.

| Current module | Category and responsibility | Primary callers/tests and owned contracts | Approved destination and gate |
| --- | --- | --- | --- |
| `__init__.py` | Empty package initializer. | Imported as the package; owns no API or artifact. | Remain a minimal root initializer. New package initializers also stay minimal. |
| `analog_treadmill_decode.py` | Exploratory signal utility. `volts2speed` is test-only; three denoising/diagnostic callables have no repository caller. | `test_manual_session_synchronization.py` tests only `volts2speed`; no internal edges or persistence. | Keep at root through R1/R2. Callable/module external-use and explicit R7 deletion decision required. Do not create `synchronization/treadmill.py`. |
| `behavior_pynap.py` | Unused exploratory Pynapple prototype with hardcoded CT014 `main()`. | No repository caller, test, command, notebook, or non-planning documentation reference. | Keep unchanged until the R7 deletion gate. |
| `ephys_sync_utils.py` | Production synchronization: IRIG decoding, timezone/offset handling, sample/time mapping, Open Ephys bounds, acquisition workflows, sync NPZs, and alignment note. | Callers: `lfp_loading`, `manual_session_synchronization`, `sync_ephys`. Tests: IRIG, Open Ephys, sync, and LFP integrations. Owns caller-named sync NPZ schemas and `time_adjustment_note.txt`. Edge: `spikeglx_sync_io`. | Split only at existing boundaries under `synchronization/alignment.py` and source workflow modules; forward the active old path. |
| `lfp_loading.py` | Production SpikeGLX/Open Ephys LFP I/O, validation, synchronization lookup, window/sample mapping, filtering, and trial traces. | Broad LFP-summary, webapp, and Spike-LFP callers; `test_lfp_loading.py` and Open Ephys/LFP-summary integrations. Reads preprocessing JSON, binary data, and sync NPZ. Edge: `ephys_sync_utils`. | `lfp/loading.py`; forward old imports and preserve value, sample-rate, unit, and source-fingerprint semantics. |
| `lfp_phase_clustering.py` | Production phase/wavelet/PLV calculations, three NPZ persistence contracts, and one plot. | Runtime, webapp, Spike-LFP, and unit plotting callers; direct and integration tests. Edges: `lfp_spectrogram`, `spike_behavior_pynapple`. Saved dataclasses are flattened to arrays/scalars plus `meta`. | Computation/persistence to `lfp/phase.py` in R2A; `plot_phase_clustering` to `lfp/plotting.py` in R3; forward old imports. |
| `lfp_power_summary.py` | Production Welch PSD, reference normalization, band integration, and trial summaries. | Caller: `lfp_summary_runtime`; direct and synthetic pipeline tests. Edge: `lfp_summary_models` for three LFP config records. | `lfp/power.py`; import configs from `lfp.config`. This removes an ownership inversion. |
| `lfp_power_validation.py` | Production CT026 Power validation/report helper, not a CLI. | Direct validation tests; used as a base by other validations. Owns timestamped report files, deterministic PNG selection, and source identifiers containing module paths. Edges: summary runtime/I/O/models/pipeline/plotting. | `lfp_summary/power_validation.py`; preserve fixed config/report bytes and path provenance intent. A real report run is separately gated. |
| `lfp_spectrogram.py` | Production Morlet power, notch/padding/decimation, reference selection, and shared color limits. | Callers: phase clustering, summary runtime, webapp. Direct spectrogram/webapp tests. Edge: `spike_behavior_pynapple`. | `lfp/spectrogram.py`; calculation remains cohesive. |
| `lfp_spike_phase_launcher.py` | Permanent documented CLI plus preflight, durable state, recovery/rerender, locking, signals, execution, measurement, and persistence. | Quickstart, Slurm wrapper, summary handoff; launcher/shell/quickstart tests. Owns launcher JSON schemas and run files. `_MODULE` is the persisted old root command string. | Root parser/dispatch remains permanently. Only established preflight/state/execution groups may move under `lfp_summary` in R6A; no real run or cleanup without authorization. |
| `lfp_spike_phase_validation.py` | Production preview/final Spike-phase report construction and cache-backed publication. | Launcher and CT adapter; direct and launcher tests. Owns report schemas, fixed exact file sets, PNG validation, and deferred exact cleanup targets. Edges: Power validation and summary runtime/I/O/models/pipeline/plotting. | `lfp_summary/spike_phase_validation.py`; preserve historical report readability and atomic publication. |
| `lfp_summary_cache_relocation.py` | Production, explicit cache-relocation API and CLI. | Tests import the exact old module and invoke `python -m` with exact arguments. Copies only manifest/Power/Synchrony and writes a caller-selected receipt. Edges: summary models/I/O. | `lfp_summary/cache_relocation.py` with old import/command forward. Do not add a receipt schema field or generalize. External copy needs separate approval. |
| `lfp_summary_ct026_profile_adapter.py` | Dataset-specific production profiling binding and isolation. | CT026 adapter/binding tests; calls grouped planning/profiling/runner and private runtime/cache seams. Uses Linux fork and scalar IPC. | `lfp_summary/ct026_profile_adapter.py`; retain CT026 specificity and tested child seam. Real-data profile gated. |
| `lfp_summary_ct026_profile_locks.py` | Dataset-specific lock ownership/recovery. | Adapter caller and adapter tests. Owns `.ct026-profile.lock`; no neural imports. | `lfp_summary/ct026_profile_locks.py`; preserve lock identity and recovery. |
| `lfp_summary_ct026_profile_runner.py` | Dataset-specific interruption-safe scalar-state/profile runner. | Adapter caller and direct tests. Owns `ct026_ppc_profile_<timestamp>`, work stages, state schema `2`, `profile.json`, and `summary.md`. | `lfp_summary/ct026_profile_runner.py`; preserve stage/resume identities and scalar-only content. Real run gated. |
| `lfp_summary_io.py` | Production final-cache I/O, manifest status, validation, transaction, and relocation rebind. | Summary runtime/pipeline/webapp/validation/launcher/relocation; broad I/O and integration tests. Owns `manifest.json`, component NPZ loading/publication, status vocabulary, and rollback. Edge: models. | `lfp_summary/cache.py`; preserve filenames, historical manifests, atomic behavior, and literal generator `src.neural_analysis.lfp_summary_io`. |
| `lfp_summary_models.py` | Production workflow records, configuration validation/JSON, defaults, source semantics, and fingerprints. | Nearly all summary modules and tests. Edge: `lfp_loading`; reusable LFP modules currently back-import configs from here. | `lfp_summary/models.py`; three LFP config records move first to `lfp/config.py`, with same-object aliases at the old path. |
| `lfp_summary_payloads.py` | Production bridge from component results to exact final NPZ array schemas. | Caller: summary runtime; payload/I/O/runtime/integration tests. Owns `power.npz`, `synchrony.npz`, `spike_phase.npz` names and array axes/units/dtypes. Edge: pipeline record. | `lfp_summary/payloads.py`; artifact-critical, no array renaming or object dtype. |
| `lfp_summary_pipeline.py` | Production component orchestration and atomic-publication coordination. | Webapp, launcher, validations, runtime/payloads; direct/component tests. Owns component manifest entries, progress stages, and exact cleanup handoff. Edge: models. | `lfp_summary/pipeline.py`; preserve entry signatures, errors, progress, and cleanup semantics. |
| `lfp_summary_plotting.py` | Production workflow-specific pure plotting. | Three validations and summary webapp; direct/synthetic/webapp tests. Owns deterministic figure selection, names, captions, and provenance layout. | `lfp_summary/plotting.py`; do not make it a generic plotting framework. |
| `lfp_summary_ppc_kernel.py` | Production performance-sensitive numerical PPC kernel with no I/O or process ownership. | PPC runtime/profile/tests. No internal edges; runtime workers call it but kernel records do not cross process boundaries. | `spike_lfp/ppc_kernel.py`; forward old API and run the approved synthetic before/after benchmark. |
| `lfp_summary_ppc_profile.py` | Profiling support for grouped production PPC and the older legacy executor profile. | CT026 adapter and direct tests. Temporarily monkeypatches runtime internals. Edges: models/PPC runtime. | `lfp_summary/ppc_profile.py`; update patches to canonical owners. Keep both generations until an R7 deletion decision. |
| `lfp_summary_ppc_runtime.py` | Production grouped PPC planner/executor/checkpoint runtime plus distinct legacy per-job executor. | Summary runtime, profile, CT adapter, broad runtime/parallel/profile tests. Owns schedules, fingerprints, locks, checkpoints, staging, spawn workers, and version literals. Back-imports three helpers from summary runtime. | R4C `ppc_planning.py` and `ppc_execution.py`, after R4B moves shared validators/membership to a common owner. Keep serial/parallel together and retain legacy path pending R7. |
| `lfp_summary_preparation.py` | Production shared trial/site/phase/spike preparation and validation. | Runtime and CT adapter; preparation/runtime/integration tests. Edges: `lfp_loading`, models, `spike_behavior_pynapple`. | `lfp_summary/preparation.py`; natural candidate to own the three shared helpers that break the runtime/PPC cycle. |
| `lfp_summary_runtime.py` | Mixed production bridge for Power, Synchrony, Spike-phase, shared preparation, and dependency composition. | Validations, webapp, launcher, CT adapter, and extensive component tests. Imports scientific domains and most workflow foundations; imports PPC runtime. Owns prepared-cache/provenance identities. | R4B split into `power_runtime.py`, `synchrony_runtime.py`, `spike_phase_runtime.py`, and the smallest real shared owner. Move private cross-module seams atomically. |
| `lfp_summary_session.py` | Production metadata-to-workflow configuration assembly. | Webapp and launcher production factory; session/webapp tests. Edges: session metadata, LFP loading, models. | `lfp_summary/session.py`; R4A migrates internals from derived v1-shaped populations/channel groups/cache alias to direct v2 probes/unit channels/cache without changing requests. |
| `lfp_summary_webapp.py` | Mixed production immutable-snapshot inspection/cache plotting and Streamlit summary composition/actions. | Caller: `psth_webapp`; extensive direct/webapp tests. Owns the exact five-file snapshot and digest receipt contracts. Edges: summary models/I/O/pipeline/plotting/runtime and LFP loading. | Snapshot logic to `lfp_summary/snapshot.py` in R4D; UI composition to `webapp/summary_view.py` in R5. Preserve launcher handoff text. |
| `lfp_summary_work_cache.py` | Production execution-only prepared-phase/PPC caches, checkpoints, locks, recovery, and exact cleanup. | Summary runtime/PPC runtime/launcher/CT adapter; direct/runtime/profile tests. Owns prepared-phase and PPC directory/file schemas. | `lfp_summary/work_cache.py`; resume/artifact critical. No real cache mutation during refactor. |
| `lfp_synchrony_summary.py` | Production ITPC/ISPC/trial-PLV summaries, bootstrap bands, exemplars, and cache-array assembly. | Caller: summary runtime; direct/runtime/synthetic tests. Edge: `lfp_summary_models.FrequencyBandConfig`. | `lfp/synchrony.py`, importing `lfp.config`; removes the second ownership inversion. |
| `lfp_synchrony_validation.py` | Production CT026 Synchrony validation/report helper, not a CLI. | Direct/synthetic tests; imports Power validation base config. Owns the fixed report set and path-bearing source identifiers. Edges: summary runtime/I/O/models/pipeline/plotting. | `lfp_summary/synchrony_validation.py`; preserve report and projection semantics. Real report gated. |
| `manual_session_synchronization.py` | Production/manual mapping, event/barcode decoding, persistence, loaders, and hardcoded example script. | No production importer; direct test. Edges: `ephys_sync_utils`, `spikeglx_sync_io`. Writes caller-named sync NPZ and alignment note. | `synchronization/manual.py`; forward imports and preserve old direct-module dispatch pending external-use decision. |
| `modified_sinc_smoother.py` | Standalone smoothing prototype with a guarded demo. | No repository caller, test, docs, notebook, dynamic import, or entry point. | Keep unchanged until R7 deletion gate; no retained equivalent was established. |
| `plot_cross_session_analysis.py` | Active tested cross-session discovery, filtering, aggregation/publication, plotting, and hardcoded script. | No production importer; direct tests/internal `main`. Reads/writes established CSVs and writes five PNG families. No internal edges. | Data work to `population/cross_session.py` in R2D, plots to `population/plotting.py` in R3; preserve old dispatch pending external-use decision. |
| `plot_single_session_analysis.py` | Empty file. | No consumer or test. | Keep unchanged until explicit R7 deletion approval. |
| `population_pca.py` | Production PCA tensor construction, normalization/fitting, and optional timing profile. | Population decoding, webapp, unit plotting; PCA/decoding/switch/webapp tests. No internal edges or persistence. | `population/pca.py`; cohesive move with forward. |
| `population_pca_decoding.py` | Production choice-aligned selection, exploratory/rigorous decoding, summaries, score extraction, and plotting. | Webapp; direct tests. Edges: `population_pca`, `spike_behavior_pynapple`. | Computation to `population/decoding.py` in R2D; plots to `population/plotting.py` in R3. |
| `population_pca_switch_trajectories.py` | Production switch selection/classification, trajectories, count summaries, and plotting. | Webapp; direct tests. Edge: `spike_behavior_pynapple`. | Computation to `population/switch_trajectories.py` in R2D; plots to `population/plotting.py` in R3. |
| `psth_behavior.py` | Production/tested PETH calculation plus plotting/save and hardcoded example execution. | No production importer; direct tests/internal `main`. Edge: `spike_behavior_pynapple`; writes PNG. | Computation to `spike_behavior/psth.py` in R2B, plotting/save to `spike_behavior/plotting.py` in R3; preserve old dispatch. |
| `psth_webapp.py` | Permanent documented Streamlit entry; currently mixes session/manual inputs, cached loading, scientific wrappers, all domain views, and saves. | Operational quickstart plus extensive direct/integration tests. Imports metadata, summary, and all scientific domains. Owns cache token and exploratory output identities. | Thin root entry over `webapp` modules in R5. Preserve manual and metadata routes, controls, save behavior, and command. |
| `session_metadata.py` | Documented production compact-v2 metadata parser, resolver, probe routing, and cheap action availability. | CLI, webapp, summary session, launcher; metadata/CLI/quickstart/integration tests. Standard library only; JSON is the persistence contract. | Remain root/direct. Remove compatibility properties only after active callers migrate and the cleanup gate is approved. |
| `session_metadata_cli.py` | Permanent documented metadata create/validate CLI. | Quickstart and CLI/quickstart tests. Edge: `session_metadata`. | Remain root/direct and preserve exit/output/error behavior. |
| `spike_behavior_analysis.py` | Unused exploratory script with import-time hardcoded metadata/array loading. | No repository consumer. Import would perform real I/O. | Keep unchanged and do not import during characterization; R7 deletion candidate. |
| `spike_behavior_binning.py` | Unused exploratory script with import-time loading, decoding, plotting, and CSV writes. | No repository consumer. Some responsibilities resemble active code, but equivalence is not established. | Keep unchanged and do not import; R7 deletion candidate. |
| `spike_behavior_pynapple.py` | Active mixed production loading, trials, binning, decoding, table publication, one plot, and hardcoded legacy script. | Broad scientific/workflow/webapp callers and direct/integration tests. Owns two CSV publication contracts. No internal neural edge. | Split under `spike_behavior` by responsibility in R2B/R3; preserve old imports and direct dispatch. |
| `spike_lfp_hilbert_phase.py` | Production Hilbert-phase computation, spike sampling, and NPZ persistence. | Summary runtime, webapp, unit plotting; direct/plot/integration tests. Edge: `lfp_loading`. | `spike_lfp/hilbert.py`; plotting remains separate. |
| `spike_lfp_phase_locking.py` | Production Morlet spike-phase sampling, occupancy/rate/PPC metrics, sparsity, and NPZ persistence. | Webapp, Spike-LFP summary, unit plotting; direct/plot/summary tests. Edge: `lfp_phase_clustering`. | `spike_lfp/phase_locking.py`; forward old API. |
| `spike_lfp_summary.py` | Production PPC summaries, permutation inference, FDR, prevalence, deterministic schedules, and metadata fingerprints. | Summary/PPC runtimes and profile; extensive direct/integration tests. Edge: `spike_lfp_phase_locking`. No file write. | `spike_lfp/ppc.py`; active underscored runtime imports move with callers rather than being deleted. |
| `spikeglx_sync_io.py` | Production SpikeGLX digital-line I/O adapter. | `ephys_sync_utils`, manual synchronization; direct test. No neural edge or persistence. | `synchronization/spikeglx.py`; forward old imports. |
| `sync_ephys.py` | Root script/function entry with NI-only, treadmill placeholder, CT014, and CT026 Open Ephys workflows. No parser. | Sync/Open Ephys tests. Edge: `ephys_sync_utils`. Hardcoded session, recording, and sorter paths are local examples rather than contracts; output names and `__main__` CT026 dispatch are behavior. | Keep as thin root entry over R1 synchronization modules; preserve callable signatures, output behavior, and dispatch without freezing example paths. |
| `unit_spike_loading.py` | Production spike/webapp loading and path/input normalization. | Launcher, CT adapter, summary runtime, webapp; direct/channel/Open Ephys/profile tests. Edge: `spike_behavior_pynapple`. | `spike_behavior/loading.py`; `build_session_from_inputs` and `load_viewer_data` have no caller/test and require R7 external-use/deletion review. |
| `unit_spike_plotting.py` | Active mixed-domain numerical plot preparation, pagination/selection, plotting, and stable PNG publication. | Webapp; broad unit/LFP/Spike-LFP/population tests. Runtime edge: `spike_behavior_pynapple`; type-only scientific imports. | Split once in R3 among `spike_behavior/plotting.py`, `lfp/plotting.py`, `spike_lfp/plotting.py`, and `population/plotting.py`; no generic plotting layer. |

## Required mixed-module symbol classifications

### Metadata compatibility adapters

- `ChannelGroupMetadata`, `PopulationMetadata`, and the derived
  `channel_groups`/`populations` properties are v1-shaped compatibility
  adapters. Active production consumers remain in `lfp_summary_session.py` and
  `psth_webapp.py`; removal waits for their R4/R5 direct-v2 migrations.
- `ResolvedSession.session_id` is heavily used. `subject_id` is used by the
  metadata webapp bridge. `session_label` has no caller. `session_date` always
  returns `None` but is passed by two webapp paths and becomes
  `"unknown-date"`; it cannot be removed before R5 migration.
- `SiteMetadata.display_label` is active. `ProbeSources.display_label` has no
  caller; `ResolvedProbeSources.display_label` is test-only.
- `synchronization_file` and `aligned_spike_file` both expose the v2 alignment
  path, but remain useful role-specific aliases with active production
  consumers. `lfp_summary_cache_directory` similarly has an active workflow
  consumer.
- `lfp_summary_snapshot_directory` and `treadmill_file` are always absent and
  have no production caller. `ResolvedBehaviorSources.session_directory` is
  active and semantically useful.
- `MetadataPopulationInputs` and `metadata_population_inputs()` are test-only,
  with no production/documentation/persistence contract. Do not move them
  into the new webapp; present removal at the cleanup gate.

### `lfp_summary_runtime.py`

- Shared: prepared-run records, trial loading, phase/spike preparation,
  validators, condition membership, common cache helpers, production
  dependency composition.
- Power: preparation, PSD assembly, payload building, and Power dependencies.
- Synchrony: phase preparation, pair/trace assembly, payload building, and
  Synchrony dependencies.
- Spike: configured unit loading, spike preparation, grouped PPC payload,
  exemplar/packing/Hilbert/cleanup helpers, and Spike dependencies.
- Cross-module private seams `_validate_prepared_phase_run`,
  `_validate_prepared_spike_run`, `_analysis_condition_membership`, and
  `_work_fingerprint` must move with every caller in one slice.

### `lfp_summary_ppc_runtime.py`

- Planning: job/allocation/component records, checked arithmetic, batching,
  schedules, edge unions, preflight, and run-fingerprint planning.
- Current execution: grouped serial/spawn reduction, staging, checkpoint
  merging, locks, progress, and final execution result.
- Legacy execution: `execute_ppc_blocks`, its separate schedule/checkpoints,
  and its legacy worker/result records. Production Spike payloads do not use
  this path, but the older production-profile API and extensive tests do.
- The grouped and legacy paths are different generations; they must not be
  conflated or removed during R4/R6.

### `lfp_summary_webapp.py` and `psth_webapp.py`

- Snapshot inspection, validation, digest/cache identity, saved-axis/source
  selection, and cached plotting belong to `lfp_summary/snapshot.py`.
- Summary Streamlit configuration, actions, progress, rendering, and launcher
  handoff belong to `webapp/summary_view.py`.
- Root webapp session/manual input, cached behavior/spike/LFP loading, domain
  view rendering, scientific wrappers, and save actions split only along the
  R5 domain boundaries.

### `lfp_spike_phase_launcher.py`

- Root-owned: command parser, documented module dispatch, signal setup, and
  the permanent `_MODULE` command identity.
- Candidate internal groups for R6A: new/resume/report preflight and retained
  work validation; durable state and report recovery/rerender; locked
  execution/progress/cleanup; process-tree resource measurement.

## Saved artifact and provenance contracts

Later moves must keep these literal identities or provide an explicitly tested
compatibility interpretation:

- Manifest generator: `{"module":"src.neural_analysis.lfp_summary_io"}`.
- Prepared phase: generator `prepare_phase_run`, analysis version
  `prepared-phase-cache-v1`.
- Grouped PPC: generator `execute_grouped_ppc_component`, code version
  `s4-grouped-serial-v1`, kernel `segmented-kernel-v1`.
- Legacy PPC code version: `wp5c-4-v1`.
- Synchrony payload version:
  `synchrony-bootstrap-quantiles-counts-v1`.
- Power/Synchrony/Spike validation source identifiers include absolute
  runtime/validation module paths; Power/Synchrony also identify the sibling
  pipeline path. Relocation cannot silently change their historical meaning.
- A valid immutable summary snapshot has exactly five nonsymlink regular
  files: `manifest.json`, `power.npz`, `synchrony.npz`, `spike_phase.npz`, and
  `cache_snapshot_identity.json`. Receipt schema is integer `1`; manifest
  schema is string `"1"`; fixed per-file size/SHA-256 records determine the
  aggregate digest.
- `hpc_ppc.sh` remains unchanged: one task, eight CPUs, 32 GB, 72 hours,
  pre-timeout TERM, clean tracked checkout, one thread per numerical backend,
  and delegation to the permanent root launcher with frozen/offline `uv`.

## Internal dependency graph

The graph was generated with a standard-library AST audit over the 51 current
Python files. A dash means no import of another `src.neural_analysis` module.

```text
analog_treadmill_decode: -
behavior_pynap: -
ephys_sync_utils: spikeglx_sync_io
lfp_loading: ephys_sync_utils
lfp_phase_clustering: lfp_spectrogram, spike_behavior_pynapple
lfp_power_summary: lfp_summary_models
lfp_power_validation: lfp_summary_io, lfp_summary_models, lfp_summary_pipeline, lfp_summary_plotting, lfp_summary_runtime
lfp_spectrogram: spike_behavior_pynapple
lfp_spike_phase_launcher: lfp_spike_phase_validation, lfp_summary_io, lfp_summary_models, lfp_summary_pipeline, lfp_summary_runtime, lfp_summary_session, lfp_summary_work_cache, session_metadata, spike_behavior_pynapple, unit_spike_loading
lfp_spike_phase_validation: lfp_power_validation, lfp_summary_io, lfp_summary_models, lfp_summary_pipeline, lfp_summary_plotting, lfp_summary_runtime
lfp_summary_cache_relocation: lfp_summary_io, lfp_summary_models
lfp_summary_ct026_profile_adapter: lfp_spike_phase_validation, lfp_summary_ct026_profile_locks, lfp_summary_ct026_profile_runner, lfp_summary_models, lfp_summary_ppc_profile, lfp_summary_ppc_runtime, lfp_summary_preparation, lfp_summary_runtime, lfp_summary_work_cache, spike_behavior_pynapple, unit_spike_loading
lfp_summary_ct026_profile_locks: -
lfp_summary_ct026_profile_runner: -
lfp_summary_io: lfp_summary_models
lfp_summary_models: lfp_loading
lfp_summary_payloads: lfp_summary_pipeline
lfp_summary_pipeline: lfp_summary_models
lfp_summary_plotting: -
lfp_summary_ppc_kernel: -
lfp_summary_ppc_profile: lfp_summary_models, lfp_summary_ppc_runtime
lfp_summary_ppc_runtime: lfp_summary_models, lfp_summary_ppc_kernel, lfp_summary_runtime, lfp_summary_work_cache, spike_lfp_summary
lfp_summary_preparation: lfp_loading, lfp_summary_models, spike_behavior_pynapple
lfp_summary_runtime: lfp_loading, lfp_phase_clustering, lfp_power_summary, lfp_spectrogram, lfp_summary_io, lfp_summary_models, lfp_summary_payloads, lfp_summary_pipeline, lfp_summary_ppc_runtime, lfp_summary_preparation, lfp_summary_work_cache, lfp_synchrony_summary, spike_behavior_pynapple, spike_lfp_hilbert_phase, spike_lfp_summary, unit_spike_loading
lfp_summary_session: lfp_loading, lfp_summary_models, session_metadata
lfp_summary_webapp: lfp_loading, lfp_summary_io, lfp_summary_models, lfp_summary_pipeline, lfp_summary_plotting, lfp_summary_runtime
lfp_summary_work_cache: -
lfp_synchrony_summary: lfp_summary_models
lfp_synchrony_validation: lfp_power_validation, lfp_summary_io, lfp_summary_models, lfp_summary_pipeline, lfp_summary_plotting, lfp_summary_runtime
manual_session_synchronization: ephys_sync_utils, spikeglx_sync_io
modified_sinc_smoother: -
plot_cross_session_analysis: -
plot_single_session_analysis: -
population_pca: -
population_pca_decoding: population_pca, spike_behavior_pynapple
population_pca_switch_trajectories: spike_behavior_pynapple
psth_behavior: spike_behavior_pynapple
psth_webapp: lfp_loading, lfp_phase_clustering, lfp_spectrogram, lfp_summary_session, lfp_summary_webapp, population_pca, population_pca_decoding, population_pca_switch_trajectories, session_metadata, spike_behavior_pynapple, spike_lfp_hilbert_phase, spike_lfp_phase_locking, unit_spike_loading, unit_spike_plotting
session_metadata: -
session_metadata_cli: session_metadata
spike_behavior_analysis: -
spike_behavior_binning: -
spike_behavior_pynapple: -
spike_lfp_hilbert_phase: lfp_loading
spike_lfp_phase_locking: lfp_phase_clustering
spike_lfp_summary: spike_lfp_phase_locking
spikeglx_sync_io: -
sync_ephys: ephys_sync_utils
unit_spike_loading: spike_behavior_pynapple
unit_spike_plotting: lfp_phase_clustering, spike_behavior_pynapple, spike_lfp_hilbert_phase, spike_lfp_phase_locking
```

The only strongly connected component is
`lfp_summary_runtime <-> lfp_summary_ppc_runtime`. R4B owns its resolution by
moving the three already-shared validation/membership helpers to the natural
common preparation/runtime owner before R4C splits PPC planning/execution. No
new protocol or framework is needed.

The prohibited ownership inversion is
`lfp_power_summary`/`lfp_synchrony_summary -> lfp_summary_models -> lfp_loading`.
R2A resolves it by making `lfp.config` the canonical owner of the three
LFP-facing configuration records. Other cross-domain scientific edges shown
above reflect real current dependencies and are allowed by the approved
direction.

## Discrepancies requiring reconciliation

1. **CT026 synchronization examples (resolved):** neither the source's August
   3/versioned-Kilosort paths nor the tests' August 1/`kilosort4` paths are an
   authoritative contract. R1 tests should verify probe routing, path
   relationships, output naming, calls, and returns without freezing those
   replaceable examples. Preserve the current source examples absent a clear
   need to change them.
2. **Webapp startup wording (resolved):** with complete metadata, the default
   `Unit raster/PSTH` view calls `load_metadata_viewer_data_cached()` and reads
   `spike_clusters.npy` plus the aligned-spike NPZ on initial render. The plan
   and quickstart were corrected to describe the baseline. Preserve the
   current eager default view unless a clear need and separate behavior-change
   approval arise.
3. **Command count wording (resolved):** the design now distinguishes all
   three documented entry paths from the two commands whose implementations
   will be decomposed.
4. **Stale root README (resolved):** the removed `preprocess_daq.py` reference
   now points to the current manual synchronization module.

## Deletion and external-use gates

Current module candidates are `behavior_pynap.py`,
`spike_behavior_analysis.py`, `spike_behavior_binning.py`,
`modified_sinc_smoother.py`, and empty `plot_single_session_analysis.py`.
Callable candidates additionally include three non-`volts2speed` functions in
`analog_treadmill_decode.py` and `build_session_from_inputs`/`load_viewer_data`
in `unit_spike_loading.py`.

For each candidate, R7 must repeat exact module, filename, distinctive-symbol,
command, notebook, dynamic-import, and package-entry-point searches; compare
apparently duplicated behavior with retained tested code; obtain confirmation
of no external lab use; add and commit a meaningful tests-first removal
contract; and obtain explicit deletion approval. Nothing is deleted during
R0-R6.

Legacy `execute_ppc_blocks` and its older production-profile path are not
currently deletion candidates merely because the grouped path is the current
production payload implementation. They remain tested and are retained until
the same R7 evidence and approval gate.

## Proposed R1 freeze for review

The first R1 slice should be limited to:

- source modules: `spikeglx_sync_io.py`, `ephys_sync_utils.py`,
  `manual_session_synchronization.py`, and the thin root `sync_ephys.py`;
- direct production callers: `lfp_loading.py` and the root workflows;
- correction of the two obsolete strict CT026 example-path tests, preserving
  behavior assertions for two-probe routing, path relationships, output
  names, dependency arguments, and return values;
- tests written and committed first for absent canonical imports, numerical
  sample/time equivalence, timezone/bounds/warnings/missingness, exact NPZ and
  note output, preserved callable signatures/returns/dependencies,
  stubbed direct-module dispatch, and old-path forwarding;
- focused tests: `test_spikeglx_sync_io.py`,
  `test_manual_session_synchronization.py`, `test_irig_sync_utils.py`,
  `test_open_ephys_sync_utils.py`, `test_sync_ephys.py`,
  `test_ct026_open_ephys_defaults.py`, and affected LFP/Open Ephys integration
  tests; and
- acceptance: focused tests and the complete neural suite green, no extra
  source reads, no new source format or parser, and no treadmill move.

This proposed list is not authorization to write R1 tests or source. R0 is
complete only after this inventory and first-edit/test list are reviewed and
approved.
