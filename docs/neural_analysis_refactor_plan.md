# Neural Analysis Refactor Implementation Plan

## Status and authority

**Status:** implementation authorized by the user on 2026-09-23. NR0 is
complete. The user-approved NR1 metadata/path/resource dry run and bounded
small representative-window check in Section 5.2 items 1-2 completed and passed
fresh Sol review on 2026-09-23. Corrected Power, its Power-only report, and the
read-only legacy Power comparison in Section 5.2 item 3 completed and passed
fresh Sol scientific/evidence review on 2026-09-23. The user visually approved
the Power report and authorized corrected Synchrony in Section 5.2 item 4 on
2026-09-23. Corrected Synchrony, its revised report, and the legacy comparison
are now scientifically, operationally, and visually approved after fresh Sol
numerical, provenance, and systematic visual reviews plus explicit user
approval on 2026-09-23. Earlier user review flagged that every observed point
in the PFC theta-whole
ITPC summary lies below its percentile-bootstrap 95% interval. The bars are not
IQRs and the legacy figure has the same pattern. The user selected the
presentation-only resolution in Section 5.5 for both ITPC and ISPC because they
share the same phase-clustering/resampling calculation. The NR1P code/test
package implementing that presentation is complete, committed, and approved by
a fresh Sol/xhigh implementation review. The NR1C-A cache-relocation package is
also complete, committed, and approved after tests-first concurrency and
provenance hardening. The NR1C-B explicit launcher-target/preflight package is
complete, committed, and approved as well. The separately authorized NR1V run
created a scientifically approved versioned Synchrony cache, but fresh visual
review rejected its first immutable report because observed-marker footprints
overlapped the offset boxes in 12 of 18 ISPC summaries. The tests-first
presentation-only correction is complete through `389e4fd`; a separately
authorized immutable cache-only rerender completed and passed fresh Sol/xhigh
artifact review, and the user visually approved it on 2026-09-23. The approved
commits were pushed and the cluster checkout was safely fast-forwarded to
`c859afe`, but NR1E preflight exposed the NR1C-C shared-work-root defect below.
Spike-phase transfer and cluster actions remain separately gated.
Scientific recomputation, cache mutation, artifact replacement, and cluster
submission remain separately gated exactly as specified below.

This plan translates the requirements and settled decisions in
`docs/neural_analysis_refactor.md` into tests-first work packages. The design
document remains authoritative for scientific and architectural intent, with
the later user-approved refinements recorded in Section 25 incorporated into
both documents before implementation.
`docs/Tasks_neural.md` remains authoritative for the completed LFP-summary
methods and historical CT026 evidence. If this plan conflicts with either
document, stop and resolve the conflict before implementation.

The immediate prerequisite is a separately versioned correction to the Open
Ephys LFP value scaling. The structural refactor must use the corrected CT026
outputs as its regression baseline; it must not preserve the known unscaled
amplitude behavior as if that behavior were scientifically correct.

### Live implementation handoff

- **Branch and planning baseline:** `refactor` at `2245475` (`neural analysis
  refactor prep`), equal to `origin/refactor` when implementation began.
- **Active package:** tests-first NR1C-C launcher shared-work-root correction,
  awaiting user approval. NR0 tests commits
  `e728cea`
  and `2b497e4` plus implementation commit `177a8d6` are complete and
  approved. The NR1 metadata/path/resource dry run is complete and approved;
  its original dry-run report and profiling destinations remain absent. The
  bounded numerical small-window check is also complete and approved. The
  corrected cache now contains approved Power and scientifically approved
  Synchrony only. Corrected Synchrony passed its numerical, provenance, and
  reviewer visual gates. The user selected the Section 5.5 presentation-only
  revision for both ITPC and ISPC band summaries. NR1P tests and implementation
  are complete through implementation commit `ad8e598`. NR1C-A relocation
  tests are complete through `b994495`, and the reviewed implementation is
  `e1d99c3`. NR1C-B tests are complete through `4646b5d`, and its reviewed
  implementation is `c844adc`; no CT026 or cluster path was accessed during
  either package. NR1V published the versioned scientific cache successfully;
  its first report is preserved but rejected for marker/box overlap. The
  reviewed geometry regression is `88d7826` and its presentation-only source
  fix is `389e4fd`. The one-shot immutable cache-only rerender at clean HEAD
  `f8ecbf0` completed and passed fresh artifact review; the user visually
  approved it on 2026-09-23. The revised NR1V Synchrony cache is therefore the
  Section 5.6 transfer source. The user authorized and completed the Git push
  and first cluster checkout update to `c859afe`; preflight then stopped before
  evidence creation because the launcher rejects the valid retained shared
  prepared-phase work root. No transfer, dry run, or Slurm submission is
  authorized yet.
  The user requires any later preview to run unattended through Slurm without
  Codex monitoring; that execution is not approved.
- **Approved NR1 destinations:** session root
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference`;
  corrected cache `processed/lfp_summary_cache_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z`;
  analysis run `analysis_runs/ct026_nr1_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z`;
  report subdirectory `<run>/report`; temporary profiling subdirectory
  `<run>/profiling_tmp`. All four destinations were absent immediately before
  authorization was recorded.
- **Agent policy:** lead Sol/high orchestration; one active Terra/xhigh writer
  for NR1P or NR1C source packages, or one Terra/high evidence runner for NR1V
  or NR1E; one fresh Sol/xhigh gate reviewer; and at most one optional
  Terra/medium read-only scout. Only the active Terra writer may author package
  tests/source. Evidence runners and reviewers are read-only with respect to
  the repository; the lead alone edits documentation and stages/commits
  reviewed paths. Section 2.6 and the package-specific divisions in Sections
  5.5-5.6 are binding.
- **Worktree safety:** tracked files were clean at the start. Pre-existing
  untracked files are user-owned and excluded from refactor commits unless the
  user separately approves them. In particular, 14 pre-existing untracked
  neural test files are baseline evidence only and are not NR0 test-authoring
  targets.
- **Real-data authority:** the NR1 dry run may read CT026 path, metadata, trial,
  sorter, and source-identity information and may write reproducibility evidence
  only inside the approved analysis-run directory. It must not read numerical
  LFP windows, execute Power/Synchrony/Spike-phase kernels, create or mutate the
  corrected cache, create the report/profiling subdirectories, alter any legacy
  artifact, submit cluster work, or push Git commits.
- **NR1 dry-run result:** the approved run directory contains the exact runner
  plus nine evidence files. It records 427 trials, three sites, 50 frequencies,
  2,000 phase samples, a conservative eight-worker bound, and a
  1,152,900,000-byte phase/validity estimate. Corrected Power, Synchrony, and
  100-shuffle ProbeB identities were resolved with
  `open_ephys_affine_uV_v1`; the legacy cache is expected stale by manifest-only
  identity comparison. A rejected intermediate evidence pass accidentally
  materialized legacy component NPZ members read-only through
  `assess_component_status`; it did not change content or modification times.
  `incident.json`, the run log, preflight, and summary preserve that event. The
  corrected pass did not repeat the access, and fresh reviewer
  `/root/nr1_dryrun_reviewer` approved with no P0-P3 findings.
- **Approved NR1 small-window check:** evidence may be written only beneath
  `<session>/analysis_runs/ct026_nr1_small_window_open_ephys_affine_uV_v1_2026-09-23T12-04-46Z`.
  Read exactly three 1-second, 2,500-sample windows at the start, midpoint, and
  final valid second for PFC saved channel 5, HPC1 saved channel 222, and HPC2
  saved channel 14. Compare direct stored float32 values with the public
  loader's uV values and the authoritative per-channel affine formula; record
  shape, sample rate, extrema, mean/standard deviation, equality error,
  timing, peak allocation, source identity, and exact evidence hashes. Do not
  read sync or spike arrays, execute scientific kernels, create/mutate any
  cache, create report/profiling output, alter legacy artifacts, submit cluster
  work, or push.
- **NR1 small-window result:** all nine authorized reads used the exact start,
  centered, and final 2,500-sample windows. Direct float32 stored values and
  public float64 uV results matched the declared affine formula elementwise
  with zero absolute and relative error. Public reads took 1.677-4.304 ms with
  212,179-214,419 traced Python bytes. Evidence contains scalar summaries only;
  arrays were not persisted, and every protected path remained unchanged.
  Exact equality proves implementation agreement with the parsed sidecar gain
  and operation order; it does not independently validate the calibration
  declared by that sidecar. Fresh reviewer `/root/nr1_window_reviewer`
  approved with no P0-P3 findings.
- **Approved NR1 corrected Power:** write the Power component and manifest only
  to the reserved corrected cache
  `<session>/processed/lfp_summary_cache_open_ephys_affine_uV_v1_2026-09-23T09-49-58Z`.
  Write the exact runner, configuration/source identity, logs, comparison, and
  Power-only report beneath
  `<session>/analysis_runs/ct026_nr1_power_open_ephys_affine_uV_v1_2026-09-23T12-23-53Z`.
  The legacy cache is read-only. Compare axes, trial selection and validity,
  reference and source traces, linear PSDs, band powers, normalized dB arrays,
  report metadata, plots, runtime, and memory. Record and explain every changed
  or invariant field. Do not compute or publish Synchrony or Spike-phase, alter
  legacy artifacts, submit cluster work, or push.
- **NR1 corrected-Power result:** the corrected cache contains only validated
  `manifest.json` and `power.npz`, anchored by SHA-256 in the run evidence. All
  axes, selections, validity, exclusions, and finite masks are invariant;
  amplitude and linear-power arrays follow gain and gain-squared expectations.
  Twenty-eight of 30 arrays passed the original tolerance. Two full-resolution
  normalized-PSD arrays stopped the initial gate at approximately
  3.35e-7 dB maximum error, localized to near-zero 1,250-Hz support. A separate
  mechanistic log-ratio propagation bound plus a fixed 1e-6-dB ceiling accepted
  both with zero violations; below 100 Hz, maximum error is 6.31e-12 dB. The
  original failure record remains unchanged. Fresh reviewer
  `/root/nr1_power_reviewer` approved the scientific, safety, and evidence gate
  with no P0-P3 findings. The 12 report PNGs await explicit user visual
  approval at this result checkpoint.
- **Power visual approval:** the user approved all 12 corrected Power report
  figures on 2026-09-23. They are now accepted NR1 evidence.
- **Approved NR1 corrected Synchrony:** preserve the approved corrected Power
  bytes and add only `synchrony.npz` plus the atomic manifest update to the
  existing corrected cache. Write the exact runner, configuration/source
  identity, logs, comparison, and Synchrony-only report beneath
  `<session>/analysis_runs/ct026_nr1_synchrony_open_ephys_affine_uV_v1_2026-09-23T13-17-26Z`.
  The legacy cache is read-only. Compare phase-derived arrays, bootstrap
  schedules/results, all validity/support, source/filtered traces, exemplars,
  report metadata/plots, runtime, and memory. Explain every changed or invariant
  field. Do not compute or publish Spike-phase, alter approved Power or legacy
  artifacts, submit cluster work, or push.
- **Synchrony retry checkpoint:** the first authorized runner at
  `ct026_nr1_synchrony_open_ephys_affine_uV_v1_2026-09-23T13-17-26Z` returned
  silently after about 31 seconds before publication. It created only its
  script, configuration, and preflight evidence; the corrected cache and
  approved Power hashes remained byte-identical, and no Synchrony/report/work
  artifact appeared. Kernel logs contain no OOM evidence, so the cause is an
  unproven external/tool termination. Preserve that failed run. Retry the same
  unchanged production Synchrony settings only in
  `<session>/analysis_runs/ct026_nr1_synchrony_open_ephys_affine_uV_v1_retry_2026-09-23T13-23-01Z`,
  with durable precompute/progress/error/resource evidence and terminal-session
  polling. Do not reuse or overwrite the failed run.
- **NR1 corrected-Synchrony result:** the retry completed in 728.985 seconds
  with 4,286,124,032-byte peak RSS and published only `synchrony.npz` plus the
  atomic manifest update. Approved Power SHA-256 remains
  `164114d606cc204ff29ad173a686f0be66b54c2b944ed1f15008b2fa85367355`;
  Synchrony SHA-256 is
  `fc4d742ec42dfd00281e8759b15545c4e9488a34e85c38cd26a6e30469145983`.
  All schema, dtype, shape, finite-mask, identity, validity, support, count,
  selection-pool, and rank-order contracts match legacy. The original strict
  comparison remains failed on nine floating-point arrays. Separate reviewed
  evidence accepts them under fixed bounds: one-float32-ULP filtered-trace
  scaling, `1e-6` rad Hilbert phase, `1e-6` absolute dimensionless summaries,
  `1e-6` complex coherence-vector error, and `1e-3` rad raw circular sanity.
  Observed maxima are within those bounds. Thirteen of 108 PLV exemplars switch
  only between the same adjacent, exactly equidistant `.5`-rank percentile
  candidates; all other exemplars are unchanged. The 252-PNG report passed a
  systematic fresh visual review with no P0-P3 finding. A packaging-script
  provenance gap is explicitly disclosed: the pre-fix and intermediate
  revision bytes were not preserved, while the final authoritative packager,
  exact production runner, formulas, artifacts, and 276-file inventory are
  preserved and independently approved. That historical first report was
  later superseded by the reviewed and user-approved NR1V presentation report;
  no Spike-phase execution is authorized yet.
- **NR1V versioned Synchrony result:** the user authorized cache
  `<session>/processed/lfp_summary_cache_open_ephys_affine_uV_v1_synchrony_bootstrap_quantiles_counts_v1_2026-09-23T22-09-50Z`
  and run
  `<session>/analysis_runs/ct026_nr1v_synchrony_bootstrap_quantiles_counts_v1_2026-09-23T22-09-50Z`.
  One production invocation at pushed clean HEAD `3a50de3` completed in
  704.424 seconds with 4,288,274,432-byte peak RSS. Power remained byte-exact
  at SHA-256 `164114d606cc204ff29ad173a686f0be66b54c2b944ed1f15008b2fa85367355`;
  new Synchrony SHA-256 is
  `d9673f2183dcbb2eac7d8ec2be80916840815c727f1431b8fdf65b41d95cb86f`.
  All 38 prior arrays match exactly, and the eight added quantile/count arrays
  satisfy schema, ordering, finiteness, count, and instability contracts. The
  live CT026 Synchrony fingerprint is
  `12e5f78a347e7cc2a210172dd08bfc59c9152a79e58464cd5ba6fe5d80049e29`;
  The historical `18eeb15...` NR1P value is only the frozen one-site Open
  Ephys unit-test fixture, not a universal format or CT026 identity.
  Fresh review accepted the scientific cache and all 216 non-band figures but
  rejected the immutable first report because marker footprints overlap boxes
  in 12 of 18 ISPC summaries. Preserve that report. A tests-first geometry fix
  (`88d7826`, `389e4fd`) now uses a 0.30-row box offset and 0.20-row box height
  while preserving all values and meanings. A new exact cache-only rerender
  at
  `<session>/analysis_runs/ct026_nr1v_synchrony_presentation_rerender_2026-09-23T23-30-02Z`
  completed once from the accepted cache at clean HEAD `f8ecbf0`. Its sole
  report leaf has exactly 252 PNGs plus the five required metadata files. All
  216 non-band PNGs are byte-identical to both preserved reports, all 36 band
  summaries changed from the rejected geometry, and fresh read-only review
  found zero marker/box overlaps across all 324 condition rows with at least
  seven clear pixel rows. Cache and prior-report hashes remained unchanged.
  The user explicitly visually approved this revised report on 2026-09-23.
  It is now the sole Section 5.6 Synchrony transfer source.
- **User Synchrony finding and preview execution requirement:** the circles in
  the flagged PFC theta-whole figure are observed nonlinear ITPC estimates and
  the vertical lines are percentile-bootstrap 95% intervals, not medians and
  IQRs. With-replacement trial resampling biases the nonnegative vector
  magnitude upward; the current plotting test explicitly permits an interval
  not containing its estimate, and the legacy figure shows the same pattern.
  Treat this as a presentation problem rather than an affine-scaling
  regression. The user selected a horizontal observed-estimate plus bootstrap-
  distribution boxplot for both ITPC and ISPC, as frozen in Section 5.5. The
  100-shuffle ProbeB preview must use the standalone Slurm launcher with eight
  workers, no `--final-run`, and no Codex monitoring. Before proposing
  submission, use the now-tested required `--cache-directory` target and its
  fail-closed prerequisite validation. A pushed exact clean cluster checkout
  and a reviewed way to establish the approved Power/Synchrony state on the
  cluster are still required. No push, transfer, or Slurm submission is
  authorized by this note.
- **Durable evidence:** exact commands, results, inventory hashes, commits,
  findings, and the NR0-NR18 package ledger live in
  `docs/neural_analysis_refactor_execution_log.md`. Update this handoff and that
  log after every commit or interruption-relevant gate.
- **Verified baseline:** the corrected plotting suite passed 29 tests and the
  complete neural suite passed 1,531 tests with 22 known warnings. The complete
  repository run stops at
  collection because the pre-existing untracked
  `src/tests/behavior_analysis/test_project_utils.py` imports unavailable
  `autograd`; this is frozen unrelated user-owned baseline behavior and does
  not justify a refactor dependency. Both representative CT026 preprocessing
  sidecars are readable, and the public validator reports the immutable copied
  snapshot as valid.
- **Local commits:** the reviewed NR0 tests are `e728cea` and `2b497e4`; the
  reviewed implementation is `177a8d6`; NR0 closure is `a94559d`; NR1 dry-run
  authorization is `181b82b`. NR1P tests are `d5d9e2b`, `5ae2040`,
  `c0129fe`, and the wrapping-only harness correction `0b8e2d7`; the reviewed
  NR1P implementation is `ad8e598`. Documentation checkpoints through
  `3e15a31` were pushed to `origin/refactor` before NR1P implementation. NR1C-A
  tests are `e940671`, `72028ad`, `6e243c3`, `f10b7bd`, and `b994495`; its
  reviewed implementation is `e1d99c3`. NR1C-B tests are `0d056fa` and the
  fixture-isolation correction `4646b5d`; its reviewed implementation is
  `c844adc`. The NR1V rendered-geometry regression is `88d7826` and its
  presentation-only correction is `389e4fd`; rerender-review documentation is
  `f8ecbf0`, `1cb0263`, and `c859afe`. All are pushed to `origin/refactor` at
  exact `c859afe7d08235e4454fa15858ed8e02f6ce6feb`.

## 1. Objectives

The work has two ordered objectives:

1. Correct the Open Ephys derived-LFP read contract so stored values are
   converted using the authoritative `lfp_preprocessing.json` metadata, update
   cache identity, and establish reviewed CT026 regression artifacts.
2. Reorganize `src/neural_analysis` around explicit session metadata, source
   adapters, pure analyses, artifacts, workflows, execution, visualization,
   reports, and a metadata-driven webapp without changing the approved
   computations.

The refactor is complete only when a new supported session requires metadata
authoring, not edits to shared analysis, workflow, source-adapter, or webapp
modules. A user may either edit the clearly labeled values in the dedicated
metadata-creator Python entry point or generate a skeleton and edit the resulting
`neural_session.json`; the creator is an intentional user-facing authoring
surface rather than a scientific implementation module.

The refactor also produces one canonical, concise end-user guide at
`src/neural_analysis/README.md`. Its purpose is to let a returning user who has
forgotten the workflow quickly create or update session JSON, validate and run
the cache, launch the webapp, and find/inspect completed outputs without reading
the architecture documentation or historical task logs.

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

- Follow strict RED-GREEN-REFACTOR. Commit tests before implementation.
- Preserve existing public call signatures unless a work package explicitly
  approves a compatibility-preserving extension.
- Run all Python commands with `uv run`.
- Add no dependency for the scaling correction or initial structural work.
- Keep raw array shapes, axis order, sampling rates, channel indices, and time
  coordinates unchanged unless a separately approved requirement says
  otherwise.
- Never overwrite the approved historical CT026 caches or report directories.
- Treat the old CT026 amplitude-bearing artifacts as immutable historical
  evidence of the legacy unscaled behavior, not as the corrected baseline.
- Do not combine import moves with numerical changes. The scaling correction
  is completed and approved before responsibility moves begin.
- Every new package directory receives a README when introduced, as required by
  the design document.
- `src/neural_analysis/README.md` is the end-user landing page. Subpackage
  READMEs document developer ownership and internals and must not become
  competing user guides.

## 2.4 Design guidance for every work package

The following rules apply the principles in `docs/SoftwareDesign.md` to this
refactor. They are review criteria, not optional style preferences.

### Purpose and contracts before abstractions

- Every new module begins with one stated responsibility, its supported inputs
  and outputs, its core transformation, and its limitations.
- Every public function documents types, array/dataframe shapes, axis order,
  physical units, missing-value semantics, return values, and failure behavior.
- Use type hints for public functions, dataclass fields, and ambiguous local
  values. Do not annotate obvious local variables merely to add noise.
- Prefer plain functions and frozen dataclasses. A class is justified only when
  it provides a clearer data or interface boundary.
- Keep data records separate from complex behavior. Do not create a large
  mutable `Session` object that loads data, runs analyses, writes caches, and
  renders figures.

### Composition and dependency injection

- Assemble concrete loaders, workflows, and writers at CLI/application
  boundaries; pass narrow callable or Protocol dependencies into consumers.
- Prefer explicit arrays, columns, paths, and small configuration records over
  passing a complete session object or dataframe when only a subset is used.
- Use a small `Protocol` only where multiple real implementations share a
  stable behavioral contract. Do not create Protocols for every function or in
  anticipation of hypothetical backends.
- Prefer composition over inheritance. No new mixins or multi-level inheritance
  hierarchies are permitted without a separately approved rationale.
- Avoid nested convenience functions and lambdas. Use module-level helpers or
  `functools.partial` only when they make a real dependency boundary clearer.

### Cohesion, coupling, and simplicity

- A module has one primary reason to change. Source adapters do not select
  analyses; analyses do not load paths; plotting does not compute or save;
  reports do not reopen raw sources; webapp views do not implement numerics.
- Dependency arrows follow the target layering in the design document. A lower
  layer must not import workflows, reports, webapp modules, or CLIs.
- Avoid control flags in numerical functions. Separate operations into named
  functions; retain user-facing mode flags only at CLI/workflow boundaries.
- Do not introduce `utils.py`, `helpers.py`, wildcard imports, dynamic plugin
  discovery, a service/database, or a generic base-analysis class.
- Apply DRY only after repeated behavior is demonstrably the same. Temporary
  duplication is preferable to a premature abstraction that hides scientific
  meaning.
- Apply YAGNI: support the inspected CT026 Open Ephys and CT014 SpikeGLX layouts
  and fail clearly on unsupported variants.

### Scientific readability and reproducibility

- Prefer established NumPy, SciPy, Pandas, Pynapple, and current project
  functions over new mathematical implementations.
- Any first use of a library API in this project must be checked against the
  installed source or official documentation and captured by a focused test.
- Randomness always enters through an explicit seed or generator and is saved
  in result provenance.
- Shape, axis, unit, coordinate, or dtype transformations are explicit in code
  and documentation. Silent transposes, squeezing, unit conversions, and
  dataframe-column inference are prohibited.
- Optimize only measured bottlenecks. Keep performance-critical PPC kernels
  isolated and do not make surrounding workflow code clever for speed.

### Import and package policy

- Use relative imports within one package and absolute imports across package
  boundaries.
- Keep `neural_analysis/__init__.py` minimal; do not create broad convenience
  re-exports.
- Keep nesting shallow. The target domain subpackages in the design document
  are the maximum intended depth unless a reviewed package proves a clearer
  boundary is needed.
- Old import paths forward temporarily and emit no new behavior. Compatibility
  wrappers never become a second implementation.

## 2.5 Dependency waves and merge order

The packages are sequential unless this plan explicitly permits read-only
overlap. The dependency waves are:

| Wave | Packages | Required result before next wave |
| --- | --- | --- |
| A - corrected baseline | NR0-NR1 | Corrected, approved CT026 regression artifacts; legacy artifacts preserved. |
| B - foundations | NR2-NR5 | Session metadata, LFP/spike/behavior adapters, and synchronization contracts are green. |
| C - generic integration | NR6-NR7 | Artifact mechanics and metadata-driven compute entry point operate over compatibility workflows. |
| D - presentation | NR8-NR10 | Pure visualization and one metadata-driven webapp shell are green. |
| E - scientific domains | NR11-NR14 | Numerical code is grouped by domain with exact compatibility imports and regression evidence. |
| F - orchestration split | NR15-NR17 | Workflows, reports, execution, profiling, and compatibility bindings have their final owners. |
| G - retirement | NR18 | Approved legacy removal, documentation consolidation, and final acceptance. |

No package may consume a target interface from a later wave. A wave-boundary
review checks dependency direction with repository import scans before the next
wave starts.

## 2.6 Codex implementation team and authority

This subsection governs future implementation only after explicit user
authorization. Approval of this plan does not itself authorize source edits,
CT026 execution, cluster submission, cache mutation, or artifact replacement.

The role names refer to Codex agents, not scientific multiprocessing workers:

1. **Lead Sol orchestrator - `gpt-5.6-sol`, `high`.** The root agent owns user
   communication, requirements, worktree safety, package ordering, worker
   prompts, independent RED/GREEN reproduction, staging, commits, execution-log
   updates, and completion decisions. It may edit planning/handoff documents
   and perform mechanical staging/commits, but it does not author or repair
   package source or test changes. Findings are returned to the Terra worker.
2. **Sol gate reviewer - `gpt-5.6-sol`, `high` or `xhigh` as assigned below.**
   A fresh read-only agent audits a stable tests-only or implementation diff for
   scientific drift, missing cases, coupling, interface breaks, cache identity,
   transactions, concurrency, and performance. It never edits, commits, spawns
   agents, or broadens scope.
3. **Terra package worker - `gpt-5.6-terra`, `high` or `xhigh` as assigned
   below.** One write-enabled worker owns the active package's exact allowlist.
   The same worker writes tests, stops at RED, and resumes for implementation
   only after the lead's gate. It never commits, changes tests merely to reach
   GREEN, edits outside scope, runs unauthorized real data, or spawns agents.
4. **Optional Terra scout - `gpt-5.6-terra`, `medium`, read-only.** The lead may
   use one scout for a bounded usage map, installed-library inspection,
   performance-log analysis, or external-entry-point inventory. A scout returns
   file/line evidence and uncertainty, with no edits or generated artifacts.

Do not silently substitute models or efforts. If an assigned model/effort is
unavailable, stop and ask the user to revise the plan. Do not use `max` or
`ultra` without a documented plan change and approval.

### Concurrency and shared-worktree rules

- At most four Codex agents may be active: the lead, one Terra writer, one Sol
  reviewer, and one optional Terra scout.
- Only one agent may edit the shared worktree. All implementation packages are
  sequential. Read-only scouting may overlap only with independent work;
  review begins only after the diff is stable and the writer is idle.
- Before and after every assignment, the lead records HEAD, branch, staged and
  unstaged package diffs, and `git status --short`. Any unexpected tracked
  change stops all work for user direction.
- Every writer prompt contains an exact allowlist. A needed out-of-scope or
  public-interface edit is a replanning request, not permission to expand.
- The lead alone stages and commits reviewed paths. Unrelated existing changes
  remain untouched and unstaged.
- Read-only roles should use enforced read-only permissions when available. If
  the runtime provides only inherited write permissions, the restriction is
  procedural and the lead records and audits that limitation.
- Subagents do not spawn subagents. The lead waits for every requested report
  and ends completed assignments before opening the next package.

### Agent spawn and handoff contract

At implementation start, the lead verifies from host/runtime metadata that the
root is Sol/high and explicit Terra/Sol child overrides are accepted. Model
self-report is not evidence. With model overrides, use a bounded context fork
and put all essential constraints in the prompt; do not rely on a full-history
fork that would force inherited settings.

Every assignment prompt states:

- package ID, role, model, effort, read/write authority, and stopping gate;
- goal, approved decisions, exact files allowed, and forbidden actions;
- interfaces, scientific definitions, axes, units, and compatibility contracts
  that must remain unchanged;
- tests and commands to run, expected RED, performance constraints, and real-
  data authorization state; and
- required return: inspected/changed files, diff summary, commands/results,
  RED/GREEN evidence, unresolved risks, and confirmation of no commit or
  out-of-scope edit.

Use follow-up tasks with the same Terra worker across tests-only and
implementation phases. Do not replace it merely to skip a clean handoff.

### Package-specific agent assignments

| Package | Terra role and effort | Independent Sol gate | Test-design review | Principal risk |
| --- | --- | --- | --- | --- |
| NR0 | writer, `xhigh` | `xhigh` | mandatory | Units, cache identity, legacy compatibility. |
| NR1 completed gates | command/evidence runner, `high` | `xhigh` | not applicable | Historical real-data scientific interpretation and artifact safety. |
| NR1P | writer, `xhigh` | `xhigh` | mandatory | Synchrony payload identity, exact quantiles, shared ITPC/ISPC presentation. |
| NR1V | command/evidence runner, `high` | `xhigh` | not applicable | New versioned Synchrony artifact, Power preservation, visual approval. |
| NR1C-A | writer, `xhigh` | `xhigh` | mandatory | Source equivalence, manifest rebinding, atomic cache relocation. |
| NR1C-B | writer, `xhigh` | `xhigh` | mandatory | Explicit launcher cache target, legacy/work protection, resume identity. |
| NR1C-C | writer, `xhigh` | `xhigh` | mandatory | Safe classification of retained prepared-phase work versus active PPC work. |
| NR1E | cluster evidence runner, `high` | `xhigh` | not applicable | Authorized transfer/dry-run/submission boundaries and no monitoring. |
| NR2 | writer, `high` | `high` | mandatory | Metadata missingness, path containment, schema migration. |
| NR3 | writer, `xhigh` | `xhigh` | mandatory | Open Ephys/SpikeGLX units, channel semantics, bounded I/O. |
| NR4 | writer, `xhigh` | `xhigh` | mandatory | Time coordinates, IRIG/manual alignment, CLI compatibility. |
| NR5 | writer, `high` | `high` | mandatory | Sorter/aligned-spike identity and dataframe contracts. |
| NR6 | writer, `xhigh` | `xhigh` | mandatory | Atomic publication, stale/failed state, snapshot receipts. |
| NR7 | writer, `high` | `high` | mandatory | Metadata-to-legacy workflow binding and dry-run purity. |
| NR8 | writer, `high` | `high` | optional | Plot/computation separation and figure equivalence. |
| NR9 | writer, `high` | `high` | optional | Unit/exploratory figure splits and computation-free plotting. |
| NR10 | writer, `high` | `high` | mandatory | Streamlit state, snapshot identity, no rerender computation. |
| NR11 | writer, `xhigh` | `xhigh` | mandatory | LFP numerical equality and phase conventions. |
| NR12 | writer, `xhigh` | `xhigh` | mandatory | PPC seeds, schedules, checkpoints, memory, performance. |
| NR13 | writer, `high` | `high` | optional | Trial/unit axes and Pynapple behavior. |
| NR14 | writer, `high` | `high` | mandatory | PCA/decoding leakage, CV identities, cross-session inputs. |
| NR15 | writer, `xhigh` | `xhigh` | mandatory | Component orchestration and corrected CT026 equivalence. |
| NR16 | writer, `high` | `xhigh` | mandatory | Cache-only reports, CT014/CT026 compatibility, profiling isolation. |
| NR17 | writer, `xhigh` | `xhigh` | mandatory | Resume, locks, failure transactions, local/Slurm parity. |
| NR18 | writer, `high` | `xhigh` | mandatory | External callers, deprecation, deletion, final topology. |

Every package receives a final independent Sol review. A mandatory test-design
review occurs after the lead reproduces RED and before the test-only commit.
For packages marked optional, the lead may still require it if the test diff
changes a public contract or reveals comparable risk.

### Mandatory per-package sequence

1. Lead re-reads package source, tests, callers, documentation, and current
   worktree; unresolved requirements return to the user.
2. Optional read-only Terra scout gathers only the assigned evidence.
3. Terra writer makes tests-only changes, runs the focused command, records
   genuine expected RED, and stops.
4. Lead audits the tests, independently reproduces RED, obtains the specified
   Sol test-design review, and commits only the tests.
5. Lead follows up with the same Terra writer to authorize implementation.
6. Terra writer makes the smallest in-scope change, runs focused GREEN, and
   stops without committing.
7. Lead audits the complete diff and independently runs focused and affected
   suites. Sol reviewer audits the stable GREEN diff.
8. Accepted findings return to the same Terra worker; the lead repeats tests
   and review as needed.
9. Lead runs the package and wave gates, commits implementation separately,
   then records evidence in a documentation-only execution-log commit.
10. No next package starts until all agents are idle and the package record is
    complete.

### Interruption and disagreement policy

- An interrupted worker or reviewer report is not a completed gate.
- The lead records last verified HEAD, status, exact diff, commands/results,
  and whether tests or implementation are uncommitted.
- A replacement uses the same role/model/effort/allowlist after the lead
  independently re-audits the diff and reproduces the last claimed result.
- Conflicting agent reports, irreproducible failures, unclear library APIs,
  unexpected scientific differences, or near-threshold inference changes stop
  the package for user direction. Agent agreement is not correctness evidence.

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

## 2.8 Target integration contracts and data flow

Exact field spellings are frozen by the NR2 tests-only review, but the following
responsibilities and data flow are binding.

### Session boundary

- A frozen `SessionDescription`-style record represents decoded user metadata:
  session identity, acquisition family, probes, sites, pairs, populations, and
  cache references. It contains relative paths and optional values but no raw
  arrays or analysis defaults.
- A separate frozen `ResolvedSession`-style record contains the metadata-file
  path and resolved absolute paths for one environment. Resolution changes
  location only; it cannot change scientific identity.
- In the initial schema, the metadata path must be the canonical
  `<session_home>/neural_session.json`; its parent is the session root. External
  metadata locations and root mappings are not part of this contract.
- `load_session_metadata(path)` decodes and structurally validates JSON.
- `resolve_session_paths(description, metadata_path)` resolves contained paths.
- `validate_session_for_action(resolved, action)` returns a structured
  availability/diagnostic result or raises an exact configuration error before
  numerical loading.

These names may be refined during NR2 review, but one object must not combine
all three states or hide path resolution in attribute access.

### Source boundary

- LFP metadata records sample rate in Hz, saved-channel count/order, physical
  unit, storage dtype/layout, sample count, source reference path, and adapter
  value-semantics version.
- A bounded LFP read accepts explicit source, zero-based saved-channel index,
  and half-open sample bounds and returns a one-dimensional physical-value
  array plus immutable metadata/provenance. It performs no trial alignment.
- Synchronization functions accept explicit coordinate arrays and return
  explicit mapped coordinates; acquisition event readers stay separate.
- Spike and behavior adapters return validated native tables/arrays with stable
  row/unit identities. They do not choose filters, populations, or analyses.

### Analysis boundary

- Analysis functions accept arrays/tables and small scientific configuration
  records only. They do not receive session paths, open files, emit progress,
  write artifacts, or render figures.
- Result records identify axes, dtypes, physical units, and missingness. Large
  arrays remain NumPy arrays rather than nested object graphs.
- Random analyses require an explicit seed/generator. The resolved seed and
  derivation identities are part of provenance and scientific identity.

### Artifact boundary

- Artifact readers validate manifest/schema/identity before returning arrays.
- Component writers stage arrays and metadata, validate staged bytes, publish
  the component atomically, and replace the manifest last.
- The generic tabular writer atomically publishes workflow-supplied CSV bytes
  while preserving the supplied filename, columns, rows, and missingness; it
  has no scientific table schema.
- Work artifacts/checkpoints have separate identities from final scientific
  components and cannot masquerade as completed results.
- Snapshot validation returns saved configuration/provenance; callers never
  combine saved arrays with new live labels.

### Workflow boundary

- Each analysis descriptor states action requirements, scientific preset,
  runnable components, expected artifacts, and cached views.
- A component workflow accepts resolved validated configuration, narrow source
  callables, artifact interfaces, and a progress callback. It returns a
  structured result/status and optional exact cleanup request.
- Small exploratory workflows compose validated session metadata, source
  adapters, and pure analysis calls for bounded interactive unit, trial,
  LFP-LFP, Spike-LFP, and population views. They return explicit result records
  and perform no Streamlit rendering, artifact publication, or hidden long-run
  execution. This keeps exploratory composition out of both the webapp and the
  numerical kernels without forcing it into the production cache workflow.
- The registry is an explicit mapping of a small known set, not runtime plugin
  discovery.

### Presentation and execution boundaries

- Visualization functions accept validated result arrays plus plot context and
  return figures/axes. Reports decide what to render and where to publish.
- Cached webapp views read validated artifacts and call visualization.
  Exploratory webapp views may call only the narrow exploratory workflow
  interfaces above and then visualization; they do not import source adapters
  or numerical kernels directly. Streamlit cache keys include the complete
  resolved source identity and adapter value-semantics version supplied by the
  workflow.
- Execution freezes resolved configuration before starting work. Local and
  Slurm paths invoke the same workflow contract. Resume uses saved run state,
  not newly supplied session metadata.

The intended flow is:

```text
neural_session.json
  -> session decode / resolve / action validation
  -> explicit analysis registry + versioned preset
  -> source adapters
  -> pure analysis kernels
  -> component workflow
  -> atomic artifacts
  -> reports and/or cache-only webapp views
```

The bounded live-view branch shares the same decode/resolve/source/analysis
stages, then returns through an exploratory workflow to visualization and the
webapp without publishing a production component. It is not an alternate
scientific implementation.

Execution wraps the workflow; it does not sit between sources and analyses.

## 2.9 End-user quickstart documentation contract

`src/neural_analysis/README.md` is a short operational guide, not an API
reference or design narrative. It is written for a lab user returning after
months away. The first screen should answer: which file do I edit, which command
do I run, and where do I look afterward?

The guide must contain these sections in this order:

1. **Workflow at a glance.** A four-step summary: create/edit metadata, validate
   or dry-run, compute/cache, inspect in the webapp.
2. **Prerequisites.** Repository location, `uv` environment, session-directory
   expectation, the initial-version requirement that metadata live at the
   session root, and the rule that commands run from the repository root.
3. **Create the session JSON.** Name the canonical
   `<session_home>/neural_session.json` location; identify the clearly labeled
   variables in `cli/create_session_metadata.py`; show how to edit those values
   before running the creator; also show how to emit a default/incomplete
   skeleton and edit the resulting JSON directly. Show the exact commands; list
   the small set of user-supplied identities/paths; explain `null` versus an
   omitted probe; and show how to run schema-only and requested filesystem
   validation. Link to schema details rather than reproducing every field.
4. **Run the cache.** Show exact dry-run and local commands for Power,
   Synchrony, Spike phase, and all approved components. Explain that dry run
   performs no computation or cache mutation. Keep Slurm to one short example
   or link to a dedicated execution note; do not bury the common local workflow
   under cluster details.
5. **Know when a run finished.** Identify the run state, manifest, component
   files, log, summary, report directory, and failure/resume information a user
   should check. Explain that file existence alone does not establish a valid
   compatible result.
6. **Launch the webapp.** Show the exact command using
   `--session-metadata PATH`, the expected startup state, and how live metadata
   inspection differs from opening an approved copied snapshot.
7. **Inspect cached output.** Explain how to select a component, site/pair,
   population, condition, epoch, and cached view; where provenance and
   compatibility status appear; and how unavailable results are represented.
8. **Common recovery and troubleshooting.** Cover missing paths, invalid site or
   population references, stale/incompatible caches, incomplete runs, resume,
   missing optional inputs, and the warning not to edit manifests or NPZ files
   manually.
9. **Safety and provenance.** State that old approved snapshots are immutable,
   computation does not silently rewrite session JSON, copied-local and
   producing-cluster paths have different roles, and commands should be run
   against an explicit metadata file.

Documentation rules:

- Use generic placeholders such as `/path/to/session/neural_session.json`; do
  not present CT014/CT026 paths, fixed PFC/HPC/V1 sites, or ProbeA/ProbeB as
  universal defaults.
- State plainly that external metadata locations, absolute source paths, and
  root remapping are not supported by the initial schema. The metadata file is
  `<session_home>/neural_session.json`, and all source paths are relative to and
  contained by that directory.
- Include one compact example JSON only if it stays synchronized through a
  tested fixture or generated output. Never maintain an untested hand-written
  schema example that can silently drift.
- Commands are copied from the actual CLI `--help` contract and use `uv run`.
- Every command intended to be pasted by a user has a CLI parsing or smoke test.
- Keep advanced schema, cache internals, report schemas, profiling, and package
  topology in linked technical documentation.
- Update the guide in the same package that changes a user-visible command,
  path, output name, recovery step, or webapp control. Documentation drift is a
  failing package review condition.
- NR18 performs a novice-path review: starting only from this README and an
  example session layout, the reviewer must be able to identify the JSON
  creation, dry-run, cache, output-inspection, and webapp commands without
  consulting implementation modules.

## 2.10 Documentation and baseline gate before NR0

Before a Terra writer receives NR0, the lead Sol completes a documentation-only
baseline:

1. Re-read this plan, `neural_analysis_refactor.md`, `SoftwareDesign.md`,
   `Tasks_neural.md`, and relevant AGENTS instructions.
2. Verify the branch/HEAD and record complete staged/unstaged/untracked status
   without cleaning or changing unrelated files.
3. Audit every untracked test used by a package or baseline. With explicit user
   approval, either commit intended pre-existing tests in a standalone baseline
   commit or record them as excluded user-owned work. A package may not modify
   an untracked baseline test and then present the whole file as that package's
   tests-only RED commit.
4. Restage the exact reviewed documentation content, verify the cached diff is
   nonempty and matches the worktree, and commit the approved design and
   comprehensive plan as documentation only. Never rely on an intent-to-add or
   empty staged placeholder.
5. Create `docs/neural_analysis_refactor_execution_log.md` with the starting
   HEAD, worktree inventory, model/effort policy, authorization state, focused
   baseline commands, and empty NR0-NR18 package records; commit it separately
   if it was not part of the approved documentation baseline.
6. Run and record the focused NR0 baseline, the complete neural suite, and the
   complete repository suite. Existing failures must be resolved or explicitly
   accepted as a frozen unrelated baseline before tests-only work begins.
7. Confirm the representative CT026 files and legacy artifacts are readable but
   perform no scientific recomputation or write.
8. Verify the named Sol/Terra model and effort overrides from runtime metadata.

This is lead-orchestrator work, not a Terra source package. It creates no test
or source diff and does not count as NR0 RED evidence.

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
configuration/component provenance. SpikeGLX identity must remain unchanged by
NR0; the separately tested NR3 source-identity completion is outside this
correction.
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

### 3.4 Close the other source-identity gaps during adapter extraction

NR0 remains limited to the Open Ephys correction. The codebase audit found two
separate omissions that must be closed by the later adapter packages rather
than silently folded into NR0:

- NR3 includes each SpikeGLX LFP binary's authoritative same-stem `*.lf.meta`
  file in the source identity, including a SHA-256 content digest because the
  metadata file is small and numerically authoritative. A binary path or
  directory stat is not a substitute for the metadata that defines
  saved-channel order, sample rate, channel type, and voltage conversion.
- NR5 and NR7 carry the exact consumed spike-population sources into resolved
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
exercise other explicit units. After NR2-NR3, resolved production
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

**NR1E - external cluster evidence.** NR1E begins only after NR1V, NR1C-A, and
NR1C-B are approved and committed, NR1C-C is complete, the user approves the
next Git push, and the exact clean pushed cluster checkout is verified. One Terra/high cluster evidence
runner has no repository-edit, stage, commit, or push authority. Each external
mutation remains separately user-gated:

1. With read-only authority, the runner verifies commit, tracked cleanliness,
   environment, focused/full-neural tests, source paths, destination absence,
   legacy preservation, and resource bounds, then stops with evidence.
2. After explicit transfer/publication approval, the same runner may create
   only the exact declared transfer run/staging/final paths, execute relocation
   once, validate hashes/component states, and run one metadata-only launcher
   dry run. It stops without submitting Slurm.
3. A fresh Sol/xhigh read-only reviewer audits the stable transfer, rebinding
   receipt, dry-run identity, legacy preservation, and exact proposed `sbatch`
   command. Findings return to the runner only within already authorized paths;
   any new mutation returns to the user.
4. After separate explicit submission approval, the same Terra/high runner may
   issue exactly one reviewed `sbatch` command, record its returned job id and
   durable command, and stop immediately. It must not poll `squeue`, `sacct`,
   logs, launcher state, or results. It may not submit a resume or another job.

The lead remains the sole user-facing authority throughout NR1E and records
every authorization boundary in the execution log. A later user-requested
status/result inspection is a new bounded read-only assignment, not a
continuation monitor and not authority to resume, repair, or submit.

#### Dependencies and performance

Introduce no package dependency. Reuse the standard library, NumPy, existing
canonical configuration/fingerprint functions, safe NPZ loader, atomic JSON
writer, and public component-status validator. Hash files in fixed-size chunks;
never use `read_bytes()` for large artifacts or sources. Hash each unique
source/component once per host and reuse the recorded digest. The transfer is
approximately 232 MB for the two component NPZs; source verification is I/O
bound and may read both LFP binaries once but must not materialize them or any
NPZ array collection simultaneously. Record wall time and peak RSS.

#### Execution and no-monitoring handoff

After the tests and implementation are green, a separate user approval is
still required for each external mutation: Git push, cluster checkout update,
component transfer/manifest publication, and Slurm submission. On the exact
clean pushed cluster checkout:

1. run focused and complete neural tests;
2. run the relocation command into the exact approved versioned cache and
   inspect its receipt plus public component states;
3. run one metadata-only launcher dry run against that cache and inspect the
   ProbeB population, 100 shuffles, eight workers, paths, resource bounds, and
   absence of work/output mutation; and
4. present the exact `sbatch src/shell_scripts/hpc_ppc.sh new ...` command for
   separate approval.

If submission is approved, capture only the returned Slurm job id and durable
submission command, then stop. Codex must not poll `squeue`, `sacct`, logs, or
launcher state. Slurm writes the persistent job log and the launcher writes its
timestamped state, progress, checkpoints, exact resume command, component, and
report. The user may later request a one-time status or result inspection. A
timeout/preemption is never converted into a new run; the saved explicit
`resume` command is submitted only after separate user direction. The preview
cannot automatically escalate to 1,000 shuffles or `--final-run`.

## 6. Structural package contract

After NR1 approval, use small vertical migrations. Each package introduces one
target responsibility, migrates bounded callers, leaves a thin compatibility
boundary, and proves corrected-baseline equivalence. A repository-wide rename
or simultaneous reorganization is prohibited.

Every package specification below freezes:

- dependencies and exact responsibility;
- current files that may be edited and target files that may be created;
- tests written first and a focused RED/GREEN command;
- interfaces and behavior that must remain unchanged;
- performance and completion evidence; and
- the appropriate Terra writer and Sol review from Section 2.6.

File lists are allowlists. README and `__init__.py` files directly required for
listed new packages are implicitly included, but they contain documentation and
minimal exports only. Any other file requires a plan amendment before editing.

After focused GREEN, every code package runs its directly affected complete
test files and:

```text
uv run pytest -q -p no:cacheprovider src/tests/neural_analysis
```

The complete repository suite runs at every wave boundary and at NR18. Test
counts, warnings, timing, and skips are recorded in
`docs/neural_analysis_refactor_execution_log.md`, which the lead creates in the
documentation-only implementation baseline before NR0 tests.

## 7. NR2 - Session metadata foundation

**Depends on:** approved NR1 corrected baseline.

**Files:**

- new `src/neural_analysis/session/models.py`
- new `src/neural_analysis/session/json_io.py`
- new `src/neural_analysis/session/paths.py`
- new `src/neural_analysis/session/validation.py`
- new `src/neural_analysis/session/README.md`
- new `src/neural_analysis/README.md` with the initial workflow overview,
  prerequisites, and session-JSON creation/validation sections
- new `src/neural_analysis/cli/create_session_metadata.py`
- new `src/neural_analysis/cli/README.md`
- new `src/tests/neural_analysis/session/test_models.py`
- new `src/tests/neural_analysis/session/test_json_io.py`
- new `src/tests/neural_analysis/session/test_paths.py`
- new `src/tests/neural_analysis/session/test_validation.py`
- new `src/tests/neural_analysis/cli/test_create_session_metadata.py`

**Design and tasks:**

- Define frozen data-only records for session identity, acquisition family,
  probes, sites, site pairs, populations, and approved cache references.
- Keep scientific presets out of session JSON. Store only identity, sources,
  topology, selection rules, and approved result references.
- Decode one explicit schema version; reject future versions and provide only
  reviewed conservative migration for older versions.
- Require the initial metadata document to be named
  `<session_home>/neural_session.json`. Resolve relative paths from that session
  root and require all initial source paths to remain inside it. External
  metadata locations, absolute source paths, traversal, and root remapping are
  explicitly deferred.
- Separate structural validation from action validation. Structural validation
  permits useful incomplete descriptions; action validation reports exactly
  what a requested operation lacks.
- Keep records declarative. Loading arrays, computing analyses, inspecting
  caches, and rendering UI are outside this package.
- Provide the requested editable Python creator with clearly labeled variables
  and a `main` function. Users may edit those values so the first generated JSON
  already contains the intended session values. The same entry point also
  supports emitting an intentionally incomplete/default skeleton for users who
  prefer to edit JSON directly. It may write explicit `null` values but must not
  invent paths, sites, probes, regions, populations, or results.
- Create the canonical end-user README and document the exact JSON creation and
  validation commands implemented in this package. Mark cache and webapp
  sections as forthcoming until their tested commands exist; do not guess them.

**Tests written first:**

- deterministic JSON round trip, stable ordering, and strict schema version;
- arbitrary probe/site identifiers, one or multiple probes, and explicit
  site-pair references;
- probe identity remains independent of site/region annotation;
- `null`, omitted, empty string, invalid path, missing probe, and empty
  scientific population are distinct;
- duplicate IDs, unresolved references, population/probe mismatches, mixed
  acquisition family, and malformed cache identities fail closed;
- the canonical metadata filename and session-root location are enforced;
  relative paths resolve from that root and reject absolute paths, traversal,
  symlink escape, and external-metadata/root-remapping attempts;
- filesystem existence checks are opt-in and fail requested missing non-null
  paths;
- incomplete metadata supports inspection while action validation disables
  only unsupported work;
- both edited-creator output and default-skeleton output validate and cause no
  computation/cache side effect;
- every JSON-creation/validation command shown in the top-level README parses
  and runs against a temporary synthetic session;
- representative CT026 and CT014 fixtures express their different layouts
  without hardcoded PFC/HPC semantics.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/session \
  src/tests/neural_analysis/cli/test_create_session_metadata.py
```

**Completion evidence:** public data contracts have full types/units/path
semantics; metadata validation opens no large binary; no existing analysis
module imports the new session package yet; README inventory is complete.

## 8. NR3 - LFP source contracts and acquisition adapters

**Depends on:** NR2; preserves NR0 corrected semantics exactly.

**Files:**

- new `src/neural_analysis/sources/contracts.py`
- new `src/neural_analysis/sources/open_ephys/lfp.py`
- new `src/neural_analysis/sources/spikeglx/lfp.py`
- new READMEs in `sources`, `sources/open_ephys`, and `sources/spikeglx`
- existing `src/neural_analysis/lfp_loading.py` as a compatibility facade
- existing `src/neural_analysis/lfp_summary_models.py` only for the source-
  fingerprint compatibility bridge
- new `src/tests/neural_analysis/sources/test_contracts.py`
- new `src/tests/neural_analysis/sources/open_ephys/test_lfp.py`
- new `src/tests/neural_analysis/sources/spikeglx/test_lfp.py`
- existing `test_lfp_loading.py`,
  `test_open_ephys_behavior_integration.py`, and
  `test_lfp_summary_preparation.py`
- existing `test_lfp_summary_models.py` for source-identity regression

**Design and tasks:**

- Define the smallest explicit LFP metadata/window contracts needed by both
  acquisition families. Prefer frozen dataclasses plus functions; do not build
  a behavior-heavy loader hierarchy.
- Move corrected Open Ephys metadata validation and bounded channel-window
  reading without altering values, dtype conversions, shapes, or semantics
  version.
- Move SpikeGLX metadata, gain correction, and bounded reading. Validate saved
  channel type/order so the synchronization row cannot be selected as LFP.
- Have each adapter expose the exact authoritative metadata sidecars it
  consumed. Include the same-stem SpikeGLX `*.lf.meta` in source fingerprints;
  do not treat the `*.lf.bin` or its parent directory as sufficient identity.
- Keep synchronization and trial alignment out of the LFP adapters; they accept
  sample windows and return explicit sample rate, unit, and provenance.
- Retain old `lfp_loading` imports and functions as thin forwarding wrappers.

**Tests written first:**

- both adapters satisfy the same documented window contract;
- corrected Open Ephys results match NR1 values exactly;
- SpikeGLX gain-corrected microvolt behavior matches the existing baseline;
- changing only a SpikeGLX `*.lf.meta` file stales affected live components and
  work caches while the unchanged legacy snapshot remains inspectable; a
  same-length edit with restored size/mtime is detected by the metadata content
  digest;
- channel bounds, signal-versus-sync type, dtype, layout, byte size, scale,
  unit, and sample rate validation fail with source-specific context;
- unsupported metadata layouts fail rather than guessing;
- wrappers preserve public signatures and return values;
- static import tests forbid adapters from importing workflows, webapp,
  reports, or Streamlit.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/sources/test_contracts.py \
  src/tests/neural_analysis/sources/open_ephys/test_lfp.py \
  src/tests/neural_analysis/sources/spikeglx/test_lfp.py \
  src/tests/neural_analysis/test_lfp_loading.py \
  src/tests/neural_analysis/test_open_ephys_behavior_integration.py \
  src/tests/neural_analysis/test_lfp_summary_preparation.py \
  src/tests/neural_analysis/test_lfp_summary_models.py
```

**Performance:** preserve memmap/bounded reads; measure allocation and time for
representative windows; prohibit full-file copies or repeated gain correction.

## 9. NR4 - Synchronization and acquisition digital I/O

**Depends on:** NR3.

**Files:**

- new `src/neural_analysis/sources/open_ephys/synchronization.py`
- new `src/neural_analysis/sources/spikeglx/synchronization.py`
- new `src/neural_analysis/synchronization/alignment.py`
- new `src/neural_analysis/synchronization/irig.py`
- new `src/neural_analysis/synchronization/manual.py`
- new `src/neural_analysis/synchronization/README.md`
- existing `ephys_sync_utils.py`, `spikeglx_sync_io.py`,
  `manual_session_synchronization.py`, `sync_ephys.py`, and
  `analog_treadmill_decode.py` as source/compatibility inputs
- existing `src/irig_tools/irig_sync_utils.py` and
  `src/irig_tools/open_ephys_irig.py` as compatibility inputs for logic whose
  neural-facing owner moves in this package
- existing low-level `src/irig_tools/irig_core.py`,
  `decode_irig_logic_csv.py`, and `irig_serial_io.py` remain the non-neural
  decoder/serial owners and are regression inputs, not duplicate targets
- their existing tests, including the decoder/serial/source-layout tests, plus
  new tests under
  `src/tests/neural_analysis/synchronization`

**Design and tasks:**

- Separate acquisition-specific digital/event reading from shared time mapping,
  IRIG anchor validation, interpolation, and manual alignment.
- Keep absolute Unix seconds, acquisition samples, derived LFP samples, and
  event-relative seconds distinct in names and contracts.
- Preserve Open Ephys global/AP-to-derived-LFP mapping and SpikeGLX digital-line
  semantics.
- Split treadmill conversion needed by manual synchronization from any
  standalone exploratory signal analysis.
- Move or wrap shared affine/anchor/interpolation logic exactly once. Keep the
  low-level IRIG bit decoder and serial protocol in `src/irig_tools`; leave
  forwarding imports there when current external callers require them rather
  than copying an implementation into both packages.
- Reduce `sync_ephys.py` to a compatibility CLI over explicit adapters. Do not
  introduce session discovery or filename inference.

**Tests written first:**

- existing synthetic IRIG, serial, digital-line, and manual-alignment results
  are exact;
- monotonicity, finite anchors, minimum anchor count, UTC offset, sample-rate
  mismatch, segment assumptions, and extrapolation behavior fail clearly;
- coordinate conversions document input/output units and preserve shapes;
- Open Ephys and SpikeGLX source readers meet one narrow shared alignment
  contract without importing each other;
- old imports and CLI calls forward during compatibility;
- no synchronization module imports Streamlit, analysis, artifact, report, or
  workflow code.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_irig_core.py \
  src/tests/neural_analysis/test_decode_irig_logic_csv.py \
  src/tests/neural_analysis/test_irig_logic_neurokairos_source.py \
  src/tests/neural_analysis/test_irig_module_reorg.py \
  src/tests/neural_analysis/test_irig_serial_io.py \
  src/tests/neural_analysis/test_irig_sync_utils.py \
  src/tests/neural_analysis/test_open_ephys_irig.py \
  src/tests/neural_analysis/test_open_ephys_sync_utils.py \
  src/tests/neural_analysis/test_spikeglx_sync_io.py \
  src/tests/neural_analysis/test_sync_ephys.py \
  src/tests/neural_analysis/test_manual_session_synchronization.py \
  src/tests/neural_analysis/synchronization
```

**Completion evidence:** exact coordinate equality where deterministic; stated
tolerance only for fitted/interpolated floats; no raw LFP numerical behavior
changes; legacy command smoke passes.

## 10. NR5 - Spike, channel-quality, and behavior source adapters

**Depends on:** NR2 and NR4.

**Files:**

- new `src/neural_analysis/sources/spikes/kilosort.py`
- new `src/neural_analysis/sources/spikes/aligned.py`
- new `src/neural_analysis/sources/spikes/channel_quality.py`
- new `src/neural_analysis/sources/behavior/trials.py`
- new `src/neural_analysis/sources/behavior/events.py`
- new `src/neural_analysis/sources/behavior/treadmill.py`
- new READMEs in `sources/spikes` and `sources/behavior`
- extraction/wrappers in `unit_spike_loading.py`,
  `spike_behavior_pynapple.py`, and `analog_treadmill_decode.py`
- source-fingerprint compatibility changes in `lfp_summary_models.py`
- new tests under `src/tests/neural_analysis/sources/spikes` and
  `sources/behavior`
- affected existing loading, channel-quality, behavior-integration, and manual
  synchronization tests

**Design and tasks:**

- Load and validate sorter metadata, spike sequences, aligned UTC arrays,
  channel quality, trial tables, event tables, and treadmill data without
  choosing an analysis.
- Compare sorter/aligned-spike sequence identity and shape; do not require
  literal provenance-path equality for the inspected CT014 layout.
- Return the exact files consumed by population resolution. Source identity
  includes sorter spike samples, cluster assignments, curated cluster metadata,
  aligned spikes, and channel-quality metadata when it affects selection; a
  sorter-directory stat alone is not accepted as the scientific source.
- Represent those validated paths in a small immutable spike-source provenance
  record returned by the adapter and consumed by NR7's resolved configuration.
  Do not add mandatory constructor parameters to the current
  `UnitPopulationConfig`; old configurations without the record remain legacy-
  readable and new configurations fail closed if required provenance is absent.
- Make population filtering a separate explicit operation over validated
  tables. Report retained channel/unit counts and fail closed on required
  missing/ambiguous columns.
- Validate only action-required trial columns rather than one global table
  schema.
- Use Pynapple native structures where they are already the supported contract;
  do not wrap them in redundant session classes.

**Tests written first:**

- Kilosort/aligned arrays preserve stable unit identity, one-to-one lengths,
  finite UTC seconds, and source sample coordinates;
- selected curated sorter paths may match aligned data whose stored provenance
  names the parent sorter;
- required quality columns/categories are explicit and counts are reported;
- missing channel-quality files remain valid for actions that do not require
  them;
- trial/event loaders preserve row identity and require only named action
  columns;
- CT026 and CT014 fixtures retain distinct schemas without conditional subject
  code;
- old loading functions forward with unchanged public results;
- changing any consumed sorter/aligned/channel-quality file stales only the
  affected population/component; unrelated components retain their identity;
  legacy directory-only identities remain readable but are not current.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/sources/spikes \
  src/tests/neural_analysis/sources/behavior \
  src/tests/neural_analysis/test_unit_spike_loading.py \
  src/tests/neural_analysis/test_channel_quality_integration.py \
  src/tests/neural_analysis/test_spike_behavior_pynapple.py \
  src/tests/neural_analysis/test_open_ephys_behavior_integration.py \
  src/tests/neural_analysis/test_manual_session_synchronization.py \
  src/tests/neural_analysis/test_lfp_summary_models.py
```

**Performance:** metadata/table validation may load small tables and index
arrays, but not LFP binaries or complete spike trains unnecessarily. Preserve
row order and avoid dataframe copies where validation can use views.

## 11. NR6 - Artifact, work-cache, and snapshot foundation

**Depends on:** NR2-NR5; corrected semantics identity from NR0.

**Files:**

- new `src/neural_analysis/artifacts/manifests.py`
- new `src/neural_analysis/artifacts/component_cache.py`
- new `src/neural_analysis/artifacts/work_cache.py`
- new `src/neural_analysis/artifacts/snapshots.py`
- new `src/neural_analysis/artifacts/tabular.py`
- new `src/neural_analysis/artifacts/legacy_result_files.py`
- new `src/neural_analysis/artifacts/README.md`
- compatibility wrappers in `lfp_summary_io.py` and
  `lfp_summary_work_cache.py`
- smallest snapshot-validation extraction from `lfp_summary_webapp.py`
- new artifact tests plus affected existing I/O, work-cache, webapp, pipeline,
  and launcher tests

**Design and tasks:**

- Move generic JSON/NPZ validation, atomic component publication, manifest-last
  updates, restart-only work caches, snapshot/receipt validation, and component
  status into explicit artifact owners.
- Provide a narrow, explicitly legacy named-array NPZ writer for the existing
  exploratory result-save wrappers. It owns filesystem/serialization mechanics;
  domain modules or workflows continue to define field names, units, axes, and
  metadata.
- Provide a generic atomic tabular writer for workflow-supplied CSV tables. It
  preserves supplied column order, row order, filenames, encoding, missingness,
  and overwrite behavior but knows no cross-session schema or scientific
  meaning. It stages and validates the complete replacement before atomic
  publication so an interrupted write retains the previous bytes. The writer
  accepts a validated direct-child filename and rejects absolute paths,
  separators, traversal, symlinks, and output-directory escapes.
- Keep component-specific array schemas supplied by workflows; artifacts do not
  know dataset paths or scientific formulas.
- Distinguish missing, stale, running, failed, corrupt, and compatible states.
- Preserve `allow_pickle=False` for component caches and snapshots, named axes,
  physical-unit metadata, rollback, immutable reports, cleanup ownership, and
  copied-snapshot identity.
- Retain legacy readers and wrappers; do not rewrite old snapshots.

**Tests written first:**

- previous valid bytes survive injected NPZ, JSON, validation, replace, and
  manifest failures;
- previous CSV bytes survive injected staging, validation, and replace
  failures; successful writes preserve supplied dataframe order, missingness,
  filenames, and existing overwrite behavior; unsafe or escaping table names
  fail before staging;
- manifest publishes last and never points to an invalid component;
- path traversal, symlink escapes, pickle/object arrays in component or
  snapshot readers, axis mismatches, bad checksums, incompatible semantics,
  and malformed receipts fail closed;
- legacy snapshots remain inspectable but stale for corrected live computation;
- work cache/checkpoint identity rejects scientific, representation, source,
  semantics, and execution changes according to existing contracts;
- artifact modules import no Streamlit, dataset compatibility, source loaders,
  or numerical analyses;
- old public I/O functions forward exactly;
- existing exploratory result files retain their names, named-array schemas,
  object-array `meta`, overwrite/error behavior, and current `allow_pickle=True`
  read contract when their wrappers later delegate to
  `legacy_result_files.py`. That legacy format is never accepted as a component
  cache or snapshot; replacing it with a pickle-free schema requires a separate
  versioned migration rather than an import refactor.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_io.py \
  src/tests/neural_analysis/test_lfp_summary_work_cache.py \
  src/tests/neural_analysis/test_lfp_summary_pipeline.py \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py \
  src/tests/neural_analysis/artifacts
```

**Performance:** safe loading validates without duplicating complete arrays;
snapshot hashing remains streaming/bounded; no new serialization dependency.

## 12. NR7 - Metadata-driven analysis registry and cache runner

**Depends on:** NR2-NR6.

**Files:**

- new `src/neural_analysis/workflows/contracts.py`
- new `src/neural_analysis/workflows/registry.py`
- new `src/neural_analysis/workflows/README.md`
- new `src/neural_analysis/cli/run_cache.py`
- update `src/neural_analysis/README.md` with dry-run, local cache, completion,
  output-location, and first-line troubleshooting guidance
- new tests for workflow contracts, registry, and CLI
- minimal adapters in `lfp_summary_pipeline.py`, `lfp_power_validation.py`,
  `lfp_synchrony_validation.py`, and `lfp_spike_phase_validation.py`

**Design and tasks:**

- Define a small explicit registry describing supported analysis name,
  action-required metadata, runnable components, result artifacts, and cached
  views. Do not build dynamic plugin discovery or a base-class hierarchy.
- Resolve session metadata into the existing approved LFP-summary configuration
  through a pure, reviewable adapter. Versioned code presets own scientific
  defaults; acquisition adapters supply authoritative sample rates, physical
  units, and value-semantics versions; every run saves the resolved combination.
- Provide one cache command with explicit metadata path, component selection,
  dry-run/local mode, output, and execution profile inputs.
- Dry run validates paths and relationships, reports planned inputs/outputs and
  resource bounds, and performs no numerical computation or cache mutation.
- Keep Slurm submission, long-run state, and resume in NR17. The NR7 command may
  invoke the current local compatibility workflow only.
- Document only the cache modes actually implemented at this stage. The guide
  identifies manifest, component, log/summary, and report outputs at a high
  level and links to technical artifact documentation for schemas.

**Tests written first:**

- registry entries are explicit, unique, and dependency-valid;
- arbitrary probe/site labels map to existing analysis contracts without CT026
  branches;
- Power, Synchrony, Spike phase, and Compute All require only their documented
  metadata and produce deterministic resolved configurations;
- scientific presets and execution settings remain separated in fingerprints;
- dry run opens no raw arrays and writes nothing;
- incomplete metadata disables only affected components with exact messages;
- corrected CT026 metadata resolves to NR1 identities and numerical defaults;
- CLI parsing is thin and delegates immediately.
- README dry-run/cache commands are exercised against synthetic metadata, and
  documented output names agree with the produced artifact plan/result.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/workflows/test_contracts.py \
  src/tests/neural_analysis/workflows/test_registry.py \
  src/tests/neural_analysis/cli/test_run_cache.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_pipeline.py
```

**Completion evidence:** metadata-driven CT026 dry run matches the corrected
configuration; synthetic arbitrary probe/site dry run succeeds; no cluster or
real computation occurs in package verification.

## 13. NR8 - LFP-summary visualization extraction

**Depends on:** NR6-NR7.

**Files:**

- new `src/neural_analysis/visualization/common.py`
- new `src/neural_analysis/visualization/lfp_lfp/power.py`
- new `src/neural_analysis/visualization/lfp_lfp/phase.py`
- new `src/neural_analysis/visualization/lfp_lfp/synchrony.py`
- new `src/neural_analysis/visualization/spike_lfp/phase.py`
- new `src/neural_analysis/visualization/spike_lfp/ppc.py`
- new READMEs under `visualization`, `visualization/lfp_lfp`, and
  `visualization/spike_lfp`
- compatibility facade in `lfp_summary_plotting.py`
- plotting-only extraction from `lfp_phase_clustering.py` and the three LFP
  validation modules
- existing and new LFP-summary plotting tests

**Design and tasks:**

- Move pure Matplotlib construction only. Functions accept validated arrays and
  small immutable plot-context records, return figures/axes, and neither load,
  compute, save, nor import Streamlit.
- Keep report selection, captions tied to report schema, file naming, and
  artifact publication outside visualization.
- Centralize only genuinely shared style/caption primitives. Domain meanings,
  axes, and panels remain named in their domain modules.
- Preserve light-mode defaults, opaque white background, black axes/text,
  readable fonts, labels, and captions with statistical meaning.
- Keep the old module as forwarding imports until NR18.

**Tests written first:**

- old and new entry points produce the same axes, artists, labels, captions,
  panel ordering, and selections for fixed cached arrays;
- figure functions do not call raw loaders, workflows, caches, or computation;
- unit/radian/Hz/seconds/dB labels are exact, including corrected microvolt
  amplitude axes;
- unavailable exemplars remain explicit and never create fake plots;
- figures close cleanly and repeated calls do not retain global state;
- import-direction tests forbid Streamlit and upper-layer dependencies;
- the phase-rate sparsity assessment is passed into visualization as prepared
  presentation context; visualization does not calculate it. The existing
  assessment remains at its compatibility location until NR12 moves it beside
  the Spike-LFP numerical result it summarizes.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_plotting.py \
  src/tests/neural_analysis/test_lfp_phase_clustering.py \
  src/tests/neural_analysis/test_lfp_power_validation.py \
  src/tests/neural_analysis/test_lfp_synchrony_validation.py \
  src/tests/neural_analysis/test_lfp_spike_phase_validation.py \
  src/tests/neural_analysis/visualization/lfp_lfp \
  src/tests/neural_analysis/visualization/spike_lfp
```

**Performance:** load no component twice, use array views for selections, save
and close one report figure at a time, and retain the approved bounded PNG set.

## 14. NR9 - Unit, spike-behavior, and population visualization extraction

**Depends on:** NR5 and NR8.

**Files:**

- new `src/neural_analysis/visualization/units/raster.py`
- new `src/neural_analysis/visualization/units/psth.py`
- new `src/neural_analysis/visualization/population/pca.py`
- new `src/neural_analysis/visualization/population/decoding.py`
- new `src/neural_analysis/visualization/cross_session/decoding.py`
- new READMEs for those visualization packages
- plotting-only changes in `unit_spike_plotting.py`, `psth_behavior.py`,
  `population_pca_decoding.py`, `population_pca_switch_trajectories.py`, and
  `plot_cross_session_analysis.py`
- corresponding existing/new plotting tests
- existing `test_single_trial_spike_lfp_hilbert_plotting.py` and
  `test_spike_lfp_phase_rate_plotting.py`

**Design and tasks:**

- Separate figure construction from selection, computation, file search, and
  saving in each oversized module.
- Keep plot inputs narrow: prepared arrays/tables plus explicit labels and
  physical context, not complete session objects.
- Avoid one generic plotting framework. Share only stable style primitives from
  `visualization/common.py`.
- Leave numerical functions in their current modules until NR13-NR14.

**Tests written first:**

- deterministic prepared inputs reproduce existing layouts, labels, pagination,
  trial ordering, event markers, and summary statistics;
- plotting does not mutate input arrays/dataframes or perform analysis/source
  loading;
- save functions, where retained temporarily, delegate to pure builders and
  preserve filenames;
- population and cross-session figures preserve date/session ordering and do
  not refit PCA or decoders;
- old public plotting imports forward.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_unit_spike_plotting.py \
  src/tests/neural_analysis/test_single_trial_spike_lfp_hilbert_plotting.py \
  src/tests/neural_analysis/test_spike_lfp_phase_rate_plotting.py \
  src/tests/neural_analysis/test_psth_behavior.py \
  src/tests/neural_analysis/test_population_pca_decoding.py \
  src/tests/neural_analysis/test_population_pca_switch_trajectories.py \
  src/tests/neural_analysis/test_plot_cross_session_analysis.py \
  src/tests/neural_analysis/visualization/units \
  src/tests/neural_analysis/visualization/population \
  src/tests/neural_analysis/visualization/cross_session
```

**Completion evidence:** no new numerical logic in visualization; rendering
smokes pass under a noninteractive backend; representative figures receive Sol
structural review, not a new scientific visual-approval claim.

## 15. NR10 - Metadata-driven Streamlit application shell

**Depends on:** NR2-NR9.

**Files:**

- new `src/neural_analysis/webapp/app.py`
- new `src/neural_analysis/webapp/routes.py`
- new `src/neural_analysis/webapp/state.py`
- new `src/neural_analysis/webapp/controls.py`
- new `src/neural_analysis/webapp/data_access.py`
- new `src/neural_analysis/webapp/views/cached_summary.py`
- new `src/neural_analysis/webapp/views/unit_activity.py`
- new `src/neural_analysis/webapp/views/trial_activity.py`
- new `src/neural_analysis/webapp/views/lfp_lfp.py`
- new `src/neural_analysis/webapp/views/spike_lfp.py`
- new `src/neural_analysis/webapp/views/population.py`
- new `src/neural_analysis/workflows/exploration/units.py`
- new `src/neural_analysis/workflows/exploration/lfp_lfp.py`
- new `src/neural_analysis/workflows/exploration/spike_lfp.py`
- new `src/neural_analysis/workflows/exploration/population.py`
- new `src/neural_analysis/workflows/exploration/README.md`
- update `src/neural_analysis/workflows/README.md` with the bounded exploratory
  workflow boundary and domain inventory
- new `src/neural_analysis/webapp/README.md`
- new `src/neural_analysis/cli/launch_webapp.py`
- new `src/tests/neural_analysis/workflows/exploration/test_units.py`
- new `src/tests/neural_analysis/workflows/exploration/test_lfp_lfp.py`
- new `src/tests/neural_analysis/workflows/exploration/test_spike_lfp.py`
- new `src/tests/neural_analysis/workflows/exploration/test_population.py`
- update `src/neural_analysis/README.md` with the final webapp launch and
  cache/snapshot inspection walkthrough
- compatibility wrappers in `psth_webapp.py` and `lfp_summary_webapp.py`
- new webapp/CLI tests and affected existing webapp tests
- concise launch how-to documentation

**Design and tasks:**

- Build one shell that loads an explicit metadata path, shows structural/action
  availability, and delegates through the NR7 explicit registry.
- Put bounded live-computation composition in the four explicit exploratory
  workflow modules, not in Streamlit views or `webapp/data_access.py`. These
  workflows consume validated resolved metadata, call the already-supported
  source and numerical compatibility APIs available at NR10, enforce explicit
  input/resource bounds, and return typed result/context records. They do not
  import Streamlit or plotting, publish artifacts, or invoke production cache
  execution. NR11-NR14 rebind them to the final pure analysis modules as those
  modules become available; NR10 must not import a later-wave target interface.
- Generate probe, site, pair, population, path, and cache controls from metadata
  while showing hardware identity and anatomy separately.
- Separate cached production views from exploratory live views in route labels
  and behavior.
- Snapshot inspection uses the saved snapshot configuration and receipt; live
  metadata cannot relabel it.
- Long computation remains outside rerenders. The app may validate, inspect,
  render cached results, run only the bounded exploratory workflows, and
  construct an explicit command handoff for production or long-running work.
- `webapp/data_access.py` owns metadata/artifact access only. Webapp modules may
  import session, artifact, visualization, and narrow workflow interfaces but
  never source adapters or numerical analysis modules directly.
- Missing optional inputs disable only affected routes with actionable reasons.
- The launcher accepts `--session-metadata PATH`; it contains no scientific
  configuration logic.
- The top-level README explains expected startup behavior, live versus snapshot
  inspection, selectors, compatibility/provenance status, and unavailable
  results without duplicating the view implementation.

**Tests written first:**

- explicit metadata is required and initializes stable state across rerenders;
- arbitrary labels and one-probe/incomplete sessions create correct controls;
- cached versus live source modes remain distinct and do not leak state;
- snapshot manifest/config/receipt control labels and provenance;
- cached routes call only artifact/data-access and visualization boundaries;
- exploratory routes call only bounded exploratory workflow and visualization
  boundaries; dependency scans reject direct webapp imports from `sources` or
  `analyses`;
- exploratory cache keys include every resolved source fingerprint and adapter
  value-semantics version consumed by the workflow, including authoritative
  Open Ephys and SpikeGLX sidecar digests;
- production/long-run compute actions produce reviewed command requests and
  never execute in Streamlit; bounded exploratory computation runs only after
  an explicit user action, never during startup or an unrelated rerender, and
  unchanged requests reuse their complete-identity cache entry;
- old entry points forward during compatibility;
- startup smoke uses a synthetic metadata file with no raw data;
- the documented launch command parses successfully and the synthetic smoke
  reaches the metadata-derived shell without computation.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_psth_webapp.py \
  src/tests/neural_analysis/workflows/exploration \
  src/tests/neural_analysis/webapp \
  src/tests/neural_analysis/cli/test_launch_webapp.py
```

**Performance:** use bounded Streamlit caches for metadata, small tables, and
explicitly requested bounded exploratory results; never cache a second
production component copy; close figures; no raw wavelet, PPC, or batch
computation during startup or an ordinary rerender. Any supported bounded
exploratory transform runs only on explicit request under its tested
trial/unit/time/frequency allocation limits.

## 16. NR11 - LFP-LFP scientific analysis package

**Depends on:** NR3-NR4 and NR8; no workflow move yet.

**Files:**

- new modules under `src/neural_analysis/analyses/lfp_lfp`: `power.py`,
  `spectrogram.py`, `wavelet_phase.py`, `relative_phase.py`, `synchrony.py`, and
  `README.md`
- compatibility facades in `lfp_power_summary.py`, `lfp_spectrogram.py`,
  `lfp_phase_clustering.py`, and `lfp_synchrony_summary.py`
- rebind `src/neural_analysis/workflows/exploration/lfp_lfp.py` from the
  compatibility APIs to the new pure analysis imports without changing its
  public workflow contract
- configuration extraction/compatibility changes in `lfp_summary_models.py`
- corresponding existing tests and mirrored target-package tests

**Design and tasks:**

- Move pure numerical kernels over explicit arrays/configuration only.
- Split result serialization and residual plotting out of current modules before
  moving functions.
- Move LFP-analysis configuration records currently imported from
  `lfp_summary_models.py` beside their owning pure analyses, leaving exact
  compatibility re-exports for workflow callers.
- Keep `make_phase_condition_masks` and `select_reference_trial_indices` in
  compatibility code until NR13 moves their behavior-table selection logic to
  `analyses/spike_behavior`; the new LFP numerical modules must not import
  `spike_behavior_pynapple`.
- Keep the three existing `lfp_phase_clustering.py` save entry points as thin
  compatibility wrappers over `artifacts/legacy_result_files.py`; preserve
  their NPZ schemas and move no file I/O into `analyses/lfp_lfp`.
- Preserve Welch definitions, canonical frequency interpolation, notch behavior,
  reference normalization, band integration, wavelet parameters, phase
  conventions, bootstrap seeds, exclusions, and missingness.
- Avoid a shared analysis base class. Small immutable result dataclasses may
  remain next to the functions whose arrays they describe.

**Tests written first:**

- old/new imports and deterministic outputs match exactly where stable;
- corrected NR1 Power/Synchrony regression fixtures match arrays, selections,
  seeds, and identities;
- phase-angle comparisons use circular differences and reviewed tolerances;
- synthetic edge cases protect half-open windows, constant/nonfinite traces,
  line exclusions, pair order, and unavailable references;
- analysis modules perform no file I/O, cache writes, plotting, or Streamlit
  import;
- the bounded exploratory LFP-LFP workflow retains exact results and source
  identity while importing the new pure analysis modules;
- API/source checks confirm required SciPy/Pynapple calls against installed
  implementations when first relocated behind a new seam.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_power_summary.py \
  src/tests/neural_analysis/test_lfp_spectrogram.py \
  src/tests/neural_analysis/test_lfp_phase_clustering.py \
  src/tests/neural_analysis/test_lfp_synchrony_summary.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/analyses/lfp_lfp \
  src/tests/neural_analysis/workflows/exploration/test_lfp_lfp.py
```

**Performance:** characterize Welch/wavelet/synchrony runtime and memory before
and after moves; no recomputation per condition; preserve bounded block design.

## 17. NR12 - Spike-LFP scientific analysis package

**Depends on:** NR11 and approved existing PPC performance baseline.

**Files:**

- new modules under `src/neural_analysis/analyses/spike_lfp`:
  `phase_sampling.py`, `phase_locking.py`, `ppc.py`, `ppc_kernel.py`, and
  `README.md`
- compatibility facades in `spike_lfp_hilbert_phase.py`,
  `spike_lfp_phase_locking.py`, `spike_lfp_summary.py`, and
  `lfp_summary_ppc_kernel.py`
- rebind `src/neural_analysis/workflows/exploration/spike_lfp.py` to the new
  pure analysis imports without changing its public workflow contract
- existing spike-LFP/PPC kernel tests plus mirrored target tests
- no launcher, work-cache, or multiprocessing-runtime move in this package

**Design and tasks:**

- Move only pure Hilbert/wavelet sampling, phase-rate, descriptive phase-rate
  sparsity assessment, PPC, null, FDR, exemplar, and sufficient-statistic
  kernels. The sparsity assessment remains a presentation diagnostic rather
  than a significance or eligibility criterion; visualization receives its
  prepared result and does not recompute it.
- Preserve stable IDs, exact-sample semantics, epoch boundaries, overlap,
  schedules, seeds, inference eligibility, BH families, histogram definitions,
  and numerical tolerance policy.
- Keep execution planning, workers, checkpoints, mmap ownership, and progress in
  their current runtime until NR15/NR17.
- Retain performance-critical functions as explicit readable kernels rather
  than hiding them behind general abstractions.
- Keep `save_single_trial_spike_lfp_hilbert_result` and
  `save_spike_lfp_phase_locking_result` as compatibility wrappers over
  `artifacts/legacy_result_files.py`. Their named-array/metadata contracts remain
  covered by existing round-trip tests; target analysis modules perform no
  filesystem I/O.

**Tests written first:**

- old/new pure-kernel results and errors match;
- PPC and null schedules are deterministic and worker/block independent;
- near-grid, boundary, overlap, sparse, tie, and empty eligibility fixtures
  retain approved decisions;
- phase-rate sparsity counts, occupancy warnings, and thresholds match the old
  entry point, and visualization consumes the prepared assessment without
  numerical work;
- corrected scaling changes only cached amplitude traces, not phase/PPC
  scientific meaning, under NR1 tolerances;
- no source, artifact, workflow, report, plotting, or Streamlit imports;
- the bounded exploratory Spike-LFP workflow retains exact results and source
  identity while importing the new pure analysis modules;
- representative kernel allocation and runtime remain within the approved PPC
  gates.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_spike_lfp_hilbert_phase.py \
  src/tests/neural_analysis/test_spike_lfp_phase_locking.py \
  src/tests/neural_analysis/test_spike_lfp_phase_rate_sparsity.py \
  src/tests/neural_analysis/test_spike_lfp_summary.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_kernel.py \
  src/tests/neural_analysis/analyses/spike_lfp \
  src/tests/neural_analysis/workflows/exploration/test_spike_lfp.py
```

**Performance:** run the frozen seeded microbenchmarks and allocation estimates;
material regression or extra full-size copies blocks completion.

## 18. NR13 - Spike-behavior analysis package

**Depends on:** NR5 and NR9.

**Files:**

- new `src/neural_analysis/analyses/spike_behavior/binning.py`
- new `src/neural_analysis/analyses/spike_behavior/psth.py`
- new `src/neural_analysis/analyses/spike_behavior/decoding.py`
- new `src/neural_analysis/analyses/spike_behavior/README.md`
- numerical extraction/facades in `psth_behavior.py` and
  `spike_behavior_pynapple.py`
- rebind `src/neural_analysis/workflows/exploration/units.py` to the new pure
  spike-behavior analysis imports without changing its public workflow contract
- existing PSTH/spike-behavior tests and new mirrored target tests
- category-5 `spike_behavior_analysis.py` and `spike_behavior_binning.py` are
  read-only evidence in this package, not deletion targets

**Design and tasks:**

- Separate trial/unit selection, binning, PSTH, feature-table construction, and
  decoding kernels from loading, plotting, and CLI code.
- Move the established condition-mask logic used by
  `lfp_phase_clustering.make_phase_condition_masks` and
  `lfp_spectrogram.select_reference_trial_indices` here, then leave those old
  names as compatibility forwards. This prevents final LFP-LFP analysis modules
  from depending on the current mixed-responsibility spike/behavior module.
- Pass explicit columns/arrays when helpers use only part of a dataframe;
  retain whole dataframes only for operations whose contract is row-aligned
  table validation/transformation.
- Preserve Pynapple-native supported paths and do not revive superseded code to
  satisfy the target organization.

**Tests written first:**

- trial masks, bin edges, half-open windows, rates, unit identities, missing
  events, and choice/context conventions remain exact;
- decoding split identities and seeds are deterministic with no train/test
  leakage;
- numerical modules do not load files, plot, save, or import Streamlit;
- wrappers preserve public behavior;
- arbitrary nonempty metadata-provided region/site labels remain opaque values
  in target analyses and exploratory workflows; any legacy HPC/V1/PFC choice
  restriction remains only in a compatibility entry point;
- the bounded exploratory unit/trial workflow retains exact results and source
  identity while importing the new pure analysis modules;
- category-5 code has a documented caller/notebook inventory but remains
  untouched.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_psth_behavior.py \
  src/tests/neural_analysis/test_spike_behavior_pynapple.py \
  src/tests/neural_analysis/analyses/spike_behavior \
  src/tests/neural_analysis/workflows/exploration/test_units.py
```

**Completion evidence:** existing analysis tables and figures consume the new
pure outputs without numerical drift; no new dependency or speculative API.

## 19. NR14 - Population and cross-session analysis packages

**Depends on:** NR5, NR9, and NR13.

**Files:**

- new modules under `analyses/population`: `pca.py`, `decoding.py`,
  `switch_trajectories.py`, and `README.md`
- new `analyses/cross_session/decoding.py` and README
- new `workflows/cross_session/decoding.py` and README
- rebind `src/neural_analysis/workflows/exploration/population.py` to the new
  pure population analysis imports without changing its public workflow contract
- compatibility facades in `population_pca.py`,
  `population_pca_decoding.py`, `population_pca_switch_trajectories.py`, and
  `plot_cross_session_analysis.py`
- existing population/cross-session tests and new mirrored analysis/workflow
  tests

**Design and tasks:**

- Move rate-tensor construction, normalization, PCA, decoding, switch
  trajectory, cross-session table transformation/aggregation, and statistics
  separately from plotting and CLI concerns.
- Keep `analyses/cross_session` pure: it accepts already loaded tables and
  returns transformed/aggregated tables or statistical results without paths or
  file I/O. `workflows/cross_session` owns explicit artifact selection, path/date
  filtering, CSV loading, output naming, and publication through
  `artifacts/tabular.py`.
- Generic output names use validated stable IDs or a tested deterministic safe
  encoding, never raw display/region labels. Path separators, traversal, and
  collisions fail closed. The CT014 compatibility adapter preserves the
  established HPC/V1/PFC filenames exactly.
- Preserve observation axes, units, zero-variance handling, explained variance,
  CV grouping, scaler/PCA fitting inside training folds, trial/date ordering,
  and missingness.
- The cross-session workflow consumes documented single-session
  artifacts/tables; it must not reopen raw neural sources when cached sufficient
  data exists.

**Tests written first:**

- exact tensor axes and seeded synthetic PCA results;
- no CV leakage and stable split identities;
- switch definitions, direction, trial filters, and trajectory ordering;
- workflow-level cross-session file selection, date filters, and filenames;
  analysis-level table aggregation remains path-free;
- arbitrary nonempty metadata-provided region labels remain opaque values in
  target population/cross-session analyses and workflows; legacy
  HPC/V1/PFC-only normalization is confined to compatibility;
- atomic tabular publication preserves previous bytes on injected failure and
  preserves the existing successful filenames, columns, rows, and overwrite
  behavior;
- arbitrary labels cannot escape the output directory or collide after safe
  encoding, while CT014 compatibility retains its reviewed legacy filenames;
- the bounded exploratory population workflow retains exact results and source
  identity while importing the new pure analysis modules;
- old/new import equivalence; analysis modules have no plotting, source, or
  workflow dependencies;
- benchmark representative tensor construction and decoder fits.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_population_pca.py \
  src/tests/neural_analysis/test_population_pca_decoding.py \
  src/tests/neural_analysis/test_population_pca_switch_trajectories.py \
  src/tests/neural_analysis/test_plot_cross_session_analysis.py \
  src/tests/neural_analysis/analyses/population \
  src/tests/neural_analysis/analyses/cross_session \
  src/tests/neural_analysis/workflows/cross_session \
  src/tests/neural_analysis/workflows/exploration/test_population.py
```

## 20. NR15 - LFP-summary workflow decomposition

**Depends on:** NR3-NR7 and NR11-NR12.

**Files:**

- new files under `src/neural_analysis/workflows/lfp_summary`: `models.py`,
  `payloads.py`, `preparation.py`, `pipeline.py`, `power.py`, `synchrony.py`,
  `spike_phase.py`, and `README.md`
- compatibility facades in `lfp_summary_models.py`,
  `lfp_summary_preparation.py`, `lfp_summary_pipeline.py`,
  `lfp_summary_payloads.py`, `lfp_summary_runtime.py`, and
  `lfp_summary_ppc_runtime.py`
- affected complete LFP-summary model/preparation/runtime/PPC/pipeline tests and
  new mirrored workflow tests

**Design and tasks:**

- Split shared preparation and component orchestration by scientific component.
- Move the Power/Synchrony/Spike-phase array schemas and payload validation from
  `lfp_summary_payloads.py` to workflow-owned `payloads.py`; generic artifact
  code accepts supplied schemas and does not learn component science.
- Workflows receive validated session/configuration, source callables, artifact
  writers, and progress callbacks through narrow explicit dependencies.
- Keep pure analysis in `analyses`, acquisition I/O in `sources`, and generic
  run lifecycle in `execution`.
- Preserve component fingerprints, corrected source semantics, trial/site/unit
  identities, progress, checkpoint behavior, cleanup deferral, worker-count
  invariance, seeds, final array schemas, and Compute All ordering.
- Move PPC runtime pieces only when their owner is clear: scientific planning
  stays with the Spike-phase workflow; generic process lifecycle waits for
  NR17.

**Tests written first:**

- old/new workflow entry points return exact component payloads/status;
- injected sources prove no acquisition inference or hidden loading;
- Power/Synchrony/Spike failure and retry states remain isolated by component;
- Compute All does not recompute shared preparation incorrectly;
- cold/warm prepared phase, checkpoint resume, transaction, progress, and
  deferred cleanup remain exact;
- corrected CT026 regression fixture matches NR1 arrays and scientific
  configuration under the approved equality policy; source-identity changes
  are limited to the explicitly tested NR3/NR5 completeness migrations;
- dependency scans enforce layering.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_lfp_summary_preparation.py \
  src/tests/neural_analysis/test_lfp_summary_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_payloads.py \
  src/tests/neural_analysis/test_lfp_synchrony_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py \
  src/tests/neural_analysis/test_lfp_summary_pipeline.py \
  src/tests/neural_analysis/test_lfp_summary_synthetic_integration.py \
  src/tests/neural_analysis/workflows/lfp_summary
```

**Performance:** preserve prepared-phase mmap sharing, bounded PPC allocation,
eight-worker CT026 production preference without changing portable default,
and accepted runtime/memory gates. Run no real CT026 compute in this package.

## 21. NR16 - Reports, dataset compatibility, and profiling isolation

**Depends on:** NR8-NR15.

**Files:**

- new `src/neural_analysis/reports/common.py`, `power.py`, `synchrony.py`,
  `spike_phase.py`, and `README.md`
- new `src/neural_analysis/compatibility/ct014.py`,
  `compatibility/ct026.py`, and README
- new `src/neural_analysis/profiling/ppc.py`, `profiling/ct026.py`, and README
- report/builder extraction from `lfp_power_validation.py`,
  `lfp_synchrony_validation.py`, `lfp_spike_phase_validation.py`, and CT026
  profile modules
- CT014/CT026-binding extraction, if still present after their earlier
  migrations, from the top-level compatibility facades
  `unit_spike_loading.py`, `sync_ephys.py`, `psth_webapp.py`,
  `psth_behavior.py`, `spike_behavior_pynapple.py`,
  `plot_cross_session_analysis.py`, `lfp_summary_models.py`, and
  `lfp_summary_webapp.py`
- CT026 production-binding extraction from `lfp_spike_phase_launcher.py`; NR16
  moves only dataset-specific paths/defaults/builders, while NR17 retains
  ownership of the generic execution-lifecycle decomposition
- exact CT026/profile inputs:
  `lfp_summary_ct026_profile_adapter.py`,
  `lfp_summary_ct026_profile_locks.py`,
  `lfp_summary_ct026_profile_runner.py`, and
  `lfp_summary_ppc_profile.py`
- corresponding report, validation, profile, adapter, runner, and production-
  binding tests

**Design and tasks:**

- Reports consume validated cache arrays/manifests and visualization functions,
  create captions/summaries/logs, and publish immutable human-readable output.
  They never calculate scientific results or reopen raw sources.
- Move CT014 and CT026 paths, subject/session identities, fixed sites/channels,
  probe-to-region maps, date selections, active-population builders, legacy
  snapshot bindings, and dataset-specific CLI/example defaults under the
  corresponding compatibility module only. Compatibility preserves reviewed
  old commands/builders; generic analyses and workflows accept arbitrary
  metadata-provided identities and regions.
- Move representative profiling fixtures, cost models, and profile lifecycle
  under profiling. Production numerics remain imported from analyses/workflows.
- Generic packages and supported top-level facades must contain no dataset-
  specific implementation value, including CT014/CT026 absolute experiment
  paths or subject/session defaults, fixed probe-to-region meaning, fixed
  channel selection, fixed date range, or fixed two-probe construction after
  this package. Temporary
  top-level compatibility facades may contain explicitly deprecated
  dataset-named symbols and forwarding imports until their NR18 disposition is
  approved, but the values and behavior they forward live only in
  `compatibility`. Category-5 modules remain separately quarantined, are not
  imported by supported code, and stay explicit scan exceptions only until
  their NR18 user-approved disposition.

**Tests written first:**

- cache-only report rendering calls no sources or analyses;
- staged JSON/log/Markdown/PNG validation and atomic publication retain exact
  failure behavior and cleanup ownership;
- report schemas, selections, captions, unavailable-versus-zero semantics, and
  measurements match current contracts;
- corrected CT026 builders reproduce NR1 configuration and reports;
- CT014 compatibility builders reproduce the existing inspected sorter,
  channel-preset, synchronization, cross-session, and example-command defaults;
- arbitrary nonempty subject, probe, site, and region labels pass through
  generic session, analysis, workflow, and webapp contracts without a
  CT014/CT026 branch;
- old CT026 snapshots remain inspectable and explicitly legacy-unscaled;
- profiling produces only scalar/work evidence and cannot publish a scientific
  component;
- repository scan finds dataset-specific implementation values and paths,
  including CT014/CT026, only in approved compatibility, profiling,
  tests/fixtures, historical docs, and the explicitly inventoried category-5
  quarantine. The scan covers subject/session literals and patterns, absolute
  experiment paths, fixed channel/date arrays,
  probe-to-region maps, and fixed two-probe constructors rather than relying
  only on the strings `CT014` or `CT026`. Any dataset text in a temporary
  top-level facade is limited to deprecated symbol names, forwarding imports,
  or deprecation documentation; an AST/source audit proves that it contains no
  dataset-specific value or construction logic.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_power_validation.py \
  src/tests/neural_analysis/test_lfp_synchrony_validation.py \
  src/tests/neural_analysis/test_lfp_spike_phase_validation.py \
  src/tests/neural_analysis/test_ct026_open_ephys_defaults.py \
  src/tests/neural_analysis/test_lfp_summary_models.py \
  src/tests/neural_analysis/test_unit_spike_loading.py \
  src/tests/neural_analysis/test_sync_ephys.py \
  src/tests/neural_analysis/test_psth_webapp.py \
  src/tests/neural_analysis/test_psth_behavior.py \
  src/tests/neural_analysis/test_spike_behavior_pynapple.py \
  src/tests/neural_analysis/test_plot_cross_session_analysis.py \
  src/tests/neural_analysis/test_lfp_summary_webapp.py \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py \
  src/tests/neural_analysis/test_lfp_summary_ct026_profile_adapter.py \
  src/tests/neural_analysis/test_lfp_summary_ct026_profile_runner.py \
  src/tests/neural_analysis/test_lfp_summary_ct026_profile_production_binding.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_profile.py \
  src/tests/neural_analysis/reports \
  src/tests/neural_analysis/compatibility \
  src/tests/neural_analysis/profiling
```

## 22. NR17 - Generic execution, launcher, and Slurm boundary

**Depends on:** NR7, NR15, and NR16.

**Files:**

- new `src/neural_analysis/execution/run_state.py`
- new `src/neural_analysis/execution/resources.py`
- new `src/neural_analysis/execution/launcher.py`
- new `src/neural_analysis/execution/slurm.py`
- new `src/neural_analysis/execution/README.md`
- compatibility facade in `lfp_spike_phase_launcher.py`
- generic process/executor/resource extraction from
  `lfp_summary_ppc_runtime.py`; its scientific PPC job planning/reduction stays
  in `workflows/lfp_summary/spike_phase.py`
- CLI integration in `cli/run_cache.py`
- update `src/neural_analysis/README.md` with the tested Slurm handoff, run
  completion checks, resume/recovery command, and final output-inspection steps
- existing launcher/HPC tests and new execution tests
- existing repository `src/shell_scripts/hpc_ppc.sh` only if a thin
  compatibility forwarding
  change is required and separately visible in the package diff

**Design and tasks:**

- Generalize persisted run state, locks, resource measurement, signals,
  local/Slurm invocation, resume, cleanup-only recovery, report recovery, and
  rerender around narrow workflow run contracts.
- Scientific configuration is resolved and frozen before submission. Shell
  code supplies environment/resources only.
- Resume/recovery accepts the saved run directory and rejects replacement
  session/configuration inputs.
- Preserve process-tree RSS/PSS measurement, atomic state/log/summary writes,
  signal behavior, lock recovery, repository identity, and deferred cleanup.
- Keep local and Slurm scientific identity identical; execution resources do
  not enter final component fingerprints.
- Keep the README's cluster material short and operational: one supported
  submission example, how to find state/log/report paths, and how to resume.
  Link to detailed execution documentation for scheduler/resource options.

**Tests written first:**

- parsing and preflight are side-effect-free and metadata driven;
- local and Slurm commands contain identical saved scientific configuration;
- run-state transitions and ordered stages reject invalid recovery;
- lock ownership, stale recovery, foreign host/live PID, signal interruption,
  worker failure/cancellation, and cleanup are exact;
- resume/report recovery/rerender cannot switch session or component identity;
- path containment and symlink defenses remain enforced;
- old launcher commands forward;
- synthetic end-to-end execution publishes only after every component/report
  gate and leaves recoverable state on injected failure.
- every README execution/resume command is covered by parser or no-submit smoke
  tests, and every named output path matches the synthetic run contract.

**Focused RED/GREEN:**

```text
uv run pytest -q -p no:cacheprovider \
  src/tests/neural_analysis/test_lfp_spike_phase_launcher.py \
  src/tests/neural_analysis/test_hpc_ppc_shell.py \
  src/tests/neural_analysis/test_lfp_spike_phase_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py \
  src/tests/neural_analysis/test_lfp_summary_ppc_parallel.py \
  src/tests/neural_analysis/execution \
  src/tests/neural_analysis/cli/test_run_cache.py
```

**Performance and safety:** dry run opens no raw arrays; preflight uses existing
bounded estimates; worker spawn occurs only after aggregate allocation checks;
no real cluster submission is part of automated verification.

## 23. NR18 - Compatibility retirement and final documentation

**Depends on:** all prior packages and a user-reviewed external-caller audit.

**Files:** exact removal/update list is produced and approved during NR18 test-
design review. No file is deleted merely because it appears unused.

**Tasks:**

1. Search repository imports, commands, notebooks, README/docs, saved run
   commands, and likely external scripts identified by the user.
2. Classify every compatibility wrapper and category-5 module as retain,
   deprecate, or remove, with evidence and recovery implications.
   The complete category-5 inventory is `behavior_pynap.py`,
   `spike_behavior_analysis.py`, `spike_behavior_binning.py`,
   `modified_sinc_smoother.py`, and `plot_single_session_analysis.py`.
3. Present the exact deletion/deprecation list for explicit approval.
4. Remove only approved wrappers/modules; update callers and documentation.
5. Verify every package README against actual files, responsibilities, public
   entry points, shapes/units, algorithms, determinism, caches, parallelism,
   runtime, and memory.
6. Finalize `src/neural_analysis/README.md` as the single concise end-user
   quickstart. Remove stale or duplicate operational instructions from other
   current docs by replacing them with links, without rewriting historical
   audit records.
7. Perform the novice-path review required by Section 2.9 and correct any step
   that requires undocumented code knowledge or a command that does not match
   `--help`.
8. Run final metadata creation, dry-run, cache-only webapp, legacy snapshot,
   corrected CT026 snapshot, and supported non-CT026 session smokes.

**Tests/checks written first:**

- import/command tests prove approved old paths are no longer required;
- retained compatibility readers cover every artifact that must remain
  inspectable;
- dependency scan has no upward arrows or dataset-specific leakage outside the
  approved compatibility/profiling/quarantine locations;
- package and README inventories agree;
- every quickstart command is covered by a parser/smoke test and every named
  output exists in the corresponding synthetic workflow fixture;
- a read-only reviewer following only `src/neural_analysis/README.md` can locate
  JSON creation, validation, local cache, Slurm handoff, run status, outputs,
  resume, webapp launch, and snapshot inspection;
- category-5 removals have no known caller and explicit approval record;
- all public functions satisfy type/shape/axis/unit/return documentation rules;
- full neural and repository suites are green from the final topology.

**Focused command:** the tests-only phase defines exact removed-import RED
checks after the deletion list is approved. Final verification always includes:

```text
uv run pytest -q -p no:cacheprovider src/tests/neural_analysis
uv run pytest -q -p no:cacheprovider
```

## 24. Wave gates and acceptance evidence

### After every package

- Tests-only commit exists before implementation commit and records genuine RED.
- Lead reproduced RED and GREEN independently.
- Assigned Sol review is accepted with every finding resolved or explicitly
  deferred by the user.
- Only allowlisted files changed; public interfaces remain or have reviewed
  wrappers.
- Docstrings specify types, shapes, axes, units, returns, and errors.
- Package README and execution log are current.
- No unauthorized real-data, cluster, cache, or external-state mutation
  occurred.
- All agents are idle; lead records model/effort/permissions and whether
  read-only enforcement was technical or procedural.

### Wave-boundary review

- Complete repository suite passes.
- Import/dependency scan matches the intended layer direction.
- Corrected CT026 regression fixtures remain compatible.
- Performance-sensitive paths show no material unexplained regression.
- Worktree inventory matches the expected package commits and preserves
  unrelated user files.
- User receives the completed-wave evidence and approves any real-data or
  deletion action required by the next wave.

### Final acceptance

- A new supported session can be described with `neural_session.json`, fully
  validated, dry-run, computed locally or through Slurm, resumed from saved run
  state, copied/verified, and inspected in the webapp without edits to shared
  analysis, workflow, source-adapter, or webapp code. Editing the dedicated
  metadata-creator values remains an approved authoring option.
- CT026 corrected Power, Synchrony, and authorized Spike-phase artifacts remain
  the regression reference; legacy unscaled artifacts remain explicitly
  inspectable.
- CT014 SpikeGLX metadata validates through the same session/workflow boundary.
- Numerical definitions, seeds, identities, axis/unit contracts, component
  schemas, cache transactions, and report meaning are unchanged except for the
  explicitly approved Open Ephys scaling correction.
- No generic production module contains dataset-specific paths or identities,
  fixed PFC/HPC/V1 meanings, dataset channel/date presets, or fixed two-probe
  assumptions.
- Category-5 code and compatibility wrappers have explicit final dispositions.
- Final package topology and all READMEs match the design document.
- `src/neural_analysis/README.md` is the single tested end-user quickstart for
  creating JSON, running/inspecting caches, launching the webapp, and recovering
  a forgotten workflow.

## 25. Approved decisions and remaining authorization gates

The user approved planning decisions 1-9 on 2026-09-22 and the follow-up audit
resolutions 10-17 on 2026-09-23:

1. Use a source-value semantics version plus preprocessing sidecar fingerprint,
   rather than a global manifest schema bump, to invalidate legacy Open Ephys
   numerical components.
2. Preserve legacy CT026 artifacts unchanged and label them as historical
   unscaled outputs; publish corrected artifacts at new paths.
3. Use corrected CT026 artifacts as the structural regression baseline only
   after staged Power, Synchrony, 100-shuffle preview, and—if separately
   authorized—1,000-shuffle validation.
4. Keep NR0 narrowly scoped to scaling and compatibility identity; defer the
   new session model and source-package moves to later packages.
5. Preserve an editable Python metadata creator so users can generate JSON with
   the intended values, while also supporting an incomplete/default skeleton
   that can be edited directly as JSON.
6. Require initial metadata at `<session_home>/neural_session.json`; defer
   external metadata locations, absolute source paths, and root remapping.
7. Implement the exact observed Open Ephys affine value contract using
   per-channel gain, offset, and unit metadata. Do not treat
   `lfp_binary_scaling` as a scalar or bare vector, and do not silently ignore
   offsets.
8. Content-hash small authoritative preprocessing/metadata sidecars while
   retaining the reviewed size/mtime policy for production-sized binaries.
9. Keep cross-session file I/O in a workflow boundary, keep visualization free
   of sparsity computation, and move dataset-specific values under
   compatibility before applying the generic-code isolation scan.
10. Treat every consumer of the corrected Open Ephys loader as part of NR0's
    unit/provenance blast radius. Exploratory Streamlit caches use the adapter
    semantics version plus authoritative sidecar digest, and exploratory labels
    and saved metadata report physical microvolts truthfully.
11. Require current production LFP-summary unit and sample-rate configuration to
    match authoritative Open Ephys metadata before computation; later resolved
    session configurations obtain those values from the adapter rather than
    user transcription.
12. Preserve legacy canonical-config deserialization by storing adapter
    semantics in component provenance/fingerprint input and treating an absent
    historical field as legacy only inside receipt-validated snapshot
    inspection.
13. Isolate supported CT014 as well as CT026 paths, identities, channel/date
    presets, and fixed topology under compatibility, quarantine other legacy
    dataset examples pending NR18, and scan for dataset-specific values rather
    than only known dataset-name strings.
14. Put bounded live exploratory composition in explicit non-Streamlit workflow
    modules; webapp code consumes those interfaces and never imports source or
    analysis implementations directly.
15. Publish cross-session CSV outputs through a generic atomic tabular artifact
    writer that has no scientific schema knowledge.
16. Support only the observed canonical Open Ephys `float32` storage declaration
    in the initial adapter and fail closed on unreviewed dtype variants.
17. Treat workflow-supplied output names as untrusted path input: the generic
    tabular writer accepts only validated direct-child filenames, generic
    workflows use stable identifiers or collision-checked safe encodings rather
    than raw display labels, and compatibility alone preserves reviewed legacy
    filenames.

The following remain separate explicit authorization gates:

- implementation start after final review of this comprehensive plan;
- every real CT026 numerical run in NR1, beginning with dry run and small-window
  checks according to their stated mutation level;
- the corrected 1,000-shuffle final run after preview review;
- any Slurm submission or external copy/registration action; and
- the NR18 deletion/deprecation list.

Approval of this plan does not imply any of those actions.
