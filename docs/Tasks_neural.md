# LFP Summary Analysis Implementation Plan

## Live handoff snapshot

- **Snapshot:** 2026-09-20 on branch `refactor`.
- **Current accepted implementation HEAD before this documentation update:**
  `ce37409` (`adding uv to git tracking`), equal to `origin/refactor`. The
  launcher implementation is `814f253`; the launcher handoff is `ed350ff`.
- **Last verified neural baseline:** 1,139 passed with 20 warning instances after
  the launcher implementation. The warnings are the existing
  multiprocessing-fork, intentional duplicate-ZIP-member, and Pynapple
  zero-duration fixture warnings recorded in the launcher handoff below.
- **WP5C state:** the exact PPC speedup S0-S8 sequence is complete. The accepted
  work-only S8 run is
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_ppc_s8_2026-09-18T15-06-33Z`.
  It selected eight workers for the benchmarked CT026 production workload and
  published no scientific component, manifest, preview, or `spike_phase.npz`.
- **Approval and execution state:** Power and Synchrony reports remain approved.
  The local ProbeB 100-shuffle Spike-phase computation completed all 105 blocks
  and committed a compatible final component, but its launcher stopped during
  report rendering with `bottom cannot be >= top`. The failure is a bounded
  caption/layout defect, not a numerical failure. Its work directory remains
  intact and no report or success cleanup was published.
- **Approved immediate sequence:** finish a documentation-first, test-first
  report-recovery package; recover and inspect the local 100-shuffle report
  without recomputation; then implement the simple SLURM/uv execution wrapper
  and cluster-side cache inspection path before WP11. The separately approved
  ProbeB 1,000-shuffle final run will use the cluster after wrapper/dry-run
  validation. Future 100-shuffle runs should use the same cluster path. WP13
  remains deferred and every nonempty absolute-amplitude request remains
  rejected before computation.
- **Do not redo completed packages:** WP0, WP1, WP2 preparation, WP3, WP4,
  WP5A, WP5B, and the Spike-phase runtime bridge are historical completed work.
  Remaining integration work must extend them through the packages below, not
  reassign or reimplement them.

Worktree warning at the snapshot: preserve all unrelated untracked files and
directories. After `pyproject.toml`, `uv.lock`, and `hpc_ppc.sh` became tracked,
the 168-entry normal untracked inventory has NUL-delimited status SHA-256
`9e406da5c668447cda38b3f8387d620df72e3bd5b705a2b6cac756116056f87f`.
Agents must stage only files assigned to their package and must stop on
unexpected tracked changes.

Source-of-truth hierarchy:

1. `docs/Tasks_neural.md` owns current status, frozen contracts, gates, package
   ownership, and remaining work.
2. `docs/ppc_speedup_plan.md` owns the approved detailed S0-S8 sequencing for
   the remaining exact PPC computation upgrade.
3. `docs/webappDesign.md` owns scientific and user-interface requirements.
4. `docs/DevJournal.md` is a historical record only and is not authoritative
   for current gates or sequencing.

## Current implementation status

| Area | State | Principal evidence | Remaining work |
| --- | --- | --- | --- |
| WP0 baseline repair | Complete | Implementation `8720011`; baseline command above is green at 617 passed | None |
| WP1 contracts/cache I/O | Complete | Tests `8e4a2dc`, `3edb956`, `ede43de`; implementation `4ecccd7`; `test_lfp_summary_models.py`, `test_lfp_summary_io.py` | WP5C execution-only cache contracts must remain separate from final component compatibility |
| WP2 preparation | Complete for existing production preparation | Tests `0fa1f18`, `7ab4e85`, `b3db525`; implementations `596d6c5`, `23d643e`; `test_lfp_summary_preparation.py` | Apply configured absolute amplitude thresholds in WP13 |
| WP3 Power | Complete and user-approved | Numerical tests/implementation `fed9435`, `355307e`, `d6c3477`; runtime `8fafe04`, `2225eb5`; report `analysis_runs/CT026_2026-08-01_130853_lfp_power_validation_2026-08-25T16-17-40Z` | Full composed webapp integration in WP10-WP11 |
| WP4 Synchrony | Numerics, runtime, report, and user approval complete | Numerical implementation `d6c3477`; production runtime `ea542d1`; report tests/implementation `fa19ad0`, `f312957`; approved report `analysis_runs/CT026_2026-08-01_130853_lfp_synchrony_validation_2026-08-26T16-57-33Z` | Full composed webapp integration in WP10-WP11 |
| WP5A observed PPC | Complete | Tests `fed9435`, `355307e`; implementation `d6c3477`; `test_spike_lfp_summary.py` | None |
| WP5B shuffle inference | Complete | Tests `c94e152`, `7d22d6f`; implementation `01b7507`; `test_spike_lfp_summary.py` | Performance redesign only; scientific reference must remain unchanged |
| Spike runtime bridge | Complete, including grouped payload and exact exemplar-trial integration | Original bridge `c8e76d5`; grouped payload integration completed in S5; WP12 implementation `bd6fa50`; `test_lfp_summary_runtime.py` and synthetic cache/reload/plot coverage | UI integration in WP11 |
| WP5C optimization | Complete through S8 | Grouped-parallel implementation `e89a0ee`; S7 documentation `8741d71`; full S7 neural suite 1,113 passed; accepted work-only S8 run `ct026_ppc_s8_2026-09-18T15-06-33Z`; `docs/ppc_speedup.md`; `docs/ppc_speedup_plan.md`; `docs/ppc_speedup_execution_log.md` | Preserve eight-worker production configuration and exact preflight; use R1/C1 for report recovery and later cluster execution |
| WP6 plotting | Numerically complete; one real-data PPC report layout defect open | Original tests `b1f51f3`; implementation `d1c3e21`; paired PPC/report tests `2c80a9b`; local 100-shuffle failure evidence below | Replace the unbounded prevalence caption with the approved three-panel population map in recovery package R1 |
| WP7 pipeline | Complete for component, Compute All, composition, and report-before-cleanup seams | Original tests `b1f51f3`; implementation `067fdef`; WP10 `98ac2ed`; WP12 deferred-cleanup implementation `bd6fa50`; `test_lfp_summary_pipeline.py` | UI progress/state integration remains in WP11 |
| WP8 webapp | Partial; Power path only | Tests `c7ec7c1`, `003394a`, `f3954de`; implementations `2885ffd`, `926801b`; `test_lfp_summary_webapp.py` | After cluster package C1, add remote cluster-side cache inspection, Synchrony/Spike views, active population, and progress in WP11 |
| WP9 validation | Partial; 100-shuffle numerics complete, report incomplete | Approved Power and Synchrony reports; S8 benchmark; committed ProbeB cache completed 2026-09-18; exact evidence below | Complete R1 report recovery and inspect it; validate C1; then separately authorize the ProbeB 1,000-shuffle cluster run |
| WP10 composed dependencies | Complete | Tests and implementation through `98ac2ed`; completion handoff `3aebba8`; PPC contracts and worker decision are stable | None; consume the composed boundary from the forthcoming launcher and WP11 |
| R1 local preview recovery | Implementation verified; exact real-run recovery pending | Tests `b77f4a5`; implementation `97fbf3b`; failed launcher run and compatible `spike_phase.npz` identified 2026-09-20 | Recover the exact run from the next documented clean descendant commit, inspect every artifact, and record the result before C1 |
| C1 cluster execution | Planned after R1 inspection | Tracked `hpc_ppc.sh`, `pyproject.toml`, and `uv.lock` at `ce37409`; uv installed on cluster access; contract below | Verify uv hello world, implement/test the thin 72-hour SLURM wrapper, run metadata-only cluster checks, then request final-run approval |
| WP11 Streamlit integration | Planned after R1 and C1 infrastructure | Package below; remote topology frozen below | Inspect numerical caches using cluster-side Streamlit over an SSH tunnel; support one explicitly selected ProbeA or ProbeB and complete cached views |
| WP12 PPC plotting/reporting | Complete except the R1 real-data layout correction | Contract `5dfd439`; tests `2c80a9b`, corrections `7d9c322` and `220f5b0`; implementation `bd6fa50`; launcher integration `814f253` | R1 adds the approved count panel and bounded caption without changing numerical/report transactions |
| WP13 optional absolute-amplitude thresholds | Explicitly deferred; not a blocker while thresholds are empty | Package below; validation/fingerprint support exists; current CT026 threshold list is empty | Reject every nonempty request before computation until WP13 is separately implemented |

This document replaces the older decoding task list. It translates the scientific
requirements in `docs/webappDesign.md` into an implementation plan for offline LFP
summary computation and webapp-based inspection. Confirmed decisions are recorded
below so implementing agents do not invent scientific requirements.

## 1. Objective

Add coordinated single-session analyses for:

1. LFP power and band-power summaries.
2. Single-site ITPC, inter-site ISPC, and within-trial PLV.
3. Unit-level spike-LFP PPC, population summaries, and permutation significance.
4. High/low synchrony and phase-locking exemplars.
5. Interactive inspection of saved results in the existing Streamlit webapp.

The plan covers all areas in the scientific specification, but implementation
should proceed in independently testable phases.

## 2. Confirmed decisions

### 2.1 Offline computation and webapp visualization

- Summary calculations will run outside the interactive Streamlit render cycle.
- Computation will save an inspection cache to the session's `processed`
  directory by default, with an optional user-selected output directory.
- The webapp will load the saved cache and provide selectors and plots.
- Small single-trial previews may continue to compute interactively when their
  runtime is bounded.
- The cache may use a generic filename and may be overwritten by an explicit
  recomputation request.
- Because the file is overwriteable, it must be labeled and treated as a
  disposable inspection cache rather than a versioned publication result.
- Before loading, the webapp must verify cache metadata against the active
  session, source files, channels, parameters, and analysis schema version.
- Cache writes must be atomic so an interrupted run cannot leave a partially
  valid file at the final path.

### 2.2 Scope and order

The implementation plan covers power, synchrony, and spike-phase locking. This
dependency order is architectural history, not a list of pending work; the live
status matrix and bounded packages govern current execution:

1. Shared configuration, channel/site identities, preprocessing, trial masks,
   cache schema, and validation.
2. Trial PSDs, reference spectra, normalized spectra, and band power.
3. Frequency-resolved ITPC/ISPC and trial-level PLV.
4. Frequency-resolved PPC, band summaries, trial-shuffle nulls, and optional
   temporal-shift nulls.
5. Saved-result loaders and webapp summary views.
6. Single-session validation followed by optional batch execution.

### 2.3 LFP sites

- An LFP site is one user-selected saved channel, not an average across
  channels, a bipolar derivation, or a region-wide signal.
- The first implementation will support one selected PFC site and multiple
  selected HPC sites, including the PFC-HPC1, PFC-HPC2, and HPC1-HPC2 pairs in
  the scientific specification.
- Site identity must be stored explicitly as acquisition format, source path,
  probe label, saved-channel index, user-facing site label, sample rate, and
  source voltage unit.
- Data structures and saved schemas must allow additional PFC sites later
  without changing existing axis meanings or public contracts.
- Pair identities must be stored by stable site identifiers rather than by
  assuming fixed positional meanings for array indices.

### 2.4 Referencing and acquisition preprocessing

- The current LFP signals have the effects of existing 1-500 Hz bandpass
  preprocessing.
- They have not been explicitly common-average referenced for this analysis.
- Neuropixels reference and ground are wired together during preparation and
  connected to an external brain site in the opposite hemisphere.
- The summary pipeline will not silently apply CAR, bipolar referencing, or
  channel averaging.
- Cache metadata and figure captions must state the reference/preprocessing
  status so inter-site synchrony is not presented as reference-independent.
- Synchrony captions must explicitly note that activity at the external brain
  reference can contribute shared phase or power to recorded sites.
- The plan must include quality-control inspection for common-reference,
  volume-conduction, broadband transient, and movement-artifact concerns.

### 2.5 Trial conditions and optional behavioral filters

- Compute and retain all primary and overlapping conditions:
  `correct_rewarded`, `omission`, `incorrect`, `switch`, `stay`,
  `omission_switch`, `omission_stay`, `incorrect_switch`, and
  `incorrect_stay`.
- Reuse `spike_behavior_pynapple.make_trial_type_masks` as the authoritative
  condition definition. Legacy experimenter-reward columns are normalized at
  that existing boundary.
- Conditions are intentionally overlapping. Saved metadata and plot labels
  must not imply that condition samples are mutually exclusive.
- Left/right choice and left/right context are independent optional filters
  within a selected condition.
- Do not suppress a result solely because its group contains few trials.
- Every display and saved summary must expose the contributing trial count.
- Results with fewer than 10 contributing trials must carry a visible
  `unstable_low_trial_count` flag and warning. This is an inspection warning,
  not a validity threshold.

### 2.6 Alignment events

- `choice_time` is the default alignment event.
- `start_time` is also supported.
- LED alignment is out of scope.
- Trials missing the selected alignment timestamp are excluded with their
  indices and exclusion reason retained in the cache.

### 2.7 PSD and line-noise defaults

- Trial PSDs use Welch's estimator on the requested whole, before, or after
  interval.
- Defaults are a Hann window, constant detrending, 50 percent overlap, and
  approximately 2 Hz frequency resolution.
- At sample rate `fs`, the tentative segment length is `round(fs / 2 Hz)`,
  subject to validation against the available interval length.
- All compared trials and reference spectra must use an identical frequency
  grid and estimator configuration.
- Linear PSD values and physical units are retained internally. Absolute dB
  display and normalized dB ratios are derived without replacing the linear
  values used for band integration.
- The approximately 10 s pre-session reference is marked unavailable when a
  full valid interval immediately before the first trial is unavailable. The
  pipeline will not silently substitute a shorter baseline.
- The 60 Hz notch remains configurable and defaults to enabled.
- Gamma summaries exclude 58-62 Hz from integration. Mean band power is
  normalized by the retained frequency bandwidth rather than treating the
  excluded interval as zero power.
- Figures and metadata must show whether the notch and 58-62 Hz exclusion were
  applied.

### 2.8 Band aggregation and synchrony uncertainty

- ITPC and ISPC band/window summaries use the arithmetic mean across valid
  frequency-by-time pixels in the selected frequency band and temporal window.
- Trial-level PLV is calculated across time separately at every retained
  Morlet frequency, then averaged across the selected frequency band.
- A trial/pair/frequency/epoch PLV is valid only when at least two paired phase
  samples and at least 80 percent of the epoch's expected samples are valid.
  This is a phase-data coverage check, not a phase-locking threshold. The valid
  sample count and fraction are saved even when the PLV is unavailable.
- Band-level PPC is the arithmetic mean of frequency-resolved PPC values in the
  selected band.
- Frequency-resolved values remain available; band summaries do not replace
  them.
- ITPC and ISPC maps retain their raw across-trial magnitude estimates and
  effective trial-count arrays.
- Scalar ITPC/ISPC band/window summaries use percentile bootstrap 95 percent
  confidence intervals from 1,000 trial resamples with a saved fixed seed.
- Full frequency-by-time maps do not receive pixelwise bootstrap intervals.
- The first milestone will not add trial-count matching or an alternative
  debiased phase-consistency metric. Counts and the fewer-than-10-trials
  instability warning must remain prominent when conditions are compared.

### 2.9 PPC reliability and significance

- Calculate observed PPC when at least two valid spike-phase samples exist.
- Save and display the exact contributing spike count for every
  unit/condition/filter/epoch/site result.
- Flag PPC based on fewer than 50 valid spike-phase samples as unstable.
- Do not assign permutation significance below 50 valid spike-phase samples.
- Use 1,000 same-condition trial shuffles by default and allow an explicit
  100-shuffle preview configuration.
- Each shuffle must be a derangement: no trial spike train may be paired with
  phase from the same trial.
- Do not calculate permutation significance when fewer than two eligible trials
  exist.
- Use one deterministic permutation schedule, saved or reproducible from the
  fixed seed, across units within a condition/filter/epoch/site job.
- Apply Benjamini-Hochberg FDR across frequencies separately for each
  unit-by-condition-by-epoch-by-LFP-site spectrum.
- The within-trial circular-shift null is architecturally reserved but deferred
  from the first implementation.

### 2.10 Inspection cache layout

- The default cache is a generic directory:

  ```text
  processed/lfp_summary_cache/
      manifest.json
      power.npz
      synchrony.npz
      spike_phase.npz
  ```

- `manifest.json` contains the shared schema, parameters, source fingerprints,
  site definitions, completion state, and component-file identities.
- Component NPZ files contain named arrays and component-specific metadata.
- Independent component files prevent the webapp from loading PPC products for
  a PSD-only view and allow a failed later phase to be diagnosed separately.
- A complete explicit recomputation writes and validates temporary component
  files, atomically replaces each successful component, and atomically replaces
  the manifest last as the transaction commit point.
- The first milestone will not introduce Zarr, HDF5, or another storage
  dependency.

### 2.11 Initial channels and figure defaults

- Initial experimental validation uses session `CT026_2026-08-01_130853` and
  saved-channel indices PFC=5, HPC1=222, and HPC2=14.
- Plotting defaults to light mode with opaque white backgrounds, black axes and
  text, and PNG export.
- PDF and SVG export are deferred from the first milestone.

### 2.12 Frequency grid, references, prevalence, and exemplars

- Wavelet phase analyses use a configurable linear 2 Hz frequency grid from
  2-100 Hz by default. The 2 Hz value is frequency sampling, not a claim that
  the Morlet transform has a fixed 2 Hz spectral bandwidth at every frequency.
- Frequencies from 58-62 Hz are excluded from gamma ITPC, ISPC, PLV, and PPC
  band summaries as well as gamma power summaries.
- The session-median reference is the frequency-wise median of whole-window
  `-2 to +2 s` PSDs across all valid trials, independent of condition and
  optional choice/context filters.
- The same session-median reference normalizes whole, before, and after trial
  spectra so normalization does not remove a session-wide before/after change.
- Significant-locking prevalence uses only units eligible for significance in
  its denominator: at least 50 valid spike-phase samples and at least two
  eligible trials.
- The numerator is the number of eligible units with FDR-adjusted `q < 0.05`.
- Prevalence displays include eligible and total selected unit counts. They
  return unavailable/NaN, not zero, when no units are eligible.
- Default fixed PPC unit ordering uses theta-band PPC from correct-rewarded,
  whole-window trials at the currently selected LFP site.
- The PPC cache saves observed PPC, exceedance count, permutation count,
  uncorrected p-value, FDR q-value, significance flag, null mean, null standard
  deviation, and null 2.5/50/97.5 percentiles. It does not save all 1,000 null
  samples.
- High/low exemplars use the valid observation nearest the requested 5th or
  95th percentile. Ties use earliest trial index for PLV and smallest stable
  unit identifier for PPC. Fewer than two eligible observations makes the
  exemplar unavailable.

### 2.13 Webapp-launched computation

- The Streamlit webapp assembles and validates the analysis configuration from
  its active session, site/channel, unit, condition-filter, and parameter
  controls.
- A deliberate compute/recompute action invokes the reusable numerical
  pipeline from the webapp, even when the run is long.
- The call is synchronous in the first milestone. The webapp must show the
  current stage and progress where the underlying work exposes progress.
- The numerical pipeline remains directly callable from Python for tests and
  future batch work, but a standalone CLI/config-file workflow is not required.
- On success, the webapp validates and loads the newly written cache. On
  failure or interruption, it reports the failed stage and does not replace a
  previously valid cache.

### 2.14 Filter scope and component execution

- The first implementation computes all nine trial conditions for one optional
  choice filter and one optional context filter per component run.
- Choice and context each use one of `all`, `left`, or `right`; their values and
  contributing trial counts are saved in component metadata.
- Changing either filter makes the affected cached summaries incompatible and
  requires recomputation.
- This is a provisional performance choice, not a permanent schema limitation.
- The first CT026 validation run must profile wall time and peak memory for one
  filter configuration, then estimate the cost and storage of all nine
  choice-by-context combinations.
- The Sol orchestrator must present those measurements for review before a
  later task changes the webapp to precompute every filter combination.
- The webapp provides separate compute/recompute actions for Power, Synchrony,
  and Spike-phase components, plus an optional Compute All action.
- A component can be reused only when its shared configuration fingerprint
  matches the manifest. Recomputing one component must not silently combine it
  with incompatible components from an older configuration.

### 2.15 PPC populations and exemplars

- Data contracts support multiple named unit populations and stable cross-probe
  identifiers of the form `probe_label:cluster_id`.
- The first webapp implementation computes PPC for the currently selected unit
  population only. Additional populations require an explicit later run rather
  than being loaded implicitly.
- Unit-population metadata includes probe/source identity, selected channels,
  quality filters, cluster ids, and sorter source fingerprint.
- High/low PPC exemplars are ranked by band-mean PPC.
- Their polar histograms use a fixed representative frequency: 8 Hz for theta
  and 40 Hz for gamma. The displayed frequency is explicit, and the full
  frequency-resolved PPC curve is shown alongside the polar view.

### 2.16 Artifact, amplitude, and exemplar-trace policy

- Automatically exclude objectively invalid inputs: nonfinite traces,
  incomplete requested windows, constant traces, or failed preprocessing.
- Compute and save per-trial QC values including RMS and peak-to-peak amplitude.
- Potential amplitude outliers are flagged for inspection but are not removed
  automatically.
- The webapp accepts an explicit list of user-excluded trial indices before
  computation. The manifest preserves those indices and the user-exclusion
  reason.
- Summary phase analyses use numerical coefficient validity only by default.
- They do not use condition-specific percentile amplitude masks.
- Optional absolute amplitude thresholds can be configured per site and must be
  applied identically across conditions. Effective samples and spikes after
  masking are saved. This describes WP13's eventual scientific behavior. Until
  WP13 is complete, only the empty threshold tuple is supported and every
  nonempty production request is rejected before computation.
- Cache event-aligned source traces decimated to 500 Hz, theta/gamma filtered
  traces, and corresponding Hilbert phase needed for exemplar plots.
- Do not include the complete site-by-frequency-by-trial-by-time wavelet tensor
  in `power.npz`, `synchrony.npz`, `spike_phase.npz`, or their public inspection
  schema. A separate fingerprinted prepared-phase intermediate cache is
  permitted for PPC restartability and repeat computation. It must record its
  generator, analysis version, source/configuration fingerprints, axes, units,
  dtype, and validity representation, and it must be rejected rather than
  partially reused after any mismatch.

### 2.17 Reliable-unit summaries and final transform defaults

- Unit-by-frequency PPC heatmaps show every unit with computable PPC. They also
  expose the spike-count matrix and an unstable mask for entries below 50 valid
  spike-phase samples.
- Condition-by-frequency median PPC maps and band-level median/IQR PPC summaries
  include only entries meeting the 50-spike reliability threshold.
- Significant-prevalence maps use the same reliability threshold plus the
  existing two-eligible-trial requirement.
- Morlet defaults are output sample rate 500 Hz, Gaussian width 1.5, window
  length 1.0, precision 16, and L1 normalization.
- The summary pipeline enables the 60 Hz notch by default with quality factor
  30.
- Wavelet edge padding is `8 * window_length / minimum_frequency_hz`; the
  default 2 Hz minimum therefore loads four seconds of padding on both sides.
- Maximum unpadded continuous processing-block duration is 120 seconds.
- All transform parameters remain configurable and are saved in component
  metadata.

### 2.18 Component state and benchmark review

- Each webapp component reports one of `missing`, `compatible`, `stale`,
  `running`, or `failed`.
- Stale components show their configuration differences and are not rendered
  as current results.
- A user can explicitly open stale results for inspection only with a
  persistent warning.
- Component NPZ and manifest updates are atomic. An interruption preserves the
  previous valid component.
- Intermediate PPC checkpoints are written atomically under a run fingerprint
  and never update component compatibility state. A restarted run may reuse
  only fully validated blocks with matching source, configuration, schedule,
  code/schema, axis, and dtype metadata.
- Compute All runs Power, Synchrony, then Spike phase sequentially.
- Progress reports the current stage, completed/total work units, elapsed time,
  and a projected remaining time when enough completed blocks exist.
- An explicit cancellation system is deferred from the first milestone.
- The choice/context all-filter decision remains a user-reviewed performance
  gate. No arbitrary wall-time cutoff is encoded. The validation report must
  provide observed and projected timing, peak memory, and cache size.

### 2.19 PPC illustrative-trial rule

- Unit PPC, preferred phase, and the representative-frequency polar histogram
  pool spikes across all eligible trials in the selected
  condition/filter/epoch.
- The accompanying LFP-and-spike trace is explicitly an illustrative single
  trial and must not be presented as the source of the pooled PPC statistic.
- Among eligible trials containing at least one valid spike in the selected
  epoch, select the trial whose valid spike count is nearest the median.
- Break equal-distance and equal-count ties by earliest trial index.
- Figure labels and captions distinguish pooled metrics from the illustrative
  trial.

### 2.20 Approved implementation clarifications

- Cross-site phase products use an exact common 500 Hz event-relative grid.
  Independently computed complex Morlet coefficients are interpolated by real
  and imaginary component onto that grid, then normalized to unit phase. Phase
  angles are never interpolated directly. Insufficient continuous support makes
  the affected site/trial or pair result unavailable with an explicit reason.
- Welch PSD is first calculated in linear physical units using the approved
  estimator. When native Welch grids differ across sites, linear PSD is
  interpolated onto the configured canonical 2 Hz cache grid before references,
  normalization, cross-site display, or band integration. The native and
  canonical grid definitions are retained in metadata.
- Gamma band integration treats the line-noise gap as two disjoint retained
  intervals, `[30, 58)` and `(62, 80]` by default, rather than zeroing sampled
  trapezoid weights. Boundary interpolation is performed in linear PSD units,
  and normalization uses the exact retained bandwidth.
- PPC preparation preserves a separate event-relative spike train for every
  trial. If trial windows overlap, a physical spike may belong to each trial
  whose half-open window contains it; the cache records an overlap warning and
  the affected trial indices. Trial shuffles never use the legacy pooled,
  merged-window representation.
- Cache transactions use atomic component-file replacement followed by atomic
  manifest replacement. The manifest is the commit point. Compute All may
  leave earlier successfully recomputed compatible components current when a
  later component fails; the failed or older component is marked failed or
  stale and is not presented as compatible. Portable atomic replacement of a
  populated directory is not required.
- `power.npz` includes decimated event-aligned source traces needed for the
  cached single-trial power view. The optional single-trial spectrogram remains
  in the existing bounded interactive view and is regression-tested as an
  existing route rather than duplicated in the summary cache.

### 2.21 Approved PPC execution design

- Preserve the existing scientific estimator: unweighted frequency-resolved
  PPC, exact linear interpolation of complex phase on the common 500 Hz grid,
  same-condition trial derangements, the 50-spike inference threshold, and
  frequency-wise BH FDR. Do not substitute Buzcode's nearest-sample lookup,
  amplitude-weighted resultant magnitude, Rayleigh test, or temporal-jitter
  null without a separately approved scientific change.
- Follow the standard spike-LFP processing pattern of sampling a shared phase
  representation once and reducing it immediately. For every required
  source-trial-to-target-trial pairing, retain only the complex unit-vector sum
  and valid spike count by unit and frequency. These are sufficient to recover
  exact PPC as `(|sum|^2 - count) / (count * (count - 1))` when count is at
  least two. Do not retain or concatenate all interpolated spike-phase vectors
  for permutation work.
- Invert the production loop order from unit-first execution to bounded blocks
  organized by condition, site, epoch, source trial, and target trial. Sample
  concatenated spikes for multiple units with vectorized frequency operations,
  then use deterministic segmented reductions to restore the unit axis.
- Generate the deterministic derangement schedule before phase sampling. For a
  preview, compute only unique source-target edges used by that schedule when
  this requires less work than the complete off-diagonal pair table. For a
  larger run, the implementation may compute the complete pair table when its
  estimated cost is lower. Both paths must return identical results for the
  same schedule and seed.
- Observed PPC, preferred phase, resultant length, and representative
  histograms are computed once from same-trial samples. The permutation kernel
  consumes only vector sums and counts; it must not invoke the generic phase
  histogram, occupancy, firing-rate, amplitude, or result-object paths for
  every shuffle.
- Exact-sample classification uses equality to the canonical stored time grid
  for both PPC and representative histograms. A merely nearby spike, including
  one within `1e-12`, remains a between-sample case and requires both neighbors.
  This deliberately corrects the legacy histogram helper's looser `isclose`
  behavior so displayed counts describe the same accepted samples as PPC.
- Preserve observed PPC for entries with at least two valid phases, but skip
  all permutation phase sampling for entries that cannot be inferentially
  eligible because they have fewer than 50 valid spikes or fewer than two
  spike-contributing trials.
- Null draws are reduced in bounded shuffle and unit blocks. Full-session null
  draws are never retained. Exact per-spectrum percentile calculation may keep
  only the currently processed bounded block of draws.
- Parallel execution is permitted only after the serial sufficient-statistic
  implementation is correct and profiled. Workers share read-only prepared
  phase data, use bounded memory, and must not duplicate the full phase tensor
  per process. Results must be invariant to worker count and all configured
  block sizes.
- A fingerprinted prepared-phase cache and fingerprinted in-progress PPC block
  checkpoints are approved under the session's processed analysis directory.
  They are intermediate computation artifacts, never compatible final
  components and never entries rendered by the webapp as scientific results.
- Progress must expose preparation, observed reduction, trial-pair reduction,
  shuffle aggregation, FDR, checkpoint, and commit stages with completed/total
  work, elapsed time, and an estimated remaining time when enough observations
  exist.

### 2.22 WP5C-0 execution contract proposed for freeze

This subsection records the exact proposed WP5C execution interface. It becomes
binding only after the user approves the WP5C-0 gate. Approval of the scientific
design or of the WP5C-before-preview ordering does not authorize tests or source
changes.

Scientific PPC settings remain in `PPCAnalysisConfig`: epochs, minimum spike
counts, shuffle count, FDR alpha/method, phase-bin edges, and seed. Replace the
ambiguous execution fields `worker_count` and `chunk_size` with one frozen
`PPCExecutionConfig` owned by `LFPSummaryConfig`:

- `unit_block_size: int = 8`
- `shuffle_block_size: int = 25`
- `trial_edge_block_size: int = 64`
- `worker_count: int = 1`
- `prepared_phase_cache_enabled: bool = True`
- `checkpoint_enabled: bool = True`
- `checkpoint_retention: str = "incomplete_only"`
- `progress_update_interval: int = 1`, measured in completed blocks

`checkpoint_retention` accepts only `"incomplete_only"` and `"retain"`. The
default retains interrupted work for restart but deletes the exact completed
run-fingerprint directory only after the final component and manifest commit.
Production starts with one worker. Block-size defaults are conservative
execution choices and may change after profiling without changing scientific
component compatibility. Every execution field is validated and serialized in
work metadata, but none enters the final scientific component fingerprint.

Preserve these existing public entry points and signatures:

- `generate_trial_derangement_schedule(trial_count, shuffle_count, *, seed)`
- `compute_trial_shuffle_ppc(*, trial_relative_spike_times_s, phase_time_s,
  trial_phase_vectors, frequencies_hz, schedule,
  overlap_trial_indices=None)` as the WP5B equivalence reference
- `summarize_permutation_null(*, observed_ppc, null_ppc_chunks, spike_count,
  eligible_trial_count)` as the null-summary equivalence reference
- `adjust_ppc_pvalues_bh(*, p_value, null_eligible)`
- `build_spike_phase_payload(config, prepared_phase, prepared_spikes)`
- `compute_spike_phase_component(config, dependencies,
  progress_callback=None, *, prepared_phase=None)`

Add only these numerical execution interfaces to `spike_lfp_summary.py`:

- `compute_edge_sufficient_statistics(*, trial_relative_spike_times_s,
  phase_time_s, trial_phase_vectors, frequencies_hz,
  source_trial_position, target_trial_position) -> EdgeSufficientStatistics`
- `reduce_scheduled_shuffle_block(*, edge_statistics, schedule,
  shuffle_positions) -> np.ndarray`

The first function receives nested unit/trial spike arrays and a bounded stable
edge list. The second returns only the current temporary
`(shuffle_block, unit, frequency)` dimensionless PPC draws. It must be consumed
immediately by bounded null aggregation. `EdgeSufficientStatistics` is a frozen
dataclass containing only `source_trial_position`, `target_trial_position`,
`phase_vector_sum`, and `valid_spike_count` with the contracts below.

Add work-cache interfaces in `lfp_summary_work_cache.py`:

- `write_prepared_phase_cache(cache_root, metadata, phase, valid, axes) -> Path`
- `load_prepared_phase_cache(cache_root, expected_metadata) -> PreparedPhaseCache | None`
- `write_ppc_checkpoint(run_directory, block_id, arrays, metadata) -> Path`
- `load_valid_ppc_checkpoints(run_directory, expected_metadata) -> tuple[PPCCheckpoint, ...]`
- `cleanup_ppc_run(run_directory, expected_run_fingerprint) -> None`

Add restart orchestration in `lfp_summary_ppc_runtime.py`:

- `execute_ppc_blocks(*, config, execution, prepared_phase, prepared_spikes,
  schedule, work_root, progress_callback=None) -> PPCExecutionResult`

These are project-internal interfaces. Their full docstrings must specify types,
axes, units, missingness, return values, and failure behavior before their
implementations are accepted.

### 2.23 Prepared-phase and edge schemas

The prepared-phase work representation is:

- `phase`: complex64 shape `(site, frequency, trial, time)`.
- `valid`: Boolean with exactly the same shape and axes as `phase`.
- `relative_time_s`: float64 shape `(time,)`, the exact 500 Hz half-open grid
  `[-2.0, 2.0)` for the current approved window.
- Stable fixed-width Unicode `site_ids` and numeric `frequency_hz` and
  `trial_indices`, each in the same order used by the final component.
- Separate site/trial validity and fixed-width Unicode exclusion-reason arrays;
  invalid values are never inferred only from a collapsed global mask.
- `phase.npy` and `valid.npy` are opened read-only with memory mapping when
  shared with workers. Downstream sums use complex128 and counts/accumulators
  use int64 or float64 as appropriate.

Trial-edge arrays are numeric and stable:

- `source_trial_position`: int64 shape `(edge,)`.
- `target_trial_position`: int64 shape `(edge,)`.
- Null edges never contain a source equal to its target.
- Edges are ordered lexicographically by source position, then target position.
- The schedule is int64 shape `(shuffle, source_trial_position)` containing
  target positions. It is generated exactly once from the saved scientific
  seed and every row is a derangement.

For every bounded edge/unit block:

- `phase_vector_sum`: complex128 shape `(edge, unit, frequency)`.
- `valid_spike_count`: int64 with identical axes.
- No per-spike shuffled phase vectors survive the reduction.
- PPC is `NaN` for count zero or one and otherwise equals
  `(|phase_vector_sum|**2 - valid_spike_count) /
  (valid_spike_count * (valid_spike_count - 1))`.

The phase validity rule remains the WP5B rule: interpolate real and imaginary
components only between adjacent finite, nonzero complex samples; exact samples
use the exact valid sample; outside support or across an invalid neighbor is
invalid. Overlapping trial membership is preserved.

### 2.24 Frozen null-summary numerics

- Permutation p values use the plus-one formula already implemented by WP5B.
- Null standard deviation is the population standard deviation, `ddof=0`.
- Exact 2.5, 50, and 97.5 percentiles use
  `numpy.percentile(..., method="linear")`; the method must be explicit rather
  than inherited from a NumPy default.
- Exact percentiles may retain only the current bounded unit block's shuffle
  draws and must not retain full-session null draws.
- BH correction runs along frequency only, separately for every
  unit/condition/site/epoch spectrum.
- Ineligible entries retain observed PPC when computable, never invoke
  permutation phase sampling, keep p/q values unavailable, and never contribute
  to the prevalence denominator.

### 2.25 Intermediate layout and transaction rules

Use this exact session-local layout, separate from the final inspection cache:

```text
processed/lfp_summary_work/
    prepared_phase/<phase_fingerprint>/
        metadata.json
        phase.npy
        valid.npy
        axes.npz
        complete.json
    ppc/<run_fingerprint>/
        metadata.json
        schedule.npz
        blocks/
            <block_id>.npz
            <block_id>.complete.json
        complete.json
```

- Temporary files use `.<final-name>.tmp-<uuid>` in the same directory as the
  final path. Data/metadata are validated before atomic `os.replace`; the
  completion marker is written last and is itself atomically replaced.
- Every NPZ load uses `allow_pickle=False`. NPY arrays must have the exact
  approved dtype, shape, and axes before memory mapping or reuse.
- Metadata includes generator function, analysis/schema version, source and
  scientific fingerprints, representation fingerprint, identities, axes,
  shapes, dtypes, units, schedule seed, execution settings, and completed block
  identities. Missing or mismatched fields reject reuse.
- A writer obtains an exact-directory `writer.lock` by exclusive creation.
  Another writer for that fingerprint fails clearly; it must not share partial
  files or silently remove the lock. Stale locks and incomplete artifacts are
  ignored for reuse until an explicit exact-fingerprint resume/cleanup action.
- Restart selects only the exact `run_fingerprint`. A block resumes only when
  both its NPZ and completion marker validate. Orphan temporary files and
  unmarked blocks are incomplete and are never loaded.
- Atomic write failure cannot create a valid completion marker. Work artifacts
  never appear in `manifest.json` and can never satisfy final component
  compatibility checks.
- Cleanup receives and verifies the exact run fingerprint stored in metadata;
  it cannot remove a sibling prepared cache or PPC run. With the default
  `"incomplete_only"` policy, completed block checkpoints are deleted only
  after `spike_phase.npz` and the final manifest have committed successfully.

### 2.26 Progress contract

WP5C progress events use these ordered stages: `prepare_phase`,
`observed_reduction`, `trial_edge_reduction`, `shuffle_aggregation`, `fdr`,
`checkpoint`, and `commit`. Each event carries component, stage, completed and
total work units, a concise message, elapsed seconds, and optional ETA seconds.
Completed and total counts never decrease within a stage. ETA is absent until
at least two completed timed blocks exist. Resumed blocks count as completed
only after validation, and final commit remains the only event that makes the
scientific component compatible.

### 2.27 Approved post-speedup integration decisions

The following decisions were approved on 2026-09-18 before any post-speedup
source implementation began. The first sequencing bullet is historical; the
2026-09-20 R1/C1 contracts above supersede its remaining-work order after the
real preview reached component completion and failed in reporting:

- The implementation sequence is WP10 composed production dependencies, WP12
  PPC plotting/reporting, the standalone WP5C-6 launcher, and then the
  user-invoked 100-shuffle preview. WP11 follows the preview infrastructure; it
  is not a prerequisite for the standalone preview.
- The webapp supports ProbeA and ProbeB, but exactly one probe-qualified unit
  population is active in a component run. A combined cross-probe population
  is out of scope. Both choices use the same defaults: good or MUA units on
  channels labeled good and inside brain. Stable unit identifiers remain
  `probe_label:cluster_id`, and the selected sorter path, aligned-spike path,
  channels, quality rules, and unit ids are explicit fingerprinted inputs.
- WP13 is deferred. An empty `absolute_amplitude_thresholds` tuple is the only
  supported production configuration until WP13 is complete. Any nonempty
  value must fail validation at the production boundary before phase
  preparation, checkpoint creation, or final-component writes. A warning that
  continues computation is insufficient because it would knowingly publish a
  result that did not apply the requested mask.
- The launcher creates and records its timestamped analysis-run directory and
  exact resume command before full PPC planning begins. Planning remains
  deterministic but is not checkpointed. If interrupted during planning,
  resume may repeat planning from the beginning; once compatible phase or PPC
  checkpoints exist, resume reuses them under their existing exact identities.
- Default `incomplete_only` PPC cleanup must not run before the launcher has
  successfully written and validated the scientific component, manifest,
  plots, detailed report, run log, and summary. The launcher may use retained
  checkpoints during computation and then invoke exact-fingerprint cleanup as
  its final successful step. Report failure leaves compatible resumable work
  and must not delete a sibling run.
- The preview uses exactly 100 shuffles and a CT026-specific request for eight
  workers, subject to planned active-worker and allocation preflight. The
  portable `PPCExecutionConfig.worker_count` default remains one. A 1,000-
  shuffle run requires a different timestamped run, a different scientific
  fingerprint, an explicit final-run flag, and separate user approval after
  preview inspection.

## 3. Target architecture

The implementation uses the following module boundaries. Implementing agents
must not move numerical logic into Streamlit modules.

### 3.1 Configuration and contracts

`src/neural_analysis/lfp_summary_models.py` defines validated frozen dataclasses
and serialization helpers for:

- Session and output paths.
- Selected LFP sites and site pairs.
- Alignment event, analysis windows, and trial conditions.
- PSD, wavelet, notch/exclusion, and band definitions.
- PPC permutation and reliability settings.
- Analysis schema version and random seed.

Every array contract must document shape, axis order, time coordinate, units,
and missing-value behavior.

### 3.2 Shared trial and LFP preparation

`src/neural_analysis/lfp_summary_preparation.py` provides shared functions that:

- Build trial masks using the existing project condition semantics rather than
  duplicating them.
- Select finite alignment events and record excluded trial indices and reasons.
- Load continuous padded LFP blocks through the existing SpikeGLX/Open Ephys
  routing.
- Apply only approved preprocessing.
- Produce common event-relative grids where cross-site phase comparisons need
  matched samples.
- Retain source-rate traces for exemplar and quality-control plots.

### 3.3 Analysis modules

Numerical work is separated into:

- `lfp_power_summary.py`: Welch PSD, reference spectra, normalization, and
  trial-level band power.
- `lfp_synchrony_summary.py`: ITPC, ISPC, preferred inter-site phase offset,
  trial PLV, band/window summaries, and bootstrap intervals.
- `spike_lfp_summary.py`: observed PPC, preferred spike phase, phase histograms,
  same-condition trial-shuffle nulls, p/q values, and population summaries.

Numerical functions must not import Streamlit or depend on webapp state.

### 3.4 Cache writer and loader

`src/neural_analysis/lfp_summary_io.py` owns the inspection-cache manifest,
component validation, compatibility fingerprints, atomic NPZ/JSON writes, and
stale-configuration diffs. The inspection cache contains named numerical arrays
and structured metadata. It preserves at least:

- Generator and analysis schema version.
- Source paths and source-file fingerprints.
- Session id and processing timestamp.
- Full parameter configuration and random seed.
- Site and site-pair identities.
- Trial indices, condition membership, split membership, and exclusions.
- Frequencies, time grids, windows, bands, axis conventions, and units.
- Effective trial, unit, spike, and sample counts for each result.
- Numerical results needed to reproduce every summary plot without reopening
  the raw recording.

Large reusable continuous arrays should not be duplicated for every condition.
The first-milestone storage format is the JSON manifest plus numeric/boolean/
fixed-width-Unicode NPZ component files defined in Section 2.10.

### 3.5 Pipeline and progress boundary

`src/neural_analysis/lfp_summary_pipeline.py` exposes directly callable
`compute_power_component`, `compute_synchrony_component`,
`compute_spike_phase_component`, and `compute_all_components` functions. Each
accepts validated configuration plus a framework-independent progress callback.
The pipeline coordinates existing loaders and numerical modules but contains no
plotting or Streamlit controls.

### 3.6 Plotting and webapp result views

`src/neural_analysis/lfp_summary_plotting.py` creates Matplotlib summary and
exemplar figures from validated cached arrays. It does not load raw data.

`src/neural_analysis/lfp_summary_webapp.py` owns Streamlit controls, component
status displays, compute buttons, progress presentation, cache loading, and
result selectors. `psth_webapp.py` only adds the route/navigation entry and
passes the active session/probe inputs into this view.

The webapp will load and validate the cache, then render separate views for:

- Condition PSDs and band-power summaries.
- ITPC/ISPC maps and band/window summaries.
- Trial-level PLV distributions and exemplar trials.
- Unit-by-frequency PPC maps.
- Condition-by-frequency median PPC and significant-fraction maps.
- Band-level PPC summaries and exemplar units.

The webapp should remain a visualization and inspection layer. It should not
contain the core numerical implementations.

## 4. Test-first implementation protocol

Each implementation phase must follow RED, GREEN, REFACTOR:

1. Add focused pytest tests and commit them before implementation.
2. Run the focused tests and record the expected failures.
3. Implement the smallest clear numerical or integration change.
4. Run focused and relevant regression tests until green.
5. Refactor without changing the approved data contracts.

The exact test cases are listed per work package in Section 12. Their shared
test categories include:

- Synthetic sinusoid/noise PSD recovery and normalization.
- Band integration on irregular or excluded frequency grids.
- Known-phase synthetic ITPC, ISPC, PLV, and phase-offset recovery.
- PPC behavior for fixed, uniform, and low-count spike phases.
- Trial-shuffle preservation and destruction properties.
- Multiple-comparison correction and deterministic random seeds.
- Site/pair identity, axis order, units, and cache round trips.
- Stale-cache rejection and atomic overwrite behavior.
- Webapp selection and plotting from saved synthetic results.
- Regression coverage for existing single-trial LFP and phase views.

## 5. Performance constraints

- Continuous wavelet transforms must be computed once per site and reusable
  processing block, not once per condition, unit, or permutation.
- Trial condition and choice/context selections should index shared trial-level
  results rather than trigger repeated raw-LFP loading.
- PPC null computation must use the sufficient-statistic and adaptive
  scheduled-edge design in Section 2.21. Its memory must be proportional to the
  configured unit/shuffle block sizes, not to every retained phase sample or to
  all unit-by-condition-by-site-by-epoch null draws.
- Exact phase interpolation uses direct indices and weights on the validated
  uniform 500 Hz grid and vectorizes across frequency. It must preserve the
  existing adjacent-valid-sample rule and normalize the interpolated complex
  value to unit magnitude; nearest-neighbor substitution is not an optimization
  permitted by this plan.
- Trial-shuffle work is chunked for bounded memory. Activate within-session
  workers only after serial equivalence tests pass and profiling demonstrates
  benefit. Benchmark 1, 2, 4, and 8 workers; do not assume linear scaling to all
  logical CPUs. Prefer shared-memory threads or another design that demonstrably
  avoids duplicating the prepared phase tensor.
- A cold CT026 run includes phase preparation; a warm run may reuse the
  validated prepared-phase cache. Initial engineering targets, not correctness
  cutoffs, are at most 60 minutes for the cold 100-shuffle preview, at most two
  hours for the cold 1,000-shuffle run, and less than 16 GiB peak PPC working
  memory. Record the phase preparation, observed reduction, trial-pair
  reduction, shuffle aggregation, FDR, and write times separately.
- Before either complete CT026 shuffle run, benchmark the largest 249-trial
  condition with blocks containing low-, median-, and high-rate units. Use the
  measured phase-sample throughput and shuffle-reduction throughput to project
  the complete run. Do not start the full run if the projection is materially
  above the target without reporting the evidence for review.
- Cache contents should favor arrays with explicit axes over deeply nested
  Python objects.
- The plan must specify a bounded preview configuration separately from final
  permutation settings if both are required.

## 6. Agent-sized work packages

Section 12 divides work into bounded packages suitable for Terra medium
implementers. Each package states files in scope, tests to write first, a RED
command, implementation constraints, and completion evidence.

A Sol medium/high orchestrator will:

- Approve contracts before parallel work begins.
- Prevent two agents from editing overlapping integration files concurrently.
- Review scientific definitions and axis/unit preservation.
- Run integration and regression tests after each merge point.
- Inspect the first synthetic and experimental-session outputs.
- Reject implementations that move numerical logic into Streamlit callbacks or
  recompute transforms per condition/permutation.

## 7. Decision and authorization status

The scientific estimator, WP5C-before-preview sequence, and internal execution
contracts in Sections 2.22-2.26 are approved. WP5C-1 through WP5C-5 and the
complete S0-S8 exact-speedup plan are complete. Eight workers are preferred for
the CT026 production configuration, without changing the portable library
default or bypassing preflight.
The post-speedup decisions in Section 2.27 are approved documentation contracts.
On 2026-09-18 the user approved the WP10 test-first implementation plan and
separate test/implementation commits, conditional on recording the complete
implementation contract in this document and committing that documentation
before any test or source edit. That documentation-first gate does not authorize
a CT026 scientific run.
Implementers must escalate newly discovered ambiguity instead of choosing new
scientific or execution defaults.
Material changes to metrics, thresholds, cache contracts, or user-visible
behavior require user approval and an update to this plan before implementation
continues. The work-only benchmark did not authorize or produce a scientific
Spike-phase component. The 100-shuffle preview and 1,000-shuffle final result
remain separate, explicitly launched runs.

## 8. Exact shared data semantics

### 8.1 Trial axes and behavioral values

- Cache `trial_indices` are integer row positions into the loaded session trial
  table and remain stable across component arrays.
- `action == 0` means right choice and `action == 1` means left choice.
- `state_int == 0` means right context and `state_int == 1` means left context.
- Choice/context values outside `{0, 1}` or missing values do not pass the
  corresponding left/right filter; they remain eligible when that filter is
  `all` if the underlying condition mask otherwise permits them.
- `condition_membership` has shape `(trial, condition)` and records all nine
  overlapping condition masks. A trial can have multiple `True` entries.
- `filter_membership` has shape `(trial,)` and records the current optional
  choice/context filter independently of condition membership.
- User exclusions and objective invalidity are separate masks with separate
  reason codes.

### 8.2 Epoch bounds

All event-relative windows use seconds and half-open bounds:

- `whole`: `[-2.0, 2.0)`
- `before`: `[-2.0, 0.0)`
- `after`: `[0.0, 2.0)`

The sample at exactly zero belongs only to `after`. The `whole` result is
computed directly over the whole interval; it is not an average of separately
calculated before/after statistics.

### 8.3 Site and pair validity

- Power and ITPC use site-specific valid trials.
- ISPC and PLV use the pairwise intersection of valid trials/samples for the
  selected pair.
- For each PLV cell, the expected sample count is the configured epoch duration
  multiplied by the configured sampling rate. Paired-valid samples are time
  points where both sites have a finite phase value after all configured masks.
  PLV requires at least two paired-valid samples and paired-valid coverage of at
  least 0.80. Otherwise PLV is `NaN` with an insufficient-phase-coverage reason.
- A missing HPC2 trial must not remove that trial from PFC-HPC1 calculations.
- Effective trial/sample/spike counts are saved at the same granularity as the
  metric whenever validity can differ across output cells.
- Site pairs are generated from configured stable site ids and stored as
  ordered `(site_a_id, site_b_id)` identities. Phase offset follows
  `phase_a - phase_b` everywhere.

### 8.4 Numerical missingness

- Unavailable or undefined results are `NaN`, never numeric zero.
- Zero or nonpositive reference power makes the associated normalized value
  unavailable; the implementation must not add an undocumented epsilon.
- PPC with fewer than two valid phase samples is `NaN`.
- Significance fields are NaN/false-with-ineligible-status when reliability
  criteria are not met. Ineligibility must not be represented as a failed null.

## 9. Configuration, dependencies, and fingerprints

### 9.1 Configuration dataclasses

`lfp_summary_models.py` defines clear frozen dataclasses, without a deep
inheritance hierarchy:

- `LFPSiteConfig`: stable id, label, acquisition format, LFP path, aligned-sync
  path when needed, probe label, saved-channel index, and source voltage unit.
- `UnitPopulationConfig`: stable label, probe label, sorter/aligned-spike paths,
  selected channels, quality settings, and stable unit ids.
- `TrialFilterConfig`: choice filter, context filter, and explicit user-excluded
  trial indices.
- `AnalysisWindowConfig`: alignment event and whole/before/after bounds.
- `FrequencyBandConfig`: band name, inclusive outer bounds, and excluded sampled
  frequency intervals.
- `PowerAnalysisConfig`: Welch settings, notch settings, bands, and baseline
  settings.
- `PhaseAnalysisConfig`: frequency grid, Morlet settings, output rate, numerical
  and optional absolute amplitude thresholds, block duration, bands, bootstrap
  count, seed, and prepared-phase intermediate-cache policy.
- `PPCAnalysisConfig`: epochs, minimum computable/reliable spikes, shuffle
  counts, FDR alpha/method, phase-bin edges, worker/unit/shuffle block settings,
  checkpoint policy, and seed.
- `LFPSummaryConfig`: session identity/paths, output directory, sites, optional
  unit population, filters, component configs, and schema version.
- `ProgressEvent`: component, stage, completed count, total count, concise
  message, elapsed seconds, and optional ETA seconds.

Validation rejects duplicate site ids, duplicate unit ids, invalid bands,
overlapping/incorrect epochs, unsupported alignments, invalid channels, and
nonfinite parameters before raw files are opened.

Worker counts, block sizes, progress cadence, checkpoint location, and
prepared-phase cache enablement are execution settings and do not change the
scientific numerical result fingerprint. Shuffle count, seed, interpolation
method, phase validity rules, and every scientific threshold remain
fingerprinted. Intermediate cache/checkpoint fingerprints additionally include
all execution-representation details needed to reject incompatible blocks.

### 9.2 Dependencies

- Introduce no new runtime dependencies.
- Use NumPy and Pandas for contracts and table/index operations.
- Use NumPy array files or NPZ metadata plus standard-library atomic file
  replacement for prepared-phase and PPC work artifacts. A memory-mappable
  representation is permitted for the approximately 1 GiB prepared tensor so
  workers can share it without copying. Do not introduce Zarr or HDF5 for this
  optimization.
- Use `scipy.signal.welch` for PSD and the existing SciPy/Pynapple filtering and
  wavelet paths already used by the package.
- Use `scipy.stats.false_discovery_control(..., method="bh")` for BH correction;
  the current environment exposes this API. The implementing agent must still
  inspect the installed package source or official documentation before first
  use and add a focused contract test.
- Use Matplotlib for figures and Streamlit only in `lfp_summary_webapp.py`.
- Use standard-library JSON, hashing, temporary-path, and atomic replace tools
  for manifests and writes.
- Run every Python command through `uv run`.

### 9.3 Compatibility fingerprints

- Serialize configuration to canonical JSON with sorted keys and stable
  primitive types, then compute a SHA-256 configuration fingerprint.
- Source fingerprints use resolved path, file size, and nanosecond modification
  time for large binaries, plus the same information for required metadata or
  sync sidecars. They are change detectors, not cryptographic full-file hashes.
- Behavior CSVs, small JSON metadata, and small configuration files may also use
  full SHA-256 content hashes.
- Each component records exactly which configuration subset and source
  fingerprints affect it. Changing PPC shuffle count should stale Spike phase,
  not Power.

## 10. Component computation and cache contracts

All NPZ files contain only numeric, boolean, or fixed-width Unicode arrays and
must load with `allow_pickle=False`. Axis names and physical units live in the
JSON manifest and are validated against array shapes.

### 10.1 Shared prepared data

Preparation returns in-memory structures with:

- Selected trial indices, alignment times in UTC Unix seconds, condition and
  filter membership, and exclusion reason codes.
- Per-site source sample rate and voltage unit.
- Event-aligned traces or continuous block-loader callbacks as required by the
  component.
- Stable sites, site pairs, units, and relative spike times.

Prepared event traces use shape `(site, trial, time)` and a documented exact
500 Hz event-relative grid when cross-site plotting requires stacking. Complex
Morlet coefficients are interpolated by real and imaginary component before
unit-phase normalization. PSD may operate at the native LFP sample rate, but
linear PSD is interpolated to the configured canonical 2 Hz cache grid when
native Welch grids differ.

### 10.2 Power component

Welch defaults are explicit:

- Periodic Hann window.
- Constant detrending.
- `nperseg = round(sample_rate_hz / 2.0)` for approximately 2 Hz bins.
- `noverlap = floor(nperseg / 2)`.
- One-sided density scaling in source-unit-squared/Hz.
- No silent zero padding to claim finer spectral resolution.

`power.npz` contains at least:

- `trial_indices`: `(trial,)`.
- `site_ids`: `(site,)`.
- `condition_names`: `(condition,)`.
- `condition_membership`: `(trial, condition)`.
- `frequency_hz`: `(frequency,)`.
- `epoch_names`: `(epoch,)` in whole/before/after order.
- `psd_linear`: `(site, trial, epoch, frequency)`.
- `psd_valid`: `(site, trial, epoch)`.
- `session_reference_psd_linear`: `(site, frequency)`.
- `presession_reference_psd_linear`: `(site, frequency)`, all NaN where
  unavailable.
- `presession_reference_available`: `(site,)`.
- `normalized_psd_session_db`: `(site, trial, epoch, frequency)`.
- `normalized_psd_presession_db`: same shape.
- `band_names`: `(band,)` in theta/gamma order.
- `band_power_linear`: `(site, trial, epoch, band)`.
- `band_power_session_db` and `band_power_presession_db`: same shape.
- `trial_rms` and `trial_peak_to_peak`: `(site, trial)` in each site's source
  voltage unit.
- `relative_time_s`: `(time,)` on the exact 500 Hz half-open event grid.
- `source_trace`: `(site, trial, time)` in each site's source voltage unit,
  decimated for the cached single-trial trace view.
- Objective-validity, user-exclusion, and instability flags/counts needed by
  plots.

Band integration uses trapezoidal quadrature on each disjoint retained frequency
interval, with boundary values interpolated in linear PSD units where needed.
For default gamma this integrates `[30, 58)` and `(62, 80]` separately so no
trapezoid bridges the excluded interval. Divide by the exact sum of retained
interval bandwidths to produce mean band power.

The session reference is computed from whole-window PSDs over all valid,
non-user-excluded trials before applying the selected condition or choice/context
filter. The pre-session interval is exactly `[first_start_time - 10 s,
first_start_time)`. Availability is recorded separately per site; a missing
baseline at one site does not invalidate another site's reference.

### 10.3 Synchrony component

Wavelet tensors are temporary computation products. They use complex64 storage
where safe, float64 accumulation for circular sums, and are processed in bounded
continuous blocks.

`synchrony.npz` contains at least:

- Shared trial/site/condition/frequency/epoch/band identities.
- `relative_time_s`: `(time,)` at 500 Hz over `[-2, 2)`.
- `pair_site_a_ids` and `pair_site_b_ids`: `(pair,)`.
- `itpc`: `(condition, site, frequency, time)`.
- `itpc_effective_trial_count`: same shape.
- `ispc`: `(condition, pair, frequency, time)`.
- `ispc_phase_offset_rad`: same shape, following site A minus site B.
- `ispc_effective_trial_count`: same shape.
- `itpc_band_mean`, `itpc_ci_low`, `itpc_ci_high`:
  `(condition, site, epoch, band)`.
- `ispc_band_mean`, `ispc_ci_low`, `ispc_ci_high`:
  `(condition, pair, epoch, band)`.
- `plv_by_frequency`: `(trial, pair, epoch, frequency)`.
- `plv_phase_offset_rad`: same shape.
- `plv_valid_sample_count`: same shape.
- `plv_valid_sample_fraction`: same shape, relative to the expected sample
  count for that epoch.
- `plv_computable`: same shape; false when fewer than two paired-valid samples
  or paired-valid coverage is below 0.80.
- `plv_band_mean`: `(trial, pair, epoch, band)`.
- Cached `source_trace`, `band_filtered_trace`, and `hilbert_phase_rad` with
  explicit site/trial/band/time axes for exemplar plots.
- Site-specific and pair-specific validity masks and exclusion counts.

ITPC/ISPC bootstrap resamples selected trial positions with replacement within
one condition and recomputes only the scalar band/window statistic. It uses
1,000 resamples, percentile 2.5/97.5 limits, and the saved seed. With very small
trial groups it still returns the requested estimate/interval but attaches the
under-10 instability flag and count.

### 10.4 Spike-phase component

Observed PPC pools valid spike phases across the selected trials. The
same-condition null maps each source trial's event-relative spike times onto a
different trial's phase trace using a true derangement. Shuffles are generated
once per condition/filter/epoch/site and reused across units.

Production execution uses exact pair sufficient statistics. A required
source-target trial edge produces `phase_vector_sum` as complex128 and
`valid_spike_count` as int64 on explicit unit-by-frequency axes. Shuffle PPC is
calculated directly after summing the scheduled edges. Individual shuffled
spike-phase vectors, artificial spike timestamps, phase histograms, occupancy,
firing rates, and amplitude arrays are not constructed. The implementation may
choose scheduled unique edges or the complete off-diagonal edge table according
to the smaller measured/projected workload, without changing the schedule or
result.

`spike_phase.npz` contains at least:

- Stable unit, population, site, condition, epoch, band, frequency, and phase
  bin identities.
- `ppc`, `resultant_length`, `preferred_phase_rad`, and `spike_count` with shape
  `(unit, condition, site, epoch, frequency)`.
- `computable`, `reliable`, and null-eligibility masks with the same shape.
- `null_exceedance_count`, `permutation_count`, `p_value`, `q_value`,
  `significant`, `null_mean`, `null_std`, `null_p025`, `null_p50`, and
  `null_p975` with the same shape.
- `ppc_band_mean`: `(unit, condition, site, epoch, band)`.
- `representative_phase_hist_count`:
  `(unit, condition, site, epoch, band, phase_bin)` for 8 and 40 Hz.
- `phase_bin_edges_rad`: `(phase_bin + 1,)`.
- Relative per-unit/per-trial spike times stored as one numeric concatenated
  array plus integer offsets, not an object/ragged array.
- Cached source/band/Hilbert traces needed to keep Spike-phase exemplars
  independent of whether Synchrony was computed.
- Deterministically selected high/low unit ids and illustrative trial indices,
  or enough saved counts to verify and reproduce their selection.

Apply BH FDR independently along the frequency axis for each eligible
unit/condition/site/epoch spectrum. Population median/IQR arrays may be derived
on load from reliable unit entries; significant prevalence uses reliable,
null-eligible units only.

With optional amplitude masking, a unit-level band PPC is reliable only when
every retained frequency contributing to that band has at least 50 valid phase
samples. Otherwise the band value may be computed from available frequencies
for inspection but is excluded from reliable population summaries.

### 10.5 Manifest and atomic updates

`manifest.json` includes schema version, session id, source/reference statement,
processing timestamp, code/git version when available, shared configuration,
site/unit/filter definitions, and one component entry per NPZ. A component
entry includes file name, configuration fingerprint, source fingerprints,
array schema, units, axes, completion time, warnings, runtime, peak memory, and
last successful status.

Component writes follow:

1. Write a uniquely named temporary NPZ in the cache parent.
2. Load it with `allow_pickle=False` and validate every required array.
3. Atomically replace the component NPZ.
4. Write and validate a temporary manifest.
5. Atomically replace `manifest.json`.

A failed attempt records a concise error for the current webapp session/log but
does not replace the last valid component. Compute All may share in-memory
prepared phase data, but each successful component remains independently
validated.

Intermediate prepared-phase and PPC restart artifacts follow a separate
transaction boundary:

1. Store them beneath a source/configuration/run-fingerprinted work directory
   in the session's processed analysis directory, never in the repository.
2. Save named arrays and metadata describing the generating class/function,
   analysis version, source and configuration fingerprints, schedule seed,
   axes, shapes, units, dtypes, and completed block identities.
3. Write and validate each work block atomically before marking it complete.
4. On restart, reject the entire prepared cache or individual work block when
   any required fingerprint, schema, axis, shape, dtype, or completion marker
   differs.
5. Never expose a work artifact through `manifest.json` as a compatible
   scientific component. Only the validated final `spike_phase.npz` commit can
   change component state to compatible.
6. Retain the prepared-phase cache for approved repeat runs. Completed PPC work
   checkpoints may be removed after final commit or retained for audit according
   to the configured checkpoint policy; cleanup must be explicit and scoped to
   the exact run fingerprint.

## 11. Plot contracts

Every figure uses an opaque white background, readable fonts, labeled axes,
trial/unit/sample counts, instability warnings, and a caption. Captions state
alignment, epoch bounds, bands, line-noise policy, reference configuration, and
normalization where applicable.

### 11.1 Power figures

- Single-trial cached trace plus absolute PSD in dB; normalized PSD is a
  selectable alternative.
- One condition-spectrum figure per selected site/epoch/normalization showing
  every condition's median with IQR shading.
- One band-power figure per selected site/normalization with conditions spaced
  widely and theta-before, theta-after, gamma-before, gamma-after spaced tightly
  within condition.
- Pre-session-normalized controls are disabled with a reason when the full
  baseline is unavailable.

### 11.2 Synchrony figures

- Frequency-by-time ITPC maps per site/condition and ISPC maps per
  pair/condition, with effective-count inspection.
- Separate theta and gamma ITPC/ISPC summary figures. Within condition, order
  configured sites followed by configured pairs; show bootstrap 95 percent
  intervals.
- Separate theta and gamma PLV distribution figures with adjacent before/after
  values per pair and median/IQR across trials.
- High/low PLV exemplars show every configured site, filtered traces, phase
  overlays, selected pair, percentile rule, trial index, and PLV.
- Preferred phase-offset values remain inspectable but are not forced into the
  primary magnitude summary.

### 11.3 Spike-phase figures

- One unit-by-frequency PPC heatmap per condition with stable reference order,
  computable/reliability mask inspection, and unit spike counts.
- Condition-by-frequency median PPC map based on reliable units.
- Condition-by-frequency significant-fraction map with eligible/total unit
  counts and NaN for no eligible units.
- Band summary with condition spacing and theta-before, theta-after,
  gamma-before, gamma-after ordering; median/IQR uses reliable units.
- Matched high/low PPC exemplar layouts containing pooled frequency-resolved
  PPC, representative-frequency polar histogram, pooled preferred phase/count,
  and the deterministic illustrative trial trace/spikes.

Plot functions return the Matplotlib figure plus axes in a documented order and
do not save automatically. The webapp's explicit save action writes PNGs to the
session figure directory using filenames containing session, component, site or
pair, condition/filter, epoch, and normalization/order settings.

## 12. Terra work packages and test-first tasks

The live status matrix identifies which packages are already complete. Their
original specifications remain below as historical contract/evidence and must
not be interpreted as authorization to reassign or redo them.

Each Terra task begins with a test-only commit, runs the listed focused command
to demonstrate RED, then makes implementation commits to reach GREEN. The Sol
orchestrator records the test-only commit, RED failure, implementation commit,
and final command output. Tests may change afterward only for an approved
requirement change or a demonstrated test defect.

### WP0 - Restore the neural-analysis baseline

Owner: Sol runner or one isolated Terra task before feature work.

Files in scope:

- Existing failing neural tests and their directly related modules only.
- The duplicate Hilbert plotting definition in `unit_spike_plotting.py`.

Tasks:

- Reconcile the two stale LFP sync tests with the current
  `lfp_loading.ephys_sync_utils` boundary.
- Reconcile the Open Ephys workflow test with the current CT026 August session
  and two-probe defaults without weakening path assertions.
- Remove the shadowed duplicate
  `plot_trial_spike_lfp_hilbert_phase_and_behavior` definition while preserving
  the tested public behavior.

Gate:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis` must be green
before new feature tests are merged. This is baseline repair, not a reason to
change scientific behavior.

### WP1 - Contracts, fingerprints, and cache I/O

Files:

- New `lfp_summary_models.py`, `lfp_summary_io.py`.
- New `test_lfp_summary_models.py`, `test_lfp_summary_io.py`.

Tests written first:

- Default configuration serializes deterministically and round-trips.
- Invalid sites, pairs, windows, bands, filters, and duplicate stable ids fail.
- Component-specific fingerprints change only for relevant settings.
- Source fingerprints detect size/mtime/sidecar changes.
- Numeric/Unicode NPZ files load with `allow_pickle=False`.
- Array-axis mismatches and missing required arrays are rejected.
- Injected write/validation failure preserves the previous valid component and
  manifest.
- Compatible, stale, missing, and failed-attempt metadata produce the expected
  status/diff.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_models.py src/tests/neural_analysis/test_lfp_summary_io.py`

### WP2 - Shared preparation and trial/site routing

Files:

- New `lfp_summary_preparation.py` and its tests.
- Minimal extraction/reuse from `psth_webapp.py`, `lfp_loading.py`, and
  `unit_spike_loading.py`; preserve their public interfaces.

Tests written first:

- Existing SpikeGLX and Open Ephys loaders route through injected fake loaders
  with documented units/sample rates.
- All nine condition masks exactly match
  `spike_behavior_pynapple.make_trial_type_masks`.
- Choice/context filters use the documented 0=right, 1=left conventions.
- User exclusions remain distinct from objective invalidity.
- Missing alignment, incomplete window, nonfinite, and constant traces receive
  explicit reason codes.
- Site-specific and pairwise validity intersections do not over-exclude an
  unrelated pair.
- RMS/peak-to-peak QC shapes and units are correct.
- Relative spike times preserve stable probe-qualified unit identity.
- Progress events are framework-independent and monotonic.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_preparation.py`

### WP3 - Power computation

Files:

- New `lfp_power_summary.py` and `test_lfp_power_summary.py`.

Tests written first:

- Synthetic 8 Hz and 40 Hz signals produce peaks on the expected 2 Hz grid.
- Whole/before/after PSD arrays have exact documented axes and half-open event
  handling.
- Every site/trial/reference uses the same frequency grid.
- Welch segment and overlap values match the approved contract.
- A trial identical to its reference normalizes to 0 dB.
- Session reference uses whole-window PSD across all valid trials and does not
  change with condition/filter membership.
- The pre-session reference uses exactly ten seconds and is unavailable for a
  shorter interval.
- Constant/nonpositive reference power yields NaN normalization, not epsilon
  substitution.
- Constant synthetic PSD integrates to the expected theta/gamma mean; gamma
  integration treats the intervals on either side of 58-62 Hz separately,
  retains the exact bandwidth, and does not discard adjacent half-intervals.
- Sites with different native Welch grids are interpolated in linear PSD units
  onto the same canonical 2 Hz cache grid before reference and band operations.
- Median/IQR input arrays preserve trial-level values and units.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_power_summary.py`

### WP4 - Synchrony computation

Files:

- New `lfp_synchrony_summary.py` and `test_lfp_synchrony_summary.py`.
- Reuse `lfp_phase_clustering.py` and `lfp_spectrogram.py` without duplicating
  their transform logic.

Tests written first:

- Fixed and random synthetic phase recover expected ITPC and ISPC magnitudes.
- Ordered constant phase differences recover the correct signed offset.
- Trial PLV is calculated across time per frequency; band output is the
  arithmetic mean of retained frequencies.
- Whole is computed directly and zero belongs only to after.
- 58/60/62 Hz are excluded from gamma band means.
- Pairwise trial/sample intersections are correct for three sites with
  different missing trials.
- Complex-coefficient interpolation onto the exact 500 Hz grid recovers a known
  cross-site lag without directly interpolating wrapped phase angles.
- PLV is available at exactly 80 percent paired phase coverage, unavailable
  below 80 percent or with fewer than two paired samples, and always saves the
  count and fraction. The test distinguishes this coverage rule from PLV
  magnitude.
- Bootstrap intervals are deterministic, use 1,000 resamples, and preserve
  condition-specific trial selection.
- Under-10 trial outputs retain values/counts and set the instability flag.
- Exemplar percentile/tie selection is deterministic.
- Cached trace arrays have source/filtered/phase units and axes; the full
  wavelet tensor is absent from the saved contract.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_synchrony_summary.py`

### WP5A - Observed PPC and population summaries

Files:

- New `spike_lfp_summary.py` and `test_spike_lfp_summary.py`.
- Reuse formulas/helpers from `spike_lfp_phase_locking.py` rather than creating
  a second PPC definition.

Tests written first:

- Explicit pairwise PPC matches the current tested formula and retains negative
  values.
- Fewer than two phases is NaN; 2-49 is computable/unstable; 50 or more is
  reliable.
- Preferred phase and resultant length recover known synthetic phase.
- Band means use the linear 2 Hz grid and exclude 58/60/62 Hz.
- Stable unit ordering uses correct-rewarded whole-window theta by default and
  is preserved across displayed conditions.
- All-computable heatmaps and reliable-only population medians use the correct
  populations.
- Representative 8/40 Hz histograms use fixed phase bins and pooled trials.
- High/low unit and median-spike-count illustrative trial selection are
  deterministic and separately labeled.
- Probe-qualified ids prevent same-number clusters on different probes from
  colliding.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_spike_lfp_summary.py -k "not shuffle"`

### WP5B - PPC trial-shuffle inference

Files:

- Continue `spike_lfp_summary.py` and its same test file after WP5A is green.

Tests written first in a separate commit:

- Every generated permutation is a derangement and is deterministic by seed.
- A shared schedule is reused across units without modifying spike timing.
- Overlapping trial windows retain trial-local spike membership, record an
  overlap warning, and never collapse the representation to merged intervals.
- Synthetic trial-specific coupling exceeds the shuffled null while independent
  phases do not systematically do so.
- Permutation p-values use the approved plus-one formula.
- Fewer than 50 spikes or fewer than two trials produces ineligible, not a
  significant/non-significant claim.
- `scipy.stats.false_discovery_control` is called with BH semantics along the
  intended frequency family and matches a small hand-calculated case.
- Null exceedance/count/moments/percentiles round-trip without saving full null
  draws.
- Shuffle chunks produce the same results as an unchunked small reference.
- Preview and final shuffle counts are explicit and fingerprint-distinct.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_spike_lfp_summary.py -k "shuffle or permutation or fdr or prevalence"`

### WP5C - Exact PPC performance redesign

WP5C follows the correct WP5A/WP5B implementation. Its packages are sequential
unless explicitly stated otherwise and must not change the scientific estimator
or public final result schema.

#### WP5C-0 - Freeze execution contracts

Owner: Sol. Documentation only.

Tasks:

- Review and freeze Sections 2.22-2.26: names/signatures, array schemas and
  axes, execution fields, schedule/edge representation, intermediate layout,
  atomicity/restart/cleanup behavior, and progress stages.
- Record the existing WP5B reference functions used for every equivalence test.
- Resolve any objection by editing this document before test work begins.

Gate (satisfied before WP5C-1 began): the user must explicitly approve the
frozen WP5C-0 contracts. Approval of the earlier documentation edit, the
Synchrony report, or the package ordering alone was not WP5C implementation
authorization.

#### WP5C-1 - Serial sufficient-statistic kernel

Owner: one implementer. Only this package may own `spike_lfp_summary.py` while
it is active.

Files:

- `src/neural_analysis/spike_lfp_summary.py`
- `src/tests/neural_analysis/test_spike_lfp_summary.py`

Tests written first:

- Complex sums/counts reproduce explicit pairwise PPC.
- Counts zero and one return unavailable PPC; count two uses the exact formula.
- Negative PPC is preserved.
- Missing phase values use the existing WP5B validity rule.
- Same-trial observed statistics remain unchanged.
- Shuffle results exactly match `compute_trial_shuffle_ppc` for the same
  schedule and seed.
- Existing p values, `ddof=0` moments, explicit-linear percentiles, and null
  eligibility remain identical.

The runtime-owned assertion that inference-ineligible entries never invoke
permutation phase sampling is tested in WP5C-4. The two WP5C-1 numerical
interfaces intentionally receive no eligibility mask and do not own runtime
dispatch; moving that assertion preserves their approved signatures and keeps
`compute_trial_shuffle_ppc` unchanged as the WP5B equivalence reference.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_spike_lfp_summary.py -k "sufficient or edge_statistic"`

No runtime, cache, pipeline, checkpoint, or parallel code belongs in WP5C-1.

Completion evidence (2026-08-28):

- Test-only commits: `cec2c91` and `00794ed`.
- Genuine RED first showed the missing numerical interfaces; the added
  zero-magnitude adjacent-support regression then failed with count 6 instead
  of the required count 4 before the interpolation validity fix.
- Implementation commit: `2460ac3`.
- Focused GREEN: 5 passed from the package RED command.
- Complete `test_spike_lfp_summary.py`: 25 passed.
- Complete `src/tests/neural_analysis`: 622 passed with the 16 known Pynapple
  warnings.
- Review confirmed that only `spike_lfp_summary.py` changed in the
  implementation commit. No runtime, cache, pipeline, checkpoint, parallel,
  external-library, or CT026 computation changes occurred.

#### WP5C-2 - Scheduled-edge and segmented reduction engine

Owner: one implementer after WP5C-1. This package retains exclusive ownership
of `spike_lfp_summary.py`.

Files:

- `src/neural_analysis/spike_lfp_summary.py`
- `src/tests/neural_analysis/test_spike_lfp_summary.py`

Tests written first:

- Source/target edge order is stable and lexicographic.
- Scheduled-edge and complete-pair modes agree for the same schedule.
- Unit blocks agree with independent unit calculations.
- Trial-edge blocks agree with the unblocked reference.
- Unit, shuffle, and trial-edge block sizes do not change results.
- Overlapping trial membership is preserved.
- Uniform-grid complex interpolation agrees with the current exact WP5B
  implementation at exact samples, between samples, and around invalid support.
- Generic histogram, firing-rate, amplitude, and result-object paths are never
  called by permutation reduction.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_spike_lfp_summary.py -k "scheduled_edge or segmented or block"`

This package remains serial and introduces no workers.

Completion evidence (2026-09-08):

- Test-only commits `ce4e3f3` and `b02da63` produced genuine RED with eight
  failures for the absent scheduled-edge engine. Test correction `6510ca2`
  replaced a reference to a nonexistent WP5B result field with the established
  null-summary interface.
- Initial implementation commit `88e9846` passed the numerical tests. Sol
  review found that it repeated edge interpolation for every shuffle block.
- Corrective test-only commit `32e6955` recorded genuine RED by detecting 36
  redundant edge samples. Corrective implementation commit `91ecc7c` samples
  each scheduled edge once per unit block before bounded shuffle aggregation.
- Final focused GREEN: 9 passed with 25 deselected. Complete
  `test_spike_lfp_summary.py`: 34 passed. Complete `src/tests/neural_analysis`:
  631 passed with the 16 known Pynapple warnings.
- Implementation commits changed only `spike_lfp_summary.py`. The package
  remained serial, preserved WP5B and the frozen public interfaces, added no
  runtime/cache/pipeline work, and ran no CT026 computation.

#### WP5C-3 - Prepared-phase persistent cache

Owner: one implementer. Work is limited to the dedicated cache module and
minimal model/runtime integration; it may begin only when those integration
files have no other owner.

Files:

- New `src/neural_analysis/lfp_summary_work_cache.py`
- New `src/tests/neural_analysis/test_lfp_summary_work_cache.py`
- Minimal `lfp_summary_models.py` and `lfp_summary_runtime.py` changes with their
  focused existing tests

Tests written first:

- Cold write and warm load are equivalent.
- Complex64 phase and Boolean validity arrays preserve exact axes.
- All NPZ loads enforce `allow_pickle=False`.
- Source, scientific configuration, schema, axis, shape, dtype, or
  incomplete-write mismatch rejects reuse.
- Injected atomic-write failure leaves no valid-looking cache.
- Work artifacts never appear compatible as final components.
- Exact run-scoped cleanup cannot affect another fingerprint.
- Concurrent-writer locking and stale/incomplete handling follow Section 2.25.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_work_cache.py`

Completion evidence (2026-09-08):

- Test-only commits `dfa37ef` and `057a9d0` produced genuine collection RED for
  the absent work-cache module and `PPCExecutionConfig`.
- Initial implementation commit `0c98c59` added the work-cache primitives and
  separated execution settings from scientific PPC configuration. Sol review
  found that failed rewrites could leave new data certified by an old completion
  marker, cleanup did not enforce the required `ppc/` parent, and Boolean block
  sizes passed integer validation.
- Corrective test-only commit `c3bab9c` recorded three cache and three model RED
  failures. Corrective implementation commit `d4bbda7` invalidates old markers
  under the exact writer lock, restricts cleanup to the exact PPC layout, and
  rejects Boolean execution counts.
- Final GREEN: `test_lfp_summary_work_cache.py` 22 passed; combined model/runtime
  tests 60 passed; complete `src/tests/neural_analysis` 665 passed with the 16
  known Pynapple warnings.
- Implementation changed only `lfp_summary_models.py` and the new
  `lfp_summary_work_cache.py`. Production reuse/restart remained deferred to
  WP5C-4; no pipeline/webapp integration, workers, dependencies, or CT026
  computation were introduced.

#### WP5C-4 - PPC checkpoints, restart, and progress

Owner: one implementer after WP5C-2 and WP5C-3. Only this package may own
runtime/pipeline integration files while it is active.

Files:

- New `src/neural_analysis/lfp_summary_ppc_runtime.py`
- New `src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py`
- Minimal changes to `lfp_summary_runtime.py`, `lfp_summary_pipeline.py`, and
  their focused tests

Tests written first:

- Completed block identity is deterministic.
- Only validated complete blocks resume.
- Interrupted work does not publish `spike_phase.npz`.
- Resumed and cold runs produce identical component arrays.
- The final manifest remains the commit point.
- Progress stages and totals are monotonic.
- ETA is absent until two timed blocks have completed.
- Execution-only settings do not stale a valid scientific result.
- Entries with fewer than 50 valid observed spike phases or fewer than two
  spike-contributing trials never invoke permutation phase sampling.
- Default post-success cleanup removes only the exact completed run while
  retained/incomplete checkpoints follow configuration.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_ppc_runtime.py src/tests/neural_analysis/test_lfp_summary_runtime.py src/tests/neural_analysis/test_lfp_summary_pipeline.py -k "checkpoint or resume or progress or execution"`

Completion evidence (2026-09-09):

- Test-only commits `456c993`, `974321f`, `66d7ed9`, `8eddc02`, `536d190`,
  `c82cd8f`, `478b4fa`, `8845496`, `8cb2188`, `c34425b`, `b8162a3`,
  `1806dbf`, `f4641d3`, `50e8b96`, `9655d89`, `00400be`, and `d19c51b`
  established the runtime/pipeline contracts, integration seams, cache reuse,
  correct fixture assumptions, identity hardening, mixed eligibility bypass,
  resumable blocks, and single edge sampling. The final sampling-once test
  recorded genuine RED against the shuffle-outer implementation before the
  edge-outer reduction was restored.
- Implementation commit `6352db4` added the serial `lfp_summary_ppc_runtime`
  execution boundary and minimal model/runtime/pipeline integration. The final
  path uses exact job/content/trial/overlap identity; an exact executor lock;
  independently resumable unit-block checkpoints with corrupt-sibling repair;
  scheduled-edge sampling once per unit block; bounded current-unit exact
  percentile draws; per-unit/frequency inference eligibility bypass; and
  summary-only assembly into the final Spike-phase payload.
- Production phase preparation now supports warm, read-only memory-mapped
  prepared-phase reuse under the fingerprinted work cache. PPC progress events
  carry stable job identities, while the pipeline remains the sole final
  component/manifest commit boundary and performs exact incomplete-only work
  cleanup only after that commit succeeds.
- Focused GREEN: `test_lfp_summary_ppc_runtime.py` 26 passed. Affected
  model/runtime/pipeline/work-cache/Spike-PPC suites passed 120 and 128 tests
  in the final integration checks. Complete `src/tests/neural_analysis`: 700
  passed with the 16 known Pynapple warnings.
- WP5C-4 remained serial (`worker_count=1`); it added no workers, dependencies,
  webapp behavior, or CT026 computation. Intermediate work artifacts remain
  outside compatible scientific components and are never rendered by the
  webapp.

#### WP5C-5 - Profiling and optional parallelism

Owner: one implementer after the complete serial path is green and Sol has
reviewed its profile.

Steps:

1. Benchmark the serial representative 249-trial workload with low-, median-,
   and high-rate unit blocks.
2. Record phase preparation, edge reduction, shuffle aggregation, peak memory,
   and throughput.
3. Decide from the profile whether within-session parallelism offers a
   meaningful benefit.
4. Only if justified, commit worker-count invariance tests and record RED before
   implementing workers.
5. Benchmark 1, 2, 4, and 8 workers with read-only shared prepared phase.
6. Select the smallest worker count near the throughput plateau.

Parallel execution is optional. Keep `worker_count=1` if workers materially
increase memory or provide little improvement.

Serial benchmark checkpoint (2026-09-09):

- The completed work-only run is
  `analysis_runs/ct026_ppc_profile_2026-09-09T21-33-06Z` under the CT026
  session. It contains atomic scalar state/profile/summary files and resumable
  phase/PPC work only; it contains no manifest or `spike_phase.npz`.
- The deterministic descriptor selected condition `incorrect`, site `PFC`, the
  half-open whole epoch `[-2, 2)` s, all 249 site-valid trials, and 263 eligible
  units from the 273-unit active ProbeB population. Low/median/high units were
  `ProbeB:39` (293 spikes), `ProbeB:181` (2,325 spikes), and `ProbeB:63`
  (15,676 spikes).
- Cold phase preparation took 136.12 s and warm mmap reuse took 7.87 s. The
  process-lifetime phase peak was 4,102,889,472 bytes. Work storage was 1.1 GiB.
- Serial 100-shuffle PPC took 20.90, 21.87, and 23.83 s for the low, median,
  and high one-unit jobs. The combined three-unit job took 66.80 s and peaked
  at 2,329,223,168 child bytes. Its observed and inclusive shuffled-edge stages
  took 26.57 and 39.69 s respectively; checkpoint overhead was 0.012 s.
- A transparent linear-per-unit projection puts the largest 249-trial
  273-unit condition/site/epoch job near 98-106 minutes at 100 shuffles. Edge
  saturation from the exact deterministic schedules projects the corresponding
  1,000-shuffle job near 3.6-3.9 hours. Applying the same measured edge/trial
  scaling to the actual nine condition trial counts, three sites, and three
  epochs gives an intentionally conservative all-eligible range of roughly
  47-51 hours at 100 shuffles and 90-100 hours at 1,000 shuffles. These are
  engineering projections, not measured full-component runtimes.
- The serial projection is materially above both review thresholds. The later
  review in `docs/ppc_speedup.md` identified enough redundant serial work that
  benchmarking 1/2/4/8 workers at this point would measure a task boundary that
  is expected to be replaced. The approved documentation plan therefore
  defers worker measurements until after S1-S6 serial optimization and
  profiling. No implementation, WP5C-6 preview, or 1,000-shuffle result is
  authorized by this profiling checkpoint or plan approval.
- Production launch review exposed and corrected three pre-computation defects:
  the runner-lock callback signature, JSON-safe identity for missing alignment
  times, and unsafe child/recovery behavior. The two failed attempts stopped
  before a phase transform or PPC scenario; their timestamped directories are
  preserved for audit. Tests and implementations remain separate commits.
- Interruption checkpoint: worker contract tests are committed in `d1bed17`,
  `32aac40`, `4dfe5fc`, and fixture correction `ccbac69`. Initial implementation
  `4cc5c83` is numerically green but is not accepted: supervisory review found
  that its `submit(...).result()` loop serializes every block and that it uses
  the platform default fork context. Corrective test-only commit `9fe7f1f`
  records genuine RED (2 failed, 12 passed) for an explicit spawn context, an
  initial bounded concurrent submission window, canonical result order, and
  failure cancellation/shutdown. Commit `37657a5` added the spawn context and
  bounded window, but the focused file still reports 2 failed and 12 passed:
  normal fake-executor shutdown is missing and the future whose `result()`
  raises is not cancelled. S0 in `docs/ppc_speedup_plan.md` is the exact first
  implementation package if implementation is separately authorized. It must
  restore the focused and full neural suites to GREEN; it must not launch a
  worker benchmark.

Completion checkpoint (2026-09-18):

- S0-S7 replaced the redundant per-job production path with the exact grouped
  executor, connected that executor once at the payload boundary, added bounded
  restartable checkpoints and shared-input workers, and preserved the public
  `spike_phase.npz` schema.
- The accepted S8 work-only benchmark used 64 rate-stratified units in eight
  blocks, 249 trials, `incorrect/PFC`, all three epochs, 50 frequencies, and
  100 shuffles. It measured three fresh repetitions each at 1/2/4/8 workers.
- Median wall times were 820.833/471.556/293.419/211.397 seconds; median
  scheduled-edge throughputs were 91.005/158.412/254.585/353.363 edges/s; and
  median aggregate PSS values were 1.718/2.291/2.646/3.352 GiB.
- Eight workers are preferred for CT026 production. This setting belongs in
  the CT026 production configuration rather than the universal library default
  and remains subject to exact per-run allocation preflight.
- Legacy overlap and 1/2/4/8-worker correctness passed. The 51 exact q-value
  differences were at most one binary64 rounding step
  (`1.1102230246251565e-16` absolute), passed the frozen tolerance, and changed
  no significance decision.
- Exact benchmark plans contained 74,700/61,586/43,301
  scheduled/independent/union edges at 100 shuffles and
  747,000/182,056/61,752 at 1,000. The 26.4-35.2 minute 1,000-shuffle estimate
  applies only to that 64-unit workload. Full 427-trial/273-unit planner
  attempts were stopped after 10 and 30 minutes, so full-component planning
  time remains a preview risk rather than a resolved projection.
- The accepted directory is
  `/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference/analysis_runs/ct026_ppc_s8_2026-09-18T15-06-33Z`.
  It contains work/evidence only and no scientific component or manifest.

#### WP5C-6 - CT026 preview validation

Owner: one implementer prepares the standalone launcher after WP10's production
dependency boundary and WP12's complete preview report path are green. The
actual CT026 execution remains a separate user-invoked run.

Files:

- New `src/neural_analysis/lfp_spike_phase_launcher.py`.
- New `src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`.
- Minimal reuse/integration changes in `lfp_spike_phase_validation.py`,
  `lfp_summary_runtime.py`, or their focused tests only when the launcher cannot
  consume an existing public seam. Do not move CLI parsing or terminal logging
  into numerical runtime modules.

The existing `lfp_spike_phase_validation.py` is a fixed 100-shuffle validation
helper, not this launcher. WP5C-6 may reuse its cache-only plotting/report
adapters, but it must not treat that helper's current API or report contents as
satisfying the launcher contract.

Launcher requirements:

- Provide a normal command-line entry point runnable with `uv run python -m ...`
  from a shell, `tmux`, `screen`, or another long-lived process supervisor. It
  must have no dependency on a live Codex task.
- Default the CT026 production configuration to eight requested workers while
  retaining exact planned active-worker and allocation preflight. Do not change
  the universal `PPCExecutionConfig` default.
- Require explicit mutually exclusive new-run and resume modes. A new run
  creates a timestamped analysis directory; resume requires that exact run
  directory and exact source/config/code identity.
- Create the analysis-run directory, initial identity/state record, log, and
  exact resume command before full PPC planning begins. The PPC work
  fingerprint may still be derived by the existing exact planner afterward.
- Planning is deterministic and deliberately not checkpointed. Resume after a
  planning-stage interruption repeats planning from the beginning. This is an
  accepted limitation, must be stated in the log/summary, and must not be
  mislabeled as reused planner work.
- Support a preflight/dry-run mode that loads metadata, resolves the active
  population, and reports the conservative metadata-only worker/phase-memory
  bounds described below without phase transforms, PPC schedules, checkpoints,
  final arrays, or manifests. It must not call this an exact grouped plan:
  exact schedules, site-valid trial membership, planner time, and allocation
  require prepared phase and are measured only by a real new/resumed run.
- Stream progress to both the terminal and a run-local log. Record the command,
  environment, git/source/config fingerprints, session and unit identities,
  worker counts, timing, memory, cache sizes, warnings, and final status.
- Interruption or failure after work publication must leave compatible
  resumable phase/PPC work and no false final component/manifest. Interruption
  during planning may leave only the analysis-run identity/log and repeats
  planning on resume. A successful run validates the scientific component,
  manifest, plots, detailed report, run log, and summary before cleaning only
  its exact completed PPC work.
- The 100-shuffle preview and 1,000-shuffle final run use the same tested
  launcher but different timestamped run directories and configuration
  fingerprints. The launcher must never continue from 100 to 1,000 shuffles
  automatically.
- The 1,000-shuffle mode requires an explicit final-run flag in addition to the
  shuffle count, so an unattended command cannot accidentally promote a
  preview invocation.
- At startup and after interruption, print the exact resume command and run
  directory. Long execution is expected; lack of Codex tool-session lifetime
  must not be treated as a computational failure.
- Run PPC with checkpoint retention set to retain through report generation,
  or provide an equivalent orchestration seam that prevents pipeline
  post-commit cleanup from running early. Exact `incomplete_only` cleanup is the
  final success step after every required human-readable artifact validates.

Tests written first:

- CLI parsing preserves the exact CT026 session, active population, worker
  count, shuffle count, and run-directory identity.
- Dry-run performs metadata-only preflight without transforms, PPC schedule
  construction, checkpoints, final arrays, or manifests; exact-plan fields are
  explicitly unavailable rather than fabricated from the S8 slice.
- New-run state and the exact resume command exist before a fake long planner
  starts; interrupting planning causes resume to repeat deterministic planning
  without claiming a planner-cache hit.
- New 100-shuffle and new 1,000-shuffle invocations cannot share or overwrite a
  run directory; 1,000 requires explicit final-run confirmation.
- Interrupted synthetic execution leaves resumable work, prints the exact
  resume command, and publishes no final component.
- Resume reuses compatible phase/PPC work, rejects every identity mismatch,
  and produces the same final result as an uninterrupted run.
- A report or plot failure after component commit retains exact PPC work and
  reports an incomplete launcher run; it must not perform success cleanup.
- A successful synthetic run atomically publishes and validates the component,
  manifest, plots/report, log, and summary, then removes only its exact
  completed PPC work.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`

Performance considerations:

- New/resumed scientific execution uses the exact grouped planner and reports
  planner wall time separately. Dry-run reports only its documented
  conservative metadata bounds, with exact planner/allocation fields
  unavailable. Neither path may use the 64-unit S8 estimate as a substitute
  for the complete active-population plan.
- Planning may repeat after interruption, but phase transforms, completed PPC
  blocks, and final component computation must not repeat when their exact
  compatible artifacts already exist. An incomplete staged report may be
  regenerated from the compatible final component.
- Logging/progress output is scalar metadata and must not serialize phase,
  spike, schedule, or component tensors.

Execution gate:

1. The user launches the active ProbeB preview with exactly 100 shuffles.
2. Record cold and warm behavior, planning and execution time, process-tree
   memory, throughput, cache sizes, unit/trial/spike counts, warnings, and
   exclusions. Full-component planner time is reported separately because S8
   could not complete the 273-unit plan within its bounded attempt.
3. Generate the complete plots/report required by WP12 and pause for user
   inspection.
4. Do not launch 1,000 shuffles automatically. After separate approval, the
   user invokes a new 1,000-shuffle run through the same standalone launcher.

#### WP5C-6 frozen launcher implementation contract (approved 2026-09-18)

This subsection supersedes any less-specific launcher wording above. The user
approved the explicit-probe, evidence-producing dry-run, clean-tracked-checkout,
and cleanup-failure recommendations before launcher tests or source edits. The
baseline is pushed commit `a8c4f3f` on `refactor`, equal to
`origin/refactor`; the tracked worktree is clean. The 171-entry normal
untracked inventory remains out of scope with NUL-delimited status SHA-256
`140fe2baeec9985753c89efe0fe2d68eb341c625065868e0437ce333e0c6a1a0`.

CLI and exit contract:

- The module entry point is
  `uv run python -m src.neural_analysis.lfp_spike_phase_launcher` and has
  required, mutually exclusive `new` and `resume` subcommands. `main(argv=None)
  -> int` performs parsing and returns an exit code; the module guard raises
  `SystemExit(main())`. Importing the module performs no filesystem, Git, data,
  plotting, signal, or process action.
- `new` requires `--session-path PATH`, `--probe ProbeA|ProbeB`, and
  `--shuffles 100|1000`. It accepts `--workers N` with CT026 default eight,
  `--analysis-root PATH` defaulting to `<session>/analysis_runs`, optional
  `--dry-run`, and `--final-run`. Probe selection is always explicit; there is
  no silent ProbeB default and no combined population.
- `--final-run` is required exactly when `--shuffles 1000` and is rejected for
  100 shuffles. Dry-run accepts either reviewed configuration but never makes
  the 1,000-shuffle mode executable without `--final-run`. Worker counts are
  positive integers and never change the universal library default.
- `resume` requires only `--run-directory PATH`. Session, probe, shuffle count,
  worker request, final-run acknowledgement, cache/report paths, and exact
  identities are loaded from validated state rather than repeated on the
  command line. Resume rejects dry-run evidence directories and completed runs;
  it never converts a 100-shuffle run into 1,000 shuffles.
- Exit zero means either terminal metadata-only `preflight_complete` for a
  dry-run or fully validated `complete` for a scientific run. CLI/config/source/
  identity errors return 2, other execution/report/cleanup failures return 1,
  SIGINT returns 130, and SIGTERM returns 143. Every nonzero path atomically
  records the status when the run directory already exists and flushes the log.

CT026 configuration and population contract:

- Add a general `build_ct026_active_population(session_path, probe_label,
  cluster_metadata_loader, channel_metadata_loader)` cache-free metadata helper
  in `lfp_spike_phase_validation.py`. Preserve
  `build_ct026_default_active_population(...)` as the unchanged ProbeB wrapper.
  ProbeA uses `Record_Node_101_Neuropix-PXI-110.ProbeA/kilosort4` and
  `probeA_sync.npz`; ProbeB uses the existing ProbeB paths. Both select good or
  MUA clusters only on channels labeled good and inside brain, sort by cluster
  id, and retain stable `ProbeA:<cluster>` or `ProbeB:<cluster>` identities.
- Build from the existing CT026 three-site configuration, set the explicitly
  selected population, requested shuffle count, eight-worker CT026 default or
  explicit worker override, checkpointing enabled, and
  `checkpoint_retention="incomplete_only"`. Every other scientific/default
  value remains unchanged. Nonempty absolute-amplitude thresholds fail before
  run-directory creation, phase preparation, or work publication.
- A new scientific run requires a clean tracked checkout. The check is
  `git status --porcelain --untracked-files=no`; unrelated untracked files do
  not make the checkout dirty and are never added, removed, hashed recursively,
  or copied. State records the exact lowercase commit. Resume requires the same
  commit and a still-clean tracked checkout.

Run-directory and artifact contract:

- A new run is a nonsymlink direct child of its resolved analysis root named
  `<session_id>_spike_phase_<probe>_<preview|final|dry_run>_<UTC timestamp>`.
  Creation uses `exist_ok=False`. Resume accepts only that existing naming and
  parent relationship; no arbitrary path, symlink, or collision is followed.
- Before exact PPC planning, a scientific new run atomically writes
  `launcher_state.json`, `configuration.json`, `source_identity.json`,
  `preflight.json`, creates `run.log`, and prints plus logs the shell-quoted
  absolute resume command. `run.log` is ASCII JSON Lines, one mapping per line,
  flushed after every event. JSON artifacts use sorted keys, `allow_nan=False`,
  ASCII encoding, and a trailing newline. `run_summary.md` is nonempty ASCII
  Markdown and is atomically refreshed at every terminal/incomplete status.
- Reports are published beneath `<run>/report/` by the generalized WP12
  cache-only report seam; the returned report directory is recorded in state.
  The generic numerical component and manifest remain in the configuration's
  `processed/lfp_summary_cache` output directory. The launcher neither copies
  numerical arrays into the analysis directory nor treats the report as the
  scientific commit point.
- `launcher.lock` is an advisory exclusive local-process lock held across state
  validation and execution. Its scalar owner record contains run id, hostname,
  pid, and acquisition UTC. A live lock fails closed; kernel lock release after
  process death permits explicit resume. The launcher never deletes another
  run's lock or any sibling work directory.
- A dry-run creates the same timestamped evidence directory and writes validated
  identity/configuration/source/preflight/log/summary artifacts, then ends in
  `preflight_complete`. It writes no resume command, report directory, prepared
  phase cache, PPC work directory, component, or manifest and is not resumable.

Identity and state schema:

- `launcher_state.json` has schema `spike_phase_launcher_state.v1`. Its fixed
  top-level keys are schema version, run id, created/updated UTC, run kind,
  dry-run Boolean, terminal/active status, ordered completed stages, resume
  command, identity, paths, measurements, cleanup request, report directory,
  warnings, and error. Optional unavailable values are JSON null, never zero or
  an empty string.
- Identity contains resolved session and repository paths; session id; probe;
  ordered stable units and their SHA-256; shuffle count; final-run Boolean;
  requested workers; canonical configuration and Spike-component fingerprints;
  the existing `fingerprint_source_files(config, "spike_phase")` mapping and
  its canonical SHA-256; Git commit; and tracked-clean Boolean. Paths contain
  the analysis root/run directory, numerical output directory, phase/PPC work
  root, and report parent. Resume rebuilds all derivable values and requires
  exact equality before any phase, spike, planner, writer, or plot call.
- Ordered completed stages are an exact prefix of `initialized`,
  `preflight_complete`, `cleanup_prepared`, `component_complete`,
  `report_complete`, `launcher_artifacts_validated`, `cleanup_complete`, and
  `complete`. `active_status` may additionally be `preparing_phase`,
  `planning`, `executing`, `reporting`, `cleaning`, `interrupted`, `failed`, or
  `cleanup_failed`. State replacement is atomic after every durable boundary.
  Deterministic planning has no completed-stage marker and is repeated when
  interrupted before `component_complete`; logs explicitly say it repeated.
- Source/config mismatch, malformed state, missing committed artifacts, an
  invalid stage prefix, or changed population/source/Git identity fails before
  numerical callbacks. A compatible committed component may be reused on
  resume to regenerate an absent/failed report without recomputation. A new run
  that finds a compatible component records `component_reused=true`, validates
  it, and renders from cache rather than silently overwriting it.

Persistent cleanup handoff:

- WP12's in-memory callback is insufficient by itself because a process can
  stop after component commit but before report publication. Add an immutable,
  JSON-safe cleanup-target record containing only the exact resolved PPC run
  directory and 64-lowercase-hex run fingerprint. The Spike payload carries
  both its existing callable and the ordered target tuple. Defaults remain empty
  for every existing caller and non-Spike payload.
- Add an optional cleanup-preparation observer to Spike component execution.
  When deferral is requested, the pipeline validates and sends the targets to
  that observer after payload construction but before the final component
  writer. The launcher atomically persists them and completes
  `cleanup_prepared`; observer failure prevents component commit. A failed
  writer never exposes the in-memory callback but may leave the saved exact
  request for later validated retry/recomputation.
- A successful deferred `ComponentRunResult` retains the callable and also
  exposes the same immutable targets. Default nondeferred execution still calls
  cleanup immediately and returns neither. Resume reconstructs cleanup only by
  validating that each target is a nonsymlink direct child of the configured
  `lfp_summary_work/ppc` parent, basename equals fingerprint, and its metadata
  matches that fingerprint, then calls the existing `cleanup_ppc_run` API.
- Cleanup occurs only after component/manifest reload, WP12 report validation,
  launcher JSON/log/summary reload, source identity recheck, and atomic
  `launcher_artifacts_validated` state. Cleanup failure records
  `cleanup_failed`, returns nonzero, preserves the component/report/work, and
  is resumable. Resume revalidates all outputs and retries only exact cleanup;
  it does not recompute. `complete` is written only after every target is absent
  and `cleanup_complete` is durable.

Measurements and reporting contract:

- Additive JSON-scalar execution metadata flows from the grouped runtime through
  `ComponentPayload` and `ComponentRunResult`; existing callers receive an
  empty mapping by default. The Spike runtime records exact PPC planning and
  grouped-execution seconds, requested and planner-active worker counts,
  planned parent/worker/aggregate/shared-phase bytes, scheduled/independent/
  union edge counts, completed/resumed block counts, run fingerprint, and run
  directory. It retains no plan/schedule/phase/spike arrays in this mapping.
- Phase preparation reports `cold` or `warm` on `PreparedPhaseRun` through an
  additive default-`None` categorical field. The launcher measures phase,
  overall component, and report wall time with a monotonic clock. It measures
  final component/cache and exact intermediate-work bytes from persisted files.
- A standard-library Linux process-tree sampler periodically reads the parent
  and recursively discovered children under `/proc`, deduplicates pids, and
  records peak process RSS, process-tree RSS, and process-tree PSS in bytes with
  its sampling interval/provenance. If `/proc` values are unavailable, affected
  fields remain null and a warning is recorded; unavailable never becomes zero.
  The sampler is injected in tests and adds no package dependency.
- Dry-run `preflight.json` has schema `spike_phase_preflight.v1` and records the
  population/unit/trial metadata counts, requested workers, conservative
  maximum active workers from unit-block count, configured allocation limits,
  estimated phase tensor/validity bytes from documented axes, and source/cache/
  report paths. Exact site-valid membership, schedules, planner seconds,
  planned PPC allocation, active processes, and runtime/cache-hit fields are
  null with a categorical explanation. The 64-unit S8 timings are never copied
  into this file.
- Generalize the cache-only WP12 publication internals into a public report seam
  usable by both 100- and 1,000-shuffle launcher runs. Existing fixed preview
  entry points and `spike_phase_preview_report.v1` remain backward compatible.
  Launcher reports use `spike_phase_report.v1`, include `run_kind` and the
  actual configured shuffle count, and otherwise preserve the WP12 bounded
  selections, artifacts, strict staged validation, and null-versus-zero rules.

Progress, interruption, and resume contract:

- One launcher progress adapter timestamps each framework-independent event,
  validates nondecreasing counts within a stage, appends it to `run.log`, and
  prints a concise flushed terminal line containing UTC, run id, component,
  stage, completed/total, elapsed, ETA when available, and message. Logs contain
  no arrays, schedules, spike trains, or phase values.
- Install SIGINT/SIGTERM handlers only while `main` owns a scientific run. The
  first signal records its number and raises one launcher interruption through
  the main thread so existing executor/finally logic stops new submission,
  releases locks, and preserves atomic checkpoints. The launcher then writes
  `interrupted`, the exact resume command, and signal-specific exit code. A
  second signal restores/immediately invokes the default behavior rather than
  pretending orderly shutdown succeeded.
- Any interruption before a run directory exists simply returns the signal exit
  code. Any interruption after initialization preserves state/log/work and no
  false complete marker. Report failure after component commit preserves exact
  work and resumes at reporting. Interruption or failure after
  `launcher_artifacts_validated` but before cleanup completion resumes at exact
  cleanup. No resume path reruns a compatible component.

Tests written before implementation:

- Parser tests cover required subcommands/probe, CT026 eight-worker default,
  100/1,000/final-run matrix, positive overrides, dry-run restrictions, and a
  resume command that accepts only the exact run directory.
- ProbeA/ProbeB metadata tests prove identical quality rules, qualified stable
  ids, correct sorter/aligned-spike paths, no combined population, and backward
  compatibility of the ProbeB helper.
- New-run tests prove all identity/state/log/config/preflight files and the
  printed exact resume command exist before an injected blocking planner.
- Dry-run tests prove the evidence directory and unavailable exact-plan fields,
  and prove zero phase/plan/checkpoint/component/manifest/report calls.
- Clean-Git, threshold, symlink/path, collision, state-schema, stage-prefix,
  source/config/population/commit, and live-lock mismatches fail before work.
- Planning interruption and SIGINT/SIGTERM tests prove nonzero exit, flushed
  state/log/resume command, preserved work, and repeated deterministic planning
  without a false cache-hit claim.
- Structured cleanup tests prove targets persist before the writer, writer
  failure exposes no callback, successful default execution remains immediate,
  and deferred result/observer targets are exact and identical.
- Synthetic interrupted execution resumes valid phase/PPC checkpoints and
  matches uninterrupted output. Compatible committed components skip compute
  and regenerate missing reports cache-only.
- Report/plot/launcher-artifact failures retain targets and never clean. Success
  validates component, manifest, bounded report, state, every JSON log line,
  summary, and source identity before exact cleanup and final completion.
- Cleanup failure produces `cleanup_failed`; resume performs no phase/planner/
  executor/writer/report work, revalidates, retries cleanup, and reaches
  complete. Sibling/fingerprint/path substitution is rejected.
- Measurement tests distinguish null from measured zero, cold from warm,
  planner from grouped/component/report time, requested/planner-active/measured
  workers, process from process-tree RSS/PSS, final from intermediate bytes,
  and preview from final report schema/shuffle count.
- CLI/module tests prove no Streamlit import, no import-time side effects, no
  external dependency, terminal/log progress flushing, and documented exit
  codes. Existing WP12 preview, immediate-cleanup, composed-runtime, synthetic
  integration, and full neural tests remain green.

Implementation order and verification:

1. Commit this documentation-only contract before any launcher test or source
   edit.
2. Add and commit tests only; run the launcher RED command and record only
   failures attributable to the absent launcher/general population/report and
   additive metadata/cleanup seams.
3. Implement the smallest launcher plus the explicitly documented additive
   validation/runtime/pipeline/report seams. Do not add a dependency, UI, SLURM
   script, CT026 result, or unrelated refactor.
4. Run launcher tests, directly affected runtime/pipeline/report tests,
   synthetic interruption/resume, synthetic LFP integration, and the complete
   `src/tests/neural_analysis` suite. Review every write/cleanup target and run
   `git diff --check`.
5. Commit implementation separately. Append exact commits, RED/GREEN counts,
   warnings, changed-file scope, and residual execution risks here before any
   CT026 command is proposed.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_spike_phase_launcher.py`

Interruption recovery:

- Before the documentation commit, no launcher test/source file exists and no
  CT026 run has been started. Resume from pushed `a8c4f3f`, verify the tracked
  worktree and NUL-delimited untracked hash above, then continue with the
  tests-only commit.
- No partial launcher implementation authorizes a dry-run or scientific CT026
  invocation. The first authorized execution after all launcher code/tests/docs
  are green is a separately user-invoked 100-shuffle ProbeB preview. The
  1,000-shuffle run remains separately gated by inspection and approval.

#### WP5C-6 implementation handoff (completed 2026-09-18)

Status and commits:

- The frozen documentation contract is commit `9139a58` (`docs: freeze Spike
  launcher contract`). The initial tests-only checkpoint is `696d823` (`test:
  define resumable Spike launcher contract`). Two omissions found during the
  implementation audit were handled with their own RED tests before their
  fixes: phase-preparation timing in `9bda3cd` and resume source/lock hardening
  in `d916bc3`.
- The implementation checkpoint is `814f253` (`feat: add resumable Spike phase
  launcher`). It adds only `lfp_spike_phase_launcher.py` and localized additive
  changes to `lfp_spike_phase_validation.py`, `lfp_summary_pipeline.py`,
  `lfp_summary_ppc_runtime.py`, and `lfp_summary_runtime.py`. No dependency,
  UI, SLURM script, data result, or unrelated refactor was added.
- The launcher now supports exactly one explicit ProbeA or ProbeB population
  per new run, 100-shuffle preview and separately acknowledged 1,000-shuffle
  final modes, metadata-only evidence dry runs, exact-identity resume,
  manifest-last computation, cache-only report publication, persistent
  fingerprint-qualified cleanup handoff, cleanup-only recovery, process-tree
  memory sampling, and first-signal orderly interruption.

RED evidence:

- The initial launcher command produced 6 expected failures because the module
  did not exist. The directly affected auxiliary selection produced 6 expected
  failures with 254 deselected, and the cache-state selection produced 1
  expected failure with 5 deselected.
- The later phase-timing contract produced 1 expected failure with 15
  deselected (`phase_preparation_seconds` absent). The source-evidence/live-lock
  hardening selection produced 2 expected failures with 6 deselected (tampered
  source evidence accepted and a competing lock owner state mutated). Each was
  committed before the corresponding source fix.

GREEN evidence:

- The standalone launcher suite passes: 8 passed in 1.20 seconds.
- The directly affected launcher/pipeline/runtime compatibility rerun passes:
  35 passed in 1.63 seconds. The earlier six-file focused run passed 266 tests
  with one deliberate duplicate-NPZ warning.
- The complete command
  `uv run pytest -q -p no:cacheprovider src/tests/neural_analysis` passes 1,139
  tests in 150.29 seconds with 20 warnings. The warnings are pre-existing test
  fixture/runtime warnings: three multiprocessing fork deprecations, one
  deliberate duplicate ZIP member warning, and sixteen Pynapple empty-epoch or
  zero-duration rate warnings.
- `python -m py_compile` passed for all five changed production modules,
  `git diff --check` passed, and the module `--help` entry point loaded without
  data access. No CT026 dry run, phase preparation, PPC execution, report, or
  cleanup was invoked during implementation or verification.

Residual execution risks and next gate:

- Synthetic seams cover interruption, cleanup-only resume, source tampering,
  live locking, and measurement propagation, but the production filesystem
  layout, real sorter/channel metadata, Linux `/proc` process tree, and report
  plotting stack have not yet been exercised together for CT026. The first
  operational step should therefore be the metadata-only ProbeB 100-shuffle
  dry run using the real session path, followed by inspection of its identity,
  population count, paths, and conservative memory bounds.
- A dry run does not authorize scientific execution. After its evidence is
  reviewed, the 100-shuffle ProbeB preview remains a separate explicit user
  invocation. The 1,000-shuffle final run remains blocked until that preview
  report is inspected and separately approved. ProbeA uses the same defaults
  only when explicitly selected in a distinct run.

#### R1 - Local ProbeB preview report recovery (approved plan 2026-09-20)

Purpose and evidence:

- The ProbeB preview run
  `CT026_2026-08-01_130853_spike_phase_ProbeB_preview_2026-09-18T20-36-32Z`
  completed all 105 grouped blocks and atomically committed
  `processed/lfp_summary_cache/spike_phase.npz`. Its manifest component state
  is `complete` with configuration fingerprint
  `b5ff9895295c1946408fe9e493b7bd698c91b5ac825ce1eb3f199056863e520b`.
- The launcher completed `initialized`, `preflight_complete`,
  `cleanup_prepared`, and `component_complete`, then failed during report
  rendering with `bottom cannot be >= top`. `report_directory` remains null,
  the report parent is empty, and success cleanup did not run.
- The final component is 211876438 bytes. It contains 273 ProbeB units, 427
  trials, nine overlapping condition labels, three sites, three epochs, and 50
  frequencies. There are 1080900 computable cells, 914700 reliable/null-
  eligible cells, and 167860 eligible cells marked significant at the
  100-shuffle FDR resolution. These are preview diagnostics, not final
  scientific inference.
- The exact report defect is an unbounded population caption that serializes
  two complete `(condition, frequency)` count matrices. For the real `9 x 50`
  cache the caption is 4789 characters and 36 wrapped lines; `_caption` asks
  Matplotlib for bottom margin 1.71, which cannot be below the subplot top.
  Numerical arrays and the manifest are not implicated.
- Measured component time was 11719.959 seconds: 115.086 seconds phase
  preparation, 3653.792 seconds PPC planning, and 7940.727 seconds grouped
  execution. Peak process RSS was 3911606272 bytes; process-tree PSS was
  4146417664 bytes; process-tree RSS was 7690588160 bytes. The retained exact
  PPC work directory occupies approximately 1.3 GB. The previous 45-120 minute
  1,000-shuffle expectation is invalid and must not be reused.

Ownership and files:

- Production: `lfp_summary_plotting.py`,
  `lfp_spike_phase_validation.py`, and `lfp_spike_phase_launcher.py`.
- Tests: `test_lfp_summary_plotting.py`,
  `test_lfp_spike_phase_validation.py`, and
  `test_lfp_spike_phase_launcher.py`.
- No numerical PPC kernel, phase-preparation, payload, cache schema, scientific
  parameter, unit selection, shuffle schedule, or final component array is in
  scope. A demonstrated test failure is required before adding another source
  file.

Population-map correction:

- `plot_population_ppc_maps` becomes a three-panel figure containing reliable-
  unit median PPC, fraction of eligible units passing FDR, and exact eligible-
  unit count on identical `(condition, frequency)` axes. The count-panel title
  records the total configured unit population. The plain-language prevalence
  label is `Fraction of eligible units passing FDR`.
- For each displayed cell the prevalence is
  `count(significant & null_eligible) / count(null_eligible)`. A zero eligible
  denominator remains NaN. This is a prevalence diagnostic, not an effect
  size; PPC magnitude remains the separate first panel.
- The third panel, rather than a truncated matrix string, preserves
  frequency-specific denominator inspection. The caption contains bounded
  scalar provenance only and never serializes a condition-by-frequency array.
  Caption size must be independent of unit, condition, frequency, trial, or
  shuffle count.
- Existing cache arrays and schemas are sufficient. There is no cache rewrite
  or numerical recomputation.

Explicit report-only recovery:

- Ordinary `resume` remains exact-identity recovery and continues to reject a
  changed Git commit. Before report recovery its required commit is the saved
  computation commit. After a recovered report reaches `report_complete`, a
  cleanup-only resume requires the exact recorded report-code commit while
  still validating the unchanged computation identity. Do not otherwise weaken
  or silently bypass the rule.
- Add the explicit command
  `uv run python -m src.neural_analysis.lfp_spike_phase_launcher recover-report --run-directory PATH`.
  It is the only cross-commit recovery path and never accepts new session,
  probe, shuffle, worker, output, or scientific parameters.
- Recovery accepts only a nonsymlink launcher directory whose saved stage
  prefix ends at `component_complete`, whose report is absent, and whose
  cleanup target remains saved. It acquires the existing launcher lock before
  mutable validation or output.
- The current checkout must be tracked-clean, use the same repository, and
  descend from the saved computation commit. The saved configuration snapshot,
  component fingerprint, population ids/order/hash, source fingerprints,
  source-file metadata, output paths, and exact completed component must all
  revalidate. A mismatch fails before a plot, report, cleanup, phase, planner,
  spike, or worker callback.
- Recovery must have no reachable numerical-computation path. It loads the
  committed component through the cache validator, renders the immutable
  report, validates every staged and published artifact, advances
  `report_complete` and `launcher_artifacts_validated`, and only then removes
  the saved fingerprint-qualified work target and completes the remaining
  launcher stages.
- Preserve `source_identity.json` as the computation identity. Add an atomic
  `report_recovery.json` audit artifact and scalar log/summary/state fields
  recording the computation commit, report-code commit, original error,
  recovery UTC, reused-component status, and recovery status. Conditional
  launcher-artifact validation requires this artifact for recovered reports.
  No array, phase tensor, schedule, or spike train may enter it.
- If report rendering/publication/validation fails, record the new error,
  retain `component_complete`, retain the exact work target, publish no partial
  final report, and allow another explicit recovery after a later clean commit.
  If cleanup alone fails, the established cleanup-only resume semantics remain
  applicable only at the exact report-code commit recorded by recovery.

Tests written before implementation:

- A realistic `9 x 50` population map returns `median_ppc`, `prevalence`, and
  `eligible_count` axes, preserves supplied exact counts and NaN zero-
  denominator cells, records total units, uses a bounded non-matrix caption,
  and has valid subplot margins.
- A full-size synthetic 273-unit cache-backed report publishes atomically and
  closes every figure. Existing small-array population-map behavior remains
  numerically unchanged apart from the additive count panel and label.
- Ordinary resume still rejects a different Git commit for its active stage,
  including a cleanup-only recovered run at anything other than its recorded
  report commit. Recovery rejects a
  dirty checkout, same-commit use when ordinary resume is available, a
  non-descendant commit, a stage before `component_complete`, an already
  complete report, symlink/path escape, malformed state, missing cache,
  incompatible component, changed configuration/population/source identity,
  and a live competing lock.
- Eligible recovery calls component validation and report publication but
  injected phase, planner, worker, and component-write seams fail the test if
  reached. Report failure retains the target. Successful recovery records both
  commits, validates before cleanup, removes only the exact saved target, and
  reaches the existing complete stage sequence.
- The existing `spike_phase_launcher_state.v1` file from the real run remains
  readable; recovery provenance is additive within the flexible measurement
  mapping plus its conditional audit artifact rather than an undocumented
  rewrite of numerical identity.

RED and verification commands:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_plotting.py src/tests/neural_analysis/test_lfp_spike_phase_validation.py src/tests/neural_analysis/test_lfp_spike_phase_launcher.py -k "population_ppc or report or recover"`

R1 tests-only RED checkpoint (2026-09-20):

- Test-only commit `b77f4a5` adds the approved population-layout, realistic
  full-report, explicit recovery, provenance, failure-retention, locking, and
  recovered cleanup-only resume contracts in exactly the three test files
  listed above. No production or numerical source changed in that commit.
- The exact focused command above selected 19 tests and produced the intended
  genuine RED: 11 failed, 8 passed, and 25 were deselected in 5.74 seconds.
  The failures are attributable to the absent `eligible_count` panel, the
  reproduced real-size `bottom cannot be >= top` layout failure, and the
  absent `recover-report` command. Collection, fixtures, and the eight
  pre-existing compatible contracts succeeded.
- Production implementation remains unstarted at this checkpoint. The next
  permissible edit is the smallest change within the three R1 production
  files, followed by the focused GREEN and broader verification below.

R1 implementation checkpoint (2026-09-20):

- Implementation commit `97fbf3b` adds the bounded three-panel population map
  and explicit report-only recovery transaction. Only
  `lfp_summary_plotting.py` and `lfp_spike_phase_launcher.py` changed; the
  numerical kernel, runtime, payload, configuration, cache schema, and report
  compositor required no edits.
- The focused command is GREEN: 19 passed and 25 deselected in 2.11 seconds.
  The complete three-file R1 set passed 44 tests in 2.47 seconds. Directly
  affected synthetic integration, pipeline, runtime, and grouped-PPC tests
  passed 243 tests in 24.68 seconds with one deliberate duplicate-NPZ warning.
- The complete neural suite passed 1,149 tests in 128.45 seconds with 20 known
  warnings: three multiprocessing fork deprecations, one deliberate duplicate
  ZIP member warning, and sixteen Pynapple empty-epoch or zero-duration rate
  warnings. `python -m py_compile` passed for all three R1 production files,
  and `git diff --check` passed.
- No experimental component, report, or cleanup callback was invoked during
  implementation verification. The remaining R1 action is the explicit
  report recovery of the exact run named above from a clean descendant commit,
  followed by artifact and image inspection. C1 remains blocked.

After focused GREEN, run the complete three files, directly affected
pipeline/runtime/report tests, and then:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis`

Implementation and execution gates:

1. Commit this complete documentation contract before changing tests or
   production source.
2. Commit tests only and record failures attributable to the absent count panel
   and recovery command before implementation.
3. Implement the smallest three-file production change, run focused/full
   verification, and commit implementation separately.
4. Invoke `recover-report` on the exact run above. It must reuse the committed
   component and perform zero scientific computation.
5. Inspect all PNGs, `report.json`, launcher log/summary/state, and recovery
   provenance. Only successful report validation authorizes exact cleanup and
   user approval of the local preview.
6. Do not launch 1,000 shuffles locally. Completion of R1 authorizes C1 cluster
   work, not the final scientific run by itself.

#### C1 - Simple uv/SLURM execution and remote inspection (approved plan 2026-09-20)

Purpose, ordering, and dependencies:

- C1 begins only after the recovered local 100-shuffle report is inspected.
  It precedes WP11 and is the required execution path for the ProbeB
  1,000-shuffle final run and preferably all later 100-shuffle previews.
- Reuse the existing Python module entry point; do not create a second
  computation implementation. `src/shell_scripts/hpc_ppc.sh` is a thin SLURM
  wrapper that forwards the launcher's `new`, `resume`, and later approved
  `recover-report` arguments exactly.
- `pyproject.toml`, `uv.lock`, and the sample `hpc_ppc.sh` became tracked at
  pushed commit `ce37409`. The user has installed uv on cluster access. Bash,
  Slurm, Git, uv, a pre-synchronized locked project environment, and the
  cluster data filesystem are the only operational dependencies. No new Python
  dependency is planned.

Harmless uv gates before wrapper implementation or scientific access:

1. On cluster access run `uv --version`.
2. Run
   `uv run --no-project --no-python-downloads --offline python -c 'print("uv cluster check: OK")'`.
   It must print exactly the success text without importing project code,
   opening data, contacting a package index, or starting Slurm.
3. Environment creation/synchronization is a separate explicit login-node
   operation. After it is ready, run
   `uv run --frozen --no-sync --offline python -c 'import numpy, scipy; import src.neural_analysis.lfp_spike_phase_launcher; print("project imports: OK")'`.
   This is still read-only and performs no scientific computation.

SLURM wrapper contract:

- Use partition `unlimited`, one task, eight CPUs per task, 32 GB memory, and
  an initial 72-hour wall time. Request `--signal=B:TERM@300` so the launcher's
  tested SIGTERM path has five minutes to persist state and flush logs.
  Retain the user's configured log directory and mail address from the tracked
  sample unless cluster validation requires a documented path correction.
- Replace the contradictory sample `-n 16`/`--ntasks=1` pair with
  `--ntasks=1` and `--cpus-per-task=8`. Eight is the launcher request and the
  process-pool size; the parent process does not justify sixteen Slurm tasks.
- Set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
  `OPENBLAS_NUM_THREADS=1` to prevent hidden nested threading. Validate that an
  explicit `--workers` value equals `SLURM_CPUS_PER_TASK`; omission is accepted
  only when both resolve to the approved launcher default of eight.
- Change into one explicit cluster repository checkout, print hostname, UTC,
  Git commit/status, Slurm job metadata, uv version, and the shell-quoted
  launcher arguments, then use
  `uv run --frozen --no-sync --offline python -m src.neural_analysis.lfp_spike_phase_launcher "$@"`.
  Use `exec` or an equivalent forwarding trap so signals and exit codes reach
  the launcher. The job never installs, synchronizes, upgrades, or downloads a
  package.
- Session and analysis roots remain explicit launcher paths. The wrapper never
  hardcodes local `/home/matt/Documents` paths, chooses a probe, combines
  probes, changes shuffle count, adds `--final-run`, finds a latest run, or
  automatically resumes/submits another job.
- The same script handles 100 and 1,000 shuffles. A preview uses
  `--shuffles 100` without `--final-run`; a final run uses
  `--shuffles 1000 --final-run`. Both keep the approved scientific defaults
  and explicitly select either ProbeA or ProbeB.

Wrapper tests written before implementation:

- A fake `uv` executable proves exact argument preservation, including paths
  containing spaces, and proves the wrapper reaches only the existing module.
- Missing repository paths, dirty tracked checkout, absent Slurm CPU metadata,
  and worker/CPU mismatch fail before Python. Session, probe, shuffle count,
  and final-run validation remain the launcher's responsibility and are never
  invented by Bash.
- Thread-limit variables, working directory, offline/no-sync/frozen uv flags,
  stdout/stderr behavior, launcher exit-code propagation, and SIGTERM
  forwarding are covered without requiring a live scheduler.
- `bash -n src/shell_scripts/hpc_ppc.sh` passes. Tests never submit `sbatch`,
  open CT026 data, or perform package/network operations.

Cluster validation before a scientific run:

1. Commit documentation, then wrapper RED tests, then implementation in
   separate commits under the same TDD rules as R1.
2. On the cluster, verify the exact pushed commit and clean tracked checkout;
   run the project-import command and focused/full neural tests.
3. Submit a non-scientific wrapper smoke job and verify Slurm logs, eight CPU
   allocation, thread limits, signal/exit handling, and uv offline execution.
4. Run metadata-only launcher dry runs for both explicit ProbeA and ProbeB
   paths. Inspect identities, populations, source paths, output paths, and
   preflight records. Dry runs do not authorize scientific execution.
5. Present the exact ProbeB final submission command and evidence for explicit
   approval. The launcher must still require `--shuffles 1000 --final-run`.

Performance and resume policy:

- The 72-hour request is deliberately conservative for the first cluster
  result. The observed local 100-shuffle component took 3.26 hours, including
  1.01 hours of planning and 2.21 hours of grouped execution. Some work is
  fixed and edge unions can saturate, but schedule construction and shuffle
  aggregation grow with shuffle count; the final job must be treated as a
  tens-of-hours workload until measured.
- The 32 GB request is conservative relative to 4.15 GB observed process-tree
  PSS, 7.69 GB observed process-tree RSS, 5.11 GB planned aggregate arrays,
  and 1.17 GB shared phase mmap. Do not request the sample's 128 GB without new
  measured evidence. Revisit CPUs, memory, and time only after the first
  1,000-shuffle result.
- A timeout/preemption uses the launcher's persisted run directory and a new
  explicit `resume` submission. Never create a second final run to conceal an
  incomplete first run. Deterministic planning may repeat as already approved;
  compatible phase/PPC checkpoints remain exact-identity reusable.

Cluster-resident cache and SSH inspection:

- Numerical caches, manifests, source data, and final validation remain on the
  cluster under their cluster absolute paths. Do not use SSHFS as the primary
  cache path and do not rewrite absolute identities during transfer.
- Human-readable immutable reports may be copied locally for review. Numerical
  inspection runs Streamlit on the cluster filesystem and binds only to
  `127.0.0.1`; a local browser connects through SSH port forwarding. Only UI
  messages and rendered figures cross SSH, not the complete NPZ on every
  interaction.
- WP11 must cache one validated selected component server-side using manifest
  and file identity, keep arrays on the cluster, close every figure, and avoid
  reloading/decompressing the component on each Streamlit rerun. This makes the
  approximately 212 MB preview component, and a similarly shaped final
  component, practical for interactive inspection after an initial load.
- If cluster policy forbids a persistent login-node process, launch Streamlit
  in an interactive or scheduled compute allocation and tunnel through the
  login node. Never bind the application to a public interface. A lost SSH
  connection cannot mutate or invalidate the cache.
- Root-portable numerical identity and local inspection of a copied cache are
  explicitly deferred. They are unnecessary for the approved cluster-side
  server topology and must not be smuggled into C1 or WP11 without a separate
  documentation/test contract.

Final-run gate and handoff:

- After all cluster validation passes, the user separately approves one
  ProbeB 1,000-shuffle submission. Monitor its Slurm log and launcher state;
  resume exact incomplete work when needed. Do not alter scientific parameters
  in response to runtime.
- Completion requires a validated final component, immutable report, launcher
  artifacts, and exact cleanup. Inspect its figures and scalar diagnostics,
  compare preview/final stability without treating overlapping conditions as
  independent, and record measured wall time, memory, storage, and utilization.
- Only then reduce future resource requests and continue WP11 against the
  cluster-resident cache. The cluster wrapper does not itself implement WP11.

### WP6 - Plotting

Files:

- New `lfp_summary_plotting.py`, `test_lfp_summary_plotting.py`.

Tests written first:

- Every required figure returns documented axes and uses opaque white output.
- Condition spectra plot median/IQR for all requested conditions.
- Grouped band plots preserve the required within/between-condition spacing and
  measurement order.
- ITPC/ISPC and PLV plots display counts, uncertainty, and instability labels.
- PPC maps preserve reference unit order and distinguish unreliable entries.
- Significant prevalence uses eligible denominators and displays NaN when none
  are eligible.
- Exemplar plots distinguish pooled metrics from the illustrative trial.
- Captions contain reference, alignment, line-noise, count, and unit statements.
- Filename builders encode selections without overwriting unrelated figures.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_plotting.py`

### WP7 - Component pipeline and progress

Files:

- New `lfp_summary_pipeline.py`, `test_lfp_summary_pipeline.py`.

Tests written first:

- Each component invokes only its required preparation/computation/writer path.
- Compute All calls Power, Synchrony, Spike phase in order.
- Progress callbacks receive monotonic stages and totals.
- Shared in-memory phase preparation is reused in Compute All when compatible.
- A failed component preserves prior valid files and reports the failing stage.
- Component fingerprints/states update independently.
- No pipeline or numerical module imports Streamlit.
- Fixed seeds make repeated synthetic runs numerically reproducible apart from
  declared timestamp fields.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_pipeline.py`

### WP8 - Webapp integration

Files:

- New `lfp_summary_webapp.py`, `test_lfp_summary_webapp.py`.
- Minimal route addition in `psth_webapp.py` and focused existing test updates.

Tests written first:

- UI-helper defaults reproduce the approved CT026 session/channels and analysis
  defaults.
- Configuration assembly maps active session, sites, unit population, filters,
  exclusions, advanced parameters, and output directory correctly.
- Separate component and Compute All actions call the injected pipeline entry
  points.
- Missing/compatible/stale/running/failed states render correct messages.
- Stale differences are visible and stale plots require explicit override.
- Successful computation reloads the cache; failure leaves old valid results.
- Plot selectors load only required component files.
- Trial/unit/spike counts and instability warnings are not hidden.
- Matplotlib figures are closed after Streamlit rendering.
- Existing raster, PCA, spectrogram, phase, and Hilbert routes remain reachable.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_webapp.py src/tests/neural_analysis/test_psth_webapp.py`

### WP9 - Synthetic integration and CT026 validation

Owner: Sol runner, with a Terra task only for bounded diagnostic fixes.

Steps:

1. Run the complete pipeline on seeded artificial LFP/spike/trial data with
   known 8/40 Hz power, known phase offsets, locked/unlocked trial groups, and
   locked/unlocked units.
2. Verify every cache component and render every plot type.
3. Power on `CT026_2026-08-01_130853` using PFC=5, HPC1=222, HPC2=14 is
   complete and user-approved.
4. Synchrony on the same session is complete; the user approved the latest
   maps, summaries, counts, exemplars, and report on 2026-08-27.
5. Complete WP5C and benchmark the 249-trial condition before starting another
   complete Spike-phase calculation. Use measured throughput to project the
   cold and warm 100- and 1,000-shuffle runtimes and report any projection
   materially above 60 minutes or two hours, respectively.
6. Run Spike phase first in 100-shuffle preview mode for the active unit
   population; user inspects reliability/count behavior and the measured
   performance report.
7. Run the 1,000-shuffle result only after separate preview approval and
   explicit execution authorization.
8. Record stage-specific wall time, peak memory, final and intermediate cache
   size, cache hit/miss state, worker/block settings, unit/trial/spike counts,
   exclusions, warnings, and projected nine-filter cost.
9. Present the performance report to decide whether all filter combinations
    should be precomputed in a later task.
10. Do not run other sessions until the user approves the CT026 outputs.

Validation human-readable outputs go under a timestamped session analysis-run
directory, outside the repository, containing PNGs, configuration/manifest
snapshot, pipeline source identifiers, a log, and `run_summary.md`. The
overwriteable numerical inspection cache remains at the generic processed path.

### WP10 - Composed production dependencies

Owner: one implementer; this is the next implementation package. The
implementer exclusively owns runtime/pipeline integration files while active.

Files:

- `src/neural_analysis/lfp_summary_runtime.py`
- `src/neural_analysis/lfp_summary_pipeline.py`
- `src/tests/neural_analysis/test_lfp_summary_runtime.py`
- `src/tests/neural_analysis/test_lfp_summary_pipeline.py`

Architecture:

- Add this project-internal factory to `lfp_summary_runtime.py`:

  ```python
  make_lfp_summary_pipeline_dependencies(
      *,
      trial_table_loader,
      unit_spike_loader=load_configured_unit_spikes,
      phase_preparer=None,
      site_phase_tensor_builder=lfp_phase_clustering.compute_site_phase_trial_tensor,
      block_loader_factory=None,
      spikeglx_loader=None,
      open_ephys_loader=None,
  ) -> PipelineDependencies
  ```

  Its docstring must specify callable inputs, prepared record types, axes,
  seconds/Hz/source-voltage units, returned dependency bundle, and failure
  behavior. Do not add a second configuration or component-payload type.
- The composed factory binds the existing `prepare_power_run`,
  `prepare_phase_run`, `prepare_spike_run`, `build_power_payload`,
  `build_synchrony_payload`, `_build_spike_phase_payload`,
  `load_or_initialize_manifest`, and `write_component_transaction` seams. It
  must not copy their numerical implementations.
- `phase_preparer` is an optional complete `LFPSummaryConfig ->
  PreparedPhaseRun` seam for focused tests or an already-composed upstream
  caller. When it is `None`, phase preparation delegates to
  `prepare_phase_run` with the configured transform/load seams and the existing
  sibling `lfp_summary_work` directory.
- Keep `make_power_pipeline_dependencies`,
  `make_synchrony_pipeline_dependencies`, and
  `make_spike_phase_pipeline_dependencies` unchanged as compatible focused
  entry points. WP10 does not remove, redirect, or broaden their unsupported
  component behavior.
- The composed bundle is the only production boundary used by Compute All,
  the later webapp, and the launcher. It must not route a requested component
  through a component-specific factory that silently rejects the other
  components.
- Compute All prepares Power independently, prepares compatible phase once,
  and passes that same prepared phase object to Synchrony and Spike phase.
  Existing component and manifest transactions remain independent.
- Preserve framework-independent `ProgressEvent` records from both the
  component pipeline and grouped PPC executor. The production boundary accepts
  a callback; it does not import Streamlit or format terminal output.
- Add one small runtime validator for the currently supported phase-amplitude
  policy. Each composed `prepare_power`, `prepare_phase`, and `prepare_spike`
  callable invokes it before its first loader, cache, planner, or writer-capable
  seam. It rejects a nonempty
  `config.phase.absolute_amplitude_thresholds` with a clear WP13-oriented
  `ValueError`. Calling it in `prepare_power` is deliberate: a composed
  production request must not partially run under an unsupported configuration.
  Empty thresholds reproduce the accepted current result. This guard is
  temporary and must be removed only by WP13's tested implementation.
- The composed manifest loader retains the same active validated configuration
  supplied to the preceding preparation call and delegates to
  `load_or_initialize_manifest`. Calling it before any composed preparation
  fails clearly rather than inventing a default configuration.
- The progress-aware Spike payload adapter forwards the exact callback to
  `_build_spike_phase_payload`. It must not translate, reorder, strip, or copy
  grouped PPC progress fields. The frozen three-argument
  `build_spike_phase_payload` remains the non-progress fallback stored in the
  bundle.
- Do not alter scientific fingerprints, PPC numerics, worker defaults, final
  component schemas, or the existing exact grouped preflight.

Tests written first:

- One production dependency bundle supports Power, Synchrony, Spike phase, and
  Compute All.
- Compatible phase preparation is shared once.
- Component-specific failures remain isolated and do not invalidate successful
  compatible siblings.
- No component-specific factory silently handles unsupported work.
- Progress remains component/stage specific and final manifests remain the
  component commit points.
- Power-only, Synchrony-only, and Spike-only calls through the composed bundle
  invoke only their required preparation and payload paths.
- Compute All preserves Power, Synchrony, Spike-phase order and reuses one
  compatible phase preparation for the latter two components.
- Grouped PPC progress fields pass through unchanged, including elapsed time,
  optional ETA, and job identity.
- A nonempty absolute-amplitude threshold fails before trial/LFP/unit loaders,
  prepared-phase cache access, PPC planning, or manifest writes. The empty
  tuple remains bitwise/numerically identical to the current tested path.
- Existing component-specific factory tests remain green.
- The composed factory's manifest loader rejects use before preparation and
  uses the exact most recently prepared configuration afterward.
- The optional injected `phase_preparer` is used once by Compute All and its
  exact returned `PreparedPhaseRun` object reaches both Synchrony and Spike
  payload paths.
- Spike preparation rejects a non-`PreparedPhaseRun` object and preserves the
  existing `PreparedSpikeRun` axes and trial-local spike semantics.
- No new imports of Streamlit, CLI parsing, plotting, or external dependencies
  appear in runtime or pipeline modules.

Performance considerations:

- Composition adds no transform or source reload. Compute All must not duplicate
  the approximately 1 GiB prepared-phase representation.
- Progress forwarding must not copy payload arrays. Eight workers remain a
  caller-supplied CT026 execution setting, not a factory default.
- Run focused runtime/pipeline tests first, then the complete neural suite. No
  CT026 computation is part of WP10 verification.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_runtime.py src/tests/neural_analysis/test_lfp_summary_pipeline.py -k "composed or threshold or progress or compute_all"`

Implementation and verification sequence:

1. Modify only the two focused test files and commit the tests before source.
2. Run the RED command and record failures caused by the absent composed
   factory/guard behavior, not fixture or import mistakes.
3. Implement the smallest clear runtime change. Change
   `lfp_summary_pipeline.py` only if a new RED test demonstrates that its
   existing Compute All or callback seam cannot satisfy the frozen contract.
4. Run the RED command to GREEN, then complete runtime/pipeline files, affected
   Power/Synchrony/Spike runtime tests, grouped PPC progress regressions, and
   the complete `src/tests/neural_analysis` suite.
5. Review the source diff for public-interface preservation, exact prepared
   object reuse, absence of numerical duplication, data contracts, no
   Streamlit import, and unchanged scientific/execution defaults.
6. Commit implementation separately from tests. Append exact commits, RED/GREEN
   counts, full-suite counts/warnings, changed-file scope, and unresolved risks
   to the final handoff section before WP12 begins.

Interruption recovery:

- Documentation-only baseline: commit `8286d15` plus the forthcoming WP10
  contract commit, on branch `refactor` tracking the same `origin/refactor`
  commit before this documentation edit.
- Pre-WP10 worktree inventory contains 171 pre-existing untracked entries and
  no tracked modifications, with `git status --short -z` SHA-256
  `140fe2baeec9985753c89efe0fe2d68eb341c625065868e0437ce333e0c6a1a0`.
  Preserve every entry; stop if unrelated tracked changes appear or the
  inventory changes unexpectedly.
- Before tests exist, resume at the documentation-only commit and write the
  listed tests. After the test-only commit, reproduce RED before source. After
  an implementation commit, rerun focused and complete neural suites before
  declaring WP10 complete. No partial state authorizes WP12 or CT026 execution.

### WP11 - Full Streamlit integration

Owner: one implementer after R1 recovery and C1 cluster infrastructure are
validated;
exclusive owner of
`lfp_summary_webapp.py`, `psth_webapp.py`, and their focused tests.

Files:

- `src/neural_analysis/lfp_summary_webapp.py`
- `src/neural_analysis/psth_webapp.py`
- `src/tests/neural_analysis/test_lfp_summary_webapp.py`
- `src/tests/neural_analysis/test_psth_webapp.py`

Architecture:

- The summary route receives both existing sorter paths and aligned-spike paths
  from `psth_webapp.py`. It exposes an active-population selector with exactly
  `ProbeA` and `ProbeB`; exactly one is selected per run.
- Construct one `UnitPopulationConfig` from the selected probe. Both probes use
  the same defaults: good or MUA clusters restricted to channels labeled good
  and inside brain. Preserve the selected probe label, sorter/aligned source,
  selected channels, quality settings, and stable `probe:cluster` unit ids.
- A combined ProbeA-plus-ProbeB population is out of scope. Switching probes
  changes Spike-phase compatibility and requires an explicit separate run; it
  must not merge units into an existing component.
- The first cluster-integrated UI is a cache inspector and launcher handoff,
  not the owner of a multi-hour process. It shows copyable local/SLURM new and
  resume commands plus validated launcher/report state. Spike phase and Compute
  All do not run synchronously inside a Streamlit callback. Existing bounded
  Power/Synchrony actions may continue to use the WP10 composed production
  boundary; any future background submission control requires its own contract.
- Progress rendering may adapt saved launcher/log `ProgressEvent` metadata but
  never performs numerical work or restarts an action on rerender.
- Cached plotting loads only the selected compatible final component. Work
  caches, checkpoints, launcher state, and incomplete reports remain invisible
  as scientific results.
- Production deployment runs Streamlit beside the numerical cache on the
  cluster, binds to loopback, and is viewed through SSH port forwarding. It
  does not load a copied component over SSHFS or remap absolute source paths.

Tests written first:

- ProbeA and ProbeB each build the expected single active population with the
  same good/MUA and good-inside-brain defaults.
- Sorter and aligned-spike paths from the existing page controls are passed to
  the selected population without substituting the other probe's paths.
- Stable ids are probe-qualified, and switching the selector changes the
  Spike-phase fingerprint. No control creates a combined population.
- Bounded Power/Synchrony actions call only the composed production boundary.
  Spike phase and Compute All present launcher/SLURM handoff commands and
  cannot enter a long-running numerical callback.
- Progress stages, counts, elapsed time, and optional ETA render without
  numerical computation inside Streamlit.
- Compatible/stale/missing/failing component states and fingerprint differences
  are inspectable.
- All cached selectors/plots load only final compatible component files.
- A cluster-side selected component is cached server-side by manifest/file
  identity and is not reopened on ordinary Streamlit rerenders; browser-facing
  output contains figures/scalars rather than the component NPZ.
- Prepared-phase caches, checkpoints, and other work artifacts never appear as
  compatible scientific components.
- A nonempty unsupported amplitude-threshold request is blocked before action
  dispatch and is never presented as applied.

Performance considerations:

- Population metadata may be loaded for selector construction, but Streamlit
  rerenders must not reopen raw LFP or recompute phase/PPC data.
- Figure rendering closes every Matplotlib figure and loads only the selected
  component NPZ. The selected validated component remains server-side on the
  cluster; SSH transports rendered UI output rather than numerical arrays.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_webapp.py src/tests/neural_analysis/test_psth_webapp.py -k "population or synchrony or spike_phase or compute_all or progress"`

### WP12 - Complete PPC plotting and reporting

Owner: one implementer immediately after WP10. Scope is plotting/report files
and the smallest report-orchestration seam required to defer PPC cleanup until
the report transaction completes. There is no WP11 ownership overlap.

Files:

- `src/neural_analysis/lfp_summary_plotting.py`
- `src/neural_analysis/lfp_spike_phase_validation.py`
- `src/neural_analysis/lfp_summary_payloads.py`
- `src/neural_analysis/lfp_summary_runtime.py`
- `src/neural_analysis/lfp_summary_pipeline.py`
- `src/tests/neural_analysis/test_lfp_summary_plotting.py`
- `src/tests/neural_analysis/test_lfp_spike_phase_validation.py`
- `src/tests/neural_analysis/test_lfp_summary_payloads.py`
- The smallest focused additions to `test_lfp_summary_runtime.py` and
  `test_lfp_summary_pipeline.py` needed to prove exact exemplar identities and
  report-before-cleanup behavior. No production UI or launcher file is in
  scope.

Architecture:

- Extend the existing cache-only plotting/report path. Figures receive only
  validated `spike_phase.npz` arrays and immutable configuration metadata; they
  never reopen raw LFP/spike sources, prepare phase, plan PPC, or recompute a
  numerical result.
- Correct the additive exemplar cache contract. The frozen cache already stores
  `selected_low_unit_ids` and `selected_high_unit_ids` on
  `(condition, site, epoch, band)` axes, but its one
  `illustrative_trial_indices` field records the high unit's median-spike-count
  trial when available and therefore cannot faithfully illustrate both units.
  Add required int64 arrays `illustrative_low_trial_indices` and
  `illustrative_high_trial_indices` with the same axes and trial-table-row
  units; `-1` means unavailable. Retain `illustrative_trial_indices` unchanged
  as the legacy high-first/fallback-low alias so existing callers do not change
  meaning. No raw-data reconstruction or approximate trial re-selection is
  allowed in plotting.
- `_select_spike_exemplars` continues to call the established deterministic
  `select_ppc_exemplars` function. It returns both exact per-unit trial arrays
  in addition to the existing unit-id and legacy arrays. The 5th/95th
  percentile rule, smallest-stable-id unit tie break, positive-spike median,
  and earliest-trial tie break remain unchanged. This additive schema makes an
  older Spike-phase NPZ fail closed during payload validation; there is no
  migration or silent default, and no completed CT026 Spike-phase cache exists
  to preserve.
- Add a frozen `PPCExemplarPanel` plotting record containing one unit id,
  percentile label, dimensionless pooled frequency PPC, preferred phase in
  radians, pooled representative-frequency histogram counts, illustrative
  trial-table row, source and band-filtered trace in the configured source
  voltage unit, Hilbert phase in radians, and event-relative spike seconds.
  Add `plot_ppc_exemplar_pair(...)`, which accepts the shared Hz/radian/seconds
  coordinates plus low/high panel records and returns one unsaved Matplotlib
  figure with explicitly separate low/high pooled PPC, polar histogram, trace,
  Hilbert-phase, and spike panels. The existing single-exemplar public function
  remains unchanged.
- The approved bounded preview plan is fixed rather than inferred from whichever
  cache cell happens to contain the most reliable values. It uses condition
  `correct_rewarded`, every configured site, epochs `before` and `after`, and
  bands `theta` and `gamma`. For three CT026 sites this produces 12 paired
  low/high exemplar figures. It also produces six corresponding unit maps, six
  population maps, and one all-condition/all-epoch band summary per site, for
  at most 27 PNGs. A scientifically unavailable exemplar cell is recorded with
  its reason and produces no fake PNG; it does not invalidate otherwise valid
  report publication.
- Add frozen `SpikePhaseFilterBenchmark` and
  `SpikePhaseReportMeasurements` records. Their explicit scalar fields cover
  phase-preparation, PPC-planner, grouped-execution, component-total, and report
  seconds; requested/planned/active worker counts; cold/warm prepared-phase
  state; process and process-tree peak RSS, process-tree peak PSS, and memory
  provenance; final component/cache and intermediate-work bytes; warning and
  exclusion strings; and the representative one-filter benchmark used for the
  nine-filter projection. Every optional unavailable measurement is Python
  `None`, JSON `null`, and Markdown `unavailable`; measured zero remains numeric
  zero. Nonnegative/finite units and categorical states are validated before
  staging writes.
- Preserve the current public preview entry points and add optional keyword
  `report_measurements` inputs. A direct computed preview supplies its measured
  component-total seconds, requested workers, current-process peak RSS when
  genuinely available, final persisted sizes, and a traceable current-filter
  benchmark. Measurements that this helper cannot observe, especially
  planner/grouped splits and process-tree RSS/PSS, remain unavailable unless
  the future launcher explicitly supplies them. Cached rendering never invents
  compute measurements.
- The machine report has schema `spike_phase_preview_report.v1`. It records the
  active population/probe, requested/planned/active workers, stage and report
  timings, cold/warm state, memory values plus provenance, final/intermediate
  sizes, total/selected trials, units, unique packed spikes, computable/reliable/
  null-eligible/significant cell counts, warnings, exclusions, rendered and
  unavailable selection records, and a nine-filter projection containing its
  representative benchmark label and inputs. Projection fields are present but
  null when an input is unavailable.
- Stage the exact artifact set beneath a temporary sibling directory:
  `report.json`, `run.log`, `manifest_snapshot.json`, `configuration.json`,
  `source_identifiers.json`, ASCII `run_summary.md`, and the deterministic PNG
  set. JSON uses sorted keys, `allow_nan=False`, and a trailing newline. Before
  publication, reload every JSON mapping, validate the report schema and
  selection/file agreement, parse each JSON log line, verify nonempty ASCII
  Markdown, verify each required PNG signature/nonempty file, and reject
  unexpected or missing artifacts. Only then atomically rename the staging
  directory to the final timestamped path.
- Extend `ComponentRunResult` with a default-`None` exact deferred-cleanup
  callable. Add keyword `defer_post_commit_cleanup=False` to
  `compute_spike_phase_component` and the internal commit seam, plus
  `defer_spike_phase_cleanup=False` to Compute All. Defaults preserve all
  existing immediate post-commit cleanup behavior. When explicitly deferred,
  the final component and manifest are committed, cleanup is not invoked, and
  the exact callback is returned in the successful component result.
- Production Spike preview computation opts into deferral. A successful report
  result exposes that same callback only after staged validation and atomic
  publication. WP12 never calls it. The forthcoming standalone launcher owns
  the last success step and invokes it only after it validates its own log and
  state artifacts as well. Compute, cache, figure, serialization, validation,
  or publication failure returns/exposes no cleanup request and leaves exact
  resumable PPC work intact.
- Preserve all scientific PPC arrays, seeds, shuffle semantics, worker/allocation
  defaults, component fingerprints, existing component-specific factories, and
  the WP10 composed boundary. The only final payload change is the two additive
  illustrative-trial arrays.

Tests written first:

- The payload schema requires both low/high illustrative-trial arrays, validates
  their exact axes/dtypes/units, rejects an older one-trial-only payload, and
  retains the legacy high-first/fallback-low field unchanged.
- Deterministic ties yield the established low/high unit ids and each unit's own
  median-positive-spike trial. The two trial ids may differ and survive the
  complete runtime payload round trip without changing PPC values.
- The paired figure uses separate low/high pooled curves, polar histograms,
  source/filtered/Hilbert traces, and trial spikes; labels and caption cannot
  confuse pooled statistics with either single illustrative trial. The actual
  nearest cached 8/40-Hz coordinate is displayed.
- The fixed report planner produces exactly 12 CT026 exemplar selections, six
  unit maps, six population maps, and three band summaries. Missing low/high
  scientific exemplars become explicit unavailable records rather than dummy
  figures or selection substitution.
- Every figure has readable labels, opaque white background, black axes/text,
  and a caption with session, alignment, epoch, band/frequency, unit/trial,
  reliability/count interpretation, and statistical-test meaning where
  applicable.
- Reports contain the complete detailed run log, warning/exclusion summary,
  population/probe identity, computable/reliable/null-eligible/significant
  counts, and stage performance/cache measurements.
- The nine-filter runtime/final/intermediate-storage projection is present and
  names the exact representative benchmark and scalar inputs. Missing
  intermediate storage stays null rather than becoming zero.
- Unavailable measurements and measured zero round-trip distinctly through
  report JSON, log, and Markdown. Cold, warm, and unavailable prepared-phase
  states remain distinct.
- Requested/planned/active workers, planner and grouped time, process/process-
  tree RSS/PSS provenance, final/intermediate sizes, units/trials/spikes,
  exclusions, and warnings all appear with their documented units.
- Default component execution still runs cleanup after a successful final
  writer. Opt-in deferred execution commits first, returns the exact callable,
  does not call it, and never returns it after a failed writer.
- Injected plot, JSON serialization, staged validation, or atomic-publication
  failures leave no final report directory and do not invoke or expose the PPC
  cleanup request.
- A fully successful staged report exposes the exact cleanup request only after
  all PNGs, report/log, configuration/manifest snapshots, source ids, and
  Markdown summary reload and validate from the published directory.
- Cache-only report rendering calls no raw loader, phase preparation, PPC
  planner/executor, or compute seam and does not create a second full component
  array copy.
- Existing single-exemplar plotting, pipeline immediate-cleanup, runtime PPC,
  component schema, synthetic integration, and WP10 composition tests remain
  green.

Performance considerations:

- Load the validated final component once. Build bounded NumPy views for one
  selection at a time, save and close each figure immediately, and never retain
  a second full PPC component or all figures concurrently.
- The approved 27-PNG upper bound replaces an exhaustive 162 paired-exemplar
  expansion. Deterministic filenames encode component, condition, site, epoch,
  band, and paired percentile identity without collisions.
- Report JSON/log/summaries contain scalar or short categorical data only.
  Report timing is measured separately from phase preparation, planning,
  grouped execution, and final component time.
- Add no dependency. Use the existing NumPy, Matplotlib, JSON, temporary-
  directory, and filesystem primitives. No CT026 data or production-sized work
  is part of WP12 verification.

RED command:

`uv run pytest -q -p no:cacheprovider src/tests/neural_analysis/test_lfp_summary_plotting.py src/tests/neural_analysis/test_lfp_spike_phase_validation.py -k "ppc or exemplar or report or cleanup or projection"`

Implementation and verification sequence:

1. Commit this exact documentation-only contract before any WP12 test or source
   edit.
2. Add and commit tests only in the listed test files. Run the RED command and
   record failures caused by absent paired plotting, two-trial schema, detailed
   report publication, or cleanup deferral behavior.
3. Implement the additive payload/runtime contract, paired plotting, report
   records/staging validation, and opt-in pipeline deferral with the smallest
   localized changes. Do not change tests merely to reach GREEN.
4. Run the RED command, complete focused files, affected payload/runtime/
   pipeline/report tests, synthetic LFP-summary integration, and the complete
   `src/tests/neural_analysis` suite.
5. Review diffs for cache-only plotting, exact low/high trial identities,
   absence of full-array copies, unavailable-versus-zero semantics, artifact
   atomicity, deferred-cleanup ownership, no Streamlit/CLI imports, unchanged
   scientific defaults, and no raw CT026 access.
6. Commit implementation separately from tests. Append exact commits, RED/GREEN
   counts, suite warnings, changed-file scope, and unresolved launcher risks to
   the handoff section before WP5C-6 begins.

Interruption recovery:

- WP12 starts from pushed commit `3aebba8` on `refactor`, equal to
  `origin/refactor`. The tracked worktree is clean. The 171 pre-existing
  untracked entries retain status-inventory SHA-256
  `140fe2baeec9985753c89efe0fe2d68eb341c625065868e0437ce333e0c6a1a0` and
  remain out of scope.
- Before this contract commit, the two existing focused files pass 17 tests.
  After the documentation commit, resume with the tests-only commit and RED.
  After the implementation commit, rerun focused/affected/full verification and
  record it here.
- No partial WP12 state authorizes WP5C-6, WP11, WP13, a CT026 dry run, or the
  100-/1,000-shuffle scientific computation. The launcher remains a separate
  documentation-first, test-first package after WP12 is fully green.

### WP13 - Apply configured absolute amplitude thresholds

Owner: one implementer after WP5C prepared-phase contracts are frozen.

This package concerns only the optional per-site absolute wavelet-amplitude
cutoff in source voltage units. It is distinct from the numerical
relative-magnitude validity threshold and from PPC's 50-spike/two-trial
reliability gates, which are already implemented. The current CT026 preview
configuration inherits `absolute_amplitude_thresholds=()`, so WP13 has no
effect on that default computation and may be deferred without changing its
scientific result.

Files:

- `lfp_summary_preparation.py`, the smallest required phase-runtime adapter,
  and their focused tests

Tests written first:

- Per-site thresholds apply in source voltage units before phase validity is
  consumed.
- Site/trial/frequency/time axes remain unchanged.
- Masked phase values and explicit exclusion reasons propagate independently by
  site and pair.
- Threshold changes stale affected scientific phase-derived components.
- Empty threshold configuration reproduces the current production result.

Until WP13 is complete, an empty threshold configuration proceeds without a
warning because it requests no absolute masking. Every nonempty production
request must be rejected before preparation or work publication; warning and
continuing is no longer an approved alternative. Implement WP13 before
advertising nonempty absolute thresholds as a supported production feature.

## 13. Sol orchestration and merge gates

### 13.1 Execution waves

1. Completed historical packages WP0 through WP5B and the Spike runtime bridge
   are not reassigned or reimplemented.
2. Sol presents WP5C-0 for explicit user approval. No WP5C tests or source
   changes begin before that approval.
3. One implementer owns `spike_lfp_summary.py` at a time: WP5C-1, then WP5C-2.
4. WP5C-3 may proceed only when its new cache files and minimal model/runtime
   integration are disjoint from any active owner. Only one implementer at a
   time owns `lfp_summary_runtime.py` or `lfp_summary_pipeline.py`.
5. WP5C-4 integrates the already-green serial numerical and cache work.
6. WP5C-5 and its approved S0-S8 sequence are complete. Eight workers are the
   preferred CT026 production setting; the universal default remains unchanged.
7. WP10, WP12, and the standalone launcher are complete. The local ProbeB
   100-shuffle component is complete; R1 is the next package and may perform
   report-only recovery but no scientific recomputation.
8. After R1 report inspection, C1 validates the simple uv/Slurm wrapper and
   cluster paths. The ProbeB 1,000-shuffle run remains a separate explicitly
   approved cluster invocation and never follows automatically.
9. WP11 follows R1/C1 and supports one explicitly selected ProbeA or ProbeB
   population per run, never a combined population. Its long-running Spike
   controls hand off to the launcher; cache inspection runs cluster-side over
   an SSH tunnel.
10. WP13 remains deferred while absolute thresholds are empty. WP10, the
   launcher, and WP11 must reject every nonempty unsupported request before
   computation; warning and continuing is forbidden.
11. No CT026 computation is an implicit consequence of a merge, test run, or
   completed package.

### 13.2 Review gates after every package

Sol checks:

- Every implementation package has a test-only commit, recorded genuine RED
  output, a separate implementation commit, focused GREEN output, complete
  neural-suite output, and Sol review.
- Every function documents types, shapes, axes, units, and returns.
- Public existing interfaces remain compatible unless explicitly listed.
- No external-library code was modified.
- No condition, unit, or sample-count threshold was silently introduced.
- Random seeds and parameters are saved.
- Numerical logic is outside Streamlit.
- Array transposes, reshapes, interpolation, decimation, and unit conversions
  are documented and tested.
- Relevant focused tests and the complete neural-analysis suite pass.
- Unexpected worktree changes stop the package and are escalated.
- Each implementer stages only its package files. Existing user documentation
  edits and all unrelated untracked files are preserved.
- Only one owner edits `spike_lfp_summary.py`; only one owner edits runtime or
  pipeline integration files at a time.

### 13.3 Performance approach

- Correctness-first implementations vectorize across frequency where clear and
  process wavelet data in bounded blocks.
- PPC permutation work uses the exact sufficient-statistic, adaptive-edge, and
  unit-block design in Section 2.21. It never materializes a
  source-by-target-by-frequency-by-spike phase cache.
- PPC shuffles run in bounded unit and shuffle blocks so memory is proportional
  to configured block sizes rather than all spike phases or 1,000 full-session
  null copies.
- The initial single-session implementation does not add Numba, Cython, MEX,
  Zarr, or a niche parallel dependency.
- The grouped benchmark is complete. Request eight workers for CT026 production
  because they delivered the only median throughput within 10 percent of the
  best measured value. Keep planned private peak allocation at or below 2 GiB
  per process, planned aggregate arrays at or below 12 GiB, and measured
  aggregate PPC working memory below 16 GiB. Continue to report job
  accumulators, kernel work, geometry, parent/worker private peaks, six-input
  shared mmap bytes, and measured process-tree RSS/PSS separately.
- The measured 26.4-35.2 minute 1,000-shuffle range applies only to the
  64-unit/one-condition/one-site/three-epoch S8 workload and is not a
  full-session estimate. The former 25-60 minute preview and 45-120 minute
  final projections are disproven by the actual 273-unit preview: 115.086
  seconds phase preparation, 3653.792 seconds planning, 7940.727 seconds
  grouped execution, and 11719.959 seconds total. Treat the 1,000-shuffle run
  as a tens-of-hours cluster workload, initially request 72 hours, and continue
  to report the three stages separately. A valid prepared-phase cache removed
  about 101 seconds in the serial profile (108.93 seconds cold versus 8.12
  seconds warm), not the earlier estimated 8-10 minutes.
- Batch parallelism across sessions is deferred until more than one session is
  approved for processing.

## 14. Final verification and approval gate

Original documentation-only handoff verification completed on 2026-08-27:

- Every cited commit resolved in this repository, including `01b7507`,
  `c8e76d5`, and the then-current/upstream tips.
- Every named existing source/test file and every historical test-command target
  exists. Future `lfp_summary_work_cache.py`,
  `test_lfp_summary_work_cache.py`, `lfp_summary_ppc_runtime.py`, and
  `test_lfp_summary_ppc_runtime.py` paths are explicitly marked new; their RED
  commands become executable when the required test-only commit creates the
  corresponding test file.
- The approved Power and Synchrony report directories exist with configuration,
  manifest, log, summary, source identifiers, and PNG outputs.
- The CT026 generic cache contains compatible Power/Synchrony artifacts and no
  `spike_phase.npz`; no Spike-phase preview report exists.
- The full neural suite passed 617 tests with the 16 known warnings at that
  handoff HEAD.
- Package ownership is sequential for every shared numerical/runtime file;
  parallel work is permitted only for disjoint files.
- Each remaining requirement is assigned to WP5C-0 through WP5C-6 or WP10-WP13.
  Absolute amplitude thresholds are visible in WP13 rather than silently
  ignored.
- `Tasks_neural.md` contains one authoritative next gate. The contradictory
  historical journal instruction is explicitly superseded by its appended
  corrective entry.
- Final Power, Synchrony, and Spike-phase scientific schemas remain unchanged;
  new schemas are work-only execution artifacts.
- This documentation action edited only `docs/Tasks_neural.md`,
  `docs/DevJournal.md`, and `docs/webappDesign.md`; all other dirty and untracked
  worktree entries were preserved.

WP5C-0 approval and WP5C-1 completion update (2026-08-28):

- The user approved the frozen WP5C-0 contracts before test or implementation
  work began.
- WP5C-1 satisfied the required test-only commit, genuine RED, separate
  implementation commit, focused GREEN, complete neural-suite run, and scope
  review.
- The exact next package is WP5C-2. It must begin with its listed tests and RED
  evidence before implementation.
- WP5C-3 through WP5C-6 remain pending. In particular, no CT026 computation is
  an implicit consequence of WP5C-1 completion or WP5C-2 authorization.

WP5C-2 completion update (2026-09-08):

- WP5C-2 satisfied separate test and implementation commits, two recorded RED
  phases, focused and complete-file GREEN, the complete neural suite, and Sol
  scope/performance review.
- The serial scheduled-edge engine produces block-size-invariant results,
  preserves overlapping trial membership and WP5B interpolation semantics, and
  samples each required edge once per unit block.
- The exact next package is WP5C-3. WP5C-4 through WP5C-6 remain pending, and no
  CT026 Spike-phase computation is authorized by this completion.

WP5C-3 completion update (2026-09-08):

- WP5C-3 satisfied separate test and implementation commits, two recorded RED
  phases, focused GREEN, the complete neural suite, and Sol safety review.
- Prepared phase is stored as validated complex64/Boolean NPY arrays with
  explicit axes and read-only memory mapping. Checkpoints and cleanup remain
  exact-fingerprint, work-only operations outside final component compatibility.
- `PPCExecutionConfig` now owns execution-only settings, which serialize but do
  not stale final scientific components.
- The exact next package is WP5C-4. WP5C-5 and WP5C-6 remain pending, and no
  CT026 Spike-phase computation is authorized by this completion.

WP5C-4 completion update (2026-09-09):

- WP5C-4 satisfied the recorded test-only commits, genuine sampling-once RED,
  implementation commit `6352db4`, focused GREEN, affected integration suites,
  complete neural-suite GREEN, and Sol contract/readability review.
- The serial execution path now resumes only exact fingerprinted job blocks,
  preserves selected stable trial and overlap identities, samples scheduled
  edges once, bypasses inference-ineligible unit/frequency entries, and keeps
  only summary products and bounded current-unit null draws.
- Warm prepared-phase reuse is read-only memory-mapped; final component
  compatibility remains manifest-last and exact completed PPC work cleanup is
  post-commit only. No workers, new dependencies, webapp changes, or CT026
  Spike-phase computation were introduced.
- The exact next package is WP5C-5. WP5C-6 remains pending, and no CT026
  Spike-phase computation is authorized by this completion.

Exact PPC speedup planning update (2026-09-09):

- `docs/ppc_speedup.md` documents the approved exact computational redesign:
  remove duplicate observed work, reuse source-spike interpolation geometry,
  derive whole-epoch results from before/after sufficient statistics, and
  reuse physical edges across overlapping conditions.
- The user approved all recommended implementation-plan decisions in
  `docs/ppc_speedup_plan.md`: preserve a single-job reference executor, add a
  grouped production executor, retain the existing epoch-specific schedules,
  require exact inferential decisions under tight floating tolerances, use a
  2 GiB planned private-peak allocation limit per worker, and defer worker
  profiling until after serial optimization.
- This approval created and reconciled documentation only. It does not
  authorize S0 source changes, synthetic implementation work, another CT026
  engineering profile, the 100-shuffle preview, or the 1,000-shuffle run.
- If implementation is requested later, S0 is first: finish the already-RED
  bounded-spawn worker correction and restore the focused and full neural
  suites to GREEN. S1-S6 then optimize and profile grouped serial work; S7-S8
  rebind and benchmark parallel workers only if still justified.

Exact PPC speedup plan clarification (2026-09-10):

- PPC and representative histograms now share strict canonical-grid equality
  for exact samples; tests cover adjacent representable floats and nearby but
  nonexact values. The legacy histogram-only `isclose` behavior is not retained.
- The 2 GiB gate now applies to a conservative named private-peak estimate,
  including accumulators, PPC draws and scratch, segmented edge work, geometry,
  and gather temporaries. Shared prepared phase is reported separately.
- A frozen `KernelAllocationEstimate` interface accounts for kernel arrays.
  Grouped preflight admits at most 12 GiB of planned arrays across shared mmap,
  the parent, and all active workers, reserving 4 GiB below the measured 16 GiB
  gate for runtime overhead and estimation error.
- Grouped work identity records the base PPC seed and each job's actual derived
  seed, derivation identities, schedule shape, and exact schedule fingerprint.
- S0 uses context-managed normal shutdown and explicit failure cancellation of
  both the failed current future and every submitted pending future.
- To conserve usage, the lead Sol runs at `high`; S0, S5, S6, and S8 use
  `high` writers/reviewers, while S1-S4 and S7 reserve `xhigh` for both the
  Terra writer and independent Sol reviewer. No `max` or `ultra` agent is
  assigned without a later documented revision and explicit approval.
- Existing single-job worker tests become frozen green regressions after S0;
  S7 must establish new grouped-worker RED failures before implementation.

Exact PPC speedup completion and handoff (2026-09-18):

- S0-S8 are complete. S7 implementation is `e89a0ee`; documentation checkpoint
  `8741d71` records the final source gate and 1,113-test neural-suite result.
- The work-only S8 benchmark selected eight workers and preserved scientific
  equivalence under the approved policy. Its accepted evidence directory is
  `ct026_ppc_s8_2026-09-18T15-06-33Z` under the CT026 `analysis_runs` root.
- No 100-shuffle scientific preview or 1,000-shuffle final component has run.
  Both remain explicit standalone launcher invocations, with inspection and
  approval between them.
- The user approved the post-speedup sequence on 2026-09-18: WP10, WP12, the
  standalone launcher, and then the 100-shuffle preview. WP11 follows the
  preview infrastructure. The UI will support either ProbeA or ProbeB as one
  active population per run with the same good/MUA and good-inside-brain
  defaults; combined populations are out of scope.
- WP13 is deferred. Empty thresholds preserve the current result; every
  nonempty request must be rejected before computation until WP13 is complete.
  Deterministic PPC planning may rerun after interruption and need not be
  checkpointed, while compatible phase/PPC work retains exact resume behavior.
- This documentation decision does not itself authorize source changes or a
  CT026 scientific run. Each package still requires its test-first plan/RED
  gate, and the preview remains a separate user-invoked launcher execution.

WP10 documentation-first authorization (2026-09-18):

- The user confirmed the Sol orchestrator is running at high effort and
  approved the bounded WP10 plan, including separate test-only and
  implementation commits, on the condition that all implementation needs be
  durable in documentation before tests or source change.
- The exact composed factory contract, amplitude-threshold guard, progress
  behavior, files, test cases, RED command, performance constraints,
  verification order, and interruption-recovery instructions are recorded in
  the WP10 package above.
- Baseline HEAD before this documentation edit was `8286d15` on `refactor`,
  equal to `origin/refactor`. The tracked worktree was clean. The 171
  pre-existing untracked entries had status-inventory SHA-256
  `140fe2baeec9985753c89efe0fe2d68eb341c625065868e0437ce333e0c6a1a0` and
  remain out of scope.
- The next allowed action after committing this documentation-only checkpoint
  is the WP10 test-only edit. The lead must commit those tests, reproduce the
  intended RED, and record it before touching runtime or pipeline source.
- WP10 authorization does not authorize WP12, the launcher, WP11, WP13, a
  synthetic production-sized profile, or any CT026 computation.

WP10 test-only RED checkpoint (2026-09-18):

- Documentation-first contract commit: `4c88f9a` (`docs: freeze WP10 composed
  runtime contract`). No test or source file changed before this commit.
- Test-only commit: `e3a6757` (`test: define WP10 composed runtime boundary`).
  It changed only `test_lfp_summary_runtime.py` and
  `test_lfp_summary_pipeline.py`; runtime and pipeline source remained
  unchanged.
- The approved RED command selected seven tests. Five existing/new pipeline
  contracts passed. Two composed-runtime tests failed because
  `lfp_summary_runtime.make_lfp_summary_pipeline_dependencies` did not exist.
  The failures were the intended missing-interface RED, not fixture, syntax,
  import-collection, or environment failures.
- The next allowed action is the minimal `lfp_summary_runtime.py`
  implementation specified in WP10. `lfp_summary_pipeline.py` must remain
  unchanged unless a subsequent source run exposes a demonstrated contract gap.

WP10 implementation completion and GREEN handoff (2026-09-18):

- Implementation commit: `98ac2ed` (`feat: compose LFP summary production
  runtime`). It changes only `src/neural_analysis/lfp_summary_runtime.py`;
  `lfp_summary_pipeline.py`, the component-specific dependency factories, all
  numerical kernels, cache schemas, fingerprints, and execution defaults are
  unchanged.
- The new `make_lfp_summary_pipeline_dependencies` factory delegates to the
  existing Power, phase, Spike, payload, manifest, and transaction seams. In a
  Compute All call, Power remains independent while one exact
  `PreparedPhaseRun` object is passed to both Synchrony and Spike phase. The
  progress-aware Spike adapter passes the original callback into the grouped
  PPC payload builder without translating `ProgressEvent` fields or copying
  payload arrays.
- Every composed preparation callable first validates the full configuration
  and rejects nonempty `absolute_amplitude_thresholds` with a WP13-specific
  `ValueError`. The guard runs before trial, LFP, unit, cache, or PPC seams.
  The manifest loader rejects use before preparation and afterward delegates
  with the exact most recently accepted configuration.
- The first source run exposed one local missing import of the existing
  `validate_lfp_summary_config` function. Adding that import was the only
  correction; no test was changed after the test-only commit.
- Focused GREEN command selected the same seven tests: 7 passed and 15 were
  deselected. The complete two-file runtime/pipeline run passed 22 tests. The
  affected Power/Synchrony/Spike/PPC runtime set passed 248 tests with one
  expected duplicate-member ZIP warning.
- The complete `src/tests/neural_analysis` suite passed 1,116 tests with 20
  warning instances in 128.17 seconds. The warnings are the existing fork,
  duplicate test-ZIP member, and Pynapple empty-epoch/divide-by-zero warnings;
  no new warning category appeared. `py_compile`, `git diff --check`, and the
  tracked-worktree scope review also passed.
- WP10 did not open CT026 recordings, run a production-sized profile, create a
  scientific cache, or execute either the 100- or 1,000-shuffle analysis. Its
  remaining integration dependencies are intentionally assigned to WP12, the
  standalone launcher, and WP11. Absolute-amplitude masking remains deferred
  to WP13 and is fail-closed meanwhile.
- WP10 is complete. The next implementation gate is WP12. Before WP12 tests or
  source changes, update this document with its exact report/cleanup interface,
  first-written tests, RED command, performance constraints, and interruption
  recovery checkpoint. The later launcher remains the first authorized
  boundary for a user-invoked 100-shuffle CT026 preview.

WP12 documentation-first authorization (2026-09-18):

- The user confirmed that all WP10 commits were pushed and authorized continued
  WP12 planning. Baseline commit `3aebba8` equals `origin/refactor`; the tracked
  worktree is clean and the pre-existing untracked inventory hash is unchanged.
- Inspection found that `selected_low_unit_ids` and
  `selected_high_unit_ids` are independently cached, but the single
  `illustrative_trial_indices` array stores the high unit's trial first. The
  user approved adding exact low/high trial-index arrays instead of displaying
  a mismatched trial, reopening raw data, or approximately reconstructing an
  excluded-trial set from incomplete cache masks.
- The user approved a bounded primary report rather than 162 exhaustive paired
  exemplar figures: `correct_rewarded`, all three configured sites,
  before/after epochs, and theta/gamma bands produce 12 paired figures. Together
  with six unit maps, six population maps, and three per-site band summaries,
  the report has an upper bound of 27 PNGs.
- The user approved launcher-owned final cleanup. WP12 adds opt-in cleanup
  deferral and returns the exact action only after report publication; WP12 does
  not execute it. Existing pipeline callers retain immediate cleanup by
  default.
- The exact additive schema, plotting records, measurement fields, report
  schema/artifacts, staged validation, cleanup interface, first-written tests,
  performance constraints, RED command, verification order, and recovery
  instructions are recorded in the WP12 package above. This documentation must
  be committed before a WP12 test or source file changes.
- The next allowed action after the documentation commit is the WP12 tests-only
  edit and commit. It must reproduce the documented RED before implementation.
  No CT026 scientific computation or launcher implementation is authorized.

WP12 tests-only RED checkpoint (2026-09-18):

- Documentation-first contract commit: `5dfd439` (`docs: freeze WP12 report
  and cleanup contract`). It changed only this task document.
- Tests-only commit: `2c80a9b` (`test: define WP12 report publication
  contract`). It changed only the five listed plotting, validation, payload,
  runtime, and pipeline test files; production source remained unchanged.
- The documented focused RED command selected 12 tests: six existing contracts
  passed and six new contracts failed. Failures were the absent
  `PPCExemplarPanel`, `SpikePhaseFilterBenchmark`, detailed report transaction
  helpers, and `ComponentRunResult.deferred_cleanup`; there was no collection,
  fixture, syntax, or environment failure.
- The auxiliary payload/runtime/pipeline RED command selected four new or
  directly extended tests. All four failed for the intended missing additive
  trial arrays and missing `defer_post_commit_cleanup` /
  `defer_spike_phase_cleanup` keywords. Thirty-four unrelated tests were
  deselected.
- The next allowed action is the minimal WP12 source implementation recorded
  above. Tests must not change unless an approved requirement changes or a
  demonstrated test defect is documented first. No CT026 computation or
  launcher work is authorized.

WP12 completion handoff (2026-09-18):

- The focused GREEN run exposed one stale pre-WP12 assertion that still
  required the legacy single-exemplar report seam even though the frozen WP12
  contract replaced that report selection with paired exemplars. Correction
  commit `7d9c322` changes only that assertion to require
  `exemplar_pair`; the existing public single-exemplar plot and its independent
  tests remain unchanged. The synthetic integration fixture also manually
  constructed the old one-trial-only payload despite the approved additive
  schema. Correction commit `220f5b0` adds both exact trial arrays to that
  fixture and changes no production behavior.
- Implementation commit `bd6fa50` (`feat: complete WP12 PPC reporting`) changes
  exactly the five authorized source files. It adds the exact low/high cached
  trial-table rows while retaining the legacy high-first alias, the frozen
  low/high plotting record and paired figure, the fixed 27-PNG maximum report
  planner, detailed scalar measurement/projection records, strict staged
  JSON/log/Markdown/PNG validation and atomic publication, and opt-in exact
  post-commit cleanup deferral. Default pipeline callers still clean
  immediately; WP12 never invokes a deferred callback.
- The computed production preview adapter opts into cleanup deferral. A callback
  becomes accessible only after both staged and published artifact validation.
  Serialization, plotting, validation, publication, or post-publication
  validation failure exposes no callback. A post-publication validation failure
  moves the newly created final directory back under the known staging path so
  the report transaction is removed rather than left partially accepted.
- Report rendering loads the validated component once, passes the original
  cache mapping to plot seams, forms only site/epoch or exemplar-sized views,
  and saves/closes each figure before continuing. The population map no longer
  requests a dtype conversion of the full PPC tensor. Reporting imports no UI
  or launcher module and opens no raw LFP, sorter, or trial source.
- The documented focused command is GREEN at 12 passed and 11 deselected. The
  auxiliary schema/runtime/pipeline command is GREEN at 4 passed and 34
  deselected. All five directly affected test modules are GREEN at 61 passed,
  and synthetic cache/write/reload integration is GREEN at 2 passed.
- The complete `src/tests/neural_analysis` suite is GREEN at 1,125 passed with
  20 warning instances in 192.95 seconds. Those are three existing
  multi-threaded-fork deprecation warnings, one intentional duplicate-NPZ-member
  warning, and 16 existing Pynapple warnings from zero-duration synthetic
  epochs. There were no WP12 failures, xfails, skips, network requests, raw
  CT026 reads, or production-sized computations.
- `git diff --check` was clean before the implementation commit. The tracked
  file scope was the five authorized source files plus the two documented test
  fixture/assertion corrections. The 171-entry normal untracked inventory is
  unchanged at NUL-delimited status SHA-256
  `140fe2baeec9985753c89efe0fe2d68eb341c625065868e0437ce333e0c6a1a0`.
- Remaining launcher risks are intentionally unresolved here. The launcher must
  record its timestamped run identity and resume command before planning,
  supply real planner/grouped/process-tree/intermediate-size measurements when
  observable rather than replacing `None`, reject nonempty amplitude thresholds,
  support exactly one ProbeA or ProbeB population per run, validate its own
  state/log in addition to the WP12 report, and only then invoke the returned
  exact-fingerprint cleanup callback. Rerunning deterministic planning after an
  interruption remains approved. No combined-probe run is requested.
- WP12 completion authorizes planning the standalone launcher as a new
  documentation-first, test-first package. It does not authorize a CT026 dry
  run, the 100-shuffle preview, the 1,000-shuffle final run, WP11, or WP13.
