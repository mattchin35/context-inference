# LFP Summary Analysis Implementation Plan

## Live handoff snapshot

- **Snapshot:** 2026-09-08 on branch `refactor`.
- **Current implementation HEAD:** `91ecc7ce6f101143c6b2150bffba38384c02b566`
  (`91ecc7c`, `reuse scheduled PPC edge samples`).
- **Upstream relationship at this implementation checkpoint:** local `refactor`
  contains the WP5C-2 commits after `origin/refactor` at `1f0bffa`.
- **Verified neural baseline at this HEAD:** 631 passed, no skipped or xfailed
  tests, with 16 known Pynapple empty-epoch/divide-by-zero warnings from
  `UV_CACHE_DIR=/tmp/context_inference_uv_cache MPLCONFIGDIR=/tmp/context_inference_mpl uv run
  pytest -q -p no:cacheprovider src/tests/neural_analysis`.
- **Approval state:** the user approved the Synchrony report dated
  `2026-08-26T16-57-33Z` and approved completing WP5C before the CT026
  100-shuffle Spike-phase preview. The WP5C-0 contracts are approved and
  WP5C-1 and WP5C-2 are complete. WP5C-3 is the next implementation package;
  no CT026 Spike-phase computation has been authorized or run.
- **Do not redo completed packages:** WP0, WP1, WP2 preparation, WP3, WP4,
  WP5A, WP5B, and the Spike-phase runtime bridge are historical completed work.
  Remaining integration work must extend them through the packages below, not
  reassign or reimplement them.

Worktree warning at the snapshot: preserve the user's existing modifications to
`docs/Tasks_neural.md`, `docs/DevJournal.md`, and `docs/webappDesign.md`; the
added `docs/spec_neural_regression.md`; and the staged deletion of
`docs/DevLog.md`. Preserve all unrelated untracked files and directories. Agents
must stage only files assigned to their package and must stop on unexpected
changes.

Source-of-truth hierarchy:

1. `docs/Tasks_neural.md` owns current status, frozen contracts, gates, package
   ownership, and remaining work.
2. `docs/webappDesign.md` owns scientific and user-interface requirements.
3. `docs/DevJournal.md` is a historical record only and is not authoritative
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
| Spike runtime bridge | Complete | Tests `453d08a`; implementation `c8e76d5`; `test_lfp_summary_runtime.py` | WP5C optimization; no CT026 Spike-phase execution has occurred |
| WP5C optimization | WP5C-0 approved; WP5C-1 and WP5C-2 complete | WP5C-2 tests `ce4e3f3`, `b02da63`, `6510ca2`, `32e6955`; implementations `88e9846`, `91ecc7c`; focused 9 passed, Spike/LFP 34 passed, full neural suite 631 passed with 16 known warnings | Implement WP5C-3 prepared-phase persistent cache test-first; WP5C-4 through WP5C-6 remain pending |
| WP6 plotting | Partially complete | Tests `b1f51f3` and later focused plotting tests; implementation `d1c3e21`; Power and Synchrony reports above | Complete PPC exemplars and reporting in WP12 |
| WP7 pipeline | Partially complete | Tests `b1f51f3`; implementation `067fdef`; `test_lfp_summary_pipeline.py` | Composed production dependencies and detailed progress in WP10 |
| WP8 webapp | Partial; Power path only | Tests `c7ec7c1`, `003394a`, `f3954de`; implementations `2885ffd`, `926801b`; `test_lfp_summary_webapp.py` | Synchrony, Spike phase, Compute All, active population, progress, and complete cached views in WP11 |
| WP9 validation | Partial | Approved Power and Synchrony reports above; Spike preview runner tests `7223072`, `0dfc1a8`, `1c2fb61`, `b9aecbe`; implementation `c3f6d41`, `c036d62` | WP5C benchmark, 100-shuffle preview, user approval, then separately authorized 1,000-shuffle run |
| WP10 composed dependencies | Planned | Package below | Build after WP5C contracts and serial runtime stabilize |
| WP11 Streamlit integration | Planned | Package below | Build after WP10 |
| WP12 PPC plotting/reporting | Planned | Package below | Complete before final CT026 interpretation |
| WP13 amplitude thresholds | Planned | Package below; current validation/fingerprint support exists but production application does not | Implement test-first or explicitly defer before broader production use |

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
  masking are saved.
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
contracts in Sections 2.22-2.26 are approved. WP5C-1 and WP5C-2 are complete,
and WP5C-3 is the next authorized implementation package. Implementers must
escalate newly discovered ambiguity instead of choosing new scientific or execution defaults.
Material changes to metrics, thresholds, cache contracts, or user-visible
behavior require user approval and an update to this plan before implementation
continues. No CT026 Spike-phase computation is authorized by these package
approvals.

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

#### WP5C-6 - CT026 preview validation

Owner: Sol after WP5C-5's serial/parallel decision and WP12's complete preview
report path are green.

- Run only the approved active ProbeB population with exactly 100 shuffles.
- Record cold and warm runs, stage timing, peak memory, throughput, final and
  intermediate cache sizes, unit/trial/spike counts, warnings, and exclusions.
- Generate the complete preview plots/report required by WP12.
- Pause for user inspection and approval.
- Do not run 1,000 shuffles automatically or as an implicit merge consequence.

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

Owner: one implementer after WP5C-5's serial/parallel decision; exclusive owner of runtime/pipeline
integration files while active.

Files:

- `lfp_summary_runtime.py`, `lfp_summary_pipeline.py`, and focused tests

Tests written first:

- One production dependency bundle supports Power, Synchrony, Spike phase, and
  Compute All.
- Compatible phase preparation is shared once.
- Component-specific failures remain isolated and do not invalidate successful
  compatible siblings.
- No component-specific factory silently handles unsupported work.
- Progress remains component/stage specific and final manifests remain the
  component commit points.

### WP11 - Full Streamlit integration

Owner: one implementer after WP10; exclusive owner of
`lfp_summary_webapp.py`, `psth_webapp.py`, and their focused tests.

Tests written first:

- The active unit population is selected and passed explicitly.
- Synchrony and Spike-phase actions and Compute All call only the composed
  production boundary.
- Progress stages, counts, elapsed time, and optional ETA render without
  numerical computation inside Streamlit.
- Compatible/stale/missing/failing component states and fingerprint differences
  are inspectable.
- All cached selectors/plots load only final compatible component files.
- Prepared-phase caches, checkpoints, and other work artifacts never appear as
  compatible scientific components.

### WP12 - Complete PPC plotting and reporting

Owner: one implementer; plotting/report files only and no ownership overlap with
WP11.

Tests written first:

- The high/low 5th/95th percentile exemplar set is deterministic.
- Pooled PPC and the illustrative median-spike-count trial are visually and
  textually distinct.
- Reports contain a detailed run log, warning/exclusion summary, reliability
  and eligibility counts, and stage performance/cache measurements.
- The nine-filter runtime/storage projection is present and traceable to the
  representative benchmark.
- Every figure has readable labels, an opaque selected background, and a
  caption summarizing scientific content and any statistical test.

### WP13 - Apply configured absolute amplitude thresholds

Owner: one implementer after WP5C prepared-phase contracts are frozen.

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

Until WP13 is complete, production controls and reports must display a visible
warning that configured absolute amplitude thresholds are validated and
fingerprinted but not applied. Broader production use must either complete WP13
or explicitly defer it with user approval; silent omission is forbidden.

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
6. WP5C-5 profiles the serial implementation. Parallel execution is added only
   if profiling justifies it and only after worker-invariance tests record RED.
7. WP12 completes the preview report path; Sol then runs WP5C-6, the CT026
   100-shuffle preview, and pauses for inspection.
   No 1,000-shuffle run follows automatically.
8. WP10-WP13 proceed only under the file ownership declared by each package.
   Parallel work is allowed only for disjoint files after shared interfaces are
   frozen.
9. No CT026 computation is an implicit consequence of a merge, test run, or
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
- Profile the serial WP5C kernel before activating within-session parallelism,
  then benchmark 1, 2, 4, and 8 shared-data workers. Select the smallest worker
  count near the measured throughput plateau and keep peak PPC working memory
  below 16 GiB.
- Treat 25-60 minutes for a cold 100-shuffle CT026 preview and 45-120 minutes
  for a cold 1,000-shuffle run as provisional engineering estimates with about
  twofold uncertainty until the representative benchmark is complete. A valid
  prepared-phase cache is expected to remove approximately 8-10 minutes from a
  repeat run, but the actual measured saving must replace this estimate in the
  validation report.
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
