# Session Timeline Phase Synchrony and Band Power: Implementation Handoff

Status: Reviewed live draft, 2026-11-04, updated after implementation-chat feedback.
Native-rate Welch and cold-run primary-first publication are now explicit user
decisions. Phase preparation also requires drift-aware UTC waveform alignment
before Morlet and an explicit numerical amplitude threshold of 0.0.
Implementation approved by the user in the receiving chat on 2026-11-04.
Round 1 implementation and synthetic/designated-session validation may proceed
through the gates below; additional experimental sessions still require user
inspection. Historical documentation-only evidence is retained in Section 10.

Scientific authority: [spec](session_timeline_phase_synchrony_spec_v1.md).
Repository rules: [AGENTS.md](../AGENTS.md), [SoftwareDesign.md](SoftwareDesign.md).
This file owns delivery order, assignments, evidence, and the next action.
Maintain it during implementation, following the existing
[prediction handoff](condition_prediction_generalization_implementation_plan.md#0-live-implementation-handoff)
and [single-unit assignments](single_unit_plan.md#10-orchestrator-and-worker-assignments).

## 0. Live implementation handoff

Update the snapshot/ledger at RED, GREEN, review, and data-inspection boundaries
and before handoff. Append exact evidence to Section 10.

### 0.1 Current snapshot

| Field | Current value |
| --- | --- |
| Date / branch / inspected HEAD | 2026-11-04 / `refactor` / `8f9443f` (WP5b source). Recheck; never reset to this observation. |
| Package / authority | WP6b corrected GREEN; WP7a browser next. Full Round 1 authorized by user: "alright, then begin implementation". |
| Team / running work | Lead runtime model/effort not exposed; scientific worker `gpt-6.1-sol` / `high`, agent `/root/scientific_worker`, stopped after read-only UI review. Lead owns integration/browser. No experimental jobs/browser server yet. |
| Feature files | WP1-WP5b source/tests and WP6a tests committed. Lead current-CLI usage additions in SESSION_TIMELINE.md/README.md are uncommitted. Scientific spec remains user-staged with unstaged refinements; HEAD alone omits it. |
| Other dirty work | Staged PCA spec, single-unit plan, and ridge spec; unstaged `src/main.py` and `src/neural_analysis/sync_ephys.py`; other untracked files. Preserve all of it. |
| Evidence | WP1-WP4 RED/GREEN/regressions and commits in Section 10; WP4 235 passing checks. No experimental signal arrays loaded or feature runtime benchmarks yet. Metadata confirms selected unit eligibility. |
| Next action | Review/commit WP6b source, inspect full seeded session in browser, then CT026 validation if accepted. CT026 uses `ProbeB:8` with `HPC1`; optional additional-unit timing question pending, no other units inferred. |

Approval of the full Round 1 sequence includes its synthetic and designated
single-session validation unless the user limits that approval. Record scope
once; do not add approval gates at every package. User inspection of CT026
output still precedes additional experimental sessions.

### 0.2 Package ledger

Assignments are planned. Advance a gate only after Sol inspects the actual
diff and evidence. Replace `not run` with commands/results/commits when obtained.

| Package | Planned owner | State / next gate | RED commit / command / result | GREEN / review evidence |
| --- | --- | --- | --- | --- |
| WP0 spec/handoff | Lead | Implementation approved | Not applicable | Section 10 doc checks |
| WP1 inputs/config/preparation contracts | Sol worker, high | GREEN reviewed; complete | `0726ee1`; fixture/acceptance follow-up `ef27189` | 44 feature tests passed; 44 metadata/loading regressions passed; source reviewed by lead |
| WP2 shared PSD/power/coherence | Sol worker, high | GREEN reviewed; complete | `705f8d8`; 1 expected missing-feature collection error, 0.52 s | Worker 107 passed; lead 96 plus 11 runtime regressions; source reviewed |
| WP3 phase/LFP metrics/spike PPC | Sol worker, high | GREEN reviewed; complete | `14fc7ea`, `cd046db`; review RED `87952d2` (6 failures) | 165 passed, no warnings; lead reviewed and reproduced |
| WP4 persistence/discovery | Sol worker, high | GREEN reviewed; complete | `6f8650f`; review RED `d867a0e` (10), `76bde37` (3) | 235 passed; lead reproduced and reviewed all source |
| WP5a execution/timing core | Sol worker, high | GREEN reviewed; complete | `11284ad`; missing-runner RED; `109986e`, 7 input/reporting failures | 20 runner tests; lead 256 combined checks passed; source reviewed |
| WP5b thin CLI/batch | Lead; Sol worker read-only review | GREEN reviewed; complete | `b800756`; fixture `6e11de7`; review `987fab4`; process context `db4aea1` | 196 combined; final 12 focused passed without warnings |
| WP6a scientific figures | Sol worker, high | GREEN reviewed; complete | `0e3f57f`; review `e929a1d` (5); label `8063a0d` (1) | 17 plotting tests; lead 247 combined; PNGs visually reviewed |
| WP6b route/widgets/Refresh | Lead; worker read-only review | Corrected GREEN; source commit next | `f4bfd03`; fixture `721307e`; selection `6fa2d40` (3); status `d099477` (3); payload `5fee607` (1) | 12 UI tests passed; browser next |
| WP7a synthetic/regression/browser audit | Lead | Seeded full pipeline passed; browser pending | Existing features already unit-TDD verified; acceptance tests `697a954` | 2 seeded integration tests passed, 3.67 s |
| WP7b CT026 validation/timings | Sol lead | Not started; user inspects output | Not applicable unless correction needed | Not run |
| WP8 usage/final handoff | Lead; optional Luna for literal docs | Not started; Round 1 acceptance | Not applicable for docs only | Not run |

### 0.3 Resume procedure and first commands

1. Read this plan, spec, repository rules, and latest Section 10 entry. Check
   implementation authority and actual branch/HEAD/dirty state.
2. Inspect recorded agent IDs and running commands before replacing them.
   Preserve completed work and resume the unfinished gate. Stop for genuinely
   unexpected changes; do not reset the checkout or rediscover settled scope.
3. Record package, phase, model/effort, exact writable paths, and focused check
   before dispatch. Respect the single-worker and lead-commit rules in Section 9.
4. Before stopping, record commands/results/commits and any tool session/PID,
   log/output path, last durable product, and exact next action.

After approval, run a bounded existing baseline only if evidence is missing or
relevant edits invalidate it. Use `uv run --frozen --no-sync pytest` with actual
files covering affected metadata, LFP loading/power/phase/synchrony,
spike-phase/PPC, and timeline/chart behavior. The first WP1 action is writing
its tests and establishing meaningful RED, not invoking a nonexistent path.

## 1. Scope, inputs, and configuration

Round 1: a separate saved-result timeline with per-site integrated theta/gamma
power, pairwise magnitude-squared coherence and LFP PLV/PPC, and explicitly
selected-unit spike-LFP PPC. Primary `[-1,+1)` s; optional secondary `[-2,+2)` s,
shared across metrics. One estimate per choice; no across-trial pooling.
Before `[-2,0)` / after `[0,+2)` s is Round 2 after Round 1 and measured timings.
Moving windows, all-unit processing, significance, and cluster deployment
are outside Round 1.

Preserve the spec's Morlet 2-100 Hz / 2-Hz spacing on the 500-Hz phase grid,
theta 6-10 Hz, gamma 30-80 Hz excluding 58-62 Hz, enabled notch, complete-support
policy, and matched 0.5-s/50%-overlap periodic-Hann Welch settings. The spec owns
full definitions and interpretation; do not substitute convenient helper defaults.

Verified validation metadata (read-only inspection, not a readiness/analysis run):

`/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260828_latent_inference/neural_session.json`

- Session ID `CT026_2026-08-28_134630`; trial/event files have session-relative
  paths. Computational timestamps are UTC seconds.
- PFC1 = ProbeA saved channel 100; HPC1/HPC2 = ProbeB saved channels 222/14.
  Metadata pairs: HPC1-HPC2, PFC1-HPC1, PFC1-HPC2. These are validation facts,
  not hard-coded channel defaults for other sessions.
- Both LFP sidecars declare 2500 Hz and affine calibration to uV (gain about
  0.195, offset 0). Stored float32 values still require the existing loader's
  calibration. Prior phase-shift/bandpass/resampling is recorded; do not repeat
  it. Recheck metadata, paths, source identities, and synchronization at run time.

Use small immutable config/result records, not new metadata schema fields.
CLI defaults: primary interval; families `power coherence lfp-phase`; available
metadata sites/ordered pairs (spec fallback if pairs absent). `spike-ppc` requires
explicit probe-qualified unit IDs and LFP site. Quality selection follows the
existing interface. Missing inputs disable only dependent work. Validate each
requested component rather than requiring all probes, spike files, or pairs.

## 2. Architecture and verified reuse boundaries

Proposed new modules under `src/neural_analysis/phase_synchrony/`; ordinary
functions, with helpers local to the owning module. These are planned project
interfaces, not assumed library APIs.

| Module | Responsibility / planned functions | Package |
| --- | --- | --- |
| `results.py` | Config/result records, component identities, NPZ save/load/discovery | WP1/WP4 |
| `preparation.py` | Independent calibrated site/unit loading, shared UTC-grid waveform sampler, continuous preprocessing, `prepare_site_lfp` / `prepare_site_phase`, compatible preparation reuse | WP1/WP3/WP4 |
| `computation.py` | `compute_site_psd`, `compute_band_power`, `compute_welch_coherence`, `compute_lfp_phase_summary`, `compute_selected_unit_ppc`, strict band reductions | WP2/WP3 |
| `plotting.py` | Plotly timeline figures and Matplotlib validation PNG export from the same saved products; `export_validation_figures` | WP6a |
| `run_session.py` / `run_batch.py` | Execution/timing core first; thin CLI and session-list wrapper second | WP5a/WP5b |
| `webapp/phase_synchrony_view.py` | Separate route, selectors, saved discovery, Refresh, timing readout; localized existing app wiring | WP6b |

```mermaid
flowchart TD
    B[Metadata and behavior] --> V[Timeline viewer]
    L[Calibrated synchronized site LFP] --> A[Welch auto-PSDs]
    A --> W[Site band power]
    L --> X[Pair cross-spectra]
    A --> C[Coherence]
    X --> C
    L --> P[Site Morlet phase]
    P --> F[Pair PLV and PPC]
    P --> S[Selected-unit PPC]
    U[Selected-unit spikes] --> S
    W --> V
    C --> V
    F --> V
    S --> V
```

Reuse narrowly, after reading callers and verifying installed APIs:

- Behavior/clock: `prepare_timeline_choices` and `load_timeline_behavior` retain
  trial IDs/rows, UTC choice times, first-trial elapsed origin, and action codes.
  `add_timeline_behavior`, `render_timeline_navigation`, and
  `render_timeline_chart` can work with a behavior-only `SessionTimelineResult`
  (empty regions); no PCA fit is needed.
- Phase: `lfp/phase.py` continuous Morlet/notch/decimation is the reference.
  Verify the actual output rate: integer decimation need not yield exactly
  500 Hz for arbitrary sources. Reject incompatible caches/configurations
  rather than relabeling their samples. The production phase block loader in
  `lfp_summary/synchrony_runtime.py` anchors once and uses nominal-rate times;
  it cannot supply drift-aware phase unchanged. Use the shared UTC-grid sampler
  before decimation/Morlet, as specified in Section 3 and spec Section 4.2.
- `make_phase_trial_tensor` can interpolate across invalid source samples;
  `compute_site_phase_trial_tensor` can reject an entire padded block on a
  source/read error. Adapt narrowly so masks/gaps stay invalid and valid primary
  windows survive missing secondary support or an unrelated boundary trial.
  Record the real Morlet padding/support; do not create an edge-correction study.
- `compute_wavelet_coefficients` retains an unprocessed source trace. Do not
  reuse that trace as an already notched/decimated waveform. Avoid the full
  `lfp_summary` workflow: even warm `prepare_phase_run` reloads source traces;
  its payload adds unrequested across-trial analyses/bootstraps.
- `compute_trial_plv_by_frequency` defaults to 80% coverage and finite-only band
  averaging elsewhere; enforce complete window support and fixed retained
  frequencies. Publish PLV/PPC together from the same phase sums/counts.
- Spike sampling: `sample_wavelet_phase_at_spikes` expects coefficient results,
  not a cached unit-phase tensor. Reuse the existing complex interpolation/PPC
  core via a narrow public adapter if needed; never fabricate amplitudes to
  satisfy that API. Interpolate normalized complex phase vectors and normalize
  the result, matching that sampler; amplitude gates validity without weighting
  the phase interpolation. Match the exact-sample validity rule in `spike_lfp/ppc.py`
  / `ppc_kernel.py`: exact samples use their own validity; interpolation needs
  both valid neighbors, no extrapolation. Keep real right-neighbor support
  beyond the half-open metric window and record excluded spikes.
- Pin/save `numerical_amplitude_threshold=0.0`, matching `PhaseAnalysisConfig`.
  Pass `minimum_relative_magnitude=0.0` explicitly wherever a normalization or
  phase-sampling helper accepts it; never inherit its generic `1e-12` default.
  Absolute amplitude gating is disabled/0.0 by default. Positive finite
  coefficients below that generic relative cutoff remain eligible; zero,
  nonfinite, and undefined interpolants remain invalid. No new downstream cutoff.
- PSD/power: `lfp/power.py::compute_welch_psd` must have per-call notch disabled
  after continuous notch. Preserve actual Welch coordinates rather than
  silently interpolating them. `mean_band_power_linear` returns mean PSD
  density plus bandwidth: multiply them for total power, retaining its
  disjoint-interval/one-sided boundary quadrature.
- Coherence needs matched cross spectra as well as auto-PSDs. Share auto-PSDs
  only for identical samples, segment placements, preprocessing, units, and
  Welch settings. Do not invoke full Power baseline/dB/before-after workflows.
- Existing app readiness checks can require all probes/spikes/pairs. Keep the
  new route component-specific and able to load saved neural results without
  reopening raw binaries or requiring unrelated inputs.

Do not add a workflow engine, registry, job service, or broad refactor.

## 3. Data and inspection contracts

Every new function/record documents types, shapes/axes, units, and returns.
Keep UTC seconds internally and one first-trial elapsed origin for display.
Use explicit absolute-time tolerances (or relative coordinates), not default
relative `allclose` on billion-second UTC values.

Welch uses native-rate waveforms (2500 Hz for CT026), not the 500-Hz phase grid.
Follow spec Section 6.2.1: map each probe's source samples to choice-relative
time using its own synchronization. For `r = -a + arange(round(2*a*fs))/fs`,
evaluate each probe's IRIG UTC-to-native-index map at every `choice+r`, then
linearly sample continuously preprocessed real voltages at those fractional
indices. A nominal choice-anchored loader time vector alone is insufficient
when probe clocks drift. Bracketing loads/filter context
are support only; exact samples use their own validity, interpolation needs
two contiguous valid neighbors. Never pass extra loader samples into Welch.
One site's aligned row/segments must be identical for power and coherence.
Pair native-rate mismatch disables that coherence pair without implicit
resampling or disabling independent power.

Phase uses that same IRIG-to-native-index sampler on padded continuous blocks,
before decimation/Morlet (spec Section 4.2). Native calibration/notch -> uniform
UTC native-rate waveform -> existing anti-alias decimation/Morlet -> unit phase
-> trial/spike sampling on the corrected UTC time base. Disable the helper's
notch when already applied, while preserving actual notch provenance. Use
block-relative seconds plus saved UTC origin for precision, retaining real
padding/validity. Do not transform an irregular corrected time vector while
declaring it uniform, or relabel old nominal-block coefficients afterward.
Trial targets are `choice+r` on the 500-Hz phase grid; selected spikes retain
their already synchronized UTC timestamps, with no second LFP clock correction.
Matching phase-coordinate arrays alone do not establish recovered phase values.

| Product | Axes / required content |
| --- | --- |
| Site waveform | `(trial, time)`, calibrated voltage, actual coordinates/validity and preprocessing identity |
| Site phase | complex64 `(frequency, trial, time)` plus bool validity, actual grid and interpolation guard support; document any removed leading site axis |
| Site auto-PSD | `(trial, frequency)` in uV^2/Hz for calibrated uV; actual Welch grid/settings and per-trial sample/segment counts |
| Integrated power | `band_power_integrated` on `(trial, band)`, uV^2; `(band,)` retained bandwidths; source PSD reference, validity/reasons |
| Pair / unit-channel metrics | `(trial, frequency)` values/counts and `(trial, band)` summaries; distinguish phase and Welch frequency axes |

All products retain original trial IDs/row positions, exact interval bounds,
site/probe/channel/unit IDs, source/configuration identity, and exclusions.
Require 1,000/2,000 primary/secondary metric phase samples at 500 Hz, and
5000/10000 waveform samples for CT026 at 2500 Hz, yielding seven/fifteen Welch
segments. Invalid secondary support must not discard valid primary data.
LFP support is complete; spike PPC retains valid spikes with per-frequency
counts/exclusions. Band spike-count labels show the min-max over retained
frequencies (or explicitly common N), with exact per-frequency counts available;
never pool frequency counts. No new 50-spike cutoff. Keep negative PPC.

Power integrates frequency, not time: constant PSD density q gives theta
`4*q` and retained gamma `46*q`. Genuine zero power is valid even if phase or
coherence is undefined. Missing retained support is unavailable, not zero.

Persist enough for the spec's selected-choice inspection: prepared calibrated
waveforms, identified frequency/phase data and validity (or durable references
to compatible preparation), selected spike timestamps, and fixed-bin per-trial/
frequency polar histogram counts/edges derived with the same sampling rule.
The viewer must not reopen raw LFP/spike sources or rerun transforms/PPC to
draw inspection plots. Avoid saving dense spike-by-frequency arrays when
histograms plus timestamps suffice.

## 4. Storage, reuse, and execution

Reusable preparation belongs under the original session's
`processed/session_timeline_phase_synchrony/`; retain existing compatible mmap
phase storage where suitable. Each invocation creates a new session-local run:

```text
analysis_runs/session_timeline_phase_synchrony_<date>_<time>_<run_id>/
  config.json, run.log, summary.md, measurements.json
  run_session.py, run_batch.py
  components/<family>/<selection>_<interval>.npz
```

Save script copies and named arrays with dictionary-valued `meta`: generator,
analysis/schema versions, effective parameters, source/preparation IDs, seed
if applicable, units/axes, timings, UTC completion time, and run ID.
Experimental files stay outside Git.

One immutable, atomically renamed NPZ is one completed component for all trials
of one site/pair/unit-channel and interval. Temporary/invalid/incompatible
files are ignored. No separate per-NPZ completion receipt is needed; existing
multi-file preparation caches retain their completion-marker contract.
Persist measurements after each completed/failed stage. Explicit reruns create
new runs and never overwrite historical results.

Allow **one preparation writer per session** plus read-only viewers; reject
or clearly report another active writer using a simple existing lock pattern.
Separate component commands run sequentially, including after an interruption;
no concurrent-writer cache coordination is required. Recover a stale lock only
after checking its recorded process. Across-session batch processes are separate.

Discovery can combine completed products across runs when relevant identities
match. For duplicate component identities, select the maximum tuple of parsed
UTC completion time, run ID, and relative product path (lexicographic string
ties). Show the selected run/time in details. Newer failed/unfinished
attempts do not displace an older compatible completed product; report both
the displayed source and newer attempt status. Ignore filesystem mtime/order.
A new unit/display selection must not invalidate site power/coherence.
PSD identity includes exact samples/segments, preprocessing, units, and Welch
settings plus waveform alignment/synchronization semantics; band-integration
changes alone do not invalidate it. Phase reuse requires matching coordinates,
support/guards, validity, transform settings, and the scientific marker
`phase_support_semantics="strict_support_v1"`, plus
`phase_alignment_semantics="irig_uniform_utc_pre_morlet_v1"`, actual IRIG/offset
identity, block UTC origin/bounds, grid rates, operation order, and effective
numerical/absolute thresholds. Reject legacy phase preparations and derived
products with missing/wrong alignment semantics or incompatible thresholds,
even if marked `strict_support_v1` with all-valid masks. Recompute; do not
relabel legacy data.
Independent power/coherence remain reusable. No viewport cache keys or general
provenance/DAG system.

The runner exposes metric families, sites/pairs/units, primary/secondary
intervals, preparation-only LFP/phase actions, explicit order, dry-run, reuse,
and additive rerun. Dry-run reads metadata/sidecars and lists missing/reusable
work without loading numerical data or creating outputs. Example proposed CLI
(not available yet; defaults in Section 1):

```bash
uv run -m src.neural_analysis.phase_synchrony.run_session \
  --session-metadata /path/to/neural_session.json \
  --components power --sites SITE --intervals primary secondary

uv run -m src.neural_analysis.phase_synchrony.run_session \
  --session-metadata /path/to/neural_session.json \
  --components spike-ppc --unit PROBE:CLUSTER --lfp-site SITE --intervals primary
```

Component-only work runs only missing prerequisites. Power needs one site, no
pairs/phase/spikes/cross spectra; coherence needs no power integrals/phase/spikes;
spike PPC needs no pairwise metrics. Publish each primary product immediately,
before secondary/other selections finish. When both intervals are requested,
run a primary sweep before secondary-only work. Within each sweep, provisionally
run power, coherence, LFP phase, then spike PPC (or explicit family order);
use measurements to recommend the eventual order.
Primary-first latency takes precedence over avoiding repeated phase work.
Reuse existing wider support or support obtained by the same required transform;
otherwise prepare/publish primary first and allow additional secondary
preparation. Record extra transforms/support/time and actual reuse. Do not
recompute or replace a completed compatible primary product solely to prepare
secondary results. This is not a promise of one transform for both cold-run
intervals, nor a requirement for a new partial-transform checkpoint system.

On failure, save the error, skip dependents, and continue independent work;
return nonzero when requested work failed. Invalid trial support is a data
status, distinct from execution failure. Completed results remain usable.

## 5. Viewer and outputs

Reuse behavior/navigation with a separate route and light mode. Load behavior
tables plus compatible saved neural products; a standalone archive viewer
without behavior files is not required. Missing/failed/unfinished components
are labeled. Manual Refresh rediscovers completed products without numerical
preparation or losing the panned range; do not cache absence indefinitely.
No app background jobs, polling service, or automatic scheduling.

Show choice-time markers and optional frequency-by-choice heatmaps with true
interval bounds and blank inter-trial gaps; no lines or stretched cells bridge
choices. Include PLV/PPC toggle, separate coherence and power labels, selected
unit counts, and selected-choice LFP/phase/spike/polar inspection. Preserve
actual time, source identities, interval, frequency, support, and units in
hover/details. Follow the spec's captions and scientific interpretation notes.

Stable scales while panning: PLV/coherence 0-1; signed PPC retaining negatives;
separate nonnegative physical-unit power axes per site/band/configuration.
Reuse native chart persistence and explicit viewport controls; no new pan
callbacks. Keep saved values invariant under navigation/display changes.
Keep Plotly for the interactive viewer. WP6a implements
`export_validation_figures` with existing Matplotlib/Agg; WP7b calls it on saved
products to write captioned, opaque-white PNGs under the run's `figures/`
directory (timeline overview and selected-choice/frequency inspection).
This export does not require Kaleido or browser downloads. Verify the saved
images visually; time export separately from numerical preparation.

## 6. Timings and measured processing order

Use exclusive monotonic wall timers. Charge shared prerequisites once and
report reused work explicitly. Required useful boundaries:

| Stage | Timing separation |
| --- | --- |
| Input setup | Metadata/behavior/sidecar validation vs numerical data loading |
| LFP preparation | Per-site I/O/synchronization vs continuous preprocessing/UTC-grid sampling, including padded phase blocks; cache validation/load vs cold work |
| Morlet preparation | Transform/extraction vs LFP phase reductions; record support/frequencies, reused support, and any extra secondary preparation |
| Welch auto-PSDs | Per site/interval preparation and cache I/O, shared once by power/coherence |
| Band power | Incremental theta/gamma integration and output writing/loading |
| Coherence | Matched cross spectra and ratio, excluding already charged auto-PSDs |
| Spike PPC | Selected-unit loading vs selection/phase sampling/PPC; counts/exclusions |
| Publication/display | Product write/load, figure construction/serialization, payload bytes, Matplotlib PNG export |

Where a verified API bundles operations, report the combined stage honestly;
do not refactor the numerical API solely to split microtimers. Use `cProfile`
on a dominant stage before proposing optimization. No pairwise PPC matrices,
dense empty session grid, or premature within-session multiprocessing.

After synthetic acceptance, measure CT026 with both centered intervals, all
requested metadata sites/pairs, one explicit unit and a small explicit multi-unit
selection. Do not benchmark all units or invent durations. Include power-only
cost, incremental power from cached PSDs, and reuse by later coherence.
Record configuration, trial/site/pair/unit/sample/frequency/segment counts,
hardware, cache state, status, elapsed time, file size, and major array sizes.
Cold analysis cache does not mean cold OS cache; process peak RSS is a lifetime
high-water mark, not a per-stage peak.

Report time to first durable neural product, subsequent publication delays,
and warm load/Refresh cost in `measurements.json`, `summary.md`, and a compact
viewer table. Inspect initial display/navigation in the browser; label
responsiveness qualitative unless actually measured end to end. Do not
mislabel Python figure-build time as browser latency or repeat costly cold
runs merely to fill a table. Recommend the work order/local offline approach
from evidence; defer major optimizations/cluster decisions to measured need.

## 7. Tests first and acceptance

Use pytest functions, seeded fixtures, and `uv run`. For each increment:
write tests -> demonstrate meaningful RED -> commit tests -> GREEN -> refactor.
Only intended paths enter commits; preserve unrelated staged user work.
Planned tests under `src/tests/neural_analysis/phase_synchrony/`:

| Files / package | Oracles and failure cases |
| --- | --- |
| `test_configuration.py`, `test_preparation.py` / WP1 | Original trial IDs/origin, half-open samples/spikes, UTC tolerance, metadata channels, calibrated units; off-grid choices and differently offset/drifting probe clocks produce the same native-rate target coordinates; bracketing load vs exact metric rows; missing/gapped support and pair-rate mismatch; existing preprocessing equivalence/no double notch |
| `test_power_coherence.py` / WP2 | Voltage x2 gives power x4; constant PSD integrates to 4q/46q; helper-times-bandwidth equivalence; excluded-gap values cannot contaminate boundaries; missing retained support vs valid zero; actual grids; seven/fifteen matched segments; SciPy coherence equivalence; one-segment/zero-denominator rejection; no trial pooling |
| `test_phase_metrics.py`, `test_spike_ppc.py` / WP3 | Drift fixture recovers phase differences, LFP PLV/PPC, off-grid UTC spike phases/PPC against independent analytic/no-drift references, far enough from block start to fail nominal-only timing; explicit 0.0 threshold retains tiny positive coefficients below the generic 1e-12 relative cutoff, rejects zero/nonfinite; other phase/count/interval oracles below |
| `test_results.py`, `test_stage_independence.py` / WP4 | Named-array/meta round trip incl. inspection inputs; otherwise matching legacy phase cache/derived results rejected despite saved all-valid mask; wrong source/settings/support rejected; temp/corrupt files ignored; immutable reruns/writer conflict; duplicate-run completion-time order/ties, older-result fallback on newer failure/unfinished work, mtime ignored; independent identities and shared PSD reuse |
| `test_runner.py` / WP5a | Only requested prerequisites; primary sweep/early publication with permitted/timed additional secondary preparation; unchanged primary product preserved; explicit order; failure isolation; shared costs charged once; reused/failed timing states; dry-run loads no signals/writes nothing |
| `test_scripts.py` / WP5b | CLI validation, explicit units, preparation-only/rerun/reuse/dry-run; batch delegates unchanged config and session-worker limit |
| `test_plotting.py` / WP6a | Real time/gaps/intervals/heatmap geometry; values/support/units/light mode; fixed scales; frequency-dependent band count ranges; saved selected-choice inspection; captioned Matplotlib PNG export into run/figures without Kaleido |
| `test_webapp.py` / WP6b | Component-specific route; behavior-only/power-only/partial states; Refresh discovers new products without recomputation; viewport persists; selected source run/time and newer failed-attempt status remain distinct |
| `test_synthetic_integration.py` / WP7a | Seeded LFP/spikes, both intervals/all families, separate invocations, failure/restart/reuse, power visible before later work, saved/direct agreement |

Use controlled stage holds for publication tests, not elapsed-speed assertions.
WP3 also retains known nonzero-offset/PLV-PPC identity, negative PPC, counts <2,
1,000/2,000 metric samples excluding guards, gap masks, valid-primary/invalid-
secondary independence, fixed frequency sets, two-spike cosine identity,
normalized-vector interpolation with unequal amplitudes, exact/neighbor/wrap
and boundary-count checks, and no unit pooling/50-spike threshold. Drift tests
must check numerical phases and PPC, not only identical target coordinates.
WP4 adds rejection of otherwise matching strict-support caches lacking the new
alignment method, threshold-change invalidation, and explicit 0.0 round trips.
Duration-scaling power tests use deterministic identical-segment PSDs, not a
requirement that two noise estimates match numerically. Independent finite
signals need not have exactly zero coherence. Add relevant existing metadata,
LFP, spike-phase/PPC, chart/timeline, and routing regressions when touched.
Browser inspection covers hover, navigation, added tracks, Refresh, and
responsiveness; Python tests alone do not establish browser behavior.

## 8. Delivery, dependencies, and data gate

Sequential delivery:
`WP1 -> WP2 -> WP3 -> WP4 -> WP5a -> WP5b -> WP6a -> WP6b -> WP7a -> WP7b -> WP8`.

Scientific contracts precede persistence/execution, then frozen figure/result
interfaces precede thin UI/CLI work. Each package uses Section 9's RED/GREEN/
review gate. WP7a completes synthetic/regression/browser acceptance; WP7b runs
and benchmarks CT026 under recorded authority. Present scientific figures,
timings, first-result latency, and a measured work-order recommendation. User
inspection precedes other experimental sessions; Round 2 awaits Round 1 and
its timing assessment. WP8 updates usage in `SESSION_TIMELINE.md`/neural README,
captions, limitations, live ledger, and final handoff.

Before spike-PPC validation, record the explicit CT026 probe-qualified unit IDs
and LFP sites selected through the existing interface. LFP validation proceeds
independently; do not choose arbitrary units or treat this outstanding selection
as a reason to stop power/coherence/phase implementation.

No new dependencies: existing NumPy, SciPy, pandas, Pynapple, Plotly, Matplotlib,
Streamlit, pytest, and standard library. Verify APIs against installed source/official
docs; do not modify dependencies. Batch wrapper parallelizes across sessions,
defaults to available CPU cores capped by session count, and permits a lower
worker limit for RAM/I/O. Building it does not authorize an experimental batch.

## 9. Agent assignments and gates

### 9.1 Roles and concurrency

**One Sol orchestrator; at most one active worker** (two active agents total).
Use zero workers for small lead-owned tasks. Reuse contexts sequentially; no
subworkers, speculative per-file agents, simultaneous Sol/Luna workers, or
standing reviewer team.

| Role | Model / effort | Scope |
| --- | --- | --- |
| Lead | `gpt-6.1-sol` / `high` | Scope, architecture, worker-diff review, integration, Git commits, live records, user communication, WP7 validation/timing interpretation |
| Scientific worker | `gpt-6.1-sol` / `high` | WP1-WP4, WP5a execution/timing, WP6a scientific plots, bounded scientific corrections/review |
| Optional routine worker | `gpt-6-luna` / `medium` | WP5b wrappers, WP6b mechanical widgets/routing, WP8 literal usage after Sol freezes contracts; `low` for literal docs only |

Calibration, synchronized time, estimators, axes/units, support, integration,
cache identity, shared-cost accounting, and plotted-data alignment stay with
Sol. Return Luna work to Sol if it touches those decisions. Report unavailable
model IDs rather than silently substituting; record actual model/effort/agent
ID. These assignments do not change this chat's runtime model and do not
promise a measured cost/speed advantage. They follow task-based
[model selection](https://developers.openai.com/api/docs/guides/model-selection)
and [reasoning effort](https://developers.openai.com/api/docs/guides/deployment-checklist).
Agent limits do not change CPU-process policy across sessions.

### 9.2 Dispatch, file ownership, and TDD gates

The module and test tables define ownership. Each dispatch names package/phase,
baseline HEAD/relevant dirty state, exact writable files, frozen data contracts,
spec sections, oracle/acceptance conditions, and focused check. Existing helper
changes must be explicitly scoped after reading callers. Sol releases
`run_session.py` before Luna edits its CLI; localized `webapp/app.py` /
`session_inputs.py` wiring belongs to WP6b. No concurrent edits to owned files.

1. Worker writes assigned tests only, reports meaningful RED and oracle, and
   stops. Missing dependencies/broken setup do not count as feature RED.
2. Lead reviews/reproduces RED and commits only intended tests, preserving
   unrelated staging. Record the test commit before releasing GREEN.
3. Worker implements, runs focused checks/regressions, reports exact results
   and unresolved issues, and becomes idle before lead integration.
4. Lead inspects diff/evidence, runs bounded acceptance, and alone commits
   accepted implementation/handoff. Record commits and next gate before
   dependent work. Do not weaken tests to get GREEN; document genuine fixture
   corrections or changed requirements.
5. Resume interruptions from actual files/evidence. Return scientific issues
   to Sol. No reset/rewrite of completed work merely to recover context.

Workers do not stage/commit, amend/reset, change dependencies, launch
experimental jobs, or spawn agents. No extra worktree is needed solely for
this sequential arrangement. When the lead writes substantive science, use
the idle Sol worker for bounded read-only review if useful; otherwise label
self-review honestly. No permanent additional reviewer is required.

## 10. Execution and evidence log

Append dated entries at decisions, test/implementation commits, reviews,
interruptions, worker/model changes, and data validation. Keep earlier evidence;
correct a mistaken historical statement with an appended correction. Use:

```text
Date/time/timezone - package / phase:
Authorization and actual lead/worker model, effort, agent ID:
Baseline/end HEAD; relevant dirty state; exact writable/changed files:
RED command, expected/actual result, test commit:
GREEN/regression/review commands, counts, warnings, implementation/review commits:
Scientific/interface changes or unresolved findings:
Running tool session/PID/command/log path, or none:
Saved synthetic/experimental products; measured resources/timings, or not measured:
Exact next action, owner, and remaining authority or user-inspection gate:
```

### 2026-11-04 - WP0 / live handoff and agent-assignment revision

- Authority: user requested a live implementation handoff matching other plans,
  with a Sol orchestrator, Sol/Luna workers by scientific complexity, and a
  minimal active team. Earlier accepted scope includes both centered intervals,
  per-site integrated power, separate processing/timings, CT026 2026-08-28, and
  light mode. This remains documentation-only authority.
- Inspected the live snapshot/ledger/resume patterns in the prediction and
  pooled-decoding plans and the Sol/Luna single-writer gates in the single-unit
  and prediction plans. Preserved the existing architecture/scientific scope.
- Baseline: branch `refactor`, HEAD
  `ca6ee10f7332d3ad9b466942b0fa45ab1ae4481e`. The feature plan is untracked;
  synchrony spec has staged and unstaged content. Other staged/source changes
  are listed in Section 0.1 and were not edited, staged, or committed here.
- Added the live snapshot, package ledger, resume procedure, model/file
  assignments, one-active-worker cap, and sequential RED/lead-commit/GREEN/
  review contract. Section 6's computation timing and early-publication
  requirements remain unchanged.
- Workers, source/test edits, RED/GREEN runs, Git commits, experimental runs,
  cluster work, and measured feature runtimes: none.
- Documentation verification: read-only Python assertions passed using
  `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync python -c` for ASCII,
  trailing whitespace, balanced code fences, local document links/anchors,
  snapshot/ledger/resume sections, all WP entries, model assignments, and
  retained scientific/timing clauses. No pytest or feature computation ran.
  `git diff --check -- docs/session_timeline_phase_synchrony_spec_v1.md` passed;
  the new plan's whitespace scan reported no diagnostics. Rechecked
  `git diff --cached --name-status`, `git diff --name-status`, and
  `git rev-parse HEAD`: existing staged/source paths and HEAD are unchanged.
- Next action: user review; after implementation approval, Sol starts WP1 RED
  using the checkout/authority procedure above. No worker should be started
  merely because assignments have been documented.

### 2026-11-04 - WP0 / correctness, completeness, and scope review

- Authority: user requested review for implementation by another lead.
  Documentation only; no workers, source/tests, analysis runs, or commits.
- Condensed Sections 0-9, removing repeated scope/assignment prose. Current
  sections supersede earlier draft mechanics; historical evidence is unchanged.
  Accepted estimands, intervals, TDD, timings, and one-worker cap are retained.
- Simplified to one preparation writer per session and atomic completed NPZs;
  removed per-component receipts and concurrent-writer cache coordination.
  Bundled APIs may report combined preparation timings instead of requiring
  numerical refactors solely for microtimers; browser latency claims need
  browser measurements.
- Source audit clarified invalid-gap interpolation, block-wide phase exclusion,
  coefficient versus unit-phase sampling, exact-sample/neighbor validity,
  calibration, component-specific route readiness, and saved inspection inputs.
  Added the verified CT026 metadata path/rates/calibration from read-only
  metadata/sidecars; no binary LFP or spike arrays were loaded.
- Read-only documentation assertions passed via
  `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync python -c`:
  ASCII/whitespace/fences, local links/anchors, ledger/model/scientific/timing
  contracts, session metadata path, and unchanged historical log. No pytest or
  feature numerical validation is claimed. Final whitespace checks had no
  diagnostics; staged/unstaged paths and HEAD match Section 0.1. The scientific
  spec is byte-for-byte unchanged by this review. Plan length fell from
  6,216 to about 4,400 words, including the preserved historical log.
- Next: user approves implementation scope; receiving Sol starts WP1 RED.
  Choose explicit benchmark units before spike-PPC validation, independently
  of LFP work. No further scientific spec decision was identified.

### 2026-11-04 - WP0 / implementation-chat feedback and resolved choices

- Corrects the prior readiness assessment: waveform grid/alignment and the
  cold-run latency/reuse precedence were insufficiently explicit. User chose
  native-rate Welch and primary-first publication with separately timed extra
  secondary preparation permitted. Updated spec and plan together.
- Native waveform rows use per-probe synchronization and a shared choice-relative
  grid; loader brackets are support, not extra observations. Added off-grid,
  cross-probe, and rate-mismatch checks. Phase remains on the 500-Hz grid.
- Strict scientific phase-support versioning now rejects legacy/derived phase
  products that cannot establish valid-gap/guard semantics. Added deterministic
  duplicate-run selection/fallback, source-run details, band spike-count ranges,
  session-relative *path* wording, and Matplotlib validation PNG export.
- Verified the loader's floor/ceil mapping, existing phase helper behavior,
  Matplotlib dependency, and project plotting/export precedent from source.
  Implementation-chat numerical checks are reported evidence; this chat did
  not reproduce experimental reads or run feature analysis/tests.
- Read-only documentation assertions passed via
  `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync python -c` for both
  files' whitespace/fences/links, plan ASCII, preserved history, resolved
  decisions, assignments/TDD contracts, and native-rate sample/segment constants.
  No workers, source edits, dependency changes, staging/commits, or experimental
  outputs. Existing HEAD/dirty work and prior evidence are preserved.
- Next: receiving Sol verifies actual implementation authority and starts or
  resumes the appropriate TDD gate. Explicit benchmark unit/site selection
  remains required only before spike-PPC validation; LFP work is independent.

### 2026-11-04 - WP0 / phase drift and amplitude-default clarification

- Authority: user supplied the remaining phase-alignment/default findings.
  Verified the production loader's block-start/nominal-rate construction,
  `PhaseAnalysisConfig.numerical_amplitude_threshold=0.0`, and generic phase
  normalization's `minimum_relative_magnitude=1e-12` default in source.
- Spec Section 4.2 and this plan now place full per-probe IRIG waveform sampling
  before anti-alias decimation/Morlet, sharing the UTC-grid sampler with Welch.
  Corrected phase retains UTC origin/guards; already synchronized spike times
  are sampled directly without a second LFP clock correction.
- Added `phase_alignment_semantics="irig_uniform_utc_pre_morlet_v1"`, actual
  synchronization/block/grid identity and explicit numerical/absolute thresholds
  to phase/derived-result compatibility. Strict support masks alone are not
  sufficient. Pin numerical threshold 0.0 instead of generic helper defaults.
- Extended WP3 oracles to recovered circular phase differences and spike
  phases/PPC under drift, using independent analytic/no-drift references and
  choices far from block start. Added tiny-positive/zero/nonfinite threshold
  checks and WP4 alignment/threshold cache rejection and metadata round trips.
- Read-only documentation assertions passed via
  `UV_CACHE_DIR=.pytest_cache/uv uv run --frozen --no-sync python -c` for both
  files' whitespace/fences/links, plan ASCII, unchanged historical evidence,
  explicit drift/threshold/cache/test requirements, and retained TDD/agent gates.
- Documentation-only revision: no source/tests, workers, experimental reads,
  analysis runs, dependency changes, staging, or commits. Prior evidence and
  unrelated working-tree changes are preserved. Next: receiving lead verifies
  authority and implements through the recorded TDD gates; benchmark unit/site
  selection remains a spike-validation prerequisite.

### 2026-11-04 20:27 America/New_York - WP1 / authorized RED dispatch

- Authority: user approved implementation: "alright, then begin implementation".
  Approved Round 1 sequence includes synthetic and CT026 validation; explicit
  spike unit/site selection requested asynchronously and remains pending.
- Baseline HEAD: `ca6ee10f7332d3ad9b466942b0fa45ab1ae4481e`, branch `refactor`.
  Existing staged specs/plans, unstaged main.py/sync_ephys.py, and untracked
  user files preserved. Lead owns this handoff and all Git operations.
- Lead runtime model/effort not exposed. One worker: `gpt-6.1-sol` / `high`,
  agent `/root/scientific_worker`; no additional worker or experimental job.
  Exact WP1 RED writable paths: new `test_configuration.py` and
  `test_preparation.py` under `src/tests/neural_analysis/phase_synchrony/`.
  No implementation changes released before lead review and test commit.
- Baseline command: `UV_CACHE_DIR=.pytest_cache/uv MPLCONFIGDIR=.pytest_cache/matplotlib
  uv run --frozen --no-sync pytest -q` with existing test_lfp_loading.py,
  test_lfp_power_summary.py, test_lfp_phase_clustering.py,
  test_lfp_synchrony_summary.py, test_session_timeline_webapp.py, and
  test_timeline_chart.py. Result: 77 passed, 1 skipped in 4.50 s; Node missing
  for the existing frontend lifecycle test. No feature validation claimed.
- Read CT026 metadata/sidecar only, not signal arrays. Next: worker reports
  substantive WP1 RED; lead inspects/reproduces and commits exact tests.

### 2026-11-04 20:32 America/New_York - WP1 / validation selection and baseline

- User selected "use hpc1 and probeB:8": canonical metadata IDs `HPC1` and
  `ProbeB:8`. This is the explicit designated spike validation selection;
  no additional units or experimental sessions are authorized by inference.
- Located bundled Node without adding a dependency. Re-ran the previously
  skipped renderer test with `NODE_BINARY=/home/matt/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node`
  and the same uv/cache environment: test_timeline_chart.py, 2 passed in 0.24 s.
- Read only the CT026 sync NPZ `meta` entry: nominal AP sample_rate_hz=30000.0
  is recorded. Preserve IRIG drift when converting AP indices to derived LFP
  indices; do not replace this known conversion rate with an anchor-slope rate.
  No raw LFP/spike arrays loaded, scientific feature timings not yet measured.

### 2026-11-04 20:35 America/New_York - WP1 / RED reviewed and committed

- Lead reviewed all 514 test lines and numerical/fixture contracts. Corrected
  genuine fixture errors before commit: existing behavior flag is
  `experimenter_reward_given`; quality CSV requires channel_id/x_um/y_um.
- Reproduced worker RED command from 20:27 entry with the two new tests:
  2 collection errors in 1.11 s, exclusively missing proposed feature package
  `src.neural_analysis.phase_synchrony`; installed dependencies are available.
  These are committed substantive feature tests, not a passing test claim.
- Test commit `0726ee1` contains only test_configuration.py and
  test_preparation.py. Existing four staged user documentation files preserved.
- Released GREEN to the same Sol/high worker with exact new writable files:
  phase_synchrony/__init__.py, results.py, preparation.py. Tests frozen except
  documented genuine errors. No experimental work or next-package writes.
- Next: inspect WP1 implementation and focused/regression evidence before
  committing GREEN and releasing WP2 RED.

### 2026-11-04 20:39 America/New_York - WP1 / GREEN reviewed

- Worker implemented only phase_synchrony/__init__.py, results.py and
  preparation.py. Lead read all source and test contracts. Existing public
  modules/dependencies unchanged; direct scalar/data records and functions.
- Initial GREEN found a genuine fixture filename error: existing quality
  loader accepts channel_quality.csv/json, not quality.csv. Corrected it in
  test commit `ef27189`; added a passing review check of UTC-representable
  exact native hits versus distinguishable fractional offsets. No separate
  RED is claimed for that extra acceptance check (implementation already
  satisfied the existing exact-hit/precision contract).
- Lead focused command: same WP1 pytest files/cache environment; 44 passed
  in 1.44 s. Existing test_session_metadata.py and test_lfp_loading.py:
  44 passed in 0.77 s. Worker independently obtained 44/44 passing results.
- UTC sampler preserves per-probe drift, calibrated uV, local gap validity,
  and exact half-open grids; notch is continuous and reused without doubling.
  Explicit selected-unit loader retains existing good/mua/channel rules.
- No signal data/experimental analyses or feature performance measurements.
  Lead commits accepted source and this handoff only; user staging preserved.
  Next gate: WP2 shared PSD/integrated power/coherence tests first.

### 2026-11-04 20:40 America/New_York - WP2 / RED dispatch

- Accepted WP1 source/handoff commit `ba44366`; unrelated staged user files
  remain untouched. Same Sol/high worker, baseline HEAD `ba44366`.
- Exact writable path: new phase_synchrony/test_power_coherence.py under
  src/tests/neural_analysis/. Tests cover independent voltage/PSD/segment
  oracles, trusted SciPy coherence, strict band support, and exact shared-PSD
  compatibility. No source changes released before lead test commit.
- Next: lead reviews/reproduces WP2 RED, commits tests, releases GREEN.

### 2026-11-04 20:44 America/New_York - WP2 / RED reviewed and committed

- Lead read the complete 322-line estimator test file and reproduced RED:
  1 collection error in 0.52 s for missing phase_synchrony.computation, with
  installed SciPy available. Independently checked the zero-DC fixture has
  exactly zero DC and positive retained band bins in this SciPy version.
- Lead-only test commit `705f8d8`; user staging preserved. Released GREEN to
  same Sol/high worker, exact writable sources computation.py and results.py.
  Tests frozen; no experimental work. Next: review focused/regression GREEN.

### 2026-11-04 20:48 America/New_York - WP2 / GREEN reviewed

- Lead read complete computation.py and added records in results.py. Actual
  Welch grid and density helper preserved, no per-window notch, exact shared
  PSD identity excludes bands, cross spectra matched within each trial.
  Coherence validity is per frequency and counts use common finite support.
- Worker command: uv/cache environment as above, pytest -q
  src/tests/neural_analysis/phase_synchrony plus existing
  test_lfp_power_summary.py, test_lfp_synchrony_summary.py and
  test_lfp_synchrony_runtime.py: 107 passed in 2.00 s, no warnings.
- Lead independently ran feature/power/synchrony tests: 96 passed in 1.56 s,
  then 11 synchrony-runtime regressions. No tests weakened or dependencies
  changed. Guards for representable intervals and consistent supplied PSD
  support implement the existing package contract; no separate RED claimed.
- No experimental arrays/jobs or feature timing claims. Lead commits only
  computation.py, results.py and this handoff. Next: WP3 RED for strict phase
  preparation, within-window LFP PLV/PPC, and selected-unit spike PPC.

### 2026-11-04 20:49 America/New_York - WP3 / RED dispatch

- Baseline/source commit `75ff0e0`; same Sol/high scientific worker.
  Exact new writable tests: test_phase_metrics.py and test_spike_ppc.py under
  src/tests/neural_analysis/phase_synchrony/. No source changes released.
- Require independent numerical drift/phase/spike oracles, explicit 0.0
  thresholds, complete metric counts excluding real guards, local gap validity,
  primary/secondary independence, normalized phase interpolation and per-unit
  counts/polar histograms. Existing bounded Morlet method remains the reference.
- Next: review/commit substantive RED, then release WP3 GREEN.

### 2026-11-04 20:56 America/New_York - WP3 / RED reviewed and committed

- Lead read both scientific test files and reproduced 2 expected missing
  SitePhase collection errors (1.15 s), then committed 396 lines as `14fc7ea`.
  Worker added two final RED checks before seeing the gate message; source
  writes remained stopped until lead reviewed and committed them as `cd046db`
  after repeating RED (2 errors, 1.12 s).
- Corrected the undefined-midpoint fixture before implementation to use UTC
  origin zero while retaining the 500-Hz grid/mask, rather than relabeling a
  4-Hz fixture with an inconsistent metric mask. No estimand/test weakening.
- GREEN writable sources: results.py, preparation.py, computation.py only;
  same Sol/high worker. Explicit drift-aware pre-Morlet sampling, real guard
  support, gap validity, complete LFP counts and per-trial/unit PPC required.
- Existing public phase metric helper can provide per-trial PPC/histograms;
  sampled phase dtype must not lose +/-pi histogram endpoints. No experimental
  reads/jobs or new dependencies. Next: inspect WP3 GREEN and regressions.

### 2026-11-04 20:58 America/New_York - WP3 / selected-unit metadata preflight

- Read only CT026 ProbeB cluster_info.tsv and channel-quality CSV through the
  existing metadata/selection functions. ProbeB:8 is group good, sorter channel
  201, eligible under existing quality rules. HPC1 is ProbeB saved LFP channel
  222. No aligned spike arrays, raw LFP traces, or experimental analyses loaded.
- WP3 GREEN remains with the same worker; no other source ownership changed.

### 2026-11-04 21:01 America/New_York - WP3 / review correction RED

- Initial WP3 implementation passed 26 new tests, then 161 feature/existing
  phase/spike/spectrogram/synchrony tests in 2.12 s. Lead read all 421 source
  additions and found missing metric-grid validation: matching malformed pair
  masks could include the guard and redefine M. Source edits stopped.
- Lead added four guard/grid/rate/bounds rejection cases, plus block-coordinate
  provenance assertions retaining UTC clock origin and small relative bounds.
  Focused pytest -k 'incompatible_metric_grid or real_guard_and_provenance'
  produced 6 behavioral failures, 12 deselected, in 1.26 s (no import failures).
- Committed tests alone as `87952d2`, then released narrow GREEN in
  computation.py phase validator and preparation.py block metadata. Numerical
  estimators remain unchanged. Same worker; no experimental work.
- Next: reproduce corrected GREEN/regressions and commit WP3 before WP4 RED.

### 2026-11-04 21:02 America/New_York - WP3 / corrected GREEN reviewed

- Lead reviewed the narrow validator/coordinate fix. Both lead and worker ran
  all feature tests plus test_lfp_phase_clustering.py,
  test_spike_lfp_phase_locking.py, test_lfp_spectrogram.py, and
  test_lfp_synchrony_summary.py: lead 165 passed in 2.12 s; worker 165 passed
  in 2.13 s, no warnings. Focused review corrections: 6 passed.
- Accepted scientific code performs bounded drift-corrected native waveform
  preparation before Morlet, frequency-specific real support around gaps,
  guarded 500-Hz trial extraction, complete-support PLV/PPC, and independent
  selected-unit PPC through the existing public estimator. Polar counts and
  selected UTC spikes are retained; negative PPC and valid zeros preserved.
- New source remains confined to the phase_synchrony package. Existing
  metadata, loaders, numerical APIs and dependencies unchanged. No experimental
  signal arrays or feature performance benchmarks yet.
- Lead commits results.py/preparation.py/computation.py and this handoff only.
  Next gate: WP4 tests for atomic persistence, compatibility and discovery.

### 2026-11-04 21:03 America/New_York - WP4 / RED dispatch

- Accepted WP3 source commit `6d58373`; same Sol/high worker. Exact writable
  tests: new test_results.py and test_stage_independence.py under
  src/tests/neural_analysis/phase_synchrony/. No source changes released.
- Cover all prepared/result records (including calibrated continuous LFP and
  its IRIG anchors), named arrays/meta, inspection inputs, no mutation,
  atomic immutable publication, legacy/alignment/threshold rejection,
  deterministic completion-time selection, independent stages and writer locks.
- No general provenance/workflow engine, extra per-NPZ receipt or dependencies.
  Next: inspect/reproduce/commit RED, then release bounded persistence GREEN.

### 2026-11-04 21:11 America/New_York - WP4 / RED reviewed and committed

- Lead reviewed 457 lines in both persistence test files and reproduced RED:
  42 expected missing-API AttributeErrors in 1.80 s; all scientific fixtures
  and dependencies ran successfully. Lead test commit `6f8650f` only.
- Frozen families: native-lfp, unit-spikes, waveforms, psd, phase, power,
  coherence, lfp-phase, spike-ppc. Native/unit interval identity is None;
  windowed families preserve primary/secondary. Nine typed records, named
  clock arrays and ragged-spike flat/offset arrays; dictionary-valued meta.
- Metadata-only discovery is separately testable and restricted to feature
  roots, so dry-run does not load saved scientific arrays. Artifact run stamps
  must not mutate records or scientific identities.
- Same Sol/high worker GREEN writable source results.py only. No source
  changes elsewhere, general workflow machinery, new dependencies or
  experimental reads. Next: inspect source and verify corrected GREEN.

### 2026-11-04 21:19 America/New_York - WP4 / scientific binding and planning RED

- Initial worker GREEN: 222 feature/metadata/work-cache/timeline-IO checks
  passed in 2.57 s. Lead read the explicit codecs, identities, publication,
  discovery and lock implementation. Source edits stopped for review findings.
- Record type alone did not bind publication to actual site/unit, interval,
  phase frequencies, Welch settings, bands or native notch settings. Lead
  added eight publication mismatch cases and one self-consistent identity
  relabeling case on load. Also added metadata-only discovery before a
  numerical preparation ID is known, required by cold/warm dry-run planning.
- Focused pytest -k 'publication_identity or relabelled or resolve_known_inputs'
  failed 10 cases, 42 deselected, in 1.32 s. Lead committed tests `d867a0e`
  before releasing fixes. No weakened tests or numerical estimator changes.
- Narrow GREEN: results.py plus computation.py only to relocate its existing
  phase validator into a shared public data-contract helper, avoiding duplicate
  validation/private cross-module import cycles. No new general framework.
- Next: verify bindings and metadata-only planning, then accept WP4 source.

### 2026-11-04 21:24 America/New_York - WP4 / preprocessing provenance RED

- Identity/planning fixes passed 52 persistence tests and 232 feature/existing
  checks. Lead reviewed the binding implementation and common phase-validator
  move (results.validate_phase_contract, imported by computation).
- Coherence lacked flat actual-notch metadata, and derived phase binding
  checked notch enabled but not source Q. Lead added three explicit mismatch
  tests: coherence false -> claimed true, and Q=30 -> claimed Q=20 in LFP
  phase/spike PPC. RED: 3 failed, 37 deselected, 1.20 s; commit `76bde37`.
- Released only coherence metadata fields and derived-source Q validation
  fixes. No numerical estimator or test weakening. Next: final WP4 GREEN.

### 2026-11-04 21:25 America/New_York - WP4 / final GREEN reviewed

- Lead read all codecs, identities, atomic publication, metadata/full loading,
  discovery and writer-lock source, then reviewed the common validator and
  actual settings bindings. Metadata-only discovery supports planning without
  numerical preparation hashes; full loading validates actual content/settings.
- Final focused preprocessing checks: 3 passed. Lead full command: uv/cache
  environment above, pytest -q src/tests/neural_analysis/phase_synchrony plus
  test_session_metadata.py, test_lfp_summary_work_cache.py and
  test_session_timeline_io.py: 235 passed in 2.74 s. Worker: 235 passed 2.77 s.
- Lead commits results.py, computation.py and this handoff only. User staging
  and unrelated changes preserved. No experimental signal reads/jobs or
  performance claims. Next: WP5a execution/timing core RED using frozen records
  and metadata-only cache discovery. CLI/batch remains the following gate.

### 2026-11-04 21:26 America/New_York - WP5a / execution-core RED dispatch

- Baseline HEAD `883803a`; same Sol/high worker. Exact writable path:
  new src/tests/neural_analysis/phase_synchrony/test_runner.py. No source writes.
- Test real temporary inputs and frozen scientific APIs: requested-only
  prerequisites, shared preparation/PSD reuse, primary-first publication under
  controlled holds, independent failure continuation, exclusive stage timing,
  durable measurements and additive runs. Dry-run forbids signal arrays and
  writes, including saved numeric preparation loads.
- Execution-only API precedes CLI. Main script snapshot is required now;
  batch snapshot is added when the actual wrapper exists in WP5b, with no fake
  implementation before its tests. Experimental output gate remains closed
  until synthetic acceptance; no experimental arrays have been loaded.
- Next: lead reviews/reproduces/commits RED, releases run_session.py core.

### 2026-11-04 21:35 America/New_York - WP5a / RED reviewed and committed

- Lead reviewed all 301 runner-test lines and reproduced the expected missing
  run_session collection error in 1.16 s. Test-only commit `11284ad`.
- Same Sol/high worker GREEN writable source: new phase_synchrony/run_session.py
  only. Frozen report: integer status, run directory, requested output paths,
  stage dictionaries and timing summary. Numerical/IO helpers remain separate.
- Rerun policy: recompute requested outputs while reusing compatible
  prerequisites; preparation-only rerun forces its chosen preparation. This
  supports incremental power-from-PSD measurement without changing bands.
- Require controlled primary-first publication, failure isolation, stage costs
  charged once, metadata-only dry-run, and new dated immutable session runs.
  No experimental arrays/jobs, CLI/batch implementation, or new dependencies.
- Next: review focused/combined GREEN and core before WP5b wrappers.

### 2026-11-04 21:41 America/New_York - upstream clock-conversion correction

- During WP5a work, lead identified that load_site_lfp's legacy inferred-AP-
  rate fallback confounds nominal AP-to-LFP resampling ratio with measured
  IRIG drift. This cannot establish verified native indices for strict alignment.
  CT026 already records nominal AP 30000 Hz and is unaffected.
- Lead added a real fixture removing only nominal sample_rate_hz. RED:
  1 failed (did not raise), 22 deselected, 1.11 s; test commit `a04b101`.
- Lead then changed only the new preparation.py loader to require a finite
  positive nominal AP rate, rejecting missing/bool/unverifiable rates clearly
  instead of silently inferring from UTC slopes. No external library changes.
  Self-review of this small input-contract correction; worker owns only the
  independent run_session.py file and was notified before lead edits.
- GREEN: test_preparation.py plus existing test_lfp_loading.py, 34 passed in
  1.26 s. No experimental arrays loaded. Next: integrate this correction and
  continue WP5a core; invalid site inputs must not disable unrelated sites.

### 2026-11-04 21:48 America/New_York - WP5a / runner review RED

- Lead reviewed the runner, its direct scientific callers, and persistence
  interfaces. Initial 13 runner tests passed. Follow-up tests expose native-only
  behavior coupling, missing/malformed behavior error escape, unknown unit-probe
  error escape, empty LFP-path failure isolation, missing support/array/count
  measurements, and missing actual timing values in the summary.
- Focused pytest -k 'native_only or behavior_input_failure or unknown_unit or
  empty_selected or retain_actual_support or summary_contains': 7 failed,
  13 deselected in 1.66 s. Tests committed `109986e` before implementation.
- Same worker released for run_session.py-only corrections. No experimental
  reads or new dependencies. Next: review corrected GREEN, commit execution
  core, then lead owns thin CLI/batch tests and wrappers (WP5b).

### 2026-11-04 21:50 America/New_York - WP5a / corrected GREEN reviewed

- Lead reviewed the complete execution source and narrow follow-up fixes.
  Native-only no longer loads behavior; ordinary behavior/unit/site failures
  are reported and isolated. Phase metric/guard dimensions, major array buffers,
  spike support counts, and summary timing values are recorded explicitly.
- Lead regression command: feature directory plus test_session_metadata.py,
  test_lfp_summary_work_cache.py and test_session_timeline_io.py: 256 passed
  in 3.74 s. Worker independently reports 20 focused and 234 combined passes.
- Worker stopped source edits. Lead accepts run_session.py core. Next WP5b:
  lead writes/commits CLI validation/delegation and session-worker tests before
  thin wrappers, reusing existing session-list/CPU-limit helpers. No experiments.

### 2026-11-04 21:52 America/New_York - WP5b / lead RED and GREEN

- Test-only `b800756`: missing run_batch feature collection error, 1.36 s.
  Lead then implemented CLI selection validation/delegation plus batch wrapper.
  Existing session-list and CPU-worker cap helpers are imported, not duplicated.
- Corrected a genuine test fixture in `6e11de7`: a one-site snapshot test had
  retained an explicit two-site pair. No production contract was changed.
- Focused: 10 passed in 1.29 s. Feature plus existing timeline-script suite:
  194 passed in 4.05 s. CLI module --help executed successfully. Batch ordinary
  failures preserve other sessions, report actual elapsed seconds, and return
  nonzero; dry-run launches no processes. No experimental batch authorized/run.
- Same scientific worker read-only review of CLI/batch source; no concurrent
  source ownership. Next: accept/commit wrappers, then WP6a figure tests.

### 2026-11-04 21:55 America/New_York - WP5b / review corrections accepted

- Worker read-only review caught ordinary writer-lock RuntimeError escaping
  CLI/batch boundaries. Lead RED `987fab4`: 1 failed, 1 passed in 1.38 s;
  added WriterLockError(RuntimeError), caught narrowly at boundaries. Existing
  RuntimeError callers remain compatible; unexpected programming errors escape.
- Combined 196 passed in 4.05 s exposed Python's multithreaded-fork warning in
  a real session-process test. Lead RED `db4aea1`: missing fresh process context,
  1 failed in 1.27 s. Use ProcessPoolExecutor's standard spawn context so workers
  do not inherit scientific-library threads. No within-session parallelism.
- Final 12 script tests passed in 2.80 s, including real child-process writer
  conflict handling; no warnings. Rerun CLI help documents prerequisite reuse.
  Lead self-reviewed these narrow corrections. Next: WP6a RED with same worker.

### 2026-11-04 21:56 America/New_York - WP6a / plotting RED dispatch

- Baseline `8f9443f`; same Sol/high worker, exact writable path new
  test_plotting.py only. No source, dependency or experimental writes.
- Frozen public functions: plot_synchrony_timeline(behavior, products, ...),
  plot_choice_inspection(behavior, products, row_position, ...), and
  export_validation_figures(behavior, products, run_directory, row_position=...,
  frequency_hz=8.0, interval='primary', site_id=None, unit_id=None) -> tuple[Path,...].
  Products are SavedProduct records; behavior is an empty-region
  SessionTimelineResult. heatmap_selection=(family, selection_tuple).
- Oracles cover saved values, original row/ID/UTC binding, true interval edges
  and gaps, fixed physical/signed scales, support/guard counts, identified phase
  frequency, spike/polar inputs, captioned opaque PNGs and immutable exports.
  All plotting forbids raw loads/estimators. Existing behavior renderer and
  persistent chart metadata are reused. Next: lead reviews and commits RED.

### 2026-11-04 22:04 America/New_York - WP6a / reviewed RED committed

- Lead read all worker plotting tests and reproduced missing plotting feature
  RED in 1.15 s; Matplotlib/Plotly dependencies imported successfully.
  Added a geometry oracle for overlapping choice intervals, requiring each
  true interval to survive instead of constructing false shared boundaries.
- Test-only commit `0e3f57f`; same worker released for new plotting.py only.
  No experimental arrays, source changes in other files or dependencies.
- While waiting at the plotting gate, lead documented the already accepted
  CLI/contracts in SESSION_TIMELINE.md and neural README. These additions make
  no viewer or experimental timing claims and will be completed at WP8.
- Next: inspect actual scientific figures and focused/combined GREEN before
  the thin UI package; retain the existing manual Refresh/navigation contract.

WP6b implementation notes for the next gate (no UI source written yet):

- Lead owns tests and thin route/widgets. Availability requires behavior only;
  compatible saved components load independently. Do not gate the view on
  raw spikes, pairs or every LFP source. Stat-level source compatibility remains
  per-product, separate from route readiness.
- Reuse runner source-identity construction through a small public helper in
  preparation.py if needed, rather than duplicating that logic or invoking the
  runner from Streamlit. Test equality before moving the existing code.
- Select scientific settings from completed products' run config.json records;
  fingerprint settings without requested units/sites/families/interval lists.
  Match each family's own relevant parameters, so selecting another saved unit
  or phase configuration does not discard compatible site power.
- Keep metadata discovery in per-session UI state until manual Refresh;
  never cache absence indefinitely. Immutable saved product loads can use
  path/size/mtime cache keys. Refresh keeps chart key/navigation revision stable.
- Read newer attempt statuses from run config/measurements JSON, ordered by
  recorded run UTC identity, and report separately from the completed source
  run/time. A newer failed attempt must not hide a completed compatible product.
- Selected-choice inspection loads only matching saved waveform/phase/PPC
  prerequisites on explicit request; ordinary display changes never prepare.

### 2026-11-04 22:08 America/New_York - CT026 CLI metadata-only preflight

- Lead ran the accepted run_session module with CT026 metadata, all four
  families, both intervals, --unit ProbeB:8 --lfp-site HPC1 --dry-run.
  Exit 0: 20 planned final components, primary sweep then secondary; per sweep
  three site powers, three coherence pairs, three LFP-phase pairs and one unit.
- In-process metadata planning elapsed 0.0055 s (not an analysis or browser
  benchmark). No run directory, products, writer lock or signal-array loads.
  Synthetic/real-browser acceptance still precedes experimental processing.

### 2026-11-04 22:10 America/New_York - shared metadata identity extraction

- Independent of plotting.py, lead prepared the metadata-only viewer boundary
  while worker retained sole plotting source ownership. Announced exact lead
  ownership: preparation.py and runner._known_identity only. No UI source.
- Tests `f2de509`: 10 genuine missing-public-helper failures in 1.35 s.
  Pre-GREEN oracle refinement `cf45453` retains independent expected keys,
  mapped channels and relevant file lists after runner delegation.
- Extracted existing source IDs into preparation.product_source_ids; runner
  delegates and its duplicate stat helper is removed. Existing preparation stat
  helper now retains the runner's explicit None/missing-file semantics. The
  initial extraction exposed that mismatch in an existing committed test;
  corrected source, without changing the test or scientific identities.
- GREEN: source-identity, runner, scripts and preparation suites, 65 passed
  in 3.96 s. Lead self-review; no array loads by the new helper. This is a
  small WP6b prerequisite with frozen existing scientific contracts, not a
  release of UI implementation before accepted scientific plotting.

### 2026-11-04 22:15 America/New_York - WP6a / plotted-data review RED

- Initial plotting GREEN: 14 passed; worker 263 feature/timeline/chart checks
  passed in 7.79 s (UI test draft excluded). Lead read all 426 source lines.
  Worker and lead visually inspected actual PNGs; static exports omitted
  site/band row titles and the inspection polar title overlapped an x label.
- Lead follow-up `e929a1d`: rich heatmap original-trial/UTC/support/source hover,
  explicit validity masks, retained-band-only LFP sample counts, actual probe/
  channel inspection captions, and exported site/band/frequency/unit labels.
  Focused RED: 5 failed, 12 deselected in 2.42 s. Before committing, corrected
  a genuine fixture padding shortage using an earlier-starting, longer synthetic
  native source; count test then failed the actual outside-band 0-1000 bug.
- Same worker released for plotting.py-only fixes and visual repair. No
  estimator/source changes, test weakening, experimental reads or dependencies.
- Lead also prepared/committed thin UI tests `f4bfd03` while figure fixes were
  pending: expected missing phase_synchrony_view collection error in 1.76 s.
  No UI source implementation is released before plotting acceptance. Tests
  freeze behavior-only routing, independent saved products, settings/source
  matching, newer failed-attempt separation, explicit inspection and real
  Streamlit Refresh with stable component/navigation identity.

### 2026-11-04 22:19 America/New_York - WP6a / corrected figures accepted

- Additional required-label RED `8063a0d`: 1 failed, 16 deselected in 1.78 s.
  Pair PPC is explicitly Within-window LFP-LFP PPC, separate from unit PPC.
- Lead reviewed all source/follow-up fixes and actual timeline/inspection PNGs.
  Labels, calibrated units, original integer trial ID, probe/channel, selected
  8-Hz phase, counts, captions and polar layout are now readable. No figure
  builder imports analysis/preparation or raw loaders. Product arrays stay intact.
- Lead feature/timeline-plot/chart command (bundled Node supplied, UI tests
  excluded): 247 passed in 7.97 s. Worker adds existing session-timeline tests:
  266 passed in 8.03 s. Focused 17 passed. No warnings.
- Worker stopped edits. Lead accepts plotting.py; WP6b source is now released
  against previously committed RED `f4bfd03`. No experimental processing until
  synthetic/browser gate. Optional async question asks whether to skip the
  plan's multi-unit timing or receive additional explicit IDs; ProbeB:8/HPC1
  single-unit validation proceeds independently.

### 2026-11-04 22:30 America/New_York - WP6b / reviewed corrections and synthetic acceptance

- Lead implemented the thin saved-only view and localized app/session-input
  routes after WP6a acceptance. Corrected genuine test-only pandas repr
  truncation in `721307e`, comparing actual dataframe path cells instead.
- Review RED `6fa2d40`: 3 failed, 6 deselected, 2.27 s; explicit empty display
  selections, actual saved unit's LFP site, and no-unit inspection now honored.
- Worker read-only review found per-selection missing labels, irrelevant-setting
  attempt filtering and missing-predecessor failure reporting. Lead RED
  `d099477`: 3 failed, 1 passed, 8 deselected in 2.10 s. Corrected matching uses
  each family's product_identity parameters and reports absent source as None;
  attempts ordered by parsed UTC run identity, independent of filesystem mtime.
- Payload RED `5fee607`: 1 failed in 1.97 s. Actual chart JSON bytes and separate
  Python serialization seconds now shown; caption honestly notes the shared
  renderer serializes again. No browser-latency claim. All 12 UI tests pass.
- Lead wrote seeded full-pipeline acceptance tests `697a954`: 2 passed in
  3.67 s on existing TDD-covered features; no new implementation/RED invented.
  Two independent drifting probe clocks, three sites/pairs, all four families,
  both intervals, 20 saved products, native reuse across separate invocations,
  shared auto-PSDs, incremental power, saved/direct agreement, warm reuse and
  restart after one pair failure. Observed 3/6 power products already durable
  at each controlled primary/secondary coherence failure; independent pairs survive.
- No experimental arrays loaded. Next: commit reviewed UI source after combined
  regressions, launch seeded session locally and inspect hover/pan/Refresh.
- Lead combined feature, metadata, timeline IO/webapp/chart and package checks:
  302 passed in 12.32 s, bundled Node enabled; no warnings. UI source accepted.
