# Session Timeline Phase Synchrony and Band Power Analysis Spec

**Status:** Version 1 — Round 1 scope and analysis defaults agreed with the user on 2026-11-04; implementation planning remains separate. Exact UI layout remains open for refinement.

**Live implementation handoff (draft):** [session_timeline_phase_synchrony_implementation_plan.md](session_timeline_phase_synchrony_implementation_plan.md), including the current snapshot, package ledger, sequential Sol/Luna assignments, separate processing stages, timing measurements, and partial-result viewing. The plan requires user review before implementation.

**Companion:** `session_timeline_pca_spec_v1.md` and the implemented viewer documented in `src/neural_analysis/SESSION_TIMELINE.md`. Build a **separate synchrony view** in the existing app, reusing the behavioral timeline and navigation. Viewing synchrony must not require preparing PCA. The user will finalize the exact panel arrangement and controls with Codex.

**Purpose:** Descriptive inspection of per-site LFP band power, LFP–LFP synchrony, and selected-unit spike–LFP locking within individual choice windows. This is not an across-trial average, an ITPC analysis, a significance-testing pipeline, or evidence of directed/causal communication.

This spec records the agreed analyses and initial operating settings. The user accepted the existing Morlet phase method and theta/gamma bands, notch and frequency-exclusion defaults, 0.5-second/50%-overlap Welch settings, and strict missing-data policy on 2026-11-04. These are inspection settings, not claims of optimal time/frequency resolution. Verify compatibility with existing code before implementing; do not silently change estimator definitions to reuse a convenient function.

The user also requested per-trial band power and selected **total integrated band power**, reported in uV^2 for calibrated uV inputs. Mean PSD density and baseline-normalized dB are different quantities and are not the primary power output here.

Following implementation-chat review, the user selected native-rate waveforms for Welch power/coherence and primary-first publication on cold runs. Additional preparation for secondary phase support is permitted and must be timed; it must not delay completed primary results merely to avoid repeated work.

## 1. Required Scope

Implement:

1. A separate real-session-time synchrony view showing choices and optional behavioral events, with band-power and synchrony tracks underneath.
2. **Per-site theta and gamma total band power**, giving one scalar per trial, site, band, and selected centered interval.
3. **LFP–LFP magnitude-squared coherence, within-window PLV, and within-window PPC** for metadata-designated LFP channel pairs.
4. **Spike–LFP PPC** for one or more explicitly selected units and metadata-designated LFP channels.
5. Use **[-1 s, +1 s)** as the primary interval and **[-2 s, +2 s)** as an optional secondary interval for both power and synchrony. Support both centered intervals in the first implementation; selecting the longer interval is optional for the user.
6. Reusable computation functions that can run offline, save results, and load those results into the viewer without recalculating the analysis during navigation.
7. Local runtime, memory, and viewer-performance assessment, including power, as part of implementation. Decide whether cluster precomputation is justified only after measurement.

All metric averages are over observations **inside one trial's selected interval**: phase samples, spike phases, or Welch segments. Do not pool observations from different trials to make a single-trial value. Do not borrow the across-trial averaging axis from the existing ITPC/ISPC analyses.

Keep this inspection-oriented. No surrogate distributions, p-values, FDR correction, effective-sample-size correction, or new temporal-confound workflow is required here. Ordinary undefined-result handling and explicit interpretation notes are sufficient.

### 1.1 Implementation rounds

| Round | Scope and ordering |
|---|---|
| **1: centered choice estimates** | Implement both centered intervals, per-site theta/gamma band power, the three LFP–LFP measures, selected-unit spike–LFP PPC, offline save/load, the separate viewer, and measured local computation/viewer performance. |
| **2: before/after estimates** | After Round 1 is complete and computation-time measurements are available, add **before [-2 s, 0)** and **after [0, +2 s)** estimates. These do not gate Round 1 completion. |

Moving-window analysis and a step-size control are outside the current implementation scope. The deferred before/after intervals are two-second intervals on either side of choice, not halves of the primary [-1 s, +1 s) interval.

## 2. Input Channels, Units, and Timestamps

### 2.1 LFP channels come from session metadata

Use the session metadata as the authoritative source of the LFP channels designated for analysis, their recording/probe identifiers, and their anatomical/site labels.

- Resolve labels such as **PFC**, **HPC1**, and **HPC2** through metadata. These are examples, not hard-coded channel numbers or guaranteed fields.
- Display the site label and actual source channel identity in the UI or hover/result details.
- Reuse the existing LFP loading, correction, referencing, calibration, synchronization, and sample-rate handling. Do not select an arbitrary channel from each region or automatically average channels into a regional signal.
- Allow selection among available metadata-designated channels/pairs. Use the existing pair definitions if present; otherwise offer the relevant distinct pairs, for example PFC–HPC1, PFC–HPC2, and HPC1–HPC2.
- Do not silently replace a missing channel, fabricate HPC2, or duplicate a channel under two site labels. Missing inputs disable only dependent tracks.
- No exhaustive all-probe channel-pair analysis is required. Restrict computation to the selected metadata channels and requested pairs.

Power uses these same metadata-designated sites but needs only one site's LFP at a time. It does not require a site pair, another recording region, spike data, or Morlet phase preparation. Keep separate site-level power results; do not average channels or infer a regional signal.

Record a consistent ordered pair convention, such as phase A minus phase B. Magnitude measures are symmetric, but the optional phase-offset sign depends on that convention.

### 2.2 Selected-unit inputs

Use existing spike sorting, unit-quality selection, stable unit IDs, and spike timestamps. The user selects the unit and the LFP phase channel; the LFP may be local or in another region.

Do not pool different units' spikes to increase sample count. An all-unit precomputation is not required merely to support selection of a few units. Missing spike data must not prevent LFP–LFP viewing.

### 2.3 Timing and behavioral events

Reuse the synchronized session clock and the existing choice timestamp for each trial. Verify timestamp units, event meanings, and left/right codes against the codebase.

Use elapsed session time for every panel. Do not round choices to a session-wide grid, time-warp trials, or replace elapsed time with equally spaced trial indices.

Reuse optional trial-start, left/right lick, and LED on/off tracks from the PCA timeline. Behavioral tracks may continue through intervals where no neural analysis is displayed.

The established task design prevents overlapping choice windows. Do not introduce overlap detection, purging, or a new boundary-trial exclusion system.

## 3. Analysis Windows and Initial Controls

### 3.1 Centered estimates, display context, and preparation support

- **Primary estimation interval:** [-1 s, +1 s), a total of two seconds centered on the actual choice.
- **Optional secondary estimation interval:** [-2 s, +2 s), a total of four seconds centered on the same choice.
- **Inspection context:** any wider LFP/behavior context shown when inspecting a choice. Displaying context does not add observations to the metric.
- **Preparation support:** real continuous LFP outside the estimation interval needed for filtering, wavelets, or phase interpolation. This support is distinct from the observations entering the statistic.

Use the same selected centered interval for site power, all LFP–LFP measures, and selected-unit spike–LFP PPC. There is no moving step in Round 1. The PCA viewer's 20/50/100/500-ms spike bins do not define these intervals, and spikes retain their available timestamp precision.

### 3.2 Accepted initial settings

| Setting | Initial value or policy |
|---|---|
| Alignment | Choice time |
| Primary estimation interval | **[-1 s, +1 s)**; total duration 2 s |
| Secondary estimation interval | **[-2 s, +2 s)**; total duration 4 s; optional selection |
| Timeline estimate placement | One estimate at each actual choice time for the selected interval |
| Interval control | Shared primary/secondary selection for power, LFP–LFP, and selected-unit spike–LFP measures |
| Selected-unit PPC initial view | Primary centered-interval value with contributing spike count |
| Before/after intervals | Deferred to Round 2: [-2 s, 0) and [0, +2 s) |
| Bands | Theta 6–10 Hz; gamma 30–80 Hz, configurable |
| Power summary | Integrated Welch PSD over retained band intervals; one absolute power value per site/trial/band, in uV^2 for uV inputs |
| Phase/frequency preparation | Existing Morlet numerical method; drift-aware UTC preparation before the transform (Section 4.2); 2–100 Hz at 2-Hz spacing on the shared 500-Hz phase grid |
| Numerical phase-amplitude threshold | `numerical_amplitude_threshold=0.0`, passed explicitly to helper normalization/sampling and saved; zero/nonfinite amplitude remains invalid |
| LFP–LFP phase metric | PLV/PPC toggle; PLV initially displayed |
| Coherence display | Separate metric panel or selectable view, explicitly labeled magnitude-squared coherence |

The primary two-second interval reflects the user's inspection priority. It is not a minimum scientific validity threshold. The four-second option includes more cycles, spectral segments, and potentially spikes, but also summarizes a longer period. Support counts and the selected interval must remain visible.

### 3.3 Interval placement and interpretation

For choice time c and selected half-width a (1 s or 2 s), use the half-open interval **[c-a, c+a)**. Plot its estimate at c on the elapsed-session timeline and retain its true start/end timestamps and relative bounds. Do not round c to a session-wide grid.

Each interval produces one value per trial, pair or selected unit/channel, and frequency, followed by the defined band summaries. A marker at choice represents the entire selected interval, not an instantaneous measurement at choice. No intervening session-time estimates are required.

For power, estimate a PSD from each site's samples within the same interval and integrate it to produce one theta and one gamma value per trial. Do not average power across trials, combine theta/gamma into one number, or accumulate signal energy in proportion to interval duration.

Both centered intervals deliberately combine activity before and after choice. A change in preferred phase offset across choice can reduce the combined phase consistency even when each side is individually consistent. These values do not identify when a change occurred; the separate before/after estimates are deferred to Round 2.

Calculate the four-second result from that interval's observations or exactly equivalent sufficient statistics, not from an average of two-second metric values. Keep interval-specific validity: unavailable four-second support must not invalidate an otherwise valid two-second result.

Build intervals within each trial. Never join the end of one trial to the beginning of another. A changed viewport must not crop the observations used, alter interval placement, or change metric values.

## 4. LFP and Phase Preparation

Reuse the established continuous-LFP preprocessing and frequency-resolved Morlet phase pipeline with the accepted defaults. Verify and record the actual frequency grid, sample rate, filter/reference settings, and phase-at-spike sampling convention. Do not substitute a broadband Hilbert phase estimator for the agreed per-frequency Morlet calculation.

Use continuous or suitably padded real LFP around each choice when preparing phase/filter outputs; then restrict the samples entering each statistic to its specified estimation interval. Reuse compatible continuous preprocessing across both centered intervals rather than independently filtering each short interval. Phase-estimation context outside the display interval is permitted and must not be confused with pooling additional observations into the statistic.

Record the transform's temporal support. These are offline centered-window measurements, not causal online estimates or precise detection of when coupling began. Reuse ordinary existing edge-validity handling; do not create a new edge-correction research project.

For LFP pairs, use synchronized phase values on a common time grid. Retain the actual analysis sampling rate; do not silently change it between trials or just to alter the PPC sample-count adjustment. Check cached data compatibility before reuse.

Prepared phase and derived phase results must identify their scientific support semantics, initially `phase_support_semantics="strict_support_v1"`: invalid source gaps stay invalid, exact-sample/interpolated validity is explicit, and real interpolation guard support is retained. Phase identity also requires the drift-aware alignment method/version and explicit amplitude settings in Section 4.2. Legacy caches without these guarantees are incompatible even when sources, shapes, frequencies, and saved validity masks otherwise match. A mask may already contain false-valid values from the old interpolation helper. Recompute rather than relabel or trust such a cache; this does not invalidate independent power/coherence products.

### 4.1 Frequencies, band summaries, and line noise

Retain frequency-resolved estimates underneath band summaries. The accepted initial phase grid is **2–100 Hz in 2-Hz steps**, matching the existing summary configuration. Preserve coherence's actual frequency coordinates separately. Record actual frequencies rather than assuming a requested grid was achieved.

For a regular grid, define each **dimensionless synchrony** band summary as the **arithmetic mean of the scalar metric values at retained frequency centers** in the band. For gamma, omit centers in 58–62 Hz, inclusive, by default. For example, a 2-Hz gamma grid excludes 58, 60, and 62 Hz. This is a mean synchrony score over retained frequencies. Power instead uses the frequency integral in Section 6.3.

- Compute PLV/PPC/coherence separately at each frequency first. Do not pool phase vectors across frequencies and then calculate one metric.
- Normalize the summary by the number of retained frequency centers, not by the original count with excluded centers treated as zero.
- Keep the retained-frequency mask fixed across compared trials. If a required retained estimate is undefined, flag the band summary rather than silently changing its frequency set.
- Do not invent a single "gamma phase" by averaging angles from different frequencies. A phase-trace inspection should identify a specific frequency or explicitly identified narrowband signal.
- Reuse the established **configurable 60-Hz notch**, enabled by default in the LFP summary configuration, without applying it twice to already processed data. Exploratory LFP views have different defaults; bind this analysis to explicit saved settings rather than the last-used widget value. Record whether the notch was actually applied.
- The notch flag and the 58–62 Hz exclusion are separate recorded settings. Mask excluded frequencies clearly in frequency-resolved views; excluded is not zero synchrony.

Do not use this frequency-center mean as a band-power calculation. Section 6.3 defines power's bandwidth-aware integration rule. If the existing phase grid is nonuniform, resolve its synchrony band-weighting convention in the plan rather than quietly treating it as uniformly sampled.

### 4.2 Drift-aware phase preparation and validity defaults

Apply synchronization throughout each padded continuous block, before decimation/Morlet calculation. After native-rate calibration and the configured continuous notch, use each probe's IRIG UTC-to-native-index mapping `s_p(T[k])` to linearly sample the real waveform onto a uniform UTC native-rate block grid. Use the same clock-sampling adapter as Welch (Section 6.2.1), with contiguous-valid-neighbor/no-extrapolation rules. Then apply the existing anti-alias decimation and Morlet calculation to that corrected waveform, without repeating the notch. Do not pass irregular IRIG-corrected source timestamps to a uniform-rate transform without first resampling the waveform.

Represent the block grid as relative seconds plus its saved UTC origin where needed for numerical precision. Transform coordinates must retain that UTC identity and actual post-decimation rate. Extract each trial's phase at `choice + r`, where `r=-a+k/500` for the metric samples; sample selected spikes at their already synchronized UTC times using the same prepared phase and circular interpolation rule. Retain real guard support. Do not apply the LFP clock correction to already synchronized spike times a second time. Merely labeling block-start-anchored nominal-rate coefficients with common 500-Hz coordinates is insufficient; the existing production phase loader cannot be reused unchanged for this path.

Save `phase_alignment_semantics="irig_uniform_utc_pre_morlet_v1"`, actual synchronization identity/offset settings, block UTC origin/bounds, grid rates, operation order, and support with phase preparation and its dependent results. Reject otherwise matching caches/results with missing or nominal-block-only alignment semantics, including those already marked `strict_support_v1`.

Pin the production default `numerical_amplitude_threshold=0.0`. Pass it explicitly as `minimum_relative_magnitude=0.0` to helpers that normalize/sample phase; do not inherit their generic `1e-12` default or reintroduce it downstream. The default absolute amplitude gate is also disabled/0.0, matching the empty production per-site threshold configuration. Save effective numerical/absolute thresholds and their meanings/units in phase identity; an explicitly changed threshold requires a different compatible configuration. At 0.0, finite strictly positive coefficients remain eligible even below `1e-12` of a block maximum; zero/nonfinite coefficients and undefined interpolants remain invalid. This is numerical validity, not a biological reliability cutoff.

## 5. LFP–LFP PLV and PPC

### 5.1 Observations and PLV

For one trial, one LFP pair A/B, one frequency f, and one estimation window W, define the M phase differences:

$$
\Delta\phi_m(f) = \phi_A(f,t_m) - \phi_B(f,t_m)
$$

$$
S(f,W) = \sum_{m=1}^{M} e^{i\Delta\phi_m(f)}
$$

$$
\mathrm{PLV}(f,W) = \frac{\lvert S(f,W)\rvert}{M}
$$

M is the number of valid time samples in this trial/window, not a trial count. Phase-only normalization gives each observation equal amplitude weight. High PLV indicates a consistent offset, not necessarily a zero offset. [R1]

If retained, the circular mean phase offset is the argument of S. Mark it undefined when the resultant is zero; do not interpret it as a direction of biological influence.

### 5.2 LFP–LFP PPC on the same observations

Use the ordinary all-observation-pairs statistic:

$$
\mathrm{PPC}_{\mathrm{LFP}}(f,W) =
\frac{\lvert S(f,W)\rvert^2 - M}{M(M-1)}
$$

For M>1, equivalently:

$$
\mathrm{PPC}_{\mathrm{LFP}}(f,W) =
\frac{M\,\mathrm{PLV}(f,W)^2 - 1}{M-1}
$$

PPC is applicable to LFP–LFP phase differences; it is not exclusive to spikes. The PLV-to-PPC identity applies to the same observations and count. [R1, R2]

Use the complex-sum expression rather than explicitly constructing all M(M-1) phase pairs. Require M>1 for this shared PLV/PPC inspection track. Preserve finite negative PPC values; do not replace them with zero or their absolute value.

Apply the conversion **per frequency**, before band averaging. Do not square band-averaged PLV to obtain band PPC. Store the count M and phase convention with the results.

### 5.3 Interpretation

For fixed M, these PLV and PPC values are transformations of the same information, not independent evidence. A metric toggle is sufficient; simultaneous plotting is optional, not required.

PPC's standard unbiasedness result concerns appropriate independent observations. Successive filtered LFP phase samples are not automatically independent, and a high sample rate is not a count of independent oscillation cycles. Do not claim that the within-time PPC correction resolves temporal dependence. [R2, R3]

Use the label **Within-window LFP–LFP PPC** and the brief caveat:

> Descriptive phase consistency over time within one trial; temporal dependence is not corrected.

No effective-sample-size substitution, cycle-thinning algorithm, or independence test is required in this inspection implementation.

## 6. Welch Spectra, LFP–LFP Coherence, and Band Power

### 6.1 Definition and within-trial averaging

Implement **magnitude-squared coherence**, not complex coherency or the unsquared coherence magnitude:

$$
C_{AB}(f,W) =
\frac{\lvert S_{AB}(f,W)\rvert^2}{S_{AA}(f,W)S_{BB}(f,W)}
$$

Here S_AA and S_BB are power spectra, and S_AB is the cross spectrum, estimated by averaging matched segments from this trial's window. Use Welch's method as the simple initial estimator. [R4]

Average the cross spectra and power spectra first, then form their ratio. Do not average one-segment coherence ratios. A single unsmoothed Fourier-product ratio is identically one wherever defined, so it is not a usable synchrony estimate; the averaging is essential.

### 6.2 Accepted Welch defaults

Use the following accepted defaults for coherence and power, matching the existing PSD settings:

| Setting | Initial value |
|---|---|
| Taper | Periodic Hann, matching the current power implementation's `hann_periodic`; record it explicitly |
| Detrending | Subtract each segment's mean |
| Segment duration | 0.5 s, converted to the nearest whole number of samples |
| Segment samples | round(fs / 2), checked against the actual sampling rate/window |
| Segment overlap | 50% |
| FFT length | Segment length, without zero padding by default |
| Averaging | Arithmetic mean of matched auto/cross spectra |

Welch averages overlapping tapered segment spectra. Use native-rate waveforms, separately from the 500-Hz Morlet phase grid. For CT026's 2500-Hz LFP, these settings give 1250-sample segments, 625-sample overlap, and a 2-Hz frequency-grid spacing. This grid spacing is not a claim of 2-Hz resolving power independent of the taper. [R5]

The complete primary [-1 s, +1 s) interval contains seven such segments; the secondary [-2 s, +2 s) interval contains fifteen. They are overlapping estimates, not seven/fifteen independent trials. Retain the actual number of contributing segments.

Require at least two complete segments as a computational minimum. That minimum is not an adequacy guarantee. Reject an incompatible shorter-window configuration instead of silently shortening segments, zero-padding missing data, or returning a one-segment value. Zero spectral denominators are unavailable, not zero coherence.

All compared auto/cross spectra must use identical settings and sample intervals. Check installed-library behavior rather than relying on version-specific defaults. The coherence frequency grid and phase-method grid may differ; preserve their actual coordinates and band masks rather than inventing equality through interpolation.

### 6.2.1 Native waveform grid and loaded support

For native rate fs, choice c, and half-width a, define relative metric coordinates `r[k] = -a + k/fs`, with `k = 0, ..., round(2*a*fs)-1`; validate that the duration is representable at that rate. CT026 has 5000/10000 primary/secondary waveform samples. Apply calibration and the configured notch to continuous/padded native data before extracting metric rows. Do not decimate Welch waveforms to 500 Hz.

Align each probe independently using the existing IRIG UTC-to-native-sample mapping: evaluate `s_p(c+r[k])` for every target timestamp, then linearly sample the continuously preprocessed real voltage at those fractional native indices. Thus all sites use the same UTC targets despite probe offsets/drift; equal array positions or the loader's choice-anchored nominal time vector alone do not establish alignment. Exact samples use their own validity; interpolation requires both adjacent, contiguous valid source samples. Do not extrapolate outside synchronization/signal support or bridge missing/invalid support; mark only dependent intervals unavailable. Record the synchronization identity and alignment method/version with the prepared waveform/PSD.

The loader's floor/ceil bracketing samples and filter context are preparation support, not extra Welch observations. Retain them for alignment, then pass only the target half-open metric row to Welch. A 5001-sample bracketing load for an off-grid two-second request must not become a 5001-observation PSD. Power and coherence reuse the identical aligned site row and segment placements. Round 1 coherence requires matching native rates within a pair; report an unsupported-rate mismatch without disabling independent site power or silently resampling one site to another rate.

### 6.3 Per-site total band power

For site A, trial i, selected interval W, and retained frequency band B, calculate:

$$
P_{A,i,B}(W) = \int_{B_{\mathrm{retained}}} \widehat{S}_{AA,i}(f,W)\,df
$$

Here the Welch auto-PSD is calculated only from that trial's selected centered interval, using Section 6.2's settings and `scaling="density"`. For calibrated uV input, PSD has units uV^2/Hz and the integrated result has units **uV^2**. This is average signal power over the selected time interval, integrated over frequency; it is not energy integrated over time. [R5]

Use theta 6–10 Hz and gamma 30–80 Hz with the same configured 58–62 Hz gamma exclusion. Reuse `lfp/power.py::mean_band_power_linear` and its established disjoint-interval quadrature/boundary conventions: it returns mean PSD density and retained bandwidth. Multiply those outputs to recover total band power. Do not label that helper's unconverted mean-density output as total power, switch to `scaling="spectrum"` and sum bins, or average sampled PSD values without frequency-width weighting.

At the default bounds, retained bandwidth is **4 Hz for theta** and **46 Hz for gamma**. Integrate retained gamma intervals separately, never bridge the excluded gap, and keep excluded 58/60/62-Hz sample values from contaminating the boundary estimates, following the existing helper's one-sided boundary treatment. Save the bounds, exclusions, actual PSD grid, retained bandwidths, and integration convention. An unavailable required retained PSD value makes that band's power unavailable.

Retain calibrated voltage units and any explicit conversion. Report uV^2 only when input calibration to uV is verified; never relabel unconverted source units. A valid all-zero trace has zero band power. It may simultaneously have undefined phase or coherence, so do not require those other metrics to be valid before publishing power.

The primary display is absolute integrated power. Session/pre-session baseline normalization, relative-power fractions, theta/gamma ratios, and across-trial power summaries are not required. Gamma covers a wider retained bandwidth than theta, so the two absolute integrated values must not be presented as bandwidth-normalized measures of oscillation strength.

Compute power independently by site and interval. Reuse a site's Welch auto-spectrum in power and coherence only when samples, segment placements, taper/detrend settings, preprocessing, units, and frequency coordinates match exactly. A power-only request must not compute cross spectra, phase, or spike metrics. A single available site remains sufficient for its power track.

Save one band-power scalar per site/trial/band/interval, plus sample and segment counts, validity/reasons, and source/setting identities. Keep the PSD as a compatible reusable product or referenced preparation result so later coherence need not recompute identical auto-spectra.

## 7. Coherence Versus PLV/PPC — Required Explanation

Keep the following distinction in the help text or result description:

| Measure | What is averaged | What the value describes |
|---|---|---|
| PLV | Unit phase-difference vectors over time | Consistency of the LFP phase offset, without direct amplitude weighting |
| LFP–LFP PPC | Pairwise consistency of those same phase-difference observations | A finite-count transformation of squared PLV; not an independent coupling measurement |
| Magnitude-squared coherence | Cross/auto spectra from segments within the trial | Normalized spectral association that depends on spectral amplitudes as well as phase relationships |

Coherence is not simply "PLV plus power" or correlation between amplitude envelopes. Conversely, phase-only weighting does not guarantee reliable phases when an oscillation is weak. [R1, R3, R4]

Coherence and PLV both span 0–1 but are not numerically interchangeable. PPC can be negative. Use distinct metric labels/scales; do not interpret matching numeric thresholds as matching coupling strengths.

Band power has physical voltage-squared units and describes a single site's signal. Give it separate axes and labels; it is not another 0–1 coupling measure or evidence of synchrony by itself.

All three depend on the available window, preprocessing, and estimation choices. A short-window high value is not a significance declaration. None resolves common reference, common input, or volume conduction. This viewer compares descriptive relationships; it does not establish communication, directionality, or mechanism. The known sensitivity of phase-based measures to shared signals motivates this caveat, not an additional corrected-connectivity estimator here. [R6]

## 8. Selected-Unit Spike–LFP PPC

### 8.1 Single-trial estimator

For a selected unit, LFP channel, trial, frequency, and window, sample the LFP phase at every valid spike time within that window. Retain spike timestamps at their available precision; do not convert the unit into coarse PCA-style spike bins before extracting phase.

Reuse and verify the existing phase-at-spike sampling method. If no suitable method exists, use a documented circular/complex interpolation convention rather than linear interpolation of wrapped angles across the -pi/pi discontinuity.

Let N be the number of retained spikes and phi_s their LFP phases:

$$
\mathrm{PPC}_{\mathrm{spike}}(f,W) =
\frac{\left\lvert\sum_{s=1}^{N} e^{i\phi_s(f)}\right\rvert^2 - N}{N(N-1)}
$$

Use **all spike pairs within this single trial/window**, equivalent in pair selection to an ordinary point-field/all-pairs PPC estimator. Do not invoke an estimator that requires spike pairs from different trials. FieldTrip, for example, distinguishes an all-pairs `ppc0` from `ppc1`, which excludes within-trial pairs. Verify the estimator itself, not just its function name. [R7]

Do not pool units, neighboring trials, or behavioral conditions to produce a valid value. Use the same selected centered interval as the LFP track: primary [-1 s, +1 s) or optional secondary [-2 s, +2 s). Before/after estimates are deferred to Round 2.

### 8.2 Support and interpretation

- N<2: PPC unavailable, not zero.
- N>=2: report the descriptive estimate and **contributing spike count**; very small counts remain poorly supported even when the formula is defined.
- Preserve finite negative values.
- Retain the number of excluded spikes if phase data were unavailable, and label partial support.
- Counts can differ by frequency. A band readout shows the minimum-to-maximum valid-spike count across its fixed retained frequencies (or one count explicitly labeled common to all of them); expose exact per-frequency counts. Never sum counts across frequencies or imply a single shared N when none exists.
- Do not impose a new arbitrary biological minimum-spike cutoff. A user-selected display cutoff may be added later, but must not be confused with the estimator's minimum or a significance threshold.

Within-trial bursts and refractory/history structure can affect the sampled phases. The all-pairs calculation is not automatically free of those dependencies; cross-trial PPC variants intended to address them are different estimators. [R3]

Keep the selected-unit track separate from LFP–LFP PPC, with labels identifying unit, source region, and LFP phase channel. Do not interpret a difference between these tracks as a difference between interchangeable sample populations.

### 8.3 Trial inspection

On selecting a trial, reuse existing inspection plots where possible: relevant LFP trace, a specifically identified phase/frequency view, spike timestamps, and a polar spike-phase histogram. Show the selected interval, PPC, spike count, and channel identity together. This does not require implementing an all-unit PPC heatmap or new exemplar-ranking analysis.

## 9. Visual Layout and Navigation

Build a separate synchrony view in the existing Streamlit app. Reuse the implemented PCA timeline's behavioral rendering and persistent navigation, without requiring a PCA result or adding synchrony tracks to the PCA view in Round 1. The PCA spike-bin settings do not apply here. Final panel dimensions, colors, and exact control placement will be refined with Codex.

The user selected **light mode**: opaque white backgrounds with black axes and text. Use readable fonts, labeled plots, and captions identifying the metric and estimation interval.

Suggested track order:

1. Choices and optional behavioral events, using the established real session clock.
2. Optional selected LFP/phase inspection tracks, not every raw trace by default.
3. Per-site theta/gamma integrated-power tracks, labeled with physical units.
4. LFP–LFP phase-consistency tracks, organized by channel pair and theta/gamma, with a PLV/PPC toggle.
5. Coherence tracks with the same pair/band grouping, available separately or alongside phase consistency.
6. Selected-unit spike–LFP PPC, with spike-count support visible.

Use one synchronized elapsed-time axis. Plot one marker per choice for each selected site/band power, metric/pair/band, or unit/channel/band, with the selected centered interval explicit in the panel label and hover. No separate across-trial summary figure is required.

For a selected pair, retain frequency-resolved results and allow inspection as a frequency-by-choice-summary heatmap. Place each choice's column at its real session-time estimation interval, leaving gaps blank. Such a column contains one scalar per frequency for the selected interval; it does not display continuously evaluated synchrony within that interval. A selected-choice frequency spectrum/readout may complement the heatmap.

Preserve real gaps between choices. Do not connect choice-summary markers with lines, interpolate across gaps, or stretch heatmap cells through inter-trial intervals. Any interval tile must be explicitly labeled as a window summary rather than constant instantaneous coupling. Missing values remain unavailable.

Keep comparable metric limits stable while panning: PLV/coherence 0–1, PPC a signed scale that preserves all values or clearly indicates any chosen display saturation. Never normalize every trial independently to look equally strong. Missing data must look different from zero.

Power uses separate nonnegative physical-unit axes with limits fixed from the completed site/band/configuration results while navigating. Do not normalize each trial, divide by interval duration again, or apply an unlabeled log/dB conversion. Power hover also identifies the site/channel, integrated value and units, retained bandwidth, and sample/segment counts.

Hover/result details should include trial ID, pair or unit/channel IDs, frequency/band, metric, actual start/end/center, sample or spike count, and Welch segment count where relevant. Keep notch/exclusion and sampling settings accessible without crowding every label.

Reuse the current timeline chart component, which preserves the browser's panned range across display changes, and the existing explicit viewport/jump controls. Native chart navigation and data already loaded into the figure are different from Python-side data loading; do not require new pan-event callbacks or assume the native `st.plotly_chart` interface supplies them. Explicit viewport controls remain the loading boundary when bounded loading is needed. [R8]

Condition/choice/context filters may highlight or select trials using existing definitions. They must not turn the metric into a cross-trial average or compress the session's time axis.

## 10. Offline Preparation, Loading, and Caching

Keep analysis independent of Streamlit widgets:

```text
Metadata + synchronized LFP/spikes/events
→ Reusable preprocessing / phase or segment spectra
→ Per-trial, per-window metrics and support counts
→ Saved analysis products
→ Timeline viewer loads/selects those products
```

The same computation functions should support explicit local preparation and offline batch execution using existing project infrastructure. Cluster execution is an option after benchmarking, not a prerequisite for viewing a session.

The user selected separate-terminal preparation plus a **Refresh results** control in the viewer. Allow power, coherence, LFP phase consistency, and selected-unit spike PPC to be requested separately with only their actual prerequisites. Publish completed site, pair, or unit/channel and interval products as they finish; those results must be viewable while other requested processing continues. Do not require an entire run to finish before displaying any neural results, or make power/coherence depend on Morlet phase preparation. Starting or monitoring background jobs inside the viewer is outside Round 1.

Save numeric arrays/results with the project's existing format and ordinary configuration metadata. Include session/data identity, mapped channel IDs, unit IDs where applicable, time origin, trial/window coordinates, frequencies, bands/exclusions, estimator/preprocessing settings, and support/validity information. A compact schema/version field is sufficient; no new provenance platform or distributed cache framework is needed.

When multiple completed compatible products have the same component identity, choose the maximum tuple of parsed UTC completion time, run ID, and relative product path, with lexicographic string tie breaks. Show the chosen source run/time in result details. Newer failed, temporary, or unfinished products do not hide an older completed compatible result; label the older result and the newer attempt's status separately. Do not use filesystem modification time or directory traversal order to select results.

Cache by analysis configuration, not viewport. Reuse phase data across LFP pairs and selected units where valid. Panning, event toggles, PLV/PPC display switching, and replotting existing bands must not reread all raw data or recompute every transform. Streamlit caching may support app reuse but does not replace saved offline products. [R9]

Changing a genuine analysis setting can require new preparation. Save the exact interval identity/bounds with each result; selecting primary versus secondary loads the matching saved estimates when available. If an interval has not been prepared, identify it and offer explicit preparation rather than hiding a long calculation behind the selection. Preparing an additional interval or selected unit should reuse compatible existing LFP products with sufficient support.

Do not store an entire dense session timeline filled mostly with empty inter-trial gaps. Keep trial-window coordinates and their results, assembling the visible portion for plotting.

## 11. Runtime Assessment and Efficient Calculation

Runtime assessment is a required implementation deliverable, not a promise of a particular duration. Benchmark representative real sessions on the local workstation before deciding what needs cluster support.

Measure separately:

- Metadata/raw-data loading and synchronization lookup.
- LFP preprocessing and phase extraction, cold versus reusable/cache-loaded.
- Welch auto-PSD preparation for power/coherence, cold versus reusable/cache-loaded, per site and interval.
- Theta/gamma power integration from prepared PSDs and its result writing/loading.
- Welch cross-spectra and coherence for the chosen pairs and window settings, separating reused auto-PSD cost.
- PLV/PPC calculation from already prepared phase data.
- Selected-unit phase sampling and PPC for one unit and a representative small multi-unit selection.
- Result writing/loading, peak memory and file size.
- Figure construction/serialization, initial display, and navigation responsiveness.

Record the session/trial/channel/pair/unit counts, frequency grid, actual sample rate, centered-interval bounds, hardware, and elapsed times. Benchmark both the primary two-second and secondary four-second intervals. Report measurements, not extrapolated guarantees. These Round 1 measurements must be available before adding the Round 2 before/after estimates.

Separate each metric's own computation from shared prerequisite preparation and cache I/O, without double-counting shared work. Report time until the first usable neural product and the additional time until later components become available. Use these measurements to recommend which work should run first or separately; no component's relative speed is assumed in advance.

Straightforward efficiency requirements:

- Reuse compatible phase across pairs/units and both intervals where support exists. Primary-first publication takes precedence over avoiding repeated preparation: on cold runs, prepare primary support first when wider support would delay it. Additional secondary preparation is permitted; record its extra time, support, and reused work. If a transform already provides wider support, reuse it without recomputation. Reuse does not promise identical primary values from newly recomputed, differently padded transforms; preserve each product's actual support identity.
- Calculate PPC from complex sums and counts; never allocate an explicit pairwise phase matrix.
- Reduce each centered interval with ordinary array operations. A rolling-window/cumulative-sum implementation is not required for two interval choices.
- Reuse channel auto-spectra where the Welch segment definitions match exactly. Preserve each interval's exact segment locations when sharing FFTs.
- Reuse those same auto-spectra for band power; record the shared Welch cost once and the incremental power-integration/coherence costs separately. Completed power may be viewed while cross-spectra or phase preparation continue.
- Compute selected units first; do not require all-unit spike–LFP preprocessing.
- Use ordinary batching if arrays are too large. Add multiprocessing/cluster orchestration only when profiling shows a need and existing infrastructure supports it.

Conclude the benchmark with one of: interactive local preparation is adequate; local offline preparation plus cached loading is preferable; or a measured compute/memory bottleneck warrants cluster precomputation. Browser/rendering bottlenecks require viewport/plot-payload changes, not simply a faster cluster.

Do not silently lower frequency resolution, alter sample counts, change estimator definitions, or omit requested trials to meet a runtime target. Any reduced configuration must be explicit.

## 12. Minimal Validity Handling

Use existing valid-signal flags and ordinary finite-data checks. Do not silently pad missing LFP samples with zeros or concatenate surviving disjoint pieces to manufacture a complete window. Mark affected pair/window/frequency results unavailable while retaining independent valid results and behavioral markers.

For phase measures, zero/undefined analytic amplitude has no valid phase. The user accepted complete valid phase support for a requested LFP window/frequency in Round 1, rather than the existing PLV helper's 80% coverage default. Spike PPC may retain the spikes with valid phase support, recording their count and exclusions. Band summaries require all retained frequencies to be available as specified in Section 4.1.

For coherence, reject insufficient complete segments and zero/nonfinite denominators. For PPC, reject fewer than two contributing observations. Floating-point roundoff is not a reason to change the estimand: any small numerical clipping must be limited to documented roundoff behavior, and negative PPC is never clipped as "noise."

Power requires complete valid waveform support for the selected site/interval, the configured complete Welch segments, and finite PSD support for the retained band integral. An affected power result is unavailable rather than zero. Genuine zero power remains valid; do not inherit coherence's positive-denominator or phase's nonzero-amplitude requirements.

No trial-shuffle significance, bootstrap intervals, automatic artifact thresholds, extra common-reference correction, or temporal-dependence correction is part of completion. Reuse existing artifact treatment; add clear inspection caveats rather than a new analysis program.

## 13. Focused Acceptance Checks

Use small seeded synthetic tests, then validate and benchmark the complete Round 1 pipeline on the user-designated session **CT026, 2026-08-28 (August 28, 2026)**. Resolve that session's existing `neural_session.json` from the actual data layout before execution; no filesystem path is assumed here. The user inspects and approves the real-session output before processing additional experimental sessions.

Acceptance checks:

- Metadata-selected channels and source IDs agree; a missing HPC2 does not substitute another channel or stop valid PFC–HPC1 output.
- Metric timestamps translate correctly from each actual choice; no observation is pooled across trials and no plotted line bridges a gap.
- Each choice has one primary [-1 s, +1 s) estimate per requested frequency and pair or unit/channel. The optional secondary selection uses [-2 s, +2 s); markers stay at the actual choice and hover intervals match the observations used.
- Interval boundaries are half-open for both LFP samples and spikes. The 500-Hz phase grid has 1,000/2,000 primary/secondary metric samples; CT026's 2500-Hz waveform grid has 5000/10000, yielding seven/fifteen Welch segments. Test off-grid choices, differently offset/drifting probe clocks, bracketing loads, and unsupported pair-rate mismatches.
- Primary and secondary results are computed from their own observations, not averages of shorter metric values. Missing secondary support does not invalidate a valid primary result.
- Constant nonzero phase offset gives PLV/PPC near one; changing offsets within an interval changes its summary. Test the PPC/PLV identity per frequency.
- Single-sample phase PPC and 0/1-spike PPC are unavailable; finite negative PPC is preserved. A two-spike test matches the cosine of their phase difference.
- Coherence agrees with a trusted Welch implementation for identical settings, and the invalid single-segment shortcut is rejected. Do not require finite-sample independent signals to give exactly zero.
- Power produces one theta and one gamma value per site/trial/selected interval, in verified voltage-squared units. Doubling input voltage multiplies power by four, and a constant PSD density integrates to density times retained bandwidth (4/46 Hz at the defaults).
- Deterministic synthetic input with identical PSDs in every complete Welch segment gives matching primary/secondary power; the four-second result is not multiplied by two merely because its interval is twice as long. Do not require unequal finite-sample estimates from stationary noise to agree within numerical tolerance.
- Power quadrature matches the established disjoint-band helper after multiplying mean PSD by bandwidth. Excluded gamma sample values cannot contaminate the result; missing retained support is unavailable, while an all-zero trace yields valid zero power.
- Power-only execution works with one LFP site and no pairs/spikes/phase products. Identical auto-PSDs can be reused by coherence, and a completed power track remains usable while another component runs or fails. Timings distinguish shared Welch work from power integration and coherence work.
- Complete valid LFP phase support is required: an invalid sample makes the affected interval/frequency unavailable, while independent valid results remain available. Do not inherit an 80% acceptance threshold from the existing helper.
- Reject an otherwise matching legacy phase cache whose saved mask cannot establish `strict_support_v1`, and reject derived phase products from incompatible support semantics.
- With known signals and independent offset/drifting probe clocks, recover the known circular phase differences, LFP PLV/PPC, and phases/PPC at off-grid UTC spike times after full preparation. Compare with an independent analytic/no-drift reference within stated numerical tolerances; matching coordinate arrays alone is insufficient. Include choices far from block start so nominal-block-only timing fails this check.
- Reject a cache with correct support semantics but missing/wrong phase-alignment semantics or different effective amplitude thresholds. At explicit numerical threshold 0.0, test tiny positive coefficients below the generic `1e-12` relative cutoff, plus zero/nonfinite coefficients, and preserve settings on save/load.
- Gamma summaries exclude the configured frequency centers without treating them as zeros; phase angles are not pooled across frequencies. An unavailable retained frequency makes the band summary unavailable rather than changing its frequency set.
- Selected-unit counts match spikes in the half-open window; circular phase sampling behaves correctly across phase wraps.
- Band spike-count readouts expose frequency-dependent ranges and exact per-frequency values without pooling counts.
- Offline-loaded and directly computed results agree for the same configuration, within ordinary numerical tolerance.
- Navigating or toggling visible tracks reuses analysis products and leaves their values unchanged. Runtime measurements are reported.
- A completed component remains viewable while another requested component is unfinished or fails. Refresh loads newly completed compatible products without recomputation or resetting the current time range. Component-only preparation does not run unrelated analyses.
- Duplicate-run discovery is deterministic, falls back to older compatible completed products when a newer attempt is unfinished/failed, and displays source-run identity. Cold-run primary-first behavior permits separately timed secondary preparation.

Round 2 adds checks for the exact before [-2 s, 0) and after [0, +2 s) intervals, including assigning a spike/sample at choice to after only. Those estimates and checks are not Round 1 acceptance requirements.

## 14. Codebase Checks and Completion Scope

Before coding, inspect the existing metadata schema, channel loaders, LFP references, corrected timestamps, frequency/phase caches, spike-phase sampler, condition definitions, plotting components, and result formats. Reuse suitable implementations, but verify their averaging dimension and PPC variant explicitly.

### 14.1 Verified reuse points and differences from this spec

The initial code review identified these concrete boundaries:

- **Behavior and clock:** `population/session_timeline.py::prepare_timeline_choices` retains original trial IDs and source row positions, uses UTC-second `choice_time`, and defines elapsed zero at the first trial start. Actions use 0=right and 1=left. `population/session_timeline_io.py::load_timeline_behavior` also loads optional events. Reuse their ordinary-choice selection and preserve behavioral markers independently of neural availability.
- **Viewer:** `population/timeline_plotting.py::add_timeline_behavior`, `webapp/timeline_view.py::render_timeline_navigation`, and `webapp/timeline_chart.py::render_timeline_chart` expose existing behavior, controls, and persistent chart rendering. Reuse them without fitting PCA or introducing a generic track framework.
- **Channels:** `session_metadata.py` maps each named site to a probe and zero-based saved LFP channel; `site_pairs` contains ordered site-name pairs. Keep these source identities rather than hard-coding anatomical labels or channel numbers.
- **Phase preparation:** `lfp/phase.py::compute_site_phase_trial_tensor` transforms padded continuous blocks before extracting trials. `lfp_summary/synchrony_runtime.py::prepare_phase_run` binds the shared production grid, and `lfp_summary/models.py::PhaseAnalysisConfig` currently specifies 2–100 Hz at 2-Hz spacing, 500-Hz output, Morlet Gaussian width 1.5/window length 1.0/precision 16/L1 normalization, notch enabled, and numerical amplitude threshold 0.0. Its production block loader anchors at block start and advances at nominal rate; reuse its numerical method behind the drift-aware adapter in Section 4.2, not its unchanged timestamp construction. Generic phase helpers default to a different relative threshold, so bind the accepted 0.0 explicitly.
- **PLV validity:** `lfp/synchrony.py::compute_trial_plv_by_frequency` already reduces time samples within individually named intervals, but its default minimum valid fraction is 0.8. This spec requires complete valid LFP phase support; reuse must explicitly enforce that policy. Its `aggregate_trial_plv_bands` helper averages available finite frequencies, which does not satisfy this spec's fixed-frequency-set rule when retained values are unavailable.
- **Different moving estimator:** `lfp/phase.py::compute_within_trial_plv` uses frequency-dependent cycle windows and permits support outside its visible interval. It is not the requested centered-interval estimator and is not needed for Round 1.
- **Spike PPC:** `spike_lfp/phase_locking.py` already provides complex phase sampling and the all-pairs complex-sum PPC formula; `spike_lfp/ppc_kernel.py` also retains trial-resolved sums/counts. Reuse suitable low-level calculations with the requested interval and separate statistics for each selected unit. Do not import the summary workflow's 50-spike reliability threshold, pooled-trial reduction, shuffles, or all-unit orchestration as prerequisites.
- **Cache distinction:** Final Synchrony summary products retain reduced results and inspection traces, not the full frequency-resolved phase tensor. The separate prepared-phase work cache can retain reusable phase/validity arrays. Verify which representation is available and whether its trial, site, frequency, interval, and preprocessing identities match before reusing it.
- **Coherence:** No LFP coherence implementation was found in the reviewed neural-analysis package. `lfp/power.py` supplies an existing Welch-settings reference, but auto-spectra alone cannot supply coherence's cross spectrum. Its per-call notch and PSD interpolation behavior must not be copied in a way that changes the continuous-preprocessing or actual-frequency-grid contracts above.
- **Power:** `lfp/power.py::compute_welch_psd` provides the existing Welch calculation, and `mean_band_power_linear` returns bandwidth-normalized PSD density plus exact retained bandwidth. Multiply them for the requested integrated power. The full Power summary workflow additionally computes before/after epochs, session/pre-session references, and normalized outputs; these are not prerequisites for this choice-centered absolute-power component. Reuse compatible auto-PSDs with coherence without applying a notch twice.

These are code-review findings, not completed implementation or measured performance. A short implementation plan, tests-first workflow, and user approval are still required before writing analysis/viewer code under `AGENTS.md`.

### 14.2 Completion by round

The exact UI layout is intentionally open for refinement with the user. Do not let UI choices alter the single-trial estimands, actual temporal support, channel identities, or support-count reporting.

Round 1 completion includes per-site theta/gamma integrated band power, the three LFP–LFP measures, and selected-unit spike–LFP PPC for both supported centered intervals, the separate timeline view, offline save/load support, and measured local performance including power. Round 2 subsequently adds before [-2 s, 0) and after [0, +2 s) estimates after the first round and its timing assessment are complete. Moving windows, cluster deployment, all-unit coupling, statistical significance, and publication-grade controls are not required.

## Technical References

These sources support estimator definitions and tooling behavior. The accepted window/grid defaults are inspection design choices, and the exact display layout remains open for refinement. The references do not verify the local codebase or metadata or establish that these defaults are optimal.

- **R1:** Spynal phase synchrony documentation — PLV, LFP–LFP PPC, and the PLV-to-PPC identity. Its averaging options must not be assumed to match this spec without inspection. `https://spynal.readthedocs.io/en/stable/modules/sync/phasesync.html`
- **R2:** Vinck et al. (2010), *The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization*. DOI: 10.1016/j.neuroimage.2010.01.073. `https://pubmed.ncbi.nlm.nih.gov/20114076/`
- **R3:** Vinck et al. (2012), *Improved measures of phase-coupling between spikes and the Local Field Potential* — estimator distinctions and dependencies between phase observations. DOI: 10.1007/s10827-011-0374-4. `https://pmc.ncbi.nlm.nih.gov/articles/PMC3394239/`
- **R4:** SciPy `signal.coherence` — magnitude-squared coherence using Welch spectra. `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html`
- **R5:** SciPy `signal.welch` — overlapping segment estimates and estimator settings. `https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html`
- **R6:** Vinck et al. (2011), *An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias*. DOI: 10.1016/j.neuroimage.2011.01.055. `https://pubmed.ncbi.nlm.nih.gov/21276857/`
- **R7:** FieldTrip spike-field tutorial and official estimator source — all-pairs versus across-trial spike-pair definitions. `https://www.fieldtriptoolbox.org/tutorial/intracranial/spikefield/` and `https://github.com/fieldtrip/fieldtrip/blob/master/contrib/spike/ft_spiketriggeredspectrum_stat.m`
- **R8:** Streamlit `st.plotly_chart` — interactive figures and exposed event interface. `https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart`
- **R9:** Streamlit caching overview — reuse of computed products across reruns. `https://docs.streamlit.io/develop/concepts/architecture/caching`
