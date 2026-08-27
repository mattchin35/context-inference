# LFP Analysis Specification

This document defines the requested analyses and visual summaries for LFP power,
LFP synchrony, and spike-phase locking. Unless otherwise noted, frequency bands
and analysis-window limits should remain configurable.

Implementation status, execution contracts, package ownership, and current
authorization gates are maintained only in `docs/Tasks_neural.md`. Status labels
below describe the current product boundary without duplicating that execution
plan.

## 1. LFP Power Analysis

### 1.1 Goal

Assess whether behavioral trial conditions are associated with changes in LFP
spectral power, including changes in specific oscillatory bands of interest.

### 1.2 Single-Trial View

For any selected trial, show:

- The LFP trace.
- The trial power spectral density (PSD) across frequencies.
- PSD on a logarithmic/dB scale.
- Optionally, a trial spectrogram for temporal inspection.

The default analysis window is -2 to +2 s around the alignment event.

### 1.3 Condition Spectra

- Calculate one PSD per trial for a defined analysis window.
- Use -2 to +2 s around the alignment event as the default window.
- Allow the PSD summary window to be selected as:
  - **Whole:** -2 to +2 s
  - **Before:** -2 to 0 s
  - **After:** 0 to +2 s
- For each trial condition, plot the median PSD across trials with variability.
- Show all trial conditions together in the same summary figure.
- Use the interquartile range (IQR) as the default measure of trial-to-trial
  variability around the median.

### 1.4 PSD Normalization

For single-session analyses, provide a toggle between:

- Raw PSD
- Normalized PSD

Perform normalization separately at each frequency rather than using a single
scalar power value. For each frequency $f$:

$$
P_{\mathrm{norm}}(f)
= 10 \log_{10}\left(
\frac{P_{\mathrm{trial}}(f)}{P_{\mathrm{reference}}(f)}
\right),
$$

where $P_{\mathrm{reference}}(f)$ is the reference PSD at the same frequency.

Provide two reference options:

#### Pre-session baseline

- Use the approximately 10 s immediately before the first trial.
- Calculate a reference PSD from this interval.
- Interpret normalized values as spectral power relative to the pre-task
  baseline.

#### Session-median baseline

- Calculate the median PSD at each frequency across the session's trials.
- Use this frequency-specific median spectrum as the reference.
- Interpret normalized values as spectral power relative to typical task-related
  power within that recording session.

The session-median baseline should be the primary normalization for comparisons
between behavioral conditions. The pre-session baseline should remain available
for assessing task activity relative to the pre-task state.

Future population pooling across sites or sessions will require normalization
within each site/session before values are combined.

### 1.5 Band-Power Summaries

#### Frequency bands

- **Theta:** 6-10 Hz
- **Gamma:** 30-80 Hz

Gamma limits should remain configurable.

#### Trial-level calculation

Calculate band power separately for every trial and for both of these windows:

- **Before event:** -2 to 0 s
- **After event:** 0 to +2 s

Calculate band power from the PSD in linear-power units by averaging or
integrating power over the relevant frequency range. For a band spanning
$f_1$ to $f_2$, the mean band power is:

$$
P_{\mathrm{band}}
= \frac{1}{f_2-f_1}\int_{f_1}^{f_2}P(f)\,df.
$$

For theta, for example:

$$
P_{\mathrm{theta}}
= \frac{1}{10-6}\int_{6}^{10}P(f)\,df.
$$

Use an equivalent calculation for gamma.

For normalized band-power values, compare the trial band power with the
corresponding band power from the selected reference spectrum and express the
ratio in dB:

$$
P_{\mathrm{band,norm}}
= 10\log_{10}\left(
\frac{P_{\mathrm{band,trial}}}{P_{\mathrm{band,reference}}}
\right).
$$

#### Band-power summary figure

- Summarize trial distributions using the median and IQR.
- Place trial conditions horizontally.
- Within each condition, show four closely spaced values in this order:
  **theta before -> theta after -> gamma before -> gamma after**.
- Leave a larger horizontal gap between conditions than between the four
  measurements within a condition.
- Show all trial conditions together in the same figure.

The figure should make it easy to compare:

- Before versus after the alignment event.
- Theta versus gamma.
- Differences across behavioral conditions.

### 1.6 Interpretation and Quality Control

- Use the full PSD condition figure to distinguish frequency-specific
  oscillatory effects from broadband power changes.
- Exclude or notch line-noise frequencies as appropriate.
- Treat broad gamma elevations cautiously because movement artifacts, spike
  contamination, or broadband transients can also increase apparent 30-80 Hz
  power.

## 2. LFP Synchrony Analysis

### 2.1 Goal

Assess whether behavioral trial conditions show changes in:

- Single-site, event-locked phase consistency across trials.
- Inter-site phase consistency across trials.
- Within-trial inter-site phase synchrony.

Analyze theta and gamma separately and preserve the before/after-event
structure.

### 2.2 Frequency Bands and Analysis Windows

Frequency bands:

- **Theta:** 6-10 Hz
- **Gamma:** 30-80 Hz

Frequency-band limits should remain configurable.

Default alignment windows:

- **Whole:** -2 to +2 s
- **Before:** -2 to 0 s
- **After:** 0 to +2 s

### 2.3 Single-Site ITPC

Calculate inter-trial phase coherence (ITPC) separately for each site/channel
and behavioral trial condition. ITPC measures whether the phase at a given site
is consistently aligned to the behavioral event across trials.

For each site and condition:

- Calculate frequency x time ITPC.
- Summarize theta-band ITPC.
- Summarize gamma-band ITPC.
- Allow summary over the whole, before, or after window.
- Retain the underlying frequency x time ITPC representation for inspection.

### 2.4 Inter-Site ISPC

Calculate inter-site phase clustering (ISPC) for each relevant site pair and
behavioral condition. For trial $k$, define the phase difference as:

$$
\Delta\phi_k(f,t)
= \phi_{1,k}(f,t)-\phi_{2,k}(f,t).
$$

Then calculate:

$$
\mathrm{ISPC}(f,t)
= \left|
\frac{1}{N}\sum_{k=1}^{N}e^{i\Delta\phi_k(f,t)}
\right|.
$$

ISPC measures whether the phase relationship between two sites is consistent
across trials. The consistent phase relationship does not need to have zero lag.

Relevant pairs should include:

- PFC-HPC1
- PFC-HPC2, where applicable
- HPC1-HPC2
- Optionally, within-PFC pairs as within-region controls

For each pair and condition:

- Calculate frequency x time ISPC.
- Summarize theta and gamma separately.
- Allow summary over the whole, before, or after window.

### 2.5 ITPC/ISPC Summary Figures

- Create separate summary plots for theta and gamma, using identical formatting.
- Show all behavioral trial conditions together horizontally.
- Within each condition, display the relevant single-site and inter-site metrics.
  For example: **PFC ITPC | HPC1 ITPC | HPC2 ITPC | PFC-HPC1 ISPC |
  PFC-HPC2 ISPC | HPC1-HPC2 ISPC**.
- Allow selection of **Whole**, **Before**, or **After** as the temporal summary
  window.
- Do not combine PLV into this plot. ITPC and ISPC are across-trial statistics,
  whereas PLV is calculated within individual trials.

### 2.6 Within-Trial PLV

Calculate phase-locking value (PLV) separately for every trial, frequency band,
temporal window, and site pair. For trial $k$:

$$
\mathrm{PLV}_k
= \left|
\frac{1}{T}\sum_t
e^{i[\phi_{1,k}(t)-\phi_{2,k}(t)]}
\right|.
$$

PLV measures how consistently two sites maintain a phase relationship over time
within an individual trial.

Calculate PLV for:

- PFC-HPC1
- PFC-HPC2, where applicable
- HPC1-HPC2

For each trial, calculate:

- Theta PLV, -2 to 0 s
- Theta PLV, 0 to +2 s
- Gamma PLV, -2 to 0 s
- Gamma PLV, 0 to +2 s
- Whole-window PLV, -2 to +2 s

### 2.7 PLV Summary Figures

- Create separate theta and gamma plots.
- Show all behavioral trial conditions together horizontally.
- Within each condition, organize site pairs with before/after values adjacent to
  one another. For example: **PFC-HPC1 before | PFC-HPC1 after | PFC-HPC2
  before | PFC-HPC2 after | HPC1-HPC2 before | HPC1-HPC2 after**.
- Use the median across trials as the central value.
- Use IQR as the default variability measure.
- Use larger horizontal spacing between behavioral conditions than between
  measurements within a condition.

### 2.8 High- and Low-Synchrony Trial Examples

Provide visual examples of trials with high and low within-trial phase
synchrony. For a selected frequency band and site pair:

- Identify a high-PLV trial, preferably near the 95th percentile rather than the
  absolute maximum.
- Identify a low-PLV trial, preferably near the 5th percentile rather than the
  absolute minimum.
- Display PFC, HPC1, and HPC2 for both examples.
- Show band-filtered LFP traces.
- Show instantaneous phase or another phase overlay that makes relative phase
  alignment visible.
- Show the PLV value used to select the exemplar.

Allow exemplar selection using:

- PFC-HPC1 PLV
- PFC-HPC2 PLV
- HPC1-HPC2 PLV

Even when an exemplar is selected using one pair, display all three signals so
that broader cross-region synchrony can be inspected visually.

### 2.9 Phase-Offset Information

For ISPC and PLV calculations, preserve the preferred phase offset in addition
to synchrony magnitude.

Synchrony magnitude:

$$
\left|\left\langle e^{i\Delta\phi}\right\rangle\right|
$$

Preferred phase offset:

$$
\arg\left(\left\langle e^{i\Delta\phi}\right\rangle\right)
$$

The preferred phase offset does not need to appear in the primary summary
figures, but it should remain available for inspection. Two conditions can have
similar synchrony strengths but different phase relationships.

### 2.10 Interpretation

The three principal synchrony analyses answer distinct questions:

- **ITPC:** Is an individual site's phase consistently aligned to the behavioral
  event across trials?
- **ISPC:** Is the phase relationship between two sites reproducible across
  trials?
- **PLV:** Within individual trials, how stably do two sites maintain a phase
  relationship over time?

## 3. Spike-Phase Locking Analysis

### 3.1 Goal

Assess whether units show behavior-dependent phase locking to LFP oscillations,
with emphasis on theta and gamma bands. Summarize both locking strength and
prevalence across behavioral conditions.

### 3.2 Frequency Bands and Analysis Windows

Frequency bands:

- **Theta:** 6-10 Hz
- **Gamma:** 30-80 Hz

Frequency ranges should remain configurable.

Calculate pairwise phase consistency (PPC) over:

- **Before event:** -2 to 0 s
- **After event:** 0 to +2 s
- **Whole window:** -2 to +2 s

The pre/post limits should remain adjustable.

### 3.3 Unit-Level PPC Calculation

For each unit, behavioral condition, LFP site, and analysis window:

- Extract the instantaneous LFP phase at each spike time.
- Calculate frequency-resolved PPC.
- Retain PPC as a function of frequency rather than only as theta/gamma summary
  values.
- Also calculate band-level theta and gamma PPC summaries for compact
  comparisons.
- Preserve slightly negative PPC values rather than clipping them to zero.

### 3.4 All-Unit PPC Heatmaps

Create one heatmap for each behavioral condition with:

- **Rows:** units
- **Columns:** frequency
- **Value:** PPC strength

Allow selection of:

- Whole, before, or after epoch
- LFP site or region used for phase
- Left versus right choice
- Left versus right context

Choice and context should be separate selectors.

#### Unit ordering

Use a consistent unit order across condition maps so the same row corresponds to
the same unit.

The default reference ordering should sort units by theta-band PPC strength in
the 6-10 Hz range, using a selected reference condition and epoch. Preserve this
ordering while displaying other conditions.

Also allow:

- Gamma-based ordering.
- Ordering based on a selected condition.
- Optional re-sorting by the currently displayed condition for exploratory
  inspection.

The primary comparison view should preserve a fixed reference ordering.

### 3.5 Population PPC Summary Map

Collapse the unit-level PPC maps across units. Create a condition x frequency
heatmap with:

- **Rows:** behavioral conditions
- **Columns:** frequency
- **Value:** median PPC across units

Provide separate views or a selector for **Before**, **After**, and **Whole**.
This map should show how the typical unit's locking strength varies across
frequency and behavioral condition.

### 3.6 Significant-Locking Prevalence Map

Create a second heatmap with the same dimensions:

- **Rows:** behavioral conditions
- **Columns:** frequency
- **Value:** fraction of units that are significantly phase-locked

Use fraction rather than raw count as the primary measure so differences in the
number of available units do not dominate comparisons. Retain or display the
number of contributing units where useful.

### 3.7 Statistical Significance

#### Primary null: same-condition trial shuffle

Use a trial-shuffle permutation test as the primary significance test. For each
unit and behavioral condition:

1. Preserve each trial's spike train and its within-trial timing.
2. Pair the spike train with LFP phase from a different trial in the same
   behavioral condition.
3. Recalculate PPC.
4. Repeat across many shuffles to generate a null PPC distribution.

This procedure preserves:

- Trial spike counts.
- Within-trial firing-rate structure.
- LFP spectral structure.
- Behavioral-condition structure.

It breaks the trial-specific spike-LFP relationship.

Calculate the permutation p-value as:

$$
p =
\frac{
1 + \sum_{s=1}^{S}
I\left(\mathrm{PPC}^{(s)}_{\mathrm{shuffle}}
\geq \mathrm{PPC}_{\mathrm{observed}}\right)
}{S+1},
$$

where $S$ is the number of shuffles.

- Apply an appropriate multiple-frequency correction when significance is
  evaluated independently across many frequencies.
- Enforce a minimum spike-count requirement before treating PPC or significance
  estimates as reliable.

#### Important trial-shuffle caveat

The trial-shuffle null is intentionally conservative when both spikes and LFP
phase are strongly aligned to the same behavioral event. For example, a neuron
may consistently increase firing 300 ms after choice while theta phase is also
highly consistent 300 ms after choice across trials. Pairing the spike train from
one trial with the LFP from another may therefore preserve apparent spike-phase
alignment because both signals are independently event-locked.

The trial-shuffle test primarily asks:

> Is spike-phase locking stronger than expected from the event-locked temporal
> structure shared across trials?

It does not strictly test whether spikes are nonuniformly distributed over phase
independently of all event-related structure. For these behavioral analyses,
this conservatism is generally desirable because it avoids labeling a unit as
intrinsically phase-locked when independent event locking of spikes and LFP can
explain the apparent locking.

#### Optional alternative null

Retain a within-trial temporal-shift null as an optional robustness analysis. For
example:

1. Circularly shift the spike train relative to the LFP by a random temporal
   offset.
2. Recalculate PPC repeatedly.

This procedure more directly tests whether coupling exceeds what would be
expected from the temporal structure within that trial. Treat it as a
complementary null rather than the default replacement for the trial shuffle.

### 3.8 Band-Level Summary Figure

For every unit and condition, calculate:

- Theta PPC before
- Theta PPC after
- Gamma PPC before
- Gamma PPC after

Summary requirements:

- Summarize across units using the median and IQR.
- Show all behavioral conditions horizontally.
- Within each condition, order measurements as: **theta before | theta after |
  gamma before | gamma after**.
- Use tighter spacing within a condition and larger spacing between conditions.

### 3.9 Behavioral Conditions and Splits

Include:

- Correct rewarded
- Correct omission
- Incorrect
- Switch
- Stay

For switch/stay analyses, preserve the before/after alignment structure.

Allow behavioral splitting by:

- Left/right choice
- Left/right context

### 3.10 High- and Low-PPC Exemplar Units

For a selected condition, frequency band, epoch, and LFP site:

- Select a high-locking exemplar near the 95th percentile of the PPC
  distribution.
- Select a low-locking exemplar near the 5th percentile of the PPC distribution.
- Do not select the absolute maximum or minimum by default.

For each exemplar, show:

- LFP trace
- Band-filtered LFP
- Instantaneous phase
- Unit spike times
- Polar histogram of spike phase
- PPC value
- Preferred phase

Use identical formatting for the high- and low-PPC plots.

### 3.11 Preferred Phase

In addition to PPC strength, retain each unit's preferred phase:

$$
\arg\left(
\frac{1}{N}\sum_{j=1}^{N}e^{i\phi_j}
\right).
$$

Preferred phase does not need to appear in the primary PPC summary figures, but
it should remain available for inspection. Two behavioral conditions can show
similar PPC strength while units lock to different portions of the oscillatory
cycle.

### 3.12 Interpretation

The main outputs answer complementary questions:

- **Unit x frequency PPC maps:** Which units lock, and at what frequencies?
- **Condition x frequency median PPC map:** How does typical locking strength
  vary with behavior?
- **Condition x frequency significant-fraction map:** How widespread is
  statistically reliable locking?
- **Band-level before/after summaries:** How do theta and gamma locking change
  around the behavioral event?
- **High/low PPC exemplars:** What does strong versus absent spike-phase locking
  look like in the underlying data?

## 4. Current webapp control status

| Control or view | Status | Current requirement |
| --- | --- | --- |
| Power compute/cache inspection | Implemented | Preserve compatible Power computation and limited cached plotting |
| Synchrony numerical computation/reporting | Implemented outside Streamlit | Add the production Streamlit action and complete cached selectors |
| Spike-phase numerical bridge/preview runner | Implemented outside Streamlit | Optimize through approved WP5C gates before CT026 preview; then add the production action |
| Synchrony and Spike-phase Streamlit controls | Planned | Use the composed production dependency boundary; do not calculate numerics in Streamlit |
| Compute All | Planned | Run Power, Synchrony, and Spike phase sequentially with isolated component failures |
| Active unit-population selection | Planned and required | The selected population, stable unit identities, channels, and quality settings must be passed explicitly into Spike-phase configuration |
| Progress display | Planned and required | Display stage, completed/total work, elapsed time, and ETA only when available |
| Cached Power selectors/plots | Partially implemented | Expand from the limited Power preview to the complete approved cached views |
| Cached Synchrony/PPC selectors and plots | Planned | Load only compatible final component files and expose counts, warnings, exclusions, and stale differences |

Prepared-phase tensors, schedules, PPC block checkpoints, incomplete work, and
other intermediate files are execution artifacts. They must never be listed,
loaded, or displayed by the webapp as compatible scientific components. Only a
validated final component committed through `manifest.json` is eligible for
scientific inspection.
