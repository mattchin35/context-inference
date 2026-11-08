# Session Timeline and PCA Activity Viewer Spec

**Status:** Version 1 — staged implementation specification for Codex.

**Purpose:** Exploratory inspection of individual choices, behavioral events, and regional population activity on the real session timeline. This is not a trial average, a decoding analysis, or a publication-finalized dimensionality estimator.

## 1. Implementation Priorities

| Stage | Status | Deliverable |
|---|---|---|
| **1. Session timeline and ongoing PCA visualization** | **Mandatory first implementation** | Real-time choices/events, synchronized regional PC activity, navigation, and session-wide 80%/90%/95% variance-coverage summaries |
| **2. Within-window variance on existing session PC axes** | **Required subsequent implementation** | Fully specified fixed-axis variance participation metric for each choice; benchmark on the local workstation before deciding whether cluster precomputation is needed |
| **3. New PCA fitted within each choice window** | **Optional alternative strategy** | Implement only if Stage 2 is scientifically unsatisfactory and this additional strategy is selected; not a prerequisite for Stages 1 or 2 |

Deliver and inspect Stage 1 before adding Stage 2. Do not make the mandatory viewer depend on per-choice dimensionality calculations, a cluster connection, or Stage 3. Stage 2 must be implemented and evaluated, not left as an unspecified future idea. Cluster execution is conditional on measured need; it is not an assumed requirement.

The main plotted PC axes always remain session-wide axes. Neither subsequent metric may replace them with independently fitted trial-specific axes.

## 2. Scientific and Display Goal

Show choices at their actual elapsed-session times, with optional behavioral events alongside them and neural activity underneath.

The neural data are binned relative to each choice but displayed at their actual session times. A user should be able to pan through the session and inspect what the regional PCs did around each individual choice.

Preserve elapsed time between choices. Draw neural activity only during the selected choice-centered windows; leave intervening neural intervals blank rather than compressing time or inserting artificial activity. Behavioral event tracks may continue through those gaps.

For example, choices at 100 s and 114 s have neural windows at 98–102 s and 112–116 s. The ten-second gap from 102 to 112 s remains ten seconds wide on the plot.

Run one simultaneously recorded session at a time. PFC and HPC have separate regional PCAs and separate neural panels.

## 3. Input Data and Time Conventions

Reuse the existing synchronized neural/behavioral timestamps, session loaders, unit-quality selection, regional assignments, and event definitions.

Required inputs:

- Session time origin and trial identifiers.
- Choice timestamp and left/right action for each trial.
- Spike times or equivalent suitable unsmoothed binned activity, with stable unit identifiers and regional membership.

Optional event inputs:

- Trial start.
- Left licks.
- Right licks.
- LED on and LED off, with channel/side identity when the source distinguishes them.

Codex must verify source column names, event meanings, synchronization, time units, and left/right codes against the existing codebase. Do not infer codes from an example or replace the established synchronization pipeline.

Use **elapsed seconds from a single session origin** for all panels; format tick labels as seconds or minutes:seconds as appropriate. Retain trial identifiers for hover and navigation. Do not round choice times to a session-wide bin grid.

The task's choice windows do not overlap between trials. Do not add window-overlap detection, purging, boundary-trial exclusion, or an elaborate overlap policy.

If an optional event stream is absent, mark that track unavailable and continue. If a choice lacks a complete valid neural window for a region, keep its behavioral marker but leave that region's neural window unavailable. Use complete valid windows for that region's PCA and variance summaries; do not pad missing neural samples with zeros.

## 4. Choice-Centered Binning

Use the half-open interval **[-2 s, +2 s)** around each choice.

| Bin width | Number of bins per four-second window |
|---|---:|
| 20 ms | 200 |
| 50 ms | 80 |
| 100 ms | 40 |
| **500 ms — default** | **8** |

Use spike counts per bin as the simple default activity representation, with no added temporal smoothing. An existing equivalent firing-rate representation may be reused if its conversion and absence of future-bin smoothing are verified and documented.

Bin edges are defined relative to the actual choice timestamp, then translated to elapsed-session time. With 500-ms bins, relative edges are -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, and 2 s.

These are individual-trial measurements. Do not average across trials, time-warp trials, interpolate between choices, or compute a trial-averaged trajectory.

The same selected bin size applies to the mandatory activity display and subsequent window metrics. Changing bin size selects a different PCA/analysis configuration; it can change the component directions. No cross-resolution PC-axis matching is required.

## 5. Stage 1 — Mandatory Session-Wide PCA

### 5.1 Fitting scope and preprocessing

For each session, bin size, fixed unit selection, and region:

1. Collect all complete valid choice windows in that region, not just the visible part of the session.
2. Form a matrix with **rows = trial × time-bin observations** and **columns = units**.
3. Z-score each unit once using its mean and standard deviation across those pooled observations.
4. Fit one shared-temporal regional PCA to the standardized activity.
5. Project each choice window into those same regional axes.
6. Use the ordinary signed PC scores directly.

Use separate PFC and HPC PCAs, **whitening disabled**, and **no post-PCA z-scoring or additional score scaling**. Ordinary PCA centering remains part of fitting. Standard PCA centers its inputs, and its components are ordered by decreasing explained variance; whitening is a separate rescaling operation. [R3]

This is descriptive session-level visualization, not held-out prediction. Do not import outer/inner cross-validation, elastic-net fitting, or training-fold PCA from the decoding spec.

Reuse the same fitted basis and unit scales when panning, selecting a trial, changing the displayed PC subset, or toggling event tracks. Do not fit PCA on the viewport or independently for each choice. Keep the unit population fixed across the windows compared in a configuration.

Exclude session-constant units from standardization/PCA using ordinary existing validity handling and report the resulting unit count. Do not change the included units independently for each choice to manufacture valid window metrics.

### 5.2 Displayed PCs versus analysis basis

The number of PCs displayed is a display setting, not the size of the variance denominator or a hard cap on the analysis basis.

- Provide a selected PC count or rank range per region; an initial view of the first 10 available PCs is a convenient UI default, not an analysis limit.
- Allow many PCs to be displayed with sufficient panel height or a PC-rank range control.
- Keep rows in **session-wide explained-variance order**: PC1, PC2, and so on.
- Include the session-wide explained-variance percentage in row labels or hover.
- Keep signs and component identities unchanged within a loaded configuration. PC sign is arbitrary; do not compare raw signs across independently fitted configurations as though they were biological labels.

Compute the available session variance spectrum sufficiently completely to determine the footer thresholds. Prefer retaining the full available nonzero session-PC spectrum and loadings for reuse in Stage 2. Projecting or retaining scores for only the visible PCs initially is acceptable as a storage choice; it must not redefine total variance or permanently prevent access to the other axes.

Do not normalize the first ten displayed components to 100%, and do not assume that the components covering 95% globally also cover 95% of every individual window.

### 5.3 Mandatory session-wide variance footer

At the bottom of the viewer, report separately for PFC and HPC the smallest **leading** PC count needed for 80%, 90%, and 95% of the variance in the full PCA-fitting dataset.

Let $e_j$ be the session variance explained by component $j$, ordered largest first. Let $V_{\mathrm{session}}$ be the total variance across the original session-standardized unit columns in the pooled fitting matrix.

$$
C(K) = \frac{\sum_{j=1}^{K} e_j}{V_{\mathrm{session}}}
$$

$$
K_q = \min\{K : C(K) \ge q\}, \qquad q \in \{0.80, 0.90, 0.95\}
$$

Use the same variance convention in numerator and denominator. For sample covariance/eigenvalues, use the corresponding sample-variance normalization. The denominator is total variance in the fitted standardized unit activity, not just in the plotted PCs. [R3]

Example format, with placeholders rather than expected results:

```text
Variance coverage — all eligible ±2 s choice windows, 500-ms bins
PFC: 80% = … PCs | 90% = … PCs | 95% = … PCs
HPC: 80% = … PCs | 90% = … PCs | 95% = … PCs
```

Include usable unit/window counts and enough configuration information to interpret the numbers. The footer does not change when the viewport changes. If a deliberately truncated computation cannot establish a threshold, report it as not reached by the retained basis, not as the retained count or a renormalized result.

## 6. Stage 1 — Mandatory Synchronized Viewer

### 6.1 Tracks

Use one shared elapsed-session-time axis for:

1. **Choices:** left/right markers in distinguishable lanes at their actual times. Show trial ID, action, and timestamp on hover.
2. **Optional behavioral events:** separate trial-start, left-lick, and right-lick tracks. Show LED-on intervals between verified on/off events where available; expose the original transition times. Keep separate LED identities when appropriate. Do not invent an on/off boundary when its state is unknown.
3. **PFC PCs:** signed-score heatmap during eligible choice windows.
4. **HPC PCs:** equivalent heatmap with its own regional PC ordering and scale.
5. **Navigation and variance footer.**

All event tracks may be independently toggled. Missing HPC data, for example, must not prevent viewing valid PFC or behavioral data.

The primary many-PC display is a heatmap. An additional stacked-trace view for a small PC selection is optional convenience, not a completion requirement. Leave room for future neural tracks without implementing a generic visualization plugin framework now.

### 6.2 Heatmap geometry and scale

Use actual temporal bin edges so every cell occupies the bin's real time interval. Plot explicit gaps between windows; do not allow a cell to stretch across an inter-choice gap. Plotly supports explicit heatmap cell-edge coordinates and missing values. [R4]

No temporal interpolation, gap-filling, or interpolation between PC rows is required. A raw PC score of zero must look different from unavailable or outside-window data. For optional traces, break the line between windows.

Use a signed color scale centered on zero, with fixed limits for each region/configuration while browsing. Do not autoscale to each choice or viewport, normalize each PC row independently, or alter scores for display. Choose limits from the configuration's scores or an explicitly labeled manual range; any saturation must be visible as a display choice rather than silently changing the measurements.

Hover should identify the region, PC rank, trial, absolute bin interval, event-relative bin interval, and score. Increasing the displayed PC count must not refit PCA, reorder existing rows, or change footer coverage counts.

### 6.3 Streamlit navigation

Use the existing Streamlit app. Prefer one Plotly figure with synchronized subplot x-axes and native horizontal pan/zoom plus a session range navigator. Streamlit supports interactive Plotly figures, and Plotly documents range sliders with stacked subplots. [R1, R2]

Provide an explicit visible-duration control and a jump-to-trial control; a 60-s initial viewport is a simple UI default. Preserve a reasonable viewport across display-only reruns where the installed tooling supports it. Do not introduce custom JavaScript solely to provide navigation already available through normal controls.

Distinguish chart panning from scrolling the page vertically. A mouse wheel need not advance session time; use documented, discoverable controls rather than relying on ambiguous wheel behavior.

Do not assume the native Streamlit chart interface sends arbitrary pan/zoom events back to Python; its documented event interface supports selections. Check the installed version. [R1]

Choose the simplest practical data-loading approach after testing a representative session:

- If the selected data fit comfortably in the figure, load them once and pan within the chart.
- If rendering/serialization is too heavy, use the explicit time-range controls to load a bounded interval and allow panning within that interval. Keep a compact behavioral/session navigator for moving elsewhere.

The entire session must remain reachable. Viewport loading must not change the PCA basis or compress gaps. Do not require a custom component or server-side streaming framework before ordinary plotting has been evaluated.

### 6.4 Caching

Separate neural binning/PCA/projection from figure construction. Cache analysis products by the actual data/analysis configuration, using existing project conventions and Streamlit facilities where appropriate. Streamlit caching is designed to avoid repeating expensive transformations across reruns. [R5]

Panning, selecting displayed PC rows, and toggling events should reuse computed products. A new session, bin size, or explicit unit selection can require a different cached analysis. Avoid a dense all-session neural grid full of empty gaps when per-window arrays suffice.

## 7. Stage 2 — Local Variance on the Existing Session PC Axes

### 7.1 Question and primary metric

After the viewer is working, implement:

> For this choice window, how many of the existing session-wide PC axes account for 95% of its within-window variance?

Use the fixed regional basis from Stage 1. **Do not refit or rotate axes for this calculation.**

The primary count uses the session axes with the **largest window-specific variance contributions**, regardless of their global rank. Sorting contributions for this calculation does not reorder the main heatmap.

This is a **session-basis variance-participation measure**, not an estimate of independent latent processes or a rotationally invariant local dimension.

### 7.2 Exact calculation

For one region, let:

- $i$ identify a choice window.
- $b$ identify one of its $B_i$ time bins.
- $u$ identify an included unit.
- $j$ identify an existing session PC.
- $x_{i,b,u}$ be unit activity after the Stage 1 session-level unit standardization.
- $z_{i,b,j}$ be its score on session PC $j$.

Use the full four-second window and the same binning and units as the viewer.

First calculate the within-window mean and sample variance along each existing PC:

$$
\bar{z}_{i,j} = \frac{1}{B_i}\sum_{b=1}^{B_i}z_{i,b,j}
$$

$$
v_{i,j} = \frac{1}{B_i-1}\sum_{b=1}^{B_i}(z_{i,b,j}-\bar{z}_{i,j})^2
$$

Calculate total within-window variance directly from the same standardized unit activity:

$$
\bar{x}_{i,u} = \frac{1}{B_i}\sum_{b=1}^{B_i}x_{i,b,u}
$$

$$
V_i = \sum_u\frac{1}{B_i-1}\sum_{b=1}^{B_i}(x_{i,b,u}-\bar{x}_{i,u})^2
$$

The contribution of session PC $j$ is:

$$
f_{i,j} = \frac{v_{i,j}}{V_i}
$$

For a complete orthonormal basis spanning these window data, the contributions sum to one, up to numerical precision. Do not substitute the stored session-wide explained-variance ratios for $f_{i,j}$; those describe different observations.

Let $\pi_i(1),\pi_i(2),\ldots$ list the original session-PC identities in descending order of $v_{i,j}$. Then:

$$
K_{95,i}^{\mathrm{fixed}} =
\min\left\{k : \sum_{m=1}^{k} f_{i,\pi_i(m)} \ge 0.95\right\}
$$

Keep the selected PC identities and their individual/cumulative contributions. The main output is the smallest count across the **existing fixed axes**, not the highest global rank encountered.

Example: if PC2, PC7, and PC12 contribute 60%, 25%, and 10% of the window's variance, the fixed-axis count is 3 and the selected identities are PC2/PC7/PC12. It is not 12. The viewer rows remain PC1, PC2, PC3, etc.

Within-window subtraction of the means is part of variance calculation, not an extra standardization step. Do not divide by within-window unit SDs or PC SDs.

### 7.3 Complete versus truncated basis

The normal Stage 2 computation should consider **all available nonzero session PCs**, not only the visible rows or the leading PCs explaining 95% globally. A globally low-variance axis may contribute strongly within a particular choice window.

Use $V_i$ from the original standardized unit activity as the denominator even when only some scores have been computed. Do not renormalize a retained subset to 100%.

If a deliberately limited basis is used because of resources:

- Mark the result as restricted to the retained session PCs.
- If their total contribution is below 95%, report **95% not reached by retained basis**.
- Even if they reach 95%, do not describe their count as the exact minimum over all session PCs: an omitted axis could change the ranking and the minimum.

The intended first implementation should avoid this ambiguity by using the full available basis. Reuse existing loadings and compute additional scores as needed; do not change the plotted basis while expanding coverage.

### 7.4 Display and interpretation

Add one compact per-choice metric track or selected-choice readout per region, aligned to the actual choice timestamps.

Suggested label:

> **Session PCs capturing 95% of within-window variance**

Retain/display on selection:

- $K_{95,i}^{\mathrm{fixed}}$.
- Selected session-PC identities and their contributions.
- Total within-window variance $V_i$.
- Bin size, number of time bins, unit count, and basis completeness.

Display values at choices; do not invent a continuously evaluated dimensionality signal in the gaps. Changing the displayed PC range must not change these numbers.

Caveat:

> Descriptive participation in fixed session axes. Measures variation during the window, not its mean displacement or the number of independent neural computations. Sensitive to sampling, bin size, unit scaling, and noise.

A PC that remains strongly elevated throughout a window can have little within-window variance. This metric deliberately removes the window mean; do not substitute mean squared score/energy without labeling it as a different analysis.

Session PCs need not remain uncorrelated inside an individual window. Multiple fixed axes can vary together and describe one diagonal trajectory. Consequently, this fixed-axis count can exceed $B_i-1$; do not impose the local-refit rank ceiling on it.

If $V_i=0$, fewer than two valid bins exist, or the required data are unavailable, mark the metric unavailable rather than zero-dimensional. Do not treat unavailable results as evidence of low dimensionality. No significance test, FDR correction, bias-correction machinery, or CV is required.

## 8. Stage 2 — Local Performance Assessment and Cluster Decision

### 8.1 Benchmark locally before choosing infrastructure

Measure the actual implementation on the workstation used for inspection. Include a representative real session and a larger available session when practical, testing both the default 500-ms setting and the 20-ms setting. The intermediate bin sizes must remain supported; exhaustive hardware benchmarking is not required.

Separate these costs:

| Measurement | Why keep it separate? |
|---|---|
| Loading/binning and session PCA | Shared Stage 1 preparation, not new work caused solely by Stage 2 |
| Projection of any additional session axes | May be needed to obtain complete window coverage |
| Per-window variances, sorting, and threshold counts | Incremental Stage 2 computation |
| Peak process memory / major array sizes | Determines whether local processing fits comfortably |
| Cache load and warm-view updates | Measures repeat inspection, not cold recomputation |
| Figure construction/serialization and browser responsiveness | A display bottleneck is not automatically a compute bottleneck |

Record session/trial/unit/PC counts, bin size, machine/CPU and available memory, elapsed times, and whether each result is measured or estimated. A small benchmark table is sufficient. Do not invent timings or adopt unmeasured latency guarantees.

### 8.2 Keep the implementation simple

When PC scores are already available, this calculation is per-window variance reduction, sorting, and cumulative sums; it does not require a new PCA or a full unit-by-unit covariance matrix for every trial.

Use straightforward array operations, cache completed summaries, and process window batches if a single large score array would unnecessarily exceed memory. Do not recompute all windows every time the viewport changes. No GPU, parallel framework, or cluster scheduler is required merely because Stage 2 exists.

### 8.3 Decision after measuring

- If local preprocessing and cached interaction are practical, keep the calculation local.
- If CPU time or memory is the actual limitation after ordinary reuse/caching, document the measurements and consider precomputing the **same products** with existing cluster infrastructure.
- If the bottleneck is browser payload or rendering, first reduce loaded/visible data or use bounded-viewport rendering. Moving numerical work to a cluster does not reduce an unchanged browser payload.

If cluster support is justified, keep it a batch producer of the same basis/scores/summaries that the viewer can load. Do not make interactive navigation depend on a live cluster job, invent remote API availability, or build a new distributed-computing framework. The need for cluster support remains a measured implementation decision, not an unresolved scientific definition.

## 9. Stage 3 — Optional Window-Specific New PCA

### 9.1 When this strategy is relevant

Describe this strategy now, but implement it only if the fixed-axis analysis is unsatisfactory for the intended scientific inspection and the additional strategy is selected.

It answers a different question:

> How many locally optimized linear dimensions describe the activity variation within this particular choice window?

For example, two session axes could rise and fall together and form one local diagonal direction. Stage 2 can require both fixed axes; a newly fitted local PCA can describe that variation with one axis.

Do not silently substitute local PCA for the fixed-axis metric, and do not assume local PCA solves a runtime bottleneck. Its computational cost would require its own measurement if implemented.

### 9.2 Procedure

For each region and choice window:

1. Start from the full **time bins × included units** activity matrix, not a truncated set of displayed session PCs.
2. Apply the same session-derived unit scaling as Stage 1.
3. Mean-center each unit within the choice window. Do not perform a second unit-wise z-score.
4. Fit a new PCA/SVD to that centered window, without whitening.
5. Order its local variance components and calculate the smallest count explaining 95% of the window's total variance.

If $d_{i,j}$ is the local variance explained by component $j$:

$$
K_{95,i}^{\mathrm{local}} =
\min\left\{k : \frac{\sum_{j=1}^{k}d_{i,j}}{V_i} \ge 0.95\right\}
$$

The denominator uses the same unit scaling and within-window total variance $V_i$ as Stage 2. Local PCA centering is not extra score standardization.

Label the result **Local-PCA dimensions for 95% of window variance**. Keep it separate from the fixed-axis count and retain both if comparing strategies. Local component identities are not the session-wide PC identities and are not necessarily aligned across choices.

The mandatory neural heatmaps continue to use the original session-wide basis. No component-matching algorithm, Procrustes alignment, or new local-axis activity viewer is required.

### 9.3 Sampling limitation

For $U$ included units and $B_i$ time bins, the maximum rank of locally centered data is:

$$
K_{\max,i} = \min(U, B_i-1)
$$

| Bin size | Time bins | Local-PCA rank ceiling, also limited by unit count |
|---|---:|---:|
| 500 ms | 8 | 7 |
| 100 ms | 40 | 39 |
| 50 ms | 80 | 79 |
| 20 ms | 200 | 199 |

Show the number of bins and rank ceiling with local-PCA results. A value near the ceiling is not evidence that the neural population truly has that many underlying dimensions. Smaller bins provide more observations but also change the data and noise properties; no cross-bin-size equivalence is assumed.

This ceiling applies to the **newly fitted local PCA**, not to the number of fixed session axes receiving variance in Stage 2. Both are descriptive measures, not a noise-corrected count of biological computations.

## 10. Focused Acceptance Checks

Use ordinary tests and a real-session visual check, not a new validation framework.

| Stage | Check |
|---|---|
| 1 | Choices and events preserve known timestamp offsets and real inter-choice gaps. |
| 1 | Each complete window contains 8/40/80/200 bins at the specified widths; edge placement and outside-window blanks are correct. |
| 1 | Panning, event toggles, and displayed-PC changes preserve the basis, existing scores, row order, and session footer. |
| 1 | Footer thresholds use total variance and are unchanged by displaying fewer PCs. |
| 1 | Both regions and many PC rows remain navigable; missing optional tracks do not stop the viewer. |
| 2 | In a complete orthonormal basis, summed window-PC variances match total window unit variance within ordinary numerical precision. |
| 2 | The PC2/PC7/PC12 example returns count 3 while the heatmap remains in session order. |
| 2 | A window spread along many fixed axes can return a count above $B_i-1$; no inappropriate clipping occurs. |
| 2 | Zero-variance and incomplete-basis cases are explicitly labeled; the displayed-PC limit does not affect the metric. |
| 2 | Benchmark results separate preprocessing, incremental calculations, memory, and display costs. |
| 3, only if implemented | A one-dimensional diagonal trajectory gives one local dimension even when it participates in multiple fixed session axes; the local rank ceiling is respected. |

## 11. Codebase Verification and Scope Boundaries

Before implementation, inspect and reuse the project's existing session timestamp mapping, neural binning, unit selection, PCA utilities, Streamlit navigation, caching, and saved-analysis conventions. Confirm installed plotting/estimator APIs rather than assuming documentation for a different version applies unchanged.

Keep mathematical computation separate from the UI, with simple functions that can later run locally or through an existing batch workflow. Preserve trial IDs, unit IDs, region, bin edges, unit scaling, fitted basis, spectrum, and analysis settings with the computed results. Do not add elaborate provenance machinery, automatic model selection, or speculative deployment support.

This spec does not request decoding, selectivity tests, Granger analysis, new LFP measures, nonlinear dimensionality estimation, or temporal-confound inference. Additional neural tracks can be designed separately after the mandatory viewer is useful.

## Technical References

These references support library capabilities and PCA conventions. They do not verify the local codebase, establish runtime, or imply that an optional stage has been implemented.

- **R1 — Streamlit interactive Plotly chart and supported event interface:** `https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart`
- **R2 — Plotly range sliders, including vertically stacked subplots:** `https://plotly.com/python/range-slider/`
- **R3 — PCA centering, whitening, component order, and explained variance:** `https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html`
- **R4 — Plotly heatmap coordinates, cell geometry, and missing values:** `https://plotly.com/python/reference/heatmap/`
- **R5 — Streamlit caching:** `https://docs.streamlit.io/develop/concepts/architecture/caching`
