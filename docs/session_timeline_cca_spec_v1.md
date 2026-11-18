# Session Timeline PFC-HPC CCA Viewer Spec

Status: Version 1 draft, updated 2026-11-13. The scientific scope and technical
conventions listed in Section 1 are confirmed. Detailed implementation still
requires a separately approved plan; no experimental-data run is authorized.

## 1. Purpose and confirmed decisions

Inspect paired PFC and HPC population activity in fixed, session-wide canonical
coordinates while navigating individual choices on the real session clock.

The user confirmed:

- Analyze simultaneously recorded PFC and HPC spike populations.
- Fit a descriptive session-wide CCA, with no held-out assessment in this version.
- Use unit activity directly, without selecting a reduced set of regional PCs.
- Retain the existing choice windows and bin-width options.
- Pair simultaneous bins at zero lag, retaining task-related activity without
  subtracting condition-average trajectories.
- Provide paired heatmaps and selected-pair traces if both fit cleanly.
- Use a light theme.
- Retain all numerically available canonical pairs; initially display the first
  ten available pairs and select CC1 for the traces.
- Provide explicit Prepare/Recompute buttons, like the PCA timeline.
- Use only windows complete in both PFC and HPC for fitting and canonical-score
  display, while retaining all behavioral choice markers.
- Normalize each regional canonical score to mean zero and sample SD one across
  the common session fitting pool, without separate trial normalization.
- Remove numerical null directions while retaining all remaining independent
  directions, without a PCA variance cutoff.

The last three technical conventions were confirmed on 2026-11-13 after review
of their meaning and trade-offs. The remaining sections specify these contracts
and propose implementation details for review, including the exact numerical
rank tolerance and estimator choice.

## 2. Relationship to the existing viewer

Add a separate **Session timeline and CCA** view in the existing Streamlit app.
Reuse the established metadata, synchronized spike loading, population-quality
selection, behavioral tracks, session clock, and navigation. Preparation must
not require a PCA result. Existing PCA archives and views remain usable.

The key scientific difference is that PCA fits each region independently,
whereas CCA fits one paired model. PFC CC1 and HPC CC1 are two regional
projections belonging to the same canonical pair; they are not one common
regional loading vector. Canonical pairs are ordered by their session-wide
correlation, not by either region's explained variance. [R1, R2]

The original PCA spec was extended after implementation. Reuse the current
behavior documented in `src/neural_analysis/SESSION_TIMELINE.md`, including
context/reward styling and persistent browser navigation, rather than reverting
to the original spec's earlier navigation limitations.

## 3. Inputs, choice eligibility, and time

Use `neural_session.json` and the existing PFC/HPC probe roles or explicit probe
selections. The two roles must identify distinct probes. Reuse the PCA
timeline's population selection: good inside-brain channels, optional metadata
channel restrictions, and the existing manual Phy good/mua policy. Record the
actual unit IDs and quality policy, including missing-label handling.

Use the existing completed ordinary choice table, including incorrect and
unrewarded choices. Preserve original row positions, trial IDs, actions,
recorded rewards, and UTC choice times. Do not add condition balancing,
outcome filtering, unit subsampling, or a minimum firing-rate threshold.

Time zero is the first trial start. All panels use elapsed seconds from that
origin; saved timestamps retain their UTC meaning. Optional trial starts,
left/right licks, verified LED transitions/intervals, and logged active-context
shading retain their existing definitions. Missing optional streams do not
prevent CCA preparation or viewing.

### Common paired windows

For each region, use the existing trusted recording-coverage rule to identify
complete `[-2,+2)` choice windows. Fit CCA on the intersection of those choice
positions, in their original chronological order. Verify that paired rows refer
to the same source choice and bin edges; equal array lengths are insufficient.

Use this common pool for both fitting and canonical-score display in version 1.
Keep all behavioral choice markers. A choice without complete support in both
regions has no canonical activity displayed in either regional panel. Report
the total choice count, regional coverage counts, and common paired count.
This deliberately differs from PCA's independent regional eligibility.

If either population is unavailable, no common windows exist, or a region has
no variable activity, display behavior and an explicit CCA-unavailable reason.
Do not fit a one-region substitute or pad missing observations with zeros.

## 4. Binning and array contracts

Use unsmoothed spike counts in the half-open interval `[-2,+2)` around each
actual choice timestamp. Preserve real inter-choice gaps and the existing
assumption that these choice windows do not overlap. No overlap framework,
time warping, resampling, interpolation across choices, or trial averaging is
requested.

| Bin width | Bins per window |
| --- | --- |
| 500 ms, default | 8 |
| 100 ms | 40 |
| 50 ms | 80 |
| 20 ms | 200 |

Reuse the existing Pynapple binning path. If that path returns Hz, multiply by
the configured bin duration to obtain counts and document the conversion.

Let W be the common window count, B the bin count, P/Q the retained regional
unit counts, and K the available canonical-pair count.

| Product | Shape and axis meaning | Units |
| --- | --- | --- |
| Regional counts before constant-unit removal | `(W, B, input_unit)` | Spike counts per bin |
| Paired fitting matrices X/Y | `(N, P)` / `(N, Q)`, N = W x B | Standardized unit activity |
| Unit means and population SDs | `(input_unit,)` per region | Counts per bin |
| Regional projection coefficients A_PFC/A_HPC | `(P, K)` / `(Q, K)` | Canonical-score units per standardized-unit value |
| Regional canonical scores | `(W, B, K)` | Dimensionless, session sample SD = 1 |
| Canonical correlations | `(K,)` | Dimensionless Pearson r |
| Common choice positions | `(W,)` | Indices into the shared choice table |
| Choice-relative bin edges | `(B+1,)` | Seconds |

Flatten windows first and bins second, identically for both regions. Preserve
explicit unit order throughout fitting, projection, saving, and loading.

## 5. Session-wide direct-unit CCA

### Preprocessing and fitting scope

For each region, compute each unit's mean and population SD (`ddof=0`) across
the common W x B observations. Omit exactly constant units globally and report
their IDs. Standardize the remaining units once using these pooled statistics.
Nonfinite count inputs are an input error, not an instruction to impute or
silently remove observations. Fully silent covered windows remain measured
zero-count windows and can enter a variable session population.

Fit ordinary, unregularized linear CCA to X and Y, retaining all numerically
available pairs. Do not select a leading-PC count, variance-coverage threshold,
ridge penalty, or noise cutoff. There is no condition-specific fit, per-choice
fit, viewport fit, temporal lag search, or second preprocessing fit when display
controls change.

The input pooling retains differences among trial conditions and shared
choice-aligned responses. There is no within-window mean subtraction or
condition-average subtraction beyond the global centering used by CCA.

### Mathematical and normalization contract

For pair j, fit regional coefficient vectors a_j and b_j to maximize the
correlation between X a_j and Y b_j subject to the usual CCA within-region
orthogonality constraints. Retain K = min(rank(X), rank(Y)) pairs, including
zero or weak canonical correlations; no correlation threshold selects pairs.

Normalize each saved canonical score column to mean zero and sample SD one on
the common fitting pool. Equivalently, with U = X A_PFC and V = Y A_HPC:

```text
U.T @ U / (N - 1) = I
V.T @ V / (N - 1) = I
U.T @ V / (N - 1) = diag(rho)
```

These relations hold to documented numerical tolerances. Store coefficients
with this normalization so projection reconstructs the saved scores. This
defines CCA's score scale; it does not inherit PCA's unwhitened score units.
Library output may require a fixed conversion: the current statsmodels
implementation normalizes centered scores to unit sum of squares, so multiplying
its coefficients by `sqrt(N - 1)` gives the accepted sample-SD convention. [R2]

Order pairs by descending rho. Use nonnegative paired correlations. Fix the
remaining joint sign by making the largest-absolute PFC coefficient positive,
breaking exact coefficient ties by saved unit order; apply that sign change to
both regions' coefficients and scores. Preserve the saved result while browsing.
Tied correlations can have nonunique axes within their tied subspace; do not
promise component matching across independent fits, sessions, or bin widths.

### Numerical rank and validity

Use float64 and explicitly record the numerical rank method and tolerances.
The proposed rank rule for each centered standardized matrix M is to retain
singular values greater than `eps_float64 * max(M.shape) * largest_singular_value`.
This removes numerical null directions, not a chosen fraction of unit variance.
Preserve all nonconstant unit identities and map coefficients back to those
units; document the minimum-norm coefficient convention for dependent columns.

Prefer a verified existing CCA implementation with a small project adapter for
normalization and rank handling. The installed statsmodels `CanCorr` uses SVD
but raises on collinear inputs; scikit-learn also provides CCA. Their APIs and
installed-source behavior have been inspected, but neither has been benchmarked
for this viewer. The implementation plan must identify the chosen estimator
and how it satisfies the full-rank and dependent-unit tests. An adapter may
use existing NumPy/SciPy linear algebra when required; do not modify dependencies.
No new dependency is expected. [R1-R3]

If N < 2 or either numerical rank is zero, mark CCA unavailable. Record rank
deficiency and conditioning diagnostics without silently introducing scientific
regularization. Material violations of normalization or correlation bounds
are fit failures, not values to conceal by clipping. Allow only documented
roundoff at the bounds of rho.

Report when the regional observation subspaces intersect because their ranks
are large relative to N - 1: some in-sample canonical correlations are then
forced to one by sample geometry. Specifically, if the regional ranks sum to
more than N - 1, at least `rank(X) + rank(Y) - (N - 1)` shared directions are
unavoidable by the dimension-intersection bound. Do not automatically reduce
units to avoid this. Distinguish this structural condition from ordinary
floating-point error, without treating every observed rho near one as forced.

## 6. Clean combined display

Use one persistent Plotly figure with synchronized elapsed-time axes. The
proposed default order is:

1. Choices, retaining current left/right, reward, and active-context styling.
2. Selected optional behavioral event rows.
3. PFC canonical-score heatmap.
4. HPC canonical-score heatmap with matching canonical-pair rows.
5. One compact selected-pair trace row containing both regional signals.
6. Compact full-session navigator.

Use opaque white backgrounds, black axes/text, and readable fonts. Keep controls
and result details outside the main plot area. Both heatmaps are the primary
overview; the trace row provides detail without duplicating every component
as a separate line plot. Allow the trace row to be hidden.

### Paired heatmaps

Use one shared canonical-pair rank-range control for the two heatmaps. Initially
show CC1 through CCmin(10,K); allow access to every retained pair. Keep matching
rows in the same session correlation order in both regions.

Use explicit actual bin edges and a diverging color scale centered on zero,
with one common symmetric limit across both regions. Set the automatic limit
from all saved scores in both regions, fixed across navigation and rank changes.
An optional manual symmetric limit changes display saturation only and is
clearly labeled. A measured zero must remain distinguishable from a blank gap.
Do not normalize each row or trial again.

Hover identifies region/probe, canonical pair, its session-wide in-sample rho,
original trial ID, elapsed/UTC/choice-relative bin intervals, and canonical
score. Label scores as canonical-score SD units, using the normalization in
Section 5. The pair correlation is a session summary, not a per-bin estimate.

### Selected-pair traces

Provide a canonical-pair selector, initially CC1. Plot PFC and HPC on one shared
y-axis in distinct labeled colors, using the same saved values as the heatmaps.
Use step traces occupying the actual bin intervals and break them between
windows. Do not connect across gaps or smooth/interpolate the measurements.

Keep the y-limits fixed from that pair's complete session scores in both
regions while navigating. Changing the selected pair can change the limits.
Include zero, clear region labels, and the session-wide rho in the panel title
or adjacent caption. If the pair lies outside the displayed heatmap range,
identify that without silently changing either control.

### Navigation and summary

Reuse the existing 60-second initial duration, Window start, Jump-to-trial,
native pan, range navigator, and persistent chart component. Display changes
preserve the browser's current time range; explicit navigation requests move
it. All rows use the same session clock and retain real gaps.

Retain the existing bounded-neural-payload option, initially on at 20 ms.
State the loaded bounds and require explicit time controls to load elsewhere;
browser panning does not request data from Python. Bounded loading must not
change fitting, normalization, correlations, or color limits.

Show a compact summary below the chart: paired windows, pooled bin observations,
regional input/retained/constant-unit counts, numerical ranks, K, bin width,
and the displayed/selected pair correlations. An expandable table exposes the
full canonical correlation spectrum, without crowding all pairs into the footer.

Do not import PCA's 80/90/95-percent variance footer or fixed-axis window-count
metric. Canonical rho and rho squared are not fractions of total regional
population variance. Per-window correlations, rolling coupling estimates,
significance thresholds, and counts of biological communication dimensions
are outside this version.

## 7. Preparation, persistence, and reuse

Separate numerical preparation, persistence/orchestration, plotting, and UI.
Use simple functions and small records with documented types, axes, and units.
Reuse current loaders and public behavioral/navigation helpers. The heatmap
geometry can be shared where useful, but do not disguise a CCA result as a PCA
result with fictional explained-variance fields. No generic track framework is
required.

Provide single-session and session-list batch entry points using the same
computation as the UI. Support dry-run, reuse by default, and explicit rerun.
Batch workers operate across sessions, defaulting to available CPU cores capped
by session count, with a smaller explicit worker limit when memory/I/O requires.

Prepare computes and saves the selected CCA configuration only on an explicit
click. Recompute creates a new timestamped run. Otherwise, load compatible saved
results; if none exist, show behavior with an unprepared message. Changing a
scientific setting must not relabel an older incompatible fit.

Save under the source session's data directory, outside the repository:

```text
analysis_runs/session_timeline_cca_<date>_<time>_<configuration_id>/
  results.npz
  config.json
  run.log
  summary.md
  run_session.py
  run_batch.py
```

The NPZ uses named arrays and a dictionary called `meta`. Preserve choices,
events, origin, common choice positions, regional coverage eligibility, bin
edges, input/retained/omitted unit IDs, means/scales, regional coefficients,
regional scores, rho, ranks, and numerical diagnostics. Raw count tensors need
not be duplicated in the final result when fitted products are sufficient.

Metadata includes generator, analysis/schema versions, actual source identity,
probe/population choices, bin/window settings, quality policy, zero-lag/raw-task
activity policy, normalization/sign/rank conventions and tolerances, numerical
library/code versions, and timings. Record no randomness used and seed `None`
for a deterministic fit; seed synthetic validation and save those seeds.

Cache by scientific inputs, analysis version, estimator/rank conventions and
their numerical parameters, excluding viewport, displayed ranks, selected trace
pair, event visibility, and display limits. Missing/invalid
prerequisites and source changes must invalidate reuse explicitly. Save new
results atomically and never overwrite earlier outputs. Do not require a live
cluster job, a new configuration file beyond actual needs, or a new cache system.

Saved viewing should use stored behavior and numerical products rather than
reload spikes or refit. Existing source-identity checks and the UI's metadata
requirements must be documented in the implementation plan; do not claim that
copied results work without raw sources until that path is tested.

## 8. Interpretation

Suggested concise caption:

> Session-wide descriptive CCA of simultaneous PFC/HPC unit activity across
> complete paired choice windows. Canonical scores use session sample-SD units;
> correlations are in-sample and task-related activity is retained.

The fitted correlations are optimized on the displayed session data. They do
not establish held-out generalization, directionality, or causal communication.
Shared task responses and other common influences remain part of the observed
association. Pooled bins are temporally dependent; N is an observation count,
not a claimed number of independent experimental samples.

High-dimensional CCA can produce inflated in-sample correlations and unstable
weights. This is a material limitation of the accepted exploratory scope, not
a request to add CV, shuffles, regularization, or automatic model selection to
version 1. High correlations or weak canonical pairs are displayed honestly,
with sample/rank context. Low-variance directions may receive strong CCA weights;
equal score SD does not imply equal regional variance or biological importance.
[R4]

## 9. Acceptance checks and tests before implementation

The implementation plan must list the tests, write and commit them before the
corresponding implementation, demonstrate RED, then GREEN and refactor under
the repository's pytest workflow. Suggested focused checks are:

| Area | Required behavior |
| --- | --- |
| Choice/time integrity | Preserve existing choice eligibility, IDs, origin, and actual choice timestamps. |
| Pairing | Unequal regional coverage produces the correct common pool; shuffled or mismatched choice/bin axes cannot pass as paired data. |
| Binning | All four widths have the specified half-open geometry; covered silence is zero, missing support is unavailable. |
| Direct-unit preprocessing | Pooled means/SDs use common windows; constants are omitted once; nonfinite data are not repaired or silently dropped. |
| Known CCA solution | Seeded or analytically constructed full-rank data recover known correlations and satisfy both within-region identity covariances and diagonal cross-covariance. |
| Library comparison | Compare a well-conditioned full-rank case to verified statsmodels CCA, allowing sign/subspace ambiguity. |
| Projection | Saved coefficients plus saved preprocessing reconstruct scores in the original window/bin/unit order. |
| Rank deficiency | Duplicate/dependent units retain the same observable canonical correlations/subspace without a scientific variance cutoff; unit mapping and rank diagnostics remain explicit. |
| Degeneracy | No variable region, too few observations, rank saturation, and tied/zero correlations have defined behavior; tied-subspace tests do not demand unique vectors. |
| Pair order/sign | Correlations descend, both regions share identities, and the joint-sign convention preserves rho and projection. |
| Display equivalence | Heatmaps and traces use the same saved scores, common units and pair identity; no local refit or renormalization occurs. |
| Geometry | Heatmaps/step traces respect actual edges and gaps; unavailable values do not become zeros or connecting lines. |
| Display invariance | Panning, bounded loading, rank/pair selection and event/color controls preserve fitted scores, rho and normalization. |
| Navigation | Display changes retain browser pan with the existing component; explicit navigation moves all time rows together. |
| Persistence | Round-trip arrays/metadata, configuration invalidation, reuse without reloading spikes, atomic completion, and immutable explicit reruns work. |
| UI/CLI/batch | Metadata routing, behavior-only fallback, explicit preparation, dry-run, worker limits, and numerical equivalence across entry points work. |

Validate the full pipeline on seeded artificial paired populations with known
shared/private activity, missing coverage, silent bins, and dependent units.
Inspect both heatmaps and the selected-pair trace in a browser, including their
combined layout at ordinary window size and after a manual pan.

After synthetic checks, the user must designate one experimental session.
Prepare that session and inspect numerical/visual output before requesting
approval for any batch. Prior PCA validation of CT026 is useful context but
does not itself authorize a CCA run or select the CCA validation session.

## 10. Performance and completion

Measure loading/binning, standardization/rank/CCA fitting, projection, output
writing, saved-result loading, figure construction, JSON serialization/payload,
and process peak memory separately. Include W, B, regional unit/rank counts, K,
machine details, bin width, and measured versus estimated quantities.

Benchmark the designated real session at 500 and 20 ms where practical. Reuse
analysis products during interaction, build only the requested plot rows, and
avoid dense session grids filled with empty gaps. Existing PCA measurements
show that fine-resolution browser payload can dominate; they are context, not
CCA timing estimates. Bounded plot loading does not necessarily reduce NPZ
decompression or full-score memory, which must be measured separately.

Start with existing optimized numerical libraries and simple local preparation.
Choose additional infrastructure or major optimization only after measurements
and user review. No unmeasured latency guarantee is part of this spec.

Completion requires the approved scientific contract, tests-first implementation,
synthetic checks, designated-session numerical and browser inspection, usage
documentation, and recorded performance. A future implementation plan should
name the concrete modules/functions, verified estimator, tests and numerical
tolerances before code is written.

## 11. Approved optional treadmill-speed extension (2026-11-13)

The user approved an optional signed-mm/s row using the existing PCA timeline
speed pipeline. A declared `behavior.treadmill` source enables explicit
Prepare speed/Recompute speed and cached Show speed controls. The step row
appears after events, shares the elapsed-time axes, retains activity between
choices and uses full-series y limits. Missing or invalid speed affects only
this row. Its cache is independent of CCA; it does not enter the fit or alter
saved scores, source identity, binning or normalization. Display toggles preserve
native pan. No smoothing, resampling or new dependency is required.

The user separately approved handling redundant same-level sync reports in
speed v3. Their intervals remain in the source clock; only pulse pairing ignores
them, with explicit row/time diagnostics. Synthetic insertion tests must preserve
frame UTC anchors and every retained speed timestamp/value. Existing short-gap
ambiguous speed exclusions remain a separate diagnostic, not part of redundant
state handling. Raw logs remain unchanged; speed cache version/identity changes.

## References and inspected reuse points

- Existing scientific spec: `docs/session_timeline_pca_spec_v1.md`.
- Current behavior/history: `src/neural_analysis/SESSION_TIMELINE.md` and
  `docs/session_timeline_pca_implementation_plan.md`.
- Choice/binning contracts: `src/neural_analysis/population/session_timeline.py`.
- Sources, selection and persistence patterns:
  `src/neural_analysis/population/session_timeline_io.py`.
- Public behavioral rendering: `add_timeline_behavior` in
  `src/neural_analysis/population/timeline_plotting.py`.
- Navigation: `render_timeline_navigation` in
  `src/neural_analysis/webapp/timeline_view.py` and the persistent
  `timeline_chart.py` / `timeline_chart.js` component.
- Separate-view precedent: `src/neural_analysis/webapp/phase_synchrony_view.py`.
- R1: [scikit-learn CCA](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.CCA.html)
  and [cross-decomposition guide](https://scikit-learn.org/stable/modules/cross_decomposition.html).
- R2: [statsmodels CanCorr](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.cancorr.CanCorr.html)
  and [implementation](https://www.statsmodels.org/stable/_modules/statsmodels/multivariate/cancorr.html).
  Local installed source was also inspected for normalization and collinearity.
- R3: [SciPy SVD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html).
- R4: [Helmer et al., 2024: On the stability of canonical correlation analysis
  and partial least squares](https://www.nature.com/articles/s42003-024-05869-4).
  Its stability findings motivate interpretation cautions, not a numerical
  sample-size threshold transferred from human imaging to these spike data.
