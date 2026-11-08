# PC-First and Single-Unit Ridge Encoding Spec

Status: Working draft v1, 2026-11-02. Records the revised analysis goal and
accepted first-pass scientific/display decisions. Source verification and an
approved implementation plan are required before coding. No implementation or
experimental-data run is authorized by this document.

[single_unit_spec_v6.md](single_unit_spec_v6.md) remains the historical
trial-summary selectivity specification. Its trial-level response aggregation,
unpenalized-only policy, rank rejection, and significance workflow do not define
this new analysis. The companion
[implementation plan](single_unit_plan.md) defines code organization,
tests-first phases, saved products, and handoff instructions for this goal.

## Goal and Initial Scope

Inspect how regional neural PCs relate to true task context, primarily through
saved coefficients, predictive performance, and ongoing session activity.
**The first implementation and inspection pass focuses on PCs.** Individual
units remain a subsequent representation using the same bin-level regression
and inspection contracts. Do not make PC inspection wait for unit fitting.
Start with a small, interchangeable predictor configuration rather than adding
the full behavioral-variable catalog immediately.

- One regression observation is **one neural time bin**, not one trial-window
  total or mean.
- Pool bin observations across eligible trials within each selected before,
  after, or whole window. These are fits for overall responsiveness, not
  separate fits at different choice-relative time points.
- Use ridge regression with a fixed, user-adjustable L2 penalty as the primary
  estimator, defaulting to `alpha=1.0` with the summed-error objective below.
  Leave the intercept unpenalized and retain native predictor coding.
- Default to the same numeric penalty across predictor configurations. Offer
  matched effective shrinkage as an explicit inspection option for the initial
  two-indicator versus signed-context comparison only.
- Retain unpenalized OLS as a selectable comparison baseline with **both**
  context encodings. Implementing signed context first is acceptable, but the
  two-indicator version is also required.
- Preserve correlated and redundant predictors intentionally; do not prune
  columns automatically to make the ridge design full rank.
- Evaluate two separate context-encoding configurations using matched data and
  evaluation rows.
- Save the full fitted coefficients and support ordering PC/activity rows by
  the strength of a selected coefficient. The same ordering capability applies
  to units when that representation is added.
- Poisson fitting is deferred. Its future count-model role is described below
  and in v6; it is not a prerequisite for the linear analysis.
- Permutation significance, BH-FDR, fraction-selective summaries, and claims of
  unique predictor contribution are outside the initial implementation goal.

Ridge provides shrinkage and can stabilize a fit. It does not establish clean
biological attribution among correlated regressors or guarantee better
held-out prediction.

### PC-First Delivery

The first usable result must combine PC ridge coefficients, available
prediction scores, and a coefficient-orderable PC activity timeline. Use the
existing session timeline/PCA visualization as the temporal reference; its
per-choice dimensionality calculations are not prerequisites for this result.
The OLS comparison must support both context encodings. Signed-context OLS can
be delivered first within that work, but the other version must not be omitted.
Unit fitting and Poisson must not block the initial PC result.

A complete PCA basis is an invertible representation of the retained,
standardized unit activity on its supported dimensions. Retaining only leading
PCs discards information and can omit a context-related low-variance dimension.
The fitted-PC set and the displayed-PC subset are separate choices; neither
coefficient sorting nor display limits may silently change the fitted set.
Fit and retain results for **all available nonzero PCs** in each region, using
ordinary numerical rank/zero-variance handling. Let the user display a selected
subset of those fitted PCs; do not impose a leading-PC analysis cap.

## Neural Observations and Retained Session Conventions

Retain the agreed choice-centered [-2, +2) second activity window, with **500-ms
bins by default** and an adjustable shared bin width. Unit and PC analyses use
the same regional activity bins, timestamps, and before/after membership.
The first implementation supports choice alignment only; trial-start alignment
is deferred.

- Unit activity has axes `(trial, time_bin, unit)` and units of spike counts
  per configured bin. Use the bin count directly as the unit response.
- PC activity has axes `(trial, time_bin, component)`. Use each bin's ordinary
  signed PC score as the response, without averaging over its trial/window.
- Flatten selected trial/bin axes to observations while preserving original
  trial ID, `cur_block`, bin edges, absolute/session time, and choice-relative
  time for every row. A response matrix has shape `(n_bin_rows, n_responses)`;
  its predictor matrix has shape `(n_bin_rows, n_predictor_columns)`.
- PFC and HPC remain independent. Require complete regional coverage of the
  four-second window; do not substitute zero activity for missing coverage.
- Manual-alignment sessions are deferred for this implementation. Require
  IRIG-backed recording coverage and report unsupported inputs explicitly.
- Retain exclusion of manual-reward, no-choice, and dark-state trials, and
  existing channel selection with both `good` and `mua` units.
- Retain saved augmented-trial features and associated feature parameters as
  inputs. Use existing source column names, including `state_int` and, when
  later selected, `FQlearning_rel_value`. Do not regenerate behavioral features
  or refit behavioral models as part of this analysis.
- Retain separate descriptive regional PCAs from pooled eligible activity,
  unit standardization once, no whitening, and no additional PC-score scaling.
  Use those fixed session axes for held-out regression evaluation as well as
  coefficient/activity inspection; do not refit PCA inside regression folds.
- Reuse the same compatible fixed regional basis and scores as the ongoing
  session timeline so a displayed PC has the same identity in the coefficient
  results and activity. Verify unit selection, input windows, bin width,
  scaling, and source identity before reusing an existing saved timeline
  product. If those differ, prepare a compatible product offline rather than
  labeling independently fitted axes as the same PCs.

Several bins from one trial are multiple observations, but they are not thereby
independent samples. Do not use random bin-level train/test splitting as though
within-trial temporal dependence had disappeared.

### Pooled Fitting Scope

Fit one coefficient vector per unit or PC, selected before/after/whole window,
predictor encoding, timing mode, and estimator configuration. Pool all eligible
trial/bin rows in that window. At the default 500-ms resolution, each complete
trial contributes 4 bin observations to before, 4 to after, or 8 to whole.
Coefficients are shared across the pooled bins in a fit; predictor values can
still change between bins according to the selected timing mode.

Do not fit a coefficient vector independently at each time offset, produce a
time-resolved coefficient scan, or add per-offset fits as an automatic extra
analysis. Keep this initial overall-responsiveness workflow fast and direct.

## Predictor Timing Modes

The implementation must offer two timing configurations for inspection using
the same neural observations. Record the timing mode with the fitted result.

### Choice-updated active values

Use the predictor value active at each bin's time on the ongoing choice
timeline. Updates occur at **choice time for this initial analysis**.

For a history feature with entering-trial value `v_i`, bins before choice `i`
use the pre-choice value; bins after that choice use the value resulting from
trial `i`. In an uninterrupted sequence, that post-choice value is the same
value present immediately before choice `i+1`. Retain predictor state across
the unobserved inter-window gaps rather than restarting it at each extracted
neural window.

This choice-time convention is deliberate even if reward delivery is recorded
later. It is an exploratory alignment rule, not a claim that an outcome-derived
quantity was available before its biological update. No reward-time update
option is required in the first version.

The task's true context can also transition after a choice. The task description
states that a correct choice can trigger a context switch; otherwise context
persists. At a switching choice, pre-choice bins use the context governing that
choice and post-choice bins use the resulting context for the next choice.

The existing parser copies context into the trial row at trial start. Use
`state_int[i]` before choice `i` and `state_int[i+1]` after it, from the complete
unfiltered chronological trial sequence. Never shift an already filtered table
or skip to the next retained choice. A next trial excluded for manual reward or
no choice can still supply its known context. Missing/invalid or dark next
state makes the post-choice bins unavailable. The final trial's post-choice
bins are unavailable when the adjacent state is absent; do not carry forward
the previous state or use an event fallback.

Recorded context-entry events remain diagnostic at their original timestamps.
Compare state order, allowing the intentional choice-time update to precede a
logged event; events do not override the adjacent-trial values or masks. Do not
infer transitions from reward or resimulate task randomness. Choice-centered
context bins have a boundary at the choice update itself. Timing modes may
retain different bin counts; do not claim identical-row score comparisons when
their eligibility masks differ.

### Trial-window values

Hold trial `i`'s **entering-trial predictor values fixed across its entire
[-2, +2) second neural window**, matching the v6 timing convention. Use those
same values for that trial's bins whether fitting before, after, or whole.
The response remains one observation per bin and is never collapsed to v6's
trial-total or window-mean response.

This is an intentional comparison with choice-updated active values. Do not
switch to post-update values in the after half of this timing mode. Record the
mode explicitly so a window-fixed association is not mislabeled as following
the active predictor timeline.

## Initial Predictor Configurations

Start with true background context only. The existing source is `state_int`:
`0=right`, `1=left`, and `2=dark` (excluded). The signed context regressor has no
special zero for absence of a previous choice; it represents context, not choice
history.

| Configuration | Derived model columns | Values |
| --- | --- | --- |
| Separate context indicators | `left_context`, `right_context` | Left: `(1, 0)`; right: `(0, 1)` |
| Signed context | `signed_context` | Left: `+1`; right: `-1` |

These are two separate fits, not three columns included simultaneously. Derived
model-column names describe the encoding; they do not rename `state_int` in the
saved input.

Use the selected timing mode to assign the underlying context before applying
either encoding. Keep response rows, eligibility, temporal scope, and evaluation
splits matched when comparing the configurations.

Keep these native `0/1` indicator and `+1/-1` signed values during fitting. Do
not standardize or otherwise rescale predictor columns automatically. Numerical
predictors added later retain their documented native values unless a future
explicit revision selects a different fitting-scale policy. Post-fit
standardized coefficient display is a separate operation.

The predictor interface must later allow replacing or extending this small
configuration with selected saved task/history/model features. The v6 predictor
catalog is a reference for future additions, not the initial default predictor
set. Do not implement every possible predictor or a generic formula framework
before this context-only analysis is usable.

## Ridge Model and Encoding Comparison

For a response vector `y`, predictor matrix `X`, unpenalized intercept `a`, and
penalized slopes `b`, write the fitting objective explicitly:

$$
\min_{a,b}\;\sum_r (y_r-a-X_r b)^2 + \alpha\sum_j b_j^2.
$$

Use **`alpha=1.0` by default**, adjustable to another positive value. This is a
simple starting setting for inspection, not a data-calibrated optimum or a
guarantee of a particular amount of denoising. The objective uses summed squared
errors, not mean squared error; do not silently multiply/divide `alpha` by the
number of bins or trials. Native predictor scaling and the number of fitting
rows therefore affect the practical strength of this fixed penalty. Record the
objective, configured penalty, and actual penalty used by each fit. Do not
introduce inner-CV tuning or an automatic penalty search. Use the explicit OLS
baseline for an unpenalized comparison rather than relying on a zero-penalty
ridge fit.

Two exhaustive context indicators plus an intercept are exactly redundant.
With positive L2 penalty on slopes and an unpenalized intercept, ridge can fit
this intentionally redundant design. Do not import v6's full-rank rejection or
silently remove one indicator. Ordinary finite-data and fit-validity checks
still apply.

The encodings contain the same context information, but their penalties are
not automatically equivalent. For native unstandardized columns, consider
predicted context means `a+d` for left and `a-d` for right:

- Signed encoding uses slope `d`, with penalty `alpha * d^2`.
- Two indicators can use slopes `d` and `-d`, with penalty
  `alpha * (d^2 + (-d)^2) = 2 * alpha * d^2`.

Thus the same numeric penalty shrinks this contrast more strongly in the
two-indicator configuration. With these native encodings, half the signed
configuration's penalty in the two-indicator configuration matches the
effective shrinkage. Standardizing predictor columns changes that relationship.

### Penalty Comparison Modes

**Default: same numeric penalty.** Apply the configured `alpha` unchanged to
each predictor configuration. This is the general policy, including future
predictor sets that are not equivalent representations of the same variable.
Do not automatically rescale the penalty based on column count, redundancy, or
an assumed equivalence between configurations. Describe observed differences
as results under the same numeric penalty; they can reflect encoding-dependent
shrinkage as well as differences in predictor information.

**Optional: matched shrinkage for the context-only encoding pair.** Allow an
explicit comparison that matches effective shrinkage between the two context
representations. For the native unstandardized columns defined above, use the
configured `alpha` for signed context and `alpha / 2` for the two-indicator
model. If another predictor-scaling policy is selected, verify the corresponding
matching factor rather than applying this native-code factor unchanged.

This optional mode is specific to the verified context-only equivalence. Do not
generalize it to arbitrary predictors or automatically apply it when additional
regressors are selected. Keep its rows, responses, timing, and evaluation splits
matched to the default comparison. Save the comparison mode, configured penalty,
and actual numeric penalty used by each fit so both comparisons can be
inspected without replacing or conflating their results.

### OLS Baseline in Both Encodings

Provide OLS for both signed context and the two separate context indicators,
with an unpenalized intercept and the same eligible rows and evaluation splits
as ridge. Signed context may be implemented first, but both configurations must
be available for inspection and comparison.

For the redundant two-indicator design, define a simple reproducible coefficient
convention: center predictor columns and the response using fitting-row means,
solve unpenalized least squares for the **minimum-Euclidean-norm slopes**, and
recover the intercept as `mean(y) - mean(X) @ slopes`. Use the same convention
for the reference fit and each training fold. This centers data without scaling
predictor columns; predictions remain in the original response units. Do not
include the intercept in the minimum-norm slope criterion.

When both context classes are present, this yields equal-and-opposite indicator
slopes. The signed-context fit and two-indicator fit should have equivalent OLS
predictions on matched data, up to numerical precision. Preserve and display
both fitted coefficient vectors. Label the two-indicator values as a
minimum-norm coefficient convention; they are not uniquely identified by the
unpenalized least-squares objective alone. Do not silently drop an indicator or
reuse v6's full-rank rejection for this intentionally redundant baseline.

## Coefficients and Predictive Performance

Primary outputs are fitted coefficients/intercepts and prediction performance
for each unit or PC, context encoding, and timing configuration. Preserve raw
coefficient units, predictor coding/scaling, penalty, response identity, and
eligible bin/trial counts. Include the penalty-comparison mode and actual
per-fit penalty in result labels or associated settings. Signed PC coefficients
depend on PC orientation and must not be interpreted as an intrinsic biological
sign.

### Saved Coefficients and Fit Identity

Retain the complete slope vector and intercept for every fitted response, not
only the displayed or strongest coefficients. A coefficient record must identify
the session, region, original PC rank or unit ID, predictor/model column,
encoding, timing mode, before/after/whole fit window, estimator, scaling, and
configured/actual penalty. Keep unavailable coefficients and their reasons
distinct from valid zeros.

Save an all-eligible-data reference fit for coefficient inspection. If
cross-validation fits are performed, retain their coefficients separately with
fold identity; do not substitute an unlabeled fold average for the reference
coefficients. Identify the fixed PCA basis, unit ordering/scales, eligible
observations, and bin geometry needed to reconnect every response with its
saved activity. Sorting or showing fewer rows must not discard fitted results.

Coefficient-display scaling and row ordering are presentation operations. Always
preserve the raw fitted values and offer a **raw / standardized coefficient
toggle**. The standardized display is `beta_j * SD(X_j) / SD(y)`, calculated on
that reference fit's eligible rows. The toggle changes the coefficient panel,
numeric coefficient readouts, and the scale used for strength ordering. It does
not refit ridge, z-score its response, whiten PCs, or rescale the activity
heatmap. Label the selected scale explicitly. An undefined display scale is
unavailable rather than zero.

Retain held-out prediction as the performance goal. Use **one deterministic
five-fold pass grouped by `cur_block`**, using ordinary GroupKFold without group
shuffling. Construct the map once per region from its baseline-valid,
full-coverage windows, before predictor/window masking, and repeat each window's
fold ID across its bins. Post-choice bins retain their source window's block.
Regions need not share a fold map. Keep all bins from a trial and behavioral
block together. No repeated CV, inner CV, tuning, or search for alternative
fold arrangements is required. Hold out blocks for regression **on the fixed
session PCA axes**. PCA uses the eligible session activity, including activity
from regression test blocks; only regression
fitting is restricted to each training fold. Label scores as held-out
regression performance conditional on descriptive session PCA, not as an
evaluation with wholly training-only neural preprocessing. Do not add
fold-local PCA or PC-axis matching to this first pass. Keep splits identical
across compared encodings, timing modes where rows match, and ridge/OLS models.

### Simple Prediction Scores

For each response/configuration, retain one out-of-fold prediction for every
eligible bin, plus fold assignments, fold-level counts/scores, and validity
reasons. Retain reference-fit predictions separately for descriptive inspection.
Report primary performance from the complete pooled out-of-fold observations:

$$
\mathrm{MSE}_{\mathrm{CV}} = \frac{\sum_r (y_r-\hat y_r)^2}{N},
\qquad
R^2_{\mathrm{CV}} = 1 -
\frac{\sum_r (y_r-\hat y_r)^2}{\sum_r (y_r-\bar y)^2}.
$$

Here `N` counts evaluated held-out bins and `mean(y)` is across those same
pooled bins. Weight bins equally; do not give unequal-sized folds equal weight
by averaging their scores. MSE is in squared PC-score units for PCs and squared
counts per bin for units. R-squared is dimensionless, can be negative, and is
not partial R-squared.

Keep failure handling simple:

- Fewer than five usable blocks, an empty required fold, or a failed/nonfinite
  fold fit/prediction makes the primary five-fold score unavailable with a
  reason. Preserve valid reference fits and any completed fold results; do not
  fall back to random bin splitting or present partial-fold scores as complete.
- Require finite predictions on all requested held-out rows for a primary
  score. If the pooled response variance is zero, retain defined MSE and mark
  R-squared unavailable rather than forcing zero or one. Apply the same rule
  independently to fold-level R-squared.
- Keep signed and two-indicator OLS coefficient conventions explicit even when
  their predictions coincide. No significance test of differences in prediction
  scores is required.

More bins do not establish temporal-confound control or create independent
evidence for significance.

## Coefficient and Ongoing Activity Inspection

Use the existing Streamlit session timeline as the interaction and time-axis
reference. Reuse its saved activity and plotting/navigation code where
appropriate, without requiring a rewrite of that viewer. Ridge fitting remains
offline. The following inspection layout is required for PCs first and for the
subsequent unit view using the same response identities.

### Coupled Coefficient and Activity Panels

| Panel | Required content |
| --- | --- |
| Analysis controls | Saved run, region, response representation, fit window, predictor encoding, timing mode, estimator, penalty-comparison mode, coefficient display scale, sorting predictor, and visible row count/range. Show actual penalty and bin width. Controls select saved configurations; they do not launch fits. |
| Behavioral/predictor tracks | Choices and optional existing events at real elapsed-session times, plus the selected model predictor values in the neural bins so choice-updated and window-fixed coding can be inspected directly. |
| Coefficient panel | Signed response-by-predictor coefficient heatmap or equivalent aligned coefficient columns, with original PC/unit labels and exact values available in hover or a table. |
| Activity panel | Signed PC score heatmap over real elapsed-session time, with the same response rows in exactly the same order as the coefficient panel. For later unit inspection use the corresponding binned unit activity with explicit count/rate units. |
| Fit/performance details | Coefficient/intercept values, coding/scaling, bin/trial counts, fit validity, and available prediction scores for the selected responses. Clearly distinguish all-data fitted quantities from held-out evaluation. |
| Selected-response trace | One compact observed/predicted overlay for a selected PC, using the same elapsed-session time range. Keep it separate from the many-row heatmap so it remains readable. |

Keep coefficient and neural-activity color scales separate and labeled. Use a
zero-centered signed coefficient scale and preserve PC score signs and the
fixed activity scale across navigation. Do not normalize activity rows merely
because coefficient display scaling or ordering changes. PC orientation remains
the saved orientation; its coefficient sign is not an intrinsic biological label.

### Ordering and Response Selection

- Allow selection of a predictor/model column and order rows by descending
  absolute coefficient strength using the raw/standardized toggle's selected
  scale. Keep positive and negative coefficients visible through color and
  numeric values.
- Also allow the original order: session explained-variance rank for PCs and
  stable unit ID order for units. Sort within each region rather than mixing
  independent regional populations.
- Apply one identical ordering to the coefficient panel, activity panel, and
  associated response details. Preserve original IDs: a PC12 moved above PC2
  is still labeled PC12, not renamed PC1.
- Use the saved reference-fit coefficients for ordering across the session,
  not coefficients recomputed from the current viewport. Resolve ties by
  original PC rank/unit ID and place unavailable strengths after valid values.
- Allow a selected number or range of the ordered rows to be shown, with the
  remaining fitted responses still accessible. A display limit must not become
  a fitting limit or redefine the PCA basis or variance spectrum.
- Ordering, selecting rows, panning, and toggling events must not refit ridge or
  PCA, alter saved coefficients, or change PC signs/loadings. Reordering is a
  feature of this regression inspection view; it does not redefine the original
  variance-ranked PCA identities or require changing the standalone viewer's
  default row order.

### Real Session Time and Configuration Comparisons

Preserve the existing timeline's actual choice times, choice-relative bin
edges, inter-choice gaps, optional event tracks, horizontal pan/zoom, explicit
time-range controls, and jump-to-trial navigation. Draw neural activity only in
available extracted windows. Leave missing/outside-window intervals blank;
do not concatenate choices into artificial continuous neural time or interpolate
across gaps. A zero score/count remains distinct from unavailable activity.

The regression fit window can be before, after, or whole while the activity
panel retains the full four-second window for inspection. Mark the bins included
in the chosen fit so displayed activity outside the fit scope is not mistaken
for fitted observations. Record absolute bin times and choice-relative times
in hover/details alongside the predictor value assigned to the bin.

Use the selected saved timing/encoding/penalty configuration for coefficients,
ordering, and fit details. Changing that selection may change the ordering;
panning alone must not.

### Compact Observed/Predicted Trace

Include a tidy detail panel for **one selected PC at a time**, with observed
activity and the selected model's predicted activity overlaid on the same time
and response axes. Distinguish the two traces by color or line style, include a
short legend, and show the PC identity and selected model configuration. Use a
separate compact panel rather than overlaying prediction curves on every
heatmap row or drawing many selected PCs together.

Offer a labeled selection between held-out predictions and reference-fit
predictions, defaulting to held-out predictions when available. Keep them
distinct: unavailable held-out predictions must not be silently replaced by
reference-fit values. Preserve missing-data/inter-window gaps in both traces,
use the existing bin geometry, and do not add temporal smoothing or a second
response normalization just to make curves appear similar. The trace follows
the current timeline range and uses saved predictions without refitting. It
visualizes the pooled model, not separate fits at different time points.

Follow the fixed-basis and gap-preserving geometry in
[session_timeline_pca_spec_v1.md](session_timeline_pca_spec_v1.md). This regression
view adds coefficient ordering and fitted-result inspection. It does not make
that viewer's optional local-PCA or variance-participation analyses mandatory
for ridge inspection.

Export light-mode PNG inspection figures with readable labels and captions.

## Offline Execution and Deferred Poisson

Run ridge and the selectable OLS comparisons offline, save progress and completed
results, and let the webapp/plotting layer inspect saved products. Do not make
the initial result wait for Poisson. Reuse compatible completed preparation and
fits when continuing work; preserve prior results when settings or inputs change.

A future Poisson model uses the same predictor rows, encoding/timing options,
unit identities, and **unit spike counts per bin** as its response. Its log-link
form is:

$$
\log E[y_r\mid X_r] = a + X_r b.
$$

It returns log-count coefficients and expected counts per bin. Expected-count
predictions permit a matched MSE comparison on the same held-out bins as the
linear models. PC scores are signed continuous responses and are not Poisson
count targets. The future Poisson penalty policy remains unspecified.

Keep the detailed Poisson model, deviance, and bootstrap discussion in v6 as a
reference. Its trial-total response, unpenalized-only requirement, and
significance machinery must not be silently activated for this bin-level
ridge-first goal. Poisson computation and inferential extensions require a
subsequent scoped decision.

## Readiness for Implementation Planning

The current first-pass scientific and display choices are settled: native
predictors, adjustable `alpha=1.0`, both ridge encodings plus both selectable OLS
baselines, simple five-fold behavioral-block prediction scores on fixed session
PCA axes, all nonzero PCs, raw/standardized coefficient inspection and ordering,
and a compact observed/predicted trace. Units and Poisson remain subsequent
work; the first delivery focuses on PC inspection.

Post-choice context uses adjacent unfiltered trial states, with terminal post
bins unavailable. Verify the source mapping with focused tests before real-data
interpretation. No per-time-offset fit or additional within-bin update policy
is required.

The companion implementation plan records verified reuse boundaries, tests
written and committed first, performance considerations, saved-result contracts,
and the synthetic-to-single-session validation gates required by AGENTS.md and
SoftwareDesign.md. Plan approval remains required before coding.
