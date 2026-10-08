# Inter-Regional Neural Regression Specification, Version 3

## Status and purpose

This document supersedes `spec_neural_regression_updated.md` for the first-pass implementation.

The analysis measures directed predictive relationships between prefrontal cortex (PFC) and
hippocampus (HPC) population activity. Its central question is:

> Does recent activity in one recorded region improve prediction of current activity in the
> other region beyond what can already be predicted from the target region's own recent
> population activity?

Analyze both HPC to PFC and PFC to HPC. Retain results for every target unit or principal
component (PC), then summarize the target distribution with individual points, median, and
interquartile range (IQR).

This is an inspection analysis, not a publication-finalized causal-inference pipeline. Results
describe prediction in the recorded populations. They do not establish direct anatomical or
mechanistic influence.

Implementation proceeds in this order:

1. Cross-validated unpenalized ordinary least-squares (OLS) regression for units and PCs.
2. Cross-validated unpenalized Poisson regression for unit counts, including comparison with
   OLS on held-out spike-count mean squared error (MSE).
3. Descriptive in-sample linear and Poisson Granger-style analyses.

Do not begin a later stage by weakening or bypassing requirements from an earlier stage.

## Analysis scope and regional populations

Analyze one simultaneously recorded session at a time. Never combine neuron columns from
different sessions as though they represented one population.

The first implementation supports the metadata-driven loading route only. The legacy manual
input route is outside the first-pass scope.

The user must explicitly select one population for the PFC role and one population for the HPC
role. Do not infer anatomy from probe names. Each selection includes its probe, unit population,
unit-quality criteria, and any region-specific channel restriction exposed by the application.
The two selected unit sets must be disjoint.

Stable unit identity is the qualified pair `(probe_id, cluster_id)`, displayed in a form such as
`probe_id:cluster_id`. A bare cluster ID is not globally unique and must not be used as the result
identity.

If one physical probe spans more than one anatomical area, the selected channel or unit subset
must already identify the desired regional population. Anatomical subdivision inference is not
part of this analysis.

## Trial identity, validity, conditions, and filters

Use zero-based trial-table row positions as the internal trial identity. Preserve the original
DataFrame index label as provenance, but do not use it for array indexing, fold assignment, or
joins.

Reuse the condition definitions in `src/neural_analysis/spike_behavior/trials.py`. Do not create
a second condition ontology. Switch and stay labels retain the codebase's established meaning:
they compare the current unrewarded trial's action with the immediately following trial-table row's
action, but only when that following row is valid and has an action. They do not skip invalid rows
and do not mean a context transition.

The accepted saved condition names, in canonical order, are:

```text
all, correct_rewarded, incorrect, omission, switch, stay,
omission_switch, omission_stay, incorrect_switch, incorrect_stay
```

The default request is `all`, `correct_rewarded`, `incorrect`, `omission`, `switch`, and `stay`.
The existing helper's `rewarded` alias is not a regression configuration name because it duplicates
`correct_rewarded` without adding a distinct scientific condition.

The `all` condition has one authoritative scientific meaning for this analysis. It is also the
shared **scientific eligibility mask**:

```text
valid experimenter-reward status
AND valid selected-alignment timestamp
AND selected choice filter
AND selected context filter
AND not user-excluded
```

Every other named behavioral condition intersects this eligibility mask with its existing condition
mask. Cross-validated analyses additionally require nonmissing `cur_block`; descriptive Granger
analyses do not. Fit separate models for each requested condition and window. Do not fit one
all-condition model and only score subsets afterward.

Trial provenance records these three booleans separately: `scientific_eligible` for the shared mask,
`condition_included` after intersecting a named condition, and `cv_included` after additionally
requiring a block. Do not use one overloaded flag for all three meanings.

Choice and context filters reuse the existing LFP-summary column names and side encoding: choice
reads `action`, context reads `state_int`, left is numeric 1, and right is numeric 0 after numeric
coercion. Unlike the existing LFP helper's silent empty selection for an absent required column,
this analysis raises a clear input error. Nonfinite values do not match left/right; `all` does not
require that filter column. Condition masks may overlap; when multiple requested conditions
contribute to a shared PCA fitting pool, use the union of eligible trials so that a trial is
included at most once.

Construct the fold mapping from every trial-table row with nonmissing `cur_block` before applying
alignment, condition, choice/context, or user-exclusion filters. A missing-block row remains in
provenance with no fold and reason `missing_block`; it is excluded from CV but may remain eligible
for descriptive Granger. The session must still contain at least five distinct nonmissing blocks
when any CV stage is requested. If a non-`all` choice or context filter is requested and its
required trial column is absent, fail validation instead of silently returning an empty selection.

An unavailable condition, target, window, representation, model, or direction must not prevent
independent valid results from being computed.

## Alignment and windows

Supported alignments are:

- Choice, the default.
- Trial start.

The validated computation configuration must provide regression-specific window settings. The
read-only application displays the saved bounds and lets the user select among already-computed
before, after, and whole result rows; it does not edit or recompute them.

Defaults are:

| Window | Half-open interval |
|---|---:|
| Before | `[-2, 0)` seconds |
| After | `[0, 2)` seconds |
| Whole | `[-2, 2)` seconds |

Window bounds are configurable through `whole_start_s`, `split_s`, and `whole_stop_s`, but
`split_s` is exactly zero seconds and before and after must exactly partition whole at the selected
event boundary. The configured whole window is the full analysis interval used to fit shared
regional PCA bases. The selected before, after, or whole window determines the prediction rows.

All neural bins and analysis windows are half-open. Record the configured bounds and generated bin
edges in results. Every window duration divided by bin size must be integer-valued within absolute
tolerance `1e-9` and zero relative tolerance; otherwise configuration validation fails.

## First-pass neural coverage assumption

This implementation assumes that successfully loaded aligned spike data completely cover every
requested trial window. A zero count is therefore interpreted as observed silence, not as missing
recording coverage.

The first pass does not infer recording coverage, build a per-bin coverage mask, or exclude trials
according to acquisition start or stop times. Loader failures, invalid alignment timestamps, and
nonfinite prepared arrays remain ordinary errors or exclusions, but an absent spike in a valid bin
is not evidence of missing data.

This is a deliberate implementation shortcut for initial inspection. Every result and figure must
carry a concise coverage-assumption note. A future revision may add an explicit coverage mask or
trial exclusion based on recording bounds. That future work must not silently reinterpret results
produced under this version.

## Binned neural activity

Build one count tensor for each selected region, using shared trial rows and bin edges:

```text
counts: integer array shaped (trial, time_bin, unit)
trial_rows: integer array shaped (trial,)
bin_edges_s: float array shaped (time_bin + 1,), seconds relative to alignment
unit_ids: qualified identifiers shaped (unit,)
```

Use raw spike counts per bin for direct-unit responses and predictors. Firing rate may be derived
for display, but it is not an alternative response mode.

Reuse the established event-alignment and half-open binning conventions. Add a regression-specific
count-tensor preparation function rather than routing the analysis through classifier helpers that
concatenate trials or discard trial/time identity.

The count function follows the existing aligned Pynapple tensor convention locally, obtains trial
metadata by zero-based positional indexing, validates finite nonnegative integer-valued output, and
casts to `int64`. It does not modify or extract a helper from the existing PCA module and does not
derive counts by rounding rates. A spike on a bin's left edge is included; a spike at the final
right edge of the configured window is excluded.

Generate the whole-window edge array once from the validated integer bin count, setting its two end
values exactly to the configured bounds. Select before/after by validated integer edge positions so
both regions and every downstream window share identical bin geometry.

The count-tensor trial axis is exactly the ascending zero-based rows in the scientific eligibility
mask, whether or not `all` is requested as an output condition. Named-condition selection is a
submask on that fixed axis. Missing-block rows remain available for descriptive Granger and are
removed only by the CV mask. Other excluded/invalid rows remain in provenance tables but are not
passed to the spike-binning call.

Use unsmoothed activity. Do not use centered or future-looking smoothing.

## Observations and history construction

One observation is one eligible original trial and one target time bin.
Persist target-bin identity as its zero-based position on the configured whole-window bin axis;
before/after selection does not renumber bins.

For target region `Y`, source region `X`, target feature `j`, trial `k`, and bin `t`:

```text
response:
    current-bin activity y_j(k, t)

restricted predictors:
    all retained target-region features at the requested historical bins

full predictors:
    the same target-region history
    plus all retained source-region features at the same historical bins
```

The target history includes the target feature's own history and every other included target-region
feature. It is not a self-history-only baseline.

Parameters are:

| Parameter | Default | Allowed values |
|---|---:|---|
| Neural bin size | 100 ms | 500, 100, 50, or 20 ms |
| Prediction lag `ell` | 1 bin | Positive integer |
| Model order `p` | 1 bin | Positive integer |

For target bin `t`, historical bins are:

```text
t - ell, t - ell - 1, ..., t - ell - p + 1
```

Target and source populations use the same history set. Feature ordering is deterministic: history
bin from most recent to oldest, then the stable feature order within each bin.

Histories must remain inside the original trial and selected window. Never:

- use a bin from a preceding trial;
- borrow a pre-event bin for an after-window response;
- concatenate selected trials into a continuous series; or
- close a missing-bin gap by using the previous available bin.

Drop initial target bins lacking the required history. With 100-ms bins, lag 1, and order 1, a
2-second window contributes 19 rows per eligible trial and a 4-second window contributes 39.
In general, if a selected window has `B` bins, its first target-bin position is
`ell + p - 1` and it contributes `B - (ell + p - 1)` rows per eligible trial. A nonpositive result
makes that condition/window configuration unavailable with `history_exceeds_window`.

Pool eligible trial by target-bin rows within a condition/window and fit one coefficient set per
target. Do not fit separate coefficients by trial or event-relative bin. Held-out predictors are
observed histories; predictions are not fed recursively into later bins.

## Neural representations

### Direct units

Direct units are the default. Use spike counts for target responses and regional history vectors.
Preserve every selected target unit's qualified identity across folds and outputs. Do not
automatically search for smaller predictor sets.

### Regional PCs

PCs are a dimensionality-reduced representation of the same regional activity. Use up to 10 PCs
per region by default, configurable by the user. Retain fewer when fewer usable dimensions exist
and record the requested and actual counts.

For each outer training fold, alignment, bin size, and selected regional unit set:

1. Pool training-trial observations over the configured whole window and the union of requested
   conditions after applying the shared filters.
2. For each region independently, calculate training means and standard deviations by unit.
3. Omit and report units with zero or nonfinite training standard deviation.
4. Z-score retained units once using those training statistics.
5. Fit separate PFC and HPC PCA projections without whitening.
6. Transform held-out observations with only the training-derived statistics and projection.
7. Do not standardize PC scores after PCA.
8. Reuse each fold's projections across directions, restricted/full fits, requested conditions,
   and windows.

No held-out observation may affect scaling, unit omission, component directions, or retained
component count. Never fit a joint PFC-plus-HPC PCA.

All PCA calculations use `float64`, population standard deviation (`ddof=0`), and scikit-learn PCA
with `svd_solver="full"` and `whiten=False`. The actual component count is
`min(requested_count, training_observation_count, retained_unit_count)`. This path uses no random
number generator; results record `randomness_used=false` and `random_seed=null`. Do not add PC sign
matching, cross-fold alignment, or post-PCA scaling. Downstream metrics are invariant to a component
sign flip.

Target PC scores are summarized by fold-specific component rank. Restricted and full models in a
fold share axes, but PC1 is not treated as one fixed session-wide trajectory across folds. A rank
missing in any requested fold has no primary complete CV summary. Emit an explicit unavailable fold
row for each missing requested rank rather than omitting its key.

For descriptive Granger analysis only, fit one separate PCA basis per region over the configured
whole window and the union of requested condition masks after scientific eligibility. This is
the same condition-pool rule as fold PCA but uses all scientifically eligible rows rather than CV
training folds. Reuse that descriptive basis across conditions, windows, and directions. Never use
it for CV scores.

PC responses use OLS only because scores are continuous and may be negative.

## Session-level grouped cross-validation

Use exactly five deterministic `GroupKFold` folds based on the trial-table column `cur_block`.
There is no existing neural-regression fold implementation to reuse; create and test one.

Build one session-level mapping from zero-based trial row to fold using one sample for each complete-
session row with nonmissing `cur_block`, before any alignment, condition, choice/context, or user-
exclusion filtering. Call `GroupKFold(n_splits=5, shuffle=False)` and number folds by split
enumeration from 0 through 4. Requirements are:

- every CV-eligible row has a nonmissing `cur_block`; missing rows are retained only as excluded
  provenance;
- the session has at least five distinct nonmissing blocks in the unfiltered row universe;
- an entire block belongs to one test fold;
- all bins from a trial remain in its trial's fold;
- the mapping is reused across directions, conditions, windows, representations, and model
  families; and
- grouping is deterministic and nonshuffled.

Nonmissing block labels may be strings, non-Boolean integers, or finite floats. Normalize integral
finite floats to integers before canonical JSON encoding, so values such as `1` and `1.0` cannot
split one logical block across folds. Nonintegral finite floats and strings remain distinct.
Canonical JSON strings supply both the GroupKFold group values and persisted provenance. Arrays,
mappings, Booleans, infinities, and NaNs are invalid.

Do not infer blocks, fall back to random splitting, split individual bins, reduce the fold count,
or search random seeds. If a requested condition has no train or test rows in one of the five
session folds, its primary five-fold result is unavailable.

All preprocessing and fitting use training observations only. No inner CV is required because the
models are unpenalized and parameters are prespecified.

## Matched restricted and full designs

For target region `Y` and source region `X`, define `h_Y` and `h_X` as the complete ordered history
vectors. Every model includes an intercept.

Restricted OLS:

```text
y_j = intercept + b_j @ h_Y + error
```

Full OLS:

```text
y_j = intercept + b_j @ h_Y + c_j @ h_X + error
```

For unit counts, the Poisson models use the same linear predictors with a log link and unpenalized
maximum-likelihood fitting.

Restricted and full models must:

- be fitted independently, including their intercepts;
- use identical training rows and identical test rows within a comparison;
- derive that common row mask from everything required by the full model;
- use identical target responses, folds, temporal parameters, and preprocessing; and
- differ only by the complete source-region history block.

Do not obtain a restricted result by zeroing full-model coefficients. Do not reduce a source region
to an average trace or selected feature pair.

With `N_Y` target features, `N_X` source features, and order `p`, the full design has
`1 + p * (N_Y + N_X)` coefficients per target.

## OLS cross-validated prediction

For held-out response vector `y`, prediction `y_hat_m`, and held-out mean `y_bar`:

```text
SSE_m = sum((y - y_hat_m)^2)
SST_test = sum((y - y_bar)^2)
R2_m = 1 - SSE_m / SST_test
delta_R2 = R2_full - R2_restricted
MSE_m = SSE_m / n_test_rows
```

The held-out mean is a scoring reference, not fitted-model input. Preserve negative finite scores
and increments. If `SST_test` is zero or nonfinite, R-squared is unavailable for that fold and
target; other defined metrics may remain available.

Label the primary OLS output **Incremental CV R-squared** and retain absolute restricted and full
CV R-squared values.

## Poisson cross-validated prediction

Poisson responses are nonnegative integer counts per bin. Do not z-score them or convert them to
rates. Predicted means must be positive and finite.

For observations `y_i` and expected counts `mu_i`, calculate Poisson deviance with the conventional
zero-count logarithmic term equal to zero:

```text
D(y, mu) = 2 * sum(y_i * log(y_i / mu_i) - (y_i - mu_i))
```

For `y_i = 0`, define `y_i * log(y_i / mu_i) = 0`, so that observation contributes `2 * mu_i`.
The null scoring prediction is the mean of the evaluated held-out counts, repeated for every row.

```text
deviance_explained_m = 1 - D_m / D_null_test
delta_deviance_explained =
    deviance_explained_full - deviance_explained_restricted
```

Use the same null deviance for a restricted/full pair. If it is zero or nonfinite, the normalized
score is unavailable. Preserve negative finite scores.

Label the primary output **Incremental CV deviance explained**. Do not call it R-squared.

For direct comparison of OLS and Poisson on units, calculate held-out spike-count MSE on the same
targets, rows, folds, and restricted/full configuration. The main comparison uses full models but
retains restricted-model MSE. Do not round expected counts or clip negative OLS predictions.
Label this an exploratory prediction comparison, not a formal test of model superiority.

Construct this comparison as a derived exact inner join of OLS and Poisson fold rows on session,
direction, condition, window, unit target, fold, and evaluation scope. A paired fold exists only
when both model families have identical canonical train- and test-row-set fingerprints and both
requested MSE values are defined. A row-set fingerprint is SHA-256 over compact JSON containing the
lexicographically sorted unique integer `(trial_row, target_bin_position)` pairs. A target-level
comparison is available only when all five paired folds are available.
For each matched fold define `mse_advantage_poisson = mse_ols - mse_poisson`, so a positive value
means Poisson had lower held-out error. The target comparison is the arithmetic mean of its five
paired fold differences. Report the full-model comparison as primary and retain the restricted
comparison for inspection. Do not persist a redundant comparison table; plotting and reporting
derive it from `fold_scores`.

## Fold and population aggregation

For each target, condition, window, direction, representation, and model family:

1. Calculate restricted and full metrics on the same test rows in each fold.
2. Calculate the paired increment within each fold.
3. Retain each fold's status, reason, train/test trial counts, and train/test row counts.
4. Form the target's primary summary as the arithmetic mean of all five valid paired fold values.
5. If any requested fold is invalid, retain its inspection data but mark the primary target summary
   incomplete and unavailable.

Only comparable normalized or per-row CV metrics enter target and population summaries. For OLS
these are `r2_restricted`, `r2_full`, `delta_r2`, `mse_restricted`, and `mse_full`. For Poisson they
are `deviance_explained_restricted`, `deviance_explained_full`,
`delta_deviance_explained`, `mse_restricted`, and `mse_full`. Raw restricted/full/null deviance is
retained only as fold-level diagnostic evidence and is never averaged across folds or targets.

Across complete targets, report individual target values, median, 25th percentile, 75th
percentile, and contributing count. IQR describes target variability, not a confidence interval.
Calculate quartiles with NumPy's `method="linear"` over complete target means; never pool fold
values before computing the target distribution.

## Descriptive Granger analyses

Granger analyses are implemented only after the OLS and Poisson CV features are complete. Once
implemented, either Granger stage may be requested without rerunning its CV counterpart. Granger
reuses the same history, condition, window, target-identity, and matched-design definitions, but
fits once to all scientifically eligible rows for a configuration; `cur_block` is irrelevant to
that in-sample fit.

### Linear Granger magnitude

For each unit or descriptive session-PC target:

```text
G_linear = log(SSE_restricted / SSE_full)
```

Both residual variances use the same fitted-row denominator, so it cancels in the ratio. Label this
**Linear Granger magnitude - in-sample log residual-variance ratio**.

### Poisson Granger-style magnitude

For each unit target:

```text
LR = 2 * (log_likelihood_full - log_likelihood_restricted)
   = deviance_restricted - deviance_full

G_poisson = LR / n_rows
```

Retain raw `LR`, log likelihoods, deviances, and `n_rows`. Display `G_poisson` as **Mean deviance
improvement - Poisson Granger-style, in-sample**.

These are descriptive nested-model improvements. Do not add p-values, significance stars, F-tests,
chi-squared inference, surrogate tests, or multiplicity correction. A substantive negative
in-sample improvement is a fit-consistency problem, not evidence of negative influence. Ordinary
floating-point roundoff around zero may be tolerated and recorded.

When the requested prediction lag exceeds one, preserve the exact requested history and label the
result a lag-restricted Granger-style comparison. Do not silently add the omitted recent lags to
make a conventional contiguous VAR history.

Do not equate the scales of linear Granger, Poisson Granger-style improvement, incremental CV
R-squared, or incremental CV deviance explained. Do not describe the median target value as joint
multivariate regional Granger causality.

## Estimability, fit validity, and unavailable results

For every fitted design:

- require matrix rank equal to the number of coefficients, including the intercept;
- require more fitted rows than coefficients so residual degrees of freedom are positive;
- require a nonconstant training response for the target;
- require finite coefficients and predictions;
- for Poisson, require optimizer convergence and positive finite means; and
- preserve the common restricted/full row mask.

Do not silently drop predictors, regularize, switch representations, reduce model order, reduce
fold count, or change bins to make a configuration fit. Constant predictor columns therefore make
the affected direct-unit design rank-invalid. This strict policy is expected to produce unavailable
unit-level configurations, especially for small bins or restricted conditions.

Metric-specific invalidity does not erase other defined metrics. For example, a constant held-out
response has undefined R-squared but may still have valid MSE.

Failures are data, not exceptions to hide. Preserve a machine-readable status and reason for each
independent result.

Poisson fitting stores only a nullable convergence Boolean and nullable IRLS iteration count for
each restricted/full fit, plus its status and reason. A convergence warning or returned
nonconvergence is locally unavailable; perfect separation or a recognized numerical fitting error
is a local fit error. Continue independent targets, but allow unexpected programming, input-
contract, or library-API errors to reach the run boundary. Do not save warning histories, optimizer
histories, or estimator objects.

Use these numerical comparison constants throughout the implementation:

```text
BIN_GEOMETRY_ATOL = 1e-9
METRIC_RTOL = 1e-9
METRIC_ATOL = 1e-12
```

Use `numpy.linalg.matrix_rank` with its documented default SVD tolerance for estimability; tests
must avoid ambiguous near-threshold matrices. Treat held-out SST or null deviance as zero under
`isclose(value, 0, rtol=METRIC_RTOL, atol=METRIC_ATOL)`. For theoretically nonnegative nested-fit
improvements, tolerate a negative value only up to
`METRIC_ATOL + METRIC_RTOL * max(abs(restricted), abs(full), 1)`; record it as zero with a roundoff
diagnostic. A more negative value is a fit inconsistency. Use the same tolerance for Poisson
likelihood-ratio/deviance-difference agreement. Do not round persisted scientific metrics.

## Frozen configuration contract

Use these exact version identifiers:

```text
config_schema_version = "1"
result_schema_version = "1"
analysis_version = "interregional-regression-v1"
coverage_assumption_version = "implicit-complete-v1"
```

The one portable JSON file contains the scientific configuration plus a separate `run` section.
The scientific part contains: absolute `session_metadata_path`; explicit `PFC`
and `HPC` population selections (`probe_id`, channel source, zero-based channels, inside-brain rule,
channel-quality labels, unit-quality column, and unit-quality labels); alignment; the three window
boundary values; requested prediction windows; bin size, lag, and order;
conditions, choice/context, and zero-based excluded trial rows; requested PC counts per region;
representations; and requested analysis stages. The `run` section may contain only optional
`output_root`. Unknown keys are errors.

Excluded trial rows must be unique, nonnegative, and within the loaded trial-table row count.

Allowed representations are `units` and `pcs`; the default is `units`. Allowed stages, in fixed
execution order, are `ols_cv`, `poisson_cv`, `linear_granger`, and `poisson_granger`; the default is
`ols_cv`. Poisson stages require units. `poisson_cv` requires `ols_cv` because their matched
count-MSE comparison is a required output. Granger stages have no runtime dependency on CV stages;
the ordered list states implementation and execution order when multiple stages are requested, not
a requirement to rerun earlier analyses. Exactly five folds and the group column `cur_block` are
specification constants, not configurable fields.

Allowed prediction-window names are `before`, `after`, and `whole`; all three are requested by
default and each receives a separate fit.

Channel source is either `metadata_quality` or `explicit`. Metadata-quality selection uses the
requested channel labels and inside-brain rule, intersected with metadata `unit_channels` when
present; explicit selection requires a nonempty channel list. Unit-quality filtering always
applies. Defaults are channel label `good`, inside-brain required, unit-quality column `group`, and
unit labels `good,mua`.

Unresolved population selections do not contain cluster IDs. Loading produces a separate resolved
population record with ordered selected channels, cluster IDs, and qualified unit IDs, and only
then validates that both sets are nonempty and PFC/HPC are disjoint. Output root, rerun, and worker
count are execution choices, not scientific configuration fields.

## Result records

The computational core returns ordinary typed records and tidy pandas DataFrames in memory. The
run boundary persists completed analyses through a small, separate persistence component; fitting,
scoring, and plotting functions do not write files themselves.

At minimum retain:

- complete analysis configuration and coverage-assumption version;
- session ID, qualified PFC/HPC unit IDs, unit ordering, and original trial-index provenance;
- session trial-to-fold mapping with `cur_block` when CV is requested;
- condition/filter masks and eligible trial counts;
- bin edges, window, lag, order, and deterministic predictor ordering;
- fold-level fit status, reason, feature counts, ranks, residual degrees of freedom, row counts, and
  canonical train/test row-set fingerprints;
- restricted/full metrics and paired increments;
- target-level complete or incomplete summaries;
- population median, quartiles, and contributing target count;
- PCA training scope, retained/omitted unit IDs, actual component counts, and fold identity;
- model family and whether a result is held-out or in-sample; and
- nullable Poisson convergence flags and IRLS iteration counts where applicable.

Use the exact named tables and column/key/status contracts in
`docs/neural_regression_plan.md`: `fold_assignments`, `trial_membership`, `fold_scores`,
`target_summaries`, `population_summaries`, `pca_fits`, and `granger_scores`. Later-stage tables are
present with empty frozen schemas before those stages are implemented. This prevents OLS, Poisson,
and Granger additions from silently changing earlier saved records.

The only saved statuses are `ok`, `metric_unavailable`, `fit_unavailable`, `incomplete_folds`, and
`not_applicable`. Machine-readable reason codes are frozen in the implementation plan. An expected
scientific unavailability uses one of them; an unexpected data-contract/programming failure remains
an exception at the run boundary.

Do not store one opaque nested result dictionary when a small number of explicit tables provides a
clearer scientific record.

Persist each execution as one immutable project run directory, normally:

```text
<session_root>/analysis_runs/
    interregional_regression_<YYYYMMDD>T<HHMMSSffffff>Z_<fingerprint12>/
```

The timestamp is UTC with microseconds and directory creation is exclusive.

It contains `config.json`, `input_manifest.json`, `result.pkl`, `run.log`, `summary.md`, exact
copies of the session and batch runner scripts, and `figures/`. There is no
second mutable or canonical result tree. Outputs must remain in the session data hierarchy or a
user-designated nonrepository directory.

The pickle contains only the pure computational result: the named tables plus versions, complete
scientific parameters, whole-window bin edges, the fixed count/history axes and units, resolved
populations, and explicit randomness fields. Execution provenance, generating entry point, runtime
versions, and file identity remain in the manifest rather than being injected into the
computational record. Loading is restricted to project-generated local run directories; untrusted
pickle files are unsupported.

The SHA-256 `run_fingerprint` covers versions, canonical scientific configuration except the
location-only `session_metadata_path`, session ID, ordered resolved unit IDs, streamed SHA-256
content identity for every consumed metadata, trial, spike, sorter, cluster, and channel-quality
input, Git HEAD, and exact Python/computation-library versions. The manifest records each normalized
absolute path, but paths are excluded from the fingerprint so relocating identical inputs does not
change scientific run identity. Timestamp, output root, rerun, worker count, and Streamlit version
also do not affect it. Dry-run validates paths and reports file sizes without reading every large
file to compute a final fingerprint; `new` computes the streamed hashes before fitting. An
identical validated finalized run is reused by default. Explicit rerun creates a second immutable
timestamped directory with the same fingerprint and never overwrites the first.

Because Git HEAD is the code identity, real `new` runs require a clean tracked worktree and no
untracked Python files under `src/neural_analysis`. Dry-run reports either violation. Other
untracked files do not change the fingerprint.

After hashing and reuse detection, a new computation writes to a hidden same-parent directory named
`.interregional_regression_<timestamp>.incomplete`. Once every required artifact validates, rename
that directory atomically to the timestamp/fingerprint name above. Finalized-run discovery ignores
the `.incomplete` suffix. On a caught failure, retain the incomplete directory with its log and a
small `failure.json`; an abrupt interruption may leave partial artifacts. Incomplete directories are
neither resumed nor automatically deleted in this first pass. This directory-level completion
marker replaces a separate run-state machine and per-file durability protocol.

The summary states the analysis goal, included session, scripts, configuration, output, warnings,
unavailable counts, coverage assumption, and scientific interpretation. The log records runtime
environment, parameters, session, stages, warnings/errors, and execution time.

Provide one single-session runner and one batch runner. The batch runner calls the single-session
pipeline independently for each session, offers a dry-run mode, and parallelizes across sessions by
default. Its initial worker count is the lesser of available CPU cores, session count, and any lower
override. Dry-run reports a clearly labeled advisory memory estimate, not an automatic safety cap.
Actual batch execution requires a separately approved exact worker count after measured peak memory
from the representative single-session inspection. It never pools unit columns or observations
across sessions.

## Figures, computation configuration, and application display

The main CV figure displays all requested conditions horizontally. Within each condition, show HPC
to PFC and PFC to HPC with individual target points and median/IQR overlays. Use stable point-to-
target identity and report unavailable/contributing counts.

Primary axes are:

- OLS: **Incremental CV R-squared**.
- Poisson: **Incremental CV deviance explained**.

Absolute restricted/full scores and fold-level values must remain inspectable. Do not overlay unlike
metrics on one axis.

Create a separate Granger summary figure with the same condition/direction structure and prominent
in-sample/descriptive labeling. Do not place Granger and CV metrics on the same numerical axis.
Export figures as PNG by default. Each figure has a readable caption summarizing its contents,
contributing targets, and scientific interpretation; no inferential result is implied.

Regression-specific computation settings are required in the validated JSON/CLI workflow:

| Setting | Options/default |
|---|---|
| PFC population | Explicit metadata population selection |
| HPC population | Explicit metadata population selection |
| Representation | Units default; PCs |
| Model family | OLS default; Poisson for units only |
| Alignment | Choice default; trial start |
| Window bounds | Configurable whole/before/after partition |
| Analyzed windows | Before; after; whole |
| Conditions | Canonical condition list; default set above |
| Choice filter | All; left; right |
| Context filter | All; left; right |
| Neural bin size | 500, 100, 50, or 20 ms; default 100 ms |
| Prediction lag | Positive integer; default 1 |
| Model order | Positive integer; default 1 |
| PCA count | Positive integer per region; default up to 10 |

The Streamlit application is a read-only viewer for atomically completed saved runs. It does not
load raw spike data, edit computation settings, launch/resume analysis, or use transient widget
state as the scientific record. It provides display-only selectors for conditions, windows,
representations, model families, and metrics present in the selected saved result. It also shows
the saved population selections, computation configuration, input/code identity, warnings, and
unavailable reasons. If no completed run exists, it shows the documented CLI dry-run/new commands.
The first-pass viewer discovers only the selected session's default `<session_root>/analysis_runs`
directory. A deliberately nondefault CLI output root remains valid but is inspected outside the
webapp.

This boundary deliberately favors reproducibility and operational simplicity over in-app
computation. A new population, bin size, lag/order, condition set, or analysis stage requires an
edited validated configuration and an explicit CLI run; Streamlit never silently computes a new
result during rerender.

Every relevant figure or description includes:

> First-pass predictive association. Loaded spike data are assumed to cover all requested windows.
> Shared event responses, common input, movement, and slow temporal fluctuations have not been
> separately controlled. Granger magnitudes are descriptive in-sample values, not significance
> tests or evidence of mechanistic causality.

## Dependencies and implementation boundaries

Use existing project dependencies only:

- NumPy and pandas for explicit array/table operations;
- Pynapple or the existing aligned-spike conventions for spike binning;
- scikit-learn `GroupKFold` and PCA with explicit deterministic/unwhitened settings;
- unpenalized NumPy linear algebra for OLS;
- statsmodels unpenalized Poisson GLM with log link; and
- Matplotlib and Streamlit for existing application presentation.

Verify each external API against the installed package source or official documentation before its
first implementation use. Do not inherit estimator defaults without an explicit test.

For the audited statsmodels 0.15.0 environment, Poisson uses `GLM` with an explicit design
intercept, `Poisson(Log())`, `missing="raise"`, and IRLS (`wls_method="qr"`, `maxiter=100`,
`tol=1e-8`), with no weights, exposure, offset, or regularized fit. The explicit QR backend avoids
the default repeated least-squares SVD/pseudoinverse path used by the bounded run that exceeded its
memory allocation. Recheck this contract if the environment changes.

Keep data preparation, folds, PCA, OLS, Poisson, Granger, aggregation, plotting, and Streamlit
integration as focused components with documented array shapes, axis meanings, units, and return
contracts. Prefer plain functions and small immutable configuration/result records over a behavior-
heavy analysis class.

## Deferred work

The following remain outside this version:

- explicit recording-coverage masks and acquisition-boundary trial exclusion;
- legacy manual-input integration;
- formal Granger inference or multiple-comparison correction;
- temporal-confound, behavioral-confound, or movement adjustment;
- temporal purging, surrogate/null analyses, or detrending;
- regularization, automatic feature selection, or lag/order optimization;
- communication subspaces, CCA, mutual information, or joint multivariate regional Granger;
- neuron-count matching and cross-fold PC-axis matching; and
- publication-level causal interpretation.

Before confirmatory interpretation, re-evaluate temporal fluctuations and across-trial dependence.
Retain the reference: Kenneth D. Harris, [*Nonsense correlations in neuroscience* - bioRxiv,
version 3](https://www.biorxiv.org/content/10.1101/2020.11.29.402719v3.full), DOI:
10.1101/2020.11.29.402719.

## Default analysis

| Setting | Default |
|---|---|
| Session | One simultaneously recorded metadata session |
| Regional roles | Explicit PFC and HPC selections |
| Representation | Units |
| Response | Spike counts per bin |
| First model | Unpenalized OLS with intercept |
| Alignment | Choice |
| Windows | Before `[-2, 0)`, after `[0, 2)`, whole `[-2, 2)` seconds |
| Bin size | 100 ms |
| Lag/order | 1/1 |
| Conditions | `all`, correct rewarded, incorrect, omission, switch, stay; separate fits |
| CV | Five deterministic `cur_block` GroupKFold folds |
| PCA | Separate training-only regional bases, up to 10 PCs, no whitening/rescaling |
| Coverage | Loaded spike data implicitly cover requested windows |
| Persistence | Hidden incomplete working directory; atomic rename to immutable finalized run |
| Application | Read-only viewer for validated completed runs; computation through CLI |
| Population display | Individual targets, median, IQR |
| Formal inference | None |

With these defaults, HPC to PFC asks whether observed HPC counts in the immediately preceding
100-ms bin improve prediction of current PFC counts beyond the complete selected PFC population's
counts in that preceding bin. PFC to HPC reverses the regional roles without changing the other
conventions.

## Version 3 changes

Version 3 resolves codebase-integration ambiguities identified after reviewing the existing neural
analysis package. It names `cur_block` as the required CV group, introduces a new deterministic
fold implementation, requires explicit dual-population selection, restricts the first pass to the
metadata route, defines the canonical conditions, `all`, trial identity, and prefilter fold universe
precisely, freezes the computation configuration, requires a local regression count tensor rather
than classifier binning, separates deterministic full-SVD PCA from the existing PCA API, documents
strict rank/tolerance-related unavailability, and records the temporary complete-coverage
assumption. It also specifies one atomically finalized immutable run directory, a read-only saved-result webapp,
session/batch runners, separate scientific/CV eligibility, and the fixed implementation sequence
OLS, then Poisson, then Granger.
