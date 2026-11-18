# Joint unit and regional-PC encoding specification

Status: approved joint-model revision, 2026-11-13. This file retains its original
link/name but replaces the earlier context-only PC-first specification. The
current scientific version is `joint-task-encoding-v2`; the old model is retired.
The approved [implementation plan](single_unit_joint_plan_v2.md) defines the
architecture and tests-first execution record. The earlier
[single_unit_plan.md](single_unit_plan.md) is historical.

## Goal

Assess conditional context, choice, reward, omission and incorrect-outcome
associations in individual units and fixed descriptive regional PCs. Use one
joint bin-level regression throughout a choice-centered **[-1,+1) second**
window. Include both pre-choice and post-choice activity so an outcome effect
must explain a change relative to activity without that outcome indicator.
This is pooled responsiveness over the window, not a per-time-offset fit.

A positive reward coefficient is insufficient evidence for reward selectivity:
generic choice-related activity can produce the same sign. Retain raw
reward-minus-omission contrasts, support diagnostics, and OLS comparisons.
These are descriptive conditional associations, without significance or causal
labels. A selectivity-index approach is a possible future change, not part of
this implementation.

## Activity and eligibility

- Choice alignment only. All coverage, binning, PCA, fitting and neural display
  use exactly [-1,+1). Supported widths: 500 ms (default), 100, 50 and 20 ms.
  Edges include zero; windows and bins are half-open.
- Unit counts have `(window,bin,unit)` axes. Divide by bin duration in seconds
  for linear-model responses in Hz. Keep constant and silent units.
- Independent PFC/HPC PCAs use pooled eligible two-second activity, standardize
  variable unit counts once, and retain all numerically nonzero unwhitened PCs.
  Keep fixed signs/loadings, means, scales and original ranks. Do not refit PCA
  inside regression folds. Missing PCA never invalidates unit analyses.
- Preserve existing good/mua and channel-quality selection. Exclude manual
  rewards, no animal choice, missing choice time and dark/invalid entering
  contexts before PCA. Require complete regional IRIG recording coverage;
  manual-alignment sessions remain deferred. Regions can truncate independently.
- Source and augmented trials must match completely in chronological order,
  original IDs, block IDs, timestamps, context, action, reward, correctness and
  normalized manual-reward flags. Use original row positions; never shift a
  filtered sequence. Logged context events remain diagnostic and unchanged.
- Unknown outcome classifications exclude the entire trial from joint fitting,
  but do not alter the descriptive activity/PCA pool when baseline-eligible.
- Flatten trial/bin axes to observations, retaining original trial, block, bin,
  response IDs and UTC/elapsed/choice-relative second edges. Elapsed zero is
  the first unfiltered trial start. No averaging, smoothing or trial collapse.
- Old four-second PCA products cannot supply a compatible basis or raw units.
  Prepare new activity; do not relabel a PC from a different basis as identical.
  Unrelated standalone timeline APIs retain their four-second defaults.

## Native joint predictors

Context and current choice remain fixed across the entire window. In
particular, a switching choice's post-choice bins retain the context that
led to that choice. Terminal trials do not need a following state.

Two separate designs use matched observations and fold assignments:

| Encoding | Fixed context columns | Fixed choice column |
| --- | --- | --- |
| Signed | `signed_context`: left +1, right -1 | `signed_choice`: left +1, right -1 |
| Separate | `left_context`, `right_context`: native 0/1 indicators | Same signed choice |

Append these six ordered columns to either design:

`left_reward`, `right_reward`, `left_omission`, `right_omission`,
`left_incorrect`, `right_incorrect`.

Every outcome indicator is zero in [-1,0) and one only for its trial's matching
condition in [0,+1). For simplicity, **choice time** is the outcome boundary;
this does not assert actual delivery occurred at choice. Reward=1 means reward,
regardless of whether correctness is available. Reward=0/correct=1 means
omission; reward=0/correct=0 means incorrect. Left/right means the chosen port
(`action`: right=0, left=1), including incorrect choices. Unrecognized action,
reward or required correctness has an explicit unavailable reason.

Do not include a generic post-choice regressor: it equals the sum of these six
columns. Do not offer before-only or after-only fits. After-only fits alias
outcome effects with context/choice and omit the required pre-choice baseline.

## Estimators and penalties

For native predictors `X(N,P)` and responses `Y(N,R)`, fit
`Y_hat = intercept + X @ slopes`, with slopes `(P,R)` and intercept `(R,)`.
Fit one coefficient vector per response and saved configuration. No predictor
or response standardization occurs inside the solver.

Ridge minimizes summed squared error plus `sum_j penalty_j * slope_j**2`,
with an unpenalized intercept. Alpha is finite and positive, default 1.

- `same_numeric`: all feature penalties are alpha.
- `matched_context`: only the two separate-context columns have alpha/2;
  choice and all six outcomes retain alpha. This matches signed-context
  shrinkage without changing non-context penalties. Signed ridge shares its
  equivalent penalty labels rather than refitting duplicate cells.
- OLS is a selectable comparison for both encodings, using centered
  minimum-norm slopes with an unpenalized intercept. Retain redundant context
  columns and record the convention; do not prune them automatically.

Save actual per-feature penalties. Reference fits use all eligible rows.
A display-only standardized coefficient is `beta * SD(X) / SD(Y)` on those
rows. Undefined response scaling stays unavailable. This toggle affects
coefficient display/ordering only; activity and raw contrasts retain native
units.

## Reward contrasts and support

Save raw `left_reward - left_omission` and `right_reward - right_omission`
slopes, in Hz or PC-score units, for the reference fit and every training
fold. Do not subtract separately standardized coefficients.

For each fit, save condition counts, predictor nonzero counts/nonconstant
support, centered-design rank, contrast estimability, and explicit reasons.
A contrast requires both relevant conditions and membership in the centered
design's estimable row space. Retain raw regularized estimates even when
unsupported, but show them as unavailable in supported-effect displays.
Also save train/test condition counts per fold. A missing condition can leave
prediction available while its attribution remains unsupported.

Unequal condition frequencies can produce unequal ridge shrinkage and a
nonzero contrast for a unit that responds equally to every post-choice
condition. Do not force such a ridge result to zero or alter category weights
to obtain that behavior. Check ridge against an independent penalized-system
oracle. Correctly specified noiseless OLS gives zero generic-response
contrasts when those contrasts are estimable. Compare estimator, support and
penalty settings when interpreting reward responsiveness.

## Held-out evaluation

Use deterministic five-fold GroupKFold by `cur_block`, assigned once per
region to baseline-valid full-coverage windows before outcome masking.
All bins from a window/block stay together. Units/PCs, context encodings and
estimators use the same eligible rows and folds. No random bin splitting,
repeated CV, alpha search or inner CV.

Pool all eligible held-out bins for primary MSE and R-squared; do not average
fold scores equally. MSE has squared-Hz or squared-PC-score units. Preserve
negative R-squared. Constant responses retain MSE and unavailable R-squared.
Fewer than five usable blocks, empty required folds or failed/nonfinite fits
leave primary CV unavailable with reasons; keep valid reference/completed-fold
results. Never substitute reference predictions for missing held-out values.

Unit evaluation requires no PCA. PC evaluation is conditional on fixed
**descriptive session PCA that includes test-block neural activity**. It is
not wholly training-only neural preprocessing. More bins do not establish
independent evidence or control temporal confounds.

## Persistence and inspection

One current configuration/schema is supported. Require explicit
`analysis_version: "joint-task-encoding-v2"`. Remove legacy timing/window
selectors and reject old options. Old archives remain untouched; current run
discovery skips them and direct loading/resume rejects them. No compatibility
reader, migration or legacy model dispatch is required.

Prepare and save counts once, common row/fold geometry and optional PCA once.
Process/publish one region/representation/encoding/estimator/penalty cell at a
time. Save named arrays with `meta` carrying generator, version, parameters,
source/code/runtime identities, units/axes and randomness provenance. Counts
are session-local under `processed/context_encoding`; models and human-readable
outputs live in timestamped `analysis_runs/context_encoding_*`. Never overwrite
scientific products. Resume from compatible saved products without live sources,
including adoption of a published cell whose index update was interrupted.

Saved viewer controls select units/PCs, original response IDs, model, raw or
standardized coefficients, coefficient/raw-contrast ordering, visible rows,
selected-response prediction scope, and elapsed range. Coefficient/activity
rows share identical identities/order. Unit heatmaps are nonnegative Hz with
fixed regional limits; PCs retain fixed signed score limits. Preserve actual
time, blank gaps, pan/zoom and categorical coefficient axes. An optional
within-unit heatmap z-score uses each unit's mean/population SD over all saved
two-second bins, excluding gaps; constant units display zero. Its adjustable
signed color range defaults to +/-3 SD and stays fixed across navigation.
This visualization leaves prediction traces in Hz and never changes fitted
responses, coefficients or saved products. Show fixed context/choice and all
six gated outcome tracks with included-bin status.
No display control loads spikes, bins, refits PCA or fits regression.

Export light opaque PNGs for each available region/representation. Captions
state the joint model, exact two-second activity scope, choice-time outcome
convention, original response ID, units, coefficient scale, prediction scope
and contrast interpretation. No significance labels. Record measured timing,
payload/storage bytes and peak process RSS, with its process-lifetime scope.

## Validation and experimental gate

Use pytest with RED, tests-only commit, GREEN and review for each implementation
phase. Generated-spike validation includes side-specific reward/omission/
incorrect, context, choice, generic post-choice, silent/constant and noisy units;
verify unit coefficients and fixed-PC projections with independent oracles.
Include balanced/imbalanced frequencies, fold fits, missing conditions,
unavailable PCA, source-free resume, saved UI and exports. Verify unrelated
timeline defaults. Use existing dependencies, uv, and session-level workers.

The user subsequently designated CT026 2026-08-28 and authorized its numerical
run, which completed with all 20 cells and five-fold scores available. CT026
2026-08-04 was dry-run only. The user inspects the 8/28 output before wider
batch processing. The 2026-11-14 z-score visualization addition uses the same
saved fits without reprocessing. Poisson,
selectivity-index methods, causal claims, automatic alpha tuning, permutation
significance and fraction-selective summaries remain outside this revision.
