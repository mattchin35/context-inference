# Joint Context, Choice, and Outcome Encoding: Revision Plan

Status: J1-J6 implemented and synthetically validated, 2026-11-13, following
user approval. Experimental numerical execution remains a separate gate.
Revised after SoftwareDesign review: the user confirmed that the old encoding
model does not need to persist. This plan replaces it with one current joint
model; existing files are preserved without legacy execution or reader support.

The user selected a choice-centered +/-1-second analysis, trial-fixed context
and choice, separate left/right reward and omission/incorrect indicators, a
joint regression, and both unit and PC responses. Units in Hz and regression
reward-minus-omission contrasts were accepted in the preceding discussion.
The user subsequently confirmed the whole-window joint fit: it must include
the pre-choice baseline with no active outcome indicator as well as the
post-choice outcome period.

This approved revision supersedes the conflicting scientific choices in the
earlier [single_unit_plan.md](single_unit_plan.md), whose P1-P6 execution record
remains historical. The linked
[scientific specification](single_unit_ridge_spec_v1.md) now records this joint
model while retaining its established filename.
Follow [AGENTS.md](../AGENTS.md) and [SoftwareDesign.md](SoftwareDesign.md).

## 1. Deliverable and limits

Extend the existing encoding package to fit one joint context/choice/outcome
model to raw units and descriptive regional PCs, then inspect saved
coefficients, predictions, and reward contrasts in the existing webapp.

The revised scientific window is exactly `[-1,+1)` seconds relative to choice.
This applies to coverage, spike binning, PCA preparation, regression, scores,
and displayed neural windows. No activity from the outer portions of the old
`[-2,+2)` window enters a new fit or PCA.

Keep independent PFC/HPC populations, existing channel selection and good/mua
units, exclusion of manual-reward/no-choice/dark trials before preparation,
native predictor coding, unpenalized intercepts, ridge/OLS, five block-grouped
folds, and saved-only inspection. Manual alignment remains deferred.

No selectivity-index implementation, Poisson model, significance testing,
automatic alpha selection, time-resolved coefficient scan, or new dependency
is included. Results describe associations; they do not establish a causal
response to reward or a significant responsive/nonresponsive classification.

## 2. Exact predictor contract

Use the current trial's entering `state_int` and its recorded `action`
throughout the window. A context switch after that choice does not change the
window's context. There is no adjacent-next-trial lookup in the new model, so
a completed terminal trial can contribute its full window.

Choice and outcome side mean the port actually chosen, not the currently
rewarded context. Preserve source codes `0=right`, `1=left`.

| Column | Before choice `[-1,0)` | After choice `[0,1)` |
| --- | --- | --- |
| `signed_context` | Right=-1, left=+1 | Same value |
| Alternatively `left_context`, `right_context` | Native 0/1 pair | Same pair |
| `signed_choice` | Right=-1, left=+1 | Same value |
| `left_reward`, `right_reward` | Both zero | One if rewarded on that side; otherwise zero |
| `left_omission`, `right_omission` | Both zero | One if correct and unrewarded on that side; otherwise zero |
| `left_incorrect`, `right_incorrect` | Both zero | One if incorrect and unrewarded on that side; otherwise zero |

The signed and two-indicator context encodings are separate fits. Choice stays
signed in both, so context-encoding comparisons do not also change choice
coding. With signed context there are eight slopes; with separate context
there are nine. Neither predictor matrix includes a ones column.

Classify outcome from recorded fields: reward=1 is rewarded; reward=0 with
correct=1 is omission; reward=0 with correct=0 is incorrect. A rewarded trial
does not require guessing correctness to classify reward. Missing/invalid
reward, or unknown correctness on an unrewarded trial, makes the joint outcome
unknown. Exclude that entire trial window from the joint regression, retaining
an explicit reason. Do not encode unknown outcomes as six zeros. Such trials
can remain in the baseline descriptive activity/PCA pool.

Every fully classified eligible trial has exactly one active outcome column
in every post-choice bin, and none in a pre-choice bin. Boundaries are
half-open: zero is post-choice and +1 is outside the window. Retain 500-ms bins
by default and support 100/50/20 ms; these give 4/20/40/100 total bins/window.
No smoothing or bin collapsing is introduced.

This is a retrospective choice association: the selected action is assigned
to pre-choice neural bins for analysis. Likewise, a zero pre-choice outcome
regressor does not assert that anticipatory neural activity is absent.

## 3. Primary fit scope and identifiability

**Confirmed fit scope: one joint fit over the whole
`[-1,+1)` window.** Before/after remain useful inspection intervals, rather
than separate joint model fits for this revision.

The pre-choice bins constrain the intercept/context/choice baseline with all
outcome indicators zero. Post-choice bins allow the corresponding outcome
coefficient to describe an additional change relative to that baseline.
Estimate the change from the data; do not require or force a lower baseline
or a positive outcome response. Negative changes can describe suppression.
Reward-minus-omission contrasts further compare two different post-choice
outcomes, rather than treating a generic post-choice increase as reward-specific.

With these six exhaustive outcome categories:

`left_reward + right_reward + left_omission + right_omission
 + left_incorrect + right_incorrect = post_choice`.

Do not add a seventh generic post-choice indicator: it is exactly redundant.
Activity common to all outcomes can be represented by equal outcome slopes.

An after-only model has a more serious attribution problem: outcome category
already determines side and, under the ordinary task contingencies, context.
Context, choice, intercept, and outcome effects then cannot be separated
uniquely. Before-only data contain no active outcome predictors at all.
Ridge would produce numbers in either case, but its numerical uniqueness
would not solve scientific identifiability.

A small synthetic design with all six trial categories has rank 9 for the
nine signed-context parameters including intercept across the whole window,
rank 3 before choice, and rank 6 after choice. This calculation is a design
check, not a claim that every experimental session has adequate support.

Save condition counts, centered-design rank, and predictor support for the
reference fit and each training fold. Exact context/choice aliasing, absent
conditions, or a missing side must be visible. Do not reject every model just
because one effect is unsupported. Predictions and supported contrasts may
remain useful. Separate context indicators intentionally retain their
intercept redundancy; saved slopes retain the declared ridge/minimum-norm
convention rather than being called uniquely identified category effects.

## 4. Regression and reward contrasts

For X(N,P), Y(N,R), slopes B(P,R), and intercept a(R), retain summed SSE:

`sum((Y - 1*a.T - X@B)**2) + sum_j lambda[j] * sum(B[j,:]**2)`.

Use raw unsmoothed unit rates in Hz and fixed unwhitened PC scores as separate
response representations. Fit all responses of one representation in a
multi-response library call. Do not normalize Y for fitting or clip negative
linear predictions for units. OLS retains centered minimum-norm slopes.

`same_numeric` assigns alpha to every slope, default alpha=1. For
`matched_context`, only the two context-indicator penalties become alpha/2;
choice and all six outcome penalties remain alpha. Signed-context penalties
remain unchanged, so its two penalty labels share one computed fit.

The existing implementation halves a scalar alpha for a context-only model.
That must not be applied to every column of the joint model. Implement the
per-column penalty by a documented internal column reparameterization around
the existing Ridge SVD solver, then map slopes back to native predictor units.
Save the actual penalty vector. This is a numerical solve, not SD-based
predictor standardization. Verify the installed package API before changes.

For each response, save these planned raw contrasts:

* `left_reward_minus_omission = beta_left_reward - beta_left_omission`.
* `right_reward_minus_omission = beta_right_reward - beta_right_omission`.

Each compares reward with omission at the same chosen side and ordinary
correct context. Incorrect coefficients remain separately inspectable; they
are not pooled into the reward-versus-omission comparison. For a correctly
specified noiseless OLS example with estimable contrasts, an equal generic
post-choice increase has equal outcome slopes and zero contrasts within
floating-point tolerance.

That zero-contrast expectation does not apply generally to ridge. Unequal
condition counts and correlated predictors can cause different shrinkage of
reward and omission coefficients, even when their true responses are equal.
A deterministic design check with a common 3-unit post-choice increase and
unequal category counts produced ridge contrasts of approximately 0.20 and
0.29 at alpha=1, while OLS contrasts were zero. These are illustrative design
calculations, not thresholds or experimental results. Keep OLS available for
comparison, report condition support, and interpret ridge contrasts as
regularized associations rather than evidence of reward-specific activity.

Compute reference and fold contrasts from their corresponding fits. Report
support and whether the contrast is estimable from the unpenalized design,
using a small SVD/row-space check. Penalization must not turn an absent
condition into evidence of zero selectivity. Preserve numeric solver outputs
for audit, but display unsupported effects/contrasts as unavailable with reasons.

Raw contrasts have Hz or PC-score units. They must not be calculated by
subtracting the existing standardized coefficients, because the two outcome
columns can have different SDs. The standardized-coefficient display remains
`beta * SD(X) / SD(Y)` on reference-fit rows; raw contrast columns retain
explicit raw-unit labels when this display is selected. No p-values or
significance labels are produced.

The outcome slopes describe post-choice associations conditional on the
included context/choice terms. The model does not include separate baseline
intercepts for each future outcome category, so these contrasts must not be
labeled a formal within-condition pre/post difference-in-differences estimate.
Outcome-related anticipatory activity or unmodeled movement can still affect
interpretation. Preserve the pre-choice activity in inspection rather than
forcing it to zero or asserting a sensory reward mechanism.

## 5. Preparation and unit/PC data contracts

Use the existing session metadata, aligned spikes, augmented trial CSV, and
feature-parameter JSON. Validate source/augmented trial correspondence on all
fields used by the joint model, including correctness and reward. Read small
tables first and reject ambiguous joins before loading neural data.

For each region:

1. Apply the existing baseline behavioral exclusions.
2. Require trusted IRIG recording coverage of choice-1 through choice+1.
3. Load the existing selected good/mua unit population and bin once.
4. Retain the full unit count array and original probe/cluster IDs.
5. Build bin rows from choice positions/times and bin edges, and assign five
   GroupKFold blocks on the regional baseline-covered trial pool. Neither
   operation depends on PCA or its success.
6. Prepare an optional full descriptive PCA from those two-second windows,
   using existing unit standardization and unwhitened projection. Fit all
   numerical nonzero PCs when PCA is available.
7. Apply joint predictor eligibility afterward; use the same rows and fold IDs
   for unit/PC fits and all encoding/penalty comparisons.

Counts have axes `(window, bin, unit)`, unit IDs `(unit,)`, relative edges
`(bin+1,)`, and PC scores `(window, bin, original_PC)`. Bin rows retain original
trial/row/block identity, UTC and elapsed bin edges, and trial-major/bin-minor
ordering. Slopes have axes `(predictor,response)` and predictions
`(eligible_bin_row,response)`. Document all transformations and units.

Convert unit counts to Hz with `counts / bin_width_s` when obtaining the unit
response matrix. Save counts once, not both a duplicate complete rate tensor
and counts. Retain selected units that are constant or silent; they are still
unit responses even though PCA excludes constant inputs. Constant responses
have valid MSE and raw fits, unavailable zero-SST R-squared, and unavailable
standardized coefficients.

Unit availability must be independent of PC availability. If a region has
usable covered unit activity but no variable PCA input, retain its unit
analysis and report its PC representation unavailable. No missing coverage
is ever filled with zeros.

Make the existing PreparedRegion record an activity-first data record. Its
primary fields are region/probe identity, unit_counts(W,B,U), unit_ids(U,),
choice_positions(W,), bin_edges_s(B+1,), bin_rows(W*B,C), and
window_fold_ids(W,). An optional pca field holds the existing regional PCA
result, with original nonzero pc_ranks and an explicit unavailable reason
when absent. Do not retain a mandatory PCA-shaped timeline field or create
dummy PC objects for unit-only activity.

The primary activity fields own window/bin geometry. Build bin rows and
validate unit observations from those fields. Optional PCA geometry must
agree with the primary activity, but PC scores must never be used to infer
the number of unit windows/bins. Save/load the common activity section
independently; only read/validate PCA arrays when a PCA section is present.

The current preparation saves PCs and their transform but discards unit counts.
Therefore old PC archives cannot supply the requested raw units. Also, cropping
scores from a PCA trained on +/-2 seconds does not satisfy the new PCA scope.
Prepare a new compatible +/-1-second basis; preserve old products and never
claim that PC12 from the two bases is the same axis.

## 6. Local architecture and reuse

Keep the current shallow package and function-oriented design. Extend the
existing small data records; do not add an estimator registry, formula parser,
general workflow framework, or external dependency.

| Owner | Planned changes |
| --- | --- |
| `encoding/configuration.py` | Validate the current analysis version and response-representation selection; derive the fixed two-second window and trial-fixed timing. Remove old model/timing/fit-window selection controls and reject obsolete configs before numerical work. |
| `encoding/predictors.py` | Add pure outcome classification and joint-design construction. Return ordered X, row validity/reasons, column names, and condition/support counts. Reuse native context encoding; remove unused context-only/choice-updated builders after a call-site audit. |
| `encoding/preparation.py` | Make PreparedRegion activity-first with counts/IDs/positions/edges/rows/folds and optional PCA. Refactor build_bin_rows to take explicit choice positions and bin edges instead of a RegionalTimelinePCA. Build the two-second pool once. |
| `population/session_timeline.py` | Add narrowly scoped optional window arguments to bin-edge, coverage, count-binning, and PCA entry points that currently hardcode +/-2 seconds. Keep existing four-second defaults and propagate explicit edges. Reuse the current Pynapple binning and PCA routines. |
| `population/session_timeline_io.py` | Allow source-configuration metadata to describe the explicit window without changing existing viewer defaults or weakening old saved-timeline checks. |
| `encoding/modeling.py` | Reuse multi-response ridge/OLS and CV; accept native X/Y and explicit feature penalties, with small support/contrast computations. Remove the old one/two-column assumption; no legacy/joint model flag enters numerical functions. |
| `encoding/pipeline.py` | Assemble one joint X per region/encoding, then fit units and PCs against matched rows/folds. Save one region/representation/encoding/estimator/penalty cell at a time. |
| `encoding/results.py` | Save/load the current activity-first schema with optional PC products, response IDs/kind/units, penalties, support and contrasts. Resolve unit observations without PCA. Reject old schema versions clearly; no compatibility adapters or migrations. |
| `encoding/plotting.py`, `webapp/encoding_views.py` | Add units/PC selection, correct labels/scales, joint predictor tracks, raw contrasts, support messages, and optional contrast ordering. Consume saved data only. |
| `encoding/run_session.py`, `run_batch.py` | Preserve CLI shape, dry-run/new/status/resume, session workers, logging, and immutable products; report planned representations and actual timing/memory/bytes. |

Proposed small public additions are `classify_choice_outcomes`,
`build_joint_predictors`, `response_activity`, and `compute_reward_contrasts`,
plus the revised data contract of `build_bin_rows`.
They receive the necessary tables/arrays, not a whole live analysis session.
Their proposed contracts, finalized in tests before implementation, are:

| Function | Inputs | Outputs |
| --- | --- | --- |
| `classify_choice_outcomes` | Aligned action/reward/correct vectors (T,), dimensionless source codes | Outcome codes, validity and explicit unknown reasons (T,) |
| `build_joint_predictors` | Complete trial table, original regional row positions (W,), relative edges (B+1,) in seconds, context encoding | Native X(W*B,P), eligible-row mask/reasons, ordered column names and condition diagnostics |
| `build_bin_rows` | Trial/choice identity tables, choice_positions(W,), bin_edges_s(B+1,), UTC time origin, and window_fold_ids(W,); no PCA object | Bin-row table (W*B,C) with original identities, time edges and folds |
| `response_activity` | One prepared regional record, units/PCs selection | Activity (W,B,R), ordered response IDs (R,), physical unit label |
| `compute_reward_contrasts` | One LinearFit, its exact native X(N,P) and ordered column names | Raw contrast values (2,R), support/estimability flags and reasons, contrast names |

No behavior-heavy classes are needed. Keep shared unit/bin geometry separate
from optional PCA fields. Unit result cells use `response_ids` and a response
kind; do not disguise cluster IDs as `pc_ranks`. Unit observation validation,
plotting and saving must work when pca is absent. Keep the representation
selection at the activity-access boundary; the solver receives ordinary X/Y.

For unit heatmaps, reuse the existing actual-time bin/gap geometry. Do not pass
unit indices through PC-specific validation or attach explained-variance hover
to units. Extract/share only the small time-geometry helper if needed; keep
the existing public PC heatmap behavior and return contract intact.

## 7. One current model, configuration and versioning

The user does not require the old model to persist. Replace the context-only
encoding model rather than maintaining two execution profiles. Keep the CLI
command entry points, but use one current configuration and one current saved
schema. There is no model_kind dispatch, implicit legacy default, legacy
reader, or old scientific-mode control in the revised UI.

The revised example requires `analysis_version: "joint-task-encoding-v2"` and
defaults `response_representations` to `["units", "pcs"]`. Window `[-1,1)`,
whole-window fitting, choice alignment and trial-fixed context/choice are
scientific constants saved explicitly in provenance, not selectable controls.
Retain supported bin widths, context encodings, estimators, alpha and penalty
names. Remove the old windows/timing_modes options from the example and
configuration contract. Missing/old version fields or obsolete options produce
an actionable error pointing to the new example; do not silently reinterpret
an old configuration.

Use one new schema/analysis version. Old configurations and archives are not
executed, resumed, migrated or loaded as current products. Preserve their
files on disk. Run discovery skips old-version metadata so older directories
cannot block a new run or appear as selectable current results. Directly
requesting an old run returns an unsupported-version message directing the
user to create a new run with the revised config. Current-version code/input
identity checks still determine whether current work can be resumed.

This retirement applies to the old encoding model, not unrelated timeline
analyses. Preserve public four-second defaults used by the existing session
timeline and other callers; pass explicit two-second windows from this
analysis. Audit uses of changed encoding functions and update their dependent
code/tests rather than retaining compatibility aliases or duplicating helpers.

Preparation identity includes the actual window, bin edges, baseline/coverage
policy, unit selection, source data including outcome fields, PCA policy,
requested prepared representations, and numerical code/runtime identity.
Model identity additionally includes the joint design version, response kind,
coding, fit scope, penalties, CV policy and contrast definitions. Reuse only
compatible current-version preparation containing the required arrays;
+/-2-second PCA reuse is rejected for this model. No migration or rewrites of
old NPZs and no proliferation of version-handling branches through scientific
helpers; version validation belongs at config/file-loading boundaries.

## 8. CV, saving, and incomplete data

Retain deterministic five-fold grouping by `cur_block` at the window level.
All bins from one source trial keep its fold. No random bin-level splitting,
PCA refitting inside folds, trial averaging, or fold-average primary R-squared.
Pool held-out MSE and R-squared across eligible bin rows, preserving negatives.
For units MSE is Hz squared; PC MSE is squared PC-score units. Unit CV does
not depend on a PCA, while PC CV remains conditional on descriptive session
PCA that includes test-block neural activity; label each scope correctly.

Save predictor support and condition counts for reference, train and test
rows, plus fold contrast values/availability. A fold lacking a condition can
still generate ridge predictions and meaningful aggregate prediction scores;
its unsupported condition attribution must be disclosed separately. Fewer
than five usable blocks retains the reference fit and explicit unavailable CV.

Keep named-array NPZs with a dictionary called meta, ordinary JSON progress,
and one-cell-at-a-time publication. Shared preparation owns counts/scores;
do not repeat observed activity in every cell. Include response identity,
units/axes, actual windows, design/penalties, exclusions, fold assignments,
contrast reasons, generator/version, parameters and randomness provenance.

Preserve interruption recovery: a fully published cell is adoptable if the
progress index was not updated, and completed compatible cells are not refit.
Status and saved inspection require no live source data. Mutable progress,
logs, summaries and performance reports remain separate from immutable
scientific products. Keep outputs session-local and outside the repository.

## 9. Saved viewer and exports

Add a Units/PCs selector and select original probe:cluster IDs or original PC
ranks accordingly. Coefficient and neural rows always share the same selected
responses and order. Sort by a selected coefficient or one of the two raw
reward contrasts; unsupported values sort last and are visibly unavailable.

Show fixed context/choice tracks, all six outcome pulses, and included-bin
status alongside activity. Preserve elapsed-session time, real gaps, pan/zoom,
bounded loading, original response identity, selected-response prediction
traces, and categorical coefficient axes. No control reloads raw spikes or fits.

Units use a nonnegative raw-Hz heatmap with fixed regional limits from the
saved full pool; PCs keep the existing fixed signed PC-score scale. A
constant-zero unit region needs a safe display range without inventing
nonzero activity. The coefficient scaling toggle never rescales neural traces.
The table identifies contrast units and unsupported conditions explicitly.

Export light-mode opaque PNG coefficient summaries and bounded timelines/
prediction examples for each available region/representation. Captions state
the +/-1-second activity scope, choice-aligned outcome convention, model,
response identity, coefficient scale, prediction scope, and contrast meaning.
No significance statements or positive-coefficient responsiveness labels.

## 10. Tests to write before implementation

Follow RED -> tests-only commit -> GREEN -> review/refactor/source commit for
each phase. Do not fix an incorrect oracle by changing a requirement silently.

| Phase | Tests committed first | Completion gate |
| --- | --- | --- |
| J1: configuration and joint predictors | Current-version example/default units+PCs accepted; old/missing versions and obsolete scientific options rejected; exact +/-1 edges at all widths; fixed context/current choice through switch and terminal trials; all six categories and chosen-port side on incorrect trials; zero pre-outcome pulses; unknown outcomes excluded with reasons; half-open zero/+1 boundaries. | One current configuration and pure design builders, explicit column order and units. |
| J2: two-second activity, units and PCA | Spikes outside +/-1 have no influence; exact two-second coverage can include a trial rejected by four-second coverage; regional truncation; unit order/IDs and count-to-Hz conversion; rows/folds derive from count geometry with no PCA; constants/silent units retained for units; only variable inputs for PCA; absent PCA does not change unit activity; incompatible old PCA rejected; unrelated timeline defaults unchanged. | Activity-first PreparedRegion plus optional PCA with correct metadata. |
| J3: joint fitting and contrasts | Independent weighted-ridge normal-equation/augmented-system oracle; OLS minimum-norm oracle; matched-context equivalence with unchanged choice/outcome penalties; signed/indicator prediction agreement under matched penalties; whole/before/after rank example; no redundant post indicator; missing conditions/aliasing support; raw contrasts versus incorrectly subtracted standardized slopes; generic equal-response OLS zero contrasts and imbalanced ridge contrasts checked against the penalized oracle; constant-response metrics. | Correct joint regression, diagnostics and estimator-specific contrast expectations. |
| J4: pipeline and persistence | Identical rows/folds across representations/encodings; counts saved once; response-kind/ID/shape mismatch rejection; one current schema; old configs/archives rejected and old run discovery skipped without modifying files; model/cache invalidation for changed window/outcomes; source-free current-version resume; orphan-cell recovery; unit save/load/fit without any PCA section or object; correct unique-cell counts and dry-run without neural loads. | Offline current-version complete/resumable products with auditable identities and support. |
| J5: plots and saved UI | Nonmonotonic unit IDs and PC12 ordering; correct Hz/PC labels and hover; contrast sorting/unsupported display; all predictor tracks; exact two-second neural windows and real gaps; categorical axis protection and pan persistence; no fits/live source loads on controls; unit display/export when PCA is absent; old-version runs omitted from selection; PNG captions/scale/scope. | Synthetic current-model saved viewer and exports pass tests and browser/visual review. |
| J6: end-to-end synthetic validation | Seeded populations containing a left-reward unit, right-reward unit, omission/incorrect units, context/choice units, a generic post-choice unit, silent/constant controls; enough blocks/conditions for CV; independent expected unit coefficients and PC effects from the fixed transform; balanced and imbalanced condition variants with OLS zero contrasts and analytically predicted ridge shrinkage/contrasts; missing-condition and PCA-unavailable variants. | Inspect unit/PC figures, record metrics/performance, update usage docs and execution record. |

J6 should verify both exact/noiseless effect recovery with OLS and analytically
predicted ridge shrinkage; a noisy synthetic example demonstrates robustness
without inventing a significance threshold. Save the synthetic seed. Verify
that a post-choice association is not mislabeled as proof of reward causality.
Use explicit column-name mappings in independent contrast oracles. Require
zero equal-response OLS contrasts only when the model is correctly specified
and the contrasts are estimable. For ridge, compare slopes and contrasts
with an independent penalized-system solution in both reference and fold
fits; do not force a zero result or alter penalties/category weights to make
the generic-response example produce zero. Update tests of the retired model
to the explicitly changed requirements, retaining applicable numeric tests
and tests protecting unrelated timeline callers.

Run focused encoding tests and directly affected timeline/metadata/webapp tests
at each relevant gate, then one broad neural regression pass after integration.
Classify unrelated existing failures rather than modifying their source/tests.
Use uv for Python and the existing bundled Node for browser-renderer tests.
Do not install dependencies to make unrelated collections pass.

## 11. Performance and dependency discipline

Reuse NumPy, pandas, SciPy, scikit-learn, Pynapple, Plotly, Matplotlib,
Streamlit and the standard library. Verify library APIs in installed source
or official documentation. Do not modify external dependencies.

Bin each region once, fit PCA once, and reuse fold/design arrays. Store one
count tensor and one PC tensor, and process one result cell at a time. Do not
save every fold's predictions for its training rows or retain all model cells
in RAM. Reuse signed penalty aliases. Load only the selected cell for the UI
and keep bounded neural display available, especially at 20 ms.

For W windows, B bins, U units and K PCs, float64 shared activity alone is
approximately `8*W*B*(U+K)` bytes, before tables/transforms/working copies.
One response cell's reference and held-out predictions add approximately
`16*N*R` bytes. At the same bin width the revised activity window has half
the old bins, but adding raw units and unit predictions can still increase RAM.
Use these dimensions in dry-run estimates only when counts are known; measure
actual peak process RSS, timings, payload bytes and compressed output sizes.

Keep session-level batch parallelism, CPU-capped workers and existing nested
thread limits. Do not add within-session parallelism or major optimization.
If measured RAM makes default batch concurrency unsuitable, report it before
launching a real batch. Synthetic timings do not establish CT026 performance.

## 12. Execution and review gates

1. Implementation approval received, 2026-11-13. The whole-window-only joint
   fitting scope is confirmed by the user.
2. Implement J1-J6 in tests-first phases. Preserve the initial dirty worktree,
   unrelated user changes, unrelated public defaults and historical files.
3. Report synthetic scientific behavior, browser verification, regression
   results and measured performance. Update the ridge spec/example/README to
   match the approved revision and append dated RED/GREEN commit records.
4. CT026 2026-08-04 is the previously identified candidate for one-session
   validation. Prepare the exact new config/dry-run for review; real numerical
   execution follows explicit authorization for this revised analysis.
5. Inspect context/choice tracks, all outcome conditions, reward contrasts,
   unit identities and PC identities on that session before any wider batch.

## 13. Execution record

2026-11-13: User authorized implementation after plan corrections. Initial
encoding and six directly related timeline test files passed 345 tests in
27.82 seconds. Unrelated worktree changes were recorded and are preserved.
J1 tests first reproduced the missing build_joint_predictors import. No
experimental data regeneration, numerical run, cluster job or batch is authorized.

J1: RED tests `9fe49e7`; GREEN source/example `46ddf33`, 102 focused tests passed.
J2: RED tests `949a542`; GREEN source `a1019dc`, 45 preparation and existing
timeline checks passed. Unit activity survives absent variable PCA inputs.
J3: RED tests `8e2286b` (14 failures; 31 unaffected checks passed), with retired
metadata assertion correction `3ace87b`; GREEN source `f7ca31e`, all 45 numeric
tests passed, including feature penalties and imbalanced ridge contrast oracles.
J4: RED tests `febd6a5`; core persistence/pipeline passes 47 tests. Runner
metadata/dry-run checks pass 11 tests; real export integration awaits J5.

J4/J5 integration: core source `bc41b38`; J5 RED tests `233dae1`, test-main
isolation correction `8fc9193`, GREEN viewer/runner/export source `5757c83`.
All 65 focused plot/UI/runner/batch and existing timeline plotting checks passed.

J6: generated-spike tests `911ebfe` validate both balanced and imbalanced
20-cell runs against independent unit-Hz/PCA-projection/penalty oracles, all
reference and fold fits, missing conditions and absent PCA. Two added source
correspondence checks reproduced missing block/manual-flag validation. Five
archive-integrity checks in `cce524f` reproduced acceptance of invalid support
counts, negative penalties and invalid declared contrasts. GREEN source and
readability review `964ed32`. Browser-discovered colorbar overlap was reproduced
by RED test `cc6955e` and corrected in the same GREEN source commit.

Final focused integration: 335 tests passed in 49.42 seconds, including the
bundled-Node frontend checks. Real browser inspection verified units, PC12,
raw/standardized display, raw contrast ordering, coupled axes, categorical
columns and pan persistence. Both unit and PC PNGs were inspected. Measured
performance, scientific limitations and broad-regression classification are in
[the validation report](single_unit_joint_validation_20261113.md).

CT026 2026-08-04 read-only dry-run passed: 716 baseline trials, 20 planned cells,
no input blockers or context-event diagnostics. A review config using verified
existing input paths is in `.pytest_cache/ct026_joint_encoding_review.json`.
The existing session config/results are unchanged. No experimental neural
arrays were loaded or fitted; the next gate remains the authorized real-session
run and user inspection, followed by any authorized wider batch.
