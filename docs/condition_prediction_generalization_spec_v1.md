# Condition Generalization and Mixed-Outcome Decoding on the Session Timeline

Status: scientific decisions confirmed, document revision 4, 2026-11-09.
Full/Before/After fits, actual reward as the third mixed target, and training/
held-out outcome/laterality summaries are implemented. Synthetic integration
(408 tests) and saved-only browser checks are complete; the linked plan records
the evidence. Original Full-window v1/v2 products remain inspectable under their
original seven-/two-target contracts. No new experimental preparation was run;
CT026 validation and user scientific/visual acceptance remain separate gates.
Keep this existing filename and the revision-3 baseline evidence in the plan.

Implementation plan and validation record:
[condition_prediction_generalization_implementation_plan.md](condition_prediction_generalization_implementation_plan.md).

This document records the user's confirmed decisions for session prediction
analyses. It is not an implementation plan. These decisions take
precedence over suggestions made earlier in the discussion. Section 11 records
the five-fold policy confirmed after the user's requested explanation.

Related documents:

- [Existing condition generalization specification](task_variable_condition_generalization_spec_v1.md).
- [Session timeline and PCA specification](session_timeline_pca_spec_v1.md).
- [Timeline implementation and refinements](session_timeline_pca_implementation_plan.md).
- [Optional treadmill speed track](session_timeline_speed_implementation_plan.md).
- [Software design guidance](SoftwareDesign.md).

## 1. Goal

Provide two explicitly named analyses using the same session timeline:

- **Correct-rewarded generalization:** train the existing seven task-variable
  readouts on correct-rewarded trials and apply them across choice conditions.
- **Mixed-outcome decoding:** train separate state, action and actual-reward
  classifiers on correct-rewarded, correct-omission and incorrect training
  trials. This measures decoding of recorded state, action and reward across
  outcomes; it is a separate analysis from the correct-rewarded reference.

Display predictions during individual choice windows throughout the session.
Show choices, selected behavioral events, and predictions on a shared
elapsed-session-time axis, with actual trial-level targets overlaid.

This expands the condition generalization idea from aggregate decoding
performance to the actual predicted values for individual trials and bins.
It uses the existing session-wide regional PCA bases and the existing
timeline's event and navigation behavior.

Each analysis supports Full, Before and After supervised windows defined in
Section 2; each model is scored in its own training window and displayed
across the same complete prediction window. Existing Full results remain
available. The readouts remain offline analyses: Full/After fitting includes
post-choice activity, and even Before uses session-wide preprocessing and
block folds that can include later blocks. Do not describe them as a fully
prospective or online pipeline.

## 2. Confirmed decisions

| Item | User decision |
| --- | --- |
| Analysis modes | Correct-rewarded generalization and mixed-outcome decoding; save and inspect separately. |
| Generalization training / targets | Canonical `correct_rewarded`; all seven targets in Section 3. |
| Mixed-outcome training / targets | Union of canonical `correct_rewarded`, `omission`, and `incorrect`; `current_state`, `current_action`, and new `current_reward`. Canonical `omission` means correct omission. |
| Sampling / balancing | All eligible training trials; equal total weight per trial. No subsampling, class/outcome balancing, or additional training fraction. |
| Training / scoring windows | Three independent fits: Full (`full`) `[-1, +1)` (existing default), Before (`before`) `[-1, 0)`, After (`after`) `[0, +1)` seconds relative to `choice_time`. Score each fit over its own selected interval. |
| Prediction interval | The complete `[-2, +2)` second choice window. |
| Intertrial intervals | Leave gaps; do not predict outside choice windows. |
| Evaluation | One deterministic five-fold behavioral-block assignment shared across modes, windows, targets, and regions. |
| Outcome summaries | Both exact training trials and held-out trials, separately by outcome; retain per-fold training scores and unique pooled held-out scores. |
| Left / right summaries | Both actual-choice and recorded-state/context groupings, for each split/outcome and each region. |
| Reward analysis | Actual trial-i `reward` flag only; additional mixed-outcome target, not a separate analysis mode or model-expectation target. |
| PCA fitting scope | Session-wide PCs, including conditions beyond correct-rewarded. |
| Predictor PCs | All available saved session PCs for each region. |
| Regional outputs | Separate PFC and HPC models and tracks. |
| Default bin width | 500 ms; 100 ms must also be available. |
| Binary model | L2-penalized logistic regression. |
| Continuous model | Ridge regression. |
| Penalty selection | Fixed penalties; keep preparation simple. |
| Penalty settings | Unchanged constants: ridge `alpha=1.0`, logistic `C=1.0`, unpenalized intercepts; no user configuration knobs. |
| Decoder feature scale | Existing unwhitened PC scores, without another PC standardization. |
| Actual target | Overlay the actual trial-i target across each choice window. |
| Optional neural view | Original PC activity or signed weighted PC contributions. |
| Optional PC ordering | Original rank or contribution strength; fixed across session navigation. |
| Entirely missing continuous target | Keep the target unavailable and continue other targets; malformed present values remain input errors. |
| Condition hover | Save and show all applicable canonical condition memberships, including overlaps. |
| Preparation | Provide an explicit Prepare control. |
| Initial webapp location | Controls within Session timeline and PCA; revisit a separate view if crowded. |
| Validation session | CT026 2026-08-04. |

The regional decision excludes a combined PFC+HPC model from this revision.
Use all available session PCs as predictors. The number of visible heatmap
rows remains independent of the model's predictor count.

The name "full window" in this discussion means the two-second training
interval `[-1, +1)`. It must not silently reuse the existing pooled decoder's
`full` setting, which means `[-2, +2)`. Save exact numeric bounds in the new
configuration and results.

All bounds are relative to choice, not trial start or reward delivery. Windows
are half open: time zero belongs to After and not Before. Full is an
independent pooled-bin fit, not an average or combination of the two half-window
models. No before-to-after or after-to-before scoring matrix is included;
the full prediction trace remains inspectable outside the matched scoring window.

## 3. Target sets by analysis mode

Reuse current definitions and add the actual-reward target below. Each target
refers to trial i; do not shift labels to a previous or next trial.

| Target identifier | Source column | Family | Displayed prediction |
| --- | --- | --- | --- |
| `current_state` | `state_int` | Binary | Probability of left state, class 1. |
| `current_action` | `action` | Binary | Probability of left action, class 1. |
| `current_reward` (new, mixed only) | `reward` | Binary | Probability that trial i was rewarded, class 1. |
| `qlearning_relative_value` | `Qlearning_rel_value` | Continuous | Predicted relative value in source units. |
| `forgetting_q_relative_value` | `FQlearning_rel_value` | Continuous | Predicted relative value in source units. |
| `hmm_signed_belief` | `HMM_rel_value_logodds` | Continuous | Predicted HMM belief log-odds. |
| `hmm_decay_signed_belief` | `HMM_rel_value_logodds_decay` | Continuous | Predicted decayed HMM belief log-odds. |
| `relative_doubt` | `relative_doubt_index` | Continuous | Predicted relative doubt in source units. |

Correct-rewarded generalization retains its original seven-target order:
`current_state`, `current_action`, `qlearning_relative_value`,
`forgetting_q_relative_value`, `hmm_signed_belief`, `hmm_decay_signed_belief`,
`relative_doubt`. All seven gain Before and After variants.
New mixed preparations use `current_state`, `current_action`, `current_reward`,
in that order, for all three windows. Existing two-target mixed archives keep
their original meaning and remain inspectable; see Section 8.
Do not train the five continuous targets on mixed outcomes or duplicate their
existing fits in the mixed-outcome product. Target identifiers, source
columns, and binary orientation do not change between modes.

Registering reward in the shared target definitions must preserve the old
decoder's default target set, both when targets are omitted from a config and
when the session setup command generates a new config. This expansion must
not silently add reward to those existing analysis defaults.

### Actual reward and expectation/outcome interpretation

Use the recorded trial-i `reward` flag directly: 1 = rewarded, 0 = unrewarded.
Do not substitute correctness, infer reward from state/action or condition
membership, or reuse a previous-trial reward target. Preserve existing manual
reward and no-choice exclusions. Missing reward values remain missing, not
zero. For an otherwise eligible mixed trial, they invalidate that target's
training/scoring rows; independent canonical outcome eligibility still applies
to every target. Missing required columns or malformed present binary values
remain input errors.
An unfit reward target must not prevent valid state/action readouts from running.

Actual reward is added only to mixed-outcome decoding. Correct-rewarded-only
training would have a single reward class and is not a requested reward model.
Before predicts the recorded eventual reward outcome from pre-choice activity;
After decodes reward-related activity after choice. Neither is a ground-truth
measure of subjective expectation. After is not automatically post-delivery:
retain choice alignment even when reward delivery falls later. No new model
expectation, value-model fit, or reward-centered alignment is included.

Use the existing augmented behavioral table and feature provenance for model
values. Do not refit the behavioral models or invent new units, normalization,
clipping, or target definitions as part of prediction preparation.

Construct and align targets on the original chronological table before
filtering the training condition. Preserve original row positions and trial
IDs when connecting this table to saved timeline choices.

If a requested continuous source column exists but has no usable values,
retain its target coordinate and mark its models/predictions unavailable;
continue processing the other targets. Preserve NaN actual values and false
target-validity masks, with an explicit reason in the saved result and UI.
Missing required columns and malformed present values still raise input
errors. This policy is separate from a finite constant target, which ridge
can fit normally.

### State and action interpretation

State and action normally agree on correct-rewarded trials. If their eligible
training rows, labels, and settings are identical, the two binary models
should produce identical predictions. Their actual labels can differ on
other trials. The specification retains both requested targets, but their
names do not establish independently identified state and action readouts.
Report this limitation rather than forcing the models to differ. The
correct-rewarded state readout projects activity onto a reference learned
where recorded state and action agree. It can motivate hypotheses about the
mouse's inferred state, but does not independently establish an internal
belief representation or distinguish state from action coding.

Mixed-outcome state decoding predicts recorded task state; mixed-outcome
action decoding predicts actual choice. Both classifiers use the same
training-condition pool, with their own target-validity masks. Incorrect
trials allow state and action labels to differ. Correct omissions alone do
not remove their agreement on correct trials. Different labels allow the
models to differ but do not guarantee different fitted outputs or establish
causal separation of neural representations. Preserve both original
generalization outputs rather than replacing or relabeling them.

## 4. Neural inputs and the fixed session PCA basis

Use separate session-wide PFC and HPC bases from the existing timeline
analysis. That analysis pools complete regional choice windows across
conditions, standardizes units once, and fits an unwhitened regional PCA.
Retain the saved axes, signs, unit selection, scaling, and score units.

The PCA fitting pool remains the timeline's complete `[-2, +2)` windows. Full,
Before and After restrict supervised readout fitting/scoring, not PCA fitting.
Reuse the same saved regional basis across all three windows at one bin width.
Do not refit PCA on correct-rewarded trials, per supervised window, per fold,
per viewport, or per target. Retain complete prediction-window eligibility:
Before/After do not introduce a new partial-window neural loading policy.

Use original saved scores for prediction. The existing optional display-only
PC z-scoring must not change model inputs, coefficients, predictions, or
contribution rankings.

Use one bin width consistently for the selected saved PCA basis, supervised
training, prediction, and time-bin placement. Training bins are pooled as
individual observations rather than averaged into one vector per trial.
The supported timeline widths are 500, 100, 50, and 20 ms. Default to 500 ms
and keep 100 ms available as an ordinary selectable option. Bin counts per
complete choice window are:

| Bin width | Full training/scoring | Before | After | Prediction |
| --- | --- | --- | --- | --- |
| 500 ms | 4 | 2 | 2 | 8 |
| 100 ms | 20 | 10 | 10 | 40 |
| 50 ms | 40 | 20 | 20 | 80 |
| 20 ms | 100 | 50 | 50 | 200 |

Every selected interval must match exact saved bin edges. Changing width
selects the matching session PCA configuration rather than reusing a basis
from a different width.

For a region with W complete windows, B bins per prediction window, and K
selected session PCs, the feature tensor has shape `(W, B, K)`. Its bins
remain unsmoothed measurements at their actual choice-relative positions.
Incomplete regional windows remain unavailable; do not zero-pad missing
activity. Preserve the other region's usable results independently.

## 5. Training and held-out predictions

Use behavioral blocks as the cross-validation groups. All bins from a trial
and all trials from one block must remain in the same fold. Establish the
group assignment before selecting either mode's training rows. For the same
source table, both modes and all three supervised windows use the identical
map. Every prediction must identify its mode, selected window, model and
held-out block fold. Adding the reward target does not redraw the split.

Select training-condition membership from the existing canonical masks:

```text
correct_rewarded_generalization: correct_rewarded
mixed_outcome_decoding: correct_rewarded OR omission OR incorrect
```

Intersect this membership with baseline validity, regional complete-window
coverage, target validity, and the other four folds. Preserve existing manual
reward/no-choice exclusions; do not substitute the broad canonical `all` mask
or count overlapping `switch`/`stay` memberships as additional observations.
Use every eligible training trial once. A fold need not contain all three
outcomes to fit, but its binary training labels must contain both classes.
Do not rebalance outcomes, resample trials, or select another training
fraction. The held-out fold remains entirely excluded even when an outcome
is absent from the other folds.

For each selected mode/window, target, region and outer fold:

1. Select eligible trials for the chosen analysis mode from the other folds.
2. Select the bins fully inside the chosen Full, Before or After interval.
3. Stack `(training trial, training bin, PC)` into
   `(training trial * training bin, PC)` in trial-major order.
4. Repeat the trial-i target for its selected bins. Follow the existing pooled
   convention of row weights `1 / training_bin_count`, so each training trial
   has total weight one. Record this policy with the penalty settings.
   At 500 ms the row weight is 1/4 for Full and 1/2 for either half-window.
   Using fewer bins must not halve a trial's total weight.
5. Fit one intercept and PC-weight vector using the selected L2 model.
6. Apply that model to every valid bin in the `[-2, +2)` windows of the
   held-out blocks, across conditions.
7. Save individual predictions rather than reducing them to a fold score.

The user selected fixed penalties. Both strengths remain constants,
not configurable defaults; expose no JSON, CLI, or GUI penalty overrides.
There is no inner hyperparameter search. Use ridge `alpha=1.0` and logistic
`C=1.0`, with unpenalized intercepts, original PC feature scales, and native
continuous target units.
The implementation plan records the weighted loss conventions; equal numeric
values for logistic and ridge parameters do not imply equal penalties.

Mixed training changes sample size as well as outcome composition. With
summed logistic loss, equal total weight per trial, and fixed `C=1`, more
training trials reduce regularization relative to the average loss. A
comparison of the modes therefore does not isolate outcome composition.
Keep the confirmed weights and penalty; do not rescale C, match sample sizes,
or add balancing to make this comparison appear controlled.

Within a given mode/target/region/fold, the three windows share trial and
complete-neural-window eligibility. Their selected bin counts differ, but
total trial weights remain one. Selecting a half-window therefore does not
by itself change the total trial weight or training-trial count. The new
reward target can have different label availability from state/action; keep
its factual counts rather than forcing equal samples across targets.

Each ordinary choice gets the prediction from its own held-out model. Do not
average models that trained on that choice into its displayed prediction.
Model changes at fold boundaries should be inspectable in hover or model
details; the trace is not generated by one session-wide regression fit.

Label availability is required to train and compare against the actual
target. Keep prediction availability distinct from scoring availability:
missing actual values or a single-class destination subset do not by
themselves prevent a valid model from producing predictions from available
neural features. The exact treatment of trials lacking usable block IDs
must be explicit; they cannot be silently called held out.

### Scope of the held-out claim

The PCA basis and its unit scaling use session data that include held-out
blocks. Only the supervised model fitting excludes those blocks. Label the
analysis as:

> Block-held-out prediction using a fixed session-wide PCA basis.

Do not claim that the full preprocessing-and-prediction pipeline was fitted
without held-out neural data. This is the user's chosen exploratory design.

## 6. Model outputs and contribution display

For continuous target q, region r, held-out fold f, and bin t, save:

```text
prediction(q, r, f, t) = intercept(q, r, f)
                       + sum_j weight(q, r, f, j) * PC_score(r, t, j)
```

The prediction and weighted contributions use the target's native units.
Keep signed, potentially out-of-range regression predictions; do not silently
clip them to the observed target range.

For binary targets, the same weighted sum is the logistic linear predictor:

```text
logit(t) = intercept + sum_j weight_j * PC_score_j(t)
probability_of_class_1(t) = sigmoid(logit(t))
```

Display probability on `[0, 1]`, with class 1 explicitly labeled as left
state, left action, or rewarded for the respective target. The reward axis
is `P(rewarded)`, never `P(left)`. Weighted PC contributions are additive
log-odds terms, not additive probability contributions. The continuous HMM
log-odds target remains a separate numerical regression target.

### Training, held-out performance, and laterality

For both modes and each supervised-window variant, summarize the selected
target with one prediction per trial: average its bin predictions over that
model's selected Full, Before or After interval. Training, held-out, outcome
and laterality summaries all use this same matched interval. Never score
the complete `[-2,+2)` trace or silently retain Full scoring for a half-window
fit. The rest of the displayed trace remains available for inspection.

For binary targets, average probabilities first, then classify as class 1
when the mean is at least 0.5: left for state/action, rewarded for reward.
Do not classify an averaged logit or count bins as independent trials.
Continuous means retain native target units.

Report binary accuracy and balanced accuracy (fractions displayed as
percentages); continuous R2 and RMSE (RMSE in target units). Empty groups have
zero trials and unavailable metrics, not zero performance. Single-class groups
retain ordinary accuracy but balanced accuracy is unavailable. Constant or
singleton continuous groups retain RMSE but R2 is unavailable; negative R2
values remain visible.

Training scores apply each fold's saved coefficients to that model's exact
saved training-choice mask. Present these per fold: the training sets overlap
and must not be pooled into an inflated independent trial count. Training
availability belongs to the model being examined: failure of a trial's own
held-out model does not exclude it from another valid model's training score.
Held-out scores use each choice's own saved held-out predictions, pooled with every
trial counted once per region. Require baseline/target validity, an assigned
fold, regional coverage and complete finite predictions in the scoring interval.
Scoring availability does not change prediction availability or model fitting.

Split and outcome are separate dimensions. For every model fold, report both
training and held-out success, with correct-rewarded, correct-omission and
incorrect breakdowns. Keep overall training-pool summaries and held-out
training-outcome/all-eligible-choice summaries. Also retain pooled held-out
outcome summaries with each trial counted once. Show trial counts, fit
availability, mode and selected interval alongside the metrics.

Training outcome rows are intersections with the exact saved training mask,
not every choice in the other four folds. Therefore correct-rewarded
generalization has no training omission/incorrect trials: report zero counts
and unavailable metrics there. Do not call predictions on untrained
other-fold outcomes "training accuracy". Mixed training can contain all three
outcomes, with target-specific validity exclusions. Outcome memberships do
not change which model made a held-out prediction.

Within each split/outcome, show both independent laterality groupings: actual
left/right choice and recorded left/right state/context. Retain these for
reward decoding too; left/right identifies trial groups, not reward classes.
Use saved trial-i labels, which can disagree on incorrect trials. Report both
sides' metrics/counts and left-minus-right differences separately for PFC/HPC.
Training laterality remains per fold; held-out laterality can be pooled once
per trial, with fold detail available. Missing side labels affect only that
grouping. Higher accuracy/R2 is better; lower RMSE is better. A laterality
difference alone does not establish a hemisphere effect: outcome composition,
class balance and behavior can also differ by side.
Keep the existing binary difference display in ordinary-accuracy percentage
points; show balanced accuracy for each side where defined. Continuous
differences retain R2 and native-unit RMSE. Counts are not performance metrics
to subtract into an accuracy gap.

Reward groups require explicit interpretation. Correct-rewarded membership
has reward class 1, and correct-omission membership has class 0. An incorrect
group may also be single class; always use its recorded reward labels.
Single-class outcome/side groups have ordinary accuracy and counts but no
balanced accuracy. A pooled reward score can look successful from class
imbalance, so retain balanced accuracy where both classes exist and expose
rewarded/unrewarded counts. No score is manufactured for an empty group.

Compute scores from saved models/PCA in a pure scoring module, with cached
small UI summaries for the selected target, saved fit window and scoring
version. Summaries do not refit models, read raw data, or add significance
tests or hemisphere metadata. They describe the entire saved session,
independent of viewport or displayed PC count. Fit/archive extensions for
the additional window and target contracts are specified separately below;
old archive bytes remain unchanged.

### Optional PC rows

Provide a display choice between:

- Original signed session-PC scores.
- Signed weighted contributions, `weight_j * PC_score_j(t)`.

Match each displayed score to its weight by the same original PC identity,
even after selection/reordering. Display-row position is not a coefficient
index.

Weighted rows use each trial's own held-out model. Together with that model's
intercept, their sum must reconstruct its continuous prediction or binary
linear predictor. Keep intercepts separately inspectable.

Offer original PC-rank order and contribution-strength order. Preserve the
original regional PC identities in row labels and hover after sorting.

For fold f, define a PC's contribution-strength measure using the supervised
training rows:

```text
strength(f, j) = abs(weight(f, j)) * SD_training(PC_score_j)
```

This is the standard deviation of its additive contribution. It accounts
for different PC score scales without renormalizing features or changing
the fit. Use population SD (`ddof=0`) and the same pooled training rows used
for that model.

Aggregate by the arithmetic mean of this strength across available fitted
folds for each target and region. Sort descending, using original PC rank
to break ties. Report the number of contributing folds and make unavailable
models explicit. This gives one ordering for the entire session, independent
of the viewport, selected trial, or visible PC count.

This ranking describes varying additive contributions, not unique predictive
importance, causal importance, or a fraction of explained output variance.
PCs can become correlated within a condition even though the full-session
basis is orthogonal. Contributions can reinforce or cancel one another.

## 7. Timeline display

Reuse the existing timeline's shared elapsed-second axis, actual choice
timestamps, event toggles, context shading, jump-to-trial controls, duration
controls, session navigator, and preserved pan/zoom behavior. Reuse optional
speed where available; it is not an input to the prediction models.

Provide an **Analysis** selector with **Correct-rewarded generalization** and
**Mixed-outcome decoding**. Filter saved runs and target options to that mode.
Add a **Training/scoring window** selection with **Full [-1,+1) s**,
**Before [-1,0) s**, and **After [0,+1) s**. Full remains the default.
Saved run selection and captions must identify the actual fitted window;
changing this selector cannot reinterpret an existing Full model as Before
or After. A new window is a separately prepared fit, not a display crop.
On first saved-only entry, initialize to a mode/window pair with saved
products if the default pair has none. Honor subsequent explicit selections,
including an unprepared mode/window without substituting another model.
When switching to an unprepared mode/window, keep the last valid saved
behavior/PCA visible and identify its source and bin width. Suppress prediction
tracks, target overlays and weighted contributions until the selected mode
and window have a saved result. Raw PC activity may remain visible. This state
must work with raw sources removed, without recomputation or raw-input fallback; showing
saved background activity does not make the selected analysis prepared.
Use the following visible binary output names in track titles and inspection:

| Analysis | State output | Action output | Reward output |
| --- | --- | --- | --- |
| Correct-rewarded generalization | State generalization | Action generalization | Not included |
| Mixed-outcome decoding | State decoding | Action decoding | Reward decoding |

Track titles must also identify the training population, for example
`PFC - State generalization (correct-rewarded training)` and
`PFC - State decoding (mixed-outcome training)`. Saved-run labels, hover,
model details, and summaries must identify the mode and supervised window.
For example, `PFC - Reward decoding (mixed-outcome training; Before [-1,0) s)`.
Explain the three
mixed training outcomes in the caption. Avoid the standalone name
"sessionwide state decoding": coverage remains restricted to choice windows.
Retain the continuous target labels in the generalization mode.
Actual reward overlays show the recorded trial outcome as a constant 0/1
label over the choice window; they are not reward-delivery event traces and
must not imply that reward had already occurred before choice.

Horizontal navigation belongs to the session and elapsed-time origin, not
the saved PCA configuration. Preserve the current time range when switching
modes, supervised windows or runs with different bin widths/regional PCA bases,
including 500-ms generalization to 100-ms mixed decoding. Keep PC controls and vertical
axes specific to the displayed basis/model. Explicit navigation still changes
the range; a different session/time origin or a necessary bounds clamp is a
separate coordinate change. Preserve the actual browser pan as well as the
server's navigation controls.

Bounded neural loading is the explicit exception: preserve the browser range
only when it fits inside the destination loaded interval. Otherwise reset to
the interval selected by the server's navigation controls and explain this
behavior in the bounded-loading caption. This also applies when switching to
20-ms data, where bounded loading is enabled by default. Keep the existing
bounded loading approach; do not add browser-to-Python pan callbacks.

For a selected target, show separate PFC and HPC prediction panels beneath
the behavioral tracks. Overlay the actual trial-i target on both panels,
constant within its choice window. Do not interpolate the actual target or
predictions across intertrial gaps. Missing actual targets and unavailable
predictions remain distinguishable.

Place optional regional PC activity/contribution heatmaps underneath the
prediction tracks. Their visibility, ordering, and displayed PC count are
display settings; they do not change which features the saved model used.
Keep the existing timeline's light appearance and readable labels.

For a reordered heatmap, use equally spaced display-row positions. Keep the
original PC numbers in tick labels and hover; do not use a nonmonotonic list
of original PC numbers as numeric row coordinates.

Hover should identify the target, region, trial, all applicable canonical
condition memberships, elapsed and choice-relative bin interval, held-out
fold, fitted/scoring interval, prediction and actual target when present.
Save the existing `all`,
`correct_rewarded`, `omission`, `incorrect`, `switch`, and `stay` masks during
preparation. These are overlapping memberships, not one exclusive outcome
label. Contribution hover also identifies the original PC, weight, and units.

Clearly label the selected Full, Before or After training/scoring interval.
Predictions in the remaining parts of `[-2, +2)` apply the frozen readout
outside its training interval. They are still held out by block, with the PCA-scope limitation
described in Section 5.

## 8. Preparation and saved results

Provide an explicit Prepare control for the selected mode and window.
Preparation fits models and saves predictions/provenance; display interactions
load saved products.
Reuse compatible saved preparation by default. Require an explicit rerun
action for a new result when an equivalent preparation already exists.
Preparing expanded mixed-outcome decoding fits its three classifiers per
region/fold: at most 30 fits with both regions per selected window, versus
70 for generalization. These are per-window counts, not a requirement to
prepare all three windows automatically.
Reuse matching saved PCA; do not rebuild spikes/PCA or refit the other mode
merely to add these readouts. An explicitly requested preparation can still
prepare a missing source PCA through the existing preparation path.

Keep a separate versioned identity for this analysis. Existing task-variable,
pooled condition-generalization, and session-PCA results retain their current
schemas and meanings. In particular, the existing pooled `full` training
window must not change.

Each saved run contains one analysis mode and one supervised window. Include
stable identifiers, exact training/scoring bounds, ordered target set and
canonical training-condition names in effective configuration/provenance and
scientific identity. Persist the selected
choice-aligned `training_condition_mask` separately from `correct_rewarded`.
Validate it against the saved canonical masks; never overwrite the meaning
of `correct_rewarded` to represent mixed membership.

Continue discovering/loading original v1/v2 results under their original
contracts: Full-window generalization with seven targets, and legacy v2
Full-window mixed decoding with state/action only. Identify their supervised
window as Full through their known versioned contract, not a filename guess.
Do not edit archives, rerun their fits, synthesize missing reward predictions,
or require raw data for saved inspection. Old two-target mixed products must
remain usable without pretending the reward extension is prepared.

The implementation plan must specify the new analysis/schema versions and
exact array/metadata validation before coding the new contracts. Require the
selected window/bounds and expanded target set explicitly in new products;
do not reinterpret missing new metadata as a half-window or a reward model.
Mode/window/target contracts cannot satisfy each other's preparation-reuse
requests. Include the scored window and scoring version in summary cache
identity. Viewing a legacy product keeps its Full scoring definition.

Save named arrays in a session-local NPZ with `meta` metadata, consistent
with repository rules. At minimum retain:

- Session, source PCA archive/configuration identity, target source identities,
  analysis version, code provenance, parameters, and random seed where used.
- Original row positions, trial IDs, choices, time origin, exact bin edges,
  target values, target validity, and training-condition membership.
- Canonical condition names and full membership masks aligned to those same
  choice rows, so portable hover never needs the raw behavioral table.
- Regional window mappings, original PC IDs, feature counts, model fold IDs,
  training masks/counts, coefficients, intercepts, and fit status/reasons.
- Individual per-bin predictions, availability, and binary output orientation.
- Fixed penalties, estimator/loss settings, row weighting, PCA scope, and
  contribution-strength definitions and summaries.
- Selected supervised-window identifier, exact training/scoring bounds and
  bin mask, plus unchanged prediction bounds. Reward class labels must remain
  separate from the choice/state labels used for laterality.

For example, regional predictions can use `(regional window, bin, target)`
and weights `(target, fold, regional PC)`. The implementation plan fixes exact
array names and metadata contracts for these axes before coding.

Use a dated run directory, summary, logs, and script snapshots according to
repository conventions. Record runtime and unavailable models. Preserve
existing outputs rather than overwriting them after configuration changes.
Do not allocate or save a redundant full
`(target, fold, trial, bin, PC)` contribution tensor; contributions can be
constructed from saved weights and matching PC scores for the displayed data.

## 9. Existing code to reuse and boundaries

| Existing code | Reuse / boundary |
| --- | --- |
| `src/neural_analysis/population/session_timeline.py` | Regional fixed session bases, choice identities, and `(window, bin, PC)` scores. |
| `src/neural_analysis/population/session_timeline_io.py` | Timeline source loading, source identities, and saved PCA products. |
| `src/neural_analysis/population/timeline_plotting.py` | Shared choices/events/context, real-time geometry, and original-PC ordering support. |
| `src/neural_analysis/webapp/timeline_view.py` and `timeline_chart.py` | Existing preparation/navigation conventions and persistent chart rendering. |
| `src/neural_analysis/task_decoding/config.py` and `targets.py` | Exact target names, class orientation, and chronological target construction. |
| `src/neural_analysis/task_decoding/conditions.py` | Canonical correct-rewarded, correct-omission, incorrect, and overlapping hover masks. |
| `src/neural_analysis/task_decoding/session_prediction/` | Existing preparation, numerical models, saved results, scoring, runners, and plotting; extend these modules for the new windows and reward target. |
| `src/neural_analysis/task_decoding/modeling.py` | Existing block-split primitives where their contracts fit; its current elastic-net estimator factory is not the selected new L2 model. |
| `src/neural_analysis/task_decoding/pooled/decoding.py` | Reference for pooled row ordering, repeated labels, and trial weights. Its training-only PCA and aggregate-only outputs are not this analysis's contracts. |

Keep numerical fitting, saved results, and plotting separate. Do not refactor
unrelated pipelines or introduce a general track/plugin framework. Reuse
installed NumPy, SciPy/scikit-learn, and Plotly capabilities after checking
their installed APIs. Follow the linked plan's module/file contracts.

## 10. Validation invariants

The implementation plan owns test commands/gates. They must establish these
scientific and display invariants before experimental preparation:

- Correct trial-i target alignment despite excluded or missing rows.
- Exact per-mode target sets and canonical training membership; Full/Before/After
  bin masks partition at zero and preserve the original Full interval.
- Matched training/scoring windows for every mode/target, while prediction
  arrays still cover every valid `[-2,+2)` bin. At 500 ms verify 4/2/2
  training/scoring bins, weights 1/4 or 1/2, and unchanged total trial weight.
- Shared source PCA, complete-window eligibility, and full-table folds across
  all window variants; no hidden PCA refitting or trial repartitioning.
- Mixed-outcome training includes incorrect and correct-omission trials without
  including manual/no-choice rows or double-counting switch/stay membership.
- Shared block exclusion for every training/prediction pair, including all
  bins from each trial; recorded fixed session-wide preprocessing scope.
- Prediction at every valid `[-2, +2)` bin with real intertrial gaps preserved.
- Separate regional availability and training failures.
- Binary probability orientation, continuous native units, and distinctions
  between missing targets, unavailable fits, and valid predictions.
- Trial-i reward target uses recorded `reward`, class 1 rewarded, with manual/
  no-choice exclusions and no shift, correctness substitution, or expectation
  model. Missing labels affect valid fits/scoring as specified in Section 3.
- New mixed target order is state/action/reward; generalization retains all
  seven targets. Each target's Before and After fits are separate models.
- Shared target registration preserves the old decoder's omitted-target and
  session-setup-generated defaults; neither silently acquires reward.
- Reconstruction of outputs from fold-specific weights and PC contributions.
- Contribution ranking that accounts for PC scale, retains original PC IDs,
  and remains fixed while navigating.
- Equal settings and labels yielding equal state/action predictions where
  their eligible training subsets coincide.
- A synthetic mixed-outcome example with different recoverable state/action
  labels; do not require unequal predictions for every possible dataset.
- Mode-specific identities/reuse, strict new-schema validation, original
  seven-target archive readability, and visible mode/training labels.
- Window-specific identities/reuse, exact saved training/scoring bounds and
  masks, and legacy Full seven-/two-target archive readability without edits.
- Training and held-out breakdowns by outcome and both lateralities use exact
  split masks; training omission/incorrect rows in generalization remain empty.
  Training folds overlap and are not pooled as independent observations;
  pooled held-out rows contain each eligible trial once per region/population.
- Probability means precede class thresholds, scoring ignores bins outside
  the selected interval, and reward axes/hover never label class 1 as left.
  Reward single-class outcome/side groups retain accuracy but not balanced
  accuracy; empty groups remain unavailable and retain zero counts.
- Switching to an unprepared mode in a saved-only session retains background
  behavior/PCA and suppresses all prediction-derived tracks without raw reads.
- Navigation survives switches between different saved PCA configurations;
  PC controls and vertical axes still follow the displayed basis/model.
- With bounded neural loading, retain an in-bounds browser range and reset an
  out-of-bounds range to the control-selected interval with an explanatory caption.
- Preparation reuse, saved-result round trips, and display changes that never
  refit a model or mutate source PCA products.

Use synthetic data first. The user designated CT026 2026-08-04 for the first
experimental validation. Preparation follows implementation approval and
successful synthetic checks; selecting the session does not authorize an
immediate run. The user inspects its results before any broader rollout.

## 11. Confirmed fold policy

The user confirmed the following common five-fold assignment and exceptional
case policies after the explanation of how the session trace is constructed:

- Assign whole behavioral blocks to five groups once, independently of target
  labels or region, using deterministic `GroupKFold(shuffle=False)`.
- For each held-out group, train on eligible trials for the selected mode
  from the other four groups and predict all eligible windows in that group.
- Use the same assignment for both modes, all windows/targets and both regions,
  while retaining target-specific label eligibility and regional neural coverage.
- Choices with missing block IDs have no held-out prediction; behavior and
  actual values can still be displayed.
- Fewer than five nonmissing blocks produces an explicit preparation error;
  individual unfit training subsets are marked unavailable without silently
  changing fold count, model, or training condition.

These groups need not be consecutive session intervals, and training can
include blocks later than a predicted choice. The trace therefore
evaluates offline block generalization, not future-only prediction.

## 12. Scope limits

This revision covers same-session regional PC readouts in two fixed modes:
seven-target correct-rewarded generalization and three-target mixed-outcome
decoding (state/action/actual reward), each with Full/Before/After matched
training/scoring and individual choice-window predictions. It does not require
continuous intertrial prediction, cross-session models, combined PFC+HPC
models, direct-unit decoders, arbitrary training-condition/target builders,
hyperparameter searches, new behavioral-model fits, significance testing,
cross-window scoring matrices, reward-centered alignment, model-expectation
targets, a third analysis mode, or new Slurm resource settings.

The revision-3 baseline and its recorded experimental scope are preserved.
The user subsequently authorized revision-4 implementation; source/test changes
and synthetic/browser checks are complete. That instruction did not include new
experimental preparation. The linked plan records the remaining validation and
user-acceptance gates.

## 13. Confirmed expansion decisions, 2026-11-09

The user confirmed all five clarifications:

1. Windows remain choice-aligned, and every summary uses its model's own
   training interval. The `[-2,+2)` prediction view is unchanged.
2. Before/After apply to all seven correct-rewarded generalization targets,
   as well as the mixed-outcome classifiers. Preserve the existing Full fit.
3. Show both training and held-out summaries by outcome, with actual-choice
   and recorded-state/context left/right groupings and factual counts.
4. Reward decoding uses actual reward only. Model-derived expectation is a
   separate, more involved analysis outside this expansion.
5. Add reward as a mixed-outcome target alongside state and action; retain
   the two existing named analysis modes. "Context generalization" in this
   discussion refers to the existing Correct-rewarded generalization mode.

This is an in-place extension of the existing spec, not a separate pipeline
or replacement of previous results. The existing implementation plan pairs
these scientific decisions with test-first work packages and preserved evidence.
