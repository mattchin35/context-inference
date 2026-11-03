# Condition Prediction Generalization on the Session Timeline

Status: scientific decisions confirmed, revision 1, 2026-11-03.
Implementation has not started.

Draft implementation plan:
[condition_prediction_generalization_implementation_plan.md](condition_prediction_generalization_implementation_plan.md).

This document records the user's confirmed decisions for a new session
prediction analysis. It is not an implementation plan. These decisions take
precedence over suggestions made earlier in the discussion. Section 11 records
the five-fold policy confirmed after the user's requested explanation.

Related documents:

- [Existing condition generalization specification](task_variable_condition_generalization_spec_v1.md).
- [Session timeline and PCA specification](session_timeline_pca_spec_v1.md).
- [Timeline implementation and refinements](session_timeline_pca_implementation_plan.md).
- [Optional treadmill speed track](session_timeline_speed_implementation_plan.md).
- [Software design guidance](SoftwareDesign.md).

## 1. Goal

Train a population readout on correct-rewarded trials, then display its
predicted task variable during individual choice windows throughout the
session. Show choices, selected behavioral events, and the prediction on a
shared elapsed-session-time axis. Overlay the actual trial-level target for
comparison.

This expands the condition generalization idea from aggregate decoding
performance to the actual predicted values for individual trials and bins.
It uses the existing session-wide regional PCA bases and the existing
timeline's event and navigation behavior.

The readout is an offline analysis. Training includes activity after the
choice, so the resulting trace must not be described as an online or strictly
pre-choice forecast.

## 2. Confirmed decisions

| Item | User decision |
| --- | --- |
| Training condition | Canonical `correct_rewarded` trials. |
| Targets | The seven targets in Section 3. |
| Training interval | One pooled interval, `[-1, +1)` seconds relative to choice. |
| Prediction interval | The complete `[-2, +2)` second choice window. |
| Intertrial intervals | Leave gaps; do not predict outside choice windows. |
| Evaluation | One deterministic five-fold behavioral-block assignment shared across targets and regions. |
| PCA fitting scope | Session-wide PCs, including conditions beyond correct-rewarded. |
| Predictor PCs | All available saved session PCs for each region. |
| Regional outputs | Separate PFC and HPC models and tracks. |
| Default bin width | 500 ms; 100 ms must also be available. |
| Binary model | L2-penalized logistic regression. |
| Continuous model | Ridge regression. |
| Penalty selection | Fixed penalties; keep preparation simple. |
| Penalty settings | Ridge `alpha=1.0`; logistic `C=1.0`; unpenalized intercepts. |
| Decoder feature scale | Existing unwhitened PC scores, without another PC standardization. |
| Actual target | Overlay the actual trial-i target across each choice window. |
| Optional neural view | Original PC activity or signed weighted PC contributions. |
| Optional PC ordering | Original rank or contribution strength; fixed across session navigation. |
| Preparation | Provide an explicit Prepare control. |
| Initial webapp location | Controls within Session timeline and PCA; revisit a separate view if crowded. |
| Validation session | CT026 2026-08-04. |

The regional decision excludes a combined PFC+HPC model from this revision.
Use all available session PCs as predictors. The number of visible heatmap
rows remains independent of the model's predictor count.

The name "full window" in this discussion means the new two-second training
interval `[-1, +1)`. It must not silently reuse the existing pooled decoder's
`full` setting, which means `[-2, +2)`. Save exact numeric bounds in the new
configuration and results.

## 3. Initial target set

Reuse the current target definitions and source columns. Each target refers
to trial i; do not shift labels to a previous or next trial.

| Target identifier | Source column | Family | Displayed prediction |
| --- | --- | --- | --- |
| `current_state` | `state_int` | Binary | Probability of left state, class 1. |
| `current_action` | `action` | Binary | Probability of left action, class 1. |
| `qlearning_relative_value` | `Qlearning_rel_value` | Continuous | Predicted relative value in source units. |
| `forgetting_q_relative_value` | `FQlearning_rel_value` | Continuous | Predicted relative value in source units. |
| `hmm_signed_belief` | `HMM_rel_value_logodds` | Continuous | Predicted HMM belief log-odds. |
| `hmm_decay_signed_belief` | `HMM_rel_value_logodds_decay` | Continuous | Predicted decayed HMM belief log-odds. |
| `relative_doubt` | `relative_doubt_index` | Continuous | Predicted relative doubt in source units. |

Use the existing augmented behavioral table and feature provenance for model
values. Do not refit the behavioral models or invent new units, normalization,
clipping, or target definitions as part of prediction preparation.

Construct and align targets on the original chronological table before
filtering the training condition. Preserve original row positions and trial
IDs when connecting this table to saved timeline choices.

### State and action interpretation

State and action normally agree on correct-rewarded trials. If their eligible
training rows, labels, and settings are identical, the two binary models
should produce identical predictions. Their actual labels can differ on
other trials. The specification retains both requested targets, but their
names do not establish independently identified state and action readouts.
Report this limitation rather than forcing the models to differ.

## 4. Neural inputs and the fixed session PCA basis

Use separate session-wide PFC and HPC bases from the existing timeline
analysis. That analysis pools complete regional choice windows across
conditions, standardizes units once, and fits an unwhitened regional PCA.
Retain the saved axes, signs, unit selection, scaling, and score units.

The PCA fitting pool remains the timeline's complete `[-2, +2)` windows. The
narrower `[-1, +1)` interval applies to supervised readout fitting, not a new
PCA fitting scope. Do not refit PCA on correct-rewarded trials, per fold, per
viewport, or per target.

Use original saved scores for prediction. The existing optional display-only
PC z-scoring must not change model inputs, coefficients, predictions, or
contribution rankings.

Use one bin width consistently for the selected saved PCA basis, supervised
training, prediction, and time-bin placement. Training bins are pooled as
individual observations rather than averaged into one vector per trial.
The supported timeline widths are 500, 100, 50, and 20 ms. Default to 500 ms
and keep 100 ms available as an ordinary selectable option. At 500 ms, each
choice supplies four training bins and eight prediction bins; at 100 ms it
supplies twenty training bins and forty prediction bins. Changing width
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
group assignment before selecting correct-rewarded training rows. Every
prediction must identify the model and held-out block fold that generated it.

For each target, region, and outer fold:

1. Select eligible correct-rewarded trials from the other folds.
2. Select the bins inside `[-1, +1)` seconds.
3. Stack `(training trial, training bin, PC)` into
   `(training trial * training bin, PC)` in trial-major order.
4. Repeat the trial-i target for its selected bins. Follow the existing pooled
   convention of row weights `1 / training_bin_count`, so each training trial
   has total weight one. Record this policy with the penalty settings.
5. Fit one intercept and PC-weight vector using the selected L2 model.
6. Apply that model to every valid bin in the `[-2, +2)` windows of the
   held-out blocks, across conditions.
7. Save individual predictions rather than reducing them to a fold score.

The user selected fixed penalties. There is no inner hyperparameter search
in this revision. Use ridge `alpha=1.0` and logistic `C=1.0`, with unpenalized
intercepts, original PC feature scales, and native continuous target units.
The implementation plan records the weighted loss conventions; equal numeric
values for logistic and ridge parameters do not imply equal penalties.

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
state or left action. Weighted PC contributions are additive log-odds terms,
not additive probability contributions. The continuous HMM log-odds target
remains a separate numerical regression target.

### Optional PC rows

Provide a display choice between:

- Original signed session-PC scores.
- Signed weighted contributions, `weight_j * PC_score_j(t)`.

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

For a selected target, show separate PFC and HPC prediction panels beneath
the behavioral tracks. Overlay the actual trial-i target on both panels,
constant within its choice window. Do not interpolate the actual target or
predictions across intertrial gaps. Missing actual targets and unavailable
predictions remain distinguishable.

Place optional regional PC activity/contribution heatmaps underneath the
prediction tracks. Their visibility, ordering, and displayed PC count are
display settings; they do not change which features the saved model used.
Keep the existing timeline's light appearance and readable labels.

Hover should identify the target, region, trial, condition/training-condition
membership, elapsed and choice-relative bin interval, held-out fold,
prediction, and actual target when present. Contribution hover should also
identify the original PC, applicable weight, and contribution units.

Clearly label the supervised training interval `[-1, +1)`. Predictions in
the remaining parts of `[-2, +2)` apply the frozen readout outside its training
time interval. They are still held out by block, with the PCA-scope limitation
described in Section 5.

## 8. Preparation and saved results

Provide an explicit Prepare control. Preparation performs model fitting and
saves predictions and provenance; display interactions load saved products.
Reuse compatible saved preparation by default. Require an explicit rerun
action for a new result when an equivalent preparation already exists.

Keep a separate versioned identity for this analysis. Existing task-variable,
pooled condition-generalization, and session-PCA results retain their current
schemas and meanings. In particular, the existing pooled `full` training
window must not change.

Save named arrays in a session-local NPZ with `meta` metadata, consistent
with repository rules. At minimum retain:

- Session, source PCA archive/configuration identity, target source identities,
  analysis version, code provenance, parameters, and random seed where used.
- Original row positions, trial IDs, choices, time origin, exact bin edges,
  target values, target validity, and training-condition membership.
- Regional window mappings, original PC IDs, feature counts, model fold IDs,
  training masks/counts, coefficients, intercepts, and fit status/reasons.
- Individual per-bin predictions, availability, and binary output orientation.
- Fixed penalties, estimator/loss settings, row weighting, PCA scope, and
  contribution-strength definitions and summaries.

For example, regional predictions can use `(regional window, bin, target)`
and weights `(target, fold, regional PC)`. These are conceptual axes, not a
frozen on-disk schema. The implementation plan must specify exact array names
and mappings before coding.

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
| `src/neural_analysis/task_decoding/conditions.py` | Canonical correct-rewarded mask. |
| `src/neural_analysis/task_decoding/modeling.py` | Existing block-split primitives where their contracts fit; its current elastic-net estimator factory is not the selected new L2 model. |
| `src/neural_analysis/task_decoding/pooled/decoding.py` | Reference for pooled row ordering, repeated labels, and trial weights. Its training-only PCA and aggregate-only outputs are not this analysis's contracts. |

Keep numerical fitting, saved results, and plotting separate. Do not refactor
unrelated pipelines or introduce a general track/plugin framework. Reuse
installed NumPy, SciPy/scikit-learn, and Plotly capabilities after checking
their installed APIs. A concrete module/file plan is a subsequent step.

## 10. Required validation topics for the later implementation plan

The later plan must cover these scientific and display behaviors before
experimental preparation:

- Correct trial-i target alignment despite excluded or missing rows.
- Correct-rewarded-only supervised training and exact `[-1, +1)` bin selection.
- Shared block exclusion for every training/prediction pair, including all
  bins from each trial; recorded fixed session-wide preprocessing scope.
- Prediction at every valid `[-2, +2)` bin with real intertrial gaps preserved.
- Separate regional availability and training failures.
- Binary probability orientation, continuous native units, and distinctions
  between missing targets, unavailable fits, and valid predictions.
- Reconstruction of outputs from fold-specific weights and PC contributions.
- Contribution ranking that accounts for PC scale, retains original PC IDs,
  and remains fixed while navigating.
- Equal settings and labels yielding equal state/action predictions where
  their eligible training subsets coincide.
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
- For each held-out group, train on eligible correct-rewarded trials from the
  other four groups and predict all eligible choice windows in that group.
- Use the same assignment for all seven targets and both regions, while
  retaining target-specific label eligibility and regional neural coverage.
- Choices with missing block IDs have no held-out prediction; behavior and
  actual values can still be displayed.
- Fewer than five nonmissing blocks produces an explicit preparation error;
  individual unfit training subsets are marked unavailable without silently
  changing fold count, model, or training condition.

These groups need not be consecutive session intervals, and training can
include blocks later than a predicted choice. The trace therefore
evaluates offline block generalization, not future-only prediction.

## 12. Scope limits

This revision covers same-session, correct-rewarded-trained regional PC
readouts and individual choice-window predictions. It does not require
continuous intertrial prediction, cross-session models, combined PFC+HPC
models, direct-unit decoders, destination-native model comparisons,
hyperparameter searches, new behavioral-model fits, significance testing,
or new Slurm resource settings.

Preparing this specification does not authorize experimental computation or
implementation. The scientific settings are now resolved; the linked draft
implementation plan is ready for review before implementation begins.
