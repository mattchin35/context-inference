# Cross-Condition Generalization of Context, Choice, Actual Reward, and Value

Status: scientific decisions confirmed, revision 10, 2026-11-10. The original
analysis and the Context outcome extension in Section 3.1 are implemented.
This specification does not authorize implementation or experimental runs.
The implementation plans record delivery and validation requirements.

References:

- [Implementation plan and live handoff](cross_condition_generalization_implementation_plan.md).
- [Context outcome extension implementation plan](context_outcome_generalization_implementation_plan.md).
- [Implemented session-prediction specification](condition_prediction_generalization_spec_v1.md).
- [Its implementation and validation record](condition_prediction_generalization_implementation_plan.md).
- [Earlier condition-generalization specification](task_variable_condition_generalization_spec_v1.md).
- [Session timeline/PCA specification](session_timeline_pca_spec_v1.md).
- [Software design guidance](SoftwareDesign.md).

## 1. Goal and scope

For each binary target, train a neural readout on two condition cells containing
both target classes. Assess whether that readout transfers to complementary
conditions containing both target classes. Keep the original four transfer
directions individually identifiable for each target. Add the three Context
outcome-training models and six separate evaluations defined in Section 3.1.

Targets are recorded context, actual choice, actual reward, and binary
correctness-based value. Existing seven-target correct-rewarded generalization
and three-target mixed-outcome decoding remain separate and unchanged.
Continuous behavioral-model targets are outside scope.

Generalization concerns a change in trial conditions. Train on all eligible
source-cell trials and evaluate all eligible complementary-cell trials, with
no additional training fraction or block cross-validation. The test cells are
excluded from fitting. Source-cell scores describe training performance, not
same-condition held-out decoding.

Focus inspection on decoding/generalization summaries. Retain the Full,
Before, and After supervised windows, but compute decoder outputs only within
the selected training/scoring interval. No full `[-2,+2)` decoder traces,
session-timeline prediction view, or PC/contribution heatmaps are requested.
The existing session PCA remains a neural input, not a new visualization task.
Use version-3 session metadata. Full supervised windows do not overlap in this
task; no overlap checks or handling are needed.

## 2. Targets and condition coordinates

Use the existing trial-i target definitions and original trial chronology:

| Visible target | Existing target ID | Source | Class 1 | Class 0 |
| --- | --- | --- | --- | --- |
| Context | `current_state` | `state_int` | Left context | Right context |
| Choice | `current_action` | `action` | Left choice | Right choice |
| Actual reward | `current_reward` | `reward` | Rewarded | Unrewarded |
| Value (correctness-based) | `current_action_is_correct` | `correct` | Correct, rewarded or omission | Incorrect |

All labels are dimensionless. Do not shift labels, infer reward from
correctness, or infer context from choice. Recorded correctness comes from
`correct`; it is both a condition coordinate and the existing binary target
reused for Value. No new continuous or modeled-value target is introduced.

Normalize numeric label/flag columns on a local copy before constructing
condition masks, using the target loader's numeric and missing-value semantics.
Numeric strings must produce the same memberships as numeric values. Preserve
the original table, row order and IDs; malformed labels are not coerced to missing.
Ignore blocks entirely: no required block column, block-value validation,
eligibility filtering or fold assignment.

For the binary context task, a left choice in left context and a right choice
in right context are correct; the opposite combinations are incorrect. For an
ordinary trial whose context, choice, and correctness are all present and
binary, a conflict means recorded `correct` differs from `(state_int == action)`.
Exclude conflicting rows from both fitting and scoring for this new analysis,
and continue valid trials. Save and report their count, original row positions,
trial IDs, recorded values, and exclusion reason. Do not infer replacement
labels, delete chronological rows, or change existing analyses. Compute
training weights after these exclusions and regional/target eligibility.

Apply existing manual-reward/no-choice/alignment exclusions before checking
context. Missing context on those excluded rows must not stop the analysis.
Among the remaining ordinary choices, exclude dark state (`state_int == 2`)
from all four targets and report its count. All other ordinary choices must
have context 0/1; missing/invalid context is an input error. Other label validity
follows the existing target loader. These are eligibility masks: no rows are
deleted, labels imputed or PCA refitted.

For the original context/choice/value transfers, correct means the union of
canonical correct-rewarded and correct-omission trials. The additional Context
outcome transfers use separate outcome training pools (Section 3.1).
For reward, the unrewarded pool combines correct
omissions and incorrect choices with recorded reward 0; the rewarded pool uses
the eligible actual-reward-1 trials. These are combined fitting pools, with
outcome-separated summaries of the same fitted models. Separate outcome training
is added only for Context; Choice, Actual reward and Value keep their existing
definitions. Missing values never become class 0.

## 3. Context generalization: four directed transfers

Here L/R means recorded **context**, not action. Each decoder predicts context.
The transfer descriptions involving action assume the correctness relationship
in Section 2.

| Split ID | Training cells | Complementary test cells | Interpretation |
| --- | --- | --- | --- |
| `context_correct_to_incorrect` | L-context correct + R-context correct | L-context incorrect + R-context incorrect | Correct to incorrect |
| `context_left_choice_to_right_choice` | L-context correct + R-context incorrect | L-context incorrect + R-context correct | Left choice to right choice |
| `context_right_choice_to_left_choice` | L-context incorrect + R-context correct | L-context correct + R-context incorrect | Right choice to left choice |
| `context_incorrect_to_correct` | L-context incorrect + R-context incorrect | L-context correct + R-context correct | Incorrect to correct |

In the middle two splits, action is constant within the training pool and
changes for testing, while both context classes remain represented in each
pool. In the first and last splits, the relationship between context and action
reverses between training and testing.

### 3.1 Additional Context outcome transfers

Train on both recorded contexts within one outcome, then evaluate the same
readout separately on each other outcome:

| Training outcome, Left + Right context | Separate test outcome 1 | Separate test outcome 2 |
| --- | --- | --- |
| Correct-rewarded | Correct-omission | Incorrect |
| Correct-omission | Correct-rewarded | Incorrect |
| Incorrect | Correct-rewarded | Correct-omission |

These are three source models and six directed evaluations. Each source model
is fitted once per window/weighting/region; its two test summaries share the
same coefficients, intercept and source weights. The incorrect-trained model
is the existing `context_incorrect_to_correct` readout, reused in both Context
views. Only rewarded-only and omission-only source models are added. Retain the
original four Context condition transfers alongside this new outcome view.

Use canonical correct-rewarded, omission and incorrect memberships after the
common exclusions. Incorrect includes otherwise eligible incorrect choices
regardless of recorded reward; it is not restricted to reward 0. Correct trials
with unknown reward belong to neither correct-rewarded nor omission. Class 1
remains Left context and class 0 Right context, including on incorrect choices.
Never use action to relabel context.

Training and test outcomes are disjoint. Do not pool the two test outcomes into
the requested individual summaries or refit a model for each test outcome.
Preserve all eligible training trials and existing weighting/estimator settings.

Display total, Left-context and Right-context training trial counts directly in
the outcome-results table for each region. Counts are unique eligible source
trials after behavioral/target exclusions and regional neural coverage, not
bins or weight sums. They do not change when selecting a test outcome or
weighting. Keep known eligible counts after a fit failure; unknown regional
coverage stays unknown. Show test counts and availability reasons separately.
No automatic small-sample threshold, subsampling or new significance test is
specified. Source training performance remains an in-sample diagnostic.

## 4. Choice generalization: four directed transfers

Here L/R means actual **choice**, not context. Each decoder predicts choice.

| Split ID | Training cells | Complementary test cells | Interpretation |
| --- | --- | --- | --- |
| `choice_correct_to_incorrect` | L-choice correct + R-choice correct | L-choice incorrect + R-choice incorrect | Correct to incorrect |
| `choice_left_context_to_right_context` | L-choice correct + R-choice incorrect | L-choice incorrect + R-choice correct | Left context to right context |
| `choice_right_context_to_left_context` | L-choice incorrect + R-choice correct | L-choice correct + R-choice incorrect | Right context to left context |
| `choice_incorrect_to_correct` | L-choice incorrect + R-choice incorrect | L-choice correct + R-choice correct | Incorrect to correct |

The middle two are context transfers: their training context is fixed, their
test context is the opposite, and both choice classes occur in each pool.

Context/choice correct-to-incorrect fits share source trials and training labels,
so identical settings can produce identical readouts while test labels differ.
Their incorrect-to-correct fits share trials with reversed training-label
orientation. Do not force these readouts to differ or treat them as independent.

## 5. Actual-reward generalization: four directed transfers

Here L/R means actual **choice**. Every decoder predicts recorded reward,
regardless of which side was chosen.

| Split ID | Training cells | Complementary test cells | Interpretation |
| --- | --- | --- | --- |
| `reward_left_choice_to_right_choice` | L-choice rewarded + L-choice unrewarded | R-choice rewarded + R-choice unrewarded | Left choice to right choice |
| `reward_left_rewarded_right_unrewarded` | L-choice rewarded + R-choice unrewarded | R-choice rewarded + L-choice unrewarded | Reverse reward/choice pairing |
| `reward_right_rewarded_left_unrewarded` | R-choice rewarded + L-choice unrewarded | L-choice rewarded + R-choice unrewarded | Reverse pairing, opposite direction |
| `reward_right_choice_to_left_choice` | R-choice rewarded + R-choice unrewarded | L-choice rewarded + L-choice unrewarded | Right choice to left choice |

The middle two deliberately reverse the reward/choice association. A readout
driven by choice alone could perform below chance on the complementary cells.
Keep the original reward orientation and report that result; do not flip labels
or invert predictions to raise the score.

Expectation/Outcome is the user's conceptual grouping. The operational target
is actual reward, not subjective expectation. Before predicts the eventual
recorded reward and After uses post-choice activity; After is not necessarily
after reward delivery.

## 6. Value generalization: four directed transfers

Here L/R means actual **choice**. The confirmed target is binary recorded
correctness: 1 for a correct choice, combining correct-rewarded and correct
omission trials, and 0 for an incorrect choice. The visible target name is
**Value (correctness-based)**; its probabilities have the meaning P(correct),
not expected reward magnitude, Q-value, or subjective belief.

| Split ID | Training cells | Complementary test cells | Interpretation |
| --- | --- | --- | --- |
| `value_left_choice_to_right_choice` | L-choice correct + L-choice incorrect | R-choice correct + R-choice incorrect | Left choice to right choice |
| `value_left_correct_right_incorrect` | L-choice correct + R-choice incorrect | R-choice correct + L-choice incorrect | Reverse correctness/choice pairing |
| `value_right_correct_left_incorrect` | R-choice correct + L-choice incorrect | L-choice correct + R-choice incorrect | Reverse pairing, opposite direction |
| `value_right_choice_to_left_choice` | R-choice correct + R-choice incorrect | L-choice correct + L-choice incorrect | Right choice to left choice |

Correct omissions distinguish Value from Actual reward: their Value label is
1 and their Actual reward label is 0. Keep those source labels and saved target
identities distinct. Do not relabel omissions as incorrect, derive Value from
the reward flag, or add behavioral-model fitting. Value is recorded correctness,
not a measured subjective value or calibrated reward probability.

The middle two splits train within one context and test the opposite context
under Section 2's consistency rule. Retain below-chance scores without relabeling
or inverting outputs.

## 7. Fitting and evaluation

### Confirmed reuse of existing settings

- Separate PFC/HPC readouts on all PCs from the fixed session-wide regional PCA.
- Full `[-1,+1)`, Before `[-1,0)`, and After `[0,+1)` choice-relative second
  training windows; score each model over its own training interval.
- Evaluate neural bins only in the selected supervised interval. No decoder
  outputs outside it, intertrial predictions, or cross-window scoring matrix
  are needed. Full is an independent fit, not an average of Before and After.
- Reuse the existing session PCA and its complete source-window eligibility.
  Its original `[-2,+2)` input windows remain unchanged; select supervised bins
  from its scores without adding partial-window PCA preparation or refitting.
  Reuse any saved regional basis whose scientific compatibility can be verified;
  unrelated configuration differences alone do not justify recomputation.
- Original unwhitened PC scores; L2 logistic regression with fixed `C=1.0`
  and an unpenalized intercept; no tuning or target-specific rescaling.
- Pool selected bins as training rows. Divide each trial's total weight across
  its selected bins. A shorter interval must not reduce a trial's total weight
  simply because it has fewer bins. Prepare both weighting schemes below.
- Default 500-ms bins, with existing 100/50/20-ms choices and exact half-open
  boundaries. Time zero belongs to After.
- Existing manual-reward/no-choice exclusions, valid labels, and complete
  regional neural coverage. Missing or unavailable cells remain visible in
  counts and fit/score reasons. If a region is unavailable, show its reason and
  continue with usable regions. If neither is usable, show both reasons without
  crashing; no model is fitted for an unavailable region.

Reuse estimator settings, not the old five-fold orchestration or saved fold
axes. Fit one all-source model per target/split/weight/window/region combination.

### Combined fitting pools and separate outcome summaries

The original transfers use the following combined pools; Section 3.1 defines
the additional outcome-specific Context source pools:

| Target | Combined fitting definition | Outcome separation |
| --- | --- | --- |
| Context/choice | Correct = correct-rewarded OR correct-omission; incorrect uses the canonical incorrect mask. | Show correct outcomes together and as correct-rewarded / correct-omission, plus incorrect trials. |
| Actual reward | Rewarded = actual reward 1; unrewarded = correct-omission OR (incorrect AND actual reward 0). | Show unrewarded outcomes together and as correct-omission / incorrect-unrewarded, plus rewarded trials. |
| Value (correctness-based) | Positive = correct-rewarded OR correct-omission; negative = canonical incorrect. | Show correct outcomes together and as correct-rewarded / correct-omission, plus incorrect trials. |

Apply these definitions wherever the condition occurs in either the source or
test cells in Sections 3-6. Source/test membership remains disjoint. Use the
actual reward flag rather than assuming every incorrect trial is unrewarded.

Every summary is the intersection of its selected training/test role and its
outcome membership. Report the combined eligible population and each relevant
outcome subset without changing coefficients or training weights. Never move
training trials into a test summary to populate an otherwise empty outcome
row. Empty outcome/role intersections remain empty and visibly unavailable.
Outcome summaries can overlap, so their counts must not be summed as though
all rows were disjoint categories.

Reward's correct-omission/incorrect-unrewarded rows contain class 0 only;
rewarded-only contains class 1. Value's correct-rewarded/omission rows contain
class 1 only; incorrect contains class 0. Ordinary accuracy is a class-specific
success rate for these rows; balanced accuracy is unavailable. Show actual class
counts for every summary. Combined populations support balanced accuracy only
when both classes are present.

Outcome selection is display/scoring only and requires no fit. These summaries
show where a combined-pool readout succeeds; they do not establish what a
readout trained exclusively on either component would have learned.

### Two training-weight schemes

Prepare both schemes for comparison using saved results:

- `equal_trial`: retain every eligible source trial with total weight 1.
- `equal_cell`: retain every eligible source trial, with equal total training
  weight assigned to the two source cells. Each source cell contains one
  target class, so this also balances the target classes within this fit.

Confirmed normalization: after regional coverage and label eligibility, let
N = n0 + n1 be the source trial count and nk the count in source cell k. Each
trial in cell k receives N/(2*nk), then each of its B supervised bins receives
N/(2*nk*B). Both schemes therefore sum to total weight N, preserving
the overall weighted-loss scale at fixed C while changing its allocation.
No trial subsampling or random draws are needed. Do not also apply estimator
class weights, which would balance twice. Internal outcome subgroups within a
combined source cell are not separately balanced.

Training weights do not redefine evaluation: retain equal weight per scored
trial and report both ordinary and balanced accuracy for both fitted schemes.
Saved model identities include scheme and normalization. Switching schemes
selects separately fitted outputs within the bundle; it never refits or merely
reweights one model's scores.

### All-source / complementary-condition evaluation

For each target, directed split, weight scheme, window, and region:

1. Select all eligible trials in its two source cells and fit one model.
2. Evaluate that model on all eligible trials in the two complementary test
   cells, using the matched scoring window.
3. Report the exact source trials' in-sample scores separately as training
   diagnostics. Do not label them held-out or cross-validated.
4. Compute per-bin probabilities only for the selected supervised interval,
   then average probabilities once per trial for scoring. Retain training/test
   membership and trial-level outputs for inspection and reproducibility.
   Trials outside this analysis's cells do not need decoder outputs.

Never use test cells to select penalties, thresholds, PCs, a preferred window,
or a preferred weight scheme and then describe the chosen score as an
independent validation. Comparing these predeclared settings remains
exploratory. Missing source classes make only that fit unavailable; a missing
test class leaves balanced accuracy unavailable, with counts and reasons.
Do not borrow test trials, synthesize missing cells, or silently fall back to
another fitting pool. Each eligible test trial is counted once within its
selected target/split/weight/window/region/outcome summary; the same trial can
appear in overlapping summary populations.

The original sixteen splits imply up to 192 binary fits across three windows,
two weighting schemes and two regions. The Context extension adds two unique
source definitions (24 fits), bringing the complete bundle to at most 216 fits.
The reused incorrect-trained definition is counted once. The six requested
outcome evaluations are summaries of three source models, not six source fits.
Reuse saved PCA scores across fits and retain trial means rather than full-window
decoder or contribution tensors. Measure runtime and memory during implementation.

## 8. Results, inspection, and interpretation

The primary outputs are one-trial matched-window decoding summaries,
with source-training and complementary-test roles clearly distinguished.
There is no source-held-out or block-held-out score in the selected policy.
Reuse trial-mean probabilities before applying the fixed 0.5 threshold. Report
ordinary accuracy, balanced accuracy, class counts, and cell counts. Empty
metrics remain unavailable; balanced accuracy requires both actual target
classes in the scored pool.
For unavailable fits, distinguish eligible cell counts from scored trials:
eligible data may exist even though no valid predictions can be scored.
Keep the four directed transfers individually visible; an average across them
is not yet requested. The additional Context outcome view shows all six
train-outcome/test-outcome pairs separately, with visible training sample sizes.

Store sufficient data to inspect a result without raw neural/behavioral
sources: original trial identities; actual labels and validity; canonical
conditions and split-specific cell memberships; exact training/test membership,
combined fitting-pool definitions, weight scheme and trial weights; time grids
and selected bins; regional coefficients, intercepts and fit reasons;
matched-window trial-mean probabilities; conflict/exclusion records; target
orientation; source PCA identity; parameters; analysis/schema version; and
reproducibility provenance. Per-bin decoder traces are not a required saved
product. New products must not overwrite or masquerade as existing prediction modes.
Final file layout and public interfaces belong in the later implementation plan.

Inspection is a summary-only webapp view. Display target, source-to-test
direction, summary outcome, weighting scheme, supervised window, region, and
evaluation role explicitly, along with fit availability and exclusions. Keep
four transfer directions identifiable without requiring session navigation.
Reuse existing summary-view conventions; exact module placement, table/figure
layout, and preparation entry points belong in the implementation plan.

A summary-outcome selector reuses the same model; a weighting selector
chooses separately fitted saved outputs. A saved result must keep its own
settings when viewed, with no fitting on selector changes. If a requested
fitted variant has no product, identify it as unprepared rather than silently
displaying another weighting scheme. Summary-outcome filters do not define
new model identities. Do not add this analysis's controls or prediction tracks
to Session timeline and PCA.

Interpretation limits:

- Context means recorded task context. Transfer does not establish the mouse's
  inferred context or a causal neural representation.
- Value means recorded correctness with correct omissions kept positive. It
  does not measure subjective value; context and choice jointly determine
  correctness under the binary task rule, so their coding can affect transfer.
- Session-wide PCA includes test-condition neural activity, as in the existing
  offline analysis. Supervised separation does not make preprocessing held out.
- Condition-based test trials can share behavioral blocks and nearby time
  periods with training trials. This is not temporal/block-held-out validation.
- Low transfer scores can reflect weak source decoding. Source training
  performance is only an in-sample diagnostic; without a same-condition
  held-out reference, overfitting and selective transfer failure are not fully
  separated.
- Sample sizes, class balance, reward/correctness composition, and timing can
  differ between pools. At fixed C with summed weighted loss, different total
  training weight also changes regularization relative to average loss.
- Reverse-direction fits and overlapping target analyses are not independent
  repetitions. No significance tests, label permutations, confidence intervals,
  or causal/subjective-expectation claims are specified in this draft.

## 9. Preparation and handoff

One explicit preparation call produces all eighteen unique readout definitions,
three supervised windows and both weightings at the selected bin width after
the Context extension is implemented. The summary view includes
Prepare/Recompute controls alongside the metadata-based CLI. Preparation may
create missing PCA through the existing runner; inspection never prepares or fits.

Revision 9 incorporates the review decisions: version-3 metadata only, blocks
ignored, numeric masks normalized locally, context checked after baseline
exclusions, confident PCA reuse, and per-region unavailability reported without
crashing. No backward-compatibility, overlap-handling or recovery subsystem is
requested. Revision 10 adds Section 3.1's Context outcome fits and visible
regional training counts. See the Context extension plan for the new result
version, contracts, tests and approval gates; original implementation evidence
remains in the completed baseline plan.
