# Population context, choice, outcome and choice-switch decoding

Status: scientific choices confirmed, 2026-11-14. The user selected both raw
unit and PC population inputs and accepted the remaining defaults. This is a
scientific specification; an implementation plan and implementation approval
remain separate. Implementation of the revised plan was approved on 2026-11-14;
an experimental run is not yet authorized.

Related contracts:

- [Single-unit encoding specification](single_unit_ridge_spec_v1.md).
- [Existing population decoding specification](task_variable_spec_v5.md).
- [Existing session prediction specification](condition_prediction_generalization_spec_v1.md).
- [Software design guidance](SoftwareDesign.md).
- [Implementation-plan draft](population_task_decoding_implementation_plan.md).

## 1. Goal and direction

Decode recorded task/behavioral values from simultaneously recorded neural
population activity within a session:

`neural population activity -> context, choice, outcome, upcoming choice switch`

The population is the model input. This is a different analysis from fitting
each unit's activity as the response. Encoding coefficients cannot simply be
inverted to obtain these decoders. Do not label population decoder weights as
single-unit selectivity.

The requested displayed values are left/right context, left/right current
choice, left/right reward presence, left/right reward absence, and an upcoming
left-to-right or right-to-left change of chosen port. A choice switch is not
a task-context switch.

## 2. Confirmed scientific decisions

| Item | Confirmed definition |
| --- | --- |
| Context | Recorded current trial's left/right task context. |
| Choice | Recorded current trial's left/right chosen port. |
| Context/choice timing | Fixed current-trial labels across choice-relative `[-1,+1)`. |
| Outcome fitting/scoring activity | Choice-relative `[0,1)` seconds only. |
| Outcome model | One four-class population decoder with four probabilities summing to one. |
| Reward present | Recorded `reward == 1` at the chosen side. |
| Reward absent / omission | Recorded `reward == 0` at the chosen side, including incorrect trials. |
| Left switch trial | A current left choice followed by a right choice on the next trial. |
| Right switch trial | A current right choice followed by a left choice on the next trial. |
| Choice-switch fitting/scoring activity | Whole choice-relative `[-1,+1)` second window. |
| Switch outputs | Two directional switch indicators across all eligible trials; no within-side switch-versus-stay analyses. |
| Population inputs | Both standardized raw-unit population vectors and training-fold PC vectors, saved as separate comparisons. |
| Regions | Independent PFC and HPC decoders; no combined-region model. |
| Observations | One population vector per bin, pooled within the target's window; 500-ms default bins. |
| Preprocessing | Unit scaling and PCA fitted on training-fold activity only. |
| Estimator | Fixed L2 logistic regression, binary or four-class as appropriate; no tuning or class balancing. |
| Evaluation | One deterministic five-fold behavioral-block assignment, shared across targets and representations. |

Use source codes `state_int: 0=right, 1=left`; `action: 0=right, 1=left`.
Do not infer context from action or reward from correctness.

The user's meaning of omission here differs from the existing canonical
`omission` condition and the single-unit model's omission regressors, which
require correct unrewarded choices. Do not reuse that narrower mask. Prefer
explicit identifiers `left_no_reward` and `right_no_reward`, with the UI label
"Omission / no reward" if desired. Correctness is unnecessary for determining
the four requested outcome labels.

## 3. Neural windows and observations

Extract only choice-relative `[-1,+1)` seconds, initially in 500-ms bins, with
100/50/20-ms resolutions available. "Trial-wide" means a constant current-trial
label across this extracted two-second window, not the entire
trial-start-to-start period. At 500 ms, full-window models use four bins per
eligible trial and outcome models use the two post-choice bins.

Each observation is one neural population vector from one bin. Pool bins
across eligible trials for a readout; do not average an entire window into
one vector or fit an independent classifier at each offset without a separate
decision. Trial and bin identity must survive flattening.

For context/choice, hold the recorded label fixed through the window even if
context changes following that choice. Outcome models fit and score only
`[0,1)`. There are no forced all-zero pre-choice outcome targets. Display
outcome tracks only in their supported post-choice interval by default.

Choice time remains the alignment reference; `[0,1)` does not assert that
reward delivery or its expected delivery time coincides exactly with choice.

Choice-switch models fit and score across the whole `[-1,+1)` window, using
the same directional switch-indicator labels throughout that window. This
includes post-choice activity: it is an offline readout of an eventual next-choice
label, not automatically a prospective prediction or a measure of intention
before the current choice.

## 4. Target construction

Build labels on the complete chronological trial table before any neural,
condition, or coverage filtering. Preserve original row positions and trial
IDs. Unknown labels remain unavailable rather than becoming negatives.

Context and current choice are each binary variables. Fit one classifier per
variable, displaying `P(left)` and
`P(right)=1-P(left)` rather than fitting inconsistent independent complements.

For known actions/rewards, post-choice outcomes are mutually exclusive:

| Outcome | Exact source predicate |
| --- | --- |
| `left_reward` | `action[i] == 1 and reward[i] == 1` |
| `right_reward` | `action[i] == 0 and reward[i] == 1` |
| `left_no_reward` | `action[i] == 1 and reward[i] == 0` |
| `right_no_reward` | `action[i] == 0 and reward[i] == 0` |

An incorrect right choice without reward belongs to `right_no_reward`, even
if the task context is left. Side always means the actual chosen port.

Fit one four-class outcome classifier. In the table's fixed class order,
return `P(left_reward)`, `P(right_reward)`, `P(left_no_reward)` and
`P(right_no_reward)`, summing to one per evaluated post-choice bin. Do not fit
four independent binary readouts. Incorrect unrewarded choices belong to the
appropriate no-reward class, not a fifth class. Pre-choice bins are outside
the outcome fitting/scoring scope, not a fifth class or four negative labels.

Upcoming directional choice-switch labels are:

- `left_choice_switch`: `action[i] == 1 and action[i+1] == 0`.
- `right_choice_switch`: `action[i] == 0 and action[i+1] == 1`.

Use two binary indicators across all eligible trials. The target belongs to
trial i, whose activity is used to decode the upcoming transition:

| Current choice -> next choice | `left_choice_switch` | `right_choice_switch` |
| --- | ---: | ---: |
| Left -> right | 1 | 0 |
| Right -> left | 0 | 1 |
| Left -> left | 0 | 0 |
| Right -> right | 0 | 0 |

Other valid transitions are negatives for the respective switch indicator.
Do not restrict each decoder to the current chosen side, create separate stay
targets, or add the proposed within-side switch-versus-stay analyses.

The successor is the immediately following unfiltered trial,
consistent with "on the next trial." Do not silently skip a no-choice/manual
trial to the next retained neural window. Missing/invalid successor animal
choices, manual-reward successors and the terminal trial have unavailable
switch labels. A manual-reward successor is excluded even if its action field
contains a side code. A valid successor does not require neural coverage or a
known reward for that future trial. Missing switch labels must not invalidate
otherwise known context, current-choice or outcome labels.

## 5. Population features, eligibility and preprocessing

Use independent PFC/HPC populations, existing good/mua unit and channel-quality
selection, and IRIG-backed complete regional coverage of `[-1,+1)`. Manual
alignment remains deferred. Exclude current manual-reward, no-animal-choice,
missing-choice-time and dark/invalid-context trials. Retain rewarded and
unrewarded choices, including incorrect choices. Do not restrict context/choice
training to correct trials, where their labels may coincide.

Use target-specific missingness: unknown reward excludes the outcome target,
not valid context/choice/switch targets; an unknown successor excludes switch
targets only. Require the complete source and augmented tables to agree on
chronological identity and used source fields. Do not require unrelated
behavior-model value columns or refit behavior models for this target set.

Run both population representations on matched eligible rows and folds:

- Raw-unit population: unsmoothed counts divided by bin duration in seconds,
  giving rates in Hz with `(window,bin,unit)` axes. Standardize each input unit
  using its training-bin mean and population SD (`ddof=0`). "Raw units" here
  means direct unit features, not unscaled fitting inputs.
- PC population: fit PCA on those same standardized training-unit vectors,
  without whitening or subsequent PC-score standardization. Retain all
  numerically nonzero training PCs; do not impose an unrequested leading-PC cap.

For each region, target model and fold, estimate preprocessing from that
model's eligible training bins within its supervised interval only. Fit one
unit scaler and retained-unit mask shared by both representations, then fit
PCA on those standardized training vectors. Apply the saved means, scales,
retained-unit mask and optional PCA unchanged to held-out bins.
Training-constant units are omitted from both representations for that
fit, with their original IDs and exclusion reasons saved. All-constant inputs
make the affected fit unavailable. Held-out activity must not change feature
selection, scaling, PC rank or axes.

Do not reuse descriptive session-wide PCA as the held-out decoder input.
V1 reuses only this decoder's own compatible saved preparation after
identity/geometry validation. Importing other analyses' saved count tensors
is deferred until preparation costs justify the additional adapter.
Fold-local PC ranks and orientations can differ across targets
and folds; they are not the existing fixed session PCs. With all nonzero PCs
retained, the PC comparison does not claim deliberate dimensionality reduction
and may closely match the standardized-unit linear decoder.

No behavior labels, future choices or trial identities enter the neural feature
matrix. Using labels to define evaluation summaries is distinct from feeding
those labels into the decoder. Display-only z-scores are not substituted for
training-fold preprocessing.

## 6. Models and held-out evaluation

For each region and representation, fit five target models: binary context,
binary current choice, one four-class outcome model, and two binary directional
switch models. Use L2 logistic regression with an unpenalized intercept and
fixed `C=1.0`, carrying forward the existing population-readout default. Use
the joint multinomial model for the four outcomes, not four one-versus-rest
fits. No tuning, subsampling, class balancing or class weights are added.

Pool bins without averaging their activity. Give each eligible trial equal
total training weight: distribute weight `1/B` to each of its B supervised
bins. This retains the existing population decoder's per-trial weighting and
keeps total trial weight fixed when bin resolution changes.

Assign deterministic five-fold behavioral-block groups once from the complete
baseline-eligible trial sequence before target masking or regional coverage
filtering. Keep every bin of a trial/block together. Use that assignment across
regions, representations and targets; do not redraw or stratify separately to
obtain favorable class support. Region/target-specific eligibility can reduce
the available rows without changing their fold IDs.

Display and score each eligible trial using the model that held out its block.
Binary decisions use probability 0.5; four-class decisions use the largest
predicted class probability. Primary scores are pooled held-out balanced
accuracy and log loss. Also save ordinary accuracy, confusion counts, per-class
precision/recall, ROC-AUC and average precision with actual class prevalence.
Average precision is the specified precision-recall summary, not an
unspecified area calculation. All bins have equal scoring weight within a
target because complete supervised windows contribute the same bin count.
Do not form primary scores by equally averaging unequal-sized fold scores.

Outcome decoding can succeed by recognizing the chosen side alone. Alongside
the four-state scores, report reward-versus-no-reward discrimination separately
within actual left and right choices, with explicit trial counts. For this
summary compare the two probabilities at that side, normalized by their sum;
this is a summary of the same four-class fit, not extra outcome decoders.
Undefined conditional probabilities remain unavailable with reasons.

Switch trials may be rare. Report class prevalence, positive/negative counts
and an always-negative baseline alongside balanced accuracy and discrimination
scores.
High ordinary accuracy alone does not establish useful switch decoding.
These are directional switch-event readouts, not a test of switching information
independent of the current chosen port.
Post-choice switch decoding can also reflect reward-related activity correlated
with the next choice; it is not proof of a distinct switch-intention signal.

If training lacks either required binary class or any required outcome class,
mark that model/fold unavailable rather than inventing probabilities for an
unseen class. A missing test class can make class-dependent metrics unavailable
without invalidating otherwise valid predictions or log loss. Save class
counts and reasons. Incomplete required held-out predictions leave primary
five-fold scores unavailable; preserve valid completed-fold outputs. Keep
training/reference fits clearly separate and never substitute them for missing
held-out values. No significance labels, permutation tests or causal claims
are part of this specification.

## 7. Saved inspection, reuse and validation

Save population features/row identity once and publish model results separately
from visualization. Save native rates in Hz, training means/SDs in Hz, and
standardized unit inputs in dimensionless SD units. Record PC-score conventions
and the explicit class order for dimensionless probabilities. Preserve original unit IDs, any PC basis,
train/test transforms, trial/bin/time mappings, labels/masks, fold assignments,
class counts, parameters, code/input identities and randomness provenance.
Outputs remain session-local, versioned and non-overwriting.

Expose this analysis as **Population task decoding**, a separate entry in the
webapp's **Plot view** dropdown. Keep **Single unit selectivity** as its own
separate view, with independent controls and saved-result discovery. This is
a routed view within the existing app, not a mode inside the single-unit view.

The viewer should show actual labels and held-out probability tracks on real
session time, preserve gaps, and expose confusion/performance summaries.
Within-unit display z-scores may be available for a neural heatmap, but they
must not silently become decoder training inputs. A fold-specific PCA cannot
be labeled as the same fixed session-PC basis in a contribution plot.

Existing decoding modules already support population matrices, grouped
evaluation, logistic readouts and saved probability/contribution views. The
existing `next_choice_switch_stay` target uses adjacent actions rather than
contexts. The implementation-plan draft records the reuse audit: current
session-prediction targets are trial-level, whereas new target-specific timing,
directional switch indicators and broad no-reward labels require explicit
contracts. Do not change existing analyses' omission meanings or introduce a
general framework.

Required scientific validation should distinguish side-only, generic post-
choice, reward-absence and next-choice signals on synthetic populations.
Include incorrect unrewarded choices, rare switches, omitted/invalid successor
trials, terminal trials, class absence, target-specific missingness and leakage
checks. Tests, synthetic validation, one designated experimental session and
user inspection precede any wider run. No experimental session is designated
for this new analysis yet.

## 8. Readiness

The requested scientific choices and accepted defaults are recorded above.
The [implementation-plan draft](population_task_decoding_implementation_plan.md)
covers audited reuse boundaries, the separate dropdown view, data contracts,
dependencies, tests written first, and performance/storage considerations.
Implementation of that revised plan is approved. An experimental session must
still be designated before running this new analysis on experimental data.
