# Behavioral model benchmark scenarios

## 1. Purpose

This specification defines short, controlled mouse-choice histories for comparing
trial-level behavior models. The scenarios are intended to reveal when each
fixed model corresponds well or poorly to behavior. Poor correspondence is an
informative result, not a failed test.

The scenarios must not be used to select a different model on each trial or to
introduce a latent strategy-switching mechanism whose purpose is to improve
agreement. Each candidate model must use one consistent update rule and one
fixed parameter set throughout a scenario and, when models are compared across
scenarios, throughout the complete benchmark suite.

This document distinguishes two types of evaluation:

1. **Behavioral benchmark:** replay a prescribed behavior sequence and measure
   each model's correspondence with the prescribed choices. There is no required
   winner and no requirement that a model match every choice.
2. **Implementation conformance test:** verify that a model's documented state
   updates and outputs are calculated correctly for a prescribed sequence. A
   conformance failure is a software defect; poor behavioral correspondence is
   not.

The initial suite focuses on reward-triggered probes, reward-count expectancy,
staying and switching after omissions, and return from an incorrect excursion.
It does not specify a new fitting framework or a hidden model-selection process.

## 2. General requirements

### 2.1 Consistent replay

For every candidate model:

- Use the same equations and parameter values on every trial.
- Do not tune parameters separately for individual scenarios.
- Reset model state only at the start of an independent scenario.
- Compute the prediction for trial `t` using observations through `t-1` only.
- Update model state only after the prescribed choice and outcome on trial `t`
  have been observed.
- Replay the prescribed choices; do not replace them with choices sampled from
  the model.
- Record probabilities or continuous values directly. Random choice sampling is
  unnecessary for benchmark replay.

A model may have causally updated memory, such as reward count or omission
doubt. Such memory is part of a consistent model and is not equivalent to
selecting among strategies trial by trial.

### 2.2 Observable and hidden information

The benchmark table may contain the objectively correct side and whether a
choice was correct, but these are hidden task annotations in an uncued block.
They must not be supplied to a model unless that model is explicitly defined to
receive a cue containing that information.

By default, a behavioral model may receive:

- previous valid choices;
- previous observed rewards or omissions;
- fixed task parameters that the model is documented to know; and
- its own causally updated internal state.

By default, it may not receive:

- the experimenter's current correct-side label;
- whether an unrewarded choice was objectively correct or incorrect;
- a future choice or outcome; or
- a benchmark case identifier.

This distinction is especially important because a correct stochastic omission
and an incorrect unrewarded choice can produce the same observation for the
mouse.

### 2.3 Task assumptions

Unless a scenario states otherwise:

- There are two choices, left and right.
- Exactly one side is objectively correct on each trial.
- A correct choice may be rewarded or may undergo a stochastic omission.
- An incorrect choice is unrewarded, matching an inactive-side reward
  probability of zero.
- The correct side remains fixed for the entire scenario.
- Blocks are uncued, so an explicit reversal marker is not shown to the model.

Scenarios prescribe realized outcomes. They do not assert that every listed
sequence is equally likely under the task's reward schedule.

### 2.4 Fixed parameter configuration

Canonical parameters are defined in Section 10. The implementation must also
accept one optional validated JSON parameter override at the benchmark-runner
boundary. The resolved parameters must be applied unchanged to every selected
sample and recorded in the report. Per-sample overrides are forbidden.

The JSON document is an object whose top-level keys are model IDs. Each model
value is a partial object: omitted fields retain their canonical values. The
accepted override groups and fields are:

| Model ID | Accepted fields |
|---|---|
| `simple_probe_persistence` | `persistence_weight`, `probe_weight` |
| `expectancy_persistence_doubt` | `expectancy_threshold`, `expectancy_scale`, `doubt_lambda`, `persistence_weight`, `expectancy_weight`, `doubt_weight` |
| `probe_persistence_doubt` | `persistence_weight`, `probe_weight`, `doubt_weight`, `doubt_lambda` |
| `direct_expectancy_switch_doubt` | `expectancy_threshold`, `expectancy_scale`, `minimum_switch_probability`, `switch_probability_span`, `omission_persistence_weight`, `doubt_weight`, `doubt_lambda` |
| `global_reward_expectancy_doubt` | `reward_decay`, `expectancy_threshold`, `expectancy_scale`, `minimum_switch_probability`, `switch_probability_span`, `omission_persistence_weight`, `doubt_weight`, `doubt_lambda` |
| `home_referenced_expectancy_doubt` | `expectancy_threshold`, `expectancy_scale`, `expectancy_weight`, `doubt_weight`, `doubt_lambda` |

Validation rules are:

- unknown model IDs and fields are rejected;
- booleans are rejected where a number is required;
- all numeric values must be finite;
- weights and expectancy thresholds must be nonnegative;
- expectancy scales and doubt lambdas must be greater than zero;
- decay and individual probability values must be in `[0, 1]`; and
- `minimum_switch_probability + switch_probability_span` must not exceed one.

Reference policies, the history observer, forgetting-Q, and the two threshold
configurations do not accept overrides in the initial implementation. This
keeps the first override surface focused on the expectant-switching models that
motivated parameter inspection. Changing a fixed policy or a threshold
configuration requires a new model ID or a later specification revision.

For example, the following changes only the full existing exemplar and applies
the result to every selected sample:

```json
{
  "expectancy_persistence_doubt": {
    "persistence_weight": 1.0,
    "expectancy_weight": 1.5,
    "doubt_weight": 1.2
  }
}
```

The saved configuration contains the fully merged values for every model, not
only the supplied override fragment.

## 3. Notation and data contract

### 3.1 Compact notation

Each token represents one completed valid trial:

| Token | Meaning |
|---|---|
| `L+` | Chose left; left was correct; reward was delivered. |
| `R+` | Chose right; right was correct; reward was delivered. |
| `Lc0` | Chose left; left was correct; reward was omitted. |
| `Rc0` | Chose right; right was correct; reward was omitted. |
| `Li0` | Chose left; left was incorrect; no reward was delivered. |
| `Ri0` | Chose right; right was incorrect; no reward was delivered. |
| `[invalid]` | No valid choice/outcome pair; model state must not update. |
| `=>` | The prediction immediately before the following trial is a highlighted checkpoint. |
| `| reversal to L/R |` | The hidden correct side changes; the uncued model is not notified directly. |

The outcome attached to a token is observed only after the pre-trial prediction
for that token has been emitted. For example, in `L+ => Ri0`, the model predicts
the choice on the `Ri0` trial using the preceding `L+`, without access to the
`Ri0` outcome.

### 3.2 Tabular representation

When these scenarios are later represented in code or data files, one scenario
must contain `n_trials` aligned rows with at least:

| Field | Type and shape | Meaning |
|---|---|---|
| `scenario_id` | string scalar repeated for all rows | Stable identifier from this specification. |
| `trial_index` | integer, shape `(n_trials,)` | Zero-based index within the independent scenario. |
| `correct_side` | integer, shape `(n_trials,)` | Hidden annotation: `0=right`, `1=left`. |
| `action` | integer, shape `(n_trials,)` | Prescribed choice: `0=right`, `1=left`. |
| `correct` | boolean, shape `(n_trials,)` | Whether `action` equals `correct_side`. |
| `reward` | float, shape `(n_trials,)` | Realized reward in task reward units; initially `0.0` or `1.0`. |
| `valid` | boolean, shape `(n_trials,)` | Whether the row contains a valid choice/outcome pair. |
| `checkpoint` | boolean, shape `(n_trials,)` | Whether the pre-trial prediction is highlighted by `=>`. |

`correct_side` and `correct` are scoring and audit metadata, not default model
inputs. All model outputs must retain the same `(n_trials,)` row alignment, with
the prediction on row `t` representing history only through row `t-1`.

## 4. Required atomic scenarios

The atomic scenarios span the basic reward/omission by stay/switch combinations.
They isolate immediate outcome sensitivity before longer histories are added.

### AT-01: Initial state

```text
=> L+
```

Purpose: record each model's documented initial prediction before any valid
history. A symmetric history-dependent model should normally be neutral, but a
model with an explicit fixed side bias may differ if that bias is part of its
definition.

### AT-02: Reward followed by stay

Correct side: left.

```text
L+ => L+
```

Purpose: canonical win-stay behavior and a negative example for a pure
post-reward switch probe.

### AT-03: Reward followed by switch

Correct side: left.

```text
L+ => Ri0
```

Purpose: canonical one-reward probe. The switch is away from the rewarded,
correct side.

### AT-04: Correct omission followed by stay

Correct side: left.

```text
Lc0 => L+
```

Purpose: persistence after a stochastic omission on the correct side.

### AT-05: Correct omission followed by switch

Correct side: left.

```text
Lc0 => Ri0
```

Purpose: switch following an omission that was objectively correct. The
immediate reward probe must be zero.

### AT-06: Incorrect omission followed by switch

Correct side: right.

```text
Li0 => R+
```

Purpose: canonical loss-switch behavior after sampling the incorrect side.

### AT-07: Incorrect omission followed by stay

Correct side: right.

```text
Li0 => Li0
```

Purpose: perseveration on an unrewarded incorrect side and a negative example
for immediate lose-switch behavior.

## 5. Required multi-trial probe and return scenarios

### PR-01: Three-reward probe and three-omission return

Correct side: left throughout.

```text
L+ L+ L+ => Ri0 Ri0 Ri0 => L+
```

Purpose: idealized expectant-switching behavior. It combines:

- a switch away after three rewarded correct trials;
- two intervening stays on the incorrect side despite omissions; and
- a return after the third unrewarded incorrect trial.

All trials must be scored. In particular, a model that predicts returning after
the first omission should correspond poorly with the two prescribed incorrect
stays even if it corresponds with the eventual return.

Required state checkpoints:

- immediately before the initial switch;
- after each of the three incorrect omissions; and
- immediately before the return.

### PR-02: One-reward probe and one-omission return

Correct side: left throughout.

```text
L+ => Ri0 => L+
```

Purpose: fastest reward-probe and omission-return sequence. This is the
low-threshold counterpart to `PR-01`.

### PR-03: Switch after a correct omission

Correct side: left throughout.

```text
L+ Lc0 => Ri0
```

Purpose: isolate switching whose immediately preceding outcome was not a
reward. A reward-triggered probe must be zero at the checkpoint. Any switching
pressure must arise from another documented component.

### PR-04: Accumulated correct omissions before switching

Correct side: left throughout.

```text
L+ Lc0 Lc0 Lc0 => Ri0
```

Purpose: test whether repeated stochastic omissions can overcome persistence
even though the selected side remains objectively correct.

## 6. Required reward-expectancy scenarios

### EX-01: Rewards separated by correct omissions

Correct side: left throughout.

```text
L+ Lc0 L+ Lc0 L+ => Ri0
```

Purpose: distinguish total reward count, consecutive reward count, leaky reward
integration, and omission-sensitive evidence. The current reward-confirmed
expectancy definition counts the three rewards and preserves that count across
the intervening omissions.

### EX-02: High expectancy without switching

Correct side: left throughout.

```text
L+ => L+ => L+ => L+ => L+ => L+
```

Purpose: required negative control. Reward count and expectancy become high,
but the prescribed behavior continues to stay. A model that consistently
predicts a high-count probe should intentionally show poor correspondence late
in this sequence.

### EX-03: Rewarded switch followed by adoption

The correct side changes from left to right before the marked switch.

```text
L+ L+ L+ | reversal to R | => R+ => R+ R+
```

Purpose: test whether a model adopts a newly rewarded side after an initially
successful probe. The reversal marker is hidden from an uncued model.

### EX-04: Rewarded switch followed by immediate rejection

The correct side changes from left to right before the marked switch.

```text
L+ L+ L+ | reversal to R | => R+ => Li0 => R+
```

Purpose: test continued switching after a successful rewarded probe. This is
the synthetic counterpart of behavior in which reward on the newly selected
side does not terminate sampling.

## 7. Required repeated-sampling scenario

### RS-01: Repeated reward-switch and omission-return loop

Correct side: right throughout.

```text
R+ => Li0 => R+ => Li0 => R+ => Li0
```

Purpose: represent repeated departure from the rewarded correct side followed
by return after an incorrect omission. This pattern directly tests whether:

- a reward probe repeatedly promotes departure;
- an omission component promotes return; and
- a reward-count state resets or remains elevated when choices alternate.

The model remains fixed throughout the loop. High correspondence for one part
of the loop and low correspondence for another must be reported rather than
hidden by selecting another model.

## 8. Additional plotted samples and conformance cases

`RS-02`, `RS-03`, `CT-01`, `CT-02`, `CT-03A`, and `CT-03B` are part of the
default plotted suite. They add longer persistence, non-reward-dependent
alternation, and hidden-context controls. `IV-01` through `IV-04` are
implementation conformance cases and do not receive routine atlas subplots.

### RS-02: Persistent incorrect sampling before return

Correct side: left throughout.

```text
L+ => Ri0 Ri0 Ri0 Ri0 => L+
```

Purpose: distinguish immediate lose-switch from gradually accumulating doubt
and persistent incorrect-side choice.

### RS-03: Alternation without continuing reward triggers

Correct side: left throughout.

```text
L+ => Ri0 => Lc0 => Ri0 => L+
```

Purpose: separate general alternation or transition momentum from a strictly
reward-triggered probe.

### CT-01: True reversal after negative evidence

The correct side changes from left to right immediately before the omitted left
choice.

```text
L+ L+ L+ | reversal to R | Li0 => R+
```

### CT-02: Stochastic correct omission without reversal

Correct side: left throughout.

```text
L+ L+ L+ Lc0 => Ri0
```

`CT-01` and `CT-02` are a paired observability control. Before the marked
choice, the mouse and any uncued model have observed the same action/reward
history:

```text
L+ L+ L+ L0
```

The hidden correctness of the final `L0` differs, but the model state and
pre-trial output at the checkpoint must be identical unless the model received
an explicit informative cue. The different outcome after choosing right may
affect later updates.

### CT-03A and CT-03B: Same prospective switch under different hidden contexts

`CT-03A` retains the left-correct context:

```text
L+ L+ L+ => Ri0
```

`CT-03B` reverses the hidden context immediately before the same prospective
right choice:

```text
L+ L+ L+ | reversal to R | => R+
```

The two samples therefore share the observed pre-switch history:

```text
L+ L+ L+ => R
```

The pre-switch prediction must be identical in both variants. The model may
diverge only after observing the right-choice outcome.

### IV-01: Left/right mirror invariance

Generate a mirrored version of every default plotted sample by exchanging left
and right. For a symmetric model with symmetric parameters:

- switch probability must be unchanged;
- probability of the mirrored observed action must be unchanged;
- signed left/right predictors and values must reverse sign; and
- unsigned state magnitudes must remain unchanged.

A model with an explicitly documented side-bias parameter is exempt only from
the probability symmetry assertions, not from correct mirroring of inputs and
sign conventions.

### IV-02: Invalid trial continuity

Correct side: left throughout.

```text
L+ [invalid] => Ri0
```

Compare against:

```text
L+ => Ri0
```

Purpose: verify that an invalid or no-choice row preserves alignment but does
not update or erase model state. The prediction at the valid `Ri0` trial must
match between the two histories under the repository's previous-valid-trial
continuity convention.

### IV-03: Independent-session reset

Replay any nontrivial scenario, start a new independent scenario, and then
replay `AT-01`. The initial state and prediction must not depend on the earlier
scenario.

### IV-04: Positive reward magnitude

If the model defines reward as binary, replay a scenario after replacing reward
`1.0` by another positive magnitude. State updates and predictions must be
identical. If a future model explicitly uses reward magnitude, it must document
the units and expected difference instead.

## 9. Required outputs and correspondence measures

At minimum, benchmark replay must retain the following on every aligned trial:

### 9.1 Observed behavior

- prescribed action;
- reward;
- hidden correct side and correctness annotation;
- previous valid action and outcome;
- switch/stay indicator;
- outcome-transition category, including reward-to-switch,
  reward-to-stay, omission-to-switch, and omission-to-stay; and
- whether an omission was objectively correct or incorrect, for retrospective
  stratification only.

### 9.2 Model output

- all interpretable internal states specified by the model;
- pre-link decision drive, when applicable;
- signed left/right model value;
- left and right choice probabilities;
- probability of switching relative to the previous valid action.

For a previous-left choice,

```text
P(switch) = P(right)
```

and for a previous-right choice,

```text
P(switch) = P(left).
```

`P(switch)` is undefined before the first valid choice and should be represented
as missing rather than silently assigned to zero.

Models defined directly by choice or switch probability need not manufacture
a pre-link decision drive. Their trialwise decision-drive field is missing on
valid rows, while their probabilities and common model value remain required.

The probability assigned to the prescribed observed action is derived only
after the pre-trial model prediction has been produced:

```text
P(observed) = P(left)  when the prescribed action is left
P(observed) = P(right) when the prescribed action is right
```

It is a current-action-conditioned evaluation field, not part of model replay
state, and it must never be used to update a model. It remains required in the
trialwise audit table and summary calculations.

### 9.3 Summary measures

Report at least:

- trialwise probability of the observed action;
- trialwise switch probability;
- mean negative log probability of the observed choices;
- number of observed-choice probabilities floored at `1e-12` only for the
  negative-log-probability calculation;
- Brier score for switch/stay predictions where a previous choice exists; and
- sample-as-block greedy agreement, with exact ties contributing `0.5` and
  retained as explicit ties in the trialwise output.

For eligible trials, the two probabilistic scores are:

```text
mean_NLL = mean(-log(max(P(observed), 1e-12)))
switch_Brier = mean((P(switch) - observed_switch)^2)
```

The NLL includes every valid choice. The Brier score excludes the first valid
choice because switch/stay is undefined there, and it excludes invalid rows.

Summary scores must never replace the trialwise trace. The intended use is to
see where a model becomes a good or poor description within each prescribed
pattern.

### 9.4 Required per-model behavior-atlas figures

Produce one or two wide, light-mode multipanel figure pages for each model.
Each selected behavior sample must appear exactly once as a subplot in that
model's atlas. Place up to 12 samples on one page and balance larger selections
across two pages. The default 22-sample suite therefore uses two pages with 11
samples per page for every model.

Every behavior-sample subplot must place observed behavior and model value on
the same signed axis:

- trial index is the horizontal axis;
- `-1` denotes right and `+1` denotes left;
- model value is a thin continuous line;
- observed choices are unconnected circle markers at `-1` or `+1`;
- rewarded choices use a larger marker than unrewarded choices; and
- markers must be drawn above the value trace so that behavior remains visible.

Do not connect observed choices with a line. Do not draw correct side as a
continuous value trace. Instead, label the initial correct-side context `L` or
`R` near the beginning of the primary panel. Whenever correct side changes,
draw a vertical dashed line at the transition and label the newly active
context `L` or `R`. Hidden correct side and correctness remain available in the
retained trialwise data for scoring and audit.

The atlas does not require separate probability or internal-state panels.
Those values remain required trialwise outputs under Section 9.2 and remain
available for numerical inspection. Each subplot must identify its behavior
sample. Each page title or caption must identify the model and fixed parameter
set.

### 9.5 Required combined sample-as-block agreement figure

Treat each complete behavior sample as exactly one block. Checkpoints and
phases within a sample do not create additional blocks.

Produce one combined figure with:

- behavior-sample ID on the horizontal axis;
- block agreement in `[0, 1]` on the vertical axis;
- every required model shown on the same axes; and
- a horizontal `0.5` reference for chance-level greedy agreement.

For each valid trial, greedy agreement is `1` when the uniquely preferred side
matches the prescribed choice and `0` when it does not. An exact model tie
contributes `0.5` and must also remain explicitly marked as a tie in the
trialwise output. The block score is the arithmetic mean across every valid
trial in that sample. This avoids assigning a biologically meaningless side to
a neutral first-trial prediction or another exact tie.

The combined agreement figure is a compact descriptive comparison, not the
primary probabilistic evaluation. Negative log probability, Brier score, and
the aligned trialwise probabilities remain required outputs.

### 9.6 Saved report artifacts

The benchmark report must save:

- the per-model atlas pages;
- the combined sample-as-block agreement figure;
- `trialwise_model_outputs.csv` with one row per model/sample/trial;
- `scenario_model_scores.csv` with one row per model/sample pair;
- `model_parameters.json` containing the resolved fixed configuration;
- a human-readable Markdown summary; and
- a run log.

`trialwise_model_outputs.csv` is a derived audit table, not a synthetic session
or saved behavior run. It contains prescribed behavior, common model outputs,
and namespaced model-specific states. Do not write session tables, augmented
trial files, serialized replay objects, outputs beneath an experimental
session directory, or generated report artifacts inside the source repository.

Resolve the repository root and requested output root before testing path
containment. Reject paths whose resolved target is inside the repository,
including `..` and symlink aliases. Create the timestamped run directory
exclusively and refuse an existing target; every artifact must be written below
that resolved run directory.

## 10. Models to evaluate

### 10.1 Shared model conventions

All models use the timing and information constraints in Section 2. The
following notation applies unless a model explicitly states otherwise:

| Symbol / name | Definition |
|---|---|
| `c_prev` | Previous valid choice, encoded `+1=left`, `-1=right`, and `0` before any valid choice. |
| `r_prev` | Previous valid outcome, encoded `1` for any positive reward and `0` for omission. |
| `decision_drive` | Model-specific left-positive policy drive before conversion to choice probability. This was previously abbreviated `S`; it does not mean sides. |
| `reward_probe` | Signed post-reward switch signal, `-c_prev * r_prev`. This was previously abbreviated `G`. |
| `expectancy_strength` | Scalar in `[0, 1]` produced by a documented reward-memory rule. This was previously abbreviated `E`. |
| `V` | Common bounded left-positive model value, defined as `P(left) - P(right)`, in `[-1, 1]`. |

For every model:

```text
V = P(left) - P(right) = 2 * P(left) - 1
```

Models with the direct tanh link use:

```text
V = tanh(decision_drive)
P(left) = (1 + V) / 2
P(right) = (1 - V) / 2
```

The existing history ideal observer and forgetting-Q policy instead retain
their implemented logistic link:

```text
P(left) = expit(decision_drive)
P(right) = 1 - P(left)
V = 2 * P(left) - 1 = tanh(decision_drive / 2)
```

Thus `decision_drive` is not assumed to have the same scale across model
families, while `V` always has the same probability-derived meaning.

For a trial with a previous valid choice:

```text
P(switch) = (1 - c_prev * V) / 2
```

If a model specifies `P(switch)` directly, convert it to a signed value using:

```text
V = c_prev * (1 - 2 * P(switch))
```

Before the first valid choice, history-dependent models emit `V=0` and
`P(left)=P(right)=0.5`, unless a fixed side bias is an explicit part of the
model. None of the initial benchmark models has such a bias.

Hard-policy references may emit probabilities of zero or one. For negative log
probability only, use `max(P(observed), 1e-12)` and report the number of trials
whose raw probability was below that floor. Stored raw probabilities and all
other scores must remain unchanged.

### 10.2 Required reference models

These simple policies establish interpretable comparison points. They are
complete fixed models rather than isolated regressors.

#### REF-01: `chance`

On every trial:

```text
V = 0
P(left) = P(right) = 0.5
```

The model has no state beyond session initialization and ignores all outcomes.

#### REF-02: `immediate_persistence`

After a valid choice:

```text
V = c_prev
```

It always predicts staying. Reward and omission do not affect its policy.

#### REF-03: `wsls`

This is the existing `wsls_regressor` policy:

```text
if r_prev == 1: V = c_prev
if r_prev == 0: V = -c_prev
```

It predicts win-stay and lose-switch. The first valid trial is neutral.

#### REF-04: `win_switch_lose_stay`

This is the outcome-reversed counterpart to WSLS:

```text
if r_prev == 1: V = -c_prev
if r_prev == 0: V = c_prev
```

It is the simplest complete model of a constant post-reward probe: reward
predicts switching, while omission predicts staying.

#### REF-05: `oracle_correct_side`

The oracle emits `V=+1` when the hidden correct side is left and `V=-1` when it
is right. It is the only benchmark reference allowed to read `correct_side`.
It is a task-performance ceiling, not a plausible uncued behavioral model, and
must be labeled as such in every table and plot. Its switch probability is
derived relative to the previous valid observed choice, just as for every other
model; the current choice and outcome are not used in its pre-trial prediction.

### 10.3 Required existing models

#### EXIST-01: `simple_probe_persistence`

Use the implemented `SimpleProbePersistenceAgent` / offline
`simple_probe_persistence` replay without changing its equations:

```text
decision_drive = w_P * c_prev + w_R * reward_probe
V = tanh(decision_drive)
```

Canonical benchmark parameters:

- `w_P=1.0`
- `w_R=1.0`

The equal-weight configuration is intentionally indifferent after reward. It
must not be retuned separately for scenarios containing reward-switch behavior.

#### EXIST-02: `expectancy_persistence_doubt`

Use the implemented `ExpectancyPersistenceDoubtAgent` / offline
`expectancy_persistence_doubt` replay and the reward-confirmed-side and
counterfactual-doubt update rules in
`expectant_switching_exemplar_spec.md`:

```text
expectant_switch = reward_probe * expectancy_strength
decision_drive = w_P * c_prev + w_E * expectant_switch + w_D * doubt_choice
V = tanh(decision_drive)
```

Canonical benchmark parameters:

- expectancy midpoint `3.0` rewarded trials;
- expectancy scale `1.0` rewarded trials;
- doubt lambda `0.5` inverse omission trials;
- `w_P=1.0`;
- `w_E=1.0`; and
- `w_D=1.0`.

This initial configuration can approach indifference but cannot prefer a
post-reward switch using expectancy alone. That limitation is part of the
model being benchmarked.

#### EXIST-03: `history_ideal_observer`

Use the existing mouse-history ideal observer represented by `observer_value`
and `HMM_reward_decay_relative_doubt`. It is one fixed environment-belief
model, not a GLM-HMM that selects behavioral strategies.

Canonical benchmark parameters are:

- believed context transition probability `0.2` per trial;
- active-side reward probability `0.8`;
- inactive-side reward probability `0.0`;
- HMM reward-decay lambda `0.1`;
- counterfactual-doubt lambda `0.5`;
- log-odds tanh scale `1.0`;
- logistic action-history bias `0.0`; and
- action temperature `1.0`.

The observer receives only the prescribed choices and outcomes. The benchmark
correct-side sequence is used to score it, not to update it. Its pre-trial
decision drive is the implemented reward-decay belief value minus raw relative
doubt. Its choice probability and common bounded value use the logistic link
in Section 10.1, matching the existing executable agent.

#### EXIST-04: `forgetting_q_learning`

Use the existing `FQlearning_rel_value` update with one explicitly documented
benchmark readout. Initialize `Q_right=Q_left=0`. Before trial `t`:

```text
decision_drive = Q_left - Q_right
P(left) = expit(decision_drive)
V = 2 * P(left) - 1
```

After a valid observed action `a` and reward `r`:

```text
Q_right = 0.7 * Q_right
Q_left = 0.7 * Q_left
Q_a = Q_a + 0.3 * r
```

Thus decay is `0.7`, the reward update rate is `0.3`, logistic action-history
bias is `0.0`, and action temperature is `1.0`. Standard Q-learning and the
fast-learning forgetting-Q variant are optional secondary comparisons, not
members of the required initial roster.

### 10.4 Shared counterfactual doubt for proposed models

Four proposed model families below reuse the existing counterfactual doubt
state.
Maintain nonnegative effective omission counts `n_left` and `n_right`, both
initialized to zero. After every valid trial:

- an omission increments only the chosen side's count by one;
- any positive reward resets both counts to zero; and
- invalid rows do not update either count.

Before trial `t`, with canonical `lambda=0.5`:

```text
doubt_raw = (1 - exp(-lambda * n_left))
            - (1 - exp(-lambda * n_right))
doubt_choice = -doubt_raw
```

`doubt_raw` points toward the more doubted side. `doubt_choice` points toward
the choice preferred by this omission rule and is the quantity added to a
left-positive decision drive.

### 10.5 Proposed model 1: `probe_persistence_doubt`

This model tests whether a constant post-reward probe plus omission-driven
return is sufficient, without reward-count expectancy.

Before every valid trial:

```text
persistence = c_prev
reward_probe = -c_prev * r_prev
decision_drive = 1.0 * persistence
                 + 2.0 * reward_probe
                 + 1.4 * doubt_choice
V = tanh(decision_drive)
```

Canonical parameters are therefore:

- persistence weight `1.0`;
- constant probe weight `2.0`;
- doubt weight `1.4`; and
- doubt lambda `0.5`.

After reward, doubt has been reset and the model assigns approximately `0.881`
probability to switching. Following omissions on the currently chosen side, it
assigns approximately `0.289`, `0.443`, and `0.544` switch probability after
omissions one, two, and three. It therefore becomes return-preferring only
after approximately three omissions, but it probes after the first reward and
does not wait for three rewards.

### 10.6 Proposed model 2: `direct_expectancy_switch_doubt`

This model retains side-specific reward-confirmed expectancy but maps high
expectancy directly to switch probability.

#### Reward-confirmed state

Maintain `confirmed_side` and scalar `reward_count`, initialized to `None` and
zero. After a valid trial:

- the first reward on side A sets `confirmed_side=A` and `reward_count=1`;
- another reward on A increments `reward_count`;
- omission on either side preserves both fields; and
- reward on the opposite side B sets `confirmed_side=B` and resets
  `reward_count=1`.

The count is a scalar attached to the currently confirmed side, not two
lifetime side totals.

#### Reward branch

Canonical parameters are:

- `expectancy_threshold=2.5` rewarded trials;
- `expectancy_scale=0.5` rewarded trials;
- `minimum_switch_probability=0.1`;
- `switch_probability_span=0.8`;
- `omission_persistence_weight=1.0`;
- `doubt_weight=1.4`; and
- `doubt_lambda=0.5` inverse omission trials.

Before trial `t`, calculate:

```text
expectancy_strength =
    (1 + tanh((reward_count - expectancy_threshold)
              / expectancy_scale)) / 2
```

If `r_prev == 1`, use:

```text
P(switch) = minimum_switch_probability
            + switch_probability_span * expectancy_strength
V = c_prev * (1 - 2 * P(switch))
```

The midpoint `2.5` makes count two stay-preferring and count three
switch-preferring. Approximate switch probabilities at reward counts one, two,
and three are `0.102`, `0.195`, and `0.805`.

#### Omission branch

If `r_prev == 0` and a previous choice exists, use:

```text
decision_drive = omission_persistence_weight * c_prev
                 + doubt_weight * doubt_choice
V = tanh(decision_drive)
```

Thus the omission response and doubt parameters match proposed model 1. Before
any valid choice, emit the neutral initial output specified in Section 10.1.

### 10.7 Proposed model 3: `global_reward_expectancy_doubt`

This model deliberately uses a side-independent reward accumulator. Probe
direction is always opposite the immediately previous choice, so no side label
is required to determine direction.

Canonical parameters are:

- `reward_decay=0.8` per valid trial;
- `expectancy_threshold=2.0` accumulator units;
- `expectancy_scale=0.5` accumulator units;
- `minimum_switch_probability=0.1`;
- `switch_probability_span=0.8`;
- `omission_persistence_weight=1.0`;
- `doubt_weight=1.4`; and
- `doubt_lambda=0.5` inverse omission trials.

Initialize scalar `reward_accumulator=0`. After each valid trial `t`, update:

```text
reward_accumulator_next =
    reward_decay * reward_accumulator + reward_indicator_t
```

Reward increases the accumulator regardless of the chosen side. Omission
causes passive decay. A reward after switching sides does not reset it.

Before trial `t`, calculate:

```text
expectancy_strength =
    (1 + tanh((reward_accumulator - expectancy_threshold)
              / expectancy_scale)) / 2
```

If `r_prev == 1`, use:

```text
P(switch) = minimum_switch_probability
            + switch_probability_span * expectancy_strength
V = c_prev * (1 - 2 * P(switch))
```

With decay `0.8`, the accumulator after three consecutive rewards is `1.0`,
`1.8`, and `2.44`. The corresponding switch probabilities are approximately
`0.114`, `0.348`, and `0.782`.

If `r_prev == 0`, use the same omission branch as proposed model 2:

```text
decision_drive = omission_persistence_weight * c_prev
                 + doubt_weight * doubt_choice
V = tanh(decision_drive)
```

The scientific contrast with proposed model 2 is the reset rule: model 2
resets reward count when the rewarded side changes, whereas model 3 continues
integrating rewards across sides.

### 10.8 Proposed model 4: `threshold_probe_return`

This is a fixed excursion-state model, not a latent model-selection procedure.
Its update rules and parameters remain unchanged throughout the benchmark.

#### Canonical configurations

Evaluate two separately named, fixed configurations across the complete suite:

| Model ID | Reward threshold `K_R` | Excursion-omission threshold `K_O` |
|---|---:|---:|
| `threshold_probe_return_3_3` | 3 | 3 |
| `threshold_probe_return_1_1` | 1 | 1 |

Neither configuration may be selected case by case. A policy-prescribed action
receives probability `0.9`; its alternative receives probability `0.1`.

#### State

Maintain:

- `home_side`, initialized to `None`;
- `home_reward_count`, initialized to zero;
- `in_excursion`, initialized to `False`;
- `excursion_side`, initialized to `None`; and
- `excursion_omission_count`, initialized to zero.

Before the first reward, the model is neutral. The first rewarded choice becomes
`home_side` and sets `home_reward_count=1`.

#### Prediction outside an excursion

When `home_side` is known and `in_excursion=False`:

- if the previous trial was rewarded and `home_reward_count >= K_R`, predict
  the side opposite `home_side` with probability `0.9`;
- otherwise predict `home_side` with probability `0.9`.

A home-side omission preserves `home_reward_count` but does not itself trigger
a reward probe. If the mouse stays home when the model predicts a probe, score
the mismatch and continue applying the same rule.

#### Updating after departure from home

When the observed mouse choice leaves `home_side`:

- if the choice is unrewarded, enter an excursion, set `excursion_side` to the
  observed side, set `excursion_omission_count=1`, and reset
  `home_reward_count=0`;
- if the choice is rewarded, adopt it immediately as the new `home_side`, set
  `home_reward_count=1`, and remain outside an excursion.

The model uses reward, not hidden correctness, to distinguish these outcomes.

#### Prediction and update during an excursion

While `in_excursion=True`:

- if `excursion_omission_count < K_O`, predict staying on `excursion_side`
  with probability `0.9`;
- otherwise predict returning to `home_side` with probability `0.9`.

After observing the mouse:

- another omission on `excursion_side` increments the omission count;
- reward on `excursion_side` adopts it as the new home with reward count one
  and ends the excursion;
- observed return to `home_side` ends the excursion; and
- a rewarded return begins the new home reward count at one.

An early return or an overlong excursion is recorded as poor correspondence;
the model does not change its thresholds in response.

### 10.9 Proposed model 5: `home_referenced_expectancy_doubt`

This model combines persistent home-referenced reward expectancy and
side-specific omission doubt on every trial. Reward and omission update the
two component states; they do not select mutually exclusive decision rules.

Reuse `RewardConfirmedExpectancyState` and the shared counterfactual doubt
state. A reward on the confirmed side increments its reward count. A reward on
the opposite side makes that side the new confirmed home with reward count one.
Any reward clears both omission counts. An omission preserves the confirmed
home and reward count while incrementing only the chosen side's omission
count.

With `home_sign=+1` for left and `-1` for right, calculate:

```text
expectancy_strength =
    (1 + tanh((reward_count - expectancy_threshold)
              / expectancy_scale)) / 2
expectancy_choice = home_sign * (1 - 2 * expectancy_strength)

decision_drive = expectancy_weight * expectancy_choice
                 + doubt_weight * doubt_choice
V = tanh(decision_drive)
P(left) = (1 + V) / 2
```

Before the first reward, `expectancy_choice=0`; omission doubt may still
contribute. There is no generic previous-choice persistence term. Below the
expectancy threshold the expectancy component points home, while above the
threshold it points to the opposite/probe side. Doubt always points away from
the side with more accumulated omissions.

Canonical parameters are:

- `expectancy_threshold=2.5` rewarded trials;
- `expectancy_scale=0.5` rewarded trials;
- `expectancy_weight=1.0`;
- `doubt_weight=1.0`; and
- `doubt_lambda=0.5` inverse omission trials.

These values make reward counts one and two home-preferring and count three
probe-preferring. Following a below-threshold failed probe, expectancy and
doubt both favor immediate return. Following an above-threshold failed probe,
expectancy initially favors remaining on the probe while doubt grows toward
home; the greedy preference changes to home after the third consecutive probe
omission. A rewarded probe becomes the new home and restarts its reward count
at one.

### 10.10 Models excluded from the initial roster

The initial benchmark does not include:

- trial or block GLM-HMM strategy models;
- any mechanism that selects a different behavioral model by trial;
- a model fitted separately to each benchmark scenario;
- raw component signals presented as if they were complete choice models;
- objective-correctness-triggered policies unavailable to an uncued mouse; or
- the previously suggested reward-transition-momentum model, which does not
  yet have sufficient motivation to justify another benchmark family.

## 11. Benchmark interpretation by model family

The suite must not declare one expected winner, but the following comparisons
motivate its construction:

| Model property | Most informative scenarios |
|---|---|
| Constant post-reward probe | `AT-02`, `AT-03`, `PR-02`, `EX-02`, `RS-01` |
| Reward-count expectancy | `PR-01`, `PR-02`, `EX-01`, `EX-02`, `EX-03` |
| Side-specific expectancy reset | `EX-03`, `EX-04`, `RS-01` |
| Side-independent reward integration | `EX-02`, `EX-04`, `RS-01` |
| Persistent home-referenced expectancy plus doubt | `PR-01`, `PR-02`, `EX-01`, `EX-03`, `RS-01`, `CT-01`, `CT-02` |
| Omission-driven doubt | `AT-04` through `AT-07`, `PR-01`, `PR-03`, `PR-04`, `RS-02` |
| WSLS-like outcome response | `AT-02` through `AT-07`, `PR-02` |
| Explicit threshold probe/return | `PR-01`, `PR-02`, `PR-04`, `EX-03`, `EX-04` |
| Persistence/perseveration | `AT-02`, `AT-04`, `AT-07`, `EX-02`, `RS-02` |
| General alternation momentum | `RS-01`, `RS-03` |
| Context inference from observable evidence | `EX-03`, `CT-01`, `CT-02`, `CT-03A`, `CT-03B` |

High performance on one row does not authorize changing the model on other
rows. The pattern of successes and failures is the scientific output.

## 12. Implementation conformance checks

Future code implementing this benchmark must be developed test-first. At a
minimum, tests must verify:

1. Explicit scenario arrays match the documented action, correctness, reward,
   checkpoint, and hidden-side sequences.
2. All scenario arrays have matching one-dimensional shapes and stable IDs.
3. Predictions are pre-trial and do not use the current token's outcome.
4. `correct` and `correct_side` are excluded from default model inputs.
5. State resets between independent scenarios.
6. Invalid rows preserve alignment and do not update state.
7. Left/right mirroring reverses signed values and preserves switch-relative
   probabilities for symmetric models.
8. Every default plotted sample produces all required trialwise outputs.
9. A model's parameters remain unchanged across the full suite.
10. Benchmark correspondence failures do not cause implementation tests to
    fail when the documented model calculations are correct.
11. Every required model emits aligned `P(left)`, `P(right)`, and `P(switch)`
    values, and the evaluation layer derives aligned observed-choice
    probabilities without feeding them back into model state.
12. Proposed model reward and omission branches reproduce their documented
    canonical checkpoint probabilities.
13. The `threshold_probe_return_3_3` and `threshold_probe_return_1_1`
    configurations are evaluated as separately named fixed models, never
    selected according to scenario identity.
14. Every evaluated model produces one or two behavior-atlas pages. Every
    selected sample appears exactly once as a subplot that overlays unconnected
    behavior markers and model value, labels the initial correct-side context,
    and uses labeled vertical dashed markers rather than a continuous
    correct-side trace for context changes.
15. Combined agreement treats each complete sample as one block and scores
    exact ties as `0.5` rather than assigning an arbitrary side.
16. The history ideal observer and forgetting-Q adapters reproduce their
    existing logistic action probabilities rather than applying the tanh link
    a second time.
17. Negative log probability uses the `1e-12` floor only for logarithms,
    preserves raw probabilities, and reports the clipped-trial count.
18. A validated JSON override is applied uniformly across all selected samples
    and its resolved values are saved with the report.
19. Unknown configuration keys, booleans used as numbers, non-finite values,
    invalid types, and out-of-range values are rejected before replay.
20. A missing decision drive is accepted only for a model defined directly by
    probability; all common probability and value outputs remain present.
21. For every non-oracle model, identical observable histories through trial
    `t-1` produce identical predictions on trial `t`, regardless of the current
    outcome or hidden context.
22. Invalid-row continuity is verified for every model, not only the simple
    reference policies.
23. Only the separately invoked oracle adapter receives `correct_side`.
24. The saved trialwise audit table retains alignment and leaves an
    inapplicable decision drive missing rather than inventing one.
25. Duplicate scenario IDs, duplicate ordinary-model IDs, and an ordinary model
    using the reserved oracle ID are rejected before replay results are mapped.
26. Output-root containment uses resolved paths, the run directory is created
    exclusively, and every report artifact remains beneath it.

No new dependency is required to represent these scenarios. Plain NumPy arrays
or pandas columns are sufficient if and when executable benchmark support is
implemented.

## 13. Required suite summary

The default plotted benchmark consists of:

- `AT-01` through `AT-07`;
- `PR-01` through `PR-04`;
- `EX-01` through `EX-04`;
- `RS-01` through `RS-03`;
- and `CT-01`, `CT-02`, `CT-03A`, and `CT-03B`.

These 22 named samples are plotted for every required model. `IV-01` through
`IV-04` remain conformance checks rather than routine plotted samples.

The required initial model roster is:

- `chance`;
- `immediate_persistence`;
- `wsls`;
- `win_switch_lose_stay`;
- `oracle_correct_side` as a labeled performance ceiling;
- `simple_probe_persistence`;
- `expectancy_persistence_doubt`;
- `history_ideal_observer`;
- `forgetting_q_learning`;
- `probe_persistence_doubt`;
- `direct_expectancy_switch_doubt`;
- `global_reward_expectancy_doubt`;
- `home_referenced_expectancy_doubt`;
- `threshold_probe_return_3_3`; and
- `threshold_probe_return_1_1`.
