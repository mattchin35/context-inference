# Home-referenced expectancy + doubt: implementation plan

Status: approved for implementation on 2026-10-12.

## 1. Purpose and scope

Add one fixed, causally observable benchmark model that combines a persistent
home-referenced reward expectancy with side-specific omission doubt on every
trial. The new model will be evaluated across the same 22 independent behavior
samples as the existing roster.

The model will be additive. It will not select separate reward and omission
policies, infer a latent model identity, use hidden correct-side annotations, or
change parameters between samples. The existing
`direct_expectancy_switch_doubt` model will remain unchanged so its behavior
can be compared with the new formulation.

Proposed identifiers:

- model ID: `home_referenced_expectancy_doubt`;
- display label: `Home-referenced expectancy + doubt`.

## 2. Model contract

### 2.1 Observable inputs and timing

The model receives the existing `ObservableHistory` fields:

- `action`, integer NumPy array with shape `(n_trials,)`, encoded
  `0=right, 1=left`;
- `reward`, float NumPy array with shape `(n_trials,)`, in task reward units;
- `valid`, Boolean NumPy array with shape `(n_trials,)`.

Every output on row `t` is a pre-trial prediction based only on valid rows
through `t-1`. Hidden `correct_side` and `correct` values are never inputs.
Every behavior sample starts with fresh model state.

### 2.2 State and updates

Reuse `RewardConfirmedExpectancyState`:

- `confirmed_side`: the most recently rewarded side, or `None` before the
  first reward;
- `reward_count`: rewards accumulated during the current confirmed-side
  episode.

Reuse `CounterfactualDoubtState`:

- an omission increments only the chosen side's omission count;
- any positive reward resets both omission counts;
- invalid rows update neither state.

A positive reward updates expectancy as follows:

- reward on `confirmed_side` increments `reward_count`;
- reward on the opposite side makes that side the new `confirmed_side` and
  resets `reward_count` to one.

An omission preserves `confirmed_side` and `reward_count`.

### 2.3 Continuous combined readout

Use left-positive side signs:

```text
home_sign = +1 for left, -1 for right
```

Calculate the existing continuous reward-count curve:

```text
expectancy_strength =
    (1 + tanh((reward_count - expectancy_threshold)
              / expectancy_scale)) / 2
```

Convert it to a bipolar home-relative preference:

```text
expectancy_choice = home_sign * (1 - 2 * expectancy_strength)
```

Therefore:

- below threshold, `expectancy_choice` points toward home;
- at the threshold, it approaches zero;
- above threshold, it points toward the opposite/probe side.

Before the first reward, `expectancy_choice=0`. Omission doubt may still
contribute after an unrewarded initial choice.

Reuse the existing doubt convention:

```text
doubt_raw = saturated_left_omissions - saturated_right_omissions
doubt_choice = -doubt_raw
```

`doubt_choice` points away from the side with greater accumulated omission
doubt. Combine both components on every valid trial:

```text
decision_drive = expectancy_weight * expectancy_choice
                 + doubt_weight * doubt_choice
model_value = tanh(decision_drive)
P(left) = (1 + model_value) / 2
P(right) = 1 - P(left)
```

There is no generic previous-choice persistence term. Below-threshold
home-directed expectancy supplies stable home preference. There is also no
branch that discards expectancy after an omission.

### 2.4 Canonical initial parameters

Use one fixed parameter record across every sample:

| Parameter | Initial value | Units |
|---|---:|---|
| `expectancy_threshold` | `2.5` | rewarded trials |
| `expectancy_scale` | `0.5` | rewarded trials |
| `expectancy_weight` | `1.0` | unitless drive |
| `doubt_weight` | `1.0` | unitless drive |
| `doubt_lambda` | `0.5` | inverse omission trials |

All five values will be included in the existing validated JSON override
mechanism. Threshold and weights must be nonnegative; scale and lambda must be
strictly positive. Values remain constant across samples.

### 2.5 Expected qualitative checkpoints

The canonical parameters should produce these pre-trial preferences:

- after one or two home-side rewards: prefer home;
- after three home-side rewards: prefer the opposite/probe side;
- after one home omission below threshold: retain home preference, weakened by
  doubt;
- after one home omission above threshold: expectancy and doubt both favor the
  probe side;
- after a below-threshold probe omission: expectancy and doubt both favor
  returning home;
- after an above-threshold probe omission: expectancy favors remaining on the
  probe side while doubt favors returning home;
- following approximately three consecutive above-threshold probe omissions,
  accumulated doubt overtakes canonical expectancy and favors return;
- reward on a probe side makes it the new home, resets its reward count to one,
  clears doubt, and produces preference for that new home.

This permits immediate return in `PR-02` but delayed return in `PR-01` without
changing parameters: the two probes begin with different reward-count
expectancies.

Approximate canonical probabilities, included as implementation checkpoints,
are:

| History state before prediction | Preferred side probability |
|---|---:|
| Home after reward count 1 | home `0.880` |
| Home after reward count 2 | home `0.821` |
| Home after reward count 3 | probe `0.821` |
| Home after count 1 plus one home omission | home `0.770` |
| Home after count 3 plus one home omission | probe `0.910` |
| Probe after count 1 plus one probe omission | return home `0.941` |
| Probe after count 3 plus one probe omission | remain probe `0.676` |
| Probe after count 3 plus two probe omissions | remain probe `0.564` |
| Probe after count 3 plus three probe omissions | return home `0.507` |

“Home” and “probe” are interpretations derived from `confirmed_side` and the
current or previous action. They are not additional latent states.

## 3. Architecture and files

### 3.1 Model replay

Modify `src/behavior_modeling/benchmark_models.py` only within the benchmark
model layer:

- add `home_referenced_expectancy_doubt` to `OBSERVABLE_MODEL_IDS`;
- add a small deterministic replay function using the existing expectancy and
  doubt state/update functions;
- record aligned internal states:
  `expectancy_confirmed_side`, `expectancy_reward_count`,
  `expectancy_strength`, `expectancy_choice`, `doubt_raw_value`,
  and `doubt_choice`; the aligned `decision_drive` remains the existing
  top-level `ModelReplay` field;
- add the canonical parameter record and registry entry.

No new class hierarchy or general model framework is needed. A short private
function may calculate the signed expectancy choice if that keeps the replay
loop readable and independently testable.

### 3.2 Runner and report integration

Modify `src/behavior_analysis/run_behavior_model_benchmark.py` to accept and
validate overrides for the five parameters. Existing generic replay, scoring,
CSV, plotting, summary, and logging code should include the model
automatically once it is registered.

The default report will increase from 14 to 15 total models, including the
oracle, and from 28 to 30 diagnostic pages. No new plot type is required.

### 3.3 Documentation

Update:

- `docs/behavior_model_benchmark_spec.md` with the complete model equations,
  state transitions, parameters, limitations, and required checkpoints;
- `docs/behavior_model_benchmark_plan.md` so roster and output counts reflect
  the additive model.

The existing direct-expectancy specification and implementation remain
unchanged.

## 4. Tests written before implementation

Tests will be added first, run to confirm the RED phase, and committed before
production changes.

1. **Registry contract:** the ordered observable registry contains the new ID
   exactly once and all models retain immutable parameter mappings.
2. **Initial state:** before any rewarded or omitted history, output is neutral.
3. **No-home omission:** an initial omission supplies doubt away from the
   omitted side without inventing a confirmed home.
4. **Below-threshold home reward:** one and two rewards favor the confirmed
   home side.
5. **Above-threshold home reward:** three rewards favor a probe to the opposite
   side.
6. **Below-threshold home omission:** retained expectancy outweighs one
   omission and continues to favor home.
7. **Above-threshold home omission:** expectancy and doubt align toward the
   probe side.
8. **Below-threshold probe omission:** expectancy and doubt align toward an
   immediate return home.
9. **Above-threshold probe omissions:** the first two omissions retain probe
   preference and the third changes the greedy preference to home.
10. **Rewarded probe adoption:** reward on the probe side changes confirmed
    home, resets reward count to one, clears doubt, and favors the new home.
11. **Scenario checkpoints:** canonical signs are verified on `PR-01`,
    `PR-02`, `EX-01`, `EX-03`, `RS-01`, `CT-01`, and `CT-02` without requiring
    good correspondence on every trial.
12. **Causality and context:** `CT-01` and `CT-02` have identical predictions
    at their shared observable-history checkpoint; current outcome and hidden
    context cannot affect a pre-trial prediction.
13. **Mirror invariance:** exchanging left and right reverses signed values and
    preserves choice-relative probabilities.
14. **Invalid-row continuity and reset:** invalid rows do not update state and
    independent scenarios start cleanly.
15. **Existing-model regression:** selected checkpoint outputs from
    `direct_expectancy_switch_doubt` remain unchanged.
16. **Override validation:** all five parameters accept valid uniform
    overrides and reject booleans, non-finite values, unknown fields, negative
    weights/thresholds, and nonpositive scales/lambdas.
17. **Report integration:** a selected-model run saves two diagnostic pages,
    trialwise internal states, parameters, scores, summary, and a warning-free
    log under the new model ID.

After implementation, run the focused model and runner tests, the full
behavior-model regression suite, and one all-model smoke report. Visually
inspect both new-model pages and the combined agreement figure.

## 5. Dependencies and performance

No new dependency is required. The implementation reuses NumPy and existing
project functions.

Replay remains linear in trial count with constant-size state. Adding one
model increases benchmark computation and plotting approximately in proportion
to one additional model. With 22 short samples, this is negligible compared
with PNG rendering. No parallelism, caching, or optimization is warranted.

## 6. Safety and scope boundaries

- Do not modify `src/neural_analysis`.
- Do not modify `src/main.py` or the real-session workflow.
- Do not alter existing model equations or historical benchmark outputs.
- Do not use hidden correct-side information.
- Do not fit or select parameters by sample or trial.
- Do not add an HMM or other latent model-selection mechanism.
- Do not save mock samples as real behavior sessions.

## 7. Approval decisions

Implementation should begin only after approval of:

1. the model ID `home_referenced_expectancy_doubt`;
2. the additive equation with no generic persistence term;
3. the five canonical parameter values in Section 2.4; and
4. allowing pre-reward omission doubt to act even though no home side has yet
   been confirmed.
