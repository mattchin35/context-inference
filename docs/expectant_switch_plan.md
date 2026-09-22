# Expectant-switching exemplars: implementation plan

## 1. Objective

Implement two modular trial-level exemplar policies described in
`docs/expectant_switching_exemplar_spec.md`:

1. `simple_probe_persistence`: immediate previous-choice persistence plus a
   constant opposing signal after reward.
2. `expectancy_persistence_doubt`: immediate previous-choice persistence plus
   reward-count-dependent expectant switching and the verified existing
   counterfactual-doubt signal.

Both models will support:

- pre-trial mouse-history feature generation;
- bounded signed values and direct left/right choice probabilities;
- greedy compatibility comparisons with mouse choices;
- use as trial GLM-HMM predictors;
- executable closed-loop agents in `behavior_modeling`;
- light-mode diagnostic and behavioral-summary plots.

This work does not fit model weights, add a new model-selection framework, or
add multi-session/chunk-processing APIs.

## 2. Verified codebase contracts

The implementation must preserve the following existing conventions.

### 2.1 Trial and sign conventions

- Task choices are `0=right`, `1=left`.
- Signed trial predictors are left-positive: right is `-1`, left is `+1`.
- A positive numeric reward is treated as rewarded; zero or a negative value
  is treated as omitted.
- Feature values are recorded before the current trial updates model state.
- No-choice and experimenter-reward rows receive missing outputs and do not
  update model history. The next valid row therefore uses the previous valid
  trial's history.
- `prev_action` in the augmented dataframe is the previous physical row, not
  necessarily the previous valid trial, so it must not drive the new models.

### 2.2 Existing doubt semantics

The reusable doubt source is the counterfactual side-specific omission state
that currently produces `left_cf_omissions`, `right_cf_omissions`, and
`relative_doubt_index`.

For each valid outcome:

- omission increments the chosen side's counter;
- omission preserves the other side's counter;
- choice change without reward does not reset either counter;
- any reward resets both counters to zero;
- session initialization sets both counters to zero.

The existing raw value is

`relative_doubt_index = H_left - H_right`,

where `H_side = 1 - exp(-lambda * omissions_side)`. It points toward the
doubted side. The new preferred-choice signal is therefore

`doubt_choice_signal = -relative_doubt_index`.

The existing observer must retain its current public raw `doubt_value` and
combined `hmm_value - doubt_value` behavior. The refactor changes ownership of
the update logic, not its results.

### 2.3 Initial parameters

The initial adjustable preset is:

| Parameter | Default |
|---|---:|
| expectancy threshold `k0` | 3.0 rewards |
| expectancy scale `s` | 1.0 rewards |
| doubt saturation `lambda` | existing `omission_lam`, currently 0.5 |
| simple persistence weight | 1.0 |
| simple probe weight | 1.0 |
| full persistence weight | 1.0 |
| full expectancy weight | 1.0 |
| full doubt weight | 1.0 |

These are demonstrative defaults, not fitted values. With all weights equal to
one, opposing signals can reduce persistence to indifference but cannot create
a switching probability above 0.5 on their own. The parameters will be easy
to change for later demonstrations.

## 3. Tests to write and commit before implementation

Testing will use `pytest` and strict RED-GREEN-REFACTOR. The failing tests will
be committed before production implementation, as required by the repository
workflow.

### 3.1 Shared component unit tests

Create `src/tests/behavior_modeling/test_expectant_switching.py` with tests for:

1. **Expectancy curve**
   - `k0=3, s=1` reproduces the specification's values for counts 0 through 5.
   - output lies in `[0, 1]`;
   - nonpositive scale raises `ValueError`.

2. **Previous-outcome signals**
   - first-trial persistence and probe are zero;
   - persistence equals the previous valid choice sign;
   - probe equals `-previous_choice * previous_reward`;
   - omission gates probe and expectant-switch signals to exactly zero.

3. **Reward-confirmed expectancy state**
   - first reward sets the confirmed side and count to one;
   - same-side rewards increment the count;
   - omissions and unrewarded choice changes preserve side and count;
   - opposite-side reward changes side and resets count to one;
   - true context/block labels are not accepted as update inputs.

4. **Shared counterfactual doubt state**
   - chosen-side omissions increment only that side;
   - choice changes without reward preserve both counts;
   - reward on either side clears both counts;
   - raw doubt matches `relative_doubt_index`;
   - preferred-choice doubt is the exact negation of raw doubt;
   - left/right mirroring negates both signed outputs.

5. **Readouts and probability mapping**
   - simple and full decision drives follow the specified equations;
   - signed value is `tanh(drive)`;
   - left probability is `(1 + value) / 2` and right probability is its
     complement;
- probabilities remain finite and sum to one;
- exact ties produce `p_left=p_right=0.5`.
- greedy compatibility decisions repeat the previous model choice on exact
  ties, using the established initial tie choice, instead of relying on
  `argmax` category order.

6. **Modularity and defaults**
   - full readout with fixed expectancy one, zero doubt weight, and
     `w_E=w_R` equals the simple readout;
   - the all-ones simple preset is indifferent after reward;
   - the all-ones full preset shows increasing departure probability as doubt
     accumulates, without falsely asserting that it exceeds 0.5.

7. **Trial ordering and validity**
   - outputs at trial `t` depend only on observations through `t-1`;
   - modifying trial `t` or later cannot change trial `t` output;
   - invalid rows are missing and do not update state;
   - two separately constructed states begin identically and do not share
     mutable state.

### 3.2 Existing-doubt parity tests

Extend existing tests to prove that the refactor is behavior-preserving:

- `src/tests/behavior_analysis/test_session_analysis.py`: the decision-variable
  counter sequence remains unchanged, including manual-reward/no-choice skips.
- `src/tests/behavior_analysis/test_trial_features.py`:
  `relative_doubt_index` retains its current values and vectorized shape.
- `src/tests/behavior_modeling/test_agents.py`:
  `HMMRewardDecayRelativeDoubt` produces exactly the same pre-trial raw doubt,
  HMM component, and combined value as the offline functions.
- `src/tests/behavior_modeling/test_switch_modes.py` and
  `test_switch_parallel_mode.py`: passive inactive-agent doubt decay remains
  numerically unchanged while switching code stops rewriting agent internals.

### 3.3 Batch feature integration tests

Extend `src/tests/behavior_analysis/test_gather_trial_features.py` to check:

- all diagnostic and model output columns are present;
- values match a hand-calculated short sequence;
- output rows remain aligned with input rows;
- invalid rows contain the established missing sentinel and preserve history;
- parameter JSON contains the configurable threshold, scale, lambda, and
  weights;
- directional outputs mirror correctly under left/right reversal;
- final model values are included in side-equivalent leave/stay conversion,
  while unsigned counts and strengths are not.

### 3.4 Agreement, registry, and plotting tests

- `src/tests/behavior_analysis/test_session_analysis.py`: both model-value
  columns are included in default blockwise mouse-agent agreement.
- `src/tests/behavior_analysis/test_trial_state_space_modeling.py`: both final
  model values are accepted by the predictor registry with stable labels.
- `src/tests/behavior_analysis/test_performance_plots.py`: both agreement
  series have stable labels/colors, and a light-mode diagnostic plot is saved
  with labeled axes and a caption/summary area.
- Create
  `src/tests/behavior_analysis/test_expectant_switching_analysis.py` with
  synthetic tests for post-reward excursion frequency, incorrect-side run
  lengths, return intervals, recurrence, and reward-count stratification.
  These tests validate metric definitions, not a particular biological result.

### 3.5 Executable-agent and controller tests

Extend `src/tests/behavior_modeling/test_agents.py` and
`src/tests/behavior_modeling/test_controller.py` to check:

- controller selection recognizes both stable model identifiers;
- online agent outputs match batch replay for the same supplied action/outcome
  history;
- constructors/controller calls receive an explicit seeded random generator;
- the agents use the direct `(1 +/- value)/2` mapping rather than the inherited
  temperature/softmax mapping;
- seeded sampling is reproducible;
- controller performance tables contain model diagnostics;
- a seeded closed-loop smoke run succeeds for both agents;
- hidden task state affects rewards through the task but is never passed into
  model-state updates.

### 3.6 RED commit and commands

After adding the tests:

1. Run the focused tests with `uv run pytest ...`.
2. Confirm failures are caused by missing expectant-switching functionality,
   not malformed fixtures or unrelated environment errors.
3. Commit only the tests, with a message such as
   `test: specify expectant switching exemplars`.

No production implementation begins before this RED commit.

## 4. Architecture

### 4.1 Package direction and design constraints

Place reusable model mechanics in `behavior_modeling`, then let
`behavior_analysis` adapt their array outputs into dataframes, summaries, and
plots. This preserves the existing useful dependency direction already used by
`ideal_observer.py`: analysis may consume model implementations, while core
model code does not import session analysis, pandas, or plotting.

Use data-only frozen dataclasses plus deterministic module-level functions for
component transitions and readouts. A transition receives the old state and
an explicit observation and returns a new state; it does not mutate external
objects. Executable agents may assign the returned state to their own private
attribute, because the agent itself is the stateful integration boundary.

Do not introduce mixins, nested convenience functions, global mutable state,
or control flags that select multiple lower-level behaviors. The two models
have separate explicit readout functions and separate executable agent
classes. No new Protocol is needed for the internal components because there
is only one implementation of each behavior; introducing an abstraction here
would be premature. The new agents retain the existing one-level
`BehaviorAgent` interface because the controller already depends on it.

### 4.2 Shared doubt module

Create `src/behavior_modeling/counterfactual_doubt.py` as the single source of
truth for the existing counterfactual-doubt update and value calculations. It
will depend only on the standard library and NumPy. This keeps doubt
independent of the new expectancy hypothesis and allows the existing observer,
batch counters, and new full exemplar to compose the same implementation.

#### `CounterfactualDoubtState`

Immutable data:

- `right_omissions: float`;
- `left_omissions: float`.

Counts are integral in ordinary batch/agent updates. They are typed as
nonnegative floats because the existing inactive-agent switching mode converts
doubt strength back into effective fractional omission counts during passive
decay.

Deterministic functions will provide:

- raw and choice-oriented values from state and lambda;
- one valid action/reward transition returning a new state;
- the existing inactive-mode passive decay returning a new state;
- vectorized raw/choice-value calculations for array callers.

The existing
`sided_omissions_counter(..., counterfactual=True)` path can delegate to the
same rule without changing its public tuple interface.

### 4.3 Expectant-switching module

Create `src/behavior_modeling/expectant_switching.py` as the single source of
truth for expectancy, previous-outcome signals, exemplar composition, and
array replay. It imports the shared doubt component but does not own or rewrite
its algorithm. It too will depend only on the standard library and NumPy, so
analysis adapters can import it without creating a reverse package dependency.

The module will contain the following components.

#### Focused parameter dataclasses

Use separate frozen data-only configurations so the simple model does not
depend on expectancy or doubt parameters it never uses:

- `ExpectancyCurveParams`: threshold and positive scale;
- `SimpleProbePersistenceParams`: persistence and probe weights;
- `ExpectancyPersistenceDoubtParams`: curve params, positive doubt lambda,
  and persistence/expectancy/doubt weights;
- `ExpectantSwitchingFeatureConfig`: a batch-boundary container holding the
  simple and full configurations when both outputs are requested together.

Module-level validation functions reject nonfinite values, nonpositive scale
or doubt lambda, and negative weights.

Small creator functions at the analysis and controller boundaries translate
existing aggregate parameter dataclasses into the focused configuration needed
by that caller. Computational functions receive only the focused configuration
rather than an entire session or task object.

#### `PreviousOutcomeState`

Immutable data:

- `previous_choice: int | None`;
- `previous_reward: int`.

Pure functions calculate persistence/probe signals and return the next state
from a parsed valid action/outcome. Before any valid choice, both signals are
zero. A new default instance represents reset state; no mutating `reset()`
method is needed.

#### `RewardConfirmedExpectancyState`

Immutable data:

- `confirmed_side: int | None`;
- `reward_count: int`.

Pure functions calculate strength and return the next state from a parsed
valid action/outcome. Omissions never modify the stored side/count. Reward
updates follow the exact episode rules in the specification. A new default
instance represents reset state.

#### Readout functions and result type

A frozen result dataclass will hold:

- `decision_drive: float`;
- `signed_value: float`;
- `p_right: float`;
- `p_left: float`.

Pure functions will compute:

- expectancy strength;
- `simple_probe_persistence` readout;
- `expectancy_persistence_doubt` readout;
- direct probability conversion from decision drive/value.

The readout functions will not mutate state. The caller must calculate all
pre-trial outputs before applying each state update.

### 4.4 Batch replay adapter

Add one array-oriented replay function to the shared model module that:

1. receives aligned actions, rewards, and a precomputed valid mask;
2. initializes fresh component state for the session;
3. preallocates aligned output arrays;
4. records diagnostics and both exemplar outputs before every valid update;
5. leaves invalid output rows missing and does not update any component;
6. updates previous outcome, expectancy, and doubt exactly once after the
   valid observation.

`gather_trial_features.py` remains the dataframe-level adapter because row
alignment and missing-value conventions are central to its responsibility. It
passes only aligned action/reward arrays, a validity mask, and the focused
configuration into the core replay.

The public feature pipeline remains one session per call. No session grouping,
chunk continuation, or state persistence API will be added.

### 4.5 Executable agents

Add two `BehaviorAgent` implementations in
`src/behavior_modeling/agents/agents.py`:

- `SimpleProbePersistenceAgent`;
- `ExpectancyPersistenceDoubtAgent`.

Each will compose the same shared state/readout components used by batch
replay. They will expose the existing `value` and `action_dist` interface, but
override action-distribution updates so that

`action_dist = [(1 - value)/2, (1 + value)/2]`.

They must not call the inherited temperature-based logistic mapping. The
controller will continue to perform probabilistic sampling unless its existing
greedy option is explicitly enabled. If greedy mode is enabled, an exact tie
will repeat the agent's previous choice, with the established initial tie
choice on the first trial; it will not use bare `argmax` tie ordering.

Random choice sampling will use an explicitly injected
`np.random.Generator`; the new agents will not create or use a module-global
generator. The controller remains the creation/factory boundary and supplies
the seeded generator before passing the constructed agent to the run loop.

### 4.6 Localized doubt refactor

Refactor only the duplicated doubt portions:

- `decision_variable_counters.sided_omissions_counter` delegates its
  counterfactual case to the shared pure updater;
- `trial_features.relative_doubt_index` delegates its saturation calculation
  to the shared vectorized raw-value function;
- `HMMRewardDecayRelativeDoubt` owns a shared `CounterfactualDoubtState`
  instead of manually reproducing transitions.
- `agent_switching/switching.py` invokes an explicit agent/shared-component
  operation for passive doubt decay rather than reading and rewriting the
  doubt formula itself.

Compatibility attributes such as `left_omissions_cf`,
`right_omissions_cf`, and raw `doubt_value` are used by current switching code
and tests. Preserve them as documented compatibility properties backed by the
composed doubt state; new code must use the focused state operations rather
than those properties. No other decision-variable or HMM logic will be
refactored.

## 5. Exact output schema

### 5.1 Augmented-trial columns

Add the following pre-trial columns:

Shared diagnostics:

- `previous_choice` — signed previous valid choice, or 0 initially;
- `previous_reward` — previous valid binary reward, or 0 initially;
- `reward_triggered_probe` — signed constant probe gate;
- `expectancy_confirmed_side` — `0`, `1`, or missing before first reward;
- `expectancy_reward_count` — integer episode reward count;
- `expectancy_strength` — unsigned value in `[0, 1]`;
- `expectant_switch` — signed expectancy-gated probe;
- `doubt_raw_value` — existing doubted-side-positive quantity;
- `doubt_choice_signal` — its preferred-choice-oriented negation.

Simple model:

- `simple_probe_persistence_drive`;
- `simple_probe_persistence_value`;
- `simple_probe_persistence_prob_left`.

Full model:

- `expectancy_persistence_doubt_drive`;
- `expectancy_persistence_doubt_value`;
- `expectancy_persistence_doubt_prob_left`.

Counts and unsigned strengths are diagnostic state, not directional GLM
predictors. Directional signals, drives, and values use the left-positive
convention. Invalid rows follow the current `None`/CSV `"None"` convention.

### 5.2 Parameter persistence

Extend `gather_trial_features.TaskParams` with explicit expectancy and exemplar
weight fields. `save_trial_features` will continue using `asdict`, so
`trial_feature_params.json` records the exact preset used. Do not overwrite or
rename existing parameter keys.

Extend `behavior_modeling.parameters.task_config.AgentParams` with equivalent
fields for executable agents. Controller run-name/metadata generation will
include the relevant parameters for these two agents so saved simulations are
reproducible.

### 5.3 Agreement and GLM registry

Add default agreement mappings:

- `simple_probe_persistence_mouse_agreement` ->
  `simple_probe_persistence_value`;
- `expectancy_persistence_doubt_mouse_agreement` ->
  `expectancy_persistence_doubt_value`.

Add short plot labels and stable colors in `performance_plots.py`.

Register only these final signed values in
`TRIAL_GLM_PREDICTOR_LABELS`:

- `simple_probe_persistence_value`;
- `expectancy_persistence_doubt_value`.

Do not add either to `DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS`; this avoids silently
changing established GLM-HMM fits. Callers can request them explicitly.

Add signed diagnostics, drives, and final values to
`LEFT_RIGHT_VALUE_COLUMNS_FOR_SIDE_EQUIVALENCE` where leave/stay conversion is
meaningful. Do not add confirmed side, reward count, expectancy magnitude, or
probability columns to that directional registry.

### 5.4 Controller diagnostics

Expose pre-trial agent diagnostics through attributes or a small diagnostic
mapping. Extend controller performance collection with stable `agent_...`
columns for:

- previous choice and reward;
- probe;
- confirmed side and reward count;
- expectancy strength and signed expectant-switch signal;
- raw and choice-oriented doubt;
- decision drive, signed value, and left probability.

Unrelated agent types receive missing values through the controller's existing
safe attribute lookup pattern.

## 6. Behavioral summaries and light-mode diagnostics

Create `src/behavior_analysis/expectant_switching_analysis.py` for pure
array-oriented behavioral summary helpers, with explicit types, shapes, choice
encoding, and units, for:

- probability/frequency of an excursion immediately after reward;
- incorrect-side excursion run lengths, in valid trials;
- return intervals, in valid trials;
- recurrence count or fraction without an intervening true context change;
- the preceding measures grouped by `expectancy_reward_count` or documented
  reward-count bins.

These summaries describe actual mouse or simulated-agent sequences. They do
not fit parameters and do not use hidden context to update the model; hidden
state is used only afterward to label correct/incorrect behavior.

Keep dataframe extraction in a thin analysis adapter that validates required
columns and passes only the needed aligned arrays into these helpers. This
avoids coupling core metric calculations to unrelated dataframe columns.

Add a light-mode diagnostic figure with an opaque white background, readable
fonts, labeled axes, and a concise caption. At minimum it should show:

1. observed choice/reward and, when available, true context;
2. both models' `p_left` traces;
3. persistence, probe, expectant-switch, and doubt-choice components;
4. reward count and expectancy strength;
5. compact behavioral-summary annotations or companion panels.

The plot should accept an axis/figure-friendly dataframe rather than recompute
model state. Save PNG by default using existing plotting save conventions.

## 7. File-by-file change map

### New files

- `src/behavior_modeling/counterfactual_doubt.py`
  - shared existing doubt state, pure updates, and raw/choice-oriented values.
- `src/behavior_modeling/expectant_switching.py`
  - configuration, previous-outcome and expectancy state, pure exemplar
    readouts, and aligned array replay; composes the separate doubt module.
- `src/behavior_analysis/expectant_switching_analysis.py`
  - post hoc array metrics and a thin dataframe extraction/validation adapter;
    never updates model state.
- `src/tests/behavior_modeling/test_expectant_switching.py`
  - focused model component, replay, timing, and symmetry tests.
- `src/tests/behavior_analysis/test_expectant_switching_analysis.py`
  - behavioral-summary metric and dataframe-adapter tests.

### Existing behavior-analysis files

- `src/behavior_analysis/decision_variable_counters.py`
  - delegate counterfactual omission updates to the shared implementation.
- `src/behavior_analysis/trial_features.py`
  - preserve `relative_doubt_index` API while delegating its calculation.
- `src/behavior_analysis/gather_trial_features.py`
  - add parameters, run batch replay, attach output columns, update the
    side-equivalence registry, and persist configuration.
- `src/behavior_analysis/session_analysis.py`
  - include both final values in default agreement outputs.
- `src/behavior_analysis/trial_state_space_modeling.py`
  - add optional predictor-registry entries only.
- `src/behavior_analysis/performance_plots.py`
  - add agreement labels/colors and the dedicated light-mode diagnostic plot.

### Existing behavior-modeling files

- `src/behavior_modeling/agents/agents.py`
  - refactor existing doubt state and add both executable agents.
- `src/behavior_modeling/parameters/task_config.py`
  - add explicit adjustable parameters.
- `src/behavior_modeling/controller.py`
  - select the agents, record diagnostics, and serialize identifiable run
    parameters.
- `src/behavior_modeling/agent_switching/switching.py`
  - delegate passive doubt decay through the composed doubt operation instead
    of duplicating formulas and mutating counter attributes directly.
- `src/behavior_modeling/sample_switch_run.py`
  - include the new stable names where strategy choices are enumerated, while
    preserving existing demos.

### Tests to extend

- `src/tests/behavior_analysis/test_session_analysis.py`
- `src/tests/behavior_analysis/test_trial_features.py`
- `src/tests/behavior_analysis/test_gather_trial_features.py`
- `src/tests/behavior_analysis/test_trial_state_space_modeling.py`
- `src/tests/behavior_analysis/test_performance_plots.py`
- `src/tests/behavior_modeling/test_agents.py`
- `src/tests/behavior_modeling/test_controller.py`
- `src/tests/behavior_modeling/test_switch_modes.py`
- `src/tests/behavior_modeling/test_switch_parallel_mode.py`
- relevant sample-switch tests if their fixed strategy registries change.

## 8. Implementation sequence after plan approval

1. **RED: shared behavior contract**
   - Write all focused component and parity tests.
   - Run them and verify expected failures.
   - Commit tests before production code.

2. **GREEN: shared components and doubt parity**
   - Implement the shared module.
   - Delegate existing batch and agent doubt logic.
   - Run component, trial-feature, session-counter, and existing agent tests.

3. **GREEN: batch feature integration**
   - Add configuration and aligned output columns.
   - Verify missing-row handling, parameter JSON, and leave/stay conversion.

4. **GREEN: executable agents and controller**
   - Add both agents, direct probability mapping, selection, metadata, and
     diagnostics.
   - Verify offline/online parity and seeded closed-loop smoke tests.

5. **GREEN: agreement, GLM registry, summaries, and plots**
   - Add compatibility agreement mappings and optional GLM predictors.
   - Add behavioral summaries and light-mode diagnostic figures.

6. **REFACTOR**
   - Remove duplicate adapters, improve names/docstrings, and keep all tests
     green without broad unrelated changes.

7. **Verification**
   - Run focused suites after every stage with `uv run pytest`.
   - Run the complete `src/tests/behavior_analysis` and
     `src/tests/behavior_modeling` suites.
   - Run the full project test suite if focused suites pass and runtime is
     reasonable.
   - Inspect at least one generated PNG for labels, clipping, white background,
     and readable layout.

## 9. Dependencies and performance

No new third-party dependency is needed. Use existing NumPy, pandas,
matplotlib, pytest, and project utilities.

Within `behavior_modeling`, use explicit relative imports between the two new
core modules and existing agent code. From `behavior_analysis`, use explicit
absolute imports from `src.behavior_modeling`. Do not add wildcard imports or
new nested subpackages.

Creation remains separate from use:

- focused configuration creator functions translate aggregate saved params;
- the analysis adapter creates fresh replay state and passes arrays to the
  computational function;
- `controller.select_agent` constructs concrete agents and the run loop uses
  only the established agent interface;
- plotting receives already computed columns and never constructs or updates a
  behavior model.

All public functions, dataclasses, and agent classes will have type hints and
data-contract docstrings. Local annotations will be limited to cases where the
type is ambiguous. Helpers will be module-level functions rather than lambdas,
nested convenience functions, closures, or behavior-heavy utility classes.

Expected complexity is linear in the number of trials:

- one sequential pass updates the three small state components;
- both exemplar outputs are computed during that pass;
- output arrays are preallocated;
- no dataframe row concatenation or repeated full-history recomputation occurs
  inside the loop;
- memory is `O(n_trials)` for saved outputs and `O(1)` for model state.

Closed-loop agents perform `O(1)` work per trial. Behavioral summaries should
also use single-pass run extraction or vectorized grouping. Optimization beyond
this is unnecessary unless profiling later identifies a real bottleneck.

## 10. Risks and safeguards

1. **Doubt sign confusion**
   - Preserve raw `relative_doubt_index` and introduce a separately named
     negated `doubt_choice_signal`; never silently reinterpret the old column.

2. **Pre/post-trial leakage**
   - Centralize the compute-then-update loop and test future-observation
     invariance.

3. **Double state updates**
   - One replay/controller location owns updates; readout functions are pure.

4. **Accidental use of inherited softmax**
   - Agents explicitly construct probabilities from signed value, with parity
     tests for the formula.

5. **Changing existing observer behavior**
   - Run exact parity tests before and after the shared-doubt refactor.

6. **Overinterpreting the all-ones preset**
   - Label it demonstrative and report probabilities/behavioral distributions,
     not only greedy agreement.

7. **Changing established GLM-HMM fits**
   - Register new predictors as opt-in and leave the default predictor tuple
     unchanged.

8. **Model/context leakage**
   - State-update interfaces accept only action and outcome. Context is used
     only by the task reward generator and post hoc behavioral summaries.

9. **Misdescribing task transitions**
   - Closed-loop runs use the selected task object's existing transition
     semantics. In particular, current `BaseMDP.success_trigger` is
     correct-choice gated, not observed-reward gated.

## 11. Explicit non-goals

- fitting or optimizing exemplar weights;
- claiming the all-ones preset is scientifically optimal;
- adding the constant-probe-plus-doubt ablation in this implementation;
- changing the existing doubt reset rule;
- adding an HMM/belief term or side-bias intercept;
- resetting at task blocks or hidden-context transitions;
- multi-session grouping, chunk continuation, or state serialization APIs;
- replacing current default trial GLM-HMM predictors;
- refactoring unrelated legacy agents.

## 12. Completion criteria

Implementation is complete only when:

- tests were committed in a verified RED state before implementation;
- batch and executable paths use the same shared state/readout logic;
- existing doubt and observer behavior pass parity tests;
- both models produce aligned diagnostics, drives, values, and probabilities;
- all four requested integrations work;
- seeded closed-loop simulation is reproducible;
- behavioral summaries cover excursions, run lengths, returns, recurrence, and
  reward-count dependence;
- the light-mode diagnostic PNG passes visual inspection;
- focused behavior-analysis and behavior-modeling test suites pass;
- public functions/classes document types, shapes, choice encoding, timing,
  units, and return values.
