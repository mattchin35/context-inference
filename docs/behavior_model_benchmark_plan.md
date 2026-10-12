# Behavioral model benchmark: implementation plan

Status: revised draft for review; not approved for implementation.

Companion specification:
`docs/behavior_model_benchmark_spec.md`.

## 1. Purpose and scope

Implement a deterministic benchmark that replays the 22 prescribed behavior
samples through all 14 fixed model configurations in the specification. The
benchmark is intended to show where each consistent model describes behavior
well or poorly. It will not select models by trial, fit parameters separately
by sample, or simulate new mouse behavior.

The initial implementation will:

- construct and validate the samples in memory;
- replay each selected model using only causally available inputs;
- calculate aligned probabilities and sample-level correspondence scores;
- create two wide behavior-atlas pages per model for the full suite;
- create one combined sample-as-block agreement figure; and
- save a compact, auditable report outside the source repository.

It will not load experimental sessions, call `src/main.py`, create fake session
directories, alter the real-session workflow, or edit `src/neural_analysis`.
GLM-HMMs and other trialwise model-selection mechanisms remain out of scope.

## 2. Verified contracts and reuse

### 2.1 Data and timing conventions

- Choice encoding is `0=right`, `1=left`.
- Signed values are left-positive: `-1` favors right and `+1` favors left.
- Positive reward is rewarded; zero or negative reward is an omission.
- A prediction on row `t` uses valid observations only through row `t-1`.
- Invalid rows retain alignment but do not emit a prediction or update state.
- Every sample begins with fresh model state.
- Correct side is hidden from all models except the labeled oracle ceiling.
- Each complete sample is one agreement block; a greedy-policy tie contributes
  `0.5` and remains marked as a tie.

### 2.2 Existing code to reuse

- `src/behavior_modeling/expectant_switching.py`: existing exemplar state,
  equations, validation, and aligned replay.
- `src/behavior_modeling/counterfactual_doubt.py`: omission-state updates and
  doubt readouts.
- `src/behavior_analysis/trial_features.py`: WSLS, forgetting-Q, and HMM
  reward-decay parity references.
- `src/behavior_modeling/agents/agents.py`: executable forgetting-Q and HMM
  reward-decay-plus-doubt action policies.
- `src/behavior_analysis/ideal_observer.py`: mouse-history observer defaults
  and diagnostic conventions.
- `src/behavior_analysis/performance_plots.py`: light-mode PNG saving behavior.

The benchmark will not call session discovery, preprocessing, session CSV
writers, `gather_trial_features.main`, or the existing block-agreement helper.
That helper resolves ties to one side, whereas this benchmark must score exact
ties as `0.5`.

### 2.3 Common probability value

The plotted value for every model is:

```text
model_value = P(left) - P(right) = 2 * P(left) - 1
```

Tanh-linked models retain their documented tanh policy. The history observer
and forgetting-Q adapters retain their implemented logistic choice policies,
so their common value is `tanh(decision_drive / 2)`, not
`tanh(decision_drive)`. Tests will compare both adapters with the existing
executable policies.

Some models specify probabilities directly and have no meaningful pre-link
decision drive. Their `decision_drive` is absent rather than manufactured from
the probability.

## 3. Architecture

The design uses small data records and deterministic functions. No inheritance
hierarchy, model-selection controller, or general simulation framework is
needed.

### 3.1 Scenario definitions

New module: `src/behavior_modeling/benchmark_scenarios.py`

```python
@dataclass(frozen=True)
class BehaviorScenario:
    scenario_id: str
    title: str
    action: np.ndarray
    reward: np.ndarray
    valid: np.ndarray
    correct_side: np.ndarray
    correct: np.ndarray
    checkpoint: np.ndarray
```

All arrays have shape `(n_trials,)`. Choice and correct-side arrays use
`0=right`, `1=left`; reward is in task reward units; trial index is derived as
zero-based `np.arange(n_trials)`.

Public functions:

- `build_default_scenarios() -> tuple[BehaviorScenario, ...]`
- `mirror_scenario(scenario) -> BehaviorScenario`
- `validate_scenario(scenario) -> None`

The 22 definitions will be explicit Python data, not parsed from Markdown or
from a second compact-token language. Constructors defensively copy arrays and
mark them read-only so that the frozen record is not only superficially
immutable. Mirror and invalid-row cases used only for conformance remain tests,
not additional plotted samples.

### 3.2 Observable model replay

New module: `src/behavior_modeling/benchmark_models.py`

```python
@dataclass(frozen=True)
class ObservableHistory:
    action: np.ndarray
    reward: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class ModelReplay:
    model_id: str
    decision_drive: np.ndarray | None
    model_value: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    p_switch: np.ndarray
    exact_tie: np.ndarray
    internal_states: Mapping[str, np.ndarray]
```

Every present trial array has shape `(n_trials,)`. Invalid rows are `NaN` in
numeric arrays and `False` in `exact_tie`; `P(switch)` is also `NaN` before the
first valid choice. A missing decision drive is valid for models defined
directly by probabilities. Other valid-row probabilities and values must be
finite and correctly normalized.

Replay constructors defensively copy their arrays, mark them read-only, and
expose `internal_states` through a read-only mapping. `ObservableHistory`
receives the already read-only scenario arrays.

```python
ReplayFunction = Callable[[ObservableHistory], ModelReplay]


@dataclass(frozen=True)
class BenchmarkModelSpec:
    model_id: str
    display_label: str
    parameters: Mapping[str, float | int | str]
    replay: ReplayFunction
```

The parameter mapping is a read-only copy of the resolved values.
`build_observable_models(config)` returns the 13 models that use only observable
history. It does not include the oracle. A common finalizer derives right,
choice and switch probabilities, the common signed value, and the tie mask from
a model's primary output. Observed-choice probability is not part of `ModelReplay`
because it depends on the current prescribed action rather than only on the
model's pre-trial prediction.

### 3.3 Oracle boundary

The oracle is deliberately not forced into the ordinary replay interface:

```python
replay_oracle(history: ObservableHistory, correct_side: np.ndarray) -> ModelReplay
```

The benchmark orchestrator replays the 13 ordinary models using
`ObservableHistory`, then invokes the oracle adapter separately and appends its
result. The oracle uses observable previous choices only to derive its aligned
switch probability; `correct_side` determines its side prediction. This is the
only code path that receives `correct_side`. It avoids both a hidden-information
flag on every model and an ordinary-model input object that contains forbidden
context.

### 3.4 Model implementations

The complete roster is defined in Section 13 of the specification:

- five reference models, including the separately replayed oracle;
- four existing model adapters; and
- five proposed model configurations.

The two threshold configurations share one replay function with distinct fixed
parameter records. Proposed models reuse existing previous-outcome, expectancy,
and doubt functions wherever their rules match. Only the global reward
accumulator and threshold excursion state require new state records. Scenario
identity is never passed to a replay function.

### 3.5 Orchestration, scoring, and export

New module: `src/behavior_analysis/behavior_model_benchmark.py`

Primary functions:

- `run_benchmark(scenarios: Sequence[BehaviorScenario], observable_models: Sequence[BenchmarkModelSpec], include_oracle: bool) -> BenchmarkResult`
- `score_model_replay(scenario, replay) -> ScenarioModelScore`
- `build_trial_output_table(result, scenarios) -> pd.DataFrame`
- `build_score_table(result) -> pd.DataFrame`

`observable_models` is the already constructed, ordered sequence of selected
`BenchmarkModelSpec` objects. The command-line runner validates and resolves
configuration, calls `build_observable_models`, filters that registry, and then
passes the prepared objects into `run_benchmark`. `include_oracle` controls the
separate fixed oracle path; no ordinary model is constructed inside the
analysis function.

Before populating result mappings, `run_benchmark` rejects duplicate scenario
IDs, duplicate ordinary-model IDs, and any ordinary model using the reserved
`oracle_correct_side` ID. This prevents silent replacement of a replay keyed by
model and scenario ID.

The two result records have explicit contracts:

```python
@dataclass(frozen=True)
class ScenarioModelScore:
    model_id: str
    scenario_id: str
    n_valid: int
    mean_observed_probability: float
    mean_negative_log_probability: float
    n_nll_floored: int
    switch_brier: float
    greedy_agreement: float
    n_exact_ties: int


@dataclass(frozen=True)
class BenchmarkResult:
    replays: Mapping[tuple[str, str], ModelReplay]
    scores: tuple[ScenarioModelScore, ...]
```

Probabilities and scores are unitless. `switch_brier` is `NaN` when no trial
has a previous valid choice. The result mapping is read-only. Plots read replay
arrays directly; they do not depend on a heterogeneous DataFrame.

`score_model_replay` derives the probability assigned to each prescribed
choice from the replay's pre-trial `p_left` and `p_right` arrays. The trialwise
export uses the same scoring helper. This current-action-conditioned quantity is
never fed back into model state.

The exported trial table has one row per model/sample/trial. It contains the
prescribed behavior, common model outputs, and namespaced model-specific state
columns. `decision_drive` is blank when it is not applicable. The score table
has one row per model/sample pair with:

- valid-trial count;
- mean probability assigned to the observed choice;
- mean negative log probability;
- number of probabilities floored at `1e-12` for the logarithm only;
- switch/stay Brier score, excluding the first valid choice;
- sample-as-block greedy agreement; and
- exact-tie count.

Raw probabilities are never clipped. Poor behavioral correspondence is a
scientific result, not an implementation error.

### 3.6 Plotting

New module: `src/behavior_analysis/behavior_model_benchmark_plots.py`

`plot_model_behavior_atlas(...)` returns one page for at most 12 selected
samples and two balanced pages for larger selections. The complete 22-sample
suite therefore produces two pages with 11 subplots each per model. Pages use
at most four columns and are approximately 24 by 12 inches.

Every sample subplot uses one signed axis:

- a thin continuous model-value trace;
- unconnected choice circles at `-1` or `+1`;
- larger rewarded-choice circles;
- an initial `L` or `R` context label;
- labeled vertical dashed lines at context changes.

There is no continuous correct-side line and no separate probability or
internal-state panel. Pages are light mode with opaque white backgrounds,
shared legends, and a caption explaining the marks and fixed parameters.

`plot_combined_sample_agreement(...)` creates one figure with sample IDs on the
x-axis, agreement in `[0, 1]`, one series per selected model, and a horizontal
`0.5` reference. A heatmap is not part of the initial implementation.

### 3.7 Runner and configuration

New module: `src/behavior_analysis/run_behavior_model_benchmark.py`

```text
uv run python -m src.behavior_analysis.run_behavior_model_benchmark \
    --output-root /absolute/output/path
```

Options:

- required `--output-root PATH`, outside the source repository;
- repeatable `--model MODEL_ID` and `--scenario SCENARIO_ID` filters; and
- optional `--config PATH` for one validated JSON parameter override.

The default evaluates all 22 samples and all 14 models. The override schema,
allowed fields, and value constraints are defined in Section 2.4 of the
specification. Overrides are merged with canonical defaults once, then the same
resolved configuration is applied to every selected sample. Unknown keys,
booleans supplied as numbers, non-finite values, invalid types, and values
outside documented ranges are rejected before replay.

After validation, the runner constructs and filters the ordered model registry
and passes those prepared model specifications to `run_benchmark`. Model
selection and construction therefore remain outside the computation function.

## 4. Report artifacts

The runner resolves both the repository root and requested output root before
checking containment. It rejects an output root that resolves inside the
repository, including paths containing `..` or symlinks. It then creates a new
directory using a UTC timestamp with microseconds and `exist_ok=False`; every
report write is made beneath that resolved run directory. An existing path is
never overwritten:

```text
OUTPUT_ROOT/behavior_model_benchmark_YYYYMMDDTHHMMSSffffffZ/
    diagnostics/
        MODEL_ID/
            page_01.png
            page_02.png
    combined_sample_agreement.png
    trialwise_model_outputs.csv
    scenario_model_scores.csv
    model_parameters.json
    summary.md
    run.log
```

`trialwise_model_outputs.csv` is a derived audit table, not a synthetic session
or saved behavior run. No session objects, augmented trial tables, `.npz`
replay state, fake mouse IDs, or outputs beneath experimental session
directories are created.

The summary records selected samples and models, score definitions, resolved
parameters, code version, and dirty-worktree status. The log records timing,
warnings, and errors. `model_parameters.json` contains the complete resolved
configuration, including fixed models that are not overrideable.

## 5. Test-first implementation sequence

Every work package follows RED-GREEN-REFACTOR. Tests are committed in a failing
RED state before production code for that package.

### WP1: Scenario contract

Tests in `src/tests/behavior_modeling/test_benchmark_scenarios.py`:

1. All 22 stable IDs and arrays match the specification.
2. Arrays are one-dimensional, aligned, correctly typed, and read-only.
3. Correctness, reward, checkpoints, and reversal boundaries are consistent.
4. Invalid rows remain aligned.
5. Mirroring swaps side-dependent fields without mutating or sharing arrays.
6. Registry order is stable and IDs are unique.

### WP2: Common replay contract and reference models

Tests in `src/tests/behavior_modeling/test_benchmark_models.py`:

1. Present outputs have aligned shapes and normalized probabilities.
2. Missing decision drive is accepted; a present drive is finite on valid rows.
3. Model value, switch probability, and tie masks are derived correctly.
4. Invalid rows remain missing and do not update history.
5. Chance, persistence, WSLS, reversed WSLS, and oracle match hand calculations.
6. The ordinary-model registry contains no correct-side input or oracle.
7. Only the separate oracle adapter accepts `correct_side`; its observable
   history input is used to derive switch probability without current-trial
   leakage.

### WP3: Existing and proposed models

Continue tests in `test_benchmark_models.py`:

1. Existing exemplar traces match `replay_expectant_switching`.
2. WSLS and forgetting-Q drives match `trial_features` references.
3. History-observer state and choice probabilities match the executable agent.
4. History-observer and forgetting-Q use their logistic, not doubled-logit,
   probability links.
5. All 14 models satisfy the common replay contract on all default samples.
6. For every non-oracle model, identical observable prefixes produce identical
   pre-trial `p_left`, `p_right`, `model_value`, applicable `decision_drive`,
   `p_switch`, tie, and internal-state outputs despite different current
   actions, outcomes, or hidden context. Current-action-conditioned scoring
   fields are excluded from this comparison.
7. For every model, inserting an invalid row preserves subsequent state and
   valid-trial predictions.
8. State resets between samples and parameters remain fixed across the suite.
9. Mirrored histories produce the documented symmetric outputs.
10. `CT-03A` and `CT-03B` have identical non-oracle pre-switch predictions.
11. Proposed-model checkpoint values match the specification.
12. Threshold variants never inspect scenario identity.

### WP4: Scoring and tables

Tests in `src/tests/behavior_analysis/test_behavior_model_benchmark.py`:

1. Replay and score counts match selected models and samples.
2. Observed-choice probability is derived from the prescribed current action
   and the pre-trial side probabilities; it is never present in model replay or
   fed back into state.
3. Probability, NLL, Brier, agreement, and tie scores match hand-worked cases.
4. The `1e-12` floor affects only NLL and its count.
5. Every complete sample produces exactly one agreement score.
6. Poor agreement is returned normally rather than raised.
7. The trial table retains alignment, leaves inapplicable drives blank, and
   namespaces model-specific state columns.
8. Reordering samples cannot change replay results.
9. Duplicate scenario IDs, duplicate model IDs, and use of the reserved oracle
   ID by an ordinary model are rejected before any result mapping is built.

### WP5: Figures

Tests in
`src/tests/behavior_analysis/test_behavior_model_benchmark_plots.py`:

1. The full suite produces two balanced pages per model; small selections use
   one page, and every selected sample appears exactly once.
2. Behavior and model value share an axis; choices are not connected and
   rewarded markers are larger.
3. Context labels and transition boundaries are placed correctly, without a
   continuous correct-side trace.
4. No extra probability or internal-state panels are created.
5. Pages are opaque light-mode PNGs with shared legends and captions.
6. Combined agreement includes every selected model and the `0.5` reference.
7. Saving creates the expected path and closes only the saved figure.

### WP6: Configuration and report runner

Tests in
`src/tests/behavior_analysis/test_run_behavior_model_benchmark.py`:

1. Filters preserve registry order and reject unknown IDs.
2. Partial JSON overrides merge with defaults and reject all documented invalid
   forms.
3. The resolved configuration is identical across samples and saved in full.
4. The output root must be outside the repository.
5. A smoke run creates exactly the documented report artifacts.
6. The trialwise file is a benchmark table, not a session-shaped output.
7. Microsecond run names are distinct and existing paths are never overwritten.
8. Summary and log contain the documented metadata.
9. Repository containment checks use resolved paths and reject `..` and symlink
   aliases into the repository.
10. The run directory is created exclusively and all artifacts remain beneath
    it.

### WP7: Verification and visual approval

1. Run focused tests and affected existing behavior tests with `uv run pytest`.
2. Run the full non-neural test subset without editing neural-analysis code.
3. Generate one single-model smoke atlas and the combined agreement figure.
4. Obtain user approval of the visual design.
5. Generate the complete 14-model report only after visual approval.

The scientific pattern of good and poor correspondence is not a test
assertion; tests verify calculations, causality, and presentation contracts.

## 6. Dependencies and performance

No new dependency is required. Use NumPy, pandas, SciPy, Matplotlib, and pytest
already present in the project. All Python commands use `uv run`.

The histories are short, so sequential replay is clearer than parallelism.
The full run produces 28 atlas PNGs plus one agreement PNG. Figures are created
and closed sequentially, and each replay is reused across its atlas pages.
Profiling and low-level optimization are unnecessary unless measurements later
show an actual bottleneck.

## 7. Expected file changes

New production files:

- `src/behavior_modeling/benchmark_scenarios.py`
- `src/behavior_modeling/benchmark_models.py`
- `src/behavior_analysis/behavior_model_benchmark.py`
- `src/behavior_analysis/behavior_model_benchmark_plots.py`
- `src/behavior_analysis/run_behavior_model_benchmark.py`

New tests:

- `src/tests/behavior_modeling/test_benchmark_scenarios.py`
- `src/tests/behavior_modeling/test_benchmark_models.py`
- `src/tests/behavior_analysis/test_behavior_model_benchmark.py`
- `src/tests/behavior_analysis/test_behavior_model_benchmark_plots.py`
- `src/tests/behavior_analysis/test_run_behavior_model_benchmark.py`

Only minimal package `__init__.py` edits are permitted if required. Explicitly
excluded files are `src/main.py`, all of `src/neural_analysis`, session loaders,
and session-output writers. If implementation requires changing an existing
shared equation or public interface, stop and revise this plan first.

## 8. Approval gate

Implementation begins only after explicit user approval of this revised plan.
The first implementation action is the failing WP1 test commit.
