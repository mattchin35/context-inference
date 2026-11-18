"""Orchestration, scoring, and derived tables for behavior model benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.behavior_modeling.benchmark_models import (
    ORACLE_MODEL_ID,
    BenchmarkModelSpec,
    ModelReplay,
    ObservableHistory,
    replay_oracle,
)
from src.behavior_modeling.benchmark_scenarios import (
    BehaviorScenario,
    validate_scenario,
)


NLL_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class ScenarioModelScore:
    """Unitless summary scores for one model/sample pair.

    ``switch_brier`` is ``NaN`` when the sample has fewer than two valid
    choices. Counts are numbers of aligned valid trials.
    """

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
    """Immutable replay and score collection for one benchmark invocation."""

    replays: Mapping[tuple[str, str], ModelReplay]
    scores: tuple[ScenarioModelScore, ...]

    def __post_init__(self) -> None:
        """Copy the replay index and expose it through a read-only mapping."""
        object.__setattr__(self, "replays", MappingProxyType(dict(self.replays)))
        object.__setattr__(self, "scores", tuple(self.scores))


def run_benchmark(
    scenarios: Sequence[BehaviorScenario],
    observable_models: Sequence[BenchmarkModelSpec],
    include_oracle: bool,
) -> BenchmarkResult:
    """Replay fixed models independently over prepared behavior samples.

    Parameters
    ----------
    scenarios : sequence of BehaviorScenario
        Independent samples with aligned arrays of shape ``(n_trials,)``.
    observable_models : sequence of BenchmarkModelSpec
        Already constructed ordinary models that receive no hidden context.
    include_oracle : bool
        Whether to append the separately invoked correct-side ceiling.

    Returns
    -------
    BenchmarkResult
        Read-only replay mapping keyed by ``(model_id, scenario_id)`` and
        scores in scenario-major, model-minor order.

    Raises
    ------
    ValueError
        If IDs are duplicated, an ordinary model uses the reserved oracle ID,
        or a replay returns a model ID different from its specification.
    """
    scenario_tuple = tuple(scenarios)
    model_tuple = tuple(observable_models)
    _validate_unique_ids(scenario_tuple, model_tuple)

    replay_index: dict[tuple[str, str], ModelReplay] = {}
    scores: list[ScenarioModelScore] = []
    for scenario in scenario_tuple:
        validate_scenario(scenario)
        history = ObservableHistory(
            action=scenario.action,
            reward=scenario.reward,
            valid=scenario.valid,
        )
        for model in model_tuple:
            replay = model.replay(history)
            if replay.model_id != model.model_id:
                raise ValueError(
                    f"Replay ID {replay.model_id!r} does not match model "
                    f"specification {model.model_id!r}."
                )
            _append_result(replay_index, scores, scenario, replay)
        if include_oracle:
            oracle_replay = replay_oracle(history, scenario.correct_side)
            _append_result(replay_index, scores, scenario, oracle_replay)
    return BenchmarkResult(replays=replay_index, scores=tuple(scores))


def score_model_replay(
    scenario: BehaviorScenario,
    replay: ModelReplay,
) -> ScenarioModelScore:
    """Calculate correspondence scores for one aligned replay.

    Parameters
    ----------
    scenario : BehaviorScenario
        Prescribed choices and outcomes with shape ``(n_trials,)``.
    replay : ModelReplay
        Pre-trial probabilities aligned to the same rows.

    Returns
    -------
    ScenarioModelScore
        Unitless probability, NLL, switch Brier, and greedy-agreement scores.
        Exact ties contribute ``0.5`` to greedy agreement.
    """
    validate_scenario(scenario)
    if replay.p_left.shape != scenario.action.shape:
        raise ValueError("Replay and scenario trial shapes must match.")
    valid = scenario.valid
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("A scored scenario must contain at least one valid choice.")

    observed_probability = _observed_choice_probability(scenario, replay)
    retained_probability = observed_probability[valid]
    n_floored = int(np.sum(retained_probability < NLL_PROBABILITY_FLOOR))
    nll = -np.log(np.maximum(retained_probability, NLL_PROBABILITY_FLOOR))

    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 2:
        switch_brier = np.nan
    else:
        eligible = valid_indices[1:]
        previous = scenario.action[valid_indices[:-1]]
        observed_switch = (scenario.action[eligible] != previous).astype(float)
        predicted_switch = replay.p_switch[eligible]
        if np.any(~np.isfinite(predicted_switch)):
            raise ValueError("Switch probabilities must be finite after the first choice.")
        switch_brier = float(np.mean((predicted_switch - observed_switch) ** 2))

    agreement = np.empty(n_valid, dtype=float)
    retained_ties = replay.exact_tie[valid]
    predicted_left = replay.p_left[valid] > replay.p_right[valid]
    observed_left = scenario.action[valid] == 1
    agreement[retained_ties] = 0.5
    agreement[~retained_ties] = (
        predicted_left[~retained_ties] == observed_left[~retained_ties]
    ).astype(float)
    return ScenarioModelScore(
        model_id=replay.model_id,
        scenario_id=scenario.scenario_id,
        n_valid=n_valid,
        mean_observed_probability=float(np.mean(retained_probability)),
        mean_negative_log_probability=float(np.mean(nll)),
        n_nll_floored=n_floored,
        switch_brier=switch_brier,
        greedy_agreement=float(np.mean(agreement)),
        n_exact_ties=int(np.sum(retained_ties)),
    )


def build_trial_output_table(
    result: BenchmarkResult,
    scenarios: Sequence[BehaviorScenario],
) -> pd.DataFrame:
    """Build the derived aligned audit table for a benchmark result.

    Parameters
    ----------
    result : BenchmarkResult
        Replays keyed by ``(model_id, scenario_id)``.
    scenarios : sequence of BehaviorScenario
        Source prescribed samples. Arrays use the benchmark side/reward units.

    Returns
    -------
    pandas.DataFrame
        One row per model/sample/trial. Common columns are aligned; model state
        columns use the ``state__NAME`` namespace and are missing when absent.
    """
    scenarios_by_id = _index_scenarios(scenarios)
    rows: list[dict[str, object]] = []
    for (model_id, scenario_id), replay in result.replays.items():
        if scenario_id not in scenarios_by_id:
            raise ValueError(f"Missing source scenario {scenario_id!r}.")
        scenario = scenarios_by_id[scenario_id]
        observed_probability = _observed_choice_probability(scenario, replay)
        drive = (
            np.full(scenario.action.shape, np.nan, dtype=float)
            if replay.decision_drive is None
            else replay.decision_drive
        )
        for trial_index in range(scenario.action.size):
            row: dict[str, object] = {
                "model_id": model_id,
                "scenario_id": scenario_id,
                "trial_index": trial_index,
                "action": int(scenario.action[trial_index]),
                "reward": float(scenario.reward[trial_index]),
                "valid": bool(scenario.valid[trial_index]),
                "correct_side": int(scenario.correct_side[trial_index]),
                "correct": bool(scenario.correct[trial_index]),
                "checkpoint": bool(scenario.checkpoint[trial_index]),
                "decision_drive": float(drive[trial_index]),
                "model_value": float(replay.model_value[trial_index]),
                "p_left": float(replay.p_left[trial_index]),
                "p_right": float(replay.p_right[trial_index]),
                "p_switch": float(replay.p_switch[trial_index]),
                "p_observed": float(observed_probability[trial_index]),
                "exact_tie": bool(replay.exact_tie[trial_index]),
            }
            for state_name, state_values in replay.internal_states.items():
                row[f"state__{state_name}"] = state_values[trial_index]
            rows.append(row)
    return pd.DataFrame(rows)


def build_score_table(result: BenchmarkResult) -> pd.DataFrame:
    """Return one unitless summary row per scored model/sample pair.

    Parameters
    ----------
    result : BenchmarkResult
        Benchmark scores in deterministic scenario-major order.

    Returns
    -------
    pandas.DataFrame
        Columns correspond exactly to :class:`ScenarioModelScore` fields.
    """
    return pd.DataFrame(asdict(score) for score in result.scores)


# ---------------------------------------------------------------------------
# Helpers


def _append_result(
    replay_index: dict[tuple[str, str], ModelReplay],
    scores: list[ScenarioModelScore],
    scenario: BehaviorScenario,
    replay: ModelReplay,
) -> None:
    """Append one uniquely keyed replay and its score."""
    key = (replay.model_id, scenario.scenario_id)
    if key in replay_index:
        raise ValueError(f"Duplicate replay key {key!r}.")
    replay_index[key] = replay
    scores.append(score_model_replay(scenario, replay))


def _validate_unique_ids(
    scenarios: tuple[BehaviorScenario, ...],
    models: tuple[BenchmarkModelSpec, ...],
) -> None:
    """Reject identifiers that could collide in result mappings."""
    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    model_ids = [model.model_id for model in models]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Duplicate scenario IDs are not allowed.")
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Duplicate model IDs are not allowed.")
    if ORACLE_MODEL_ID in model_ids:
        raise ValueError(
            f"The model ID {ORACLE_MODEL_ID!r} is reserved for the separate oracle."
        )


def _index_scenarios(
    scenarios: Sequence[BehaviorScenario],
) -> dict[str, BehaviorScenario]:
    """Build a unique scenario lookup for tabular export."""
    scenario_tuple = tuple(scenarios)
    identifiers = [scenario.scenario_id for scenario in scenario_tuple]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate scenario IDs are not allowed.")
    return {scenario.scenario_id: scenario for scenario in scenario_tuple}


def _observed_choice_probability(
    scenario: BehaviorScenario,
    replay: ModelReplay,
) -> np.ndarray:
    """Derive current-action-conditioned probability without changing replay."""
    if replay.p_left.shape != scenario.action.shape:
        raise ValueError("Replay and scenario trial shapes must match.")
    probability = np.full(scenario.action.shape, np.nan, dtype=float)
    left_rows = scenario.valid & (scenario.action == 1)
    right_rows = scenario.valid & (scenario.action == 0)
    probability[left_rows] = replay.p_left[left_rows]
    probability[right_rows] = replay.p_right[right_rows]
    if np.any(~np.isfinite(probability[scenario.valid])):
        raise ValueError("Observed-choice probabilities must be finite on valid rows.")
    return probability
