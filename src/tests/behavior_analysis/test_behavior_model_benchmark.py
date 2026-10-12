"""Tests for benchmark orchestration, scoring, and derived audit tables."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.behavior_model_benchmark import (
    BenchmarkResult,
    build_score_table,
    build_trial_output_table,
    run_benchmark,
    score_model_replay,
)
from src.behavior_modeling.benchmark_models import (
    BenchmarkModelSpec,
    ModelReplay,
    build_observable_models,
)
from src.behavior_modeling.benchmark_scenarios import (
    BehaviorScenario,
    build_default_scenarios,
)


def _scenario(
    scenario_id: str = "manual",
    action: tuple[int, ...] = (1, 1, 0),
) -> BehaviorScenario:
    """Build a compact valid scenario for hand-scored tests."""
    actions = np.asarray(action, dtype=np.int8)
    correct_side = np.ones(actions.shape, dtype=np.int8)
    return BehaviorScenario(
        scenario_id=scenario_id,
        title="Manual scoring case",
        action=actions,
        reward=np.array([1.0, 0.0, 0.0])[: actions.size],
        valid=np.ones(actions.shape, dtype=bool),
        correct_side=correct_side,
        correct=actions == correct_side,
        checkpoint=np.zeros(actions.shape, dtype=bool),
    )


def _replay(
    model_id: str = "manual-model",
    p_left: tuple[float, ...] = (0.5, 0.75, 0.25),
    decision_drive: np.ndarray | None = None,
) -> ModelReplay:
    """Build one aligned replay with a single model-specific state."""
    left = np.asarray(p_left, dtype=float)
    right = 1.0 - left
    value = left - right
    p_switch = np.full(left.shape, np.nan, dtype=float)
    if left.size > 1:
        # The hand-worked scenario's previous choice is left on later rows.
        p_switch[1:] = right[1:]
    return ModelReplay(
        model_id=model_id,
        decision_drive=decision_drive,
        model_value=value,
        p_left=left,
        p_right=right,
        p_switch=p_switch,
        exact_tie=left == right,
        internal_states={"latent": np.arange(left.size, dtype=float)},
    )


def test_score_model_replay_matches_hand_calculation() -> None:
    """Likelihood, Brier, tie, and greedy agreement use the documented rules."""
    score = score_model_replay(_scenario(), _replay())

    expected_observed = np.array([0.5, 0.75, 0.75])
    assert score.n_valid == 3
    assert score.mean_observed_probability == pytest.approx(expected_observed.mean())
    assert score.mean_negative_log_probability == pytest.approx(
        -np.log(expected_observed).mean()
    )
    assert score.n_nll_floored == 0
    assert score.switch_brier == pytest.approx(0.0625)
    assert score.greedy_agreement == pytest.approx((0.5 + 1.0 + 1.0) / 3.0)
    assert score.n_exact_ties == 1


def test_nll_floor_does_not_change_raw_probabilities_or_other_scores() -> None:
    """A zero observed probability is floored only inside the logarithm."""
    scenario = _scenario(action=(1,))
    replay = _replay(p_left=(0.0,))
    score = score_model_replay(scenario, replay)

    assert replay.p_left[0] == 0.0
    assert score.mean_observed_probability == 0.0
    assert score.mean_negative_log_probability == pytest.approx(-np.log(1e-12))
    assert score.n_nll_floored == 1
    assert np.isnan(score.switch_brier)
    assert score.greedy_agreement == 0.0


def test_run_benchmark_returns_every_selected_replay_and_score() -> None:
    """Selected observable models and the oracle produce one result per sample."""
    scenarios = build_default_scenarios()[:3]
    models = build_observable_models()[:2]
    result = run_benchmark(scenarios, models, include_oracle=True)

    assert isinstance(result.replays, MappingProxyType)
    assert len(result.replays) == 3 * 3
    assert len(result.scores) == 3 * 3
    assert {key[0] for key in result.replays} == {
        "chance",
        "immediate_persistence",
        "oracle_correct_side",
    }
    assert {key[1] for key in result.replays} == {
        "AT-01",
        "AT-02",
        "AT-03",
    }


def test_run_benchmark_rejects_duplicate_and_reserved_ids() -> None:
    """Mapping keys cannot silently replace duplicate or mislabeled results."""
    scenario = _scenario()
    model = build_observable_models()[0]
    with pytest.raises(ValueError, match="Duplicate scenario"):
        run_benchmark((scenario, scenario), (model,), include_oracle=False)
    with pytest.raises(ValueError, match="Duplicate model"):
        run_benchmark((scenario,), (model, model), include_oracle=False)

    reserved = BenchmarkModelSpec(
        model_id="oracle_correct_side",
        display_label="Unsafe ordinary oracle",
        parameters={},
        replay=model.replay,
    )
    with pytest.raises(ValueError, match="reserved"):
        run_benchmark((scenario,), (reserved,), include_oracle=False)


def test_trial_output_table_is_aligned_and_derives_observed_probability() -> None:
    """The audit table derives scoring fields without modifying model replay."""
    scenario = _scenario()
    replay = _replay()
    score = score_model_replay(scenario, replay)
    result = BenchmarkResult(
        replays=MappingProxyType({(replay.model_id, scenario.scenario_id): replay}),
        scores=(score,),
    )

    table = build_trial_output_table(result, (scenario,))

    assert isinstance(table, pd.DataFrame)
    assert table.shape[0] == 3
    assert table["trial_index"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        table["p_observed"].to_numpy(), [0.5, 0.75, 0.75]
    )
    assert table["decision_drive"].isna().all()
    np.testing.assert_array_equal(table["state__latent"], [0.0, 1.0, 2.0])
    assert "p_observed" not in replay.internal_states


def test_score_table_preserves_deterministic_result_order() -> None:
    """The score table is one row per model/sample pair in result order."""
    scenarios = build_default_scenarios()[:2]
    result = run_benchmark(
        scenarios,
        build_observable_models()[:2],
        include_oracle=False,
    )
    table = build_score_table(result)

    assert table[["model_id", "scenario_id"]].to_records(index=False).tolist() == [
        ("chance", "AT-01"),
        ("immediate_persistence", "AT-01"),
        ("chance", "AT-02"),
        ("immediate_persistence", "AT-02"),
    ]


def test_reordering_scenarios_does_not_change_replay_values() -> None:
    """Each independent sample result is invariant to suite ordering."""
    scenarios = build_default_scenarios()[:3]
    models = build_observable_models()[:3]
    forward = run_benchmark(scenarios, models, include_oracle=False)
    reverse = run_benchmark(tuple(reversed(scenarios)), models, include_oracle=False)

    assert forward.replays.keys() == reverse.replays.keys()
    for key in forward.replays:
        np.testing.assert_allclose(
            forward.replays[key].p_left,
            reverse.replays[key].p_left,
        )
