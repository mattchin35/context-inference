"""Contract and parity tests for deterministic benchmark model replays."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from scipy.special import expit

from src.behavior_analysis import trial_features
from src.behavior_modeling.agents.agents import HMMRewardDecayRelativeDoubt
from src.behavior_modeling.benchmark_models import (
    OBSERVABLE_MODEL_IDS,
    ORACLE_MODEL_ID,
    BenchmarkModelSpec,
    ModelReplay,
    ObservableHistory,
    build_observable_models,
    replay_oracle,
)
from src.behavior_modeling.benchmark_scenarios import build_default_scenarios
from src.behavior_modeling.expectant_switching import (
    ExpectantSwitchingFeatureConfig,
    replay_expectant_switching,
)
from src.behavior_modeling.parameters.task_config import AgentParams, TaskParams


EXPECTED_OBSERVABLE_MODEL_IDS = (
    "chance",
    "immediate_persistence",
    "wsls",
    "win_switch_lose_stay",
    "simple_probe_persistence",
    "expectancy_persistence_doubt",
    "history_ideal_observer",
    "forgetting_q_learning",
    "probe_persistence_doubt",
    "direct_expectancy_switch_doubt",
    "global_reward_expectancy_doubt",
    "home_referenced_expectancy_doubt",
    "threshold_probe_return_3_3",
    "threshold_probe_return_1_1",
)


def _history(action, reward, valid=None) -> ObservableHistory:
    """Build one immutable observable history for a replay test."""
    action_array = np.asarray(action, dtype=np.int8)
    if valid is None:
        valid = np.ones(action_array.shape, dtype=bool)
    return ObservableHistory(
        action=action_array,
        reward=np.asarray(reward, dtype=float),
        valid=np.asarray(valid, dtype=bool),
    )


def _models_by_id() -> dict[str, BenchmarkModelSpec]:
    """Return the canonical observable registry indexed by stable ID."""
    return {model.model_id: model for model in build_observable_models()}


def _assert_replay_contract(replay: ModelReplay, valid: np.ndarray) -> None:
    """Check aligned, immutable probabilities and states for one replay."""
    n_trials = valid.shape[0]
    assert replay.model_id
    for values in (
        replay.model_value,
        replay.p_left,
        replay.p_right,
        replay.p_switch,
        replay.exact_tie,
    ):
        assert values.shape == (n_trials,)
        assert not values.flags.writeable
    if replay.decision_drive is not None:
        assert replay.decision_drive.shape == (n_trials,)
        assert not replay.decision_drive.flags.writeable

    np.testing.assert_allclose(
        replay.p_left[valid] + replay.p_right[valid], 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        replay.model_value[valid],
        replay.p_left[valid] - replay.p_right[valid],
        atol=1e-12,
    )
    assert np.all(np.isfinite(replay.p_left[valid]))
    assert np.all((replay.p_left[valid] >= 0) & (replay.p_left[valid] <= 1))
    assert np.all(np.isnan(replay.p_left[~valid]))
    assert np.all(~replay.exact_tie[~valid])
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size:
        assert np.isnan(replay.p_switch[valid_indices[0]])
        assert np.all(np.isfinite(replay.p_switch[valid_indices[1:]]))
    assert isinstance(replay.internal_states, MappingProxyType)
    for state in replay.internal_states.values():
        assert state.shape == (n_trials,)
        assert not state.flags.writeable


def test_observable_model_registry_is_complete_ordered_and_immutable() -> None:
    """The registry contains exactly the 14 non-oracle fixed configurations."""
    models = build_observable_models()
    assert OBSERVABLE_MODEL_IDS == EXPECTED_OBSERVABLE_MODEL_IDS
    assert tuple(model.model_id for model in models) == EXPECTED_OBSERVABLE_MODEL_IDS
    assert ORACLE_MODEL_ID not in OBSERVABLE_MODEL_IDS
    assert len({model.model_id for model in models}) == len(models)
    for model in models:
        assert isinstance(model.parameters, MappingProxyType)
        with pytest.raises(TypeError):
            model.parameters["unsafe"] = 1.0


def test_all_models_satisfy_replay_contract_on_default_suite() -> None:
    """Every fixed configuration returns aligned probabilities for all samples."""
    models = build_observable_models()
    for scenario in build_default_scenarios():
        history = _history(scenario.action, scenario.reward, scenario.valid)
        for model in models:
            replay = model.replay(history)
            assert replay.model_id == model.model_id
            _assert_replay_contract(replay, scenario.valid)

        oracle = replay_oracle(history, scenario.correct_side)
        assert oracle.model_id == ORACLE_MODEL_ID
        _assert_replay_contract(oracle, scenario.valid)


def test_reference_models_match_hand_calculations() -> None:
    """Neutral, persistence, and immediate outcome policies are exact."""
    history = _history([1, 1, 0], [1.0, 0.0, 0.0])
    models = _models_by_id()

    np.testing.assert_array_equal(models["chance"].replay(history).model_value, 0.0)
    np.testing.assert_array_equal(
        models["immediate_persistence"].replay(history).model_value,
        [0.0, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        models["wsls"].replay(history).model_value,
        [0.0, 1.0, -1.0],
    )
    np.testing.assert_array_equal(
        models["win_switch_lose_stay"].replay(history).model_value,
        [0.0, -1.0, 1.0],
    )


def test_oracle_is_separate_and_reads_only_correct_side_and_validity() -> None:
    """The labeled ceiling has a distinct hidden-context interface."""
    valid = np.array([True, False, True])
    history = _history([1, -1, 0], [1.0, np.nan, 0.0], valid)
    replay = replay_oracle(history, np.array([1, -1, 0]))
    np.testing.assert_array_equal(replay.model_value[[0, 2]], [1.0, -1.0])
    assert np.isnan(replay.model_value[1])
    assert replay.p_switch[2] == pytest.approx(1.0)
    assert replay.decision_drive is None


def test_existing_exemplar_adapters_match_offline_replay() -> None:
    """Both existing exemplar adapters retain the current implementation."""
    history = _history([1, 1, 1, 0, 0], [1.0, 1.0, 1.0, 0.0, 0.0])
    expected = replay_expectant_switching(
        history.action,
        history.reward,
        history.valid,
        ExpectantSwitchingFeatureConfig(),
    )
    models = _models_by_id()
    simple = models["simple_probe_persistence"].replay(history)
    full = models["expectancy_persistence_doubt"].replay(history)

    np.testing.assert_allclose(
        simple.decision_drive, expected["simple_probe_persistence_drive"]
    )
    np.testing.assert_allclose(
        simple.p_left, expected["simple_probe_persistence_prob_left"]
    )
    np.testing.assert_allclose(
        full.decision_drive, expected["expectancy_persistence_doubt_drive"]
    )
    np.testing.assert_allclose(
        full.p_left, expected["expectancy_persistence_doubt_prob_left"]
    )


def test_forgetting_q_adapter_matches_existing_feature_and_logistic_link() -> None:
    """Forgetting-Q uses the existing pre-update drive and one logistic link."""
    history = _history([1, 1, 0, 0], [1.0, 0.0, 1.0, 0.0])
    expected_drive = trial_features.forgetting_qlearning_relative_value(
        history.action,
        history.reward,
        decay=0.7,
        reward_update_rate=0.3,
    )
    replay = _models_by_id()["forgetting_q_learning"].replay(history)

    np.testing.assert_allclose(replay.decision_drive, expected_drive)
    np.testing.assert_allclose(replay.p_left, expit(expected_drive))
    np.testing.assert_allclose(replay.model_value, np.tanh(expected_drive / 2.0))


def test_history_observer_matches_executable_agent_policy() -> None:
    """The observer adapter reproduces executable pre-update agent outputs."""
    history = _history([1, 1, 0, 0], [1.0, 0.0, 1.0, 0.0])
    agent_params = AgentParams(
        HMM_transition_prob=0.2,
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=1.0,
        HMM_reward_decay_lambda=0.1,
        relative_doubt_lambda=0.5,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )
    agent = HMMRewardDecayRelativeDoubt(
        agent_params,
        TaskParams(mean_correct_reward=1.0, mean_incorrect_reward=0.0),
        rng=np.random.default_rng(0),
    )
    expected_drive = []
    expected_p_left = []
    for action, reward in zip(history.action, history.reward):
        expected_drive.append(agent.value)
        expected_p_left.append(agent.action_dist[1])
        agent.update_params(int(action), float(reward))

    replay = _models_by_id()["history_ideal_observer"].replay(history)
    np.testing.assert_allclose(replay.decision_drive, expected_drive)
    np.testing.assert_allclose(replay.p_left, expected_p_left)
    np.testing.assert_allclose(replay.model_value, np.tanh(replay.decision_drive / 2.0))


def test_proposed_model_checkpoint_probabilities() -> None:
    """Canonical proposed models reproduce their documented probe/return values."""
    scenarios = {item.scenario_id: item for item in build_default_scenarios()}
    models = _models_by_id()

    pr02 = scenarios["PR-02"]
    probe = models["probe_persistence_doubt"].replay(
        _history(pr02.action, pr02.reward)
    )
    assert probe.p_switch[1] == pytest.approx(0.880797, abs=1e-6)
    assert probe.p_switch[2] == pytest.approx(0.289, abs=1e-3)

    pr01 = scenarios["PR-01"]
    pr01_history = _history(pr01.action, pr01.reward)
    direct = models["direct_expectancy_switch_doubt"].replay(pr01_history)
    global_model = models["global_reward_expectancy_doubt"].replay(pr01_history)
    threshold_33 = models["threshold_probe_return_3_3"].replay(pr01_history)
    assert direct.p_switch[3] == pytest.approx(0.805, abs=1e-3)
    assert global_model.p_switch[3] == pytest.approx(0.782, abs=1e-3)
    np.testing.assert_allclose(threshold_33.p_switch[3:7], [0.9, 0.1, 0.1, 0.9])

    threshold_11 = models["threshold_probe_return_1_1"].replay(
        _history(pr02.action, pr02.reward)
    )
    np.testing.assert_allclose(threshold_11.p_switch[1:3], [0.9, 0.9])


def test_home_referenced_expectancy_and_doubt_component_matrix() -> None:
    """Persistent expectancy and omission doubt combine in all matrix cells."""
    model = _models_by_id()["home_referenced_expectancy_doubt"]

    initial = model.replay(_history([1], [1.0]))
    assert initial.p_left[0] == pytest.approx(0.5)

    no_home_omission = model.replay(_history([1, 0], [0.0, 0.0]))
    assert np.isnan(
        no_home_omission.internal_states["expectancy_confirmed_side"][1]
    )
    assert no_home_omission.internal_states["expectancy_choice"][1] == 0.0
    assert no_home_omission.p_left[1] < 0.5

    home_rewards = model.replay(_history([1, 1, 1, 0], [1.0, 1.0, 1.0, 0.0]))
    assert home_rewards.p_left[1] == pytest.approx(0.879755, abs=1e-6)
    assert home_rewards.p_left[2] == pytest.approx(0.821007, abs=1e-6)
    assert home_rewards.p_right[3] == pytest.approx(0.821007, abs=1e-6)

    below_home_omission = model.replay(
        _history([1, 1, 1], [1.0, 0.0, 0.0])
    )
    assert below_home_omission.p_left[2] == pytest.approx(0.769088, abs=1e-6)

    above_home_omission = model.replay(
        _history([1, 1, 1, 1, 0], [1.0, 1.0, 1.0, 0.0, 0.0])
    )
    assert above_home_omission.p_right[4] == pytest.approx(0.909712, abs=1e-6)

    below_probe_omission = model.replay(
        _history([1, 0, 1], [1.0, 0.0, 0.0])
    )
    assert below_probe_omission.p_left[2] == pytest.approx(0.941423, abs=1e-6)


def test_home_referenced_high_expectancy_probe_returns_after_three_omissions() -> None:
    """High expectancy sustains a probe until accumulated doubt overtakes it."""
    scenario = next(
        item for item in build_default_scenarios() if item.scenario_id == "PR-01"
    )
    replay = _models_by_id()["home_referenced_expectancy_doubt"].replay(
        _history(scenario.action, scenario.reward)
    )

    np.testing.assert_allclose(
        replay.p_left[4:7],
        [0.323825, 0.435623, 0.507637],
        atol=1e-6,
    )
    assert replay.p_right[4] > 0.5
    assert replay.p_right[5] > 0.5
    assert replay.p_left[6] > 0.5


def test_home_referenced_rewarded_probe_becomes_new_home() -> None:
    """A rewarded probe resets expectancy and doubt around the newly home side."""
    replay = _models_by_id()["home_referenced_expectancy_doubt"].replay(
        _history([1, 0, 0], [1.0, 1.0, 0.0])
    )

    assert replay.internal_states["expectancy_confirmed_side"][2] == 0
    assert replay.internal_states["expectancy_reward_count"][2] == 1
    assert replay.internal_states["doubt_raw_value"][2] == 0.0
    assert replay.p_right[2] == pytest.approx(0.879755, abs=1e-6)


def test_home_referenced_scenario_checkpoint_directions() -> None:
    """Canonical samples exercise the intended fixed matrix without model switching."""
    scenarios = {item.scenario_id: item for item in build_default_scenarios()}
    model = _models_by_id()["home_referenced_expectancy_doubt"]
    replays = {
        scenario_id: model.replay(
            _history(scenarios[scenario_id].action, scenarios[scenario_id].reward)
        )
        for scenario_id in ("PR-02", "EX-01", "EX-03", "RS-01", "CT-01", "CT-02")
    }

    pr02 = replays["PR-02"]
    assert pr02.p_left[1] > 0.5
    assert pr02.p_left[2] > 0.5

    ex01 = replays["EX-01"]
    assert ex01.p_left[2] > 0.5
    assert ex01.p_left[4] > 0.5
    assert ex01.p_right[5] > 0.5

    ex03 = replays["EX-03"]
    assert ex03.p_right[3] > 0.5
    assert ex03.p_right[4] > 0.5

    rs01 = replays["RS-01"]
    assert rs01.p_right[1] > 0.5
    assert rs01.p_right[2] > 0.5

    ct01 = replays["CT-01"]
    ct02 = replays["CT-02"]
    assert ct01.p_right[4] > 0.5
    assert ct01.p_left[4] == pytest.approx(ct02.p_left[4])


def test_existing_direct_expectancy_model_is_unchanged_by_new_model() -> None:
    """The additive model must not silently alter the earlier direct model."""
    scenario = next(
        item for item in build_default_scenarios() if item.scenario_id == "PR-01"
    )
    replay = _models_by_id()["direct_expectancy_switch_doubt"].replay(
        _history(scenario.action, scenario.reward)
    )

    assert replay.p_switch[3] == pytest.approx(0.804638, abs=1e-6)
    np.testing.assert_allclose(
        replay.p_switch[4:7],
        [0.289403, 0.442737, 0.543697],
        atol=1e-6,
    )


def test_non_oracle_predictions_are_causal_for_every_model() -> None:
    """Current actions, outcomes, future rows, and hidden context cannot leak."""
    history_a = _history([1, 0, 0], [1.0, 0.0, 0.0])
    history_b = _history([1, 1, 1], [1.0, 1.0, 1.0])
    for model in build_observable_models():
        replay_a = model.replay(history_a)
        replay_b = model.replay(history_b)
        for field_name in (
            "decision_drive",
            "model_value",
            "p_left",
            "p_right",
            "p_switch",
            "exact_tie",
        ):
            values_a = getattr(replay_a, field_name)
            values_b = getattr(replay_b, field_name)
            if values_a is not None:
                np.testing.assert_allclose(values_a[1], values_b[1])
        assert replay_a.internal_states.keys() == replay_b.internal_states.keys()
        for state_name in replay_a.internal_states:
            np.testing.assert_allclose(
                replay_a.internal_states[state_name][1],
                replay_b.internal_states[state_name][1],
                equal_nan=True,
            )


def test_invalid_rows_preserve_state_for_every_model() -> None:
    """An aligned invalid row emits no values and has no state effect."""
    with_gap = _history([1, -1, 0], [1.0, np.nan, 0.0], [True, False, True])
    compact = _history([1, 0], [1.0, 0.0])
    for model in build_observable_models():
        replay_gap = model.replay(with_gap)
        replay_compact = model.replay(compact)
        for field_name in (
            "model_value",
            "p_left",
            "p_right",
            "p_switch",
            "exact_tie",
        ):
            gap_values = getattr(replay_gap, field_name)
            compact_values = getattr(replay_compact, field_name)
            np.testing.assert_allclose(
                gap_values[[0, 2]], compact_values, equal_nan=True
            )
        assert np.isnan(replay_gap.p_left[1])
        assert not replay_gap.exact_tie[1]
        for state_name in replay_gap.internal_states:
            np.testing.assert_allclose(
                replay_gap.internal_states[state_name][[0, 2]],
                replay_compact.internal_states[state_name],
                equal_nan=True,
            )


def test_mirrored_histories_have_symmetric_common_outputs() -> None:
    """All observable models remain symmetric under left/right exchange."""
    original = _history([1, 1, 0, 0], [1.0, 0.0, 0.0, 1.0])
    mirrored = _history(1 - original.action, original.reward)
    for model in build_observable_models():
        replay = model.replay(original)
        mirror_replay = model.replay(mirrored)
        np.testing.assert_allclose(
            mirror_replay.model_value, -replay.model_value, atol=1e-12
        )
        np.testing.assert_allclose(
            mirror_replay.p_switch, replay.p_switch, atol=1e-12, equal_nan=True
        )


def test_ct03_hidden_context_cannot_change_observable_prediction() -> None:
    """The paired hidden-context control is identical for every non-oracle model."""
    scenarios = {item.scenario_id: item for item in build_default_scenarios()}
    scenario_a = scenarios["CT-03A"]
    scenario_b = scenarios["CT-03B"]
    for model in build_observable_models():
        replay_a = model.replay(_history(scenario_a.action, scenario_a.reward))
        replay_b = model.replay(_history(scenario_b.action, scenario_b.reward))
        assert replay_a.p_left[3] == pytest.approx(replay_b.p_left[3])
        assert replay_a.p_switch[3] == pytest.approx(replay_b.p_switch[3])


def test_model_state_resets_between_replay_calls() -> None:
    """Reusing a model specification cannot carry state between samples."""
    history = _history([1, 1, 0], [1.0, 1.0, 0.0])
    for model in build_observable_models():
        first = model.replay(history)
        second = model.replay(history)
        np.testing.assert_allclose(first.p_left, second.p_left)
        for state_name in first.internal_states:
            np.testing.assert_allclose(
                first.internal_states[state_name],
                second.internal_states[state_name],
                equal_nan=True,
            )
