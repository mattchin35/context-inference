"""Contracts for candidate expectant-switching models used on real sessions."""

import numpy as np
import pytest

from src.behavior_modeling.benchmark_models import (
    ObservableHistory,
    build_observable_models,
)
from src.behavior_modeling.expectant_switching import (
    CandidateExpectantSwitchingConfig,
    HomeReferencedExpectancyDoubtParams,
    ProbePersistenceDoubtParams,
    replay_candidate_expectant_switching,
    replay_home_referenced_expectancy_doubt,
    replay_probe_persistence_doubt,
)


def _benchmark_models_by_id():
    """Return benchmark model specifications keyed by stable model ID."""
    return {model.model_id: model for model in build_observable_models()}


def test_candidate_replay_matches_benchmark_models() -> None:
    """Production and benchmark paths must use identical fixed equations."""
    actions = np.array([1, 1, 1, 0, 0, 1], dtype=int)
    rewards = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    valid = np.ones(actions.shape, dtype=bool)
    replay = replay_candidate_expectant_switching(
        actions,
        rewards,
        valid,
        CandidateExpectantSwitchingConfig(),
    )
    history = ObservableHistory(actions, rewards, valid)
    benchmark_models = _benchmark_models_by_id()

    probe = benchmark_models["probe_persistence_doubt"].replay(history)
    home = benchmark_models["home_referenced_expectancy_doubt"].replay(history)
    np.testing.assert_allclose(
        replay["probe_persistence_doubt_drive"],
        probe.decision_drive,
    )
    np.testing.assert_allclose(
        replay["probe_persistence_doubt_value"],
        probe.model_value,
    )
    np.testing.assert_allclose(
        replay["probe_persistence_doubt_prob_left"],
        probe.p_left,
    )
    np.testing.assert_allclose(
        replay["home_referenced_expectancy_doubt_drive"],
        home.decision_drive,
    )
    np.testing.assert_allclose(
        replay["home_referenced_expectancy_doubt_value"],
        home.model_value,
    )
    np.testing.assert_allclose(
        replay["home_referenced_expectancy_doubt_prob_left"],
        home.p_left,
    )


def test_candidate_replay_is_causal_and_skips_invalid_rows() -> None:
    """Invalid rows emit NaN, preserve state, and future changes cannot leak."""
    actions = np.array([1, -1, 0, 1], dtype=int)
    rewards = np.array([1.0, np.nan, 0.0, 1.0])
    valid = np.array([True, False, True, True])
    config = CandidateExpectantSwitchingConfig(
        probe=ProbePersistenceDoubtParams(),
        home=HomeReferencedExpectancyDoubtParams(),
    )
    replay = replay_candidate_expectant_switching(actions, rewards, valid, config)

    assert all(np.isnan(values[1]) for values in replay.values())
    assert replay["probe_persistence_doubt_previous_choice"][2] == 1.0
    assert replay["home_referenced_expectancy_doubt_confirmed_side"][2] == 1.0
    assert replay["home_referenced_expectancy_doubt_reward_count"][2] == 1.0

    changed_actions = actions.copy()
    changed_rewards = rewards.copy()
    changed_actions[2:] = [1, 0]
    changed_rewards[2:] = [1.0, 0.0]
    changed = replay_candidate_expectant_switching(
        changed_actions,
        changed_rewards,
        valid,
        config,
    )
    for column_name in replay:
        np.testing.assert_allclose(
            replay[column_name][2],
            changed[column_name][2],
            equal_nan=True,
        )


@pytest.mark.parametrize("replay,params,prefix", [
    (replay_probe_persistence_doubt, ProbePersistenceDoubtParams(), "probe_persistence_doubt"),
    (replay_home_referenced_expectancy_doubt, HomeReferencedExpectancyDoubtParams(),
     "home_referenced_expectancy_doubt"),
])
def test_optional_update_mask_predicts_missing_reward_without_updating(replay, params, prefix):
    """Replay/params/ID inputs separate prediction bool[T] from update bool[T]."""
    actions = np.array([1, 0, 0, 1])
    rewards = np.array([1., np.nan, 0., 1.])
    valid = np.ones(4, bool)
    update = np.array([True, False, True, True])
    actual = replay(actions, rewards, valid, params, update_mask=update)
    compressed = replay(actions[update], rewards[update], valid[update], params)
    np.testing.assert_array_equal(actual[f"{prefix}_drive"][update], compressed[f"{prefix}_drive"])
    assert np.isfinite(actual[f"{prefix}_prob_left"]).all()
    assert actual[f"{prefix}_drive"][1] == actual[f"{prefix}_drive"][2]
    for name in actual:
        np.testing.assert_array_equal(actual[name][1], actual[name][2])
    with pytest.raises(ValueError, match="reward"):
        replay(actions, rewards, valid, params)  # Legacy default still requires complete outcomes.


@pytest.mark.parametrize("update", [np.ones(2, bool), np.array([True, True, True])])
def test_update_mask_must_be_aligned_and_subset_of_prediction_rows(update):
    """Invalid bool[N] updates reject shape mismatches and updating excluded rows."""
    with pytest.raises(ValueError, match="update_mask"):
        replay_probe_persistence_doubt(
            np.array([1, 0, 1]), np.array([1., 0., 1.]), np.array([True, False, True]),
            ProbePersistenceDoubtParams(), update_mask=update)

