"""Contract tests for the expectant-switching exemplar components."""

import numpy as np
import numpy.testing as npt
import pytest

from src.behavior_modeling.counterfactual_doubt import (
    CounterfactualDoubtState,
    counterfactual_doubt_choice_value,
    counterfactual_doubt_raw_value,
    update_counterfactual_doubt,
)
from src.behavior_modeling.expectant_switching import (
    ExpectancyCurveParams,
    ExpectancyPersistenceDoubtParams,
    ExpectantSwitchingFeatureConfig,
    PreviousOutcomeState,
    RewardConfirmedExpectancyState,
    SimpleProbePersistenceParams,
    expectancy_strength,
    previous_outcome_signals,
    read_expectancy_persistence_doubt,
    read_simple_probe_persistence,
    replay_expectant_switching,
    update_previous_outcome,
    update_reward_confirmed_expectancy,
)


def make_feature_config() -> ExpectantSwitchingFeatureConfig:
    """Return the documented initial exemplar configuration."""
    return ExpectantSwitchingFeatureConfig(
        simple=SimpleProbePersistenceParams(
            persistence_weight=1.0,
            probe_weight=1.0,
        ),
        full=ExpectancyPersistenceDoubtParams(
            curve=ExpectancyCurveParams(threshold=3.0, scale=1.0),
            doubt_lambda=0.5,
            persistence_weight=1.0,
            expectancy_weight=1.0,
            doubt_weight=1.0,
        ),
    )


def test_expectancy_curve_matches_documented_values_and_bounds():
    params = ExpectancyCurveParams(threshold=3.0, scale=1.0)

    actual = expectancy_strength(np.arange(6), params)
    expected = np.array(
        [0.00247262, 0.01798621, 0.11920292, 0.5, 0.88079708, 0.98201379]
    )

    npt.assert_allclose(actual, expected)
    assert np.all((actual >= 0.0) & (actual <= 1.0))


@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_expectancy_curve_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        ExpectancyCurveParams(threshold=3.0, scale=scale)


def test_previous_outcome_signals_are_pretrial_and_reward_gated():
    state = PreviousOutcomeState()
    assert previous_outcome_signals(state) == (0.0, 0.0)

    rewarded_left = update_previous_outcome(state, action=1, reward=1.0)
    assert previous_outcome_signals(rewarded_left) == (1.0, -1.0)

    omitted_right = update_previous_outcome(rewarded_left, action=0, reward=0.0)
    assert previous_outcome_signals(omitted_right) == (-1.0, 0.0)


def test_reward_confirmed_expectancy_uses_only_observed_rewards():
    state = RewardConfirmedExpectancyState()

    state = update_reward_confirmed_expectancy(state, action=1, reward=1.0)
    assert state == RewardConfirmedExpectancyState(confirmed_side=1, reward_count=1)

    state = update_reward_confirmed_expectancy(state, action=1, reward=2.0)
    assert state == RewardConfirmedExpectancyState(confirmed_side=1, reward_count=2)

    state = update_reward_confirmed_expectancy(state, action=0, reward=0.0)
    assert state == RewardConfirmedExpectancyState(confirmed_side=1, reward_count=2)

    state = update_reward_confirmed_expectancy(state, action=0, reward=1.0)
    assert state == RewardConfirmedExpectancyState(confirmed_side=0, reward_count=1)


def test_reward_confirmed_expectancy_does_not_accept_hidden_context():
    state = RewardConfirmedExpectancyState()

    with pytest.raises(TypeError):
        update_reward_confirmed_expectancy(
            state,
            action=1,
            reward=1.0,
            hidden_context=1,
        )


def test_counterfactual_doubt_rules_and_choice_oriented_sign():
    state = CounterfactualDoubtState()
    state = update_counterfactual_doubt(state, action=1, reward=0.0)
    state = update_counterfactual_doubt(state, action=0, reward=0.0)
    state = update_counterfactual_doubt(state, action=0, reward=0.0)

    assert state == CounterfactualDoubtState(right_omissions=2.0, left_omissions=1.0)
    expected_raw = (1.0 - np.exp(-0.5)) - (1.0 - np.exp(-1.0))
    assert counterfactual_doubt_raw_value(state, omission_lambda=0.5) == pytest.approx(
        expected_raw
    )
    assert counterfactual_doubt_choice_value(
        state, omission_lambda=0.5
    ) == pytest.approx(-expected_raw)

    reset = update_counterfactual_doubt(state, action=1, reward=1.0)
    assert reset == CounterfactualDoubtState()


def test_counterfactual_doubt_is_left_right_symmetric():
    left = CounterfactualDoubtState(right_omissions=1.0, left_omissions=3.0)
    right = CounterfactualDoubtState(right_omissions=3.0, left_omissions=1.0)

    assert counterfactual_doubt_raw_value(left, 0.5) == pytest.approx(
        -counterfactual_doubt_raw_value(right, 0.5)
    )
    assert counterfactual_doubt_choice_value(left, 0.5) == pytest.approx(
        -counterfactual_doubt_choice_value(right, 0.5)
    )


def test_simple_readout_uses_direct_probability_mapping():
    params = SimpleProbePersistenceParams(
        persistence_weight=2.0,
        probe_weight=0.5,
    )

    result = read_simple_probe_persistence(
        persistence_signal=1.0,
        probe_signal=-1.0,
        params=params,
    )

    assert result.decision_drive == pytest.approx(1.5)
    assert result.signed_value == pytest.approx(np.tanh(1.5))
    assert result.p_left == pytest.approx((1.0 + np.tanh(1.5)) / 2.0)
    assert result.p_right == pytest.approx(1.0 - result.p_left)


def test_all_ones_simple_model_is_indifferent_after_reward():
    result = read_simple_probe_persistence(
        persistence_signal=1.0,
        probe_signal=-1.0,
        params=SimpleProbePersistenceParams(),
    )

    assert result.decision_drive == 0.0
    assert result.signed_value == 0.0
    assert result.p_left == 0.5
    assert result.p_right == 0.5


def test_full_readout_reduces_to_simple_with_fixed_expectancy():
    simple = read_simple_probe_persistence(
        persistence_signal=-1.0,
        probe_signal=1.0,
        params=SimpleProbePersistenceParams(
            persistence_weight=0.7,
            probe_weight=1.2,
        ),
    )
    full = read_expectancy_persistence_doubt(
        persistence_signal=-1.0,
        expectant_switch_signal=1.0,
        doubt_choice_signal=-0.8,
        params=ExpectancyPersistenceDoubtParams(
            persistence_weight=0.7,
            expectancy_weight=1.2,
            doubt_weight=0.0,
        ),
    )

    assert full == simple


def test_batch_replay_is_pretrial_aligned_and_skips_invalid_rows():
    actions = np.array([1, -1, 1, 0], dtype=int)
    rewards = np.array([1.0, np.nan, 0.0, 1.0])
    valid = np.array([True, False, True, True])

    result = replay_expectant_switching(
        actions=actions,
        rewards=rewards,
        valid_mask=valid,
        config=make_feature_config(),
    )

    assert result["previous_choice"][0] == 0.0
    assert result["expectancy_reward_count"][0] == 0.0
    assert np.isnan(result["previous_choice"][1])
    assert np.isnan(result["simple_probe_persistence_value"][1])
    assert result["previous_choice"][2] == 1.0
    assert result["previous_reward"][2] == 1.0
    assert result["expectancy_reward_count"][2] == 1.0
    assert result["previous_choice"][3] == 1.0
    assert result["previous_reward"][3] == 0.0
    assert result["expectancy_reward_count"][3] == 1.0


def test_batch_replay_has_no_current_or_future_trial_leakage():
    actions = np.array([1, 1, 0, 0], dtype=int)
    rewards = np.array([1.0, 1.0, 0.0, 1.0])
    valid = np.ones(4, dtype=bool)
    config = make_feature_config()

    baseline = replay_expectant_switching(actions, rewards, valid, config)
    changed_actions = actions.copy()
    changed_rewards = rewards.copy()
    changed_actions[2:] = 1 - changed_actions[2:]
    changed_rewards[2:] = 1.0 - changed_rewards[2:]
    changed = replay_expectant_switching(
        changed_actions,
        changed_rewards,
        valid,
        config,
    )

    for column in baseline:
        assert baseline[column][2] == pytest.approx(changed[column][2], nan_ok=True)


def test_fresh_states_do_not_share_mutable_data():
    first = update_reward_confirmed_expectancy(
        RewardConfirmedExpectancyState(), action=1, reward=1.0
    )
    second = RewardConfirmedExpectancyState()

    assert first.reward_count == 1
    assert second == RewardConfirmedExpectancyState()
