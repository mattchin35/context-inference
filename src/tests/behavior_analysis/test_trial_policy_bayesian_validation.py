"""Opt-in Bayesian evidence validation preserves legacy arithmetic by default."""

import numpy as np
import pytest

from src.behavior_analysis import trial_features


@pytest.mark.parametrize("value_mode", ["expected_reward", "bayesian_log_odds"])
def test_supported_histories_preserve_existing_numerical_values(value_mode):
    """Value-mode string input yields identical float[T] outputs with validation enabled."""
    arguments = dict(actions=[0, 1, 1, 0, 1], rewards=[1, 0, 1, 0, 1], value_mode=value_mode)
    expected = trial_features.hmm_relative_value(**arguments)
    actual = trial_features.hmm_relative_value(**arguments, validate_outcome_evidence=True)
    np.testing.assert_array_equal(actual, expected)
    if value_mode == "expected_reward":
        np.testing.assert_allclose(actual, [0, -.48, -.8 * 19 / 35, .48, .8 * 19 / 35], atol=1e-12, rtol=0)
    else:
        np.testing.assert_allclose(actual, np.tanh(np.log([1, 1 / 4, 8 / 27, 4, 27 / 8])), atol=1e-12, rtol=0)


@pytest.mark.parametrize("value_mode", ["expected_reward", "bayesian_log_odds"])
@pytest.mark.parametrize("overrides", [dict(active_reward_probability=0, inactive_reward_probability=0),
                                          dict(correct_reward_size=2),
                                          dict(active_reward_probability=np.nan),
                                          dict(active_reward_probability=np.inf)])
def test_impossible_or_nonfinite_outcome_mass_raises_with_row_and_observation(value_mode, overrides):
    """Mode string and parameter dict inputs must reject invalid evidence at row 1."""
    with pytest.raises(ValueError) as error:
        trial_features.hmm_relative_value(actions=[0, 1], rewards=[None, 1], value_mode=value_mode,
                                          validate_outcome_evidence=True, **overrides)
    message = str(error.value)
    assert "row" in message and "1" in message
    assert "action" in message and "reward" in message
    assert "active_reward_probability" in message and "correct_reward_size" in message


@pytest.mark.parametrize("value_mode", ["expected_reward", "bayesian_log_odds"])
def test_disabled_validation_retains_legacy_impossible_evidence_behavior(value_mode):
    """Mode string input preserves default zero-belief collapse for unrelated callers."""
    arguments = dict(actions=[0, 1], rewards=[1, 0], active_reward_probability=0,
                     inactive_reward_probability=0, value_mode=value_mode)
    default = trial_features.hmm_relative_value(**arguments)
    disabled = trial_features.hmm_relative_value(**arguments, validate_outcome_evidence=False)
    np.testing.assert_array_equal(default, [0, 0])
    np.testing.assert_array_equal(disabled, default)


def test_validation_is_keyword_only_and_missing_outcomes_do_not_raise():
    """No inputs; opt-in switch is keyword-only and missing outcomes freeze valid priors."""
    arguments = dict(actions=[0, 1, 1], rewards=[1, None, 0])
    validated = trial_features.hmm_relative_value(**arguments, validate_outcome_evidence=True)
    np.testing.assert_array_equal(validated[1:], [validated[1], validated[1]])
    with pytest.raises(TypeError):
        trial_features.hmm_relative_value([0], [1], .2, .8, 0, 1, 0, "expected_reward", 1, None, True)
