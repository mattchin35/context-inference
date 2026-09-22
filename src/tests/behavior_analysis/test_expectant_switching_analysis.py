"""Contract tests for post hoc expectant-switching behavioral summaries."""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from src.behavior_analysis.expectant_switching_analysis import (
    incorrect_side_run_lengths,
    post_reward_excursion_rate,
    recurrence_without_context_change,
    return_intervals,
    summarize_expectant_switching_by_reward_count,
)


def test_post_reward_excursion_rate_uses_next_valid_choice():
    actions = np.array([1, -1, 0, 1, 1, 0])
    rewards = np.array([1, np.nan, 0, 1, 1, 0], dtype=float)
    valid = np.array([True, False, True, True, True, True])

    result = post_reward_excursion_rate(actions, rewards, valid)

    # Rewarded valid trials are indices 0, 3, and 4. Their next valid choices
    # differ after 0 and 4, and stay after 3.
    assert result.n_opportunities == 3
    assert result.n_excursions == 2
    assert result.rate == pytest.approx(2 / 3)


def test_post_reward_excursion_rate_is_nan_without_opportunities():
    result = post_reward_excursion_rate(
        actions=np.array([1, 0]),
        rewards=np.array([0.0, 0.0]),
        valid_mask=np.array([True, True]),
    )

    assert result.n_opportunities == 0
    assert result.n_excursions == 0
    assert np.isnan(result.rate)


def test_incorrect_runs_and_return_intervals_count_valid_trials():
    actions = np.array([1, 0, -1, 0, 0, 1, 1, 0])
    correct_choices = np.array([1, 1, -1, 1, 1, 1, 1, 1])
    valid = np.array([True, True, False, True, True, True, True, True])

    npt.assert_array_equal(
        incorrect_side_run_lengths(actions, correct_choices, valid),
        np.array([3, 1]),
    )
    # Each interval includes the valid departure trial through the first valid
    # return trial. The final incomplete excursion has no return interval.
    npt.assert_array_equal(
        return_intervals(actions, correct_choices, valid),
        np.array([4]),
    )


def test_recurrence_requires_return_and_no_context_change():
    actions = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    correct_choices = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    valid = np.ones(actions.shape, dtype=bool)

    result = recurrence_without_context_change(actions, correct_choices, valid)

    # The second incorrect run in the initial left-correct episode recurs.
    # The later incorrect run follows a true context change and starts a new
    # episode, so it is not recurrence of the earlier excursion.
    assert result.n_eligible_completed_excursions == 1
    assert result.n_recurrences == 1
    assert result.fraction == 1.0


def test_reward_count_summary_has_stable_columns_and_counts():
    frame = pd.DataFrame(
        {
            "action": [1, 0, 1, 1, 0, 1],
            "reward": [1, 0, 1, 1, 0, 0],
            "correct_choice": [1, 1, 1, 1, 1, 1],
            "expectancy_reward_count": [0, 1, 1, 2, 3, 3],
            "valid": [True, True, True, True, True, True],
        }
    )

    summary = summarize_expectant_switching_by_reward_count(frame)

    assert list(summary.columns) == [
        "expectancy_reward_count",
        "post_reward_opportunities",
        "post_reward_excursions",
        "post_reward_excursion_rate",
        "incorrect_run_count",
        "mean_incorrect_run_length",
        "completed_return_count",
        "mean_return_interval",
    ]
    assert summary["expectancy_reward_count"].tolist() == [0, 1, 2, 3]
    assert summary["post_reward_opportunities"].sum() == 3
