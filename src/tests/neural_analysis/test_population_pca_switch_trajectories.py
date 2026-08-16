from __future__ import annotations

import numpy as np
import pandas as pd

from src.neural_analysis import population_pca_switch_trajectories


def _make_switch_trial_df() -> pd.DataFrame:
    """Build adjacent valid/invalid trials spanning all supported switch types."""

    return pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0, 0, 0, 1, 0],
            "correct": [0, 1, 0, 0, 1, 1, 1, 1],
            "reward": [0, 1, 0, 0, 0, 1, 1, 1],
            "action": [0, 1, 0, 1, 0, 1, 0, 1],
            "choice_time": np.arange(8, dtype=float) + 0.5,
        }
    )


def _make_pca_scores(trial_indices: np.ndarray) -> np.ndarray:
    """Build deterministic trial, choice-window, PC scores for extraction tests."""

    scores = np.zeros((trial_indices.size, 2, 2), dtype=float)
    for score_index, trial_index in enumerate(trial_indices):
        scores[score_index, 0, :] = [10.0 * trial_index, 100.0 + trial_index]
        scores[score_index, 1, :] = [10.0 * trial_index + 1.0, 200.0 + trial_index]
    return scores


def test_select_valid_choice_trial_indices_uses_all_valid_choice_trials():
    """PCA fitting should use every valid trial with finite choice time and action."""
    trial_df = _make_switch_trial_df()
    trial_df.loc[7, "choice_time"] = np.nan

    trial_indices = population_pca_switch_trajectories.select_valid_choice_trial_indices(trial_df)

    np.testing.assert_array_equal(trial_indices, np.arange(6, dtype=int))


def test_select_choice_switch_events_uses_any_adjacent_valid_choice_switch():
    """Switch events should not inherit the older unrewarded-trial switch convention."""
    trial_df = _make_switch_trial_df()

    events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df,
        pre_switch_filter=population_pca_switch_trajectories.SWITCH_PRE_FILTER_ALL,
    )

    assert events["previous_trial_index"].tolist() == [0, 1, 3, 4]
    assert events["next_trial_index"].tolist() == [1, 2, 4, 5]
    assert events["switch_type"].tolist() == [
        "incorrect_to_correct",
        "correct_to_incorrect",
        "incorrect_to_correct",
        "correct_to_correct",
    ]
    assert 2 not in events["previous_trial_index"].tolist()
    assert 5 not in events["previous_trial_index"].tolist()


def test_select_choice_switch_events_filters_pre_switch_outcome():
    """Pre-switch filters should separately select rewarded-correct and omission events."""
    trial_df = _make_switch_trial_df()

    rewarded_events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df,
        pre_switch_filter=population_pca_switch_trajectories.SWITCH_PRE_FILTER_CORRECT_REWARDED,
    )
    omission_events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df,
        pre_switch_filter=population_pca_switch_trajectories.SWITCH_PRE_FILTER_OMISSION,
    )

    assert rewarded_events["previous_trial_index"].tolist() == [1]
    assert omission_events["previous_trial_index"].tolist() == [4]


def test_extract_switch_event_pca_trajectories_returns_four_ordered_points():
    """Each switch event should map to previous pre/post then next pre/post scores."""
    trial_df = _make_switch_trial_df()
    trial_indices = population_pca_switch_trajectories.select_valid_choice_trial_indices(trial_df)
    events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df,
        pre_switch_filter=population_pca_switch_trajectories.SWITCH_PRE_FILTER_ALL,
    )

    trajectories = population_pca_switch_trajectories.extract_switch_event_pca_trajectories(
        pca_scores=_make_pca_scores(trial_indices),
        pca_trial_indices=trial_indices,
        switch_events=events,
    )

    first_event = trajectories.loc[trajectories["event_id"] == 0]
    assert first_event["point_order"].tolist() == [0, 1, 2, 3]
    assert first_event["point_label"].tolist() == [
        "previous_pre_choice",
        "previous_post_choice",
        "next_pre_choice",
        "next_post_choice",
    ]
    np.testing.assert_allclose(first_event["pc1"], np.array([0.0, 1.0, 10.0, 11.0]))
    np.testing.assert_allclose(first_event["pc2"], np.array([100.0, 200.0, 101.0, 201.0]))
    assert trajectories.groupby("event_id").size().eq(4).all()


def test_plot_switch_event_pca_trajectories_uses_three_shared_scale_panels():
    """The plot should show individual and mean paths on identical PC limits."""
    trial_df = _make_switch_trial_df()
    trial_indices = population_pca_switch_trajectories.select_valid_choice_trial_indices(trial_df)
    events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df,
        pre_switch_filter=population_pca_switch_trajectories.SWITCH_PRE_FILTER_ALL,
    )
    trajectories = population_pca_switch_trajectories.extract_switch_event_pca_trajectories(
        pca_scores=_make_pca_scores(trial_indices),
        pca_trial_indices=trial_indices,
        switch_events=events,
    )

    figure, axes = population_pca_switch_trajectories.plot_switch_event_pca_trajectories(
        trajectories
    )

    axes = np.asarray(axes, dtype=object).reshape(-1)
    assert axes.shape == (3,)
    assert [axis.get_title() for axis in axes] == [
        "Incorrect to correct",
        "Correct to incorrect",
        "Correct to correct",
    ]
    assert all(axis.get_xlim() == axes[0].get_xlim() for axis in axes[1:])
    assert all(axis.get_ylim() == axes[0].get_ylim() for axis in axes[1:])
    assert all(len(axis.lines) >= 2 for axis in axes)
    assert all(max(line.get_linewidth() for line in axis.lines) >= 2.0 for axis in axes)
    assert all(min(line.get_alpha() or 1.0 for line in axis.lines) < 0.5 for axis in axes)
    figure.clf()
