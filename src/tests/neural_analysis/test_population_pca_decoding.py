from __future__ import annotations

import numpy as np
import pandas as pd

from src.neural_analysis import population_pca_decoding


def _make_decoding_trial_df() -> pd.DataFrame:
    """Build trial metadata with enough classes for PCA decoder smoke tests."""

    return pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0, 0, 0],
            "correct": [1, 1, 1, 1, 1, 1],
            "reward": [1, 1, 1, 1, 1, 1],
            "action": [0, 0, 0, 1, 1, 1],
            "state_int": [0, 0, 0, 1, 1, 1],
            "start_time": np.arange(6, dtype=float),
            "choice_time": np.arange(6, dtype=float) + 0.5,
            "reward_time": np.arange(6, dtype=float) + 0.8,
        }
    )


def _make_rate_tensor_hz() -> np.ndarray:
    """Build rates with trial, choice-window, and unit axes."""

    rates = np.zeros((6, 2, 4), dtype=float)
    for trial_index in range(6):
        label_value = float(trial_index >= 3)
        rates[trial_index, 0, :] = np.array(
            [label_value, 1.0 - label_value, trial_index, 1.0],
            dtype=float,
        )
        rates[trial_index, 1, :] = np.array(
            [2.0 * label_value, 2.0 * (1.0 - label_value), trial_index + 1.0, 0.5],
            dtype=float,
        )
    return rates


def _make_score_summary_trial_df() -> pd.DataFrame:
    """Build mixed condition/target rows for average PC score tests."""

    return pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0, 0],
            "correct": [1, 1, 1, 0, 0],
            "reward": [1, 1, 1, 0, 0],
            "action": [0, 1, 0, 1, 0],
            "state_int": [0, 0, 1, 0, 1],
            "start_time": np.arange(5, dtype=float),
            "choice_time": np.arange(5, dtype=float) + 0.5,
            "reward_time": np.arange(5, dtype=float) + 0.8,
        }
    )


def _make_score_summary_pca_scores() -> np.ndarray:
    """Build score tensor with trial, pre/post, and PC axes."""

    scores = np.zeros((5, 2, 2), dtype=float)
    scores[:, 0, :] = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [4.0, 4.0],
            [6.0, 6.0],
            [8.0, 8.0],
        ],
        dtype=float,
    )
    scores[:, 1, :] = scores[:, 0, :] + np.array([10.0, 20.0], dtype=float)
    return scores


def test_build_exploratory_pca_decoder_trial_bins_preserves_choice_windows():
    """Exploratory PCA scores should be adapted to the existing decoder-bin contract."""
    trial_df = _make_decoding_trial_df()

    trial_bins, pca_result = population_pca_decoding.build_exploratory_pca_decoder_trial_bins(
        rate_tensor_hz=_make_rate_tensor_hz(),
        trial_df=trial_df,
        trial_indices=np.arange(6, dtype=int),
        n_components=2,
    )

    assert len(trial_bins) == trial_df.shape[0]
    assert pca_result.scores.shape == (6, 2, 2)
    assert trial_bins[0]["trial_ix"] == 0
    assert trial_bins[0]["binned_spikes"].shape == (2, 2)
    np.testing.assert_allclose(
        trial_bins[0]["bin_edges"],
        np.array([0.0, 0.5, 1.0], dtype=float),
    )
    np.testing.assert_allclose(trial_bins[0]["bin_states"], np.array([0.0, 0.0]))
    np.testing.assert_allclose(trial_bins[3]["bin_choices"], np.array([1.0, 1.0]))


def test_run_exploratory_pca_choice_decoding_supports_state_and_action_targets():
    """Exploratory PCA decoding should return pre/post rows for both supported targets."""
    trial_df = _make_decoding_trial_df()

    state_results = population_pca_decoding.run_exploratory_pca_choice_decoding(
        rate_tensor_hz=_make_rate_tensor_hz(),
        trial_df=trial_df,
        trial_indices=np.arange(6, dtype=int),
        n_components=2,
        condition_names=["correct_rewarded"],
        target="state_int",
        cv=2,
        n_permutations=2,
        random_state=42,
    )
    action_results = population_pca_decoding.run_exploratory_pca_choice_decoding(
        rate_tensor_hz=_make_rate_tensor_hz(),
        trial_df=trial_df,
        trial_indices=np.arange(6, dtype=int),
        n_components=2,
        condition_names=["correct_rewarded"],
        target="action",
        cv=2,
        n_permutations=2,
        random_state=42,
    )

    assert state_results["target"].tolist() == ["state_int", "state_int"]
    assert action_results["target"].tolist() == ["action", "action"]
    assert state_results["window"].tolist() == ["pre_choice", "post_choice"]
    assert set(state_results["status"]) == {"ok"}
    assert set(action_results["status"]) == {"ok"}
    assert set(state_results["mode"]) == {population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY}
    assert np.all(state_results["n_pcs_fit"].to_numpy(dtype=int) == 2)


def test_run_rigorous_pca_choice_decoding_uses_matching_result_schema():
    """Rigorous mode should expose the same display columns while fitting PCA inside CV."""
    trial_df = _make_decoding_trial_df()

    results = population_pca_decoding.run_rigorous_pca_choice_decoding(
        rate_tensor_hz=_make_rate_tensor_hz(),
        trial_df=trial_df,
        trial_indices=np.arange(6, dtype=int),
        n_components=2,
        condition_names=["correct_rewarded"],
        target="state_int",
        cv=2,
        n_permutations=2,
        random_state=42,
    )

    assert results["mode"].tolist() == [
        population_pca_decoding.PCA_DECODING_MODE_RIGOROUS,
        population_pca_decoding.PCA_DECODING_MODE_RIGOROUS,
    ]
    assert results["window"].tolist() == ["pre_choice", "post_choice"]
    assert set(results["status"]) == {"ok"}
    for column_name in population_pca_decoding.PCA_DECODING_DISPLAY_COLUMNS:
        assert column_name in results.columns


def test_summarize_pca_decoding_results_keeps_display_columns():
    """The webapp display table should hide implementation-only metadata."""
    raw_results = pd.DataFrame(
        [
            {
                "condition": "correct_rewarded",
                "window": "pre_choice",
                "target": "state_int",
                "mode": population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY,
                "status": "ok",
                "reason": "",
                "n_samples": 6,
                "n_classes": 2,
                "cv_score": 0.75,
                "cv_pvalue": 0.25,
                "permutation_score_mean": 0.5,
                "permutation_score_std": 0.1,
                "n_pcs_requested": 5,
                "n_pcs_fit": 2,
                "internal": "hidden",
            }
        ]
    )

    display_results = population_pca_decoding.summarize_pca_decoding_results(raw_results)

    assert display_results.columns.tolist() == population_pca_decoding.PCA_DECODING_DISPLAY_COLUMNS
    assert display_results.iloc[0]["condition"] == "correct_rewarded"


def test_plot_pca_decoding_pre_post_scores_groups_bars_side_by_side():
    """Pre/post bars should sit beside each other, with larger gaps between conditions."""
    results = pd.DataFrame(
        [
            {
                "condition": "correct_rewarded",
                "window": "pre_choice",
                "target": "state_int",
                "mode": population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY,
                "status": "ok",
                "reason": "",
                "n_samples": 8,
                "n_classes": 2,
                "cv_score": 0.6,
                "cv_pvalue": 0.2,
                "permutation_score_mean": 0.5,
                "permutation_score_std": 0.1,
            },
            {
                "condition": "correct_rewarded",
                "window": "post_choice",
                "target": "state_int",
                "mode": population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY,
                "status": "ok",
                "reason": "",
                "n_samples": 8,
                "n_classes": 2,
                "cv_score": 0.8,
                "cv_pvalue": 0.05,
                "permutation_score_mean": 0.5,
                "permutation_score_std": 0.1,
            },
            {
                "condition": "incorrect",
                "window": "pre_choice",
                "target": "state_int",
                "mode": population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY,
                "status": "ok",
                "reason": "",
                "n_samples": 6,
                "n_classes": 2,
                "cv_score": 0.55,
                "cv_pvalue": 0.4,
                "permutation_score_mean": 0.5,
                "permutation_score_std": 0.1,
            },
            {
                "condition": "incorrect",
                "window": "post_choice",
                "target": "state_int",
                "mode": population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY,
                "status": "failed",
                "reason": "insufficient_classes",
                "n_samples": 1,
                "n_classes": 1,
                "cv_score": np.nan,
                "cv_pvalue": np.nan,
                "permutation_score_mean": np.nan,
                "permutation_score_std": np.nan,
            },
        ]
    )

    figure, axis = population_pca_decoding.plot_pca_decoding_pre_post_scores(results)

    bar_centers = np.array(
        [patch.get_x() + patch.get_width() / 2.0 for patch in axis.patches],
        dtype=float,
    )
    bar_heights = np.array([patch.get_height() for patch in axis.patches], dtype=float)
    assert bar_centers.shape == (3,)
    np.testing.assert_allclose(np.sort(bar_heights), np.array([0.55, 0.6, 0.8]))
    within_condition_spacing = abs(bar_centers[1] - bar_centers[0])
    between_condition_spacing = abs(bar_centers[2] - np.mean(bar_centers[:2]))
    assert between_condition_spacing > within_condition_spacing * 2.0
    assert axis.get_ylim()[1] == 1.0
    figure.clf()


def test_summarize_pca_scores_by_condition_and_target_groups_pre_post_target_values():
    """Average PC summaries should group by base condition, window, and target value."""
    trial_df = _make_score_summary_trial_df()

    summary = population_pca_decoding.summarize_pca_scores_by_condition_and_target(
        pca_scores=_make_score_summary_pca_scores(),
        trial_df=trial_df,
        trial_indices=np.arange(5, dtype=int),
        condition_names=["correct_rewarded", "incorrect"],
        target="state_int",
    )

    assert summary.shape[0] == 8
    correct_state0_pre = summary.loc[
        (summary["condition"] == "correct_rewarded")
        & (summary["window"] == "pre_choice")
        & (summary["target_value"] == 0.0)
    ].iloc[0]
    assert correct_state0_pre["target_label"] == "state 0"
    assert correct_state0_pre["n_trials"] == 2
    assert correct_state0_pre["pc1_mean"] == 1.0
    assert correct_state0_pre["pc2_mean"] == 1.0

    incorrect_state1_post = summary.loc[
        (summary["condition"] == "incorrect")
        & (summary["window"] == "post_choice")
        & (summary["target_value"] == 1.0)
    ].iloc[0]
    assert incorrect_state1_post["n_trials"] == 1
    assert incorrect_state1_post["pc1_mean"] == 18.0
    assert incorrect_state1_post["pc2_mean"] == 28.0


def test_summarize_pca_scores_by_condition_and_target_requires_two_pcs():
    """PC score-space summaries require PC1 and PC2."""
    trial_df = _make_score_summary_trial_df()

    try:
        population_pca_decoding.summarize_pca_scores_by_condition_and_target(
            pca_scores=np.zeros((5, 2, 1), dtype=float),
            trial_df=trial_df,
            trial_indices=np.arange(5, dtype=int),
            condition_names=["correct_rewarded"],
            target="state_int",
        )
    except ValueError as error:
        assert "at least two PCs" in str(error)
    else:
        raise AssertionError("Expected one-PC score summaries to raise ValueError.")


def test_extract_pca_score_points_by_condition_and_target_preserves_overlapping_memberships():
    """Raw PC points should keep condition memberships even when one trial appears in multiple masks."""
    trial_df = _make_score_summary_trial_df()

    raw_points = population_pca_decoding.extract_pca_score_points_by_condition_and_target(
        pca_scores=_make_score_summary_pca_scores(),
        trial_df=trial_df,
        trial_indices=np.arange(5, dtype=int),
        condition_names=["incorrect", "switch"],
        target="state_int",
    )

    assert raw_points.columns.tolist() == population_pca_decoding.PCA_RAW_SCORE_COLUMNS
    assert raw_points.shape[0] == 6
    duplicated_trial_rows = raw_points.loc[raw_points["trial_index"] == 3]
    assert duplicated_trial_rows.shape[0] == 4
    assert set(duplicated_trial_rows["condition"]) == {"incorrect", "switch"}

    switch_pre = raw_points.loc[
        (raw_points["condition"] == "switch")
        & (raw_points["trial_index"] == 3)
        & (raw_points["window"] == "pre_choice")
    ].iloc[0]
    assert switch_pre["target_label"] == "state 0"
    assert switch_pre["pc1"] == 6.0
    assert switch_pre["pc2"] == 6.0


def test_plot_average_pca_scores_by_condition_and_target_returns_pre_post_axes():
    """Average PC score plots should have separate pre-choice and post-choice panels."""
    trial_df = _make_score_summary_trial_df()
    summary = population_pca_decoding.summarize_pca_scores_by_condition_and_target(
        pca_scores=_make_score_summary_pca_scores(),
        trial_df=trial_df,
        trial_indices=np.arange(5, dtype=int),
        condition_names=["correct_rewarded", "incorrect"],
        target="action",
    )

    figure, axes = population_pca_decoding.plot_average_pca_scores_by_condition_and_target(summary)

    axes = np.asarray(axes, dtype=object).reshape(-1)
    assert axes.shape == (2,)
    assert axes[0].get_title() == "Pre-choice (-0.5 to 0 s)"
    assert axes[1].get_title() == "Post-choice (0 to 0.5 s)"
    assert len(axes[0].collections) > 0
    assert len(axes[1].collections) > 0
    figure.clf()


def test_plot_average_pca_scores_by_condition_and_target_can_overlay_raw_points():
    """Raw PC score points should be drawn transparently beneath larger average markers."""
    trial_df = _make_score_summary_trial_df()
    pca_scores = _make_score_summary_pca_scores()
    summary = population_pca_decoding.summarize_pca_scores_by_condition_and_target(
        pca_scores=pca_scores,
        trial_df=trial_df,
        trial_indices=np.arange(5, dtype=int),
        condition_names=["incorrect", "switch"],
        target="state_int",
    )
    raw_points = population_pca_decoding.extract_pca_score_points_by_condition_and_target(
        pca_scores=pca_scores,
        trial_df=trial_df,
        trial_indices=np.arange(5, dtype=int),
        condition_names=["incorrect", "switch"],
        target="state_int",
    )

    figure, axes = population_pca_decoding.plot_average_pca_scores_by_condition_and_target(
        summary,
        raw_score_df=raw_points,
    )

    axes = np.asarray(axes, dtype=object).reshape(-1)
    pre_collections = axes[0].collections
    raw_alpha = pre_collections[0].get_alpha()
    average_alpha = pre_collections[-1].get_alpha()
    raw_sizes = pre_collections[0].get_sizes()
    average_sizes = pre_collections[-1].get_sizes()
    assert raw_alpha < 0.5
    assert average_alpha is None or average_alpha == 1.0
    assert float(np.max(raw_sizes)) < float(np.max(average_sizes))
    figure.clf()
