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
