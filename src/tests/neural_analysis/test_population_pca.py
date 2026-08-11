from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src.neural_analysis import population_pca
from src.neural_analysis import psth_webapp
from src.neural_analysis import unit_spike_plotting


def _make_spike_group() -> nap.TsGroup:
    """Build a tiny spike group for population PCA tests.

    Args:
        None.

    Returns:
        nap.TsGroup: Spike group with three integer unit ids. Spike times are
        in seconds.
    """

    return nap.TsGroup(
        {
            1: nap.Ts(t=np.array([0.05, 0.15, 1.05, 1.15], dtype=float)),
            2: nap.Ts(t=np.array([0.15, 0.35, 1.25, 1.45], dtype=float)),
            3: nap.Ts(t=np.array([0.05, 0.25, 1.05, 1.35], dtype=float)),
        }
    )


def _make_trial_df() -> pd.DataFrame:
    """Build a tiny trial dataframe with choice and start alignment times.

    Args:
        None.

    Returns:
        pd.DataFrame: Trial table with two rows. Time columns are in seconds.
    """

    return pd.DataFrame(
        {
            "start_time": [0.0, 1.0],
            "choice_time": [0.1, 1.1],
            "led_on_time": [0.2, 1.2],
            "action": [0, 1],
        }
    )


def test_build_trial_unit_rate_tensor_returns_expected_shape_and_hz_values():
    rate_tensor_hz, bin_centers_s = population_pca.build_trial_unit_rate_tensor(
        spike_group=_make_spike_group(),
        unit_ids=np.array([1, 2], dtype=int),
        trial_df=_make_trial_df(),
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.4),
        bin_size_s=0.1,
    )

    assert rate_tensor_hz.shape == (2, 4, 2)
    np.testing.assert_allclose(bin_centers_s, np.array([0.05, 0.15, 0.25, 0.35]))
    np.testing.assert_allclose(
        rate_tensor_hz[0],
        np.array(
            [
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 0.0],
                [0.0, 10.0],
            ]
        ),
    )


def test_build_trial_unit_rate_tensor_aligns_to_choice_time():
    rate_tensor_hz, bin_centers_s = population_pca.build_trial_unit_rate_tensor(
        spike_group=_make_spike_group(),
        unit_ids=np.array([1], dtype=int),
        trial_df=_make_trial_df(),
        trial_indices=np.array([0], dtype=int),
        alignment_event="choice_time",
        window=(-0.1, 0.2),
        bin_size_s=0.1,
    )

    assert rate_tensor_hz.shape == (1, 3, 1)
    np.testing.assert_allclose(bin_centers_s, np.array([-0.05, 0.05, 0.15]))
    np.testing.assert_allclose(rate_tensor_hz[0, :, 0], np.array([10.0, 10.0, 0.0]))


def test_build_trial_unit_rate_tensor_rejects_invalid_window_and_bin_size():
    with pytest.raises(ValueError, match="bin_size_s"):
        population_pca.build_trial_unit_rate_tensor(
            spike_group=_make_spike_group(),
            unit_ids=np.array([1], dtype=int),
            trial_df=_make_trial_df(),
            trial_indices=np.array([0], dtype=int),
            alignment_event="start_time",
            window=(0.0, 0.4),
            bin_size_s=0.0,
        )

    with pytest.raises(ValueError, match="window"):
        population_pca.build_trial_unit_rate_tensor(
            spike_group=_make_spike_group(),
            unit_ids=np.array([1], dtype=int),
            trial_df=_make_trial_df(),
            trial_indices=np.array([0], dtype=int),
            alignment_event="start_time",
            window=(0.4, 0.0),
            bin_size_s=0.1,
        )


def test_prepare_pca_observation_matrix_zscores_and_handles_constant_units():
    rate_tensor_hz = np.array(
        [
            [[1.0, 10.0], [3.0, 10.0]],
            [[5.0, 10.0], [7.0, 10.0]],
        ]
    )

    observations, unit_mean_hz, unit_scale_hz = population_pca.prepare_pca_observation_matrix(
        rate_tensor_hz,
        normalization=population_pca.PCA_NORMALIZATION_ZSCORE,
    )

    np.testing.assert_allclose(unit_mean_hz, np.array([4.0, 10.0]))
    assert unit_scale_hz[0] > 0
    assert unit_scale_hz[1] == 1.0
    np.testing.assert_allclose(observations[:, 1], np.zeros(4))
    np.testing.assert_allclose(observations[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(observations[:, 0].std(), 1.0)


def test_prepare_pca_observation_matrix_mean_centers_without_rescaling():
    rate_tensor_hz = np.array([[[1.0, 10.0], [3.0, 30.0]]])

    observations, unit_mean_hz, unit_scale_hz = population_pca.prepare_pca_observation_matrix(
        rate_tensor_hz,
        normalization=population_pca.PCA_NORMALIZATION_MEAN_CENTER,
    )

    np.testing.assert_allclose(unit_mean_hz, np.array([2.0, 20.0]))
    np.testing.assert_allclose(unit_scale_hz, np.ones(2))
    np.testing.assert_allclose(observations, np.array([[-1.0, -10.0], [1.0, 10.0]]))


def test_fit_population_pca_returns_scores_and_explained_variance_shapes():
    rate_tensor_hz, _ = population_pca.build_trial_unit_rate_tensor(
        spike_group=_make_spike_group(),
        unit_ids=np.array([1, 2, 3], dtype=int),
        trial_df=_make_trial_df(),
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.4),
        bin_size_s=0.1,
    )

    result = population_pca.fit_population_pca(
        rate_tensor_hz,
        n_components=2,
        normalization=population_pca.PCA_NORMALIZATION_ZSCORE,
    )

    assert result.scores.shape == (2, 4, 2)
    assert result.explained_variance_ratio.shape == (2,)
    assert result.cumulative_explained_variance.shape == (2,)
    assert np.all(np.diff(result.cumulative_explained_variance) >= -1e-12)


def test_profile_population_pca_pipeline_reports_timings_and_dimensions():
    fake_times = iter([0.0, 1.0, 3.0, 6.0])

    profile = population_pca.profile_population_pca_pipeline(
        spike_group=_make_spike_group(),
        unit_ids=np.array([1, 2, 3], dtype=int),
        trial_df=_make_trial_df(),
        trial_indices=np.array([0, 1], dtype=int),
        alignment_event="start_time",
        window=(0.0, 0.4),
        bin_size_s=0.1,
        n_components=2,
        normalization=population_pca.PCA_NORMALIZATION_ZSCORE,
        timer=lambda: next(fake_times),
        print_summary=False,
    )

    assert profile.rate_tensor_shape == (2, 4, 3)
    assert profile.observation_shape == (8, 3)
    assert profile.fitted_component_count == 2
    assert profile.timings_s == {
        "binning": 1.0,
        "normalization": 2.0,
        "pca_fit": 3.0,
        "total": 6.0,
    }


def test_fit_population_pca_rejects_too_few_units_or_observations():
    with pytest.raises(ValueError, match="At least two units"):
        population_pca.fit_population_pca(
            np.zeros((2, 4, 1), dtype=float),
            n_components=1,
            normalization=population_pca.PCA_NORMALIZATION_ZSCORE,
        )

    with pytest.raises(ValueError, match="At least two observations"):
        population_pca.fit_population_pca(
            np.zeros((1, 1, 2), dtype=float),
            n_components=1,
            normalization=population_pca.PCA_NORMALIZATION_ZSCORE,
        )


def test_plot_trial_behavior_and_population_pca_draws_pc_lines_and_behavior_events():
    pca_scores = np.zeros((2, 4, 3), dtype=float)
    pca_scores[1, :, 0] = np.array([0.0, 1.0, 0.0, -1.0])
    pca_scores[1, :, 1] = np.array([1.0, 0.0, -1.0, 0.0])
    pca_scores[1, :, 2] = np.array([0.5, 0.5, -0.5, -0.5])

    figure, axes = unit_spike_plotting.plot_trial_behavior_and_population_pca(
        trial_df=_make_trial_df(),
        trial_index=1,
        lick_times={
            "left_entry": nap.Ts(t=np.array([1.05], dtype=float)),
            "right_entry": nap.Ts(t=np.array([1.15], dtype=float)),
        },
        pca_time_s=np.array([0.05, 0.15, 0.25, 0.35], dtype=float),
        pca_scores=pca_scores,
        trial_position=1,
        alignment_event="start_time",
        window=(0.0, 0.4),
        pc_count=3,
    )

    axes = np.asarray(axes, dtype=object).reshape(-1)
    pca_axis = axes[-1]
    assert len(pca_axis.lines) == 4
    assert pca_axis.get_ylabel() == "PC score"
    assert "PC1" in pca_axis.get_legend_handles_labels()[1]
    figure.clf()


def test_plot_pca_cumulative_explained_variance_uses_pc_numbers():
    figure, axis = unit_spike_plotting.plot_pca_cumulative_explained_variance(
        cumulative_explained_variance=np.array([0.5, 0.8, 0.9], dtype=float),
    )

    assert axis.get_xlabel() == "Number of PCs"
    assert axis.get_ylabel() == "Cumulative explained variance"
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.array([1, 2, 3]))
    figure.clf()


def test_plot_pca_cumulative_explained_variance_can_limit_many_pcs_with_sparse_ticks():
    cumulative_variance = np.linspace(0.05, 0.95, 100)

    figure, axis = unit_spike_plotting.plot_pca_cumulative_explained_variance(
        cumulative_explained_variance=cumulative_variance,
        pc_count=50,
    )

    np.testing.assert_array_equal(axis.lines[0].get_xdata(), np.arange(1, 51))
    np.testing.assert_allclose(axis.lines[0].get_ydata(), cumulative_variance[:50])
    assert 3 <= len(axis.get_xticks()) < 50
    figure.clf()


def test_webapp_population_pca_uses_all_selected_units_not_paginated_units():
    selected_unit_metadata = pd.DataFrame({"cluster_id": [10, 11, 12]})

    pca_unit_ids = psth_webapp.resolve_population_pca_unit_ids(
        selected_unit_metadata=selected_unit_metadata,
        page_unit_ids=np.array([10], dtype=int),
    )

    np.testing.assert_array_equal(pca_unit_ids, np.array([10, 11, 12], dtype=int))


def test_webapp_population_pca_cache_fits_maximum_requested_component_count(monkeypatch):
    captured_component_counts = []

    def fake_build_trial_unit_rate_tensor(**kwargs):
        return np.zeros((2, 4, 60), dtype=float), np.arange(4, dtype=float)

    def fake_fit_population_pca(*, rate_tensor_hz, n_components, normalization):
        captured_component_counts.append(int(n_components))
        return population_pca.PopulationPCAResult(
            scores=np.zeros((2, 4, int(n_components)), dtype=float),
            explained_variance_ratio=np.zeros(int(n_components), dtype=float),
            cumulative_explained_variance=np.zeros(int(n_components), dtype=float),
            unit_mean_hz=np.zeros(rate_tensor_hz.shape[2], dtype=float),
            unit_scale_hz=np.ones(rate_tensor_hz.shape[2], dtype=float),
            normalization=normalization,
        )

    monkeypatch.setattr(population_pca, "build_trial_unit_rate_tensor", fake_build_trial_unit_rate_tensor)
    monkeypatch.setattr(population_pca, "fit_population_pca", fake_fit_population_pca)

    _, pca_result = psth_webapp.compute_population_pca_cached.__wrapped__(
        session_key="test-session:PFC",
        aligned_spike_path="/tmp/probeA_sync.npz",
        unit_ids=tuple(range(60)),
        trial_indices=(0, 1),
        alignment_event="choice_time",
        window_start_s=-2.0,
        window_end_s=2.0,
        bin_size_s=0.1,
        normalization=population_pca.PCA_NORMALIZATION_ZSCORE,
        trajectory_component_count=5,
        variance_component_count=50,
        _spike_group=object(),
        _trial_df=pd.DataFrame({"choice_time": [1.0, 2.0]}),
    )

    assert captured_component_counts == [50]
    assert pca_result.scores.shape[2] == 50
