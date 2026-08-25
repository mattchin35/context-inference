"""RED contracts for observed spike-LFP PPC and population summaries."""

from __future__ import annotations

import numpy as np
import pytest

from src.neural_analysis import spike_lfp_phase_locking, spike_lfp_summary


def test_observed_ppc_reuses_existing_pairwise_formula_and_retains_negative_values() -> None:
    """Observed PPC delegates to the established frequency-phase metric helper."""
    phases = np.array([0.0, np.pi, np.pi / 2.0], dtype=float)
    phase_vectors = np.exp(1j * phases)[np.newaxis, :]
    expected = spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=phase_vectors,
        valid_mask=np.ones((1, 3), dtype=bool),
        frequencies_hz=np.array([8.0]),
        spike_times_s=np.arange(3, dtype=float),
        unit_id=12,
        lfp_site_label="PFC",
    )

    observed = spike_lfp_summary.compute_observed_ppc(
        probe_label="PFC",
        cluster_id=12,
        spike_phase_vectors=phase_vectors,
        valid_mask=np.ones((1, 3), dtype=bool),
        frequencies_hz=np.array([8.0]),
        spike_times_s=np.arange(3, dtype=float),
    )

    explicit_ppc = np.mean(
        [
            np.cos(phases[first] - phases[second])
            for first in range(phases.size)
            for second in range(phases.size)
            if first != second
        ]
    )
    np.testing.assert_allclose(observed.ppc, expected.ppc)
    np.testing.assert_allclose(observed.ppc, [explicit_ppc])
    assert observed.ppc[0] < 0.0
    assert observed.unit_id == "PFC:12"


def test_observed_ppc_count_masks_and_circular_metrics_follow_contract() -> None:
    """Two samples compute PPC, while only fifty samples are reliable."""
    phase_vectors = np.array(
        [
            [1.0, np.nan, np.nan, np.nan] + [np.nan] * 46,
            [1.0, -1.0] + [np.nan] * 48,
            [1.0] * 49 + [np.nan],
            np.exp(1j * np.full(50, np.pi / 2.0)),
        ],
        dtype=complex,
    )
    valid_mask = np.isfinite(phase_vectors)

    observed = spike_lfp_summary.compute_observed_ppc(
        probe_label="HPC1",
        cluster_id=7,
        spike_phase_vectors=phase_vectors,
        valid_mask=valid_mask,
        frequencies_hz=np.array([2.0, 4.0, 6.0, 8.0]),
        spike_times_s=np.arange(50, dtype=float),
    )

    np.testing.assert_array_equal(observed.spike_count, [1, 2, 49, 50])
    assert np.isnan(observed.ppc[0])
    assert observed.ppc[1] == -1.0
    np.testing.assert_array_equal(observed.computable, [False, True, True, True])
    np.testing.assert_array_equal(observed.reliable, [False, False, False, True])
    assert observed.resultant_length[3] == pytest.approx(1.0)
    assert observed.preferred_phase_rad[3] == pytest.approx(np.pi / 2.0)


def test_band_mean_uses_2_hz_grid_and_excludes_gamma_line_frequencies() -> None:
    """Gamma summaries omit 58, 60, and 62 Hz without bridging their values."""
    frequencies_hz = np.arange(2.0, 82.0, 2.0)
    ppc = frequencies_hz.copy()

    summary = spike_lfp_summary.compute_band_ppc_means(
        frequencies_hz=frequencies_hz,
        ppc_by_frequency=ppc,
    )

    assert summary.band_names == ("theta", "gamma")
    assert summary.ppc_band_mean.shape == (2,)
    assert summary.ppc_band_mean[0] == np.mean([6.0, 8.0, 10.0])
    gamma_retained = frequencies_hz[(frequencies_hz >= 30.0) & (frequencies_hz <= 80.0)]
    gamma_retained = gamma_retained[~np.isin(gamma_retained, [58.0, 60.0, 62.0])]
    assert summary.ppc_band_mean[1] == np.mean(gamma_retained)


def test_population_order_and_medians_distinguish_computable_from_reliable() -> None:
    """Reference theta order is reused for all conditions; medians need reliable PPC."""
    unit_ids = ("HPC:2", "PFC:1", "PFC:9")
    # Axes are (unit, condition, site, epoch, frequency).
    ppc = np.array(
        [
            [[[[0.2, 0.2], [0.1, 0.1]]]],
            [[[[0.8, 0.8], [0.3, 0.3]]]],
            [[[[0.5, 0.5], [0.9, 0.9]]]],
        ],
        dtype=float,
    )
    ppc = np.concatenate([ppc, ppc[:, :, :, :, :] + 0.01], axis=1)
    spike_count = np.array(
        [
            [[[[50, 50], [49, 49]]]],
            [[[[49, 49], [50, 50]]]],
            [[[[50, 50], [1, 1]]]],
        ],
        dtype=int,
    )
    spike_count = np.concatenate([spike_count, spike_count], axis=1)

    summary = spike_lfp_summary.build_population_ppc_summary(
        unit_ids=unit_ids,
        condition_names=("correct_rewarded", "omission"),
        site_ids=("PFC",),
        epoch_names=("whole", "after"),
        frequencies_hz=np.array([6.0, 8.0]),
        ppc=ppc,
        spike_count=spike_count,
    )

    assert summary.unit_order_ids == ("PFC:1", "PFC:9", "HPC:2")
    np.testing.assert_array_equal(
        summary.computable[:, 0, 0, 0],
        [[True, True], [True, True], [True, True]],
    )
    np.testing.assert_array_equal(
        summary.reliable[:, 0, 0, 0],
        [[False, False], [True, True], [True, True]],
    )
    np.testing.assert_array_equal(
        summary.reliable[:, 0, 0, 1],
        [[True, True], [False, False], [False, False]],
    )
    assert summary.heatmap_ppc.shape == (3, 2, 1, 2, 2)
    np.testing.assert_allclose(summary.heatmap_ppc[0, 0, 0, 1], [0.3, 0.3])
    assert np.isnan(summary.heatmap_ppc[1, 0, 0, 1]).all()
    np.testing.assert_allclose(summary.heatmap_ppc[2, 0, 0, 1], [0.1, 0.1])
    np.testing.assert_allclose(summary.population_median_ppc[0, 0, 0], [0.35, 0.35])
    np.testing.assert_allclose(summary.population_median_ppc[0, 0, 1], [0.3, 0.3])


def test_representative_histograms_use_fixed_frequency_and_pool_all_trials() -> None:
    """Theta and gamma polar counts use 8/40 Hz and fixed bins across trials."""
    frequencies_hz = np.array([6.0, 8.0, 10.0, 38.0, 40.0, 42.0])
    phase_vectors = np.exp(
        1j
        * np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-2.0, -0.5, 0.5, 2.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-2.0, -0.5, 0.5, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    trial_indices = np.array([3, 3, 8, 8])
    phase_bin_edges_rad = np.linspace(-np.pi, np.pi, 5)

    histograms = spike_lfp_summary.build_representative_phase_histograms(
        frequencies_hz=frequencies_hz,
        spike_phase_vectors=phase_vectors,
        valid_mask=np.ones(phase_vectors.shape, dtype=bool),
        trial_indices=trial_indices,
        phase_bin_edges_rad=phase_bin_edges_rad,
    )

    assert histograms.representative_frequency_hz == (8.0, 40.0)
    assert histograms.phase_bin_edges_rad.tolist() == phase_bin_edges_rad.tolist()
    np.testing.assert_array_equal(histograms.spike_count_by_band, [[1, 1, 1, 1], [1, 1, 1, 1]])
    assert histograms.source_label == "pooled eligible trials"


def test_representative_histograms_do_not_mutate_caller_validity_mask() -> None:
    """Zero/nonfinite vectors are excluded locally without altering caller-owned masks."""
    frequencies_hz = np.array([8.0, 40.0])
    phase_vectors = np.array(
        [
            [1.0, 0.0, np.nan + 0.0j],
            [1.0j, 0.0, np.nan + 0.0j],
        ],
        dtype=complex,
    )
    valid_mask = np.ones(phase_vectors.shape, dtype=bool)
    original_valid_mask = valid_mask.copy()

    histograms = spike_lfp_summary.build_representative_phase_histograms(
        frequencies_hz=frequencies_hz,
        spike_phase_vectors=phase_vectors,
        valid_mask=valid_mask,
        trial_indices=np.array([0, 0, 1]),
        phase_bin_edges_rad=np.linspace(-np.pi, np.pi, 5),
    )

    np.testing.assert_array_equal(valid_mask, original_valid_mask)
    np.testing.assert_array_equal(histograms.spike_count_by_band.sum(axis=1), [1, 1])


def test_population_reference_order_excludes_subcomputable_ppc_values() -> None:
    """A count-one PPC cannot outrank a lower computable theta observation."""
    unit_ids = ("PFC:9", "HPC:2", "PFC:1", "HPC:1")
    # Axes are (unit, condition, site, epoch, frequency).
    ppc = np.array([0.99, 0.50, 0.30, 0.80], dtype=float)[:, None, None, None, None]
    ppc = np.repeat(ppc, 2, axis=-1)
    spike_count = np.array([1, 50, 50, 1], dtype=int)[:, None, None, None, None]
    spike_count = np.repeat(spike_count, 2, axis=-1)

    summary = spike_lfp_summary.build_population_ppc_summary(
        unit_ids=unit_ids,
        condition_names=("correct_rewarded",),
        site_ids=("PFC",),
        epoch_names=("whole",),
        frequencies_hz=np.array([6.0, 8.0]),
        ppc=ppc,
        spike_count=spike_count,
    )

    assert summary.unit_order_ids == ("HPC:2", "PFC:1", "HPC:1", "PFC:9")


def test_exemplar_selection_is_deterministic_and_labels_pooled_vs_illustrative() -> None:
    """High/low band PPC and median-spike trials use approved deterministic ties."""
    selection = spike_lfp_summary.select_ppc_exemplars(
        unit_ids=("PFC:11", "HPC:5", "PFC:2", "HPC:4"),
        band_ppc=np.array([0.20, 0.80, 0.20, 0.80]),
        trial_indices=np.array([9, 2, 7, 4]),
        trial_spike_counts=np.array(
            [
                [1, 5, 5, 9],
                [4, 4, 8, 8],
                [1, 1, 1, 1],
                [2, 2, 2, 2],
            ]
        ),
    )

    assert selection.low_unit_id == "PFC:11"
    assert selection.high_unit_id == "HPC:4"
    assert selection.illustrative_trial_index_by_unit["PFC:11"] == 2
    assert selection.illustrative_trial_index_by_unit["HPC:4"] == 2
    assert selection.pooled_metric_label == "pooled eligible trials"
    assert selection.illustrative_trial_label == "illustrative single trial"


def test_probe_qualified_ids_prevent_same_cluster_number_collisions() -> None:
    """Cluster ids remain distinct when the same sorter number occurs on two probes."""
    identifiers = spike_lfp_summary.build_probe_qualified_unit_ids(
        probe_labels=("PFC", "HPC1", "HPC1"),
        cluster_ids=(12, 12, 13),
    )

    assert identifiers == ("PFC:12", "HPC1:12", "HPC1:13")
    assert len(set(identifiers)) == 3
