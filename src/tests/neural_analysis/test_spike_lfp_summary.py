"""RED contracts for observed spike-LFP PPC and population summaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Literal

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from src.neural_analysis import spike_lfp_phase_locking, spike_lfp_summary
from src.neural_analysis.lfp_summary_preparation import build_trial_relative_spike_trains


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


def test_shuffle_schedule_is_seeded_derangement_and_deterministic() -> None:
    """Every shuffle remaps each source trial to a distinct nonself phase trial."""
    first = spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count=5,
        shuffle_count=12,
        seed=17,
    )
    second = spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count=5,
        shuffle_count=12,
        seed=17,
    )

    assert first.shape == (12, 5)
    np.testing.assert_array_equal(first, second)
    for permutation in first:
        np.testing.assert_array_equal(np.sort(permutation), np.arange(5))
        assert not np.any(permutation == np.arange(5))


def test_shuffle_schedule_is_reused_without_mutating_trial_local_spikes() -> None:
    """Two units can share one schedule while retaining separate input train arrays."""
    schedule = spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count=2,
        shuffle_count=3,
        seed=4,
    )
    phase_time_s = np.array([0.0, 1.0])
    phase_vectors = np.ones((2, 1, 2), dtype=complex)
    unit_a_spikes = (np.array([0.0]), np.array([1.0]))
    unit_b_spikes = (np.array([1.0]), np.array([0.0]))
    original_a = tuple(values.copy() for values in unit_a_spikes)
    original_b = tuple(values.copy() for values in unit_b_spikes)

    first = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=unit_a_spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase_vectors,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
    )
    second = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=unit_b_spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase_vectors,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
    )

    np.testing.assert_array_equal(first.schedule, schedule)
    np.testing.assert_array_equal(second.schedule, schedule)
    for actual, expected in zip(unit_a_spikes, original_a, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(unit_b_spikes, original_b, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_shuffle_inference_retains_overlapping_trial_local_spikes_with_warning() -> None:
    """Overlapping windows remain trial-local and are never replaced by merged intervals."""
    prepared = build_trial_relative_spike_trains(
        probe_label="PFC",
        cluster_id=1,
        unit_spike_times_s=np.array([10.0, 11.0]),
        event_times_s=np.array([9.0, 10.0]),
        window=(0.0, 2.0),
    )
    inference = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=prepared.relative_spike_times,
        phase_time_s=np.array([0.0, 1.0]),
        trial_phase_vectors=np.ones((2, 1, 2), dtype=complex),
        frequencies_hz=np.array([8.0]),
        schedule=np.array([[1, 0]]),
        overlap_trial_indices=prepared.overlap_trial_indices,
    )

    assert inference.overlap_warning is True
    np.testing.assert_array_equal(inference.overlap_trial_indices, [0, 1])
    assert inference.trial_relative_spike_times_s[0].tolist() == [1.0]
    assert inference.trial_relative_spike_times_s[1].tolist() == [0.0, 1.0]
    assert not hasattr(inference, "merged_intervals_s")


def test_shuffle_ineligibility_requires_two_spike_contributing_trials() -> None:
    """One nonempty train cannot make a two-condition-trial shuffle eligible."""
    inference = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=(np.zeros(50), np.array([], dtype=float)),
        phase_time_s=np.array([0.0]),
        trial_phase_vectors=np.ones((2, 1, 1), dtype=complex),
        frequencies_hz=np.array([8.0]),
        schedule=np.array([[1, 0]]),
    )

    assert not inference.null_summary.null_eligible[0]
    assert np.isnan(inference.p_value[0])


def test_trial_shuffle_coupling_exceeds_null_while_independent_data_do_not() -> None:
    """Trial-specific phase coupling has smaller plus-one p than independent traces."""
    trial_count = 4
    phase_time_s = np.arange(trial_count, dtype=float)
    schedule = np.array(
        [
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
        * 10,
        dtype=int,
    )
    trial_spikes = tuple(np.full(13, float(index)) for index in range(trial_count))
    coupled = np.empty((trial_count, 1, trial_count), dtype=complex)
    for phase_trial in range(trial_count):
        for source_trial in range(trial_count):
            angle = 0.0 if phase_trial == source_trial else 2.0 * np.pi * phase_trial / trial_count
            coupled[phase_trial, 0, source_trial] = np.exp(1j * angle)
    independent = np.broadcast_to(
        np.exp(2j * np.pi * np.arange(trial_count) / trial_count),
        (trial_count, trial_count),
    )[:, np.newaxis, :].copy()

    coupled_result = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=trial_spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=coupled,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
    )
    independent_result = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=trial_spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=independent,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
    )

    assert coupled_result.observed_ppc[0] > coupled_result.null_mean[0]
    assert coupled_result.p_value[0] < independent_result.p_value[0]
    assert independent_result.p_value[0] == pytest.approx(1.0)


def test_permutation_plus_one_and_ineligibility_are_explicit() -> None:
    """Null inference distinguishes ineligible data from a nonsignificant result."""
    eligible = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.50]),
        null_ppc_chunks=(np.array([[0.60], [0.40], [0.70], [0.20]]),),
        spike_count=np.array([50]),
        eligible_trial_count=2,
    )
    few_spikes = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.50]),
        null_ppc_chunks=(np.array([[0.10], [0.20]]),),
        spike_count=np.array([49]),
        eligible_trial_count=2,
    )
    one_trial = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.50]),
        null_ppc_chunks=(np.array([[0.10], [0.20]]),),
        spike_count=np.array([50]),
        eligible_trial_count=1,
    )

    assert eligible.p_value[0] == pytest.approx((1 + 2) / (1 + 4))
    assert eligible.null_eligible[0]
    for ineligible in (few_spikes, one_trial):
        assert not ineligible.null_eligible[0]
        assert np.isnan(ineligible.p_value[0])
        assert not ineligible.significant[0]


def test_fdr_uses_scipy_bh_per_frequency_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each unit-condition-site-epoch spectrum is adjusted only across frequency."""
    calls: list[tuple[tuple[int, ...], int, str]] = []

    def recording_bh(
        p_values: np.ndarray,
        *,
        axis: int = 0,
        method: str = "bh",
    ) -> np.ndarray:
        """Record SciPy-family arguments and return SciPy BH-adjusted p values."""
        calls.append((p_values.shape, axis, method))
        return false_discovery_control(p_values, axis=axis, method=method)

    monkeypatch.setattr(spike_lfp_summary, "false_discovery_control", recording_bh)
    p_value = np.array([[[[[0.01, 0.04, 0.03]]]], [[[[0.02, 0.20, 0.40]]]]])
    null_eligible = np.array([[[[[True, True, True]]]], [[[[True, False, True]]]]])

    q_value = spike_lfp_summary.adjust_ppc_pvalues_bh(
        p_value=p_value,
        null_eligible=null_eligible,
    )

    expected_first = false_discovery_control(np.array([0.01, 0.04, 0.03]), method="bh")
    np.testing.assert_allclose(q_value[0, 0, 0, 0], expected_first)
    assert np.isnan(q_value[1, 0, 0, 0, 1])
    assert calls == [((3,), 0, "bh"), ((2,), 0, "bh")]


def test_shuffle_null_accumulator_round_trips_numeric_summaries_without_draws() -> None:
    """Cache-ready null summaries retain moments/percentiles, never complete draws."""
    summary = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.2, 0.5]),
        null_ppc_chunks=(
            np.array([[0.1, 0.4], [0.3, 0.6]]),
            np.array([[0.2, 0.8], [0.4, 0.2]]),
        ),
        spike_count=np.array([50, 50]),
        eligible_trial_count=3,
    )
    arrays = spike_lfp_summary.permutation_null_summary_to_arrays(summary)
    restored = spike_lfp_summary.permutation_null_summary_from_arrays(arrays)

    expected_names = {
        "null_exceedance_count",
        "permutation_count",
        "p_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
    }
    assert expected_names <= arrays.keys()
    assert not any("draw" in name for name in arrays)
    for name in expected_names:
        np.testing.assert_allclose(
            getattr(restored, name),
            getattr(summary, name),
            equal_nan=True,
        )


def test_shuffle_chunks_match_an_unchunked_permutation_reference() -> None:
    """Streaming null chunks give the same statistics as one unchunked draw array."""
    draws = np.array([[0.1], [0.2], [0.5], [0.7], [0.3]], dtype=float)
    unchunked = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.4]),
        null_ppc_chunks=(draws,),
        spike_count=np.array([50]),
        eligible_trial_count=3,
    )
    chunked = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.4]),
        null_ppc_chunks=(draws[:2], draws[2:]),
        spike_count=np.array([50]),
        eligible_trial_count=3,
    )

    for name in (
        "null_exceedance_count",
        "permutation_count",
        "p_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
    ):
        np.testing.assert_allclose(getattr(chunked, name), getattr(unchunked, name))


def test_shuffle_preview_and_final_metadata_have_distinct_fingerprints() -> None:
    """Preview 100 and final 1000 shuffles are explicit incompatible cache metadata."""
    preview = spike_lfp_summary.build_shuffle_run_metadata(
        shuffle_count=100,
        seed=5,
        mode="preview",
    )
    final = spike_lfp_summary.build_shuffle_run_metadata(
        shuffle_count=1000,
        seed=5,
        mode="final",
    )

    assert preview["shuffle_count"] == 100
    assert final["shuffle_count"] == 1000
    assert preview["fingerprint"] != final["fingerprint"]


def test_significant_prevalence_uses_eligible_denominator_or_nan() -> None:
    """Prevalence divides significant units by eligible units, never all selected units."""
    q_value = np.array([[[[[0.01, 0.20]]]], [[[[0.03, np.nan]]]], [[[[0.80, 0.01]]]]])
    null_eligible = np.array([[[[[True, True]]]], [[[[True, False]]]], [[[[False, False]]]]])

    prevalence = spike_lfp_summary.compute_significant_prevalence(
        q_value=q_value,
        null_eligible=null_eligible,
        alpha=0.05,
    )

    np.testing.assert_allclose(prevalence.prevalence[0, 0, 0], [1.0, 0.0])
    np.testing.assert_array_equal(prevalence.eligible_unit_count[0, 0, 0], [2, 1])
    np.testing.assert_array_equal(prevalence.total_unit_count[0, 0, 0], [3, 3])
    no_eligible = spike_lfp_summary.compute_significant_prevalence(
        q_value=np.full((2, 1, 1, 1, 1), np.nan),
        null_eligible=np.zeros((2, 1, 1, 1, 1), dtype=bool),
        alpha=0.05,
    )
    assert np.isnan(no_eligible.prevalence[0, 0, 0, 0])


def test_edge_sufficient_statistics_reproduce_explicit_pairwise_ppc() -> None:
    """Complex edge sums and counts must recover exact unweighted pairwise PPC."""
    phase_time_s = np.array([0.0, 1.0, 2.0])
    trial_phase_vectors = np.ones((2, 2, 3), dtype=complex)
    trial_phase_vectors[1, 0] = np.exp(
        1j * np.array([0.0, np.pi / 2.0, np.pi])
    )
    trial_phase_vectors[1, 1] = np.exp(
        1j * np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    )
    unit_trial_spikes = (
        (np.array([0.0, 1.0, 2.0]), np.empty(0)),
        (np.array([0.0, 2.0]), np.empty(0)),
    )

    statistics = spike_lfp_summary.compute_edge_sufficient_statistics(
        trial_relative_spike_times_s=unit_trial_spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=trial_phase_vectors,
        frequencies_hz=np.array([8.0, 40.0]),
        source_trial_position=np.array([0]),
        target_trial_position=np.array([1]),
    )

    np.testing.assert_array_equal(statistics.source_trial_position, [0])
    np.testing.assert_array_equal(statistics.target_trial_position, [1])
    assert statistics.phase_vector_sum.shape == (1, 2, 2)
    assert statistics.phase_vector_sum.dtype == np.complex128
    assert statistics.valid_spike_count.dtype == np.int64
    recovered_ppc = (
        np.abs(statistics.phase_vector_sum) ** 2
        - statistics.valid_spike_count
    ) / (
        statistics.valid_spike_count * (statistics.valid_spike_count - 1)
    )
    expected_ppc = np.empty((2, 2), dtype=float)
    for unit_index, spike_times in enumerate((phase_time_s, phase_time_s[[0, 2]])):
        for frequency_index in range(2):
            phases = np.asarray(
                np.angle(
                    trial_phase_vectors[1, frequency_index, spike_times.astype(int)]
                ),
                dtype=float,
            )
            expected_ppc[unit_index, frequency_index] = np.mean(
                [
                    np.cos(phases[first] - phases[second])
                    for first in range(phases.size)
                    for second in range(phases.size)
                    if first != second
                ]
            )
    # WP5B stores normalized samples as complex64 before complex128 reduction.
    np.testing.assert_allclose(recovered_ppc[0], expected_ppc, rtol=0.0, atol=5e-8)
    assert recovered_ppc[0, 0, 0] < 0.0


def test_sufficient_statistic_reducer_handles_zero_one_two_counts_and_is_frozen() -> None:
    """Counts below two are unavailable; two opposing samples retain PPC -1."""
    phase_vector_sum = np.zeros((2, 3, 1), dtype=np.complex128)
    valid_spike_count = np.zeros((2, 3, 1), dtype=np.int64)
    phase_vector_sum[0, 1, 0] = 1.0
    valid_spike_count[0, 1, 0] = 1
    phase_vector_sum[0, 2, 0] = 1.0
    phase_vector_sum[1, 2, 0] = -1.0
    valid_spike_count[:, 2, 0] = 1
    statistics = spike_lfp_summary.EdgeSufficientStatistics(
        source_trial_position=np.array([0, 1], dtype=np.int64),
        target_trial_position=np.array([1, 0], dtype=np.int64),
        phase_vector_sum=phase_vector_sum,
        valid_spike_count=valid_spike_count,
    )

    draws = spike_lfp_summary.reduce_scheduled_shuffle_block(
        edge_statistics=statistics,
        schedule=np.array([[1, 0]], dtype=np.int64),
        shuffle_positions=np.array([0], dtype=np.int64),
    )

    assert draws.shape == (1, 3, 1)
    assert np.isnan(draws[0, 0, 0])
    assert np.isnan(draws[0, 1, 0])
    assert draws[0, 2, 0] == pytest.approx(-1.0)
    with pytest.raises(FrozenInstanceError):
        setattr(statistics, "source_trial_position", np.array([1, 0]))


def test_edge_sufficient_statistics_preserve_wp5b_missing_phase_validity() -> None:
    """Interpolation rejects outside support and intervals touching invalid phase."""
    phase_time_s = np.array([0.0, 1.0, 2.0, 3.0])
    trial_phase_vectors = np.ones((2, 2, 4), dtype=complex)
    trial_phase_vectors[0, 0, 1] = np.nan + 1j * np.nan
    trial_phase_vectors[0, 1, 1] = 0.0j
    spike_times_s = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

    statistics = spike_lfp_summary.compute_edge_sufficient_statistics(
        trial_relative_spike_times_s=(
            (spike_times_s, np.empty(0)),
        ),
        phase_time_s=phase_time_s,
        trial_phase_vectors=trial_phase_vectors,
        frequencies_hz=np.array([8.0, 40.0]),
        source_trial_position=np.array([0]),
        target_trial_position=np.array([0]),
    )

    np.testing.assert_array_equal(statistics.valid_spike_count, [[[4, 4]]])
    np.testing.assert_allclose(
        statistics.phase_vector_sum,
        [[[4.0 + 0.0j, 4.0 + 0.0j]]],
    )


def test_scheduled_sufficient_statistics_match_wp5b_shuffle_numerics() -> None:
    """Reduced draws must reproduce every WP5B null statistic for each unit."""
    phase_time_s = np.array([0.0, 1.0])
    trial_phase_vectors = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0j]],
            [[1.0j, 1.0j], [-1.0, 1.0]],
            [[-1.0, -1.0], [1.0j, -1.0j]],
        ],
        dtype=complex,
    )
    unit_trial_spikes = (
        tuple(np.tile(phase_time_s, 10) for _ in range(3)),
        (np.tile(phase_time_s, 13), np.tile(phase_time_s, 12), np.empty(0)),
    )
    schedule = spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count=3,
        shuffle_count=9,
        seed=23,
    )
    edge_pairs = sorted(
        {
            (source_position, int(target_position))
            for row in schedule
            for source_position, target_position in enumerate(row)
        }
    )
    statistics = spike_lfp_summary.compute_edge_sufficient_statistics(
        trial_relative_spike_times_s=unit_trial_spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=trial_phase_vectors,
        frequencies_hz=np.array([8.0, 40.0]),
        source_trial_position=np.array([pair[0] for pair in edge_pairs]),
        target_trial_position=np.array([pair[1] for pair in edge_pairs]),
    )
    shuffle_positions = np.array([0, 2, 4, 7], dtype=np.int64)
    reduced_draws = spike_lfp_summary.reduce_scheduled_shuffle_block(
        edge_statistics=statistics,
        schedule=schedule,
        shuffle_positions=shuffle_positions,
    )

    for unit_index, trial_spikes in enumerate(unit_trial_spikes):
        reference = spike_lfp_summary.compute_trial_shuffle_ppc(
            trial_relative_spike_times_s=trial_spikes,
            phase_time_s=phase_time_s,
            trial_phase_vectors=trial_phase_vectors,
            frequencies_hz=np.array([8.0, 40.0]),
            schedule=schedule[shuffle_positions],
        )
        reduced_summary = spike_lfp_summary.summarize_permutation_null(
            observed_ppc=reference.observed_ppc,
            null_ppc_chunks=(reduced_draws[:, unit_index],),
            spike_count=reference.spike_count,
            eligible_trial_count=reference.null_summary.eligible_trial_count,
        )
        for field_name in (
            "null_exceedance_count",
            "permutation_count",
            "eligible_trial_count",
            "p_value",
            "null_mean",
            "null_std",
            "null_p025",
            "null_p50",
            "null_p975",
            "null_eligible",
        ):
            np.testing.assert_allclose(
                getattr(reduced_summary, field_name),
                getattr(reference.null_summary, field_name),
                equal_nan=True,
            )


def test_sufficient_statistic_null_percentiles_request_explicit_linear_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null percentiles must pin NumPy's linear method rather than inherit defaults."""
    percentile = np.percentile
    requested_methods: list[str] = []

    def recording_percentile(
        values: np.ndarray,
        quantiles: list[float],
        *,
        method: Literal["linear"] = "linear",
    ) -> np.ndarray:
        """Record the requested interpolation method and delegate unchanged values."""
        requested_methods.append(method)
        return percentile(values, quantiles, method=method)

    monkeypatch.setattr(
        "src.neural_analysis.spike_lfp_summary.np.percentile",
        recording_percentile,
    )
    spike_lfp_summary.summarize_permutation_null(
        observed_ppc=np.array([0.2]),
        null_ppc_chunks=(np.array([[0.1], [0.3], [0.5]]),),
        spike_count=np.array([50]),
        eligible_trial_count=2,
    )

    assert requested_methods == ["linear"]


# WP5C-2 scheduled-edge engine contracts.
def _wp5c2_inputs() -> tuple[np.ndarray, np.ndarray, tuple[tuple[np.ndarray, ...], ...], np.ndarray]:
    """Return deterministic scheduled-edge inputs without missing values.

    Returns
    -------
    phase_time_s : numpy.ndarray
        Float64 shape ``(time,)`` strictly ascending event-relative seconds.
    trial_phase_vectors : numpy.ndarray
        Complex128 shape ``(trial, frequency, time)`` dimensionless unit phase
        vectors. Every entry is finite and nonzero.
    unit_trial_spikes : tuple[tuple[numpy.ndarray, ...], ...]
        Nested ``(unit, trial)`` float64 arrays of event-relative seconds. Each
        spike lies within the phase-time support; empty arrays mean no spikes.
    schedule : numpy.ndarray
        Int64 shape ``(shuffle, trial)`` complete derangements without missing
        values. Frequencies supplied by each caller are Hz.
    """
    phase_time_s = np.array([0.0, 0.5, 1.0], dtype=float)
    trial_phase_vectors = np.exp(
        1j
        * np.array(
            [
                [[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]],
                [[0.5, 0.7, 0.9], [0.6, 0.8, 1.0]],
                [[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]],
            ],
            dtype=float,
        )
    )
    unit_trial_spikes = (
        (np.array([0.0, 0.5]), np.array([0.5]), np.array([1.0])),
        (np.array([0.25]), np.array([0.0, 1.0]), np.array([0.5])),
        (np.array([], dtype=float), np.array([0.5]), np.array([0.0, 0.5, 1.0])),
    )
    schedule = spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count=3,
        shuffle_count=7,
        seed=29,
    )
    return phase_time_s, trial_phase_vectors, unit_trial_spikes, schedule


def test_scheduled_edge_table_is_lexicographic_and_can_use_only_schedule_edges() -> None:
    """Scheduled edges are unique, sorted source/target pairs with no self edge."""
    _, _, _, schedule = _wp5c2_inputs()

    source, target = spike_lfp_summary._scheduled_trial_edges(
        schedule,
        complete_pair_table=False,
    )

    pairs = list(zip(source.tolist(), target.tolist(), strict=True))
    assert pairs == sorted(set(pairs))
    assert all(left != right for left, right in pairs)
    assert set(pairs) == {
        (source_position, int(target_position))
        for row in schedule
        for source_position, target_position in enumerate(row)
    }


def test_scheduled_edge_and_complete_modes_match_for_one_schedule() -> None:
    """Both edge-table choices yield identical ordered shuffle PPC draws."""
    phase_time_s, phase, spikes, schedule = _wp5c2_inputs()

    scheduled = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=2,
        trial_edge_block_size=2,
        shuffle_block_size=3,
        complete_pair_table=False,
    )
    complete = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=2,
        trial_edge_block_size=2,
        shuffle_block_size=3,
        complete_pair_table=True,
    )

    np.testing.assert_allclose(scheduled, complete, equal_nan=True)


def test_scheduled_edge_engine_unit_blocks_match_independent_units() -> None:
    """Reducing a multi-unit block agrees with independently reduced unit blocks."""
    phase_time_s, phase, spikes, schedule = _wp5c2_inputs()
    together = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=3,
        trial_edge_block_size=4,
        shuffle_block_size=4,
        complete_pair_table=False,
    )
    separately = [
        spike_lfp_summary._compute_scheduled_shuffle_draws(
            trial_relative_spike_times_s=(unit_spikes,),
            phase_time_s=phase_time_s,
            trial_phase_vectors=phase,
            frequencies_hz=np.array([8.0, 40.0]),
            schedule=schedule,
            unit_block_size=1,
            trial_edge_block_size=1,
            shuffle_block_size=1,
            complete_pair_table=False,
        )[:, 0]
        for unit_spikes in spikes
    ]

    np.testing.assert_allclose(together, np.stack(separately, axis=1), equal_nan=True)


def test_scheduled_edge_engine_trial_edge_blocks_match_unblocked_reference() -> None:
    """Bounded trial-edge reductions preserve the complete edge-table result."""
    phase_time_s, phase, spikes, schedule = _wp5c2_inputs()
    reference = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=3,
        trial_edge_block_size=6,
        shuffle_block_size=7,
        complete_pair_table=False,
    )
    blocked = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=3,
        trial_edge_block_size=1,
        shuffle_block_size=7,
        complete_pair_table=False,
    )

    np.testing.assert_allclose(blocked, reference, equal_nan=True)


def test_scheduled_edge_engine_block_sizes_do_not_change_draws() -> None:
    """Unit, edge, and shuffle block sizes are execution-only choices."""
    phase_time_s, phase, spikes, schedule = _wp5c2_inputs()
    first = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=1,
        trial_edge_block_size=1,
        shuffle_block_size=1,
        complete_pair_table=False,
    )
    second = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=2,
        trial_edge_block_size=5,
        shuffle_block_size=4,
        complete_pair_table=False,
    )

    np.testing.assert_allclose(first, second, equal_nan=True)


def test_scheduled_edge_engine_preserves_overlapping_trial_membership() -> None:
    """The same physical spike may remain in each overlapping trial-local train."""
    phase_time_s = np.array([0.0, 1.0])
    phase = np.ones((2, 1, 2), dtype=complex)
    spikes = ((np.array([1.0]), np.array([0.0, 1.0])),)
    schedule = np.array([[1, 0]], dtype=np.int64)

    draws = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
        unit_block_size=1,
        trial_edge_block_size=1,
        shuffle_block_size=1,
        complete_pair_table=False,
    )

    assert draws[0, 0, 0] == pytest.approx(1.0)


def test_scheduled_edge_engine_interpolation_matches_wp5b_exact_and_invalid_support() -> None:
    """The engine retains exact complex interpolation and adjacent-valid support rules."""
    phase_time_s = np.array([0.0, 1.0, 2.0])
    phase = np.ones((2, 1, 3), dtype=complex)
    phase[1, 0, 1] = 0.0j
    spikes = (
        (
            np.tile(np.array([0.0, 0.5, 1.0, 1.5, 2.0]), 10),
            np.tile(np.array([0.0, 2.0]), 25),
        ),
    )
    schedule = np.array([[1, 0]], dtype=np.int64)

    draws = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
        unit_block_size=1,
        trial_edge_block_size=2,
        shuffle_block_size=1,
        complete_pair_table=False,
    )
    reference = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=spikes[0],
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0]),
        schedule=schedule,
    )

    np.testing.assert_allclose(
        draws[0, 0],
        reference.null_summary.null_mean,
        equal_nan=True,
    )


def test_scheduled_edge_engine_never_calls_legacy_result_or_display_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Permutation reduction samples edges only and bypasses generic result/display work."""
    phase_time_s, phase, spikes, schedule = _wp5c2_inputs()

    def fail(*_: object, **__: object) -> None:
        """Fail if scheduled reduction re-enters a prohibited legacy path."""
        raise AssertionError("scheduled reduction called a prohibited legacy path")

    monkeypatch.setattr(spike_lfp_summary, "compute_trial_shuffle_ppc", fail)
    monkeypatch.setattr(spike_lfp_summary, "compute_observed_ppc", fail)
    monkeypatch.setattr(spike_lfp_summary, "build_representative_phase_histograms", fail)
    monkeypatch.setattr(spike_lfp_phase_locking, "sample_wavelet_phase_at_spikes", fail)
    monkeypatch.setattr(spike_lfp_phase_locking, "compute_frequency_phase_metrics", fail)
    monkeypatch.setattr(spike_lfp_phase_locking, "compute_phase_firing_rate_hz", fail)
    monkeypatch.setattr(spike_lfp_phase_locking, "compute_phase_occupancy", fail)

    draws = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=2,
        trial_edge_block_size=2,
        shuffle_block_size=3,
        complete_pair_table=False,
    )

    assert draws.shape == (schedule.shape[0], len(spikes), 2)


def test_scheduled_edge_engine_samples_each_edge_once_per_unit_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase interpolation is performed once per edge/unit block, not per shuffle block."""
    phase_time_s, phase, spikes, schedule = _wp5c2_inputs()
    expected_source, expected_target = spike_lfp_summary._scheduled_trial_edges(
        schedule,
        complete_pair_table=False,
    )
    expected_edges = list(zip(expected_source.tolist(), expected_target.tolist(), strict=True))
    sampled_edges: list[tuple[int, int]] = []
    original = spike_lfp_summary.compute_edge_sufficient_statistics

    def recording_edge_statistics(**kwargs: object) -> spike_lfp_summary.EdgeSufficientStatistics:
        """Record every edge passed to the numerical phase-sampling kernel."""
        source = np.asarray(kwargs["source_trial_position"])
        target = np.asarray(kwargs["target_trial_position"])
        sampled_edges.extend(zip(source.tolist(), target.tolist(), strict=True))
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        spike_lfp_summary,
        "compute_edge_sufficient_statistics",
        recording_edge_statistics,
    )
    draws = spike_lfp_summary._compute_scheduled_shuffle_draws(
        trial_relative_spike_times_s=spikes,
        phase_time_s=phase_time_s,
        trial_phase_vectors=phase,
        frequencies_hz=np.array([8.0, 40.0]),
        schedule=schedule,
        unit_block_size=2,
        trial_edge_block_size=2,
        shuffle_block_size=2,
        complete_pair_table=False,
    )

    assert draws.shape == (schedule.shape[0], len(spikes), 2)
    assert sorted(sampled_edges) == sorted(expected_edges * 2)
