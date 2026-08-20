from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis import lfp_phase_clustering, spike_lfp_phase_locking, unit_spike_plotting


def test_compute_frequency_phase_metrics_matches_explicit_pairwise_ppc():
    """Efficient PPC should equal the explicit ordered-pair cosine definition."""

    phases = np.array([0.1, 0.8, 2.2, -1.0], dtype=float)
    vectors = np.exp(1j * phases)[np.newaxis, :].astype(np.complex64)
    explicit_ppc = np.mean(
        [np.cos(phases[first] - phases[second]) for first in range(4) for second in range(4) if first != second]
    )

    result = spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=vectors,
        valid_mask=np.ones(vectors.shape, dtype=bool),
        frequencies_hz=np.array([8.0]),
        spike_times_s=np.arange(4, dtype=float),
        unit_id=12,
        lfp_site_label="HPC channel 4",
    )

    np.testing.assert_allclose(result.ppc, [explicit_ppc], atol=1e-7)
    np.testing.assert_allclose(result.resultant_length, [np.abs(np.mean(vectors[0]))], atol=1e-7)
    np.testing.assert_allclose(result.preferred_phase_rad, [np.angle(np.mean(vectors[0]))], atol=1e-7)
    np.testing.assert_array_equal(result.n_spikes, [4])


def test_compute_frequency_phase_metrics_preserves_negative_ppc_and_handles_low_counts():
    """PPC must remain unclipped and be undefined with fewer than two valid spikes."""

    vectors = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.complex64)
    valid = np.array([[True, True], [True, False]])

    result = spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=vectors,
        valid_mask=valid,
        frequencies_hz=np.array([4.0, 8.0]),
        spike_times_s=np.array([1.0, 2.0]),
        unit_id=3,
        lfp_site_label="PFC channel 9",
    )

    assert result.ppc[0] == -1.0
    assert np.isnan(result.ppc[1])
    assert result.resultant_length[0] == 0.0
    assert np.isnan(result.preferred_phase_rad[0])
    np.testing.assert_array_equal(result.n_spikes, [2, 1])


def test_fixed_and_uniform_synthetic_phases_have_expected_metrics():
    """Locked phases should be strong while many uniform phases should approach zero PPC."""

    locked = np.exp(1j * np.full(200, np.pi / 2.0))
    uniform = np.exp(1j * np.linspace(-np.pi, np.pi, 200, endpoint=False))
    vectors = np.stack([locked, uniform]).astype(np.complex64)

    result = spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=vectors,
        valid_mask=np.ones(vectors.shape, dtype=bool),
        frequencies_hz=np.array([8.0, 12.0]),
        spike_times_s=np.arange(200, dtype=float),
        unit_id=1,
        lfp_site_label="site",
    )

    assert result.ppc[0] > 0.999
    assert result.resultant_length[0] > 0.999
    assert np.isclose(result.preferred_phase_rad[0], np.pi / 2.0, atol=1e-6)
    assert abs(result.ppc[1]) < 0.01
    assert result.resultant_length[1] < 0.01


def test_sample_wavelet_phase_at_spikes_interpolates_complex_vectors_and_amplitude():
    """Adjacent real/imaginary interpolation should recover a phase crossing without angle wrapping."""

    time_s = np.array([0.0, 0.25, 0.5], dtype=float)
    phase = np.array([0.0, np.pi / 2.0, np.pi], dtype=float)
    coefficients = np.stack(
        [
            np.array([2.0, 4.0, 2.0]) * np.exp(1j * phase),
            np.array([0.5, 0.5, 0.5]) * np.exp(1j * phase),
        ]
    ).astype(np.complex64)
    wavelets = lfp_phase_clustering.WaveletCoefficientResult(
        time_s=time_s,
        frequencies_hz=np.array([4.0, 8.0]),
        coefficients=coefficients,
        sample_rate_hz=4.0,
        source_time_s=time_s,
        source_lfp_values=np.zeros(time_s.size),
        source_sample_rate_hz=4.0,
    )

    sampled = spike_lfp_phase_locking.sample_wavelet_phase_at_spikes(
        wavelet_result=wavelets,
        spike_times_s=np.array([0.125, 0.375]),
        absolute_amplitude_threshold=1.0,
    )

    np.testing.assert_allclose(np.angle(sampled.phase_vectors[0]), [np.pi / 4.0, 3.0 * np.pi / 4.0], atol=1e-6)
    np.testing.assert_allclose(sampled.amplitude[0], [3.0, 3.0])
    np.testing.assert_array_equal(sampled.valid[0], [True, True])
    np.testing.assert_array_equal(sampled.valid[1], [False, False])
    np.testing.assert_allclose(np.abs(sampled.phase_vectors[sampled.valid]), 1.0, atol=1e-6)


def test_sample_wavelet_phase_requires_two_adjacent_numerically_valid_samples():
    """Interpolation must not bridge a zero-magnitude or otherwise invalid wavelet sample."""

    wavelets = lfp_phase_clustering.WaveletCoefficientResult(
        time_s=np.array([0.0, 1.0, 2.0]),
        frequencies_hz=np.array([5.0]),
        coefficients=np.array([[1.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j]], dtype=np.complex64),
        sample_rate_hz=1.0,
        source_time_s=np.array([0.0, 1.0, 2.0]),
        source_lfp_values=np.zeros(3),
        source_sample_rate_hz=1.0,
    )

    sampled = spike_lfp_phase_locking.sample_wavelet_phase_at_spikes(
        wavelet_result=wavelets,
        spike_times_s=np.array([0.5, 1.5]),
    )

    assert not sampled.valid.any()
    np.testing.assert_array_equal(sampled.phase_vectors, np.zeros((1, 2), dtype=np.complex64))


def test_merge_trial_windows_forms_a_union_without_duplicate_time():
    """Overlapping trial windows should become disjoint half-open absolute intervals."""

    intervals = spike_lfp_phase_locking.merge_trial_windows(
        event_times_s=np.array([1.0, 1.5, 4.0]),
        window=(-0.5, 0.5),
    )

    np.testing.assert_allclose(intervals, np.array([[0.5, 2.0], [3.5, 4.5]]))


def test_trial_aligned_pipeline_uses_padded_blocks_and_counts_each_spike_once(monkeypatch):
    """The block pipeline should sample only the union of selected trial windows."""

    load_calls: list[tuple[float, float]] = []

    def block_loader(start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray, float]:
        load_calls.append((start_s, end_s))
        time_s = np.arange(start_s, end_s, 0.01)
        return time_s, np.sin(2.0 * np.pi * 8.0 * time_s), 100.0

    def fake_compute_wavelet_coefficients(**kwargs):
        time_s = np.asarray(kwargs["time_s"], dtype=float)
        frequencies_hz = np.asarray(kwargs["frequencies_hz"], dtype=float)
        phase_offsets = np.arange(frequencies_hz.size, dtype=float)[:, np.newaxis] * 0.5
        coefficients = np.exp(1j * np.broadcast_to(phase_offsets, (frequencies_hz.size, time_s.size)))
        return lfp_phase_clustering.WaveletCoefficientResult(
            time_s=time_s,
            frequencies_hz=frequencies_hz,
            coefficients=coefficients.astype(np.complex64),
            sample_rate_hz=100.0,
            source_time_s=time_s,
            source_lfp_values=np.asarray(kwargs["lfp_values"], dtype=float),
            source_sample_rate_hz=100.0,
        )

    monkeypatch.setattr(
        spike_lfp_phase_locking.lfp_phase_clustering,
        "compute_wavelet_coefficients",
        fake_compute_wavelet_coefficients,
    )
    result = spike_lfp_phase_locking.compute_trial_aligned_spike_phase_locking(
        unit_spike_times_s=np.array([9.6, 10.2, 10.7, 15.0, 29.6]),
        event_times_s=np.array([10.0, 10.4, 30.0]),
        trial_indices=np.array([2, 3, 8]),
        window=(-0.5, 0.5),
        frequencies_hz=np.array([4.0, 8.0]),
        wavelet_padding_s=2.0,
        maximum_core_duration_s=5.0,
        block_loader=block_loader,
        unit_id=17,
        lfp_site_label="HPC channel 2",
    )

    assert load_calls == [(7.5, 12.9), (27.5, 32.5)]
    np.testing.assert_allclose(result.spike_times_s, [9.6, 10.2, 10.7, 29.6])
    np.testing.assert_array_equal(result.n_spikes, [4, 4])
    np.testing.assert_allclose(result.ppc, 1.0)
    np.testing.assert_array_equal(result.selected_trial_indices, [2, 3, 8])
    np.testing.assert_allclose(result.merged_intervals_s, [[9.5, 10.9], [29.5, 30.5]])


def _make_plot_result() -> spike_lfp_phase_locking.SpikePhaseLockingResult:
    """Build a compact deterministic result for plot and save tests."""

    phases = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    vectors = np.stack([np.exp(1j * phases), np.exp(1j * np.full(12, np.pi / 2.0))]).astype(np.complex64)
    return spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=vectors,
        valid_mask=np.ones(vectors.shape, dtype=bool),
        frequencies_hz=np.array([4.0, 8.0]),
        spike_times_s=np.arange(12, dtype=float),
        unit_id=21,
        lfp_site_label="PFC channel 5",
        amplitude_at_spikes=np.ones(vectors.shape, dtype=np.float32),
        selected_trial_indices=np.array([1, 4]),
        merged_intervals_s=np.array([[0.0, 2.0], [4.0, 6.0]]),
    )


def test_plot_spike_lfp_phase_locking_has_metric_and_polar_axes():
    """The primary figure should show PPC, resultant length, and a nearest-frequency polar plot."""

    result = _make_plot_result()
    figure, axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=result,
        polar_frequency_hz=7.5,
    )

    assert {"ppc", "resultant_length", "polar"} == axes.keys()
    assert axes["ppc"].get_xscale() == "log"
    assert axes["resultant_length"].get_ylim() == (0.0, 1.0)
    assert axes["polar"].name == "polar"
    assert "8 Hz" in axes["polar"].get_title()
    assert any(np.allclose(line.get_ydata(), [0.0, 0.0]) for line in axes["ppc"].lines)
    plt.close(figure)


def test_save_spike_lfp_phase_locking_result_preserves_vectors_and_metadata(tmp_path: Path):
    """Saved output should retain auditable vectors, masks, intervals, counts, and metadata."""

    result = _make_plot_result()
    output_path = tmp_path / "spike_lfp_phase_locking.npz"

    spike_lfp_phase_locking.save_spike_lfp_phase_locking_result(
        output_path=output_path,
        result=result,
        metadata={"window_s": [-0.5, 0.5], "normalization": "l1"},
    )

    saved = np.load(output_path, allow_pickle=True)
    np.testing.assert_array_equal(saved["ppc"], result.ppc)
    np.testing.assert_array_equal(saved["spike_phase_vectors"], result.spike_phase_vectors)
    np.testing.assert_array_equal(saved["spike_phase_valid"], result.spike_phase_valid)
    np.testing.assert_array_equal(saved["selected_trial_indices"], [1, 4])
    np.testing.assert_allclose(saved["merged_intervals_s"], [[0.0, 2.0], [4.0, 6.0]])
    assert saved["meta"].item()["normalization"] == "l1"

