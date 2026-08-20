from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import numpy as np

from src.neural_analysis import spike_lfp_phase_locking, unit_spike_plotting


def make_rate_plot_result(
    phase_firing_rate_hz: np.ndarray | None = None,
) -> spike_lfp_phase_locking.SpikePhaseLockingResult:
    """Build a deterministic result carrying occupancy-normalized phase rates."""
    frequencies_hz = np.array([4.0, 8.0])
    phase_bin_edges_rad = np.linspace(-np.pi, np.pi, 5)
    phase_spike_counts = np.array(
        [[1, 1, 1, 1], [2, 8, 1, 6]],
        dtype=int,
    )
    phase_occupancy_s = np.array(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 2.0]],
        dtype=float,
    )
    if phase_firing_rate_hz is None:
        phase_firing_rate_hz = phase_spike_counts / phase_occupancy_s

    phases = np.linspace(-np.pi, np.pi, 17, endpoint=False)
    phase_vectors = np.stack(
        [np.exp(1j * phases), np.exp(1j * np.full(phases.size, np.pi / 2.0))]
    ).astype(np.complex64)
    return spike_lfp_phase_locking.SpikePhaseLockingResult(
        frequencies_hz=frequencies_hz,
        ppc=np.array([0.1, 0.5]),
        resultant_length=np.array([0.2, 0.8]),
        preferred_phase_rad=np.array([0.0, np.pi / 2.0]),
        n_spikes=np.array([4, 17]),
        spike_phase_vectors=phase_vectors,
        spike_phase_valid=np.ones(phase_vectors.shape, dtype=bool),
        amplitude_at_spikes=np.ones(phase_vectors.shape, dtype=np.float32),
        spike_times_s=np.arange(phases.size, dtype=float),
        selected_trial_indices=np.array([1, 4]),
        merged_intervals_s=np.array([[0.0, 2.0], [4.0, 6.0]]),
        unit_id=21,
        lfp_site_label="PFC channel 5",
        phase_bin_edges_rad=phase_bin_edges_rad,
        phase_spike_counts=phase_spike_counts,
        phase_occupancy_s=phase_occupancy_s,
        phase_firing_rate_hz=np.asarray(phase_firing_rate_hz, dtype=float),
    )


def test_phase_rate_plot_keeps_the_existing_three_axes_and_uses_nearest_frequency():
    """The rate visualization should preserve the PPC, resultant, and polar panels."""
    figure, axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=make_rate_plot_result(),
        polar_frequency_hz=7.5,
        polar_bin_count=4,
    )

    assert set(axes) == {"ppc", "resultant_length", "polar"}
    assert axes["ppc"].get_xscale() == "log"
    assert axes["resultant_length"].get_ylim() == (0.0, 1.0)
    assert "8 Hz" in axes["polar"].get_title()
    plt.close(figure)


def test_phase_polar_bars_use_occupancy_normalized_hz_values():
    """Polar bar heights must be rates, not the raw per-bin spike counts."""
    result = make_rate_plot_result()
    figure, axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=result,
        polar_frequency_hz=8.0,
        polar_bin_count=4,
    )

    bar_heights = np.array([patch.get_height() for patch in axes["polar"].patches])
    np.testing.assert_allclose(bar_heights, [2.0, 4.0, 1.0, 3.0])
    assert not np.array_equal(bar_heights, result.phase_spike_counts[1])
    assert axes["polar"].get_ylabel() == "Firing rate (Hz)"
    assert "Firing rate" in axes["polar"].get_title()
    plt.close(figure)


def test_preferred_phase_marker_radius_is_independent_of_rate_scale():
    """The preferred-phase marker should span the axis rather than encode a rate."""
    low_rate_result = make_rate_plot_result(
        phase_firing_rate_hz=np.array([[0.5, 1.0, 0.5, 1.0], [1.0, 2.0, 1.0, 2.0]])
    )
    high_rate_result = make_rate_plot_result(
        phase_firing_rate_hz=np.array([[50.0, 100.0, 50.0, 100.0], [100.0, 200.0, 100.0, 200.0]])
    )
    low_figure, low_axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=low_rate_result,
        polar_frequency_hz=8.0,
        polar_bin_count=4,
    )
    high_figure, high_axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=high_rate_result,
        polar_frequency_hz=8.0,
        polar_bin_count=4,
    )

    assert not [child for child in low_axes["polar"].get_children() if isinstance(child, Annotation)]
    assert not [child for child in high_axes["polar"].get_children() if isinstance(child, Annotation)]
    for axis in (low_axes["polar"], high_axes["polar"]):
        preferred_phase_lines = [
            line
            for line in axis.lines
            if np.allclose(line.get_xdata(), [np.pi / 2.0, np.pi / 2.0])
        ]
        assert len(preferred_phase_lines) == 1
        np.testing.assert_allclose(preferred_phase_lines[0].get_ydata(), [0.0, 1.0])
    plt.close(low_figure)
    plt.close(high_figure)
