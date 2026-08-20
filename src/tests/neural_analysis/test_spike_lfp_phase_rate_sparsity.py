"""Tests for descriptive sparse phase-rate estimate warnings."""

from __future__ import annotations

import numpy as np

from src.neural_analysis import spike_lfp_phase_locking


def test_phase_rate_sparsity_warns_below_five_spikes_per_bin():
    """A 24-bin estimate with fewer than 120 spikes should be marked sparse."""
    assessment = spike_lfp_phase_locking.assess_phase_rate_sparsity(
        phase_spike_counts=np.full(24, 4, dtype=int),
        phase_occupancy_s=np.ones(24, dtype=float),
    )

    assert assessment.is_sparse
    assert assessment.valid_spike_count == 96
    assert assessment.phase_bin_count == 24
    assert assessment.mean_spikes_per_bin == 4.0
    assert assessment.minimum_recommended_spikes == 120
    assert assessment.zero_occupancy_bin_count == 0
    assert assessment.low_spike_count
    assert not assessment.has_zero_occupancy


def test_phase_rate_sparsity_accepts_five_spikes_per_occupied_bin():
    """The heuristic boundary should not warn when every bin has occupancy."""
    assessment = spike_lfp_phase_locking.assess_phase_rate_sparsity(
        phase_spike_counts=np.full(24, 5, dtype=int),
        phase_occupancy_s=np.ones(24, dtype=float),
    )

    assert not assessment.is_sparse
    assert assessment.valid_spike_count == 120
    assert assessment.mean_spikes_per_bin == 5.0
    assert not assessment.low_spike_count
    assert not assessment.has_zero_occupancy


def test_phase_rate_sparsity_warns_when_any_phase_bin_has_no_occupancy():
    """Unobserved phase bins should warn even when the spike count is large."""
    occupancy_s = np.ones(24, dtype=float)
    occupancy_s[[3, 11]] = 0.0

    assessment = spike_lfp_phase_locking.assess_phase_rate_sparsity(
        phase_spike_counts=np.full(24, 10, dtype=int),
        phase_occupancy_s=occupancy_s,
    )

    assert assessment.is_sparse
    assert assessment.valid_spike_count == 240
    assert assessment.zero_occupancy_bin_count == 2
    assert not assessment.low_spike_count
    assert assessment.has_zero_occupancy


def test_phase_rate_sparsity_rejects_mismatched_or_invalid_inputs():
    """Counts and occupancy must be matching finite one-dimensional arrays."""
    with np.testing.assert_raises(ValueError):
        spike_lfp_phase_locking.assess_phase_rate_sparsity(
            phase_spike_counts=np.ones(3, dtype=int),
            phase_occupancy_s=np.ones(4, dtype=float),
        )

    with np.testing.assert_raises(ValueError):
        spike_lfp_phase_locking.assess_phase_rate_sparsity(
            phase_spike_counts=np.ones((1, 3), dtype=int),
            phase_occupancy_s=np.ones((1, 3), dtype=float),
        )

    with np.testing.assert_raises(ValueError):
        spike_lfp_phase_locking.assess_phase_rate_sparsity(
            phase_spike_counts=np.ones(3, dtype=int),
            phase_occupancy_s=np.array([1.0, np.nan, 1.0]),
        )
