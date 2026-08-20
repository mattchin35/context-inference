from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis import lfp_phase_clustering, unit_spike_plotting


def _make_trial_df() -> pd.DataFrame:
    """Build trials that exercise the established unrewarded switch convention."""

    return pd.DataFrame(
        {
            "start_time": np.arange(6, dtype=float) * 10.0,
            "choice_time": np.arange(6, dtype=float) * 10.0 + 1.0,
            "give_reward": [0, 0, 0, 0, 1, 0],
            "correct": [1, 1, 0, 1, 1, 1],
            "reward": [1, 0, 0, 0, 0, 0],
            "action": [0, 0, 1, 1, 0, 0],
        }
    )


def test_normalize_wavelet_phase_returns_unit_vectors_and_masks_tiny_values():
    """Phase normalization should reject unreliable magnitudes without infinities."""
    coefficients = np.array(
        [[1.0 + 1.0j, -2.0j], [0.0 + 0.0j, 1e-15 + 0.0j]],
        dtype=np.complex128,
    )

    unit_phase, valid = lfp_phase_clustering.normalize_wavelet_phase(
        coefficients,
        minimum_relative_magnitude=1e-12,
    )

    assert unit_phase.shape == coefficients.shape
    assert valid.shape == coefficients.shape
    np.testing.assert_allclose(np.abs(unit_phase[valid]), 1.0)
    assert not valid[1, 0]
    assert not valid[1, 1]
    assert unit_phase[1, 0] == 0.0j
    assert unit_phase[1, 1] == 0.0j


def test_make_phase_trial_tensor_uses_documented_axis_order_and_common_grid():
    """Continuous coefficients should be sampled only after transform onto one trial grid."""
    coefficient_times_s = np.arange(0.0, 8.0, 0.002)
    coefficients = np.empty((2, 2, coefficient_times_s.size), dtype=np.complex64)
    for site_index in range(2):
        for frequency_index, frequency_hz in enumerate((5.0, 10.0)):
            phase_offset = site_index * np.pi / 4.0
            coefficients[site_index, frequency_index] = np.exp(
                1j * (2.0 * np.pi * frequency_hz * coefficient_times_s + phase_offset)
            )

    tensor = lfp_phase_clustering.make_phase_trial_tensor(
        coefficient_times_s=coefficient_times_s,
        unit_phase=coefficients,
        event_times_s=np.array([2.0, 5.0]),
        trial_indices=np.array([4, 9]),
        window=(-0.1, 0.2),
        output_sample_rate_hz=500.0,
        edge_buffer_s=0.01,
    )

    assert lfp_phase_clustering.PHASE_TENSOR_AXIS_ORDER == ("site", "frequency", "trial", "time")
    assert tensor.phase.shape == (2, 2, 2, 150)
    assert tensor.valid.shape == tensor.phase.shape
    np.testing.assert_array_equal(tensor.trial_indices, np.array([4, 9]))
    assert tensor.relative_time_s[0] == -0.1
    assert np.isclose(tensor.relative_time_s[-1], 0.198)
    assert np.isclose(tensor.relative_time_s[50], 0.0)
    np.testing.assert_allclose(np.abs(tensor.phase[tensor.valid]), 1.0, atol=1e-6)


def test_make_phase_trial_tensor_excludes_boundary_trials_and_reports_them():
    """Trials lacking the full requested interval plus edge buffer should be excluded."""
    coefficient_times_s = np.arange(0.0, 2.0, 0.01)
    unit_phase = np.ones((1, 1, coefficient_times_s.size), dtype=np.complex64)

    tensor = lfp_phase_clustering.make_phase_trial_tensor(
        coefficient_times_s=coefficient_times_s,
        unit_phase=unit_phase,
        event_times_s=np.array([0.05, 1.0, 1.95]),
        trial_indices=np.array([0, 1, 2]),
        window=(-0.1, 0.1),
        output_sample_rate_hz=100.0,
        edge_buffer_s=0.02,
    )

    np.testing.assert_array_equal(tensor.trial_indices, np.array([1]))
    np.testing.assert_array_equal(tensor.excluded_trial_indices, np.array([0, 2]))
    assert tensor.phase.shape == (1, 1, 1, 20)


def test_plan_continuous_processing_blocks_groups_trials_without_condition_masks():
    """Nearby trials should share transforms while distant trials use bounded blocks."""
    blocks = lfp_phase_clustering.plan_continuous_processing_blocks(
        event_times_s=np.array([10.0, 14.0, 18.0, 100.0]),
        trial_indices=np.array([1, 2, 3, 20]),
        window=(-1.0, 2.0),
        wavelet_padding_s=4.0,
        maximum_core_duration_s=20.0,
    )

    assert len(blocks) == 2
    np.testing.assert_array_equal(blocks[0].trial_indices, np.array([1, 2, 3]))
    np.testing.assert_array_equal(blocks[1].trial_indices, np.array([20]))
    assert blocks[0].load_start_s == 5.0
    assert blocks[0].load_end_s == 24.0


def test_compute_itpc_distinguishes_locked_and_random_trial_phase():
    """Trial-locked phase should approach one while independent phases approach zero."""
    random_generator = np.random.default_rng(14)
    locked = np.ones((1, 3, 200, 6), dtype=np.complex64)
    random_phase = np.exp(
        1j * random_generator.uniform(-np.pi, np.pi, size=(1, 3, 200, 6))
    ).astype(np.complex64)

    locked_result = lfp_phase_clustering.compute_itpc(locked)
    random_result = lfp_phase_clustering.compute_itpc(random_phase)

    np.testing.assert_allclose(locked_result.values, 1.0)
    assert float(np.max(random_result.values)) < 0.2
    assert np.all((random_result.values >= 0.0) & (random_result.values <= 1.0))
    assert locked_result.n_trials == 200
    np.testing.assert_array_equal(locked_result.effective_trial_count, np.full((1, 3, 6), 200))


def test_combine_phase_trial_tensors_uses_cross_site_trial_intersection():
    """Cross-probe tensors should retain only identically indexed trials in source order."""
    relative_time_s = np.array([-0.1, 0.0], dtype=float)
    first = lfp_phase_clustering.PhaseTrialTensor(
        phase=np.ones((1, 2, 3, 2), dtype=np.complex64),
        valid=np.ones((1, 2, 3, 2), dtype=bool),
        relative_time_s=relative_time_s,
        trial_indices=np.array([1, 2, 4]),
        excluded_trial_indices=np.array([0]),
    )
    second = lfp_phase_clustering.PhaseTrialTensor(
        phase=np.full((1, 2, 3, 2), 1j, dtype=np.complex64),
        valid=np.ones((1, 2, 3, 2), dtype=bool),
        relative_time_s=relative_time_s,
        trial_indices=np.array([2, 3, 4]),
        excluded_trial_indices=np.array([5]),
    )

    combined = lfp_phase_clustering.combine_phase_trial_tensors([first, second])

    assert combined.phase.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(combined.trial_indices, np.array([2, 4]))
    np.testing.assert_array_equal(combined.excluded_trial_indices, np.array([0, 1, 3, 5]))
    np.testing.assert_array_equal(combined.phase[0], np.ones((2, 2, 2), dtype=np.complex64))
    np.testing.assert_array_equal(combined.phase[1], np.full((2, 2, 2), 1j, dtype=np.complex64))


def test_select_tensor_trial_mask_maps_full_trial_table_mask_by_trial_index():
    """Condition masks should map by source trial id after site exclusions."""
    selected = lfp_phase_clustering.select_tensor_trial_mask(
        tensor_trial_indices=np.array([1, 4, 7]),
        full_trial_mask=np.array([False, True, False, False, True, False, False, False]),
    )

    np.testing.assert_array_equal(selected, np.array([True, True, False]))


def test_compute_itpc_applies_trial_mask_without_mutating_phase_tensor():
    """Conditions should subset the trial axis after phase preprocessing."""
    phase = np.ones((2, 1, 4, 3), dtype=np.complex64)
    original = phase.copy()

    result = lfp_phase_clustering.compute_itpc(
        phase,
        trial_mask=np.array([True, False, True, False]),
    )

    assert result.values.shape == (2, 1, 3)
    assert result.n_trials == 2
    np.testing.assert_array_equal(result.effective_trial_count, np.full((2, 1, 3), 2))
    np.testing.assert_array_equal(phase, original)


def test_compute_ispc_detects_constant_nonzero_offset_and_random_relative_phase():
    """Complex phase differences should preserve circular wrapping and reject random offsets."""
    random_generator = np.random.default_rng(22)
    base_phase = np.exp(
        1j * random_generator.uniform(-np.pi, np.pi, size=(4, 250, 5))
    ).astype(np.complex64)
    phase = np.empty((2, 4, 250, 5), dtype=np.complex64)
    phase[0] = base_phase
    phase[1] = base_phase * np.exp(-1j * 1.2)

    locked_result = lfp_phase_clustering.compute_ispc(phase, site_a=0, site_b=1)
    phase[1] = np.exp(
        1j * random_generator.uniform(-np.pi, np.pi, size=base_phase.shape)
    ).astype(np.complex64)
    random_result = lfp_phase_clustering.compute_ispc(phase, site_a=0, site_b=1)

    np.testing.assert_allclose(locked_result.values, 1.0, atol=1e-6)
    assert float(np.max(random_result.values)) < 0.2
    assert np.all((random_result.values >= 0.0) & (random_result.values <= 1.0))
    assert locked_result.values.shape == (4, 5)


def test_compute_ispc_uses_intersection_of_site_validity():
    """ISPC effective counts should include only trials valid at both sites."""
    phase = np.ones((2, 1, 4, 2), dtype=np.complex64)
    valid = np.ones_like(phase, dtype=bool)
    valid[0, 0, 0, 0] = False
    valid[1, 0, 1, 0] = False

    result = lfp_phase_clustering.compute_ispc(
        phase,
        site_a=0,
        site_b=1,
        valid_mask=valid,
    )

    np.testing.assert_array_equal(result.effective_trial_count, np.array([[2, 4]]))


def test_phase_condition_masks_reuse_older_unrewarded_switch_and_stay_convention():
    """Switch/stay should mark the current unrewarded trial based on its next choice."""
    masks = lfp_phase_clustering.make_phase_condition_masks(_make_trial_df())

    assert np.flatnonzero(masks["correct_rewarded"]).tolist() == [0]
    assert np.flatnonzero(masks["incorrect"]).tolist() == [2]
    assert np.flatnonzero(masks["omission"]).tolist() == [1, 3, 5]
    assert np.flatnonzero(masks["switch"]).tolist() == [1]
    assert np.flatnonzero(masks["stay"]).tolist() == [2]


def test_relative_phase_trial_selection_combines_condition_and_action():
    """Single-trial browsing should allow left/right selection within a base condition."""
    trial_indices = unit_spike_plotting.filter_trials_for_unit_plot(
        _make_trial_df(),
        condition="omission",
        action=1,
    )

    np.testing.assert_array_equal(trial_indices, np.array([3]))


def test_plot_phase_clustering_uses_fixed_scale_log_frequency_and_alignment_line():
    """Phase-clustering heatmaps should preserve their bounded scientific scale."""
    frequencies_hz = np.geomspace(2.0, 100.0, 8)
    relative_time_s = np.linspace(-1.0, 2.0, 31)
    values = np.tile(np.linspace(0.0, 1.0, relative_time_s.size), (frequencies_hz.size, 1))

    figure, axis = lfp_phase_clustering.plot_phase_clustering(
        values=values,
        frequencies_hz=frequencies_hz,
        relative_time_s=relative_time_s,
        metric_label="ITPC",
        site_label="HPC/V1 channel 4",
        condition_label="correct_rewarded",
        n_trials=24,
    )

    assert axis.get_xlabel() == "Time from alignment event (s)"
    assert axis.get_ylabel() == "Frequency (Hz)"
    assert axis.get_yscale() == "log"
    assert axis.collections[0].get_clim() == (0.0, 1.0)
    assert "n=24" in axis.get_title()
    assert any(np.allclose(line.get_xdata(), [0.0, 0.0]) for line in axis.lines)
    assert figure.axes[-1].get_ylabel() == "ITPC"
    plt.close(figure)


def test_save_phase_clustering_result_round_trips_named_arrays_and_metadata(tmp_path: Path):
    """Saved numeric results should retain axes, counts, trial ids, and metadata."""
    output_path = tmp_path / "phase_result.npz"
    values = np.full((3, 4), 0.5, dtype=float)
    effective_counts = np.full((3, 4), 12, dtype=int)
    metadata = {
        "analysis_version": "0.1.0",
        "axis_order": "frequency,time",
        "condition": "switch",
        "random_seed": None,
    }

    lfp_phase_clustering.save_phase_clustering_result(
        output_path=output_path,
        values=values,
        effective_trial_count=effective_counts,
        frequencies_hz=np.array([2.0, 5.0, 10.0]),
        relative_time_s=np.arange(4, dtype=float) / 10.0,
        trial_indices=np.arange(12, dtype=int),
        metadata=metadata,
    )

    saved = np.load(output_path, allow_pickle=True)
    np.testing.assert_array_equal(saved["values"], values)
    np.testing.assert_array_equal(saved["effective_trial_count"], effective_counts)
    np.testing.assert_array_equal(saved["trial_indices"], np.arange(12, dtype=int))
    assert saved["meta"].item() == metadata


def _make_coefficient_result(
    phase_offset_rad: float,
    coefficient_time_offset_s: float = 0.0,
) -> lfp_phase_clustering.WaveletCoefficientResult:
    """Build padded coefficients with a controlled phase offset and source trace."""

    frequencies_hz = np.array([5.0, 10.0], dtype=float)
    time_s = np.arange(5.0 + coefficient_time_offset_s, 15.0, 0.002)
    coefficients = np.stack(
        [
            (1.0 + frequency_index) * np.exp(
                1j * (2.0 * np.pi * frequency_hz * time_s + float(phase_offset_rad))
            )
            for frequency_index, frequency_hz in enumerate(frequencies_hz)
        ]
    ).astype(np.complex64)
    source_time_s = np.arange(5.0, 15.0, 0.0004)
    return lfp_phase_clustering.WaveletCoefficientResult(
        time_s=time_s,
        frequencies_hz=frequencies_hz,
        coefficients=coefficients,
        sample_rate_hz=500.0,
        source_time_s=source_time_s,
        source_lfp_values=np.sin(2.0 * np.pi * 10.0 * source_time_s),
        source_sample_rate_hz=2500.0,
    )


def test_single_trial_relative_phase_preserves_offset_sign_and_compact_dtypes():
    """The returned angle must represent phase A minus phase B with compact arrays."""
    site_a = _make_coefficient_result(phase_offset_rad=np.pi / 2.0)
    site_b = _make_coefficient_result(phase_offset_rad=0.0)

    result = lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=site_a,
        site_b=site_b,
        event_time_s=10.0,
        visible_window=(-1.0, 2.0),
        output_sample_rate_hz=500.0,
        trial_index=7,
        site_a_label="HPC channel 2",
        site_b_label="PFC channel 3",
    )

    assert result.relative_phase_complex.shape == (2, 1500)
    assert result.phase_angle_rad.shape == (2, 1500)
    assert result.relative_phase_complex.dtype == np.complex64
    assert result.phase_angle_rad.dtype == np.float32
    assert result.amplitude_a.dtype == np.float32
    assert result.amplitude_b.dtype == np.float32
    np.testing.assert_allclose(result.phase_angle_rad, np.pi / 2.0, atol=0.04)
    assert result.trial_index == 7
    assert result.event_time_s == 10.0
    assert result.site_a_label == "HPC channel 2"
    assert result.site_b_label == "PFC channel 3"


def test_single_trial_relative_phase_interpolates_sites_to_exact_common_grid():
    """Small differences in transformed timestamps must not alter a known phase offset."""
    site_a = _make_coefficient_result(phase_offset_rad=0.7)
    site_b = _make_coefficient_result(phase_offset_rad=0.0, coefficient_time_offset_s=0.0003)

    result = lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=site_a,
        site_b=site_b,
        event_time_s=10.0,
        visible_window=(-0.5, 0.5),
        output_sample_rate_hz=500.0,
        trial_index=4,
        site_a_label="A",
        site_b_label="B",
    )

    np.testing.assert_allclose(result.relative_time_s, -0.5 + np.arange(500) / 500.0)
    np.testing.assert_allclose(result.phase_angle_rad, 0.7, atol=0.04)
    assert result.source_time_a_s[0] >= -0.5
    assert result.source_time_a_s[-1] < 0.5
    assert result.source_time_b_s[0] >= -0.5
    assert result.source_time_b_s[-1] < 0.5


def test_single_trial_relative_phase_matches_existing_trial_tensor_pathway():
    """The lightweight path should agree with the established complex-phase interpolation."""
    site_a = _make_coefficient_result(phase_offset_rad=0.9)
    site_b = _make_coefficient_result(phase_offset_rad=-0.2)
    result = lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=site_a,
        site_b=site_b,
        event_time_s=10.0,
        visible_window=(-0.2, 0.3),
        output_sample_rate_hz=500.0,
        trial_index=3,
        site_a_label="A",
        site_b_label="B",
    )
    phase_a, _valid_a = lfp_phase_clustering.normalize_wavelet_phase(site_a.coefficients)
    phase_b, _valid_b = lfp_phase_clustering.normalize_wavelet_phase(site_b.coefficients)
    tensor = lfp_phase_clustering.make_phase_trial_tensor(
        coefficient_times_s=site_a.time_s,
        unit_phase=np.stack([phase_a, phase_b]),
        event_times_s=np.array([10.0]),
        trial_indices=np.array([3]),
        window=(-0.2, 0.3),
        output_sample_rate_hz=500.0,
    )
    expected = tensor.phase[0, :, 0] * np.conjugate(tensor.phase[1, :, 0])

    np.testing.assert_allclose(result.relative_phase_complex, expected, atol=1e-6)


def test_relative_phase_amplitude_mask_supports_off_percentile_and_absolute_modes():
    """Optional masking should invalidate a pixel when either site has insufficient amplitude."""
    result = lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=_make_coefficient_result(phase_offset_rad=0.0),
        site_b=_make_coefficient_result(phase_offset_rad=0.0),
        event_time_s=10.0,
        visible_window=(-0.1, 0.1),
        output_sample_rate_hz=500.0,
        trial_index=1,
        site_a_label="A",
        site_b_label="B",
    )
    result.amplitude_a[0, 2] = 0.1
    result.amplitude_b[1, 3] = 0.1

    off_mask = lfp_phase_clustering.make_relative_phase_display_mask(result, mode="Off")
    percentile_mask = lfp_phase_clustering.make_relative_phase_display_mask(
        result,
        mode="Per-frequency percentile",
        percentile=10.0,
    )
    absolute_mask = lfp_phase_clustering.make_relative_phase_display_mask(
        result,
        mode="Absolute magnitude",
        absolute_threshold_a=0.5,
        absolute_threshold_b=0.5,
    )

    np.testing.assert_array_equal(off_mask, result.numerical_valid)
    assert not percentile_mask[0, 2]
    assert not absolute_mask[0, 2]
    assert not absolute_mask[1, 3]
    assert absolute_mask[0, 0]


def test_plot_single_trial_relative_phase_draws_heatmap_traces_and_behavior():
    """The viewer figure should contain phase, two source traces, and behavior panels."""
    result = lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=_make_coefficient_result(phase_offset_rad=np.pi / 2.0),
        site_b=_make_coefficient_result(phase_offset_rad=0.0),
        event_time_s=10.0,
        visible_window=(-0.2, 0.3),
        output_sample_rate_hz=500.0,
        trial_index=1,
        site_a_label="HPC channel 2",
        site_b_label="PFC channel 3",
    )
    trial_df = pd.DataFrame(
        {
            "start_time": [0.0, 9.0],
            "choice_time": [1.0, 10.0],
            "led_on_time": [0.5, 9.5],
            "reward_time": [1.2, 10.2],
            "action": [0, 1],
        }
    )
    lick_times = {
        unit_spike_plotting.LEFT_LICK_EVENT: np.array([9.8, 10.1]),
        unit_spike_plotting.RIGHT_LICK_EVENT: np.array([9.9]),
    }

    figure, axes = unit_spike_plotting.plot_trial_lfp_relative_phase_and_behavior(
        trial_df=trial_df,
        trial_index=1,
        lick_times=lick_times,
        result=result,
        display_valid_mask=result.numerical_valid,
        alignment_event="choice_time",
        window=(-0.2, 0.3),
    )

    assert axes.shape == (4,)
    assert axes[0].get_yscale() == "log"
    assert axes[0].collections[0].get_clim() == (-np.pi, np.pi)
    assert axes[1].get_title().endswith("(unprocessed)")
    assert axes[2].get_title().endswith("(unprocessed)")
    assert axes[3].get_ylabel() == "Behavior"
    assert figure.axes[-1].get_ylabel() == "Phase difference A - B (rad)"
    np.testing.assert_allclose(
        figure.axes[-1].get_yticks(),
        [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi],
    )
    plt.close(figure)


def test_save_single_trial_relative_phase_result_round_trips_numeric_output(tmp_path: Path):
    """Relative-phase exports should preserve reusable arrays and display settings."""
    result = lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=_make_coefficient_result(phase_offset_rad=0.5),
        site_b=_make_coefficient_result(phase_offset_rad=0.0),
        event_time_s=10.0,
        visible_window=(-0.1, 0.1),
        output_sample_rate_hz=500.0,
        trial_index=8,
        site_a_label="A",
        site_b_label="B",
    )
    display_valid = lfp_phase_clustering.make_relative_phase_display_mask(result)
    output_path = tmp_path / "relative_phase.npz"

    lfp_phase_clustering.save_single_trial_relative_phase_result(
        output_path,
        result=result,
        display_valid_mask=display_valid,
        metadata={"mask_mode": "Off", "phase_units": "radians, A minus B"},
    )

    saved = np.load(output_path, allow_pickle=True)
    np.testing.assert_array_equal(saved["relative_phase_complex"], result.relative_phase_complex)
    np.testing.assert_array_equal(saved["phase_angle_rad"], result.phase_angle_rad)
    np.testing.assert_array_equal(saved["display_valid"], display_valid)
    np.testing.assert_array_equal(saved["source_lfp_a"], result.source_lfp_a)
    assert saved["trial_index"].item() == 8
    assert saved["meta"].item()["mask_mode"] == "Off"
