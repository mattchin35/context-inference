"""RED contracts for shared LFP-summary preparation and existing-loader routing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    ProgressEvent,
    TrialFilterConfig,
)
from src.neural_analysis.spike_behavior_pynapple import make_trial_type_masks
from src.neural_analysis import lfp_loading
from src.neural_analysis.lfp_summary_preparation import (
    build_common_event_grid,
    build_prepared_trials,
    build_trial_relative_spike_trains,
    interpolate_complex_coefficients_to_common_grid,
    load_site_trial_traces,
    prepare_site_trial_traces,
    validate_progress_events,
)


CONDITIONS = (
    "correct_rewarded", "omission", "incorrect", "switch", "stay",
    "omission_switch", "omission_stay", "incorrect_switch", "incorrect_stay",
)


def _observed_open_ephys_metadata(sample_rate_hz: float, channel_count: int) -> dict[str, object]:
    """Return normalized production-sidecar metadata for synthetic physical-uV routes."""
    channel_ids = [f"CH{index}" for index in range(channel_count)]
    return {
        "output_binary": "lfp.dat",
        "sampling_frequency_hz": sample_rate_hz,
        "num_channels": channel_count,
        "num_segments": 1,
        "num_samples_by_segment": [1],
        "dtype": "float32",
        "binary_layout": "time_major_channel_interleaved",
        "channel_ids_in_binary_order": channel_ids,
        "lfp_binary_scaling": {
            "data_units": "unscaled_binary_values",
            "has_scaleable_traces": True,
            "channel_ids": channel_ids,
            "gain_to_uV_by_channel": [0.195] * channel_count,
            "offset_to_uV_by_channel": [0.0] * channel_count,
            "physical_unit_by_channel": ["uV"] * channel_count,
            "export_scale_factor": 1.0,
            "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
        },
    }


def _trials() -> pd.DataFrame:
    """Return a trial table with overlapping condition membership and behavior edge cases."""
    return pd.DataFrame({
        "choice_time": [1.0, np.nan, 3.0, 4.0, 5.0],
        "start_time": [0.0, 2.0, 2.5, 3.5, 4.5],
        "experimenter_reward_given": [0, 0, 0, 0, 0],
        "correct": [1, 1, 0, 1, 0],
        "reward": [0, 0, 0, 1, 0],
        "action": [0, 1, 0, np.nan, 2],
        "state_int": [0, 1, 2, np.nan, 1],
    })


def test_prepared_trials_use_authoritative_ordered_overlapping_masks() -> None:
    """Nine masks retain the authoritative order and intentional overlap semantics."""
    trial_df = _trials()
    prepared = build_prepared_trials(trial_df, TrialFilterConfig(), "choice_time")
    authoritative_masks = make_trial_type_masks(trial_df)
    expected_membership = np.column_stack(
        [authoritative_masks[condition].to_numpy(dtype=bool) for condition in CONDITIONS]
    )

    assert prepared.condition_names == CONDITIONS
    assert prepared.condition_membership.shape == (5, 9)
    assert np.array_equal(prepared.condition_membership, expected_membership)
    omission = prepared.condition_membership[:, CONDITIONS.index("omission")]
    omission_switch = prepared.condition_membership[:, CONDITIONS.index("omission_switch")]
    assert omission[0] and omission_switch[0]
    assert prepared.condition_membership.dtype == bool


@pytest.mark.parametrize(
    ("choice", "context", "expected"),
    [
        ("right", "all", [True, False, True, False, False]),
        ("left", "all", [False, True, False, False, False]),
        ("all", "right", [True, False, False, False, False]),
        ("all", "left", [False, True, False, False, True]),
    ],
)
def test_choice_context_filters_preserve_zero_right_one_left_and_other_behavior(choice: str, context: str, expected: list[bool]) -> None:
    """Filters are independent; missing/other labels pass only an ``all`` dimension."""
    prepared = build_prepared_trials(_trials(), TrialFilterConfig(choice=choice, context=context), "choice_time")
    assert prepared.filter_membership.tolist() == expected


def test_user_and_objective_exclusions_have_distinct_masks_and_reasons() -> None:
    """User exclusions never overwrite missing-alignment objective-invalid reasons."""
    prepared = build_prepared_trials(_trials(), TrialFilterConfig(excluded_trial_indices=(0, 1)), "choice_time")
    assert prepared.user_excluded.tolist() == [True, True, False, False, False]
    assert prepared.objective_valid.tolist() == [True, False, True, True, True]
    assert prepared.objective_exclusion_reason[1] == "missing_choice_time"
    assert prepared.user_exclusion_reason[0] == "user_excluded"


def test_site_trace_validity_records_incomplete_nonfinite_constant_reasons_and_qc() -> None:
    """Site-local invalid traces preserve trial axes, uV units, and explicit reason codes."""
    site = LFPSiteConfig(
        "PFC", "PFC", "spikeglx", Path("pfc.bin"), None, "PFC", 5, "uV", 1000.0
    )
    traces = [np.array([1.0, 2.0]), np.array([np.nan, 1.0, 2.0, 3.0]), np.ones(4), np.arange(4.0)]
    prepared = prepare_site_trial_traces(
        site=site,
        trial_indices=np.array([0, 1, 2, 3]),
        alignment_times_s=np.array([1.0, 2.0, 3.0, 4.0]),
        window=(-0.002, 0.002),
        trace_loader=lambda *_: (np.array([-0.002, -0.001, 0.0, 0.001]), traces.pop(0), 1000.0),
    )
    assert prepared.valid.tolist() == [False, False, False, True]
    assert prepared.exclusion_reason.tolist() == ["incomplete_window", "nonfinite_trace", "constant_trace", ""]
    assert prepared.rms.shape == prepared.peak_to_peak.shape == (4,)
    assert np.isnan(prepared.rms[:3]).all()
    assert np.isnan(prepared.peak_to_peak[:3]).all()
    assert prepared.rms[3] == pytest.approx(np.sqrt(3.5))
    assert prepared.peak_to_peak[3] == pytest.approx(3.0)
    assert prepared.voltage_unit == "uV"


def test_site_validity_and_pair_intersection_do_not_overexclude_hpc1() -> None:
    """A missing HPC2 trial affects only pairs containing HPC2."""
    prepared = build_prepared_trials(
        _trials(),
        TrialFilterConfig(),
        "choice_time",
        site_validity={
            "PFC": [True] * 5,
            "HPC1": [True] * 5,
            "HPC2": [True, False, True, True, True],
        },
    )
    assert prepared.site_validity["PFC"].tolist() == [True]*5
    assert prepared.pair_validity[("PFC", "HPC1")].tolist() == [True]*5
    assert prepared.pair_validity[("PFC", "HPC2")].tolist() == [True, False, True, True, True]


def test_common_grid_is_exact_500_hz_half_open_and_complex_interpolation_wrap_safe() -> None:
    """Complex real/imag interpolation preserves a wrapped phase instead of interpolating angles."""
    grid = build_common_event_grid(-0.002, 0.002, 500.0)
    assert np.array_equal(grid, np.array([-0.002, 0.0]))
    source_time = np.array([-0.002, 0.002])
    coefficients = np.vstack((np.exp(1j * np.array([3.0, -3.0])), np.exp(1j * np.array([2.0, 2.2]))))
    interpolated = interpolate_complex_coefficients_to_common_grid(source_time, coefficients, grid)
    assert interpolated.shape == (2, 2)
    assert np.allclose(np.abs(interpolated), 1.0)
    assert abs(np.angle(interpolated[0, 1])) > 2.5
    assert np.angle(interpolated[1, 1]) > 2.0


def test_trial_relative_spike_trains_are_local_probe_qualified_and_report_overlaps() -> None:
    """Half-open overlapping trial windows retain one physical spike in each applicable trial."""
    result = build_trial_relative_spike_trains(
        probe_label="PFC",
        cluster_id=10,
        unit_spike_times_s=np.array([10.0, 11.0]),
        event_times_s=np.array([9.0, 10.0]),
        window=(0.0, 2.0),
    )
    assert result.unit_id == "PFC:10"
    assert result.relative_spike_times[0].tolist() == [1.0]
    assert result.relative_spike_times[1].tolist() == [0.0, 1.0]
    assert result.overlap_trial_indices.tolist() == [0, 1]


def test_existing_acquisition_routes_use_injected_loaders_and_preserve_units_rates() -> None:
    """SpikeGLX/Open Ephys dispatch is injectable and returns source units/sample rates unchanged."""
    site_glx = LFPSiteConfig(
        "PFC", "PFC", "spikeglx", Path("a.bin"), None, "PFC", 5, "uV", 2500.0
    )
    site_oe = LFPSiteConfig(
        "HPC", "HPC", "open_ephys", Path("b.dat"), Path("b.npz"), "HPC", 2, "mV", 1000.0
    )
    calls: list[str] = []

    def fake_glx(**_: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Accept routed kwargs; return one-second-axis value sample and a 2500-Hz rate."""
        calls.append("glx")
        return np.array([0.0]), np.array([1.0]), 2500.0

    def fake_oe(**_: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Accept routed kwargs; return one-second-axis value sample and a 1000-Hz rate."""
        calls.append("oe")
        return np.array([0.0]), np.array([2.0]), 1000.0
    loaded = load_site_trial_traces(
        sites=[site_glx, site_oe],
        trial_indices=np.array([7]),
        alignment_times_s=np.array([1.0]),
        window=(-1.0, 1.0),
        spikeglx_loader=fake_glx,
        open_ephys_loader=fake_oe,
    )
    assert calls == ["glx", "oe"]
    assert loaded["PFC"].sample_rate_hz == 2500.0 and loaded["HPC"].voltage_unit == "mV"
    assert loaded["PFC"].source_trace.shape == (1, 5000)
    assert loaded["HPC"].source_trace.shape == (1, 2000)
    assert not loaded["PFC"].valid[0]
    assert not loaded["HPC"].valid[0]


def test_progress_events_are_framework_independent_and_monotonic() -> None:
    """Progress validation accepts plain records and rejects decreasing completed counts."""
    events = [ProgressEvent("power", "load", 0, 2, "starting"), ProgressEvent("power", "load", 2, 2, "done")]
    validate_progress_events(events)
    with pytest.raises(ValueError):
        validate_progress_events([events[1], events[0]])


def test_default_spikeglx_route_decodes_sync_once_and_uses_real_loader_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SpikeGLX routing reuses one decoded sync table and the decoded LFP rate per site."""
    site = LFPSiteConfig("PFC", "PFC", "spikeglx", Path("pfc.bin"), None, "PFC", 5, "uV", 2500.0)
    decoded_sync = pd.DataFrame({"sample_ix": [0], "utc_unix": [0.0]})
    decode_calls: list[Path] = []
    loader_kwargs: list[dict[str, object]] = []

    def fake_decode(lfp_path: Path) -> tuple[pd.DataFrame, float]:
        """Return one IRIG anchor and its 1250-Hz LFP rate for one path."""
        decode_calls.append(lfp_path)
        return decoded_sync, 1250.0

    def fake_loader(**kwargs: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Record real SpikeGLX loader kwargs and return `(time, uV, Hz)` arrays."""
        loader_kwargs.append(kwargs)
        return np.array([0.0]), np.array([2.0]), 1250.0

    monkeypatch.setattr(lfp_loading, "decode_lfp_sync", fake_decode)
    monkeypatch.setattr(lfp_loading, "load_trial_lfp_trace_with_sample_rate", fake_loader)
    loaded = load_site_trial_traces([site], np.array([0, 1]), np.array([1.0, 2.0]), (-1.0, 1.0))

    assert decode_calls == [Path("pfc.bin")]
    assert len(loader_kwargs) == 2
    assert all(
        set(kwargs) == {
            "lfp_path", "saved_channel_index", "alignment_time_s", "window",
            "lfp_irig_df", "sample_rate_hz",
        }
        for kwargs in loader_kwargs
    )
    assert all(kwargs["lfp_irig_df"] is decoded_sync for kwargs in loader_kwargs)
    assert loaded["PFC"].sample_rate_hz == 1250.0


def test_default_open_ephys_route_reports_lfp_metadata_rate_not_ap_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open Ephys routing obtains LFP rate from metadata and passes only its real API kwargs."""
    site = LFPSiteConfig("HPC", "HPC", "open_ephys", Path("hpc.dat"), Path("hpc.npz"), "HPC", 2, "uV", 1000.0)
    loader_kwargs: list[dict[str, object]] = []

    def fake_metadata(_: Path) -> dict[str, object]:
        """Return authoritative 1000-Hz, microvolt metadata for saved channel two."""
        return _observed_open_ephys_metadata(1000.0, 3)

    def fake_loader(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Record a complete native `(time, physical-uV)` Open Ephys trial window."""
        loader_kwargs.append(kwargs)
        relative_time_s = -1.0 + np.arange(2_000, dtype=float) / 1_000.0
        return relative_time_s, 3.0 + np.sin(2.0 * np.pi * 8.0 * relative_time_s)

    monkeypatch.setattr(lfp_loading, "load_open_ephys_lfp_metadata", fake_metadata)
    monkeypatch.setattr(lfp_loading, "load_open_ephys_trial_lfp_trace", fake_loader)
    loaded = load_site_trial_traces([site], np.array([0]), np.array([1.0]), (-1.0, 1.0))

    assert len(loader_kwargs) == 1
    assert set(loader_kwargs[0]) == {
        "lfp_path", "aligned_sync_npz_path", "saved_channel_index", "alignment_time_s", "window"
    }
    assert loaded["HPC"].sample_rate_hz == 1000.0
    assert loaded["HPC"].voltage_unit == "uV"
    np.testing.assert_array_equal(
        loaded["HPC"].relative_time_s,
        -1.0 + np.arange(2_000, dtype=float) / 1_000.0,
    )
    np.testing.assert_array_equal(
        loaded["HPC"].source_trace,
        3.0 + np.sin(2.0 * np.pi * 8.0 * loaded["HPC"].relative_time_s)[np.newaxis, :],
    )


def test_open_ephys_preparation_retains_physical_uv_grid_validity_and_qc_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production adapter consumes reader-provided uV values without changing QC axes or scale."""
    site = LFPSiteConfig(
        "HPC", "HPC", "open_ephys", Path("hpc.dat"), Path("hpc.npz"), "HPC", 0, "uV", 1000.0
    )
    native_time_s = -0.00125 + np.arange(6, dtype=float) / 1000.0
    stored_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 1.0])
    prepared_by_gain: dict[float, object] = {}

    for gain_to_uv in (1.0, 0.195):
        metadata = _observed_open_ephys_metadata(1000.0, 1)
        scaling = metadata["lfp_binary_scaling"]
        assert isinstance(scaling, dict)
        scaling["gain_to_uV_by_channel"] = [gain_to_uv]
        monkeypatch.setattr(lfp_loading, "load_open_ephys_lfp_metadata", lambda _: metadata)
        monkeypatch.setattr(
            lfp_loading,
            "load_open_ephys_trial_lfp_trace",
            lambda **_: (native_time_s, gain_to_uv * stored_values),
        )
        prepared_by_gain[gain_to_uv] = load_site_trial_traces(
            [site],
            np.array([11, 12]),
            np.array([100.0, np.nan]),
            (-0.001, 0.003),
        )["HPC"]

    identity = prepared_by_gain[1.0]
    corrected = prepared_by_gain[0.195]
    expected_grid = np.array([-0.001, 0.0, 0.001, 0.002])
    assert identity.voltage_unit == corrected.voltage_unit == "uV"
    assert identity.sample_rate_hz == corrected.sample_rate_hz == 1000.0
    np.testing.assert_array_equal(identity.relative_time_s, expected_grid)
    np.testing.assert_array_equal(corrected.relative_time_s, expected_grid)
    assert identity.valid.tolist() == corrected.valid.tolist() == [True, False]
    assert identity.exclusion_reason.tolist() == corrected.exclusion_reason.tolist() == ["", "missing_alignment"]
    np.testing.assert_allclose(corrected.source_trace[0], 0.195 * identity.source_trace[0])
    np.testing.assert_allclose(corrected.rms[0], 0.195 * identity.rms[0])
    np.testing.assert_allclose(corrected.peak_to_peak[0], 0.195 * identity.peak_to_peak[0])
    assert np.isnan(identity.source_trace[1]).all()
    assert np.isnan(corrected.source_trace[1]).all()
    assert np.isnan(identity.rms[1]) and np.isnan(corrected.rms[1])
    assert np.isnan(identity.peak_to_peak[1]) and np.isnan(corrected.peak_to_peak[1])


@pytest.mark.parametrize(
    ("configured_rate_hz", "configured_unit", "expected_message"),
    (
        (
            12.5,
            "uV",
            "Open Ephys site PFC: configured sample_rate_hz=12.5 does not match authoritative sample_rate_hz=10.0",
        ),
        (
            10.0,
            "mV",
            "Open Ephys site PFC: configured voltage_unit=mV does not match authoritative voltage_unit=uV",
        ),
    ),
)
def test_open_ephys_summary_route_rejects_configured_metadata_disagreement_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    configured_rate_hz: float,
    configured_unit: str,
    expected_message: str,
) -> None:
    """Production site labels must agree with the sidecar before traces or caches exist."""
    site = LFPSiteConfig(
        "PFC",
        "PFC",
        "open_ephys",
        Path("lfp.dat"),
        Path("sync.npz"),
        "PFC",
        0,
        configured_unit,
        configured_rate_hz,
    )
    loader_called = False

    monkeypatch.setattr(
        lfp_loading,
        "load_open_ephys_lfp_metadata",
        lambda _: _observed_open_ephys_metadata(10.0, 1),
    )

    def forbidden_loader(**_: object) -> tuple[np.ndarray, np.ndarray]:
        """Fail if numerical Open Ephys loading starts after a metadata mismatch."""
        nonlocal loader_called
        loader_called = True
        raise AssertionError("numerical loading must not run after a metadata mismatch")

    monkeypatch.setattr(lfp_loading, "load_open_ephys_trial_lfp_trace", forbidden_loader)

    with pytest.raises(ValueError) as error:
        load_site_trial_traces([site], np.array([0]), np.array([1.0]), (-0.1, 0.1))
    assert not loader_called
    assert str(error.value) == expected_message


def test_open_ephys_fractional_alignment_is_interpolated_to_exact_native_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sub-sample sync offset should retain valid exact event-relative samples.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces Open Ephys metadata and trace loading with a fractional-offset
        native grid matching the production loader's sample-window behavior.
    """
    site = LFPSiteConfig(
        "HPC",
        "HPC",
        "open_ephys",
        Path("hpc.dat"),
        Path("hpc.npz"),
        "HPC",
        2,
        "uV",
        1000.0,
    )

    monkeypatch.setattr(
        lfp_loading,
        "load_open_ephys_lfp_metadata",
        lambda _: _observed_open_ephys_metadata(1000.0, 3),
    )

    def fake_loader(**_: object) -> tuple[np.ndarray, np.ndarray]:
        """Return one extra native sample around a 0.25-ms alignment residual."""
        time_s = -0.00225 + np.arange(5, dtype=float) / 1000.0
        return time_s, 2.0 * time_s + 1.0

    monkeypatch.setattr(
        lfp_loading,
        "load_open_ephys_trial_lfp_trace",
        fake_loader,
    )

    loaded = load_site_trial_traces(
        [site],
        np.array([0]),
        np.array([1.0]),
        (-0.002, 0.002),
    )["HPC"]

    assert loaded.valid.tolist() == [True]
    assert loaded.exclusion_reason.tolist() == [""]
    assert np.array_equal(
        loaded.relative_time_s,
        np.array([-0.002, -0.001, 0.0, 0.001]),
    )
    assert np.allclose(
        loaded.source_trace[0],
        2.0 * loaded.relative_time_s + 1.0,
    )


def test_malformed_last_trial_cannot_change_canonical_trace_axis() -> None:
    """Later malformed traces retain the canonical native `(trial, time)` axis."""
    site = LFPSiteConfig("PFC", "PFC", "spikeglx", Path("pfc.bin"), None, "PFC", 5, "uV", 1000.0)
    results = [
        (np.array([-0.002, -0.001, 0.0, 0.001]), np.arange(4.0), 1000.0),
        (np.array([-0.002, 0.0]), np.array([1.0, 2.0]), 1000.0),
    ]
    prepared = prepare_site_trial_traces(site, np.array([0, 1]), np.array([1.0, 2.0]), (-0.002, 0.002), lambda *_: results.pop(0))
    assert np.array_equal(prepared.relative_time_s, np.array([-0.002, -0.001, 0.0, 0.001]))
    assert prepared.source_trace.shape == (2, 4)


def test_asymmetric_overlapping_windows_use_absolute_interval_endpoints() -> None:
    """Events at 0 and 2 overlap under the asymmetric half-open window [-2, 1)."""
    result = build_trial_relative_spike_trains(
        probe_label="PFC", cluster_id=10, unit_spike_times_s=np.array([0.0]),
        event_times_s=np.array([0.0, 2.0]), window=(-2.0, 1.0),
    )
    assert result.overlap_trial_indices.tolist() == [0, 1]


def test_common_grid_and_progress_reject_invalid_invariants() -> None:
    """Nonpositive grids and decreasing/over-complete progress records are invalid."""
    with pytest.raises(ValueError):
        build_common_event_grid(1.0, -1.0, 500.0)
    with pytest.raises(ValueError):
        build_common_event_grid(-1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        validate_progress_events([ProgressEvent("power", "load", 3, 2, "too far")])
