"""Independent physical-voltage/IRIG waveform and selected-spike oracles (WP1)."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp.loading import read_open_ephys_lfp_channel_window
from src.neural_analysis.lfp.spectrogram import apply_optional_60_hz_notch
from src.neural_analysis.phase_synchrony.results import (
    ContinuousSiteLFP,
    PhaseSynchronyConfig,
    SelectedUnitSpikes,
)
from src.neural_analysis.phase_synchrony.preparation import (
    load_selected_unit_spikes,
    load_site_lfp,
    prepare_site_lfp,
    preprocess_site_lfp,
    sample_site_lfp_utc,
    select_unit_window_spikes,
    validate_pair_sample_rates,
)
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)


UTC_ORIGIN = 1_800_000_000.0


@pytest.fixture
def binary_session(tmp_path: Path):
    """Return resolved OE session and raw (25000,2) float32 stored values, unscaled."""
    rate = 2500.0
    samples = np.arange(25_000, dtype=np.float32)
    raw = np.column_stack((samples * 0.25 - 7, samples * -0.5 + 11)).astype(np.float32)
    directory = tmp_path / "probe"
    directory.mkdir()
    raw.tofile(directory / "lfp.dat")
    sidecar = {
        "output_binary": "lfp.dat", "sampling_frequency_hz": rate,
        "num_channels": 2, "num_segments": 1, "num_samples_by_segment": [len(raw)],
        "dtype": "float32", "binary_layout": "time_major_channel_interleaved",
        "channel_ids_in_binary_order": ["CH1", "CH2"],
        "lfp_binary_scaling": {
            "data_units": "unscaled_binary_values", "has_scaleable_traces": True,
            "channel_ids": ["CH1", "CH2"], "gain_to_uV_by_channel": [0.195, 2.0],
            "offset_to_uV_by_channel": [4.0, -3.0],
            "physical_unit_by_channel": ["uV", "uV"], "export_scale_factor": 1.0,
            "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
        },
    }
    (directory / "lfp_preprocessing.json").write_text(json.dumps(sidecar), encoding="ascii")
    # AP/global indices convert with known 30 kHz acquisition rate, not clock slope.
    np.savez(
        directory / "sync.npz",
        irig_sample_ix=944_908 + np.array([0, 12_500, 25_000]) * 12,
        irig_utc_unix=UTC_ORIGIN + np.array([0, 5.005, 10.02]),
        spike_utc_unix=UTC_ORIGIN + np.array([4, 4.25, 5, 5.75, 6, 4.5, 5.5, 4.75]),
        meta={"continuous_start_sample_ix": 944_908, "sample_rate_hz": 30_000.0},
    )
    sorter = directory / "ks"
    sorter.mkdir()
    np.save(sorter / "spike_clusters.npy", np.array([17, 17, 17, 17, 17, 19, 23, 29]))
    pd.DataFrame({
        "cluster_id": [17, 19, 23, 29], "ch": [1, 1, 0, 1],
        "group": ["good", "noise", "good", "mua"],
    }).to_csv(sorter / "cluster_info.tsv", sep="\t", index=False)
    pd.DataFrame({
        "ch": [0, 1], "channel_id": ["CH0", "CH1"], "label": ["bad", "good"],
        "inside_brain": [True, True], "x_um": [0.0, 20.0], "y_um": [0.0, 20.0],
    }).to_csv(
        directory / "channel_quality.csv", index=False
    )
    payload = {
        "schema_version": "2", "session": "synthetic_2030-01-02_030405",
        "acquisition": "open_ephys", "behavior": {"trials": "trials.csv", "events": None},
        "probes": {"ProbeZ": {
            "lfp": "probe/lfp.dat", "alignment": "probe/sync.npz",
            "sorter": "probe/ks", "quality": "probe/channel_quality.csv",
            "sites": {"NamedSite": 1}, "unit_channels": [1],
        }},
        "site_pairs": [],
    }
    path = tmp_path / "neural_session.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(path), path), raw


def _source(
    sample_indices: np.ndarray,
    values_uv: np.ndarray,
    irig_df: pd.DataFrame,
    rate_hz: float,
    valid: np.ndarray | None = None,
) -> ContinuousSiteLFP:
    """Build calibrated (sample,) uV source with explicit native indices and UTC anchors."""
    return ContinuousSiteLFP(
        site_id="NamedSite", probe_id="ProbeZ", saved_channel_index=1,
        sample_indices=np.asarray(sample_indices), values_uv=np.asarray(values_uv, dtype=float),
        valid=np.ones(len(sample_indices), dtype=bool) if valid is None else valid,
        sample_rate_hz=rate_hz, irig_df=irig_df,
        meta={"voltage_unit": "uV", "notch_enabled": False, "synchronization_id": "synthetic"},
    )


def _choices(times_s: list[float]) -> pd.DataFrame:
    """Return (trial,columns) identities with UTC-second choices; no neural filtering."""
    return pd.DataFrame({
        "choice_time": times_s, "trial_id": [21 + 7 * i for i in range(len(times_s))],
        "row_position": [2 + 3 * i for i in range(len(times_s))],
    })


def test_continuous_loader_uses_named_channel_and_exactly_one_affine_uv_conversion(binary_session):
    """Real binary read returns (sample,) calibrated uV; source bytes are unchanged."""
    session, raw = binary_session
    binary_path = session.probes[0].lfp_file
    before = binary_path.read_bytes()
    source = load_site_lfp(session, "NamedSite", start_sample=11, stop_sample=101)
    expected, rate = read_open_ephys_lfp_channel_window(binary_path, 1, 11, 101)
    np.testing.assert_array_equal(source.values_uv, expected)
    np.testing.assert_allclose(source.values_uv, raw[11:101, 1] * 2.0 - 3.0, rtol=0, atol=0)
    np.testing.assert_array_equal(source.sample_indices, np.arange(11, 101))
    assert source.sample_rate_hz == rate == 2500.0
    assert source.site_id == "NamedSite" and source.probe_id == "ProbeZ"
    assert source.saved_channel_index == 1
    assert source.meta["voltage_unit"] == "uV"
    assert source.valid.all()
    assert binary_path.read_bytes() == before
    with pytest.raises(FrozenInstanceError):
        source.sample_rate_hz = 500.0


def test_oe_known_acquisition_rate_preserves_irig_clock_drift(binary_session):
    """30 kHz AP to 2500 Hz LFP conversion retains altered UTC anchor slopes."""
    session, _ = binary_session
    source = load_site_lfp(session, "NamedSite", start_sample=0, stop_sample=25_000)
    np.testing.assert_allclose(source.irig_df.sample_ix, [0, 12_500, 25_000], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        source.irig_df.utc_unix - UTC_ORIGIN, [0, 5.005, 10.02], rtol=0, atol=2e-7
    )
    target = UTC_ORIGIN + 5.005
    values, valid = sample_site_lfp_utc(source, np.array([target]))
    np.testing.assert_allclose(values, [-12_481.0], rtol=0, atol=1e-6)
    assert valid.tolist() == [True]


def test_clock_offset_changes_lfp_anchors_once(binary_session):
    """A configured hour offset is applied once to LFP anchor UTC seconds only."""
    session, _ = binary_session
    source = load_site_lfp(session, "NamedSite", start_sample=0, stop_sample=100, utc_offset_hours=1)
    assert source.irig_df.utc_unix.iloc[0] == UTC_ORIGIN + 3600
    np.testing.assert_array_equal(source.sample_indices, np.arange(100))


def test_native_choice_grid_has_exact_half_open_observation_count(binary_session):
    """An off-grid choice returns 5000 metric samples, excluding extra loader brackets."""
    session, _ = binary_session
    source = load_site_lfp(session, "NamedSite", start_sample=0, stop_sample=25_000)
    choices = _choices([UTC_ORIGIN + 5.12345])
    waveform = prepare_site_lfp(source, choices, interval="primary")
    relative = -1.0 + np.arange(5000) / 2500.0
    assert waveform.values_uv.shape == waveform.valid.shape == (1, 5000)
    np.testing.assert_array_equal(waveform.relative_time_s, relative)
    assert waveform.relative_time_s[-1] < 1.0
    np.testing.assert_array_equal(waveform.trial_ids, [21])
    np.testing.assert_array_equal(waveform.row_positions, [2])
    np.testing.assert_array_equal(waveform.choice_times_s, choices.choice_time)
    assert waveform.sample_rate_hz == 2500.0
    assert waveform.valid.all()
    # The second anchor lies inside the row: one nominal clock slope cannot pass.
    targets_relative = choices.choice_time.iloc[0] - UTC_ORIGIN + relative
    indices = np.interp(targets_relative, [0, 5.005000114440918, 10.019999980926514], [0, 12500, 25000])
    np.testing.assert_allclose(waveform.values_uv[0], 19 - indices, rtol=0, atol=8e-4)


@pytest.mark.parametrize("offset_s,drift", [(0.013, 1.004), (-0.027, 0.997)])
def test_independent_probe_clock_sampling_recovers_shared_utc_voltage(offset_s, drift):
    """Affine clock fixtures yield analytic uV values at identical off-grid UTC targets."""
    rate = 2500.0
    samples = np.arange(30_000)
    native_utc_relative = offset_s + samples / (rate * drift)
    values_uv = 3 * native_utc_relative - 2
    anchors = pd.DataFrame({
        "sample_ix": [0.0, 29_999.0],
        "utc_unix": UTC_ORIGIN + np.array([native_utc_relative[0], native_utc_relative[-1]]),
    })
    source = _source(samples, values_uv, anchors, rate)
    choice = UTC_ORIGIN + 9.12345
    waveform = prepare_site_lfp(source, _choices([choice]), interval="primary")
    expected = 3 * ((choice - UTC_ORIGIN) + waveform.relative_time_s) - 2
    np.testing.assert_allclose(waveform.values_uv[0], expected, rtol=0, atol=2e-6)
    assert waveform.valid.all()
    assert waveform.meta["waveform_alignment_semantics"] == "irig_uniform_utc_v1"


def test_exact_neighbor_gap_and_extrapolation_validity_are_distinct():
    """Exact hits need only themselves; fractional samples need contiguous valid neighbors."""
    samples = np.array([0, 1, 2, 4, 5])
    anchors = pd.DataFrame({"sample_ix": [0, 5], "utc_unix": UTC_ORIGIN + np.array([0, 5])})
    source = _source(samples, samples * 10, anchors, 1.0, np.array([True, False, True, True, True]))
    targets = UTC_ORIGIN + np.array([-0.25, 0, 0.5, 1, 1.5, 2, 3, 4, 4.5, 5, 5.25])
    values, valid = sample_site_lfp_utc(source, targets)
    np.testing.assert_array_equal(valid, [False, True, False, False, False, True, False, True, True, True, False])
    np.testing.assert_array_equal(values[valid], [0, 20, 40, 45, 50])
    assert np.isnan(values[~valid]).all()


def test_nonfinite_voltage_stays_invalid_without_bridging():
    """A nonfinite (sample,) uV value invalidates itself and interpolations using it."""
    anchors = pd.DataFrame({"sample_ix": [0, 3], "utc_unix": UTC_ORIGIN + np.array([0, 3])})
    source = _source(np.arange(4), np.array([1, np.nan, 3, 4]), anchors, 1.0)
    values, valid = sample_site_lfp_utc(source, UTC_ORIGIN + np.array([0, 0.5, 1, 1.5, 2]))
    np.testing.assert_array_equal(valid, [True, False, False, False, True])
    assert np.isnan(values[~valid]).all()


def test_signal_support_does_not_override_shorter_irig_support():
    """Even present source samples cannot justify extrapolation beyond anchor UTC support."""
    anchors = pd.DataFrame({"sample_ix": [2, 6], "utc_unix": UTC_ORIGIN + np.array([2, 6])})
    source = _source(np.arange(10), np.arange(10), anchors, 1.0)
    _, valid = sample_site_lfp_utc(source, UTC_ORIGIN + np.array([1, 2, 6, 7]))
    np.testing.assert_array_equal(valid, [False, True, True, False])


def test_epoch_second_tolerance_cannot_snap_distinct_fractional_native_samples():
    """At large UTC epochs, interpolation retains sub-sample offsets with absolute tolerance."""
    rate = 4.0
    anchors = pd.DataFrame({"sample_ix": [0, 8], "utc_unix": UTC_ORIGIN + np.array([0, 2])})
    source = _source(np.arange(9), np.arange(9) * 4.0, anchors, rate)
    targets = UTC_ORIGIN + np.array([0.25, 0.28125, 0.5])
    values, valid = sample_site_lfp_utc(source, targets)
    np.testing.assert_allclose(values, [4, 4.5, 8], rtol=0, atol=1e-12)
    assert valid.all()


def test_utc_rounding_preserves_exact_native_sample_with_invalid_neighbors():
    """A 2500-Hz exact hit uses its own validity despite UTC-double rounding.

    Inputs are (2501,) native samples in uV and UTC-second targets at a large
    epoch. A distinguishable one-eighth-sample offset still needs both neighbors.
    """
    samples = np.arange(2501)
    anchors = pd.DataFrame({
        "sample_ix": [0, 2500], "utc_unix": [UTC_ORIGIN, UTC_ORIGIN + 1],
    })
    source = _source(samples, samples.astype(float), anchors, 2500.0, samples == 1)
    values, valid = sample_site_lfp_utc(
        source, UTC_ORIGIN + np.array([1 / 2500, 1.125 / 2500]),
    )
    np.testing.assert_array_equal(valid, [True, False])
    assert values[0] == 1.0
    assert np.isnan(values[1])


def test_valid_primary_survives_missing_secondary_and_unrelated_boundary_trial():
    """Per-trial (trial,time) support remains independent between centered intervals."""
    rate = 500.0
    samples = np.arange(1501)
    anchors = pd.DataFrame({"sample_ix": [0, 1500], "utc_unix": UTC_ORIGIN + np.array([0, 3])})
    source = _source(samples, samples * 0.01, anchors, rate)
    choices = _choices([UTC_ORIGIN + 0.25, UTC_ORIGIN + 1.5])
    primary = prepare_site_lfp(source, choices, interval="primary")
    secondary = prepare_site_lfp(source, choices, interval="secondary")
    assert primary.values_uv.shape == (2, 1000)
    assert secondary.values_uv.shape == (2, 2000)
    np.testing.assert_array_equal(primary.valid.all(axis=1), [False, True])
    np.testing.assert_array_equal(secondary.valid.all(axis=1), [False, False])
    np.testing.assert_array_equal(primary.trial_ids, secondary.trial_ids)
    np.testing.assert_array_equal(primary.row_positions, [2, 5])


def test_continuous_notch_matches_existing_oracle_and_is_not_applied_twice():
    """One continuous (sample,) uV preprocessing pass matches the established notch."""
    rate = 2500.0
    samples = np.arange(25_000)
    time_s = samples / rate
    values = 20 * np.sin(2 * np.pi * 8 * time_s) + 15 * np.sin(2 * np.pi * 60 * time_s)
    anchors = pd.DataFrame({"sample_ix": [0, 24999], "utc_unix": UTC_ORIGIN + np.array([0, time_s[-1]])})
    source = _source(samples, values, anchors, rate)
    original = source.values_uv.copy()
    config = PhaseSynchronyConfig()
    first = preprocess_site_lfp(source, config)
    expected = apply_optional_60_hz_notch(values, rate, enabled=True, quality_factor=30.0)
    np.testing.assert_allclose(first.values_uv, expected, rtol=0, atol=1e-12)
    second = preprocess_site_lfp(first, config)
    np.testing.assert_array_equal(second.values_uv, first.values_uv)
    np.testing.assert_array_equal(source.values_uv, original)
    waveform = prepare_site_lfp(first, _choices([UTC_ORIGIN + 5]), interval="primary")
    np.testing.assert_allclose(waveform.values_uv[0], expected[10_000:15_000], rtol=0, atol=2e-4)
    assert first.meta["notch_enabled"] is True


def test_disabled_notch_preserves_values_and_missing_gap():
    """Disabling continuous filtering preserves (sample,) uV values and validity mask."""
    anchors = pd.DataFrame({"sample_ix": [0, 3], "utc_unix": UTC_ORIGIN + np.array([0, 3])})
    source = _source(np.arange(4), np.array([1, 2, np.nan, 4]), anchors, 1.0)
    result = preprocess_site_lfp(source, PhaseSynchronyConfig(notch_enabled=False))
    np.testing.assert_array_equal(result.values_uv, source.values_uv)
    assert result.meta["notch_enabled"] is False


def test_pair_rate_mismatch_is_explicit_and_does_not_modify_independent_site():
    """Native pair rates must agree; mismatches raise without implicit resampling."""
    anchors = pd.DataFrame({"sample_ix": [0, 3], "utc_unix": UTC_ORIGIN + np.array([0, 0.003])})
    source = _source(np.arange(4), np.arange(4), anchors, 1000.0)
    other = replace(source, site_id="OtherSite", sample_rate_hz=2500.0)
    with pytest.raises(ValueError, match="rate"):
        validate_pair_sample_rates(source, other)
    assert source.sample_rate_hz == 1000.0 and other.sample_rate_hz == 2500.0
    validate_pair_sample_rates(source, replace(source, site_id="MatchingSite"))


def test_selected_unit_loading_uses_explicit_probe_id_and_existing_quality_rules(binary_session):
    """Real NPZ/TSV/CSV loads preserve selected (spike,) UTC seconds and source identity."""
    session, _ = binary_session
    unit = load_selected_unit_spikes(session, "ProbeZ:17")
    assert unit.unit_id == "ProbeZ:17" and unit.probe_id == "ProbeZ" and unit.cluster_id == 17
    np.testing.assert_array_equal(unit.times_s - UTC_ORIGIN, [4, 4.25, 5, 5.75, 6])
    # A qualifying mua unit is retained; explicit selection does not demand 50 spikes.
    mua = load_selected_unit_spikes(session, "ProbeZ:29")
    np.testing.assert_array_equal(mua.times_s - UTC_ORIGIN, [4.75])


@pytest.mark.parametrize("unit_id", ["17", "MissingProbe:17", "ProbeZ:19", "ProbeZ:23", "ProbeZ:999"])
def test_missing_or_quality_excluded_unit_is_not_replaced(binary_session, unit_id):
    """Unknown IDs, noise clusters and bad channels raise; no arbitrary unit substitution."""
    session, _ = binary_session
    with pytest.raises(ValueError):
        load_selected_unit_spikes(session, unit_id)


def test_spike_window_is_half_open_and_retains_sub_bin_precision():
    """SelectedUnitSpikes (spike,) UTC seconds selects [choice-a,choice+a) unchanged."""
    times = UTC_ORIGIN + np.array([3.999, 4, 4.123456, 5, 5.999, 6])
    unit = SelectedUnitSpikes(unit_id="ProbeZ:17", probe_id="ProbeZ", cluster_id=17, times_s=times, meta={})
    primary = select_unit_window_spikes(unit, UTC_ORIGIN + 5, interval="primary")
    np.testing.assert_array_equal(primary, times[1:5])
    secondary = select_unit_window_spikes(unit, UTC_ORIGIN + 5, interval="secondary")
    np.testing.assert_array_equal(secondary, times)
    np.testing.assert_array_equal(unit.times_s, times)
