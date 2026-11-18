import warnings

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import analog_treadmill_decode
from src.neural_analysis import manual_session_synchronization


def test_load_behavior_flipper_timestamps_reads_required_columns(tmp_path):
    flipper_csv = tmp_path / "session_flipper_output.csv"
    pd.DataFrame(
        {
            "pin_state": [1, 0, 1, 0],
            "time.time()": [1000.0, 1000.25, 1001.0, 1001.25],
        }
    ).to_csv(flipper_csv, index=False)

    timestamps = manual_session_synchronization.load_behavior_flipper_timestamps(flipper_csv)

    np.testing.assert_allclose(timestamps["positive_utc_seconds"], [1000.0, 1001.0])
    np.testing.assert_allclose(timestamps["negative_utc_seconds"], [1000.25, 1001.25])


def test_load_behavior_flipper_timestamps_rejects_missing_columns(tmp_path):
    flipper_csv = tmp_path / "bad_flipper_output.csv"
    pd.DataFrame({"pin_state": [1], "timestamp": [1000.0]}).to_csv(flipper_csv, index=False)

    with pytest.raises(ValueError, match="time.time"):
        manual_session_synchronization.load_behavior_flipper_timestamps(flipper_csv)


def test_load_behavior_flipper_timestamps_rejects_nonmonotonic_time(tmp_path):
    flipper_csv = tmp_path / "bad_flipper_output.csv"
    pd.DataFrame(
        {
            "pin_state": [1, 0, 1],
            "time.time()": [1000.0, 1000.5, 1000.4],
        }
    ).to_csv(flipper_csv, index=False)

    with pytest.raises(ValueError, match="monotonic"):
        manual_session_synchronization.load_behavior_flipper_timestamps(flipper_csv)


def test_match_timestamps_by_intervals_matches_embedded_sequence():
    daq_times_s = np.array([0.1, 1.0, 2.5, 4.0, 6.5, 8.0])
    behavior_utc_seconds = np.array([100.0, 101.5, 104.0])

    matched_daq, matched_behavior = manual_session_synchronization.match_timestamps_by_intervals(
        daq_times_s,
        behavior_utc_seconds,
        tolerance_s=1e-9,
    )

    np.testing.assert_allclose(matched_daq, [2.5, 4.0, 6.5])
    np.testing.assert_allclose(matched_behavior, behavior_utc_seconds)


def test_match_timestamps_by_intervals_raises_when_too_few_matches():
    with pytest.raises(ValueError, match="No matching timestamp sequence"):
        manual_session_synchronization.match_timestamps_by_intervals(
            np.array([0.0, 1.0, 3.0, 6.0]),
            np.array([100.0, 102.0, 105.5]),
            tolerance_s=0.001,
        )


def test_build_flipper_utc_mapper_maps_known_ni_times():
    daq_flipper_events = {
        "flipper_pos_t": np.array([2.0, 4.0, 6.0]),
        "flipper_neg_t": np.array([2.5, 4.5, 6.5]),
    }
    behavior_flipper_times = {
        "positive_utc_seconds": np.array([102.0, 104.0, 106.0]),
        "negative_utc_seconds": np.array([102.5, 104.5, 106.5]),
    }

    mapper = manual_session_synchronization.build_flipper_utc_mapper(
        daq_flipper_events,
        behavior_flipper_times,
        bounds_s=(2.0, 6.0),
        bounds_policy="reject",
    )

    mapped = mapper.map_ni_times_to_utc(np.array([3.0, 5.0]))
    np.testing.assert_allclose(mapped, [103.0, 105.0])
    np.testing.assert_allclose(mapper.negative_edge_residual_s, [0.0, 0.0, 0.0])


def test_apply_bounds_policy_reject_clip_warn_allow():
    times_s = np.array([0.0, 1.0, 2.0, 3.0])
    bounds_s = (1.0, 2.0)

    with pytest.raises(ValueError, match="outside barcode bounds"):
        manual_session_synchronization.apply_bounds_policy(times_s, bounds_s, policy="reject")

    np.testing.assert_allclose(
        manual_session_synchronization.apply_bounds_policy(times_s, bounds_s, policy="clip"),
        [1.0, 2.0],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warned = manual_session_synchronization.apply_bounds_policy(times_s, bounds_s, policy="warn")
    np.testing.assert_allclose(warned, times_s)
    assert any("outside barcode bounds" in str(item.message) for item in caught)

    np.testing.assert_allclose(
        manual_session_synchronization.apply_bounds_policy(times_s, bounds_s, policy="allow"),
        times_s,
    )


def test_map_imec_samples_to_utc_uses_imec_to_ni_then_ni_to_utc():
    daq_flipper_events = {
        "flipper_pos_t": np.array([10.0, 20.0, 30.0]),
        "flipper_neg_t": np.array([11.0, 21.0, 31.0]),
    }
    behavior_flipper_times = {
        "positive_utc_seconds": np.array([110.0, 120.0, 130.0]),
        "negative_utc_seconds": np.array([111.0, 121.0, 131.0]),
    }
    ni_to_utc = manual_session_synchronization.build_flipper_utc_mapper(
        daq_flipper_events,
        behavior_flipper_times,
        bounds_s=(10.0, 30.0),
        bounds_policy="reject",
    )
    imec_to_ni = manual_session_synchronization.make_timebase_mapper(
        source_times_s=np.array([0.0, 10.0]),
        target_times_s=np.array([10.0, 30.0]),
    )

    utc_seconds = manual_session_synchronization.map_imec_samples_to_utc(
        sample_ix=np.array([2500, 5000]),
        imec_sample_rate=1000.0,
        imec_to_ni=imec_to_ni,
        ni_to_utc=ni_to_utc,
        bounds_s=(10.0, 30.0),
        bounds_policy="reject",
    )

    np.testing.assert_allclose(utc_seconds, [115.0, 120.0])


def test_build_imec_to_ni_mapper_handles_extra_ni_sync_pulses():
    sample_rate = 100.0

    def make_pulse_signal(pulse_times_s, duration_s=0.05):
        signal = np.zeros(1800, dtype=bool)
        pulse_width = int(round(duration_s * sample_rate))
        for pulse_time_s in pulse_times_s:
            start_ix = int(round(pulse_time_s * sample_rate))
            signal[start_ix:start_ix + pulse_width] = True
        return signal

    imec_signal = make_pulse_signal([12.0, 13.5, 16.0])
    ni_signal = make_pulse_signal([1.0, 2.0, 3.5, 6.0, 9.0])

    imec_to_ni = manual_session_synchronization.build_imec_to_ni_mapper(
        imec_sync_signal=imec_signal,
        imec_sample_rate=sample_rate,
        ni_ephys_sync_signal=ni_signal,
        ni_sample_rate=sample_rate,
        tolerance_s=1e-9,
    )

    np.testing.assert_allclose(imec_to_ni(np.array([12.0, 16.0])), [2.0, 6.0])


def test_get_flipper_events_debounce_rejects_short_pulses():
    sample_rate_hz = 1000.0
    signal = np.zeros(20, dtype=bool)
    signal[2:3] = True
    signal[10:15] = True

    events = manual_session_synchronization.get_flipper_events(
        signal,
        sample_rate=sample_rate_hz,
        debounce=0.002,
    )

    np.testing.assert_array_equal(events["pos_ix"], [10])
    np.testing.assert_array_equal(events["neg_ix"], [15])


def test_decode_flipper_barcodes_without_barcode_preserves_event_keys():
    signal = np.zeros(20, dtype=bool)
    signal[4:10] = True

    events, flipper_bounds, barcodes = manual_session_synchronization.decode_flipper_barcodes(
        signal,
        sample_rate=1000.0,
        barcode_present=False,
    )

    np.testing.assert_array_equal(events["flipper_pos_t"], [0.004])
    np.testing.assert_array_equal(events["flipper_neg_t"], [0.010])
    np.testing.assert_allclose(flipper_bounds, (0.004, 0.010))
    assert barcodes is None


def test_analog_treadmill_decode_volts2speed_preserves_shape():
    millivolts = np.array([0.0, 1650.0, 3300.0])

    speed = analog_treadmill_decode.volts2speed(millivolts)

    assert speed.shape == millivolts.shape
    assert np.issubdtype(speed.dtype, np.floating)


def test_write_manual_alignment_note_identifies_flipper_sync(tmp_path):
    note_path = manual_session_synchronization.write_manual_alignment_note(tmp_path)

    note_text = note_path.read_text()
    assert "UTC hour adjustment: 0" in note_text
    assert "Processed at:" in note_text
    assert "manual line sync to the behavior flipper timestamps" in note_text


def test_load_spike_times_and_clusters_validate_matching_lengths(tmp_path):
    spike_times_file = tmp_path / "spike_times.npy"
    spike_clusters_file = tmp_path / "spike_clusters.npy"
    np.save(spike_times_file, np.array([[0], [1000], [2000]], dtype=np.int64))
    np.save(spike_clusters_file, np.array([[7], [8], [7]], dtype=np.int64))

    spike_times = manual_session_synchronization.load_spike_times_npy(spike_times_file)
    spike_clusters = manual_session_synchronization.load_spike_clusters_npy(spike_clusters_file)
    manual_session_synchronization.validate_spike_times_and_clusters(spike_times, spike_clusters)

    np.testing.assert_array_equal(spike_times, [0, 1000, 2000])
    np.testing.assert_array_equal(spike_clusters, [7, 8, 7])


def test_validate_spike_times_and_clusters_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        manual_session_synchronization.validate_spike_times_and_clusters(
            np.array([0, 1], dtype=np.int64),
            np.array([7], dtype=np.int64),
        )


def test_convert_spike_times_to_imec_seconds_from_samples():
    spike_imec_time_s, spike_sample_ix = manual_session_synchronization.convert_spike_times_to_imec_seconds(
        spike_times=np.array([0, 1000], dtype=np.int64),
        imec_sample_rate=1000.0,
        spike_time_units="samples",
    )

    np.testing.assert_allclose(spike_imec_time_s, [0.0, 1.0])
    np.testing.assert_array_equal(spike_sample_ix, [0, 1000])


def test_convert_spike_times_to_imec_seconds_from_seconds():
    spike_imec_time_s, spike_sample_ix = manual_session_synchronization.convert_spike_times_to_imec_seconds(
        spike_times=np.array([0.25, 1.5], dtype=float),
        imec_sample_rate=1000.0,
        spike_time_units="seconds",
    )

    np.testing.assert_allclose(spike_imec_time_s, [0.25, 1.5])
    assert spike_sample_ix is None


def test_map_imec_spikes_to_utc_preserves_cluster_alignment():
    daq_flipper_events = {
        "flipper_pos_t": np.array([10.0, 20.0, 30.0]),
        "flipper_neg_t": np.array([11.0, 21.0, 31.0]),
    }
    behavior_flipper_times = {
        "positive_utc_seconds": np.array([110.0, 120.0, 130.0]),
        "negative_utc_seconds": np.array([111.0, 121.0, 131.0]),
    }
    ni_to_utc = manual_session_synchronization.build_flipper_utc_mapper(
        daq_flipper_events,
        behavior_flipper_times,
        bounds_s=(10.0, 30.0),
        bounds_policy="reject",
    )
    imec_to_ni = manual_session_synchronization.make_timebase_mapper(
        source_times_s=np.array([0.0, 10.0]),
        target_times_s=np.array([10.0, 30.0]),
    )

    spike_df = manual_session_synchronization.map_imec_spikes_to_utc(
        spike_times=np.array([2500, 5000], dtype=np.int64),
        spike_clusters=np.array([7, 8], dtype=np.int64),
        imec_sample_rate=1000.0,
        imec_to_ni=imec_to_ni,
        ni_to_utc=ni_to_utc,
        spike_time_units="samples",
        bounds_policy="reject",
    )

    np.testing.assert_array_equal(spike_df["spike_ix"].to_numpy(dtype=np.int64), [0, 1])
    np.testing.assert_array_equal(spike_df["cluster_id"].to_numpy(dtype=np.int64), [7, 8])
    np.testing.assert_array_equal(spike_df["spike_sample_ix"].to_numpy(dtype=np.int64), [2500, 5000])
    np.testing.assert_allclose(spike_df["spike_imec_time_s"], [2.5, 5.0])
    np.testing.assert_allclose(spike_df["spike_ni_time_s"], [15.0, 20.0])
    np.testing.assert_allclose(spike_df["spike_utc_unix"], [115.0, 120.0])


def test_sync_imec_spikes_to_manual_utc_saves_sync_ephys_compatible_npz(tmp_path):
    spike_times_file = tmp_path / "spike_times.npy"
    spike_clusters_file = tmp_path / "spike_clusters.npy"
    output_file = tmp_path / "aligned_imec" / "imec0_sync.npz"
    np.save(spike_times_file, np.array([2500, 5000], dtype=np.int64))
    np.save(spike_clusters_file, np.array([7, 8], dtype=np.int64))

    daq_flipper_events = {
        "flipper_pos_t": np.array([10.0, 20.0, 30.0]),
        "flipper_neg_t": np.array([11.0, 21.0, 31.0]),
    }
    behavior_flipper_times = {
        "positive_utc_seconds": np.array([110.0, 120.0, 130.0]),
        "negative_utc_seconds": np.array([111.0, 121.0, 131.0]),
    }
    ni_to_utc = manual_session_synchronization.build_flipper_utc_mapper(
        daq_flipper_events,
        behavior_flipper_times,
        bounds_s=(10.0, 30.0),
        bounds_policy="reject",
    )
    imec_to_ni = manual_session_synchronization.make_timebase_mapper(
        source_times_s=np.array([0.0, 10.0]),
        target_times_s=np.array([10.0, 30.0]),
    )

    spike_df = manual_session_synchronization.sync_imec_spikes_to_manual_utc(
        spike_times_npy=spike_times_file,
        spike_clusters_npy=spike_clusters_file,
        imec_sample_rate=1000.0,
        imec_to_ni=imec_to_ni,
        ni_to_utc=ni_to_utc,
        output_file=output_file,
        spike_time_units="samples",
    )

    np.testing.assert_allclose(spike_df["spike_utc_unix"], [115.0, 120.0])
    with np.load(output_file, allow_pickle=True) as loaded:
        assert "spike_utc_unix" in loaded
        assert "spike_clusters" in loaded
        assert "spike_sample_ix" in loaded
        assert "meta" in loaded
        np.testing.assert_allclose(loaded["spike_utc_unix"], [115.0, 120.0])
        np.testing.assert_array_equal(loaded["spike_clusters"], [7, 8])
        np.testing.assert_array_equal(loaded["spike_sample_ix"], [2500, 5000])
        assert loaded["spike_utc_unix"].shape == loaded["spike_clusters"].shape
        meta = loaded["meta"].item()
        assert meta["generator"] == "sync_imec_spikes_to_manual_utc"
        assert meta["sync_method"] == "manual_line_sync_to_behavior_flipper"
        assert meta["utc_reference"] == "behavior_flipper_time.time()"
        assert meta["utc_offset_hours"] == 0.0
