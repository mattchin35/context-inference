from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import ephys_sync_utils
from src.neural_analysis import sync_ephys


def _write_continuous_sample_numbers(
    continuous_dir: Path,
    sample_numbers: np.ndarray,
) -> None:
    """Write a minimal Open Ephys continuous sample-number file.

    Args:
        continuous_dir: Destination continuous-stream directory.
        sample_numbers: Open Ephys global sample numbers with shape
            ``(n_samples,)`` in samples.

    Returns:
        None.
    """
    continuous_dir.mkdir(parents=True, exist_ok=True)
    np.save(continuous_dir / "sample_numbers.npy", sample_numbers.astype(np.int64))


def _write_kilosort_spike_times(
    kilosort_dir: Path,
    spike_times: np.ndarray,
) -> None:
    """Write a minimal Kilosort spike-time file.

    Args:
        kilosort_dir: Destination Kilosort output directory.
        spike_times: Kilosort spike sample indices with shape ``(n_spikes,)``
            in samples relative to the sorted continuous data.

    Returns:
        None.
    """
    kilosort_dir.mkdir(parents=True, exist_ok=True)
    np.save(kilosort_dir / "spike_times.npy", spike_times.astype(np.int64))


def _irig_anchor_df() -> pd.DataFrame:
    """Build a minimal Open Ephys IRIG dataframe for sync tests.

    Returns:
        pd.DataFrame: Four IRIG anchor rows with ``sample_ix`` in Open Ephys
        global samples, ``recording_time_s`` in seconds, ``pulse_len_samples``
        in samples, ``irig_bit`` values, ``utc_unix`` in seconds, and
        timezone-aware ``utc_datetime`` values.
    """
    utc_unix = np.array([1000.0, 1001.0, 1002.0, 1003.0], dtype=float)
    return pd.DataFrame(
        {
            "sample_ix": np.array([100, 110, 120, 130], dtype=np.int64),
            "recording_time_s": np.array([10.0, 11.0, 12.0, 13.0], dtype=float),
            "pulse_len_samples": np.array([2, 2, 2, 2], dtype=np.int64),
            "irig_bit": np.array([False, True, "P", False], dtype=object),
            "utc_unix": utc_unix,
            "utc_datetime": [
                datetime.fromtimestamp(value, tz=timezone.utc)
                for value in utc_unix
            ],
        }
    )


def test_load_open_ephys_continuous_sample_bounds_reads_first_last_and_count(tmp_path: Path):
    continuous_dir = tmp_path / "continuous" / "ProbeA"
    _write_continuous_sample_numbers(
        continuous_dir,
        np.arange(370_032, 370_037, dtype=np.int64),
    )

    bounds = ephys_sync_utils.load_open_ephys_continuous_sample_bounds(continuous_dir)

    assert bounds.first_sample_ix == 370_032
    assert bounds.last_sample_ix == 370_036
    assert bounds.n_samples == 5


def test_convert_kilosort_samples_to_open_ephys_samples_adds_start_offset():
    spike_sample_ix = np.array([0, 10, 25], dtype=np.int64)

    open_ephys_sample_ix = ephys_sync_utils.convert_kilosort_samples_to_open_ephys_samples(
        spike_sample_ix=spike_sample_ix,
        continuous_start_sample_ix=370_032,
    )

    assert np.array_equal(
        open_ephys_sample_ix,
        np.array([370_032, 370_042, 370_057], dtype=np.int64),
    )


def test_sync_open_ephys_kilosort_spikes_to_utc_maps_adjusted_samples(tmp_path: Path):
    kilosort_dir = tmp_path / "kilosort4"
    continuous_dir = tmp_path / "continuous" / "ProbeA"
    ttl_dir = tmp_path / "events" / "ProbeA" / "TTL"
    ttl_dir.mkdir(parents=True)
    _write_kilosort_spike_times(kilosort_dir, np.array([0, 5, 20], dtype=np.int64))
    _write_continuous_sample_numbers(continuous_dir, np.arange(100, 151, dtype=np.int64))

    spike_df, irig_df = ephys_sync_utils.sync_open_ephys_kilosort_spikes_to_utc(
        kilosort_dir=kilosort_dir,
        ttl_dir=ttl_dir,
        continuous_dir=continuous_dir,
        decode_open_ephys_ttl_irig_utc_fn=lambda **kwargs: _irig_anchor_df(),
    )

    assert np.array_equal(
        spike_df["sample_ix"].to_numpy(dtype=np.int64),
        np.array([0, 5, 20], dtype=np.int64),
    )
    assert np.array_equal(
        spike_df["open_ephys_sample_ix"].to_numpy(dtype=np.int64),
        np.array([100, 105, 120], dtype=np.int64),
    )
    assert np.allclose(spike_df["utc_unix"].to_numpy(dtype=float), [1000.0, 1000.5, 1002.0])
    assert np.array_equal(irig_df["sample_ix"].to_numpy(dtype=np.int64), np.array([100, 110, 120, 130]))


def test_sync_open_ephys_kilosort_spikes_to_utc_applies_utc_offset(tmp_path: Path):
    kilosort_dir = tmp_path / "kilosort4"
    continuous_dir = tmp_path / "continuous" / "ProbeA"
    ttl_dir = tmp_path / "events" / "ProbeA" / "TTL"
    ttl_dir.mkdir(parents=True)
    _write_kilosort_spike_times(kilosort_dir, np.array([10], dtype=np.int64))
    _write_continuous_sample_numbers(continuous_dir, np.arange(100, 151, dtype=np.int64))

    spike_df, irig_df = ephys_sync_utils.sync_open_ephys_kilosort_spikes_to_utc(
        kilosort_dir=kilosort_dir,
        ttl_dir=ttl_dir,
        continuous_dir=continuous_dir,
        utc_offset_hours=1.0,
        decode_open_ephys_ttl_irig_utc_fn=lambda **kwargs: _irig_anchor_df(),
    )

    assert np.allclose(irig_df["utc_unix"].to_numpy(dtype=float), np.array([4600.0, 4601.0, 4602.0, 4603.0]))
    assert np.allclose(spike_df["utc_unix"].to_numpy(dtype=float), np.array([4601.0]))


def test_sync_open_ephys_kilosort_spikes_to_utc_warns_but_keeps_extrapolated_spikes(
    tmp_path: Path,
):
    kilosort_dir = tmp_path / "kilosort4"
    continuous_dir = tmp_path / "continuous" / "ProbeA"
    ttl_dir = tmp_path / "events" / "ProbeA" / "TTL"
    ttl_dir.mkdir(parents=True)
    _write_kilosort_spike_times(kilosort_dir, np.array([-10, 0, 40], dtype=np.int64))
    _write_continuous_sample_numbers(continuous_dir, np.arange(100, 151, dtype=np.int64))

    with pytest.warns(RuntimeWarning, match="outside finite IRIG UTC anchor range"):
        spike_df, _ = ephys_sync_utils.sync_open_ephys_kilosort_spikes_to_utc(
            kilosort_dir=kilosort_dir,
            ttl_dir=ttl_dir,
            continuous_dir=continuous_dir,
            decode_open_ephys_ttl_irig_utc_fn=lambda **kwargs: _irig_anchor_df(),
        )

    assert np.array_equal(
        spike_df["open_ephys_sample_ix"].to_numpy(dtype=np.int64),
        np.array([90, 100, 140], dtype=np.int64),
    )
    assert np.allclose(spike_df["utc_unix"].to_numpy(dtype=float), np.array([999.0, 1000.0, 1004.0]))


def test_sync_open_ephys_kilosort_spikes_to_utc_saves_expected_npz_schema_and_meta(
    tmp_path: Path,
):
    kilosort_dir = tmp_path / "kilosort4"
    continuous_dir = tmp_path / "continuous" / "ProbeA"
    ttl_dir = tmp_path / "events" / "ProbeA" / "TTL"
    output_file = tmp_path / "aligned" / "aligned_open_ephys" / "probeA_sync.npz"
    ttl_dir.mkdir(parents=True)
    _write_kilosort_spike_times(kilosort_dir, np.array([0, 10], dtype=np.int64))
    _write_continuous_sample_numbers(continuous_dir, np.arange(100, 151, dtype=np.int64))

    ephys_sync_utils.sync_open_ephys_kilosort_spikes_to_utc(
        kilosort_dir=kilosort_dir,
        ttl_dir=ttl_dir,
        continuous_dir=continuous_dir,
        output_file=output_file,
        probe_name="ProbeA",
        utc_offset_hours=-0.5,
        decode_open_ephys_ttl_irig_utc_fn=lambda **kwargs: _irig_anchor_df(),
    )

    with np.load(output_file, allow_pickle=True) as loaded:
        assert set(loaded.files) == {
            "spike_sample_ix",
            "spike_open_ephys_sample_ix",
            "spike_utc_unix",
            "irig_sample_ix",
            "irig_recording_time_s",
            "irig_pulse_len_samples",
            "irig_bit",
            "irig_utc_unix",
            "meta",
        }
        assert np.array_equal(loaded["spike_sample_ix"], np.array([0, 10], dtype=np.int64))
        assert np.array_equal(loaded["spike_open_ephys_sample_ix"], np.array([100, 110], dtype=np.int64))
        meta = loaded["meta"].item()
        assert meta["generator"] == "sync_open_ephys_kilosort_spikes_to_utc"
        assert meta["probe_name"] == "ProbeA"
        assert meta["continuous_start_sample_ix"] == 100
        assert meta["continuous_last_sample_ix"] == 150
        assert meta["utc_offset_hours"] == -0.5


def test_main_open_ephys_workflow_syncs_probe_a_and_probe_b_with_expected_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    session_data_home = tmp_path / "CT026_20260727_alternating_latent"
    raw_recording_dir = (
        session_data_home
        / "ephys"
        / "raw"
        / "2026-07-27_14-37-43"
        / "Record Node 101"
        / "experiment1"
        / "recording1"
    )
    for probe_name in ("ProbeA", "ProbeB"):
        processor_name = f"Neuropix-PXI-100.{probe_name}"
        (raw_recording_dir / "continuous" / processor_name).mkdir(parents=True)
        (raw_recording_dir / "events" / processor_name / "TTL").mkdir(parents=True)
        (
            session_data_home
            / "ephys"
            / "derived"
            / f"Record_Node_101_{processor_name}"
            / "kilosort4"
        ).mkdir(parents=True)

    calls: list[dict[str, object]] = []

    def fake_sync_open_ephys_kilosort_spikes_to_utc(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(
        ephys_sync_utils,
        "sync_open_ephys_kilosort_spikes_to_utc",
        fake_sync_open_ephys_kilosort_spikes_to_utc,
    )

    result = sync_ephys.main_open_ephys_workflow(
        session_data_home=session_data_home,
        raw_recording_name="2026-07-27_14-37-43",
        record_node_name="Record Node 101",
        processor_prefix="Neuropix-PXI-100.",
        probe_names=("ProbeA", "ProbeB"),
        utc_offset_hours=1.0,
    )

    assert set(result.keys()) == {"ProbeA", "ProbeB"}
    assert [Path(call["output_file"]).name for call in calls] == ["probeA_sync.npz", "probeB_sync.npz"]
    assert calls[0]["ttl_dir"] == raw_recording_dir / "events" / "Neuropix-PXI-100.ProbeA" / "TTL"
    assert calls[1]["ttl_dir"] == raw_recording_dir / "events" / "Neuropix-PXI-100.ProbeB" / "TTL"
    assert calls[0]["utc_offset_hours"] == 1.0
    assert calls[1]["probe_name"] == "ProbeB"
