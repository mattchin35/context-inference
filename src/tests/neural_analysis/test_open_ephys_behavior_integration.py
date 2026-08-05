from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_loading
from src.neural_analysis import psth_webapp
from src.neural_analysis import spike_behavior_pynapple
from src.neural_analysis import unit_spike_loading


def _write_open_ephys_lfp(
    lfp_path: Path,
    lfp_matrix: np.ndarray,
    sample_rate_hz: float = 2_500.0,
) -> None:
    """Write a tiny derived Open Ephys LFP file and metadata JSON.

    Args:
        lfp_path: Destination binary path. The sibling metadata file is named
            ``lfp_preprocessing.json``.
        lfp_matrix: Time-major LFP array with shape
            ``(n_samples, n_channels)`` in neutral derived LFP units.
        sample_rate_hz: LFP sample rate in Hz.

    Returns:
        None.
    """
    lfp_path.parent.mkdir(parents=True, exist_ok=True)
    lfp_matrix.astype(np.float32).tofile(lfp_path)
    metadata = {
        "output_binary": lfp_path.name,
        "sampling_frequency_hz": float(sample_rate_hz),
        "num_channels": int(lfp_matrix.shape[1]),
        "num_segments": 1,
        "num_samples_by_segment": [int(lfp_matrix.shape[0])],
        "dtype": "float32",
        "binary_layout": "time_major_channel_interleaved",
    }
    (lfp_path.parent / "lfp_preprocessing.json").write_text(json.dumps(metadata) + "\n")


def _write_aligned_sync_npz(sync_path: Path) -> None:
    """Write a minimal aligned Open Ephys sync ``.npz`` for tests.

    Args:
        sync_path: Destination ``.npz`` path.

    Returns:
        None.
    """
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generator": "sync_open_ephys_kilosort_spikes_to_utc",
        "continuous_start_sample_ix": 1_000,
        "units": {
            "irig_sample_ix": "Open Ephys global samples",
            "utc_unix": "seconds",
        },
    }
    np.savez(
        sync_path,
        irig_sample_ix=np.array([1_000, 31_000, 61_000], dtype=np.int64),
        irig_utc_unix=np.array([100.0, 101.0, 102.0], dtype=float),
        meta=np.asarray(meta, dtype=object),
    )


def test_load_sorter_metadata_requires_phy_cluster_info(tmp_path: Path):
    sorter_dir = tmp_path / "kilosort4"
    sorter_dir.mkdir()
    np.save(sorter_dir / "spike_clusters.npy", np.array([1, 2, 1], dtype=np.int64))

    with pytest.raises(FileNotFoundError, match="manual spike curation in Phy"):
        spike_behavior_pynapple.load_sorter_metadata(sorter_dir)


def test_filter_cluster_metadata_can_use_kslabel_quality_column():
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "ch": [10, 10, 10],
            "group": [np.nan, "noise", np.nan],
            "KSLabel": ["good", "mua", "noise"],
        }
    )

    selected = unit_spike_loading.filter_cluster_metadata(
        cluster_info=cluster_info,
        region_channels=[10],
        quality_labels=("good", "mua"),
        quality_column="KSLabel",
    )

    assert selected["cluster_id"].tolist() == [1, 2]
    assert selected["quality_label"].tolist() == ["good", "mua"]


def test_load_open_ephys_lfp_metadata_reads_json_contract(tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(lfp_path, np.zeros((5, 3), dtype=np.float32), sample_rate_hz=1_250.0)

    metadata = lfp_loading.load_open_ephys_lfp_metadata(lfp_path)

    assert metadata["sampling_frequency_hz"] == 1_250.0
    assert metadata["num_channels"] == 3
    assert metadata["num_samples"] == 5
    assert metadata["dtype"] == "float32"
    assert metadata["binary_layout"] == "time_major_channel_interleaved"


def test_read_open_ephys_lfp_channel_window_reads_time_major_float32(tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    lfp_matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    _write_open_ephys_lfp(lfp_path, lfp_matrix, sample_rate_hz=2_500.0)

    lfp_values, sample_rate_hz = lfp_loading.read_open_ephys_lfp_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=2,
        start_sample=1,
        stop_sample=4,
    )

    np.testing.assert_allclose(lfp_values, np.array([6.0, 10.0, 14.0], dtype=float))
    assert sample_rate_hz == 2_500.0


def test_build_open_ephys_lfp_irig_df_converts_global_ap_samples_to_lfp_samples(tmp_path: Path):
    sync_path = tmp_path / "probeA_sync.npz"
    _write_aligned_sync_npz(sync_path)

    lfp_irig_df = lfp_loading.build_open_ephys_lfp_irig_df(
        aligned_sync_npz_path=sync_path,
        lfp_sample_rate_hz=2_500.0,
        continuous_sample_rate_hz=30_000.0,
    )

    np.testing.assert_allclose(lfp_irig_df["sample_ix"].to_numpy(dtype=float), np.array([0.0, 2500.0, 5000.0]))
    np.testing.assert_allclose(lfp_irig_df["utc_unix"].to_numpy(dtype=float), np.array([100.0, 101.0, 102.0]))


def test_load_open_ephys_trial_lfp_trace_uses_aligned_sync_npz(tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    sync_path = tmp_path / "probeA_sync.npz"
    lfp_matrix = np.arange(60, dtype=np.float32).reshape(20, 3)
    _write_open_ephys_lfp(lfp_path, lfp_matrix, sample_rate_hz=10.0)
    _write_aligned_sync_npz(sync_path)

    relative_time_s, lfp_values = lfp_loading.load_open_ephys_trial_lfp_trace(
        lfp_path=lfp_path,
        aligned_sync_npz_path=sync_path,
        saved_channel_index=1,
        alignment_time_s=101.0,
        window=(-0.2, 0.3),
        continuous_sample_rate_hz=30_000.0,
    )

    np.testing.assert_allclose(relative_time_s, np.array([-0.2, -0.1, 0.0, 0.1, 0.2]))
    np.testing.assert_allclose(lfp_values, lfp_matrix[8:13, 1].astype(float))


def test_webapp_open_ephys_lfp_route_does_not_require_spikeglx_meta(monkeypatch, tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    sync_path = tmp_path / "probeA_sync.npz"
    seen = {}

    def fake_load_open_ephys_trial_lfp_trace(**kwargs):
        seen.update(kwargs)
        return np.array([0.0], dtype=float), np.array([1.0], dtype=float)

    monkeypatch.setattr(
        psth_webapp.lfp_loading,
        "load_open_ephys_trial_lfp_trace",
        fake_load_open_ephys_trial_lfp_trace,
    )

    relative_time_s, lfp_values = psth_webapp.load_trial_lfp_trace_for_format(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path),
        saved_channel_index=3,
        alignment_time_s=101.0,
        window_start_s=-0.5,
        window_end_s=0.5,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        filter_low_hz=None,
        filter_high_hz=None,
        filter_padding_s=1.0,
        aligned_sync_npz_path=str(sync_path),
    )

    np.testing.assert_allclose(relative_time_s, np.array([0.0]))
    np.testing.assert_allclose(lfp_values, np.array([1.0]))
    assert seen["lfp_path"] == lfp_path
    assert seen["aligned_sync_npz_path"] == sync_path
    assert seen["saved_channel_index"] == 3
