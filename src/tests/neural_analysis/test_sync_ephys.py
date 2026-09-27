"""Entry-point tests that are distinct from the shared synchronization suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import ephys_sync_utils
from src.neural_analysis import sync_ephys


def test_map_spike_times_to_utc_maps_spike_sample_indices() -> None:
    """The SpikeGLX entry helper maps sample indices through IRIG anchors."""
    start_time_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    imec_irig_df = pd.DataFrame(
        {
            "sample_ix": np.array([0, 10], dtype=np.int64),
            "utc_unix": np.array(
                [start_time_utc.timestamp(), start_time_utc.timestamp() + 1.0],
                dtype=float,
            ),
        }
    )
    spike_sample_ix = np.array([3, 9], dtype=np.int64)

    spike_df = ephys_sync_utils.map_spike_times_to_utc(
        spike_sample_ix,
        imec_irig_df,
    )

    assert np.array_equal(
        spike_df["sample_ix"].to_numpy(dtype=np.int64),
        spike_sample_ix,
    )
    assert np.allclose(
        spike_df["utc_unix"].to_numpy(dtype=float),
        start_time_utc.timestamp() + spike_sample_ix / 10.0,
    )


def test_main_ni_only_prints_summary_and_returns_dataframes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The NI-only entry point returns decoder output and prints its summary."""
    ni_file = tmp_path / "run0_g0_t0.nidq.bin"
    ni_file.touch()
    pulse_df = pd.DataFrame(
        {
            "pulse_ix": np.array([0, 1], dtype=np.int64),
            "rising_sample_ix": np.array([0, 10], dtype=np.int64),
            "falling_sample_ix": np.array([2, 15], dtype=np.int64),
            "recording_time_s": np.array([0.0, 1.0], dtype=float),
            "pulse_width_samples": np.array([2.0, 5.0], dtype=float),
            "pulse_width_s": np.array([0.2, 0.5], dtype=float),
            "irig_bit": np.array([False, True], dtype=object),
            "utc_unix": np.array([100.0, 101.0], dtype=float),
            "utc_datetime": [
                datetime.fromtimestamp(100.0, tz=timezone.utc),
                datetime.fromtimestamp(101.0, tz=timezone.utc),
            ],
            "local_datetime": [
                datetime.fromtimestamp(100.0, tz=timezone.utc),
                datetime.fromtimestamp(101.0, tz=timezone.utc),
            ],
        }
    )
    frame_df = pd.DataFrame(
        {
            "frame_ix": np.array([0], dtype=np.int64),
            "frame_start_pulse_ix": np.array([0], dtype=np.int64),
            "frame_start_sample_ix": np.array([0], dtype=np.int64),
            "utc_unix": np.array([100.0], dtype=float),
            "utc_datetime": [datetime.fromtimestamp(100.0, tz=timezone.utc)],
            "local_datetime": [datetime.fromtimestamp(100.0, tz=timezone.utc)],
        }
    )

    def fake_decode_ni_irig_debug(
        ni_file: Path | str,
        digital_word: int = 0,
        irig_line: int = 0,
        bit_period_s: float = 1.0,
        utc_offset_hours: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return deterministic decoded tables without reading acquisition data."""
        return pulse_df, frame_df

    monkeypatch.setattr(
        ephys_sync_utils,
        "decode_ni_irig_debug",
        fake_decode_ni_irig_debug,
    )

    returned_pulse_df, returned_frame_df = sync_ephys.main_ni_only(
        ni_file=ni_file,
        digital_word=0,
        irig_line=0,
    )

    assert returned_pulse_df.equals(pulse_df)
    assert returned_frame_df.equals(frame_df)
    stdout = capsys.readouterr().out
    assert "Decoded 2 IRIG pulses" in stdout
    assert "Decoded 1 IRIG frames" in stdout
    assert "local_datetime" in stdout


def test_main_ni_only_raises_clear_error_for_missing_file(tmp_path: Path) -> None:
    """The NI-only entry point reports a missing acquisition file directly."""
    ni_file = tmp_path / "missing.nidq.bin"

    with pytest.raises(FileNotFoundError, match="Required input file not found"):
        sync_ephys.main_ni_only(ni_file=ni_file)
