"""SpikeGLX/ephys synchronization helpers built on IRIG-derived UTC mapping."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.irig_tools.irig_sync_utils import (
    assign_utc_to_irig_bits,
    classify_irig_h_pulses,
    decode_irig_h_frame_anchors,
    decode_sync_line_to_irig_utc,
    find_signal_edges,
    map_digital_rising_edges_to_utc,
    map_sample_indices_to_utc,
    pulse_lengths_from_edges,
)
from src.neural_analysis.spikeglx_sync_io import read_digital_line as read_spikeglx_digital_line


NEW_YORK_TZ = ZoneInfo("America/New_York")
ReadDigitalLineFn = Callable[[Path | str, int, int], tuple[np.ndarray, float]]
DecodeSyncLineFn = Callable[..., pd.DataFrame]


def save_stream_sync_npz(
    output_file: Path | str,
    arrays: dict[str, npt.ArrayLike],
    meta: dict[str, Any],
) -> Path:
    """Save synchronization outputs to a ``.npz`` file with named arrays.

    Args:
        output_file: Destination ``.npz`` path.
        arrays: Named arrays to save. Each value must be array-like with its
            own documented units in ``meta``.
        meta: Metadata dictionary describing generator, parameters, units, and
            axis conventions.

    Returns:
        Path: Saved file path.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    savez_payload: dict[str, Any] = {name: np.asarray(value) for name, value in arrays.items()}
    savez_payload["meta"] = np.asarray(meta, dtype=object)
    np.savez(output_path, **savez_payload)
    return output_path


def apply_utc_hour_offset(irig_df: pd.DataFrame, utc_offset_hours: float) -> pd.DataFrame:
    """Shift decoded UTC timestamps by a user-specified number of hours.

    Args:
        irig_df: IRIG dataframe with shape ``(n_pulses, n_columns)``. Required
            columns are ``utc_unix`` in seconds and ``utc_datetime`` as
            timezone-aware UTC datetimes or ``NaT`` values.
        utc_offset_hours: Constant offset added to all decoded timestamps, in
            hours.

    Returns:
        pd.DataFrame: Copy of ``irig_df`` with ``utc_unix`` shifted by
        ``utc_offset_hours * 3600`` seconds and ``utc_datetime`` rebuilt from
        the shifted UTC Unix values.
    """
    shifted_df = irig_df.copy(deep=True)
    shift_seconds = float(utc_offset_hours) * 3600.0
    shifted_unix = shifted_df["utc_unix"].to_numpy(dtype=float) + shift_seconds
    shifted_df["utc_unix"] = shifted_unix
    shifted_df["utc_datetime"] = [
        datetime.fromtimestamp(unix_time, tz=timezone.utc) if np.isfinite(unix_time) else pd.NaT
        for unix_time in shifted_unix
    ]
    return shifted_df


def write_alignment_note(output_root: Path | str, utc_offset_hours: float) -> Path:
    """Write a note recording the UTC hour adjustment used for a run.

    Args:
        output_root: Root aligned-output directory where the note should be
            written.
        utc_offset_hours: Constant offset added to decoded timestamps, in
            hours.

    Returns:
        Path: Path to the text note written inside ``output_root``.
    """
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    note_path = output_path / "time_adjustment_note.txt"
    note_lines = [
        f"UTC hour adjustment: {float(utc_offset_hours)}",
        f"Processed at: {datetime.now().astimezone().isoformat()}",
    ]
    note_path.write_text("\n".join(note_lines) + "\n")
    return note_path


def convert_utc_series_to_new_york(
    localizable_times: list[datetime | pd.Timestamp | Any],
) -> list[datetime | pd.NaT]:
    """Convert UTC datetimes into New York local datetimes.

    Args:
        localizable_times: Sequence with shape ``(n_times,)`` containing
            timezone-aware UTC datetimes or ``NaT``-like values.

    Returns:
        list[datetime | pd.NaT]: Sequence with the same length where finite UTC
        datetimes are converted to ``America/New_York`` and missing values
        remain ``pd.NaT``.
    """
    local_times: list[datetime | pd.NaT] = []
    for utc_time in localizable_times:
        if pd.isna(utc_time):
            local_times.append(pd.NaT)
        else:
            local_times.append(pd.Timestamp(utc_time).tz_convert(NEW_YORK_TZ).to_pydatetime())
    return local_times


def decode_binary_file_irig_utc(
    binary_file: Path | str,
    digital_word: int,
    irig_line: int,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
    read_digital_line_fn: ReadDigitalLineFn = read_spikeglx_digital_line,
    decode_sync_line_to_irig_utc_fn: DecodeSyncLineFn = decode_sync_line_to_irig_utc,
) -> tuple[pd.DataFrame, float]:
    """Decode IRIG-H UTC timestamps from one SpikeGLX digital line.

    Args:
        binary_file: Path to a SpikeGLX ``.bin`` file.
        digital_word: Digital word index used by ``ExtractDigital``.
        irig_line: Digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours. This is retained for lab setups where the IRIG/recording
            clock relationship is not fully reliable.
        read_digital_line_fn: Function that reads one digital line and returns
            ``(signal, sample_rate_hz)``.
        decode_sync_line_to_irig_utc_fn: Function that converts a digital IRIG
            line into a pulse-level UTC dataframe.

    Returns:
        tuple[pd.DataFrame, float]:
            - IRIG rising-edge dataframe with UTC columns.
            - Sampling rate in Hz.
    """
    irig_signal, sample_rate_hz = read_digital_line_fn(
        binary_file=binary_file,
        digital_word=digital_word,
        digital_line=irig_line,
    )
    irig_df = decode_sync_line_to_irig_utc_fn(
        sync_signal=irig_signal,
        sample_rate_hz=sample_rate_hz,
        bit_period_s=bit_period_s,
    )
    irig_df = apply_utc_hour_offset(irig_df, utc_offset_hours=utc_offset_hours)
    return irig_df, sample_rate_hz


def map_spike_times_to_utc(
    spike_sample_ix: npt.ArrayLike,
    imec_irig_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map IMEC spike sample indices to UTC.

    Args:
        spike_sample_ix: Spike sample indices from ``spike_times.npy`` with
            shape ``(n_spikes,)`` in IMEC AP samples.
        imec_irig_df: DataFrame from ``decode_sync_line_to_irig_utc`` for the
            corresponding IMEC stream.

    Returns:
        pd.DataFrame: Columns ``sample_ix`` in samples, ``utc_unix`` in
        seconds, and ``utc_datetime`` as timezone-aware UTC datetimes.
    """
    return map_sample_indices_to_utc(spike_sample_ix, imec_irig_df)


def sync_imec_spikes_to_utc(
    imec_ap_file: Path | str,
    spike_times_npy: Path | str,
    output_file: Optional[Path | str] = None,
    digital_word: int = 0,
    irig_line: int = 6,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
    read_digital_line_fn: ReadDigitalLineFn = read_spikeglx_digital_line,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode IMEC IRIG and assign UTC timestamps to spikes.

    Args:
        imec_ap_file: IMEC AP ``.bin`` file path.
        spike_times_npy: ``spike_times.npy`` path containing shape
            ``(n_spikes,)`` IMEC AP sample indices.
        output_file: Optional ``.npz`` output path for this IMEC stream.
        digital_word: IMEC digital word index.
        irig_line: IMEC digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.
        read_digital_line_fn: Function that reads one digital line and returns
            ``(signal, sample_rate_hz)``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Spike UTC dataframe.
            - IMEC IRIG rising-edge dataframe.
    """
    imec_irig_df, sample_rate_hz = decode_binary_file_irig_utc(
        binary_file=imec_ap_file,
        digital_word=digital_word,
        irig_line=irig_line,
        bit_period_s=bit_period_s,
        utc_offset_hours=utc_offset_hours,
        read_digital_line_fn=read_digital_line_fn,
    )
    spike_sample_ix = np.asarray(np.load(Path(spike_times_npy), allow_pickle=True), dtype=np.int64).reshape(-1)
    spike_df = map_spike_times_to_utc(spike_sample_ix, imec_irig_df)

    if output_file is not None:
        meta = {
            "generator": "sync_imec_spikes_to_utc",
            "analysis_version": "0.2.0",
            "source_file": str(imec_ap_file),
            "spike_times_file": str(spike_times_npy),
            "sample_rate_hz": float(sample_rate_hz),
            "digital_word": int(digital_word),
            "irig_line": int(irig_line),
            "utc_offset_hours": float(utc_offset_hours),
            "units": {
                "sample_ix": "samples",
                "recording_time_s": "seconds",
                "pulse_len_samples": "samples",
                "utc_unix": "seconds",
            },
            "axis_convention": "1D sample index along acquisition time",
        }
        save_stream_sync_npz(
            output_file=output_file,
            arrays={
                "spike_sample_ix": spike_df["sample_ix"].to_numpy(dtype=np.int64),
                "spike_utc_unix": spike_df["utc_unix"].to_numpy(dtype=float),
                "irig_sample_ix": imec_irig_df["sample_ix"].to_numpy(dtype=np.int64),
                "irig_recording_time_s": imec_irig_df["recording_time_s"].to_numpy(dtype=float),
                "irig_pulse_len_samples": imec_irig_df["pulse_len_samples"].to_numpy(dtype=float),
                "irig_bit": imec_irig_df["irig_bit"].to_numpy(dtype=object),
                "irig_utc_unix": imec_irig_df["utc_unix"].to_numpy(dtype=float),
            },
            meta=meta,
        )

    return spike_df, imec_irig_df


def sync_ni_rising_edges_to_utc(
    ni_file: Path | str,
    event_line: int,
    output_file: Optional[Path | str] = None,
    digital_word: int = 0,
    irig_line: int = 0,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
    read_digital_line_fn: ReadDigitalLineFn = read_spikeglx_digital_line,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode NI IRIG and assign UTC timestamps to one NI event line.

    Args:
        ni_file: NI ``.nidq.bin`` file path.
        event_line: NI digital line index for the event of interest.
        output_file: Optional ``.npz`` output path for this NI stream.
        digital_word: NI digital word index.
        irig_line: NI digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.
        read_digital_line_fn: Function that reads one digital line and returns
            ``(signal, sample_rate_hz)``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Rising-edge UTC dataframe for ``event_line``.
            - NI IRIG rising-edge dataframe.
    """
    ni_irig_df, sample_rate_hz = decode_binary_file_irig_utc(
        binary_file=ni_file,
        digital_word=digital_word,
        irig_line=irig_line,
        bit_period_s=bit_period_s,
        utc_offset_hours=utc_offset_hours,
        read_digital_line_fn=read_digital_line_fn,
    )
    event_signal, event_sample_rate_hz = read_digital_line_fn(
        binary_file=ni_file,
        digital_word=digital_word,
        digital_line=event_line,
    )
    if not np.isclose(event_sample_rate_hz, sample_rate_hz):
        raise ValueError("IRIG line and event line sample rates do not match.")

    rising_df = map_digital_rising_edges_to_utc(
        digital_signal=event_signal,
        sample_rate_hz=sample_rate_hz,
        irig_df=ni_irig_df,
    )

    if output_file is not None:
        meta = {
            "generator": "sync_ni_rising_edges_to_utc",
            "analysis_version": "0.2.0",
            "source_file": str(ni_file),
            "sample_rate_hz": float(sample_rate_hz),
            "digital_word": int(digital_word),
            "irig_line": int(irig_line),
            "event_line": int(event_line),
            "utc_offset_hours": float(utc_offset_hours),
            "units": {
                "sample_ix": "samples",
                "recording_time_s": "seconds",
                "pulse_len_samples": "samples",
                "utc_unix": "seconds",
            },
            "axis_convention": "1D sample index along acquisition time",
        }
        save_stream_sync_npz(
            output_file=output_file,
            arrays={
                "event_sample_ix": rising_df["sample_ix"].to_numpy(dtype=np.int64),
                "event_utc_unix": rising_df["utc_unix"].to_numpy(dtype=float),
                "irig_sample_ix": ni_irig_df["sample_ix"].to_numpy(dtype=np.int64),
                "irig_recording_time_s": ni_irig_df["recording_time_s"].to_numpy(dtype=float),
                "irig_pulse_len_samples": ni_irig_df["pulse_len_samples"].to_numpy(dtype=float),
                "irig_bit": ni_irig_df["irig_bit"].to_numpy(dtype=object),
                "irig_utc_unix": ni_irig_df["utc_unix"].to_numpy(dtype=float),
            },
            meta=meta,
        )

    return rising_df, ni_irig_df


def decode_ni_irig_debug(
    ni_file: Path | str,
    digital_word: int = 0,
    irig_line: int = 0,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
    read_digital_line_fn: ReadDigitalLineFn = read_spikeglx_digital_line,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode one NI IRIG line into pulse- and frame-level debug tables.

    Args:
        ni_file: NI ``.nidq.bin`` file path. This function does not check file
            existence; entry-point workflows should validate paths before
            calling it.
        digital_word: NI digital word index.
        irig_line: NI digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.
        read_digital_line_fn: Function that reads one digital line and returns
            ``(signal, sample_rate_hz)``.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Pulse dataframe with one row per paired IRIG pulse onset.
            - Frame dataframe with one row per decoded IRIG frame start.
    """
    irig_signal, sample_rate_hz = read_digital_line_fn(
        binary_file=ni_file,
        digital_word=digital_word,
        digital_line=irig_line,
    )
    rising_ix, falling_ix = find_signal_edges(irig_signal)
    paired_rising_ix, pulse_lengths_samples = pulse_lengths_from_edges(rising_ix, falling_ix)
    paired_falling_ix = paired_rising_ix + pulse_lengths_samples.astype(np.int64)
    irig_bits = classify_irig_h_pulses(
        pulse_lengths_samples=pulse_lengths_samples,
        sample_rate_hz=sample_rate_hz,
        bit_period_s=bit_period_s,
    )

    valid_mask = np.asarray([bit is not None for bit in irig_bits], dtype=bool)
    valid_bits = irig_bits[valid_mask]
    valid_pulse_ix = np.flatnonzero(valid_mask).astype(np.int64)
    frame_anchors = decode_irig_h_frame_anchors(valid_bits)
    valid_unix = assign_utc_to_irig_bits(valid_bits.size, frame_anchors)

    all_unix = np.full(irig_bits.shape[0], np.nan, dtype=float)
    all_unix[valid_mask] = valid_unix
    if utc_offset_hours != 0.0:
        all_unix = all_unix + float(utc_offset_hours) * 3600.0

    pulse_df = pd.DataFrame(
        {
            "pulse_ix": np.arange(paired_rising_ix.size, dtype=np.int64),
            "rising_sample_ix": paired_rising_ix.astype(np.int64),
            "falling_sample_ix": paired_falling_ix.astype(np.int64),
            "recording_time_s": paired_rising_ix.astype(float) / float(sample_rate_hz),
            "pulse_width_samples": pulse_lengths_samples.astype(float),
            "pulse_width_s": pulse_lengths_samples.astype(float) / float(sample_rate_hz),
            "irig_bit": irig_bits,
            "utc_unix": all_unix,
            "utc_datetime": [
                datetime.fromtimestamp(unix_time, tz=timezone.utc) if np.isfinite(unix_time) else pd.NaT
                for unix_time in all_unix
            ],
        }
    )
    pulse_df["local_datetime"] = convert_utc_series_to_new_york(pulse_df["utc_datetime"].tolist())

    frame_rows: list[dict[str, object]] = []
    for frame_ix, (frame_start_valid_ix, frame_unix) in enumerate(frame_anchors):
        frame_start_pulse_ix = int(valid_pulse_ix[frame_start_valid_ix])
        if utc_offset_hours != 0.0:
            frame_unix = float(frame_unix) + float(utc_offset_hours) * 3600.0
        frame_rows.append(
            {
                "frame_ix": int(frame_ix),
                "frame_start_pulse_ix": frame_start_pulse_ix,
                "frame_start_sample_ix": int(paired_rising_ix[frame_start_pulse_ix]),
                "utc_unix": float(frame_unix),
                "utc_datetime": datetime.fromtimestamp(float(frame_unix), tz=timezone.utc),
            }
        )
    frame_df = pd.DataFrame(frame_rows)
    if not frame_df.empty:
        frame_df["local_datetime"] = convert_utc_series_to_new_york(frame_df["utc_datetime"].tolist())
    else:
        frame_df["local_datetime"] = pd.Series(dtype=object)

    return pulse_df, frame_df
