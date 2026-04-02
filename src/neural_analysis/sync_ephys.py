"""Synchronize IMEC spikes and NI digital events to UTC using IRIG-H.

This module currently keeps the IRIG decoding logic local even though similar
code exists in ``preprocess_daq.py``. That module imports optional signal
processing dependencies that are not required for synchronization and are not
available in every environment. Once the synchronization path is stable, the
shared IRIG code should be moved into a dedicated helper module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

import src.external_tools.readSGLX as readSGLX
from src.irig_tools import irig_h_gpio as irig


def read_spikeglx_digital_line(
    binary_file: Path | str,
    digital_word: int,
    digital_line: int,
) -> tuple[np.ndarray, float]:
    """Read one SpikeGLX digital line from a binary file.

    Args:
        binary_file: Path to a SpikeGLX ``.bin`` file.
        digital_word: Digital word index used by ``ExtractDigital``.
        digital_line: Digital line index within ``digital_word``.

    Returns:
        tuple[np.ndarray, float]:
            - Boolean digital signal with shape ``(n_samples,)`` and units of
              logical high/low state.
            - Sampling rate in Hz as ``float``.
    """
    binary_file = Path(binary_file)
    meta = readSGLX.readMeta(binary_file)
    sample_rate_hz = float(readSGLX.SampRate(meta))
    n_channels = int(meta["nSavedChans"])
    n_samples = int(int(meta["fileSizeBytes"]) / (2 * n_channels))
    first_sample = 0
    last_sample = n_samples - 1

    raw_data = readSGLX.makeMemMapRaw(binary_file, meta)
    digital_array = readSGLX.ExtractDigital(
        raw_data,
        first_sample,
        last_sample,
        digital_word,
        [digital_line],
        meta,
    )
    digital_signal = np.asarray(np.squeeze(digital_array), dtype=bool)
    return digital_signal, sample_rate_hz


def find_signal_edges(binary_signal: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Find rising and falling edge sample indices for a digital signal.

    Args:
        binary_signal: Array-like digital signal with shape ``(n_samples,)`` and
            logical values where ``True`` denotes high state.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Rising edge sample indices with shape ``(n_rising,)`` in samples.
            - Falling edge sample indices with shape ``(n_falling,)`` in samples.
    """
    signal = np.asarray(binary_signal, dtype=bool).reshape(-1)
    if signal.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rising_ix = np.flatnonzero(~signal[:-1] & signal[1:]) + 1
    falling_ix = np.flatnonzero(signal[:-1] & ~signal[1:]) + 1

    if signal[0]:
        rising_ix = np.insert(rising_ix, 0, 0)

    return rising_ix.astype(np.int64), falling_ix.astype(np.int64)


def pulse_lengths_from_edges(
    rising_ix: npt.ArrayLike,
    falling_ix: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each rising edge to the next falling edge.

    Args:
        rising_ix: Rising edge sample indices with shape ``(n_rising,)`` in
            samples.
        falling_ix: Falling edge sample indices with shape ``(n_falling,)`` in
            samples.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Rising sample indices that have a following falling edge, shape
              ``(n_pulses,)`` in samples.
            - Pulse lengths with shape ``(n_pulses,)`` in samples.
    """
    rising_ix = np.asarray(rising_ix, dtype=np.int64).reshape(-1)
    falling_ix = np.asarray(falling_ix, dtype=np.int64).reshape(-1)

    if rising_ix.size == 0 or falling_ix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    next_falling_pos = np.searchsorted(falling_ix, rising_ix, side="right")
    valid_mask = next_falling_pos < falling_ix.size
    if not np.any(valid_mask):
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    paired_rising_ix = rising_ix[valid_mask]
    paired_falling_ix = falling_ix[next_falling_pos[valid_mask]]
    pulse_lengths_samples = (paired_falling_ix - paired_rising_ix).astype(float)
    return paired_rising_ix.astype(np.int64), pulse_lengths_samples


def classify_irig_h_pulses(
    pulse_lengths_samples: npt.ArrayLike,
    sample_rate_hz: float,
    bit_period_s: float = 1.0,
) -> np.ndarray:
    """Classify IRIG-H pulse widths into ``False``, ``True``, or ``'P'``.

    Args:
        pulse_lengths_samples: Pulse widths with shape ``(n_pulses,)`` in
            samples.
        sample_rate_hz: Sampling rate in Hz for converting samples to bit
            fractions.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        np.ndarray: Object array with shape ``(n_pulses,)`` containing
        ``False`` for 0-bits, ``True`` for 1-bits, ``'P'`` for position
        markers, and ``None`` for invalid widths.
    """
    pulse_lengths_samples = np.asarray(pulse_lengths_samples, dtype=float).reshape(-1)
    samples_per_bit = float(sample_rate_hz) * float(bit_period_s)
    p_thresh = 0.75 * samples_per_bit
    one_thresh = 0.45 * samples_per_bit
    zero_thresh = 0.05 * samples_per_bit

    bits = np.full(pulse_lengths_samples.shape, None, dtype=object)
    bits[pulse_lengths_samples > p_thresh] = "P"
    bits[(pulse_lengths_samples > one_thresh) & (pulse_lengths_samples <= p_thresh)] = True
    bits[(pulse_lengths_samples > zero_thresh) & (pulse_lengths_samples <= one_thresh)] = False
    return bits


def _irig_frame_to_utc_unix(frame_bits: list[object]) -> Optional[float]:
    """Decode one 60-bit IRIG-H frame into UTC unix time.

    Args:
        frame_bits: IRIG-H frame represented as a list of length ``60`` with bit
            values ``False``, ``True``, or ``'P'``.

    Returns:
        Optional[float]: UTC unix timestamp in seconds, or ``None`` if decoding
        fails.
    """
    decoded_datetime = irig.irig_h_to_datetime(frame_bits)
    if decoded_datetime is None:
        return None

    decoded_datetime_utc = decoded_datetime.replace(tzinfo=timezone.utc)
    return float(decoded_datetime_utc.timestamp())


def decode_irig_h_frame_anchors(irig_bits: npt.ArrayLike) -> list[tuple[int, float]]:
    """Decode IRIG-H frame anchors from classified bit values.

    Args:
        irig_bits: Object array with shape ``(n_bits,)`` containing IRIG-H bit
            labels ``False``, ``True``, or ``'P'``.

    Returns:
        list[tuple[int, float]]: ``(frame_start_bit_ix, frame_start_utc_unix)``
        pairs, where ``frame_start_bit_ix`` is in bit index units and
        ``frame_start_utc_unix`` is in seconds.
    """
    irig_bits = np.asarray(irig_bits, dtype=object).reshape(-1)
    if irig_bits.size < 122:
        return []

    tracking_start = None
    scan_max = min(120, irig_bits.size - 1)
    for bit_ix in range(scan_max):
        if irig_bits[bit_ix] == "P" and irig_bits[bit_ix + 1] == "P":
            tracking_start = bit_ix + 1
            break

    if tracking_start is None:
        return []

    anchors: list[tuple[int, float]] = []
    frame_bits: list[object] = []
    frame_ix: list[int] = []
    for bit_ix in range(tracking_start, irig_bits.size - 1):
        frame_bits.append(irig_bits[bit_ix])
        frame_ix.append(bit_ix)
        if irig_bits[bit_ix] == "P" and irig_bits[bit_ix + 1] == "P":
            if len(frame_bits) == 60:
                frame_unix = _irig_frame_to_utc_unix(frame_bits)
                if frame_unix is not None:
                    anchors.append((frame_ix[0], float(frame_unix)))
            frame_bits = []
            frame_ix = []

    return anchors


def assign_utc_to_irig_bits(
    n_bits: int,
    frame_anchors: list[tuple[int, float]],
) -> np.ndarray:
    """Assign UTC unix time to each IRIG bit index using decoded frame anchors.

    Args:
        n_bits: Number of IRIG bits in the sequence.
        frame_anchors: ``(frame_start_bit_ix, frame_start_utc_unix)`` pairs
            where bit indices are in bit units and timestamps are in seconds.

    Returns:
        np.ndarray: UTC unix time for each bit with shape ``(n_bits,)`` in
        seconds. Invalid or undecodable ranges are filled with ``NaN``.
    """
    unix_time = np.full(int(n_bits), np.nan, dtype=float)
    if n_bits == 0 or len(frame_anchors) == 0:
        return unix_time

    anchor_ix = np.asarray([anchor[0] for anchor in frame_anchors], dtype=np.int64)
    anchor_unix = np.asarray([anchor[1] for anchor in frame_anchors], dtype=float)
    keep_mask = np.concatenate(([True], np.diff(anchor_ix) > 0))
    anchor_ix = anchor_ix[keep_mask]
    anchor_unix = anchor_unix[keep_mask]
    if anchor_ix.size == 0:
        return unix_time

    first_anchor_ix = int(anchor_ix[0])
    unix_time[first_anchor_ix:] = anchor_unix[0] + np.arange(n_bits - first_anchor_ix, dtype=float)
    unix_time[:first_anchor_ix] = anchor_unix[0] - np.arange(first_anchor_ix, 0, -1, dtype=float)

    for anchor_position in range(1, anchor_ix.size):
        start_ix = int(anchor_ix[anchor_position])
        unix_time[start_ix:] = anchor_unix[anchor_position] + np.arange(n_bits - start_ix, dtype=float)

    return unix_time


def decode_sync_line_to_irig_utc(
    sync_signal: npt.ArrayLike,
    sample_rate_hz: float,
    bit_period_s: float = 1.0,
) -> pd.DataFrame:
    """Decode an IRIG-H digital sync line into UTC timestamps for rising edges.

    Args:
        sync_signal: Boolean or binary-valued sync signal with shape
            ``(n_samples,)``.
        sample_rate_hz: Sampling rate in Hz for ``sync_signal``.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        pd.DataFrame: One row per paired IRIG pulse onset with columns:
            - ``sample_ix``: pulse onset sample index, shape ``(n_pulses,)``,
              units samples
            - ``recording_time_s``: pulse onset time, units seconds
            - ``pulse_len_samples``: pulse width, units samples
            - ``irig_bit``: classified IRIG bit label
            - ``utc_unix``: decoded UTC unix time, units seconds
            - ``utc_datetime``: timezone-aware UTC datetime
    """
    rising_ix, falling_ix = find_signal_edges(sync_signal)
    paired_rising_ix, pulse_lengths_samples = pulse_lengths_from_edges(rising_ix, falling_ix)
    irig_bits = classify_irig_h_pulses(
        pulse_lengths_samples=pulse_lengths_samples,
        sample_rate_hz=sample_rate_hz,
        bit_period_s=bit_period_s,
    )

    valid_mask = np.asarray([bit is not None for bit in irig_bits], dtype=bool)
    valid_bits = irig_bits[valid_mask]
    frame_anchors = decode_irig_h_frame_anchors(valid_bits)
    valid_unix = assign_utc_to_irig_bits(valid_bits.size, frame_anchors)

    all_unix = np.full(irig_bits.shape[0], np.nan, dtype=float)
    all_unix[valid_mask] = valid_unix
    utc_datetime = [
        datetime.fromtimestamp(unix_time, tz=timezone.utc) if np.isfinite(unix_time) else pd.NaT
        for unix_time in all_unix
    ]

    return pd.DataFrame(
        {
            "sample_ix": paired_rising_ix.astype(np.int64),
            "recording_time_s": paired_rising_ix.astype(float) / float(sample_rate_hz),
            "pulse_len_samples": pulse_lengths_samples.astype(float),
            "irig_bit": irig_bits,
            "utc_unix": all_unix,
            "utc_datetime": utc_datetime,
        }
    )


def _interpolate_with_linear_extrapolation(
    query_x: npt.ArrayLike,
    known_x: npt.ArrayLike,
    known_y: npt.ArrayLike,
) -> np.ndarray:
    """Interpolate and linearly extrapolate a one-dimensional mapping.

    Args:
        query_x: Query positions with shape ``(n_query,)``.
        known_x: Known x positions with shape ``(n_known,)``.
        known_y: Known y values with shape ``(n_known,)``.

    Returns:
        np.ndarray: Interpolated values with shape ``(n_query,)``.
    """
    query_x = np.asarray(query_x, dtype=float).reshape(-1)
    known_x = np.asarray(known_x, dtype=float).reshape(-1)
    known_y = np.asarray(known_y, dtype=float).reshape(-1)

    interpolated_y = np.interp(query_x, known_x, known_y)
    if known_x.size < 2:
        return interpolated_y

    left_mask = query_x < known_x[0]
    right_mask = query_x > known_x[-1]

    left_dx = float(known_x[1] - known_x[0])
    right_dx = float(known_x[-1] - known_x[-2])
    left_slope = (known_y[1] - known_y[0]) / left_dx if left_dx != 0 else 0.0
    right_slope = (known_y[-1] - known_y[-2]) / right_dx if right_dx != 0 else 0.0

    interpolated_y[left_mask] = known_y[0] + (query_x[left_mask] - known_x[0]) * left_slope
    interpolated_y[right_mask] = known_y[-1] + (query_x[right_mask] - known_x[-1]) * right_slope
    return interpolated_y


def map_sample_indices_to_utc(
    sample_ix: npt.ArrayLike,
    irig_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map arbitrary sample indices to UTC using IRIG-derived anchors.

    Args:
        sample_ix: Sample indices with shape ``(n_samples,)`` in samples.
        irig_df: DataFrame from :func:`decode_sync_line_to_irig_utc` containing
            ``sample_ix`` in samples and ``utc_unix`` in seconds.

    Returns:
        pd.DataFrame: UTC mapping with columns:
            - ``sample_ix``: queried sample indices, units samples
            - ``utc_unix``: mapped UTC unix times, units seconds
            - ``utc_datetime``: timezone-aware UTC datetime
    """
    sample_ix = np.asarray(sample_ix, dtype=np.int64).reshape(-1)
    known = irig_df.loc[np.isfinite(irig_df["utc_unix"]), ["sample_ix", "utc_unix"]].copy()
    known = known.drop_duplicates(subset="sample_ix").sort_values("sample_ix")
    if known.shape[0] < 2:
        raise ValueError("Need at least two finite IRIG UTC points to map sample indices.")

    utc_unix = _interpolate_with_linear_extrapolation(
        query_x=sample_ix.astype(float),
        known_x=known["sample_ix"].to_numpy(dtype=float),
        known_y=known["utc_unix"].to_numpy(dtype=float),
    )
    utc_datetime = [datetime.fromtimestamp(unix_time, tz=timezone.utc) for unix_time in utc_unix]
    return pd.DataFrame(
        {
            "sample_ix": sample_ix.astype(np.int64),
            "utc_unix": utc_unix.astype(float),
            "utc_datetime": utc_datetime,
        }
    )


def map_daq_rising_edges_to_utc(
    digital_signal: npt.ArrayLike,
    sample_rate_hz: float,
    irig_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map DAQ digital rising edges to UTC.

    Args:
        digital_signal: Digital event signal with shape ``(n_samples,)`` and
            logical high/low values.
        sample_rate_hz: Sampling rate in Hz for ``digital_signal``.
        irig_df: DataFrame from :func:`decode_sync_line_to_irig_utc` for the
            same stream.

    Returns:
        pd.DataFrame: One row per rising edge with columns:
            - ``sample_ix``: rising edge sample index, units samples
            - ``recording_time_s``: rising edge time, units seconds
            - ``utc_unix``: mapped UTC unix time, units seconds
            - ``utc_datetime``: timezone-aware UTC datetime
    """
    rising_ix, _ = find_signal_edges(digital_signal)
    mapped_df = map_sample_indices_to_utc(rising_ix, irig_df)
    mapped_df.insert(1, "recording_time_s", rising_ix.astype(float) / float(sample_rate_hz))
    return mapped_df


def map_spike_times_to_utc(
    spike_sample_ix: npt.ArrayLike,
    imec_irig_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map IMEC spike sample indices to UTC.

    Args:
        spike_sample_ix: Spike sample indices from ``spike_times.npy`` with
            shape ``(n_spikes,)`` in IMEC AP samples.
        imec_irig_df: DataFrame from :func:`decode_sync_line_to_irig_utc` for
            the corresponding IMEC stream.

    Returns:
        pd.DataFrame: UTC mapping with columns:
            - ``sample_ix``: spike sample index, units samples
            - ``utc_unix``: mapped UTC unix time, units seconds
            - ``utc_datetime``: timezone-aware UTC datetime
    """
    return map_sample_indices_to_utc(spike_sample_ix, imec_irig_df)


def save_stream_sync_npz(
    output_file: Path | str,
    arrays: dict[str, npt.ArrayLike],
    meta: dict[str, Any],
) -> Path:
    """Save synchronization outputs to a ``.npz`` file with named arrays.

    Args:
        output_file: Destination ``.npz`` path.
        arrays: Named arrays to save. Each value must be array-like.
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


def decode_binary_file_irig_utc(
    binary_file: Path | str,
    digital_word: int,
    irig_line: int,
    bit_period_s: float = 1.0,
) -> tuple[pd.DataFrame, float]:
    """Decode IRIG-H UTC timestamps directly from one SpikeGLX digital line.

    Args:
        binary_file: Path to a SpikeGLX ``.bin`` file.
        digital_word: Digital word index used by ``ExtractDigital``.
        irig_line: Digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        tuple[pd.DataFrame, float]:
            - IRIG rising-edge dataframe from
              :func:`decode_sync_line_to_irig_utc`.
            - Sampling rate in Hz.
    """
    irig_signal, sample_rate_hz = read_spikeglx_digital_line(binary_file, digital_word, irig_line)
    irig_df = decode_sync_line_to_irig_utc(
        sync_signal=irig_signal,
        sample_rate_hz=sample_rate_hz,
        bit_period_s=bit_period_s,
    )
    return irig_df, sample_rate_hz


def sync_imec_spikes_to_utc(
    imec_ap_file: Path | str,
    spike_times_npy: Path | str,
    output_file: Optional[Path | str] = None,
    digital_word: int = 0,
    irig_line: int = 6,
    bit_period_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode IMEC IRIG and assign UTC timestamps to spikes.

    Args:
        imec_ap_file: IMEC AP ``.bin`` file path.
        spike_times_npy: ``spike_times.npy`` path containing a 1D array of IMEC
            AP sample indices with shape ``(n_spikes,)`` in samples.
        output_file: Optional ``.npz`` output path for this IMEC stream.
        digital_word: IMEC digital word index.
        irig_line: IMEC digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Spike UTC dataframe with spike sample indices and UTC timestamps.
            - IMEC IRIG rising-edge dataframe.
    """
    imec_irig_df, sample_rate_hz = decode_binary_file_irig_utc(
        binary_file=imec_ap_file,
        digital_word=digital_word,
        irig_line=irig_line,
        bit_period_s=bit_period_s,
    )
    spike_sample_ix = np.asarray(np.load(Path(spike_times_npy), allow_pickle=True), dtype=np.int64).reshape(-1)
    spike_df = map_spike_times_to_utc(spike_sample_ix, imec_irig_df)

    if output_file is not None:
        meta = {
            "generator": "sync_imec_spikes_to_utc",
            "analysis_version": "0.1.0",
            "source_file": str(imec_ap_file),
            "spike_times_file": str(spike_times_npy),
            "sample_rate_hz": float(sample_rate_hz),
            "digital_word": int(digital_word),
            "irig_line": int(irig_line),
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode NI IRIG and assign UTC timestamps to one NI event line.

    Args:
        ni_file: NI ``.nidq.bin`` file path.
        event_line: NI digital line index for the event of interest.
        output_file: Optional ``.npz`` output path for this NI stream.
        digital_word: NI digital word index.
        irig_line: NI digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.

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
    )
    event_signal, event_sample_rate_hz = read_spikeglx_digital_line(
        binary_file=ni_file,
        digital_word=digital_word,
        digital_line=event_line,
    )
    if not np.isclose(event_sample_rate_hz, sample_rate_hz):
        raise ValueError("IRIG line and event line sample rates do not match.")

    rising_df = map_daq_rising_edges_to_utc(
        digital_signal=event_signal,
        sample_rate_hz=sample_rate_hz,
        irig_df=ni_irig_df,
    )

    if output_file is not None:
        meta = {
            "generator": "sync_ni_rising_edges_to_utc",
            "analysis_version": "0.1.0",
            "source_file": str(ni_file),
            "sample_rate_hz": float(sample_rate_hz),
            "digital_word": int(digital_word),
            "irig_line": int(irig_line),
            "event_line": int(event_line),
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
def _require_existing_file(file_path: Path) -> Path:
    """Validate that an expected input file exists.

    Args:
        file_path: File path expected to exist on disk.

    Returns:
        Path: The same path after validation.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Required input file not found: {file_path}")
    return file_path


def main(
    session_data_home: Path | str = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference"),
    sess_id_full: str = "CT014_2025-12-16_153200",
    sorting_output_name: str = "Kilosort2.5.2_2026-03-18_115111",
    ni_event_lines: tuple[int, ...] = (2, 3),
    output_root: Optional[Path | str] = None,
) -> None:
    """Run a brief example synchronization workflow for one IMEC stream and NI lines.

    Args:
        session_data_home: Session root directory containing ``ephys/raw`` and
            ``ephys/catgt`` subdirectories.
        sess_id_full: Session identifier string for printed summaries.
        sorting_output_name: Name of the sorting output directory containing
            ``spike_times.npy``.
        ni_event_lines: NI digital line indices to map to UTC from NI word 0.
        output_root: Root output directory for per-stream ``.npz`` files. If
            ``None``, defaults to ``session_data_home / 'ephys' / 'aligned'``.

    Returns:
        None: This example runner writes files and prints a short summary.
    """
    session_data_home = Path(session_data_home)
    output_root = Path(output_root) if output_root is not None else session_data_home / "ephys" / "aligned"

    raw_ephys_folder = session_data_home / "ephys" / "raw" / "run0_g0"
    catgt_ephys_folder = session_data_home / "ephys" / "catgt" / "catgt_run0_g0"
    imec0_folder = catgt_ephys_folder / "run0_g0_imec0"
    sorting_output = imec0_folder / sorting_output_name

    ni_file = _require_existing_file(raw_ephys_folder / "run0_g0_t0.nidq.bin")
    ap_file = _require_existing_file(imec0_folder / "run0_g0_tcat.imec0.ap.bin")
    spike_times_file = _require_existing_file(sorting_output / "spike_times.npy")

    imec_output_dir = output_root / "aligned_imec0"
    nidaq_output_dir = output_root / "aligned_nidaq"

    spike_df, imec_irig_df = sync_imec_spikes_to_utc(
        imec_ap_file=ap_file,
        spike_times_npy=spike_times_file,
        output_file=imec_output_dir / "imec0_sync.npz",
        digital_word=0,
        irig_line=6,
    )
    print(
        f"{sess_id_full} imec0: mapped {spike_df.shape[0]} spikes using "
        f"{imec_irig_df.shape[0]} IRIG rising edges."
    )

    for event_line in ni_event_lines:
        rising_df, ni_irig_df = sync_ni_rising_edges_to_utc(
            ni_file=ni_file,
            event_line=int(event_line),
            output_file=nidaq_output_dir / f"line{int(event_line)}_sync.npz",
            digital_word=0,
            irig_line=0,
        )
        print(
            f"{sess_id_full} nidaq line {int(event_line)}: mapped {rising_df.shape[0]} rising edges using "
            f"{ni_irig_df.shape[0]} IRIG rising edges."
        )


if __name__ == "__main__":
    main()
