"""Reusable IRIG-H synchronization helpers for neural-analysis workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.irig_tools import irig_h_gpio as irig


def find_signal_edges(binary_signal: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Find rising and falling edge sample indices for a digital signal.

    Args:
        binary_signal: Array-like digital signal with shape ``(n_samples,)`` and
            logical values where ``True`` denotes the high state.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - Rising edge sample indices with shape ``(n_rising,)`` in samples.
            - Falling edge sample indices with shape ``(n_falling,)`` in
              samples.
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
    """Pair each rising edge with the next falling edge.

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
    """Classify IRIG-H pulse widths into ``False``, ``True``, ``'P'``, or ``None``.

    Args:
        pulse_lengths_samples: Pulse widths with shape ``(n_pulses,)`` in
            samples.
        sample_rate_hz: Sampling rate in Hz.
        bit_period_s: IRIG-H bit period in seconds.

    Returns:
        np.ndarray: Object array with shape ``(n_pulses,)``.
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
        frame_bits: IRIG-H frame bits with length ``60``.

    Returns:
        Optional[float]: UTC unix time in seconds, or ``None`` if decoding
        fails.
    """
    decoded_datetime = irig.irig_h_to_datetime(frame_bits)
    if decoded_datetime is None:
        return None
    return float(decoded_datetime.replace(tzinfo=timezone.utc).timestamp())


def decode_irig_h_frame_anchors(irig_bits: npt.ArrayLike) -> list[tuple[int, float]]:
    """Decode IRIG-H frame anchor positions from classified bit values.

    Args:
        irig_bits: Object array with shape ``(n_bits,)`` containing ``False``,
            ``True``, ``'P'``, or ``None``.

    Returns:
        list[tuple[int, float]]: ``(frame_start_bit_ix, frame_start_utc_unix)``
        pairs, where bit indices are in bit units and UTC values are in
        seconds.
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
    """Assign UTC unix time to each IRIG bit index.

    Args:
        n_bits: Number of IRIG bits.
        frame_anchors: ``(frame_start_bit_ix, frame_start_utc_unix)`` pairs.

    Returns:
        np.ndarray: UTC unix time for each bit with shape ``(n_bits,)`` in
        seconds. Missing values are ``NaN``.
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
    align_first_rising_edge_unix: Optional[float] = None,
) -> pd.DataFrame:
    """Decode an IRIG-H sync line into UTC timestamps for pulse onsets.

    Args:
        sync_signal: Boolean or binary-valued sync signal with shape
            ``(n_samples,)``.
        sample_rate_hz: Sampling rate in Hz for ``sync_signal``.
        bit_period_s: IRIG-H bit period in seconds.
        align_first_rising_edge_unix: Optional UTC unix time in seconds used to
            shift the decoded sequence so the first finite rising edge matches a
            known timestamp.

    Returns:
        pd.DataFrame: One row per paired IRIG pulse onset with columns
            ``sample_ix``, ``recording_time_s``, ``pulse_len_samples``,
            ``irig_bit``, ``utc_unix``, and ``utc_datetime``.
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
    if align_first_rising_edge_unix is not None and np.isfinite(align_first_rising_edge_unix):
        finite_ix = np.where(np.isfinite(all_unix))[0]
        if finite_ix.size > 0:
            shift_s = float(align_first_rising_edge_unix) - float(all_unix[finite_ix[0]])
            all_unix = all_unix + shift_s

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


def interpolate_with_linear_extrapolation(
    query_x: npt.ArrayLike,
    known_x: npt.ArrayLike,
    known_y: npt.ArrayLike,
) -> np.ndarray:
    """Interpolate and linearly extrapolate a one-dimensional mapping.

    Args:
        query_x: Query x values with shape ``(n_query,)``.
        known_x: Known x values with shape ``(n_known,)``.
        known_y: Known y values with shape ``(n_known,)``.

    Returns:
        np.ndarray: Interpolated/extrapolated values with shape ``(n_query,)``.
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
    """Map sample indices to UTC using IRIG-derived anchors.

    Args:
        sample_ix: Sample indices with shape ``(n_samples,)`` in samples.
        irig_df: DataFrame from :func:`decode_sync_line_to_irig_utc`.

    Returns:
        pd.DataFrame: Columns ``sample_ix``, ``utc_unix``, and
        ``utc_datetime``.
    """
    sample_ix = np.asarray(sample_ix, dtype=np.int64).reshape(-1)
    known = irig_df.loc[np.isfinite(irig_df["utc_unix"]), ["sample_ix", "utc_unix"]].copy()
    known = known.drop_duplicates(subset="sample_ix").sort_values("sample_ix")
    if known.shape[0] < 2:
        raise ValueError("Need at least two finite IRIG UTC points to map sample indices.")

    utc_unix = interpolate_with_linear_extrapolation(
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


def map_digital_rising_edges_to_utc(
    digital_signal: npt.ArrayLike,
    sample_rate_hz: float,
    irig_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map digital rising edges to UTC using IRIG-derived anchors.

    Args:
        digital_signal: Digital event signal with shape ``(n_samples,)``.
        sample_rate_hz: Sampling rate in Hz for ``digital_signal``.
        irig_df: DataFrame from :func:`decode_sync_line_to_irig_utc`.

    Returns:
        pd.DataFrame: One row per rising edge with columns ``sample_ix``,
        ``recording_time_s``, ``utc_unix``, and ``utc_datetime``.
    """
    rising_ix, _ = find_signal_edges(digital_signal)
    mapped_df = map_sample_indices_to_utc(rising_ix, irig_df)
    mapped_df.insert(1, "recording_time_s", rising_ix.astype(float) / float(sample_rate_hz))
    return mapped_df
