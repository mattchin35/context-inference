"""Reusable IRIG-H synchronization helpers for sampled signals and event times."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.irig_tools import irig_core


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
    return irig_core.find_signal_edges(binary_signal)


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
    return irig_core.pulse_lengths_from_edges(rising_ix, falling_ix)


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
    return irig_core.classify_pulse_widths_samples(
        pulse_lengths_samples=pulse_lengths_samples,
        sample_rate_hz=sample_rate_hz,
        bit_period_s=bit_period_s,
    )


def _irig_frame_to_utc_unix(
    frame_bits: list[object],
    irig_format: irig_core.IRIGFormat = "neurokairos",
) -> Optional[float]:
    """Decode one 60-bit IRIG-H frame into UTC unix time.

    Args:
        frame_bits: IRIG-H frame bits with length ``60``.
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.

    Returns:
        Optional[float]: UTC unix time in seconds, or ``None`` if decoding
        fails.
    """
    try:
        frame_fields = irig_core.decode_irig_frame(frame_bits, irig_format=irig_format)
        decoded_unix = float(frame_fields["decoded_utc_seconds"])
        if np.isfinite(decoded_unix):
            return decoded_unix
        return None
    except ValueError:
        return None


def decode_irig_h_frame_anchors(
    irig_bits: npt.ArrayLike,
    irig_format: irig_core.IRIGFormat = "neurokairos",
) -> list[tuple[int, float]]:
    """Decode IRIG-H frame anchor positions from classified bit values.

    Args:
        irig_bits: Object array with shape ``(n_bits,)`` containing ``False``,
            ``True``, ``'P'``, or ``None``.
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.

    Returns:
        list[tuple[int, float]]: ``(frame_start_bit_ix, frame_start_utc_unix)``
        pairs, where bit indices are in bit units and UTC values are in
        seconds.
    """
    return irig_core.decode_irig_frame_anchors(
        irig_bits=irig_bits,
        irig_format=irig_format,
    )


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
    return irig_core.assign_utc_to_irig_bits(n_bits=n_bits, frame_anchors=frame_anchors)


def decode_sync_line_to_irig_utc(
    sync_signal: npt.ArrayLike,
    sample_rate_hz: float,
    bit_period_s: float = 1.0,
    align_first_rising_edge_unix: Optional[float] = None,
    irig_format: irig_core.IRIGFormat = "neurokairos",
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
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.

    Returns:
        pd.DataFrame: One row per paired IRIG pulse onset with columns
            ``sample_ix``, ``recording_time_s``, ``pulse_len_samples``,
            ``irig_bit``, ``utc_unix``, and ``utc_datetime``.
    """
    decoded = irig_core.decode_sync_signal_to_irig(
        sync_signal=sync_signal,
        sample_rate_hz=sample_rate_hz,
        bit_period_s=bit_period_s,
        irig_format=irig_format,
    )
    irig_df = decoded.pulse_df.rename(
        columns={
            "source_onset_ix": "sample_ix",
            "source_onset_time_s": "recording_time_s",
            "pulse_width_s": "pulse_len_seconds",
        }
    )
    irig_df["pulse_len_samples"] = irig_df["pulse_len_seconds"].to_numpy(dtype=float) * float(
        sample_rate_hz
    )
    all_unix = irig_df["utc_unix"].to_numpy(dtype=float)
    if align_first_rising_edge_unix is not None and np.isfinite(align_first_rising_edge_unix):
        finite_ix = np.where(np.isfinite(all_unix))[0]
        if finite_ix.size > 0:
            shift_s = float(align_first_rising_edge_unix) - float(all_unix[finite_ix[0]])
            all_unix = all_unix + shift_s
            irig_df["utc_unix"] = all_unix

    utc_datetime = [
        datetime.fromtimestamp(unix_time, tz=timezone.utc) if np.isfinite(unix_time) else pd.NaT
        for unix_time in all_unix
    ]
    irig_df["utc_datetime"] = utc_datetime
    return irig_df.loc[
        :,
        [
            "sample_ix",
            "recording_time_s",
            "pulse_len_samples",
            "irig_bit",
            "utc_unix",
            "utc_datetime",
        ],
    ]


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
