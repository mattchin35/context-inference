"""Core IRIG-H encoding and decoding utilities.

This module holds format-aware IRIG bit handling. File-specific modules should
extract pulse widths and metadata from their own input formats, then call these
helpers to classify pulses, find frames, and decode UTC timestamps.
"""

from __future__ import annotations

import calendar
import datetime
import time
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

IRIGBit = Literal[False, True, "P"]
IRIGFormat = Literal["neurokairos", "standard_decisecond"]

BIT_PERIOD_SECONDS = 1.0
MARKER_POSITIONS = {0, 9, 19, 29, 39, 49, 59}

SECONDS_WEIGHTS = [1, 2, 4, 8, 10, 20, 40]
MINUTES_WEIGHTS = [1, 2, 4, 8, 10, 20, 40]
HOURS_WEIGHTS = [1, 2, 4, 8, 10, 20]
DAY_OF_YEAR_WEIGHTS = [1, 2, 4, 8, 10, 20, 40, 80, 100, 200]
DECISECONDS_WEIGHTS = [1, 2, 4, 8]
YEARS_WEIGHTS = [1, 2, 4, 8, 10, 20, 40, 80]


@dataclass(frozen=True)
class IRIGDecodeResult:
    """Decoded pulse and frame tables from a sequence of IRIG pulse widths.

    Attributes:
        pulse_df: One row per input pulse. Pulse widths are in seconds. UTC
            values are Unix seconds, with ``NaN`` for invalid/unassigned pulses.
        frame_df: One row per valid 60-pulse frame found in the valid pulse
            stream. Frame boundaries are reported in original input pulse
            indices and compressed valid-pulse indices.
        invalid_pulse_ix: Integer input pulse indices with invalid widths.
    """

    pulse_df: pd.DataFrame
    frame_df: pd.DataFrame
    invalid_pulse_ix: np.ndarray


def bcd_encode(value: int, weights: list[int]) -> list[bool]:
    """Encode an integer as weighted BCD bits.

    Args:
        value: Integer value to encode.
        weights: BCD weights with shape ``(n_bits,)``.

    Returns:
        list[bool]: BCD bits with shape ``(n_bits,)``.
    """

    remaining = int(value)
    encoded = [False] * len(weights)
    for bit_ix in reversed(range(len(weights))):
        if weights[bit_ix] <= remaining:
            encoded[bit_ix] = True
            remaining -= weights[bit_ix]
    return encoded


def bcd_decode(bits: list[object] | np.ndarray, weights: list[int]) -> int:
    """Decode weighted BCD bits into an integer.

    Args:
        bits: Binary bit values with shape ``(n_bits,)``.
        weights: BCD weights with shape ``(n_bits,)``.

    Returns:
        int: Decoded integer.
    """

    bit_list = list(np.asarray(bits, dtype=object).reshape(-1))
    if len(bit_list) != len(weights):
        raise ValueError("Bit and weight lists must be the same length.")

    total = 0
    for bit, weight in zip(bit_list, weights):
        if bit not in (False, True, 0, 1):
            raise ValueError(f"Cannot BCD-decode non-binary bit value: {bit!r}")
        if bool(bit):
            total += int(weight)
    return total


def encode_stratum_code(chrony_stratum: int) -> int:
    """Encode chrony stratum into the NeuroKairos two-bit bucket.

    Args:
        chrony_stratum: Chrony stratum value. Values ``1``, ``2``, and ``3``
            have exact buckets; all other values use the fourth bucket.

    Returns:
        int: Encoded two-bit bucket value in ``[0, 3]``.
    """

    if int(chrony_stratum) == 1:
        return 0
    if int(chrony_stratum) == 2:
        return 1
    if int(chrony_stratum) == 3:
        return 2
    return 3


def decode_stratum_code(raw_code: int) -> str:
    """Decode the custom two-bit NeuroKairos stratum bucket.

    Args:
        raw_code: Encoded stratum bucket in ``[0, 3]``.

    Returns:
        str: Human-readable stratum bucket label.
    """

    if int(raw_code) == 0:
        return "stratum_1"
    if int(raw_code) == 1:
        return "stratum_2"
    if int(raw_code) == 2:
        return "stratum_3"
    return "stratum_4_or_higher"


def encode_root_dispersion_code(root_dispersion_s: float) -> int:
    """Encode root dispersion into the NeuroKairos three-bit bucket.

    Args:
        root_dispersion_s: Root dispersion in seconds.

    Returns:
        int: Encoded three-bit bucket value in ``[0, 7]``.
    """

    root_dispersion_s = float(root_dispersion_s)
    if root_dispersion_s < 0.00025:
        return 0
    if root_dispersion_s < 0.0005:
        return 1
    if root_dispersion_s < 0.001:
        return 2
    if root_dispersion_s < 0.002:
        return 3
    if root_dispersion_s < 0.004:
        return 4
    if root_dispersion_s < 0.008:
        return 5
    if root_dispersion_s < 0.016:
        return 6
    return 7


def describe_dispersion_code(raw_code: int) -> str:
    """Describe the custom three-bit NeuroKairos root-dispersion bucket.

    Args:
        raw_code: Encoded root-dispersion bucket in ``[0, 7]``.

    Returns:
        str: Human-readable root-dispersion bucket label.
    """

    labels = {
        0: "<0.25 ms",
        1: "0.25-0.5 ms",
        2: "0.5-1 ms",
        3: "1-2 ms",
        4: "2-4 ms",
        5: "4-8 ms",
        6: "8-16 ms",
    }
    return labels.get(int(raw_code), ">=16 ms")


def bit_symbol(bit: object) -> str:
    """Convert one classified IRIG bit to a printable symbol.

    Args:
        bit: One classified bit value: ``False``, ``True``, ``'P'``, or
            invalid/unknown.

    Returns:
        str: ``"0"``, ``"1"``, ``"P"``, or ``"?"``.
    """

    if bit == "P":
        return "P"
    if bit is True:
        return "1"
    if bit is False:
        return "0"
    return "?"


def _validate_irig_format(irig_format: str) -> IRIGFormat:
    """Validate and normalize an IRIG format label.

    Args:
        irig_format: Format name string.

    Returns:
        IRIGFormat: Validated format name.
    """

    if irig_format not in ("neurokairos", "standard_decisecond"):
        raise ValueError(
            "irig_format must be either 'neurokairos' or 'standard_decisecond'."
        )
    return irig_format


def encode_irig_frame(
    unix_seconds: float | None = None,
    irig_format: IRIGFormat = "neurokairos",
    chrony_stratum: int = 0,
    root_dispersion_s: float = 1.0,
) -> list[object]:
    """Generate one 60-bit IRIG-H frame from UTC Unix seconds.

    Args:
        unix_seconds: UTC Unix seconds for the frame start. If omitted,
            ``time.time()`` is used. Whole seconds are encoded in the common
            IRIG calendar fields.
        irig_format: ``"neurokairos"`` for metadata bits at positions 43-48,
            or ``"standard_decisecond"`` for legacy deciseconds at positions
            45-48.
        chrony_stratum: Chrony stratum for NeuroKairos metadata encoding.
        root_dispersion_s: Root dispersion in seconds for NeuroKairos metadata
            encoding.

    Returns:
        list[object]: IRIG-H bit values with shape ``(60,)``.
    """

    irig_format = _validate_irig_format(irig_format)
    if unix_seconds is None:
        unix_seconds = time.time()
    unix_seconds = float(unix_seconds)
    utc_time = time.gmtime(unix_seconds)

    seconds_bcd = bcd_encode(utc_time.tm_sec, SECONDS_WEIGHTS)
    minutes_bcd = bcd_encode(utc_time.tm_min, MINUTES_WEIGHTS)
    hours_bcd = bcd_encode(utc_time.tm_hour, HOURS_WEIGHTS)
    day_of_year_bcd = bcd_encode(utc_time.tm_yday, DAY_OF_YEAR_WEIGHTS)
    year_bcd = bcd_encode(utc_time.tm_year % 100, YEARS_WEIGHTS)

    frame: list[object] = ["P"] + [False] * 58 + ["P"]
    for marker_pos in MARKER_POSITIONS:
        frame[marker_pos] = "P"

    frame[1:5] = seconds_bcd[0:4]
    frame[5] = False
    frame[6:9] = seconds_bcd[4:7]

    frame[10:14] = minutes_bcd[0:4]
    frame[14] = False
    frame[15:18] = minutes_bcd[4:7]
    frame[18] = False

    frame[20:24] = hours_bcd[0:4]
    frame[24] = False
    frame[25:27] = hours_bcd[4:6]
    frame[27:29] = [False, False]

    frame[30:34] = day_of_year_bcd[0:4]
    frame[34] = False
    frame[35:39] = day_of_year_bcd[4:8]
    frame[40:42] = day_of_year_bcd[8:10]

    if irig_format == "neurokairos":
        stratum_code = encode_stratum_code(chrony_stratum)
        dispersion_code = encode_root_dispersion_code(root_dispersion_s)
        frame[42] = False
        frame[43] = bool(stratum_code & 0b001)
        frame[44] = bool(stratum_code & 0b010)
        frame[45] = False
        frame[46] = bool(dispersion_code & 0b001)
        frame[47] = bool(dispersion_code & 0b010)
        frame[48] = bool(dispersion_code & 0b100)
    else:
        deciseconds = int(np.floor((unix_seconds - np.floor(unix_seconds)) * 10.0 + 1e-9))
        deciseconds = int(np.clip(deciseconds, 0, 9))
        deciseconds_bcd = bcd_encode(deciseconds, DECISECONDS_WEIGHTS)
        frame[42:45] = [False, False, False]
        frame[45:49] = deciseconds_bcd

    frame[50:54] = year_bcd[0:4]
    frame[54] = False
    frame[55:59] = year_bcd[4:8]
    return frame


def decode_irig_frame(
    frame_bits: npt.ArrayLike,
    irig_format: IRIGFormat = "neurokairos",
    century_base: int | None = None,
) -> dict[str, object]:
    """Decode one 60-bit IRIG-H frame.

    Args:
        frame_bits: IRIG-H bit values with shape ``(60,)``.
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.
        century_base: Optional century to add to the two-digit year, for
            example ``2000``. If omitted, the current UTC century is used.

    Returns:
        dict[str, object]: Decoded calendar fields, UTC Unix seconds, metadata
            fields for the selected format, and a compact bit string.
    """

    irig_format = _validate_irig_format(irig_format)
    frame = list(np.asarray(frame_bits, dtype=object).reshape(-1))
    if len(frame) != 60:
        raise ValueError("frame_bits must contain exactly 60 IRIG bits.")
    if any(frame[pos] != "P" for pos in MARKER_POSITIONS):
        raise ValueError("frame_bits does not contain markers at the expected positions.")

    seconds = bcd_decode(frame[1:5], SECONDS_WEIGHTS[0:4]) + bcd_decode(
        frame[6:9], SECONDS_WEIGHTS[4:7]
    )
    minutes = bcd_decode(frame[10:14], MINUTES_WEIGHTS[0:4]) + bcd_decode(
        frame[15:18], MINUTES_WEIGHTS[4:7]
    )
    hours = bcd_decode(frame[20:24], HOURS_WEIGHTS[0:4]) + bcd_decode(
        frame[25:27], HOURS_WEIGHTS[4:6]
    )
    day_of_year = (
        bcd_decode(frame[30:34], DAY_OF_YEAR_WEIGHTS[0:4])
        + bcd_decode(frame[35:39], DAY_OF_YEAR_WEIGHTS[4:8])
        + bcd_decode(frame[40:42], DAY_OF_YEAR_WEIGHTS[8:10])
    )
    year_two_digit = bcd_decode(frame[50:54], YEARS_WEIGHTS[0:4]) + bcd_decode(
        frame[55:59], YEARS_WEIGHTS[4:8]
    )
    if century_base is None:
        century_base = (time.gmtime(time.time()).tm_year // 100) * 100
    decoded_year = int(century_base) + year_two_digit

    decoded_datetime_utc = None
    decoded_datetime_utc_text = None
    decoded_utc_seconds = np.nan
    try:
        decoded_date = datetime.date(decoded_year, 1, 1) + datetime.timedelta(
            days=day_of_year - 1
        )
        datetime.time(hours, minutes, seconds)
        decoded_utc_seconds = float(
            calendar.timegm(
                (
                    decoded_date.year,
                    decoded_date.month,
                    decoded_date.day,
                    hours,
                    minutes,
                    seconds,
                )
            )
        )
        decoded_datetime_utc = datetime.datetime.fromtimestamp(
            decoded_utc_seconds, tz=datetime.timezone.utc
        )
    except ValueError:
        decoded_utc_seconds = np.nan

    output: dict[str, object] = {
        "seconds": seconds,
        "minutes": minutes,
        "hours": hours,
        "day_of_year": day_of_year,
        "year_two_digit": year_two_digit,
        "decoded_year": decoded_year,
        "decoded_datetime_utc": decoded_datetime_utc_text,
        "decoded_utc_seconds": decoded_utc_seconds,
        "bit_string": "".join(bit_symbol(bit) for bit in frame),
    }

    if irig_format == "standard_decisecond":
        deciseconds = bcd_decode(frame[45:49], DECISECONDS_WEIGHTS)
        if np.isfinite(decoded_utc_seconds):
            decoded_utc_seconds = float(decoded_utc_seconds) + deciseconds * 0.1
            decoded_datetime_utc = datetime.datetime.fromtimestamp(
                decoded_utc_seconds, tz=datetime.timezone.utc
            )
        output["decoded_utc_seconds"] = decoded_utc_seconds
        output["deciseconds"] = deciseconds
        output["reserved_bits_42_44"] = [int(bool(bit)) for bit in frame[42:45]]
    else:
        raw_stratum_code = int(bool(frame[43])) + (int(bool(frame[44])) << 1)
        raw_dispersion_code = (
            int(bool(frame[46]))
            + (int(bool(frame[47])) << 1)
            + (int(bool(frame[48])) << 2)
        )
        output.update(
            {
                "reserved_bit_42": int(bool(frame[42])),
                "custom_stratum_code": raw_stratum_code,
                "decoded_stratum": decode_stratum_code(raw_stratum_code),
                "reserved_bit_45": int(bool(frame[45])),
                "custom_dispersion_code": raw_dispersion_code,
                "decoded_dispersion_bucket": describe_dispersion_code(raw_dispersion_code),
            }
        )

    if decoded_datetime_utc is not None:
        output["decoded_datetime_utc"] = decoded_datetime_utc.isoformat()
    return output


def classify_pulse_widths_seconds(
    pulse_width_s: npt.ArrayLike,
    bit_period_s: float = BIT_PERIOD_SECONDS,
) -> np.ndarray:
    """Classify pulse widths in seconds into IRIG bit labels.

    Args:
        pulse_width_s: Pulse widths with shape ``(n_pulses,)`` in seconds.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        np.ndarray: Object array with shape ``(n_pulses,)`` containing
        ``False``, ``True``, ``'P'``, or ``None``.
    """

    pulse_width_s = np.asarray(pulse_width_s, dtype=float).reshape(-1)
    p_thresh = 0.75 * float(bit_period_s)
    one_thresh = 0.45 * float(bit_period_s)
    zero_thresh = 0.05 * float(bit_period_s)

    bit_labels = np.full(pulse_width_s.shape, None, dtype=object)
    bit_labels[pulse_width_s > p_thresh] = "P"
    bit_labels[(pulse_width_s > one_thresh) & (pulse_width_s <= p_thresh)] = True
    bit_labels[(pulse_width_s > zero_thresh) & (pulse_width_s <= one_thresh)] = False
    return bit_labels


def classify_pulse_widths_samples(
    pulse_lengths_samples: npt.ArrayLike,
    sample_rate_hz: float,
    bit_period_s: float = BIT_PERIOD_SECONDS,
) -> np.ndarray:
    """Classify pulse widths in samples into IRIG bit labels.

    Args:
        pulse_lengths_samples: Pulse widths with shape ``(n_pulses,)`` in
            samples.
        sample_rate_hz: Sampling rate in Hz.
        bit_period_s: IRIG bit period in seconds.

    Returns:
        np.ndarray: Object array with shape ``(n_pulses,)`` containing
        ``False``, ``True``, ``'P'``, or ``None``.
    """

    pulse_lengths_samples = np.asarray(pulse_lengths_samples, dtype=float).reshape(-1)
    pulse_width_s = pulse_lengths_samples / float(sample_rate_hz)
    return classify_pulse_widths_seconds(pulse_width_s, bit_period_s=bit_period_s)


def find_irig_frame_spans(irig_bits: npt.ArrayLike) -> list[tuple[int, int]]:
    """Locate consecutive 60-pulse IRIG frames.

    Args:
        irig_bits: Classified IRIG bits with shape ``(n_bits,)``.

    Returns:
        list[tuple[int, int]]: Half-open ``(start_ix, end_ix)`` frame spans in
        bit indices.
    """

    irig_bits = np.asarray(irig_bits, dtype=object).reshape(-1)
    frame_spans: list[tuple[int, int]] = []
    if irig_bits.size < 60:
        return frame_spans

    frame_start = None
    for bit_ix in range(irig_bits.size - 1):
        if irig_bits[bit_ix] == "P" and irig_bits[bit_ix + 1] == "P":
            candidate_start = bit_ix + 1 - 60
            if candidate_start < 0:
                continue

            candidate_end = candidate_start + 60
            candidate_bits = irig_bits[candidate_start:candidate_end]
            if all(candidate_bits[pos] == "P" for pos in MARKER_POSITIONS):
                frame_start = candidate_start
                break
    if frame_start is None:
        return frame_spans

    while frame_start + 60 <= irig_bits.size:
        frame_end = frame_start + 60
        frame_bits = irig_bits[frame_start:frame_end]
        if all(frame_bits[pos] == "P" for pos in MARKER_POSITIONS):
            frame_spans.append((frame_start, frame_end))
            frame_start = frame_end
        else:
            frame_start += 1
    return frame_spans


def find_signal_edges(binary_signal: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Find rising and falling edge sample indices for a digital signal.

    Args:
        binary_signal: Digital signal with shape ``(n_samples,)``. Nonzero
            values are treated as high.

    Returns:
        tuple[np.ndarray, np.ndarray]: Rising and falling edge sample indices,
        each with shape ``(n_edges,)`` in samples.
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
        tuple[np.ndarray, np.ndarray]: Paired rising edge indices and pulse
        lengths, each with shape ``(n_pulses,)`` in samples.
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


def decode_irig_frame_anchors(
    irig_bits: npt.ArrayLike,
    irig_format: IRIGFormat = "neurokairos",
    century_base: int | None = None,
) -> list[tuple[int, float]]:
    """Decode frame-start UTC anchors from classified valid IRIG bits.

    Args:
        irig_bits: Classified valid IRIG bits with shape ``(n_valid_bits,)``.
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.
        century_base: Optional century base for two-digit year decoding.

    Returns:
        list[tuple[int, float]]: ``(frame_start_bit_ix, frame_start_utc_unix)``
        pairs in valid-bit indices and UTC Unix seconds.
    """

    irig_bits = np.asarray(irig_bits, dtype=object).reshape(-1)
    anchors: list[tuple[int, float]] = []
    for start_ix, end_ix in find_irig_frame_spans(irig_bits):
        frame_fields = decode_irig_frame(
            irig_bits[start_ix:end_ix],
            irig_format=irig_format,
            century_base=century_base,
        )
        frame_unix = float(frame_fields["decoded_utc_seconds"])
        if np.isfinite(frame_unix):
            anchors.append((int(start_ix), frame_unix))
    return anchors


def assign_utc_to_irig_bits(
    n_bits: int,
    frame_anchors: list[tuple[int, float]],
) -> np.ndarray:
    """Assign UTC Unix seconds to each IRIG bit index.

    Args:
        n_bits: Number of classified IRIG bits.
        frame_anchors: ``(frame_start_bit_ix, frame_start_utc_unix)`` pairs.

    Returns:
        np.ndarray: UTC Unix seconds with shape ``(n_bits,)``. Missing values
        are ``NaN``.
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
    unix_time[first_anchor_ix:] = anchor_unix[0] + np.arange(
        n_bits - first_anchor_ix, dtype=float
    )
    unix_time[:first_anchor_ix] = anchor_unix[0] - np.arange(
        first_anchor_ix, 0, -1, dtype=float
    )
    for anchor_position in range(1, anchor_ix.size):
        start_ix = int(anchor_ix[anchor_position])
        unix_time[start_ix:] = anchor_unix[anchor_position] + np.arange(
            n_bits - start_ix, dtype=float
        )
    return unix_time


def decode_irig_pulses(
    pulse_width_s: npt.ArrayLike,
    source_onset_time_s: npt.ArrayLike | None = None,
    source_onset_ix: npt.ArrayLike | None = None,
    irig_format: IRIGFormat = "neurokairos",
    bit_period_s: float = BIT_PERIOD_SECONDS,
    century_base: int | None = None,
    warn_invalid: bool = True,
) -> IRIGDecodeResult:
    """Decode IRIG pulse widths into pulse and frame tables.

    Args:
        pulse_width_s: Pulse widths with shape ``(n_pulses,)`` in seconds.
        source_onset_time_s: Optional source-domain pulse onset times with
            shape ``(n_pulses,)`` in seconds.
        source_onset_ix: Optional source-domain pulse onset indices with shape
            ``(n_pulses,)`` in samples or rows.
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.
        bit_period_s: IRIG bit period in seconds.
        century_base: Optional century base for two-digit year decoding.
        warn_invalid: If ``True``, emit a warning listing invalid pulse
            indices.

    Returns:
        IRIGDecodeResult: Pulse table, frame table, and invalid pulse indices.
    """

    irig_format = _validate_irig_format(irig_format)
    pulse_width_s = np.asarray(pulse_width_s, dtype=float).reshape(-1)
    n_pulses = pulse_width_s.size
    pulse_ix = np.arange(n_pulses, dtype=np.int64)

    if source_onset_time_s is not None:
        source_onset_time_s = np.asarray(source_onset_time_s, dtype=float).reshape(-1)
        if source_onset_time_s.size != n_pulses:
            raise ValueError("source_onset_time_s must match pulse_width_s length.")
    if source_onset_ix is not None:
        source_onset_ix = np.asarray(source_onset_ix, dtype=np.int64).reshape(-1)
        if source_onset_ix.size != n_pulses:
            raise ValueError("source_onset_ix must match pulse_width_s length.")

    irig_bits = classify_pulse_widths_seconds(pulse_width_s, bit_period_s=bit_period_s)
    valid_mask = np.asarray([bit is not None for bit in irig_bits], dtype=bool)
    invalid_pulse_ix = pulse_ix[~valid_mask]
    if warn_invalid and invalid_pulse_ix.size > 0:
        warnings.warn(
            "Invalid IRIG pulses present at pulse indices: "
            + ", ".join(str(int(ix)) for ix in invalid_pulse_ix),
            UserWarning,
            stacklevel=2,
        )

    valid_bits = irig_bits[valid_mask]
    valid_to_pulse_ix = pulse_ix[valid_mask]
    frame_spans = find_irig_frame_spans(valid_bits)
    frame_anchors = decode_irig_frame_anchors(
        valid_bits,
        irig_format=irig_format,
        century_base=century_base,
    )
    valid_unix = assign_utc_to_irig_bits(valid_bits.size, frame_anchors)

    all_unix = np.full(n_pulses, np.nan, dtype=float)
    all_unix[valid_mask] = valid_unix
    utc_datetime = [
        datetime.datetime.fromtimestamp(unix_time, tz=datetime.timezone.utc)
        if np.isfinite(unix_time)
        else pd.NaT
        for unix_time in all_unix
    ]

    pulse_columns: dict[str, object] = {
        "pulse_ix": pulse_ix,
        "pulse_width_s": pulse_width_s,
        "irig_bit": irig_bits,
        "utc_unix": all_unix,
        "utc_datetime": utc_datetime,
    }
    if source_onset_ix is not None:
        pulse_columns["source_onset_ix"] = source_onset_ix
    if source_onset_time_s is not None:
        pulse_columns["source_onset_time_s"] = source_onset_time_s
    pulse_df = pd.DataFrame(pulse_columns)

    frame_rows: list[dict[str, object]] = []
    for frame_ix, (start_valid_ix, end_valid_ix) in enumerate(frame_spans):
        frame_fields = decode_irig_frame(
            valid_bits[start_valid_ix:end_valid_ix],
            irig_format=irig_format,
            century_base=century_base,
        )
        start_pulse_ix = int(valid_to_pulse_ix[start_valid_ix])
        end_pulse_ix = int(valid_to_pulse_ix[end_valid_ix - 1]) + 1
        frame_row: dict[str, object] = {
            "frame_ix": frame_ix,
            "frame_start_pulse_ix": start_pulse_ix,
            "frame_end_pulse_ix": end_pulse_ix,
            "frame_start_valid_ix": int(start_valid_ix),
            "frame_end_valid_ix": int(end_valid_ix),
        }
        if source_onset_ix is not None:
            frame_row["observed_start_ix"] = int(source_onset_ix[start_pulse_ix])
        if source_onset_time_s is not None:
            frame_row["observed_start_time_s"] = float(source_onset_time_s[start_pulse_ix])
        frame_row.update(frame_fields)
        frame_rows.append(frame_row)
    frame_df = pd.DataFrame(frame_rows)

    return IRIGDecodeResult(
        pulse_df=pulse_df,
        frame_df=frame_df,
        invalid_pulse_ix=invalid_pulse_ix,
    )


def decode_sync_signal_to_irig(
    sync_signal: npt.ArrayLike,
    sample_rate_hz: float,
    bit_period_s: float = BIT_PERIOD_SECONDS,
    irig_format: IRIGFormat = "neurokairos",
    century_base: int | None = None,
    warn_invalid: bool = True,
) -> IRIGDecodeResult:
    """Extract and decode IRIG pulses from a sampled digital signal.

    Args:
        sync_signal: Boolean or binary sync signal with shape ``(n_samples,)``.
        sample_rate_hz: Sampling rate in Hz.
        bit_period_s: IRIG bit period in seconds.
        irig_format: ``"neurokairos"`` or ``"standard_decisecond"``.
        century_base: Optional century base for two-digit year decoding.
        warn_invalid: If ``True``, emit a warning listing invalid pulse
            indices.

    Returns:
        IRIGDecodeResult: Pulse and frame tables decoded from the sampled sync
        signal.
    """

    rising_ix, falling_ix = find_signal_edges(sync_signal)
    paired_rising_ix, pulse_lengths_samples = pulse_lengths_from_edges(rising_ix, falling_ix)
    pulse_width_s = pulse_lengths_samples.astype(float) / float(sample_rate_hz)
    return decode_irig_pulses(
        pulse_width_s=pulse_width_s,
        source_onset_time_s=paired_rising_ix.astype(float) / float(sample_rate_hz),
        source_onset_ix=paired_rising_ix,
        irig_format=irig_format,
        bit_period_s=bit_period_s,
        century_base=century_base,
        warn_invalid=warn_invalid,
    )
