"""Open Ephys TTL event adapters for shared IRIG-H decoding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.irig_tools import irig_core


def load_open_ephys_ttl_events(
    ttl_dir: Path | str,
    sample_rate_hz: float | None = None,
) -> pd.DataFrame:
    """Load an Open Ephys Binary-format TTL event folder.

    Args:
        ttl_dir: Directory containing Open Ephys TTL arrays. Required files are
            ``states.npy`` and ``sample_numbers.npy``. ``timestamps.npy`` is
            preferred and stores seconds. If it is absent, ``sample_rate_hz`` is
            required to convert sample numbers to seconds. ``full_words.npy`` is
            optional.
        sample_rate_hz: Optional sample rate in Hz. Used only when
            ``timestamps.npy`` is absent.

    Returns:
        pd.DataFrame: One row per TTL transition with columns ``sample_ix`` in
        samples, ``recording_time_s`` in seconds, zero-based ``line``, binary
        ``state`` where 1 is high and 0 is low, and optional ``full_word``.
    """
    ttl_path = Path(ttl_dir)
    states = _load_required_array(ttl_path / "states.npy", "states").astype(np.int64).reshape(-1)
    sample_numbers = _load_required_array(ttl_path / "sample_numbers.npy", "sample_numbers").astype(
        np.int64
    ).reshape(-1)

    timestamps_path = ttl_path / "timestamps.npy"
    if timestamps_path.exists():
        timestamps = np.asarray(np.load(timestamps_path, allow_pickle=False), dtype=float).reshape(-1)
    else:
        if sample_rate_hz is None:
            raise FileNotFoundError(
                f"Open Ephys TTL timestamps file is missing: {timestamps_path}. "
                "Pass sample_rate_hz to compute timestamps from sample_numbers."
            )
        if float(sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be positive when timestamps.npy is absent.")
        timestamps = sample_numbers.astype(float) / float(sample_rate_hz)

    _validate_equal_lengths(
        {
            "states": states,
            "sample_numbers": sample_numbers,
            "timestamps": timestamps,
        }
    )
    if np.any(states == 0):
        raise ValueError("Open Ephys TTL states must be nonzero signed channel-state values.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Open Ephys TTL timestamps must be finite.")

    event_df = pd.DataFrame(
        {
            "sample_ix": sample_numbers,
            "recording_time_s": timestamps,
            "line": np.abs(states).astype(np.int64) - 1,
            "state": (states > 0).astype(np.int64),
        }
    )

    full_words_path = ttl_path / "full_words.npy"
    if full_words_path.exists():
        full_words = np.asarray(np.load(full_words_path, allow_pickle=False), dtype=np.uint64).reshape(-1)
        _validate_equal_lengths({"full_words": full_words, "states": states})
        event_df["full_word"] = full_words

    return event_df.sort_values(["recording_time_s", "sample_ix"], kind="stable").reset_index(drop=True)


def extract_line_transitions(event_df: pd.DataFrame, line: int = 0) -> pd.DataFrame:
    """Select transitions for one Open Ephys TTL line.

    Args:
        event_df: Transition dataframe from :func:`load_open_ephys_ttl_events`
            with shape ``(n_events, n_columns)``. Required columns are
            ``sample_ix``, ``recording_time_s``, ``line``, and ``state``.
        line: Zero-based TTL line to select.

    Returns:
        pd.DataFrame: Sorted transition dataframe for ``line``. Samples and
        timestamps preserve the Open Ephys coordinates and units.
    """
    _require_columns(event_df, {"sample_ix", "recording_time_s", "line", "state"}, "event_df")
    selected_df = event_df.loc[event_df["line"].astype(int) == int(line)].copy()
    return selected_df.sort_values(["recording_time_s", "sample_ix"], kind="stable").reset_index(drop=True)


def debounce_line_transitions(
    transition_df: pd.DataFrame,
    min_high_duration_s: float = 0.01,
    min_low_duration_s: float = 0.01,
) -> pd.DataFrame:
    """Remove short TTL glitches while preserving transition coordinates.

    Args:
        transition_df: Transition dataframe for one TTL line with shape
            ``(n_events, n_columns)``. Required columns are ``sample_ix``,
            ``recording_time_s``, and ``state``. Times are seconds and state is
            1 for high, 0 for low.
        min_high_duration_s: High intervals shorter than this duration, in
            seconds, are removed as high glitches.
        min_low_duration_s: Low intervals shorter than this duration, in
            seconds, are merged over as low glitches when high intervals exist
            on both sides.

    Returns:
        pd.DataFrame: Debounced transition dataframe with the same columns as
        ``transition_df`` and rows sorted by recording time.
    """
    _require_columns(transition_df, {"sample_ix", "recording_time_s", "state"}, "transition_df")
    if float(min_high_duration_s) < 0:
        raise ValueError("min_high_duration_s must be nonnegative.")
    if float(min_low_duration_s) < 0:
        raise ValueError("min_low_duration_s must be nonnegative.")

    rows = transition_df.sort_values(["recording_time_s", "sample_ix"], kind="stable").to_dict("records")
    rows = _collapse_repeated_states(rows)

    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(rows) - 1:
            current_state = int(rows[index]["state"])
            next_state = int(rows[index + 1]["state"])
            duration_s = float(rows[index + 1]["recording_time_s"]) - float(rows[index]["recording_time_s"])
            if current_state == 1 and next_state == 0 and duration_s < float(min_high_duration_s):
                del rows[index:index + 2]
                changed = True
                index = max(index - 1, 0)
                continue
            if (
                current_state == 0
                and next_state == 1
                and duration_s < float(min_low_duration_s)
                and index > 0
                and index + 2 < len(rows)
            ):
                del rows[index:index + 2]
                changed = True
                index = max(index - 1, 0)
                continue
            index += 1
        if changed:
            rows = _collapse_repeated_states(rows)

    if not rows:
        return transition_df.iloc[0:0].copy().reset_index(drop=True)
    return pd.DataFrame(rows, columns=transition_df.columns).reset_index(drop=True)


def extract_irig_pulses_from_transitions(transition_df: pd.DataFrame) -> pd.DataFrame:
    """Pair Open Ephys TTL transitions into IRIG high-pulse measurements.

    Args:
        transition_df: Transition dataframe for one debounced TTL line with
            shape ``(n_events, n_columns)``. Required columns are ``sample_ix``,
            ``recording_time_s``, and ``state``. ``sample_ix`` is in samples,
            ``recording_time_s`` is in seconds, and ``state`` is 1 for high and
            0 for low.

    Returns:
        pd.DataFrame: Pulse dataframe with columns ``pulse_ix``,
        ``source_onset_ix`` in samples, ``source_onset_time_s`` in seconds,
        ``pulse_width_s`` in seconds, and ``pulse_width_samples`` in samples.
    """
    _require_columns(transition_df, {"sample_ix", "recording_time_s", "state"}, "transition_df")
    ordered_df = transition_df.sort_values(["recording_time_s", "sample_ix"], kind="stable").reset_index(drop=True)
    states = ordered_df["state"].to_numpy(dtype=np.int64)
    if not np.isin(states, [0, 1]).all():
        raise ValueError("transition_df state values must contain only 0 and 1.")

    rising_event_ix = np.flatnonzero(states == 1)
    falling_event_ix = np.flatnonzero(states == 0)
    next_falling_pos = np.searchsorted(falling_event_ix, rising_event_ix, side="right")
    valid_mask = next_falling_pos < falling_event_ix.size
    paired_rising_ix = rising_event_ix[valid_mask]
    paired_falling_ix = falling_event_ix[next_falling_pos[valid_mask]]

    onset_sample_ix = ordered_df.loc[paired_rising_ix, "sample_ix"].to_numpy(dtype=np.int64)
    falling_sample_ix = ordered_df.loc[paired_falling_ix, "sample_ix"].to_numpy(dtype=np.int64)
    onset_time_s = ordered_df.loc[paired_rising_ix, "recording_time_s"].to_numpy(dtype=float)
    falling_time_s = ordered_df.loc[paired_falling_ix, "recording_time_s"].to_numpy(dtype=float)

    pulse_width_s = falling_time_s - onset_time_s
    if np.any(pulse_width_s <= 0):
        raise ValueError("Every paired falling transition must occur after its rising transition.")

    return pd.DataFrame(
        {
            "pulse_ix": np.arange(onset_sample_ix.size, dtype=np.int64),
            "source_onset_ix": onset_sample_ix,
            "source_onset_time_s": onset_time_s,
            "pulse_width_s": pulse_width_s.astype(float),
            "pulse_width_samples": (falling_sample_ix - onset_sample_ix).astype(np.int64),
        }
    )


def decode_open_ephys_ttl_irig(
    ttl_dir: Path | str,
    line: int = 0,
    bit_period_s: float = 1.0,
    irig_format: irig_core.IRIGFormat = "neurokairos",
    sample_rate_hz: float | None = None,
    min_high_duration_s: float = 0.01,
    min_low_duration_s: float = 0.01,
    century_base: int | None = None,
    warn_invalid: bool = True,
) -> irig_core.IRIGDecodeResult:
    """Decode IRIG-H from an Open Ephys TTL event folder.

    Args:
        ttl_dir: Open Ephys TTL event folder containing ``states.npy``,
            ``sample_numbers.npy``, and usually ``timestamps.npy``.
        line: Zero-based TTL line carrying IRIG-H.
        bit_period_s: IRIG-H bit period in seconds.
        irig_format: IRIG frame format, ``"neurokairos"`` or
            ``"standard_decisecond"``.
        sample_rate_hz: Optional sample rate in Hz. Used only if
            ``timestamps.npy`` is absent.
        min_high_duration_s: Debounce threshold for short high glitches, in
            seconds.
        min_low_duration_s: Debounce threshold for short low glitches, in
            seconds.
        century_base: Optional century base for two-digit year decoding.
        warn_invalid: If ``True``, warn about IRIG pulses that remain
            unclassified after debouncing.

    Returns:
        irig_core.IRIGDecodeResult: Pulse and frame tables decoded in Open
        Ephys sample/time coordinates.
    """
    event_df = load_open_ephys_ttl_events(ttl_dir, sample_rate_hz=sample_rate_hz)
    line_df = extract_line_transitions(event_df, line=line)
    debounced_df = debounce_line_transitions(
        line_df,
        min_high_duration_s=min_high_duration_s,
        min_low_duration_s=min_low_duration_s,
    )
    pulse_df = extract_irig_pulses_from_transitions(debounced_df)
    decoded = irig_core.decode_irig_pulses(
        pulse_width_s=pulse_df["pulse_width_s"].to_numpy(dtype=float),
        source_onset_time_s=pulse_df["source_onset_time_s"].to_numpy(dtype=float),
        source_onset_ix=pulse_df["source_onset_ix"].to_numpy(dtype=np.int64),
        irig_format=irig_format,
        bit_period_s=bit_period_s,
        century_base=century_base,
        warn_invalid=warn_invalid,
    )
    decoded.pulse_df["pulse_width_samples"] = pulse_df["pulse_width_samples"].to_numpy(dtype=np.int64)
    return decoded


def decode_open_ephys_ttl_irig_utc(
    ttl_dir: Path | str,
    line: int = 0,
    bit_period_s: float = 1.0,
    irig_format: irig_core.IRIGFormat = "neurokairos",
    sample_rate_hz: float | None = None,
    min_high_duration_s: float = 0.01,
    min_low_duration_s: float = 0.01,
    century_base: int | None = None,
    warn_invalid: bool = True,
) -> pd.DataFrame:
    """Decode Open Ephys TTL IRIG into the existing UTC sync dataframe schema.

    Args:
        ttl_dir: Open Ephys TTL event folder.
        line: Zero-based TTL line carrying IRIG-H.
        bit_period_s: IRIG-H bit period in seconds.
        irig_format: IRIG frame format, ``"neurokairos"`` or
            ``"standard_decisecond"``.
        sample_rate_hz: Optional sample rate in Hz for timestamp fallback.
        min_high_duration_s: Debounce threshold for short high glitches, in
            seconds.
        min_low_duration_s: Debounce threshold for short low glitches, in
            seconds.
        century_base: Optional century base for two-digit year decoding.
        warn_invalid: If ``True``, warn about unclassified IRIG pulses.

    Returns:
        pd.DataFrame: One row per debounced IRIG pulse with columns
        ``sample_ix`` in samples, ``recording_time_s`` in seconds,
        ``pulse_len_samples`` in samples, ``irig_bit``, ``utc_unix`` in
        seconds, and ``utc_datetime`` as timezone-aware UTC datetimes.
    """
    decoded = decode_open_ephys_ttl_irig(
        ttl_dir=ttl_dir,
        line=line,
        bit_period_s=bit_period_s,
        irig_format=irig_format,
        sample_rate_hz=sample_rate_hz,
        min_high_duration_s=min_high_duration_s,
        min_low_duration_s=min_low_duration_s,
        century_base=century_base,
        warn_invalid=warn_invalid,
    )
    pulse_df = decoded.pulse_df
    all_unix = pulse_df["utc_unix"].to_numpy(dtype=float)
    utc_datetime = [
        datetime.fromtimestamp(unix_time, tz=timezone.utc) if np.isfinite(unix_time) else pd.NaT
        for unix_time in all_unix
    ]
    return pd.DataFrame(
        {
            "sample_ix": pulse_df["source_onset_ix"].to_numpy(dtype=np.int64),
            "recording_time_s": pulse_df["source_onset_time_s"].to_numpy(dtype=float),
            "pulse_len_samples": pulse_df["pulse_width_samples"].to_numpy(dtype=np.int64),
            "irig_bit": pulse_df["irig_bit"].to_numpy(dtype=object),
            "utc_unix": all_unix,
            "utc_datetime": utc_datetime,
        }
    )


def _load_required_array(file_path: Path, array_name: str) -> np.ndarray:
    """Load one required ``.npy`` file from an Open Ephys TTL folder.

    Args:
        file_path: Path to the required NumPy array file.
        array_name: Human-readable array name used in error messages.

    Returns:
        np.ndarray: Loaded array with the shape and dtype stored in the file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Open Ephys TTL {array_name} file is missing: {file_path}")
    return np.asarray(np.load(file_path, allow_pickle=False))


def _validate_equal_lengths(arrays: dict[str, npt.ArrayLike]) -> None:
    """Validate that named arrays represent the same number of TTL events.

    Args:
        arrays: Mapping from array name to array-like values. Each value is
            flattened to shape ``(n_events,)`` before comparing lengths.

    Returns:
        None.
    """
    lengths = {name: np.asarray(value).reshape(-1).shape[0] for name, value in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Open Ephys TTL arrays must have the same length: {lengths}")


def _require_columns(data_frame: pd.DataFrame, required_columns: set[str], name: str) -> None:
    """Validate that a dataframe contains required columns.

    Args:
        data_frame: Dataframe with shape ``(n_rows, n_columns)``.
        required_columns: Column names that must be present.
        name: Human-readable dataframe name used in error messages.

    Returns:
        None.
    """
    missing_columns = required_columns - set(data_frame.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")


def _collapse_repeated_states(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop consecutive repeated TTL states from transition rows.

    Args:
        rows: List of transition records sorted by time. Each record must
            contain a binary ``state`` entry, where 1 is high and 0 is low.

    Returns:
        list[dict[str, object]]: Transition records with repeated adjacent
        states removed. Other record fields and units are preserved.
    """
    collapsed_rows: list[dict[str, object]] = []
    for row in rows:
        state = int(row["state"])
        if state not in {0, 1}:
            raise ValueError("transition_df state values must contain only 0 and 1.")
        if collapsed_rows and int(collapsed_rows[-1]["state"]) == state:
            continue
        collapsed_rows.append(row)
    return collapsed_rows
