"""Serial-log parsing helpers for IRIG-bearing treadmill output."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
import warnings

import numpy as np
import pandas as pd

PCTimestampFormat = Literal["datetime", "millisecondtime"] | None


def _require_reference_date(
    pc_timestamp_format: PCTimestampFormat,
    pc_reference_date: str | None,
) -> str:
    """Validate and return the hardcoded local date when one is required.

    Args:
        pc_timestamp_format: PC timestamp format for the file.
        pc_reference_date: User-provided local date string, ``YYYY-MM-DD``.

    Returns:
        str: Validated reference date string.
    """
    if pc_timestamp_format in (None, "millisecondtime") and not pc_reference_date:
        raise ValueError(
            "pc_reference_date is required when pc_timestamp_format is None or 'millisecondtime'."
        )
    if pc_reference_date:
        datetime.strptime(pc_reference_date, "%Y-%m-%d")
    return "" if pc_reference_date is None else pc_reference_date


def _parse_serial_message(serial_msg: str) -> dict[str, object]:
    """Parse one treadmill serial message.

    Args:
        serial_msg: Serial message text such as
            ``syncPinState:LOW;syncIntervalUs:499991;``.

    Returns:
        dict[str, object]: Parsed label/value and interval fields. Interval
        value is converted from microseconds to seconds.
    """
    fields = [field for field in serial_msg.split(";") if field]
    if len(fields) < 2:
        raise ValueError(f"Serial message must contain label and interval fields: {serial_msg!r}")

    label, value = fields[0].split(":", maxsplit=1)
    time_label, time_value = fields[1].split(":", maxsplit=1)
    return {
        "label": label,
        "value": value,
        "time_label": time_label,
        "time_value": float(time_value) / 1e6,
    }


def _format_time_with_milliseconds(local_datetime: datetime) -> str:
    """Format a local time string with millisecond precision."""
    return local_datetime.strftime("%H:%M:%S.%f")[:-3]


def _synthesize_pc_times(
    time_values_s: list[float],
    pc_reference_start_time: str,
) -> list[str]:
    """Build per-row local time strings from serial intervals.

    Args:
        time_values_s: Serial interval values with shape ``(n_rows,)`` in
            seconds.
        pc_reference_start_time: Local start time string, ``HH:MM:SS.mmm``.

    Returns:
        list[str]: Local time strings with shape ``(n_rows,)``.
    """
    start_time = datetime.strptime(pc_reference_start_time, "%H:%M:%S.%f")
    elapsed_s = np.zeros(len(time_values_s), dtype=float)
    if len(time_values_s) > 1:
        elapsed_s[1:] = np.cumsum(np.asarray(time_values_s[1:], dtype=float))
    return [
        _format_time_with_milliseconds(start_time + timedelta(seconds=float(elapsed)))
        for elapsed in elapsed_s
    ]


def parse_coolterm_serial_lines(
    lines: list[str],
    pc_timestamp_format: PCTimestampFormat = "datetime",
    pc_reference_date: str | None = None,
    pc_reference_start_time: str | None = None,
) -> pd.DataFrame:
    """Parse CoolTerm serial log lines into a row table.

    Args:
        lines: Text lines with shape ``(n_lines,)``.
        pc_timestamp_format: ``"datetime"`` for full local date/time prefixes,
            ``"millisecondtime"`` for local time-only prefixes, or ``None`` for
            serial-message-only lines.
        pc_reference_date: Local date string, ``YYYY-MM-DD``. Required for
            ``"millisecondtime"`` and ``None``.
        pc_reference_start_time: Optional local time string, ``HH:MM:SS.mmm``.
            Used only when ``pc_timestamp_format is None`` to synthesize PC
            times from serial intervals.

    Returns:
        pd.DataFrame: One row per parsed serial message. Columns are
        ``pc_date``, ``pc_time``, ``label``, ``value``, ``time_label``, and
        ``time_value``. ``time_value`` has units seconds.
    """

    if pc_timestamp_format not in ("datetime", "millisecondtime", None):
        raise ValueError("pc_timestamp_format must be None, 'datetime', or 'millisecondtime'.")
    reference_date = _require_reference_date(pc_timestamp_format, pc_reference_date)

    log_rows: list[dict[str, object]] = []
    previous_pc_datetime: datetime | None = None
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if pc_timestamp_format == "datetime":
            pc_timestamp, serial_msg = stripped_line.split("\t", maxsplit=1)
            pc_date, pc_time = pc_timestamp.split(" ", maxsplit=1)
        elif pc_timestamp_format == "millisecondtime":
            pc_time, serial_msg = stripped_line.split("\t", maxsplit=1)
            pc_date = reference_date
            pc_datetime = datetime.strptime(f"{pc_date} {pc_time}", "%Y-%m-%d %H:%M:%S.%f")
            if previous_pc_datetime is not None and pc_datetime < previous_pc_datetime:
                raise ValueError(
                    "PC timestamp decreased in millisecondtime log; midnight rollover is not supported yet."
                )
            previous_pc_datetime = pc_datetime
        else:
            serial_msg = stripped_line
            pc_date = reference_date
            pc_time = ""

        parsed_msg = _parse_serial_message(serial_msg)
        log_rows.append({"pc_date": pc_date, "pc_time": pc_time, **parsed_msg})

    if pc_timestamp_format is None and pc_reference_start_time is not None:
        synthesized_times = _synthesize_pc_times(
            [float(row["time_value"]) for row in log_rows],
            pc_reference_start_time=pc_reference_start_time,
        )
        for row, pc_time in zip(log_rows, synthesized_times):
            row["pc_time"] = pc_time

    return pd.DataFrame(log_rows)


def parse_coolterm_serial_text(
    text: str,
    pc_timestamp_format: PCTimestampFormat = "datetime",
    pc_reference_date: str | None = None,
    pc_reference_start_time: str | None = None,
) -> pd.DataFrame:
    """Parse CoolTerm serial log text into a row table.

    Args:
        text: Complete serial log text.
        pc_timestamp_format: ``"datetime"``, ``"millisecondtime"``, or
            ``None``.
        pc_reference_date: Local date string, ``YYYY-MM-DD``. Required for
            ``"millisecondtime"`` and ``None``.
        pc_reference_start_time: Optional local time string, ``HH:MM:SS.mmm``.

    Returns:
        pd.DataFrame: Parsed serial table from
        :func:`parse_coolterm_serial_lines`.
    """

    return parse_coolterm_serial_lines(
        lines=text.splitlines(),
        pc_timestamp_format=pc_timestamp_format,
        pc_reference_date=pc_reference_date,
        pc_reference_start_time=pc_reference_start_time,
    )


def load_coolterm_serial_txt(
    txt_path: Path | str,
    pc_timestamp_format: PCTimestampFormat = "datetime",
    pc_reference_date: str | None = None,
    pc_reference_start_time: str | None = None,
) -> pd.DataFrame:
    """Load and parse a CoolTerm serial text file.

    Args:
        txt_path: Path to a CoolTerm serial capture text file.
        pc_timestamp_format: ``"datetime"``, ``"millisecondtime"``, or
            ``None``.
        pc_reference_date: Local date string, ``YYYY-MM-DD``. Required for
            ``"millisecondtime"`` and ``None``.
        pc_reference_start_time: Optional local time string, ``HH:MM:SS.mmm``.

    Returns:
        pd.DataFrame: Parsed serial table from
        :func:`parse_coolterm_serial_lines`.
    """

    path = Path(txt_path).expanduser()
    with path.open("r") as file:
        lines = [line.strip() for line in file]
    return parse_coolterm_serial_lines(
        lines=lines,
        pc_timestamp_format=pc_timestamp_format,
        pc_reference_date=pc_reference_date,
        pc_reference_start_time=pc_reference_start_time,
    )


def treadmill_log_to_transition_df(
    parsed_df: pd.DataFrame,
    local_timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Convert parsed CoolTerm sync rows into desktop IRIG transition format.

    Args:
        parsed_df: Parsed serial log table. Required columns are ``label``,
            ``value``, ``time_value``, ``pc_date``, and ``pc_time``.
            ``time_value`` has shape ``(n_rows,)`` in seconds and represents
            the serial interval attached to each transition row.
        local_timezone: IANA timezone name for interpreting PC timestamp text.

    Returns:
        pd.DataFrame: Transition table with one row per sync transition and
        columns ``utc_seconds``, ``local_datetime``, ``pin_state``,
        ``pc_local_datetime``, ``pc_utc_seconds``, and
        ``pc_vs_reconstructed_error_s``. UTC values are in Unix seconds.
    """

    required_columns = {"label", "value", "time_value", "pc_date", "pc_time"}
    missing_columns = required_columns - set(parsed_df.columns)
    if missing_columns:
        raise ValueError(f"parsed_df is missing required columns: {sorted(missing_columns)}")
    if parsed_df["pc_time"].astype(str).str.len().eq(0).any():
        raise ValueError("pc_time is required to convert treadmill log rows into UTC transition times.")

    sync_df = parsed_df.loc[parsed_df["label"] == "syncPinState"].copy()
    if sync_df.empty:
        raise ValueError("No syncPinState rows found in parsed serial log.")

    while not sync_df.empty and sync_df.iloc[0]["value"] != "HIGH":
        sync_df = sync_df.iloc[1:].copy()
    while not sync_df.empty and sync_df.iloc[-1]["value"] != "LOW":
        sync_df = sync_df.iloc[:-1].copy()
    if sync_df.empty:
        raise ValueError("No complete HIGH/LOW transition sequence found.")
    repeated_state_mask = sync_df["value"].eq(sync_df["value"].shift())
    if repeated_state_mask.any():
        repeated_rows = sync_df.index[repeated_state_mask].to_list()
        repeated_positions = np.flatnonzero(repeated_state_mask.to_numpy(dtype=bool))
        time_value_col = sync_df.columns.get_loc("time_value")
        for repeated_position in repeated_positions:
            next_position = repeated_position + 1
            if next_position < sync_df.shape[0]:
                sync_df.iat[next_position, time_value_col] = (
                    float(sync_df.iat[next_position, time_value_col])
                    + float(sync_df.iat[repeated_position, time_value_col])
                )
        warnings.warn(
            "Repeated syncPinState values found at parsed rows "
            f"{repeated_rows}; dropping repeated rows and keeping the first state occurrence.",
            UserWarning,
            stacklevel=2,
        )
        sync_df = sync_df.loc[~repeated_state_mask].copy()
        while not sync_df.empty and sync_df.iloc[-1]["value"] != "LOW":
            sync_df = sync_df.iloc[:-1].copy()
        if sync_df.empty:
            raise ValueError("No complete HIGH/LOW transition sequence found after dropping repeats.")

    pc_local_datetime = pd.to_datetime(sync_df["pc_date"] + " " + sync_df["pc_time"])
    pc_local_datetime = pc_local_datetime.dt.tz_localize(local_timezone)
    pc_utc_seconds = pc_local_datetime.map(lambda timestamp: timestamp.timestamp())

    serial_intervals_s = sync_df["time_value"].to_numpy(dtype=float)
    transition_elapsed_s = np.zeros(sync_df.shape[0], dtype=float)
    if sync_df.shape[0] > 1:
        transition_elapsed_s[1:] = np.cumsum(serial_intervals_s[1:])

    reconstructed_utc_seconds = float(pc_utc_seconds.iloc[0]) + transition_elapsed_s
    reconstructed_local_datetime = pd.to_datetime(
        reconstructed_utc_seconds,
        unit="s",
        utc=True,
    ).tz_convert(local_timezone)

    return pd.DataFrame(
        {
            "utc_seconds": reconstructed_utc_seconds,
            "local_datetime": reconstructed_local_datetime.astype(str),
            "pin_state": (sync_df["value"].to_numpy() == "HIGH").astype(int),
            "pc_local_datetime": pc_local_datetime.astype(str).to_numpy(),
            "pc_utc_seconds": pc_utc_seconds.to_numpy(dtype=float),
            "pc_vs_reconstructed_error_s": (
                pc_utc_seconds.to_numpy(dtype=float) - reconstructed_utc_seconds
            ),
        }
    )
