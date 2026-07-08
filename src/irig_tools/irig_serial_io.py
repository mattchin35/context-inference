"""Serial-log parsing helpers for IRIG-bearing treadmill output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_coolterm_serial_lines(
    lines: list[str],
    pc_timestamp_present: bool = True,
) -> pd.DataFrame:
    """Parse CoolTerm serial log lines into a row table.

    Args:
        lines: Text lines with shape ``(n_lines,)``. When
            ``pc_timestamp_present`` is ``True``, each non-empty line must have
            ``"<pc date> <pc time>\\t<label>:<value>;<time label>:<microseconds>"``.
            When ``False``, the timestamp prefix is omitted.
        pc_timestamp_present: Whether each line begins with a PC timestamp.

    Returns:
        pd.DataFrame: One row per parsed serial message. Columns are
        ``pc_date``, ``pc_time``, ``label``, ``value``, ``time_label``, and
        ``time_value``. ``time_value`` has units seconds.
    """

    log_rows: list[dict[str, object]] = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if pc_timestamp_present:
            pc_timestamp, serial_msg = stripped_line.split("\t", maxsplit=1)
            pc_date, pc_time = pc_timestamp.split(" ", maxsplit=1)
        else:
            serial_msg = stripped_line
            pc_date = ""
            pc_time = ""

        fields = serial_msg.split(";")
        label, value = fields[0].split(":", maxsplit=1)
        time_label, time_value = fields[1].split(":", maxsplit=1)
        log_rows.append(
            {
                "pc_date": pc_date,
                "pc_time": pc_time,
                "label": label,
                "value": value,
                "time_label": time_label,
                "time_value": float(time_value) / 1e6,
            }
        )

    return pd.DataFrame(log_rows)


def parse_coolterm_serial_text(
    text: str,
    pc_timestamp_present: bool = True,
) -> pd.DataFrame:
    """Parse CoolTerm serial log text into a row table.

    Args:
        text: Complete serial log text.
        pc_timestamp_present: Whether each line begins with a PC timestamp.

    Returns:
        pd.DataFrame: Parsed serial table from
        :func:`parse_coolterm_serial_lines`.
    """

    return parse_coolterm_serial_lines(
        lines=text.splitlines(),
        pc_timestamp_present=pc_timestamp_present,
    )


def load_coolterm_serial_txt(
    txt_path: Path | str,
    pc_timestamp_present: bool = True,
) -> pd.DataFrame:
    """Load and parse a CoolTerm serial text file.

    Args:
        txt_path: Path to a CoolTerm serial capture text file.
        pc_timestamp_present: Whether each line begins with a PC timestamp.

    Returns:
        pd.DataFrame: Parsed serial table from
        :func:`parse_coolterm_serial_lines`.
    """

    path = Path(txt_path).expanduser()
    with path.open("r") as file:
        lines = [line.strip() for line in file]
    return parse_coolterm_serial_lines(
        lines=lines,
        pc_timestamp_present=pc_timestamp_present,
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

    sync_df = parsed_df.loc[parsed_df["label"] == "syncPinState"].copy()
    if sync_df.empty:
        raise ValueError("No syncPinState rows found in parsed serial log.")

    while not sync_df.empty and sync_df.iloc[0]["value"] != "HIGH":
        sync_df = sync_df.iloc[1:].copy()
    while not sync_df.empty and sync_df.iloc[-1]["value"] != "LOW":
        sync_df = sync_df.iloc[:-1].copy()
    if sync_df.empty:
        raise ValueError("No complete HIGH/LOW transition sequence found.")

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
