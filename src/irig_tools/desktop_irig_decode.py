"""Decode NeuroKairos desktop IRIG transition CSVs into pulse and frame tables."""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


SECONDS_WEIGHTS = [1, 2, 4, 8, 10, 20, 40]
MINUTES_WEIGHTS = [1, 2, 4, 8, 10, 20, 40]
HOURS_WEIGHTS = [1, 2, 4, 8, 10, 20]
DAY_OF_YEAR_WEIGHTS = [1, 2, 4, 8, 10, 20, 40, 80, 100, 200]
YEARS_WEIGHTS = [1, 2, 4, 8, 10, 20, 40, 80]

MARKER_POSITIONS = {0, 9, 19, 29, 39, 49, 59}
BIT_PERIOD_SECONDS = 1.0


def bcd_decode(bits: list[object], weights: list[int]) -> int:
    """Decode a BCD-style weighted bit slice into an integer."""

    if len(bits) != len(weights):
        raise ValueError("Bit and weight lists must be the same length.")

    total = 0
    for bit, weight in zip(bits, weights):
        if bit not in (False, True, 0, 1):
            raise ValueError(f"Cannot BCD-decode non-binary bit value: {bit!r}")
        if bool(bit):
            total += weight
    return total


def load_irig_transition_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a transition CSV emitted by ``irig_logic_neurokairos.c``."""

    transition_df = pd.read_csv(csv_path)
    required_columns = {"utc_seconds", "local_datetime", "pin_state"}
    missing_columns = required_columns - set(transition_df.columns)
    if missing_columns:
        raise ValueError(f"Transition CSV is missing required columns: {sorted(missing_columns)}")

    transition_df = transition_df.loc[:, ["utc_seconds", "local_datetime", "pin_state"]].copy()
    transition_df["utc_seconds"] = pd.to_numeric(transition_df["utc_seconds"], errors="coerce")
    transition_df["pin_state"] = pd.to_numeric(transition_df["pin_state"], errors="coerce")
    if transition_df["utc_seconds"].isna().any():
        raise ValueError("utc_seconds must be numeric for every transition row.")
    if transition_df["pin_state"].isna().any():
        raise ValueError("pin_state must be numeric 0/1 for every transition row.")

    transition_df["pin_state"] = transition_df["pin_state"].astype(int)
    if not transition_df["pin_state"].isin([0, 1]).all():
        raise ValueError("pin_state must contain only 0 and 1 values.")

    return transition_df.sort_values("utc_seconds").reset_index(drop=True)


def load_encoded_timestamp_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load the optional encoded/sending-start CSV emitted by the simulator."""

    timestamp_df = pd.read_csv(csv_path)
    required_columns = {"Encoded times", "Sending starts"}
    missing_columns = required_columns - set(timestamp_df.columns)
    if missing_columns:
        raise ValueError(f"Timestamp CSV is missing required columns: {sorted(missing_columns)}")

    timestamp_df = timestamp_df.loc[:, ["Encoded times", "Sending starts"]].copy()
    timestamp_df["Encoded times"] = pd.to_numeric(timestamp_df["Encoded times"], errors="coerce")
    timestamp_df["Sending starts"] = pd.to_numeric(timestamp_df["Sending starts"], errors="coerce")
    if timestamp_df["Encoded times"].isna().any():
        raise ValueError("Encoded times must be numeric for every row.")
    if timestamp_df["Sending starts"].isna().any():
        raise ValueError("Sending starts must be numeric for every row.")

    return timestamp_df.reset_index(drop=True)


def infer_timestamp_csv_path(transition_csv_path: Path) -> Path | None:
    """Infer the matching timestamp CSV path from a transition CSV path."""

    transition_name = transition_csv_path.name
    if not transition_name.startswith("irig_logic_neurokairos_transitions_"):
        return None

    suffix = transition_name.removeprefix("irig_logic_neurokairos_transitions_")
    candidate = transition_csv_path.with_name(f"irig_logic_neurokairos_output_timestamps_{suffix}")
    if candidate.exists():
        return candidate
    return None


def find_latest_transition_csv(search_dir: Path) -> Path:
    """Find the most recent NeuroKairos transition CSV in ``search_dir``."""

    candidates = sorted(
        search_dir.glob("irig_logic_neurokairos_transitions_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No irig_logic_neurokairos_transitions_*.csv files found in {search_dir}"
        )
    return candidates[0]


def extract_irig_pulses_from_transitions(transitions_df: pd.DataFrame) -> pd.DataFrame:
    """Pair rising and falling transitions into pulse measurements."""

    required_columns = {"utc_seconds", "local_datetime", "pin_state"}
    missing_columns = required_columns - set(transitions_df.columns)
    if missing_columns:
        raise ValueError(f"transitions_df is missing required columns: {sorted(missing_columns)}")

    ordered_df = transitions_df.sort_values("utc_seconds").reset_index(drop=True).copy()
    if ordered_df.empty:
        return pd.DataFrame(
            columns=[
                "rising_edge_ix",
                "rising_edge_utc_seconds",
                "falling_edge_utc_seconds",
                "pulse_width_s",
                "rising_edge_local_datetime",
            ]
        )

    pin_state = ordered_df["pin_state"].to_numpy(dtype=int)
    if ordered_df.shape[0] % 2 != 0:
        raise ValueError("Transition rows must come in rising/falling pairs.")
    if pin_state[0] != 1:
        raise ValueError("Transition rows must start with a rising edge where pin_state becomes 1.")
    if pin_state[-1] != 0:
        raise ValueError("Transition rows must end with a falling edge where pin_state becomes 0.")
    
    expected_pattern = np.tile(np.array([1, 0], dtype=int), ordered_df.shape[0] // 2)
    if not np.array_equal(pin_state, expected_pattern):
        raise ValueError("Transition rows must alternate rising and falling edges.")

    rising_rows = ordered_df.iloc[0::2].reset_index(drop=True)
    falling_rows = ordered_df.iloc[1::2].reset_index(drop=True)
    pulse_width_s = (
        falling_rows["utc_seconds"].to_numpy(dtype=float) - rising_rows["utc_seconds"].to_numpy(dtype=float)
    )
    if np.any(pulse_width_s <= 0):
        raise ValueError("Every falling edge must occur after its paired rising edge.")

    return pd.DataFrame(
        {
            "rising_edge_ix": np.arange(rising_rows.shape[0], dtype=int),
            "rising_edge_utc_seconds": rising_rows["utc_seconds"].to_numpy(dtype=float),
            "falling_edge_utc_seconds": falling_rows["utc_seconds"].to_numpy(dtype=float),
            "pulse_width_s": pulse_width_s.astype(float),
            "rising_edge_local_datetime": rising_rows["local_datetime"].astype(str).to_numpy(),
        }
    )


def classify_irig_pulse_widths_seconds(
    pulse_width_s: np.ndarray | pd.Series,
    bit_period_s: float = BIT_PERIOD_SECONDS,
) -> np.ndarray:
    """Classify pulse widths into ``False``, ``True``, ``'P'``, or ``None``."""

    pulse_width_s = np.asarray(pulse_width_s, dtype=float).reshape(-1)
    p_thresh = 0.75 * float(bit_period_s)
    one_thresh = 0.45 * float(bit_period_s)
    zero_thresh = 0.05 * float(bit_period_s)

    bit_labels = np.full(pulse_width_s.shape, None, dtype=object)
    bit_labels[pulse_width_s > p_thresh] = "P"
    bit_labels[(pulse_width_s > one_thresh) & (pulse_width_s <= p_thresh)] = True
    bit_labels[(pulse_width_s > zero_thresh) & (pulse_width_s <= one_thresh)] = False
    return bit_labels


def find_irig_frame_spans(irig_bits: np.ndarray | pd.Series) -> list[tuple[int, int]]:
    """Locate consecutive 60-pulse IRIG frames from a classified bit sequence."""

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
    if frame_start is None or frame_start < 0:
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


def decode_stratum_code(raw_code: int) -> str:
    """Decode the custom 2-bit NeuroKairos stratum encoding."""

    if raw_code == 0:
        return "stratum_1"
    if raw_code == 1:
        return "stratum_2"
    if raw_code == 2:
        return "stratum_3"
    return "stratum_4_or_higher"


def describe_dispersion_code(raw_code: int) -> str:
    """Describe the custom 3-bit NeuroKairos root-dispersion bucket."""

    if raw_code == 0:
        return "<0.25 ms"
    if raw_code == 1:
        return "0.25-0.5 ms"
    if raw_code == 2:
        return "0.5-1 ms"
    if raw_code == 3:
        return "1-2 ms"
    if raw_code == 4:
        return "2-4 ms"
    if raw_code == 5:
        return "4-8 ms"
    if raw_code == 6:
        return "8-16 ms"
    return ">=16 ms"


def bit_symbol(bit: object) -> str:
    """Convert a classified bit to a compact printable symbol."""

    if bit == "P":
        return "P"
    if bit is True:
        return "1"
    if bit is False:
        return "0"
    return "?"


def decode_neurokairos_frame_fields(
    frame_bits: np.ndarray | list[object],
    century_base: int | None = None,
) -> dict[str, object]:
    """Decode one 60-pulse NeuroKairos frame as emitted by the C simulator."""

    frame_bits = list(np.asarray(frame_bits, dtype=object).reshape(-1))
    if len(frame_bits) != 60:
        raise ValueError("frame_bits must contain exactly 60 IRIG bits.")
    if any(frame_bits[pos] != "P" for pos in MARKER_POSITIONS):
        raise ValueError("frame_bits does not contain markers at the expected positions.")

    seconds = bcd_decode(frame_bits[1:5], SECONDS_WEIGHTS[0:4]) + bcd_decode(
        frame_bits[6:9], SECONDS_WEIGHTS[4:7]
    )
    minutes = bcd_decode(frame_bits[10:14], MINUTES_WEIGHTS[0:4]) + bcd_decode(
        frame_bits[15:18], MINUTES_WEIGHTS[4:7]
    )
    hours = bcd_decode(frame_bits[20:24], HOURS_WEIGHTS[0:4]) + bcd_decode(
        frame_bits[25:27], HOURS_WEIGHTS[4:6]
    )
    day_of_year = (
        bcd_decode(frame_bits[30:34], DAY_OF_YEAR_WEIGHTS[0:4])
        + bcd_decode(frame_bits[35:39], DAY_OF_YEAR_WEIGHTS[4:8])
        + bcd_decode(frame_bits[40:42], DAY_OF_YEAR_WEIGHTS[8:10])
    )
    year_two_digit = bcd_decode(frame_bits[50:54], YEARS_WEIGHTS[0:4]) + bcd_decode(
        frame_bits[55:59], YEARS_WEIGHTS[4:8]
    )

    raw_stratum_code = int(bool(frame_bits[43])) + (int(bool(frame_bits[44])) << 1)
    raw_dispersion_code = (
        int(bool(frame_bits[46]))
        + (int(bool(frame_bits[47])) << 1)
        + (int(bool(frame_bits[48])) << 2)
    )

    if century_base is None:
        century_base = (datetime.datetime.now(datetime.timezone.utc).year // 100) * 100
    decoded_year = century_base + year_two_digit

    decoded_datetime_utc = None
    decoded_datetime_utc_text = None
    decoded_utc_seconds = np.nan
    try:
        decoded_datetime_utc = datetime.datetime(
            decoded_year,
            1,
            1,
            hours,
            minutes,
            seconds,
            tzinfo=datetime.timezone.utc,
        ) + datetime.timedelta(days=day_of_year - 1)
        decoded_utc_seconds = decoded_datetime_utc.timestamp()
        decoded_datetime_utc_text = decoded_datetime_utc.isoformat()
    except ValueError:
        decoded_datetime_utc = None
        decoded_datetime_utc_text = None
        decoded_utc_seconds = np.nan

    return {
        "seconds": seconds,
        "minutes": minutes,
        "hours": hours,
        "day_of_year": day_of_year,
        "year_two_digit": year_two_digit,
        "decoded_year": decoded_year,
        "decoded_datetime_utc": decoded_datetime_utc_text,
        "decoded_utc_seconds": decoded_utc_seconds,
        "reserved_bit_42": int(bool(frame_bits[42])),
        "custom_stratum_code": raw_stratum_code,
        "decoded_stratum": decode_stratum_code(raw_stratum_code),
        "reserved_bit_45": int(bool(frame_bits[45])),
        "custom_dispersion_code": raw_dispersion_code,
        "decoded_dispersion_bucket": describe_dispersion_code(raw_dispersion_code),
        "bit_string": "".join(bit_symbol(bit) for bit in frame_bits),
    }


def build_pulse_debug_table(pulse_df: pd.DataFrame) -> pd.DataFrame:
    """Add classified bit labels to the pulse table for inspection."""

    pulse_debug_df = pulse_df.copy(deep=True)
    pulse_debug_df["irig_bit"] = classify_irig_pulse_widths_seconds(pulse_debug_df["pulse_width_s"])
    return pulse_debug_df


def build_frame_debug_table(
    pulse_df: pd.DataFrame,
    timestamps_df: pd.DataFrame | None = None,
    timezone_name: str = "America/New_York",
    century_base: int | None = None,
) -> pd.DataFrame:
    """Build a NeuroKairos frame-by-frame inspection table."""

    required_columns = {
        "rising_edge_ix",
        "rising_edge_utc_seconds",
        "rising_edge_local_datetime",
        "pulse_width_s",
    }
    missing_columns = required_columns - set(pulse_df.columns)
    if missing_columns:
        raise ValueError(f"pulse_df is missing required columns: {sorted(missing_columns)}")

    irig_bits = classify_irig_pulse_widths_seconds(pulse_df["pulse_width_s"])
    frame_spans = find_irig_frame_spans(irig_bits)
    local_zone = ZoneInfo(timezone_name)
    frame_rows: list[dict[str, object]] = []

    for frame_ix, (start_ix, end_ix) in enumerate(frame_spans):
        frame_fields = decode_neurokairos_frame_fields(irig_bits[start_ix:end_ix], century_base=century_base)
        observed_start_utc_seconds = float(pulse_df.iloc[start_ix]["rising_edge_utc_seconds"])
        observed_start_local_datetime = datetime.datetime.fromtimestamp(
            observed_start_utc_seconds, tz=datetime.timezone.utc
        ).astimezone(local_zone).isoformat()

        encoded_time_from_csv = np.nan
        sending_start_from_csv = np.nan
        if timestamps_df is not None and frame_ix < len(timestamps_df):
            encoded_time_from_csv = float(timestamps_df.iloc[frame_ix]["Encoded times"])
            sending_start_from_csv = float(timestamps_df.iloc[frame_ix]["Sending starts"])

        frame_rows.append(
            {
                "frame_ix": frame_ix,
                "frame_start_pulse_ix": int(start_ix),
                "frame_end_pulse_ix": int(end_ix),
                "observed_start_utc_seconds": observed_start_utc_seconds,
                "observed_start_local_datetime": observed_start_local_datetime,
                "csv_encoded_time": encoded_time_from_csv,
                "csv_sending_start": sending_start_from_csv,
                "decoded_vs_csv_encoded_error_s": (
                    frame_fields["decoded_utc_seconds"] - encoded_time_from_csv
                    if np.isfinite(frame_fields["decoded_utc_seconds"]) and np.isfinite(encoded_time_from_csv)
                    else np.nan
                ),
                "sending_start_vs_observed_start_error_s": (
                    sending_start_from_csv - observed_start_utc_seconds
                    if np.isfinite(sending_start_from_csv)
                    else np.nan
                ),
                **frame_fields,
            }
        )

    return pd.DataFrame(frame_rows)


def summarize_pulse_labels(pulse_df: pd.DataFrame) -> dict[str, int]:
    """Count classified pulse labels for a compact summary."""

    labels = classify_irig_pulse_widths_seconds(pulse_df["pulse_width_s"])
    return {
        "P": int(np.sum(labels == "P")),
        "1": int(np.sum(labels == True)),
        "0": int(np.sum(labels == False)),
        "unclassified": int(np.sum(labels == None)),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "transition_csv",
        nargs="?",
        help="Path to an irig_logic_neurokairos_transitions_*.csv file. Defaults to the newest one in this directory.",
    )
    parser.add_argument(
        "--timestamps-csv",
        help="Optional matching irig_logic_neurokairos_output_timestamps_*.csv file.",
    )
    parser.add_argument(
        "--timezone",
        default="America/New_York",
        help="IANA timezone name used for displayed local datetimes.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="How many decoded frames to print in the summary.",
    )
    parser.add_argument(
        "--century-base",
        type=int,
        default=None,
        help="Optional century base for 2-digit year decode, for example 2000.",
    )
    return parser.parse_args()


def run_logic_csv_decode(
    transition_csv_path: Path | str,
    timestamps_csv_path: Path | str | None = None,
    timezone_name: str = "America/New_York",
    frames: int = 10,
    century_base: int | None = None,
) -> None:
    """Run a NeuroKairos desktop IRIG decode and print a compact inspection summary."""

    transition_csv_path = Path(transition_csv_path).expanduser().resolve()
    if timestamps_csv_path is not None:
        timestamps_csv_path = Path(timestamps_csv_path).expanduser().resolve()

    transition_df = load_irig_transition_csv(transition_csv_path)
    pulse_df = extract_irig_pulses_from_transitions(transition_df)
    pulse_debug_df = build_pulse_debug_table(pulse_df)

    timestamps_df = None
    if timestamps_csv_path is not None and timestamps_csv_path.exists():
        timestamps_df = load_encoded_timestamp_csv(timestamps_csv_path)

    frame_debug_df = build_frame_debug_table(
        pulse_df,
        timestamps_df=timestamps_df,
        timezone_name=timezone_name,
        century_base=century_base,
    )

    print("NeuroKairos desktop IRIG decode summary")
    print(f"Transition CSV: {transition_csv_path}")
    print(f"Timestamp CSV: {timestamps_csv_path if timestamps_df is not None else 'not used'}")
    print(f"Transitions: {transition_df.shape[0]}")
    print(f"Pulses: {pulse_df.shape[0]}")
    print(f"Frames: {frame_debug_df.shape[0]}")

    if pulse_df.empty:
        print("No pulses found in the transition CSV.")
        return

    pulse_summary = summarize_pulse_labels(pulse_df)
    print(
        "Pulse labels: "
        f"P={pulse_summary['P']} 1={pulse_summary['1']} 0={pulse_summary['0']} "
        f"unclassified={pulse_summary['unclassified']}"
    )

    print("First pulses:")
    print(
        pulse_debug_df.loc[
            :,
            [
                "rising_edge_ix",
                "rising_edge_utc_seconds",
                "pulse_width_s",
                "irig_bit",
                "rising_edge_local_datetime",
            ],
        ]
        .head(10)
        .to_string(index=False)
    )

    print("Decoded frames:")
    if frame_debug_df.empty:
        print("No valid 60-pulse NeuroKairos frames found.")
        return

    print(
        frame_debug_df.loc[
            :,
            [
                "frame_ix",
                "observed_start_utc_seconds",
                "csv_encoded_time",
                "decoded_utc_seconds",
                "seconds",
                "minutes",
                "hours",
                "day_of_year",
                "decoded_year",
                "decoded_stratum",
                "decoded_dispersion_bucket",
                "decoded_vs_csv_encoded_error_s",
            ],
        ]
        .head(frames)
        .to_string(index=False)
    )

    print("First frame detail:")
    print(frame_debug_df.iloc[0].to_string())


def main_logic_csv() -> None:
    """Run the desktop IRIG decode with IDE-friendly hardcoded inputs."""

    transition_csv_path = Path(
        "/home/matt/Documents/matt_irig_mods/irig_logic_neurokairos_transitions_2026-04-03_17-51-00.csv"
    )
    timestamps_csv_path = None
    timezone_name = "America/New_York"
    frames = 10
    century_base = None

    if timestamps_csv_path is None:
        timestamps_csv_path = infer_timestamp_csv_path(transition_csv_path)

    run_logic_csv_decode(
        transition_csv_path=transition_csv_path,
        timestamps_csv_path=timestamps_csv_path,
        timezone_name=timezone_name,
        frames=frames,
        century_base=century_base,
    )


def main_cli() -> None:
    """Run the desktop IRIG decode from command-line arguments."""

    args = parse_args()
    repo_dir = Path(__file__).resolve().parent

    if args.transition_csv:
        transition_csv_path = Path(args.transition_csv).expanduser().resolve()
    else:
        transition_csv_path = find_latest_transition_csv(repo_dir)

    if args.timestamps_csv:
        timestamps_csv_path = Path(args.timestamps_csv).expanduser().resolve()
    else:
        timestamps_csv_path = infer_timestamp_csv_path(transition_csv_path)

    run_logic_csv_decode(
        transition_csv_path=transition_csv_path,
        timestamps_csv_path=timestamps_csv_path,
        timezone_name=args.timezone,
        frames=args.frames,
        century_base=args.century_base,
    )


if __name__ == "__main__":
    main_logic_csv()
    # main_cli()
