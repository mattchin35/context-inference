"""Synchronize IMEC spikes and NI digital events to UTC using IRIG-H."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.neural_analysis.irig_sync_utils import (
    assign_utc_to_irig_bits,
    classify_irig_h_pulses,
    decode_irig_h_frame_anchors,
    decode_sync_line_to_irig_utc,
    find_signal_edges,
    map_digital_rising_edges_to_utc as map_daq_rising_edges_to_utc,
    map_sample_indices_to_utc,
    pulse_lengths_from_edges,
)
from src.neural_analysis.spikeglx_sync_io import read_digital_line as read_spikeglx_digital_line


NEW_YORK_TZ = ZoneInfo("America/New_York")


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


def apply_utc_hour_offset(irig_df: pd.DataFrame, utc_offset_hours: float) -> pd.DataFrame:
    """Shift decoded UTC timestamps by a user-specified number of hours.

    Args:
        irig_df: IRIG dataframe with one row per decoded pulse onset. Required
            columns are ``utc_unix`` and ``utc_datetime``. ``utc_unix`` has
            shape ``(n_pulses,)`` in seconds and ``utc_datetime`` contains
            timezone-aware UTC datetimes or ``NaT`` values.
        utc_offset_hours: Constant offset added to all decoded timestamps, in
            hours.

    Returns:
        pd.DataFrame: Copy of ``irig_df`` with the same row count and metadata,
        but with ``utc_unix`` shifted by ``utc_offset_hours * 3600`` seconds
        and ``utc_datetime`` rebuilt from the shifted UTC unix values.
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
    """Write a simple note describing the temporary UTC hour adjustment.

    Args:
        output_root: Root aligned-output directory where the note should be
            written.
        utc_offset_hours: Constant offset added to decoded timestamps, in
            hours.

    Returns:
        Path: Path to the two-line text note written inside ``output_root``.
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


def convert_utc_series_to_new_york(localizable_times: list[datetime | pd.Timestamp | Any]) -> list[datetime | pd.NaT]:
    """Convert UTC datetimes into New York local datetimes.

    Args:
        localizable_times: Sequence of timezone-aware UTC datetimes or ``NaT``-like
            values with shape ``(n_times,)``.

    Returns:
        list[datetime | pd.NaT]: Sequence with the same length where finite UTC
        datetimes are converted to ``America/New_York`` and missing values remain
        ``pd.NaT``.
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
) -> tuple[pd.DataFrame, float]:
    """Decode IRIG-H UTC timestamps directly from one SpikeGLX digital line.

    Args:
        binary_file: Path to a SpikeGLX ``.bin`` file.
        digital_word: Digital word index used by ``ExtractDigital``.
        irig_line: Digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours. This is a temporary workaround for known alignment issues.

    Returns:
        tuple[pd.DataFrame, float]:
            - IRIG rising-edge dataframe from
              :func:`decode_sync_line_to_irig_utc`.
            - Sampling rate in Hz.
    """
    irig_signal, sample_rate_hz = read_spikeglx_digital_line(
        binary_file=binary_file,
        digital_word=digital_word,
        digital_line=irig_line,
    )
    irig_df = decode_sync_line_to_irig_utc(
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
        imec_irig_df: DataFrame from :func:`decode_sync_line_to_irig_utc` for
            the corresponding IMEC stream.

    Returns:
        pd.DataFrame: UTC mapping with columns:
            - ``sample_ix``: spike sample index, units samples
            - ``utc_unix``: mapped UTC unix time, units seconds
            - ``utc_datetime``: timezone-aware UTC datetime
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
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours. This is a temporary workaround for known alignment issues.

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
        utc_offset_hours=utc_offset_hours,
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
            hours. This is a temporary workaround for known alignment issues.

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


def decode_ni_irig_debug(
    ni_file: Path | str,
    digital_word: int = 0,
    irig_line: int = 0,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode one NI IRIG line into pulse- and frame-level debug tables.

    Args:
        ni_file: NI ``.nidq.bin`` file path.
        digital_word: NI digital word index.
        irig_line: NI digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Pulse dataframe with one row per paired IRIG pulse onset. Columns
              include sample indices, pulse widths, classified bit labels, and
              decoded UTC values.
            - Frame dataframe with one row per decoded IRIG frame start. Columns
              include frame index, frame-start pulse/sample indices, and decoded
              UTC values.
    """
    ni_file = _require_existing_file(Path(ni_file))
    irig_signal, sample_rate_hz = read_spikeglx_digital_line(
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


def main_ni_only(
    ni_file: Path | str = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/test_runs/irig_neurokairos_20260408/run0_g0/run0_g0_t0.nidq.bin"
    ),
    digital_word: int = 0,
    irig_line: int = 0,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a lightweight NI-only IRIG decode for signal debugging.

    Args:
        ni_file: NI ``.nidq.bin`` file path.
        digital_word: NI digital word index.
        irig_line: NI digital line index carrying IRIG-H.
        bit_period_s: IRIG bit period in seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Pulse-level debug dataframe.
            - Frame-level debug dataframe.
    """
    ni_file = _require_existing_file(Path(ni_file))
    pulse_df, frame_df = decode_ni_irig_debug(
        ni_file=ni_file,
        digital_word=digital_word,
        irig_line=irig_line,
        bit_period_s=bit_period_s,
        utc_offset_hours=utc_offset_hours,
    )

    bit_counts = pulse_df["irig_bit"].value_counts(dropna=False).to_dict()
    print(f"Decoded {pulse_df.shape[0]} IRIG pulses from {ni_file}")
    print(f"Decoded {frame_df.shape[0]} IRIG frames")
    print(f"Bit counts: {bit_counts}")
    if not pulse_df.empty:
        print(pulse_df.head(5).to_string(index=False))
    if not frame_df.empty:
        print(frame_df.head(5).to_string(index=False))
    return pulse_df, frame_df


def main_workflow() -> None:
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251211_latentInference")
    sess_id_full: str = "CT014_2025-12-11_134311"
    output_root = session_data_home / "ephys" / "aligned"
    sorting0_output_name = 'Kilosort2.5.2_2026-04-27_142633'
    sorting1_output_name = 'Kilosort2.5.2_2026-04-27_145457'
    ni_event_lines = (2, 3)
    # Temporary workaround: set this to a nonzero value to shift all decoded
    # IMEC and NI timestamps by a constant number of hours.
    utc_offset_hours = 1

    raw_ephys_folder = session_data_home / "ephys" / "raw" / "run0_g0"
    catgt_ephys_folder = session_data_home / "ephys" / "catgt" / "catgt_run0_g0"
    imec0_folder = catgt_ephys_folder / "run0_g0_imec0"
    ap0_file = _require_existing_file(imec0_folder / "run0_g0_tcat.imec0.ap.bin")
    sorting_output0 = imec0_folder / sorting0_output_name
    spike_times_file0 = _require_existing_file(sorting_output0 / "spike_times.npy")

    imec1_folder = catgt_ephys_folder / "run0_g0_imec1"
    sorting_output1 = imec1_folder / sorting1_output_name
    ap1_file = _require_existing_file(imec1_folder / "run0_g0_tcat.imec1.ap.bin")
    spike_times_file1 = _require_existing_file(sorting_output1 / "spike_times.npy")

    ni_file = _require_existing_file(raw_ephys_folder / "run0_g0_t0.nidq.bin")
    imec_output_dir = output_root / "aligned_imec"
    nidaq_output_dir = output_root / "aligned_nidaq"
    write_alignment_note(output_root=output_root, utc_offset_hours=utc_offset_hours)

    spike_df, imec_irig_df = sync_imec_spikes_to_utc(
        imec_ap_file=ap0_file,
        spike_times_npy=spike_times_file0,
        output_file=imec_output_dir / "imec0_sync.npz",
        digital_word=0,
        irig_line=6,
        utc_offset_hours=utc_offset_hours,
    )
    print(
        f"{sess_id_full} imec0: mapped {spike_df.shape[0]} spikes using "
        f"{imec_irig_df.shape[0]} IRIG rising edges."
    )

    spike_df, imec_irig_df = sync_imec_spikes_to_utc(
        imec_ap_file=ap1_file,
        spike_times_npy=spike_times_file1,
        output_file=imec_output_dir / "imec1_sync.npz",
        digital_word=0,
        irig_line=6,
        utc_offset_hours=utc_offset_hours,
    )
    print(
        f"{sess_id_full} imec1: mapped {spike_df.shape[0]} spikes using "
        f"{imec_irig_df.shape[0]} IRIG rising edges."
    )

    for event_line in ni_event_lines:
        rising_df, ni_irig_df = sync_ni_rising_edges_to_utc(
            ni_file=ni_file,
            event_line=int(event_line),
            output_file=nidaq_output_dir / f"line{int(event_line)}_sync.npz",
            digital_word=0,
            irig_line=0,
            utc_offset_hours=utc_offset_hours,
        )
        print(
            f"{sess_id_full} nidaq line {int(event_line)}: mapped {rising_df.shape[0]} rising edges using "
            f"{ni_irig_df.shape[0]} IRIG rising edges."
        )


if __name__ == "__main__":
    main_workflow()
    #main_ni_only()
