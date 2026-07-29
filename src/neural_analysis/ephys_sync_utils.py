"""SpikeGLX/ephys synchronization helpers built on IRIG-derived UTC mapping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
import warnings
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
from src.irig_tools.open_ephys_irig import decode_open_ephys_ttl_irig_utc
from src.neural_analysis.spikeglx_sync_io import read_digital_line as read_spikeglx_digital_line


NEW_YORK_TZ = ZoneInfo("America/New_York")
ReadDigitalLineFn = Callable[[Path | str, int, int], tuple[np.ndarray, float]]
DecodeSyncLineFn = Callable[..., pd.DataFrame]
DecodeOpenEphysTTLIRIGFn = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class OpenEphysContinuousSampleBounds:
    """Sample-number bounds for one Open Ephys continuous stream.

    Attributes:
        first_sample_ix: First Open Ephys global sample number in the
            continuous stream, in samples.
        last_sample_ix: Last Open Ephys global sample number in the continuous
            stream, in samples.
        n_samples: Number of samples in the continuous stream.
    """

    first_sample_ix: int
    last_sample_ix: int
    n_samples: int


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


def load_open_ephys_continuous_sample_bounds(
    continuous_dir: Path | str,
) -> OpenEphysContinuousSampleBounds:
    """Load sample-number bounds for one Open Ephys continuous stream.

    Args:
        continuous_dir: Open Ephys continuous-stream directory containing
            ``sample_numbers.npy`` with shape ``(n_samples,)``. Values are Open
            Ephys global sample numbers in samples.

    Returns:
        OpenEphysContinuousSampleBounds: First sample, last sample, and sample
        count for the stream. All sample indices are in Open Ephys global
        sample-number coordinates.
    """
    sample_numbers_path = Path(continuous_dir) / "sample_numbers.npy"
    if not sample_numbers_path.exists():
        raise FileNotFoundError(f"Open Ephys continuous sample_numbers.npy not found: {sample_numbers_path}")

    sample_numbers = np.load(sample_numbers_path, allow_pickle=False, mmap_mode="r").reshape(-1)
    if sample_numbers.size == 0:
        raise ValueError(f"Open Ephys continuous sample_numbers.npy is empty: {sample_numbers_path}")

    return OpenEphysContinuousSampleBounds(
        first_sample_ix=int(sample_numbers[0]),
        last_sample_ix=int(sample_numbers[-1]),
        n_samples=int(sample_numbers.size),
    )


def convert_kilosort_samples_to_open_ephys_samples(
    spike_sample_ix: npt.ArrayLike,
    continuous_start_sample_ix: int,
) -> np.ndarray:
    """Convert Kilosort-relative spike samples to Open Ephys global samples.

    Args:
        spike_sample_ix: Kilosort spike sample indices with shape
            ``(n_spikes,)`` in samples relative to the sorted continuous data.
        continuous_start_sample_ix: First Open Ephys global sample number for
            that continuous stream, in samples.

    Returns:
        np.ndarray: Open Ephys global spike sample numbers with shape
        ``(n_spikes,)`` in samples.
    """
    spike_sample_ix = np.asarray(spike_sample_ix, dtype=np.int64).reshape(-1)
    return spike_sample_ix + int(continuous_start_sample_ix)


def warn_if_samples_extrapolated(
    sample_ix: npt.ArrayLike,
    irig_df: pd.DataFrame,
    label: str = "samples",
) -> dict[str, int | float | str]:
    """Warn when samples fall outside finite IRIG UTC anchor boundaries.

    Args:
        sample_ix: Query sample indices with shape ``(n_samples,)`` in the same
            sample-number coordinates as ``irig_df["sample_ix"]``.
        irig_df: IRIG dataframe with shape ``(n_pulses, n_columns)``. Required
            columns are ``sample_ix`` in samples and ``utc_unix`` in seconds.
        label: Human-readable label for warning messages and metadata.

    Returns:
        dict[str, int | float | str]: Extrapolation metadata containing counts
        before and after the finite IRIG anchor range plus the first/last safe
        sample and UTC boundaries.
    """
    sample_ix = np.asarray(sample_ix, dtype=np.int64).reshape(-1)
    known = irig_df.loc[np.isfinite(irig_df["utc_unix"]), ["sample_ix", "utc_unix"]].copy()
    known = known.drop_duplicates(subset="sample_ix").sort_values("sample_ix")
    if known.shape[0] < 2:
        raise ValueError("Need at least two finite IRIG UTC points to evaluate extrapolation boundaries.")

    first_safe_sample_ix = int(known["sample_ix"].iloc[0])
    last_safe_sample_ix = int(known["sample_ix"].iloc[-1])
    first_safe_utc_unix = float(known["utc_unix"].iloc[0])
    last_safe_utc_unix = float(known["utc_unix"].iloc[-1])
    n_before = int(np.count_nonzero(sample_ix < first_safe_sample_ix))
    n_after = int(np.count_nonzero(sample_ix > last_safe_sample_ix))

    first_safe_utc = datetime.fromtimestamp(first_safe_utc_unix, tz=timezone.utc).isoformat()
    last_safe_utc = datetime.fromtimestamp(last_safe_utc_unix, tz=timezone.utc).isoformat()
    if n_before > 0 or n_after > 0:
        warnings.warn(
            (
                f"{label}: {n_before} samples before and {n_after} samples after are "
                "outside finite IRIG UTC anchor range and will be linearly extrapolated. "
                f"Safe sample range is {first_safe_sample_ix} to {last_safe_sample_ix}; "
                f"safe UTC range is {first_safe_utc} to {last_safe_utc}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "label": str(label),
        "n_extrapolated_before_irig": n_before,
        "n_extrapolated_after_irig": n_after,
        "first_safe_sample_ix": first_safe_sample_ix,
        "last_safe_sample_ix": last_safe_sample_ix,
        "first_safe_utc_unix": first_safe_utc_unix,
        "last_safe_utc_unix": last_safe_utc_unix,
        "first_safe_utc_datetime": first_safe_utc,
        "last_safe_utc_datetime": last_safe_utc,
    }


def sync_open_ephys_kilosort_spikes_to_utc(
    kilosort_dir: Path | str,
    ttl_dir: Path | str,
    continuous_dir: Path | str,
    output_file: Optional[Path | str] = None,
    probe_name: str = "",
    line: int = 0,
    bit_period_s: float = 1.0,
    irig_format: str = "neurokairos",
    sample_rate_hz: float | None = None,
    min_high_duration_s: float = 0.01,
    min_low_duration_s: float = 0.01,
    utc_offset_hours: float = 0.0,
    warn_invalid: bool = True,
    decode_open_ephys_ttl_irig_utc_fn: DecodeOpenEphysTTLIRIGFn = decode_open_ephys_ttl_irig_utc,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode Open Ephys TTL IRIG and assign UTC timestamps to Kilosort spikes.

    Args:
        kilosort_dir: Kilosort output directory containing ``spike_times.npy``.
        ttl_dir: Open Ephys TTL event directory containing IRIG transitions.
        continuous_dir: Open Ephys continuous-stream directory containing
            ``sample_numbers.npy``. Kilosort spike times are assumed to be
            relative to this stream's first sample.
        output_file: Optional ``.npz`` output path for the synced probe.
        probe_name: Optional probe label, for example ``"ProbeA"``.
        line: Zero-based TTL line carrying IRIG-H.
        bit_period_s: IRIG-H bit period in seconds.
        irig_format: IRIG frame format passed to the Open Ephys TTL decoder.
        sample_rate_hz: Optional sample rate in Hz, used only if TTL
            ``timestamps.npy`` is absent.
        min_high_duration_s: Debounce threshold for short high glitches, in
            seconds.
        min_low_duration_s: Debounce threshold for short low glitches, in
            seconds.
        utc_offset_hours: Constant offset added to decoded UTC timestamps, in
            hours.
        warn_invalid: If ``True``, warn about unclassified IRIG pulses.
        decode_open_ephys_ttl_irig_utc_fn: Function that decodes Open Ephys TTL
            IRIG into the common IRIG UTC dataframe schema.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - Spike UTC dataframe with Kilosort-relative ``sample_ix`` in
              samples, ``open_ephys_sample_ix`` in Open Ephys global samples,
              ``utc_unix`` in seconds, and UTC datetimes.
            - Open Ephys IRIG rising-edge dataframe in global sample-number
              coordinates.
    """
    kilosort_path = Path(kilosort_dir)
    ttl_path = Path(ttl_dir)
    continuous_path = Path(continuous_dir)
    spike_times_path = kilosort_path / "spike_times.npy"
    if not spike_times_path.exists():
        raise FileNotFoundError(f"Kilosort spike_times.npy not found: {spike_times_path}")

    continuous_bounds = load_open_ephys_continuous_sample_bounds(continuous_path)
    irig_df = decode_open_ephys_ttl_irig_utc_fn(
        ttl_dir=ttl_path,
        line=line,
        bit_period_s=bit_period_s,
        irig_format=irig_format,
        sample_rate_hz=sample_rate_hz,
        min_high_duration_s=min_high_duration_s,
        min_low_duration_s=min_low_duration_s,
        warn_invalid=warn_invalid,
    )
    irig_df = apply_utc_hour_offset(irig_df, utc_offset_hours=utc_offset_hours)

    spike_sample_ix = np.asarray(np.load(spike_times_path, allow_pickle=True), dtype=np.int64).reshape(-1)
    open_ephys_spike_sample_ix = convert_kilosort_samples_to_open_ephys_samples(
        spike_sample_ix=spike_sample_ix,
        continuous_start_sample_ix=continuous_bounds.first_sample_ix,
    )
    extrapolation_meta = warn_if_samples_extrapolated(
        sample_ix=open_ephys_spike_sample_ix,
        irig_df=irig_df,
        label=f"{probe_name or 'Open Ephys'} spikes",
    )

    mapped_df = map_sample_indices_to_utc(open_ephys_spike_sample_ix, irig_df)
    spike_df = mapped_df.rename(columns={"sample_ix": "open_ephys_sample_ix"})
    spike_df.insert(0, "sample_ix", spike_sample_ix.astype(np.int64))

    if output_file is not None:
        meta = {
            "generator": "sync_open_ephys_kilosort_spikes_to_utc",
            "analysis_version": "0.3.0",
            "kilosort_dir": str(kilosort_path),
            "spike_times_file": str(spike_times_path),
            "ttl_dir": str(ttl_path),
            "continuous_dir": str(continuous_path),
            "probe_name": str(probe_name),
            "continuous_start_sample_ix": int(continuous_bounds.first_sample_ix),
            "continuous_last_sample_ix": int(continuous_bounds.last_sample_ix),
            "continuous_n_samples": int(continuous_bounds.n_samples),
            "line": int(line),
            "bit_period_s": float(bit_period_s),
            "irig_format": str(irig_format),
            "sample_rate_hz": None if sample_rate_hz is None else float(sample_rate_hz),
            "min_high_duration_s": float(min_high_duration_s),
            "min_low_duration_s": float(min_low_duration_s),
            "utc_offset_hours": float(utc_offset_hours),
            "extrapolation": extrapolation_meta,
            "units": {
                "spike_sample_ix": "samples relative to Kilosort continuous data",
                "spike_open_ephys_sample_ix": "Open Ephys global samples",
                "irig_sample_ix": "Open Ephys global samples",
                "irig_recording_time_s": "seconds",
                "irig_pulse_len_samples": "samples",
                "utc_unix": "seconds",
            },
            "axis_convention": "1D sample index along acquisition time",
        }
        save_stream_sync_npz(
            output_file=output_file,
            arrays={
                "spike_sample_ix": spike_df["sample_ix"].to_numpy(dtype=np.int64),
                "spike_open_ephys_sample_ix": spike_df["open_ephys_sample_ix"].to_numpy(dtype=np.int64),
                "spike_utc_unix": spike_df["utc_unix"].to_numpy(dtype=float),
                "irig_sample_ix": irig_df["sample_ix"].to_numpy(dtype=np.int64),
                "irig_recording_time_s": irig_df["recording_time_s"].to_numpy(dtype=float),
                "irig_pulse_len_samples": irig_df["pulse_len_samples"].to_numpy(dtype=np.int64),
                "irig_bit": irig_df["irig_bit"].to_numpy(dtype=object),
                "irig_utc_unix": irig_df["utc_unix"].to_numpy(dtype=float),
            },
            meta=meta,
        )

    return spike_df, irig_df


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
