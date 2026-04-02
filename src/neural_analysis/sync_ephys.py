"""Synchronize IMEC spikes and NI digital events to UTC using IRIG-H."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
            "analysis_version": "0.2.0",
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
            "analysis_version": "0.2.0",
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


def main() -> None:
    """Run a brief example synchronization workflow for one IMEC stream and NI lines.

    Args:
        session_data_home: Session root directory containing ``ephys/raw`` and
            ``ephys/catgt`` subdirectories.
        sess_id: Session identifier string for printed summaries.
        sorting_output_name: Name of the sorting output directory containing
            ``spike_times.npy``.
        ni_event_lines: NI digital line indices to map to UTC from NI word 0.
        output_root: Root output directory for per-stream ``.npz`` files. If
            ``None``, defaults to ``session_data_home / 'ephys' / 'aligned'``.

    Returns:
        None: This example runner writes files and prints a short summary.
    """
    session_data_home = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251205_latentInference")
    sess_id_full: str = "CT014_2025-12-05"
    output_root = session_data_home / "ephys" / "aligned"
    sorting0_output_name = 'Kilosort2.5.2_2026-03-19_165539'
    sorting1_output_name = 'Kilosort2.5.2_2026-03-19_173016'
    ni_event_lines = (2, 3)

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

    spike_df, imec_irig_df = sync_imec_spikes_to_utc(
        imec_ap_file=ap0_file,
        spike_times_npy=spike_times_file0,
        output_file=imec_output_dir / "imec0_sync.npz",
        digital_word=0,
        irig_line=6,
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
        )
        print(
            f"{sess_id_full} nidaq line {int(event_line)}: mapped {rising_df.shape[0]} rising edges using "
            f"{ni_irig_df.shape[0]} IRIG rising edges."
        )


if __name__ == "__main__":
    main()
