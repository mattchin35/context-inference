from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap

import src.external_tools.readSGLX as readSGLX
from src.neural_analysis import ephys_sync_utils


def load_lfp_metadata(lfp_path: Path | str) -> dict:
    """
    Load SpikeGLX metadata for one LFP binary file.

    Parameters
    ----------
    lfp_path : Path | str
        Path to a SpikeGLX ``.lf.bin`` file. The matching ``.meta`` file must
        live next to it.

    Returns
    -------
    dict
        SpikeGLX metadata dictionary. Values are strings as returned by the
        local ``readSGLX.readMeta`` helper.
    """

    lfp_file = Path(lfp_path)
    meta = readSGLX.readMeta(lfp_file)
    if not meta:
        raise FileNotFoundError(f"Could not load SpikeGLX metadata for {lfp_file}.")
    return meta


def validate_lfp_saved_channel(meta: dict, saved_channel_index: int) -> None:
    """
    Validate a saved-channel row index for one LFP binary file.

    Parameters
    ----------
    meta : dict
        SpikeGLX metadata containing ``nSavedChans``.
    saved_channel_index : int
        Zero-based saved channel index into the binary file rows. This is not
        necessarily the original/acquired channel id.

    Returns
    -------
    None
        Raises ``ValueError`` if the saved channel index is outside the binary
        row range.
    """

    if "nSavedChans" not in meta:
        raise ValueError("LFP metadata is missing nSavedChans.")
    n_saved_channels = int(meta["nSavedChans"])
    if int(saved_channel_index) < 0:
        raise ValueError("LFP saved channel index must be nonnegative.")
    if int(saved_channel_index) >= n_saved_channels:
        raise ValueError(
            f"LFP saved channel index {saved_channel_index} must be less than nSavedChans={n_saved_channels}."
        )


def decode_lfp_sync(
    lfp_path: Path | str,
    digital_word: int = 0,
    irig_line: int = 6,
    bit_period_s: float = 1.0,
    utc_offset_hours: float = 0.0,
) -> tuple[pd.DataFrame, float]:
    """
    Decode IRIG timestamps from the digital sync line in an LFP file.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    digital_word : int, default=0
        Digital word index in the saved binary stream.
    irig_line : int, default=6
        Digital line carrying IRIG-H.
    bit_period_s : float, default=1.0
        IRIG-H bit period in seconds.
    utc_offset_hours : float, default=0.0
        Constant offset applied to decoded UTC timestamps.

    Returns
    -------
    tuple[pd.DataFrame, float]
        Decoded IRIG dataframe and LFP sample rate in Hz.
    """

    return ephys_sync_utils.decode_binary_file_irig_utc(
        binary_file=Path(lfp_path),
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
    )


def map_lfp_time_window_to_samples(
    alignment_time_s: float,
    window: tuple[float, float],
    lfp_irig_df: pd.DataFrame,
    sample_rate_hz: float,
) -> tuple[np.ndarray, int, int]:
    """
    Convert a behavior-aligned time window to LFP sample indices.

    Parameters
    ----------
    alignment_time_s : float
        Absolute alignment timestamp in seconds, using the same UTC-like time
        base as decoded LFP IRIG timestamps.
    window : tuple[float, float]
        Relative window bounds in seconds around ``alignment_time_s``.
    lfp_irig_df : pd.DataFrame
        IRIG dataframe with ``sample_ix`` in samples and ``utc_unix`` in
        seconds.
    sample_rate_hz : float
        LFP sampling rate in Hz.

    Returns
    -------
    tuple[np.ndarray, int, int]
        ``(relative_time_s, start_sample, stop_sample)``. ``relative_time_s``
        has shape ``(stop_sample - start_sample,)`` in seconds. ``stop_sample``
        is exclusive.
    """

    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if float(sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    required_columns = {"sample_ix", "utc_unix"}
    missing_columns = required_columns - set(lfp_irig_df.columns)
    if missing_columns:
        raise ValueError(f"lfp_irig_df is missing required columns: {sorted(missing_columns)}")

    sync_points = lfp_irig_df.loc[:, ["sample_ix", "utc_unix"]].copy()
    sync_points["sample_ix"] = pd.to_numeric(sync_points["sample_ix"], errors="coerce")
    sync_points["utc_unix"] = pd.to_numeric(sync_points["utc_unix"], errors="coerce")
    sync_points = sync_points.replace([np.inf, -np.inf], np.nan).dropna().sort_values("utc_unix")
    if sync_points.shape[0] < 2:
        raise ValueError("At least two finite LFP IRIG sync points are required.")

    utc_unix = sync_points["utc_unix"].to_numpy(dtype=float)
    sample_ix = sync_points["sample_ix"].to_numpy(dtype=float)
    if float(alignment_time_s) < utc_unix[0] or float(alignment_time_s) > utc_unix[-1]:
        raise ValueError("alignment_time_s is outside the decoded LFP sync range.")

    alignment_sample = float(np.interp(float(alignment_time_s), utc_unix, sample_ix))
    sample_tolerance = 1e-9
    start_sample = int(np.floor(alignment_sample + float(window[0]) * float(sample_rate_hz) + sample_tolerance))
    stop_sample = int(np.ceil(alignment_sample + float(window[1]) * float(sample_rate_hz) - sample_tolerance))
    if start_sample < 0:
        raise ValueError("Requested LFP window starts before sample 0.")
    if stop_sample <= start_sample:
        raise ValueError("Requested LFP window contains no samples.")

    relative_time_s = (np.arange(start_sample, stop_sample, dtype=float) - alignment_sample) / float(sample_rate_hz)
    return relative_time_s, start_sample, stop_sample


def read_lfp_saved_channel_window(
    lfp_path: Path | str,
    saved_channel_index: int,
    start_sample: int,
    stop_sample: int,
) -> tuple[np.ndarray, float]:
    """
    Read a gain-corrected LFP segment for one saved channel.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    saved_channel_index : int
        Zero-based saved channel index into the binary file rows.
    start_sample : int
        Inclusive start sample index.
    stop_sample : int
        Exclusive stop sample index.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(lfp_uv, sample_rate_hz)`` where ``lfp_uv`` has shape
        ``(stop_sample - start_sample,)`` in microvolts.
    """

    lfp_file = Path(lfp_path)
    meta = load_lfp_metadata(lfp_file)
    validate_lfp_saved_channel(meta, saved_channel_index=int(saved_channel_index))
    sample_rate_hz = float(readSGLX.SampRate(meta))
    n_saved_channels = int(meta["nSavedChans"])
    n_samples = int(int(meta["fileSizeBytes"]) / (2 * n_saved_channels))
    if int(start_sample) < 0:
        raise ValueError("start_sample must be nonnegative.")
    if int(stop_sample) <= int(start_sample):
        raise ValueError("stop_sample must be greater than start_sample.")
    if int(stop_sample) > n_samples:
        raise ValueError(f"stop_sample {stop_sample} exceeds LFP sample count {n_samples}.")

    raw_data = readSGLX.makeMemMapRaw(lfp_file, meta)
    channel_list = [int(saved_channel_index)]
    raw_channel_window = np.asarray(raw_data[channel_list, int(start_sample):int(stop_sample)], dtype=np.int16)
    gain_corrected_volts = readSGLX.GainCorrectIM(raw_channel_window, channel_list, meta)
    lfp_uv = np.asarray(gain_corrected_volts, dtype=float).reshape(-1) * 1e6
    return lfp_uv, sample_rate_hz


def filter_lfp_trace(
    relative_time_s: np.ndarray,
    lfp_uv: np.ndarray,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float] | None,
) -> np.ndarray:
    """
    Optionally bandpass-filter one LFP trace with Pynapple.

    Parameters
    ----------
    relative_time_s : np.ndarray
        One-dimensional time vector with shape ``(n_samples,)`` in seconds.
        Times are relative to the trial alignment event.
    lfp_uv : np.ndarray
        One-dimensional LFP voltage vector with shape ``(n_samples,)`` in
        microvolts. Values should already be gain-corrected.
    sample_rate_hz : float
        LFP sample rate in Hz.
    frequency_band_hz : tuple[float, float] | None
        Bandpass cutoff frequencies in Hz as ``(low_hz, high_hz)``. If
        ``None``, the gain-corrected signal is returned unfiltered.

    Returns
    -------
    np.ndarray
        One-dimensional LFP vector with shape ``(n_samples,)`` in microvolts.
        The default path is unfiltered; band-limited paths use
        ``nap.apply_bandpass_filter(..., mode="butter")``.
    """

    relative_time_s = np.asarray(relative_time_s, dtype=float).reshape(-1)
    lfp_uv = np.asarray(lfp_uv, dtype=float).reshape(-1)
    if relative_time_s.shape != lfp_uv.shape:
        raise ValueError("relative_time_s and lfp_uv must have the same one-dimensional shape.")
    if float(sample_rate_hz) <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if frequency_band_hz is None:
        return lfp_uv

    if len(frequency_band_hz) != 2:
        raise ValueError("frequency_band_hz must be None or a two-value tuple.")
    low_hz, high_hz = (float(frequency_band_hz[0]), float(frequency_band_hz[1]))
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("frequency_band_hz must satisfy 0 < low_hz < high_hz.")

    lfp_tsd = nap.Tsd(t=relative_time_s, d=lfp_uv, time_units="s")
    filtered_tsd = nap.apply_bandpass_filter(
        lfp_tsd,
        (low_hz, high_hz),
        fs=float(sample_rate_hz),
        mode="butter",
    )
    return np.asarray(filtered_tsd.values, dtype=float).reshape(-1)


def load_trial_lfp_trace(
    lfp_path: Path | str,
    saved_channel_index: int,
    alignment_time_s: float,
    window: tuple[float, float],
    lfp_irig_df: pd.DataFrame,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float] | None = None,
    filter_padding_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one trial-aligned LFP trace from a saved-channel row.

    Parameters
    ----------
    lfp_path : Path | str
        SpikeGLX ``.lf.bin`` file path.
    saved_channel_index : int
        Zero-based saved channel index into the binary file rows.
    alignment_time_s : float
        Absolute alignment timestamp in seconds.
    window : tuple[float, float]
        Relative window bounds in seconds around ``alignment_time_s``.
    lfp_irig_df : pd.DataFrame
        IRIG dataframe with ``sample_ix`` in samples and ``utc_unix`` in
        seconds.
    sample_rate_hz : float
        LFP sampling rate in Hz.
    frequency_band_hz : tuple[float, float] | None, default=None
        Optional bandpass cutoff frequencies in Hz as ``(low_hz, high_hz)``.
        ``None`` returns the gain-corrected, unfiltered trace.
    filter_padding_s : float, default=1.0
        Seconds added to both sides of the requested window before filtering.
        The returned arrays are trimmed back to ``window`` after filtering.
        This parameter is ignored when ``frequency_band_hz`` is ``None``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_uv)``. Both arrays have shape ``(n_samples,)``.
        Time is in seconds relative to alignment; LFP is in microvolts.
    """

    relative_time_s, start_sample, stop_sample = map_lfp_time_window_to_samples(
        alignment_time_s=alignment_time_s,
        window=window,
        lfp_irig_df=lfp_irig_df,
        sample_rate_hz=sample_rate_hz,
    )
    read_start_sample = start_sample
    read_stop_sample = stop_sample
    read_relative_time_s = relative_time_s
    if frequency_band_hz is not None:
        if float(filter_padding_s) < 0:
            raise ValueError("filter_padding_s must be nonnegative.")
        padded_window = (
            float(window[0]) - float(filter_padding_s),
            float(window[1]) + float(filter_padding_s),
        )
        read_relative_time_s, read_start_sample, read_stop_sample = map_lfp_time_window_to_samples(
            alignment_time_s=alignment_time_s,
            window=padded_window,
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=sample_rate_hz,
        )
    lfp_uv, file_sample_rate_hz = read_lfp_saved_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        start_sample=read_start_sample,
        stop_sample=read_stop_sample,
    )
    if not np.isclose(float(file_sample_rate_hz), float(sample_rate_hz)):
        raise ValueError(
            f"LFP sync sample rate {sample_rate_hz} does not match file metadata sample rate {file_sample_rate_hz}."
        )
    if lfp_uv.shape[0] != read_relative_time_s.shape[0]:
        raise ValueError("Loaded LFP trace length does not match the requested sample window.")
    if frequency_band_hz is not None:
        filtered_lfp_uv = filter_lfp_trace(
            relative_time_s=read_relative_time_s,
            lfp_uv=lfp_uv,
            sample_rate_hz=float(sample_rate_hz),
            frequency_band_hz=frequency_band_hz,
        )
        trim_start = int(start_sample) - int(read_start_sample)
        trim_stop = trim_start + int(stop_sample) - int(start_sample)
        lfp_uv = filtered_lfp_uv[trim_start:trim_stop]
        if lfp_uv.shape[0] != relative_time_s.shape[0]:
            raise ValueError("Filtered LFP trace length does not match the requested sample window.")
    return relative_time_s, lfp_uv
