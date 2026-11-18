"""Cached LFP calculations, controls, and interactive LFP views."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import lfp_phase_clustering, unit_spike_plotting
from src.neural_analysis.lfp import spectrogram as lfp_spectrogram
from src.neural_analysis.spike_behavior import loading as spike_behavior_pynapple
from src.neural_analysis.webapp.data_loading import (
    OpenEphysCacheToken,
    _open_ephys_cache_token_or_none,
    _open_ephys_exploratory_provenance,
    _validate_supplied_open_ephys_cache_token,
    build_open_ephys_cache_token,
    load_trial_lfp_trace_for_format_with_sample_rate,
)
from src.neural_analysis.webapp.session_inputs import (
    ACTION_OPTIONS,
    ALIGNMENT_OPTIONS,
    COMPARE_LEFT_RIGHT_ACTION,
    CONDITION_OPTIONS,
    LFP_DROPDOWN_LABEL_HPC_V1,
    LFP_DROPDOWN_LABEL_PFC,
    LFP_FORMAT_OPEN_EPHYS_DERIVED,
    LFP_FORMAT_OPTIONS,
    MetadataLFPSiteInputs,
)


def resolve_lfp_view_sites(
    *,
    metadata_sites: tuple[MetadataLFPSiteInputs, ...] = (),
    hpc_v1_lfp_path: str = "",
    pfc_lfp_path: str = "",
    hpc_v1_aligned_spike_path: str = "",
    pfc_aligned_spike_path: str = "",
) -> tuple[MetadataLFPSiteInputs, ...]:
    """Return direct metadata sites or the equivalent two legacy view sites.

    Parameters
    ----------
    metadata_sites : tuple[MetadataLFPSiteInputs, ...]
        Version-2 site records in declared display order. Each record keeps its
        probe id, saved-channel index, source path, and synchronization path.
    hpc_v1_lfp_path, pfc_lfp_path : str
        Legacy manual continuous-LFP paths used only when ``metadata_sites`` is
        empty.
    hpc_v1_aligned_spike_path, pfc_aligned_spike_path : str
        Legacy manual synchronization paths used only when ``metadata_sites``
        is empty.

    Returns
    -------
    tuple[MetadataLFPSiteInputs, ...]
        The original metadata records unchanged, or two records that preserve
        the established manual HPC/V1 and PFC labels and zero-based default
        saved channel. No source array is opened.
    """
    if metadata_sites:
        return tuple(metadata_sites)
    return (
        MetadataLFPSiteInputs(
            site_id=LFP_DROPDOWN_LABEL_HPC_V1,
            display_label=LFP_DROPDOWN_LABEL_HPC_V1,
            probe_id=LFP_DROPDOWN_LABEL_HPC_V1,
            acquisition_family="legacy",
            saved_channel_index=0,
            lfp_file=Path(hpc_v1_lfp_path) if str(hpc_v1_lfp_path).strip() else None,
            synchronization_file=(
                Path(hpc_v1_aligned_spike_path)
                if str(hpc_v1_aligned_spike_path).strip()
                else None
            ),
        ),
        MetadataLFPSiteInputs(
            site_id=LFP_DROPDOWN_LABEL_PFC,
            display_label=LFP_DROPDOWN_LABEL_PFC,
            probe_id=LFP_DROPDOWN_LABEL_PFC,
            acquisition_family="legacy",
            saved_channel_index=0,
            lfp_file=Path(pfc_lfp_path) if str(pfc_lfp_path).strip() else None,
            synchronization_file=(
                Path(pfc_aligned_spike_path)
                if str(pfc_aligned_spike_path).strip()
                else None
            ),
        ),
    )


LFP_FILTER_BANDS = {
    "Default": None,
    "Theta (5-10 Hz)": (5.0, 10.0),
    "Gamma (50-70 Hz)": (50.0, 70.0),
}

LFP_FILTER_PADDING_S = 1.0

LFP_SPECTROGRAM_MIN_FREQUENCY_HZ = 2.0

LFP_SPECTROGRAM_MAX_FREQUENCY_HZ = 80.0

LFP_SPECTROGRAM_FREQUENCY_COUNT = 40

LFP_SPECTROGRAM_REFERENCE_TRIAL_COUNT = 24

LFP_SPECTROGRAM_COLOR_PERCENTILES = (2.0, 98.0)

LFP_SPECTROGRAM_NOTCH_DEFAULT = False

LFP_SPECTROGRAM_GAUSSIAN_WIDTH = 1.5

LFP_SPECTROGRAM_WINDOW_LENGTH = 1.0

LFP_SPECTROGRAM_PRECISION = 16

LFP_SPECTROGRAM_TARGET_SAMPLE_RATE_HZ = 500.0

LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR = 30.0

LFP_PHASE_CLUSTERING_ANALYSIS_OPTIONS = ("ITPC", "ISPC")

LFP_PHASE_CLUSTERING_MIN_FREQUENCY_HZ = 2.0

LFP_PHASE_CLUSTERING_MAX_FREQUENCY_HZ = 100.0

LFP_PHASE_CLUSTERING_FREQUENCY_COUNT = 50

LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ = 500.0

LFP_PHASE_CLUSTERING_DEFAULT_WINDOW = (-1.0, 2.0)

LFP_PHASE_CLUSTERING_MAXIMUM_CORE_DURATION_S = 120.0

LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE = 1e-12

RELATIVE_PHASE_MIN_FREQUENCY_HZ = 2.0

RELATIVE_PHASE_MAX_FREQUENCY_HZ = 100.0

RELATIVE_PHASE_FREQUENCY_COUNT = 50

RELATIVE_PHASE_OUTPUT_SAMPLE_RATE_HZ = 500.0

RELATIVE_PHASE_DEFAULT_WINDOW = (-1.0, 2.0)

RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS = lfp_phase_clustering.RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS

RELATIVE_PHASE_DISPLAY_OPTIONS = ("Phase difference", "Within-trial PLV", "Phase + PLV")

RELATIVE_PHASE_DEFAULT_DISPLAY = "Within-trial PLV"

WITHIN_TRIAL_PLV_DEFAULT_WINDOW_CYCLES = 3.0

WITHIN_TRIAL_PLV_DEFAULT_MIN_VALID_FRACTION = 0.8

WITHIN_TRIAL_PLV_MIN_WINDOW_DEFAULT_ENABLED = False

WITHIN_TRIAL_PLV_MAX_WINDOW_DEFAULT_ENABLED = False

WITHIN_TRIAL_PLV_DEFAULT_MIN_WINDOW_S = 0.05

WITHIN_TRIAL_PLV_DEFAULT_MAX_WINDOW_S = 1.0

RELATIVE_PHASE_CACHE_MAX_ENTRIES = 12

EVENT_MARKER_STYLES = {
    "start_time": {"label": "trial start", "color": "black"},
    "choice_time": {"label": "choice", "color": "tab:purple"},
    "led_on_time": {"label": "LED", "color": "tab:green"},
}

@st.cache_data(show_spinner="Computing trial LFP spectrogram...")
def _compute_trial_lfp_spectrogram_cached(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    visible_window_start_s: float,
    visible_window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    window_length: float,
    precision: int,
    norm: str,
    target_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    aligned_sync_npz_path: str | None = None,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> lfp_spectrogram.LFPSpectrogramResult:
    """
    Load, pad, decimate, and cache one trial's absolute Morlet power.

    Parameters
    ----------
    lfp_format : str
        LFP format from ``LFP_FORMAT_OPTIONS``.
    lfp_path : str
        Input LFP binary path.
    saved_channel_index : int
        Zero-based saved-channel index.
    alignment_time_s : float
        Absolute trial alignment timestamp in UTC Unix seconds.
    visible_window_start_s, visible_window_end_s : float
        Returned relative time bounds in seconds.
    digital_word : int
        SpikeGLX sync digital word.
    irig_line : int
        SpikeGLX IRIG line.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds.
    utc_offset_hours : float
        Sync timestamp offset in hours.
    frequencies_hz : tuple[float, ...]
        Morlet frequencies in Hz.
    gaussian_width : float
        Pynapple Morlet Gaussian width, dimensionless.
    window_length : float
        Pynapple Morlet window-length parameter, dimensionless.
    precision : int
        Base-2 Morlet evaluation precision.
    norm : str
        Pynapple wavelet normalization, ``"l1"`` or ``"l2"``.
    target_sample_rate_hz : float
        Preferred post-decimation sample rate in Hz.
    notch_60_hz : bool
        Whether to apply a 60 Hz notch before decimation.
    notch_quality_factor : float
        Dimensionless 60 Hz notch quality factor.
    aligned_sync_npz_path : str | None, optional
        Open Ephys aligned sync path; ignored for SpikeGLX.
    open_ephys_cache_token : OpenEphysCacheToken or None, optional
        Complete physical-uV Open Ephys identity used only for this decorated
        cache key. The public wrapper resolves it before numerical loading.

    Returns
    -------
    lfp_spectrogram.LFPSpectrogramResult
        Visible trace and absolute power. Power shape is
        ``(n_visible_samples, n_frequencies)`` in dB relative to one squared
        input unit; time is in relative seconds and frequencies are in Hz.
    """

    del open_ephys_cache_token
    frequencies = np.asarray(frequencies_hz, dtype=float)
    padding_s = lfp_spectrogram.compute_wavelet_padding_s(
        minimum_frequency_hz=float(np.min(frequencies)),
        window_length=float(window_length),
    )
    padded_start_s = float(visible_window_start_s) - padding_s
    padded_end_s = float(visible_window_end_s) + padding_s
    relative_time_s, lfp_values, sample_rate_hz = load_trial_lfp_trace_for_format_with_sample_rate(
        lfp_format=lfp_format,
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        alignment_time_s=float(alignment_time_s),
        window_start_s=padded_start_s,
        window_end_s=padded_end_s,
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
        aligned_sync_npz_path=aligned_sync_npz_path,
    )
    return lfp_spectrogram.compute_morlet_log_power(
        relative_time_s=relative_time_s,
        lfp_values=lfp_values,
        sample_rate_hz=float(sample_rate_hz),
        frequencies_hz=frequencies,
        visible_window=(float(visible_window_start_s), float(visible_window_end_s)),
        gaussian_width=float(gaussian_width),
        window_length=float(window_length),
        precision=int(precision),
        norm=norm,
        target_sample_rate_hz=float(target_sample_rate_hz),
        notch_60_hz=bool(notch_60_hz),
        notch_quality_factor=float(notch_quality_factor),
    )

def compute_trial_lfp_spectrogram_cached(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    visible_window_start_s: float,
    visible_window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    window_length: float,
    precision: int,
    norm: str,
    target_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    aligned_sync_npz_path: str | None = None,
    *,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> lfp_spectrogram.LFPSpectrogramResult:
    """Compute one visible trial spectrogram through a source-identity cache.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label selecting the Open Ephys physical-uV reader
        or unchanged SpikeGLX source path.
    lfp_path : str
        Selected LFP binary filesystem path with no array axis or physical
        unit.
    aligned_sync_npz_path : str or None
        Open Ephys synchronization filesystem path, required for that format
        and unused for SpikeGLX.
    saved_channel_index : int
        Zero-based saved-channel index on the source channel axis.
    alignment_time_s, visible_window_start_s, visible_window_end_s : float
        Absolute alignment and event-relative visible bounds in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization selectors.
    bit_period_s, utc_offset_hours : float
        SpikeGLX IRIG period in seconds and synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        One-dimensional positive Morlet frequency axis in Hz.
    gaussian_width, window_length : float
        Dimensionless Morlet settings.
    precision : int
        Base-2 Morlet evaluation precision.
    norm : str
        Morlet normalization label.
    target_sample_rate_hz : float
        Positive output sampling rate in Hz.
    notch_60_hz : bool
        Whether to apply the existing 60 Hz notch.
    notch_quality_factor : float
        Dimensionless notch quality factor.
    open_ephys_cache_token : OpenEphysCacheToken or None, keyword-only
        Optional complete source identity. Omission remains compatible and
        builds one before cache keying; a supplied valid token is reused.

    Returns
    -------
    lfp_spectrogram.LFPSpectrogramResult
        Visible one-dimensional time (seconds) and raw LFP (uV for Open
        Ephys), plus frequency-by-time log-power in dB relative to uV^2 for
        Open Ephys.

    Raises
    ------
    ValueError
        If transform settings, source/sync identity, or the Open Ephys token
        contract is invalid before the decorated cache body runs.
    OSError
        If the selected source cannot be read.
    """
    token = _open_ephys_cache_token_or_none(
        lfp_format, lfp_path, aligned_sync_npz_path, open_ephys_cache_token
    )
    return _compute_trial_lfp_spectrogram_cached(
        lfp_format, lfp_path, saved_channel_index, alignment_time_s,
        visible_window_start_s, visible_window_end_s, digital_word, irig_line,
        bit_period_s, utc_offset_hours, frequencies_hz, gaussian_width,
        window_length, precision, norm, target_sample_rate_hz, notch_60_hz,
        notch_quality_factor, aligned_sync_npz_path, token,
    )

compute_trial_lfp_spectrogram_cached.clear = _compute_trial_lfp_spectrogram_cached.clear

@st.cache_data(show_spinner="Estimating shared LFP power scale...")
def _compute_shared_lfp_power_limits_cached(
    reference_alignment_times_s: tuple[float, ...],
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    visible_window_start_s: float,
    visible_window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    window_length: float,
    precision: int,
    norm: str,
    target_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    lower_percentile: float,
    upper_percentile: float,
    aligned_sync_npz_path: str | None = None,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> tuple[tuple[float, float], int]:
    """
    Compute a shared robust power scale from session reference trials.

    Parameters
    ----------
    reference_alignment_times_s : tuple[float, ...]
        Absolute alignment times for deterministic reference trials in seconds.
    lfp_format, lfp_path, saved_channel_index : str, str, int
        LFP acquisition format, binary path, and zero-based channel index.
    visible_window_start_s, visible_window_end_s : float
        Relative displayed bounds in seconds.
    digital_word, irig_line : int
        SpikeGLX sync word and line.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds.
    utc_offset_hours : float
        Sync offset in hours.
    frequencies_hz : tuple[float, ...]
        Morlet frequencies in Hz.
    gaussian_width, window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int
        Base-2 Morlet evaluation precision.
    norm : str
        Pynapple wavelet normalization.
    target_sample_rate_hz : float
        Preferred post-decimation sample rate in Hz.
    notch_60_hz : bool
        Whether to apply a 60 Hz notch.
    notch_quality_factor : float
        Dimensionless notch quality factor.
    lower_percentile, upper_percentile : float
        Pooled robust percentile bounds in percent.
    aligned_sync_npz_path : str | None, optional
        Open Ephys aligned sync path.
    open_ephys_cache_token : OpenEphysCacheToken or None, optional
        Complete physical-uV source identity passed unchanged to every nested
        trial spectrogram cache key. It contains no numerical trace arrays.

    Returns
    -------
    tuple[tuple[float, float], int]
        Shared ``(minimum_db, maximum_db)`` limits and the number of reference
        trials successfully included. Power units are dB relative to one
        squared input unit.
    """

    reference_power = []
    for alignment_time_s in reference_alignment_times_s:
        try:
            result = compute_trial_lfp_spectrogram_cached(
                lfp_format=lfp_format,
                lfp_path=lfp_path,
                saved_channel_index=int(saved_channel_index),
                alignment_time_s=float(alignment_time_s),
                visible_window_start_s=float(visible_window_start_s),
                visible_window_end_s=float(visible_window_end_s),
                digital_word=int(digital_word),
                irig_line=int(irig_line),
                bit_period_s=float(bit_period_s),
                utc_offset_hours=float(utc_offset_hours),
                frequencies_hz=frequencies_hz,
                gaussian_width=float(gaussian_width),
                window_length=float(window_length),
                precision=int(precision),
                norm=norm,
                target_sample_rate_hz=float(target_sample_rate_hz),
                notch_60_hz=bool(notch_60_hz),
                notch_quality_factor=float(notch_quality_factor),
                aligned_sync_npz_path=aligned_sync_npz_path,
                open_ephys_cache_token=open_ephys_cache_token,
            )
        except (FileNotFoundError, OSError, ValueError):
            continue
        reference_power.append(result.log_power_db)
    limits = lfp_spectrogram.estimate_shared_log_power_limits(
        reference_power,
        lower_percentile=float(lower_percentile),
        upper_percentile=float(upper_percentile),
    )
    return limits, len(reference_power)

def compute_shared_lfp_power_limits_cached(
    reference_alignment_times_s: tuple[float, ...],
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    visible_window_start_s: float,
    visible_window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    window_length: float,
    precision: int,
    norm: str,
    target_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    lower_percentile: float,
    upper_percentile: float,
    aligned_sync_npz_path: str | None = None,
    *,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> tuple[tuple[float, float], int]:
    """Estimate one shared spectrogram scale through a source-identity cache.

    Parameters
    ----------
    reference_alignment_times_s : tuple[float, ...]
        One-dimensional sequence of absolute reference-trial timestamps in
        seconds; each element is independently loaded with the same source.
    lfp_format : str
        Acquisition-format label selecting Open Ephys or SpikeGLX behavior.
    lfp_path : str
        Selected LFP binary filesystem path with no array axis or physical
        unit.
    aligned_sync_npz_path : str or None
        Open Ephys sync filesystem path, required for that format and unused
        for SpikeGLX.
    saved_channel_index : int
        Zero-based source channel index.
    visible_window_start_s, visible_window_end_s, bit_period_s : float
        Event-relative visible bounds and IRIG period in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization selectors.
    utc_offset_hours : float
        Synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        One-dimensional Morlet frequency axis in Hz.
    gaussian_width, window_length, notch_quality_factor : float
        Dimensionless transform/filter settings.
    precision : int
        Base-2 Morlet evaluation precision.
    norm : str
        Morlet normalization label.
    target_sample_rate_hz : float
        Positive output rate in Hz.
    notch_60_hz : bool
        Whether to apply the existing notch filter.
    lower_percentile, upper_percentile : float
        Robust pooled power percentiles in percent.
    open_ephys_cache_token : OpenEphysCacheToken or None, keyword-only
        Complete physical-uV source identity shared unchanged with nested
        spectrogram calls, or ``None`` to construct one before keying.

    Returns
    -------
    tuple[tuple[float, float], int]
        ``((minimum_db, maximum_db), included_count)``. Limits describe dB
        relative to uV^2 for Open Ephys and have no trace array axis.

    Raises
    ------
    ValueError
        If the source/token/transform contract is invalid before nested cache
        calls. Individual unusable reference trials remain excluded rather
        than failing the shared-limit calculation.
    """
    token = _open_ephys_cache_token_or_none(
        lfp_format, lfp_path, aligned_sync_npz_path, open_ephys_cache_token
    )
    return _compute_shared_lfp_power_limits_cached(
        reference_alignment_times_s, lfp_format, lfp_path, saved_channel_index,
        visible_window_start_s, visible_window_end_s, digital_word, irig_line,
        bit_period_s, utc_offset_hours, frequencies_hz, gaussian_width,
        window_length, precision, norm, target_sample_rate_hz, notch_60_hz,
        notch_quality_factor, lower_percentile, upper_percentile,
        aligned_sync_npz_path, token,
    )

compute_shared_lfp_power_limits_cached.clear = _compute_shared_lfp_power_limits_cached.clear

@st.cache_data(show_spinner="Computing continuous LFP phase tensors...")
def _compute_lfp_phase_site_cached(
    lfp_format: str,
    lfp_path: str,
    lfp_file_mtime_ns: int,
    saved_channel_index: int,
    event_times_s: tuple[float, ...],
    trial_indices: tuple[int, ...],
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    window_length: float,
    precision: int,
    norm: str,
    output_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    minimum_relative_magnitude: float,
    maximum_core_duration_s: float,
    aligned_sync_npz_path: str | None = None,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> lfp_phase_clustering.PhaseTrialTensor:
    """
    Cache continuous-block Morlet phase preprocessing for one LFP site.

    Parameters
    ----------
    lfp_format : str
        LFP acquisition format from ``LFP_FORMAT_OPTIONS``.
    lfp_path : str
        LFP binary filesystem path.
    lfp_file_mtime_ns : int
        Source-file modification time in nanoseconds, used only for cache
        invalidation.
    saved_channel_index : int
        Zero-based saved-channel row index.
    event_times_s : tuple[float, ...]
        Chronological absolute event timestamps in seconds.
    trial_indices : tuple[int, ...]
        Trial-table row positions matching ``event_times_s``.
    window_start_s, window_end_s : float
        Event-relative output bounds in seconds.
    digital_word, irig_line : int
        SpikeGLX sync digital word and IRIG line.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds.
    utc_offset_hours : float
        Offset applied to synchronized LFP timestamps in hours.
    frequencies_hz : tuple[float, ...]
        Positive Morlet frequencies in Hz.
    gaussian_width, window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int
        Base-2 Pynapple wavelet evaluation precision.
    norm : str
        Pynapple wavelet normalization.
    output_sample_rate_hz : float
        Exact event-relative phase tensor rate in Hz.
    notch_60_hz : bool
        Whether to apply the optional 60 Hz notch.
    notch_quality_factor : float
        Dimensionless notch quality factor.
    minimum_relative_magnitude : float
        Relative wavelet magnitude threshold for valid phase.
    maximum_core_duration_s : float
        Maximum unpadded continuous transform block duration in seconds.
    aligned_sync_npz_path : str | None, optional
        Open Ephys aligned sync path; ignored for SpikeGLX.
    open_ephys_cache_token : OpenEphysCacheToken or None, optional
        Complete physical-uV source identity used only in this decorated cache
        key. It has no phase/trace array axes and is resolved by the wrapper.

    Returns
    -------
    lfp_phase_clustering.PhaseTrialTensor
        One-site phase tensor with shape ``(1, frequency, trial, time)``;
        phase is unitless complex phase and time is in event-relative seconds.
    """

    del lfp_file_mtime_ns, open_ephys_cache_token

    def load_absolute_block(
        block_start_s: float,
        block_end_s: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Load one absolute block and preserve its UTC-like sample times."""

        block_center_s = 0.5 * (float(block_start_s) + float(block_end_s))
        relative_time_s, lfp_values, sample_rate_hz = load_trial_lfp_trace_for_format_with_sample_rate(
            lfp_format=lfp_format,
            lfp_path=lfp_path,
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=block_center_s,
            window_start_s=float(block_start_s) - block_center_s,
            window_end_s=float(block_end_s) - block_center_s,
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
            aligned_sync_npz_path=aligned_sync_npz_path,
        )
        return block_center_s + relative_time_s, lfp_values, float(sample_rate_hz)

    return lfp_phase_clustering.compute_site_phase_trial_tensor(
        event_times_s=np.asarray(event_times_s, dtype=float),
        trial_indices=np.asarray(trial_indices, dtype=int),
        block_loader=load_absolute_block,
        frequencies_hz=np.asarray(frequencies_hz, dtype=float),
        window=(float(window_start_s), float(window_end_s)),
        output_sample_rate_hz=float(output_sample_rate_hz),
        gaussian_width=float(gaussian_width),
        window_length=float(window_length),
        precision=int(precision),
        norm=norm,
        notch_60_hz=bool(notch_60_hz),
        notch_quality_factor=float(notch_quality_factor),
        minimum_relative_magnitude=float(minimum_relative_magnitude),
        maximum_core_duration_s=float(maximum_core_duration_s),
    )

def compute_lfp_phase_site_cached(
    lfp_format: str,
    lfp_path: str,
    lfp_file_mtime_ns: int,
    saved_channel_index: int,
    event_times_s: tuple[float, ...],
    trial_indices: tuple[int, ...],
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    window_length: float,
    precision: int,
    norm: str,
    output_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    minimum_relative_magnitude: float,
    maximum_core_duration_s: float,
    aligned_sync_npz_path: str | None = None,
    *,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> lfp_phase_clustering.PhaseTrialTensor:
    """Compute a continuous site phase tensor through a source-identity cache.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label selecting Open Ephys or SpikeGLX behavior.
    lfp_path : str
        Selected LFP binary filesystem path with no array axis or physical
        unit.
    aligned_sync_npz_path : str or None
        Open Ephys synchronization filesystem path, required for that format
        and unused for SpikeGLX.
    lfp_file_mtime_ns : int
        Existing compatibility cache identity in nanoseconds.
    saved_channel_index : int
        Zero-based saved-channel index.
    event_times_s : tuple[float, ...]
        One-dimensional absolute event timestamps in seconds.
    trial_indices : tuple[int, ...]
        One-dimensional source trial-row axis aligned with ``event_times_s``.
    window_start_s, window_end_s, bit_period_s : float
        Event-relative bounds and SpikeGLX IRIG period in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization selectors.
    utc_offset_hours : float
        Synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        One-dimensional positive Morlet frequencies in Hz.
    gaussian_width, window_length, notch_quality_factor : float
        Dimensionless transform/filter settings.
    precision : int
        Base-2 Morlet evaluation precision.
    norm : str
        Morlet normalization label.
    output_sample_rate_hz : float
        Positive output time-axis rate in Hz.
    notch_60_hz : bool
        Whether to apply the existing notch filter.
    minimum_relative_magnitude, maximum_core_duration_s : float
        Dimensionless validity threshold and maximum block span in seconds.
    open_ephys_cache_token : OpenEphysCacheToken or None, keyword-only
        Optional complete physical-uV source identity, constructed before the
        keyed cache only when omitted.

    Returns
    -------
    lfp_phase_clustering.PhaseTrialTensor
        Unit complex phase and validity arrays with axes ``(frequency, trial,
        time)``; phase is dimensionless/radians and its time coordinate is in
        event-relative seconds.

    Raises
    ------
    ValueError
        If the source/synchronization/token identity, event/trial axes, or
        transform settings are invalid before the decorated cache body runs.
    OSError
        If production source loading fails.
    """
    token = _open_ephys_cache_token_or_none(
        lfp_format, lfp_path, aligned_sync_npz_path, open_ephys_cache_token
    )
    return _compute_lfp_phase_site_cached(
        lfp_format, lfp_path, lfp_file_mtime_ns, saved_channel_index,
        event_times_s, trial_indices, window_start_s, window_end_s, digital_word,
        irig_line, bit_period_s, utc_offset_hours, frequencies_hz, gaussian_width,
        window_length, precision, norm, output_sample_rate_hz, notch_60_hz,
        notch_quality_factor, minimum_relative_magnitude, maximum_core_duration_s,
        aligned_sync_npz_path, token,
    )

compute_lfp_phase_site_cached.clear = _compute_lfp_phase_site_cached.clear

@st.cache_data(
    show_spinner="Computing single-trial relative phase...",
    max_entries=RELATIVE_PHASE_CACHE_MAX_ENTRIES,
)
def _compute_single_trial_relative_phase_cached(
    lfp_format: str,
    lfp_path_a: str,
    lfp_mtime_ns_a: int,
    channel_a: int,
    site_a_label: str,
    aligned_sync_path_a: str | None,
    aligned_sync_mtime_ns_a: int,
    lfp_path_b: str,
    lfp_mtime_ns_b: int,
    channel_b: int,
    site_b_label: str,
    aligned_sync_path_b: str | None,
    aligned_sync_mtime_ns_b: int,
    trial_index: int,
    event_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    wavelet_window_length: float,
    precision: int,
    norm: str,
    output_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    minimum_relative_magnitude: float,
    plv_window_cycles: float,
    plv_min_window_s: float | None,
    plv_max_window_s: float | None,
    open_ephys_cache_token: tuple[OpenEphysCacheToken | None, OpenEphysCacheToken | None] | None = None,
) -> lfp_phase_clustering.SingleTrialRelativePhaseResult:
    """
    Load and compare two padded LFP segments for one selected trial.

    Parameters
    ----------
    lfp_format : str
        Shared acquisition format from ``LFP_FORMAT_OPTIONS``.
    lfp_path_a, lfp_path_b : str
        Probe-specific continuous LFP binary paths.
    lfp_mtime_ns_a, lfp_mtime_ns_b : int
        Source modification times in nanoseconds used for cache invalidation.
    channel_a, channel_b : int
        Zero-based saved-channel indices.
    site_a_label, site_b_label : str
        Ordered probe/channel labels defining phase A minus B.
    aligned_sync_path_a, aligned_sync_path_b : str | None
        Open Ephys aligned sync paths; ``None`` for SpikeGLX.
    aligned_sync_mtime_ns_a, aligned_sync_mtime_ns_b : int
        Sync-file modification times in nanoseconds or ``-1`` when unused.
    trial_index : int
        Source trial-table row identifier.
    event_time_s : float
        Absolute behavioral alignment time in seconds.
    window_start_s, window_end_s : float
        Visible event-relative half-open interval in seconds.
    digital_word, irig_line : int
        SpikeGLX sync digital word and IRIG line.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds.
    utc_offset_hours : float
        Constant synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        Positive Morlet frequencies in Hz.
    gaussian_width, wavelet_window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int
        Base-2 Pynapple wavelet evaluation precision.
    norm : str
        Pynapple wavelet normalization.
    output_sample_rate_hz : float
        Exact relative-phase comparison-grid rate in Hz.
    notch_60_hz : bool
        Whether heatmap processing applies a 60 Hz notch.
    notch_quality_factor : float
        Dimensionless notch-filter quality factor.
    minimum_relative_magnitude : float
        Relative numerical-validity threshold for wavelet coefficients.
    plv_window_cycles : float
        Positive cycles per local PLV window.
    plv_min_window_s, plv_max_window_s : float | None
        Optional local PLV duration bounds in seconds. These affect the retained
        support interval and therefore participate in the cache key.
    open_ephys_cache_token : tuple[OpenEphysCacheToken | None, OpenEphysCacheToken | None] or None
        Ordered complete identities for source A and source B, used only in
        this decorated cache key. The public wrapper path-validates or builds
        both identities before numerical loading.

    Returns
    -------
    lfp_phase_clustering.SingleTrialRelativePhaseResult
        Visible and PLV-support frequency-by-time phase/amplitude arrays plus
        visible unprocessed source-rate traces. Wavelet coefficients and
        Morlet-only padding are not retained in the cache.
    """

    del (
        lfp_mtime_ns_a,
        lfp_mtime_ns_b,
        aligned_sync_mtime_ns_a,
        aligned_sync_mtime_ns_b,
        open_ephys_cache_token,
    )
    frequencies = np.asarray(frequencies_hz, dtype=float)
    padding_s = lfp_spectrogram.compute_wavelet_padding_s(
        minimum_frequency_hz=float(np.min(frequencies)),
        window_length=float(wavelet_window_length),
    )
    plv_window_sample_count, _effective_window_s = lfp_phase_clustering.compute_plv_window_samples(
        frequencies_hz=frequencies,
        sample_rate_hz=float(output_sample_rate_hz),
        window_cycles=float(plv_window_cycles),
        min_window_s=plv_min_window_s,
        max_window_s=plv_max_window_s,
    )
    plv_support_s = float(np.max(plv_window_sample_count // 2)) / float(output_sample_rate_hz)

    def load_site(
        lfp_path: str,
        channel: int,
        aligned_sync_path: str | None,
    ) -> lfp_phase_clustering.WaveletCoefficientResult:
        """Load one padded source segment and calculate its complex Morlet coefficients."""

        relative_time_s, lfp_values, sample_rate_hz = load_trial_lfp_trace_for_format_with_sample_rate(
            lfp_format=lfp_format,
            lfp_path=lfp_path,
            saved_channel_index=int(channel),
            alignment_time_s=float(event_time_s),
            window_start_s=float(window_start_s) - plv_support_s - padding_s,
            window_end_s=float(window_end_s) + plv_support_s + padding_s,
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
            aligned_sync_npz_path=aligned_sync_path,
        )
        return lfp_phase_clustering.compute_wavelet_coefficients(
            time_s=float(event_time_s) + relative_time_s,
            lfp_values=lfp_values,
            sample_rate_hz=float(sample_rate_hz),
            frequencies_hz=frequencies,
            gaussian_width=float(gaussian_width),
            window_length=float(wavelet_window_length),
            precision=int(precision),
            norm=norm,
            target_sample_rate_hz=float(output_sample_rate_hz),
            notch_60_hz=bool(notch_60_hz),
            notch_quality_factor=float(notch_quality_factor),
        )

    site_a = load_site(lfp_path_a, int(channel_a), aligned_sync_path_a)
    site_b = load_site(lfp_path_b, int(channel_b), aligned_sync_path_b)
    return lfp_phase_clustering.compute_single_trial_relative_phase(
        site_a=site_a,
        site_b=site_b,
        event_time_s=float(event_time_s),
        visible_window=(float(window_start_s), float(window_end_s)),
        output_sample_rate_hz=float(output_sample_rate_hz),
        trial_index=int(trial_index),
        site_a_label=site_a_label,
        site_b_label=site_b_label,
        minimum_relative_magnitude=float(minimum_relative_magnitude),
        support_window=(
            float(window_start_s) - plv_support_s,
            float(window_end_s) + plv_support_s,
        ),
    )

def compute_single_trial_relative_phase_cached(
    lfp_format: str,
    lfp_path_a: str,
    lfp_mtime_ns_a: int,
    channel_a: int,
    site_a_label: str,
    aligned_sync_path_a: str | None,
    aligned_sync_mtime_ns_a: int,
    lfp_path_b: str,
    lfp_mtime_ns_b: int,
    channel_b: int,
    site_b_label: str,
    aligned_sync_path_b: str | None,
    aligned_sync_mtime_ns_b: int,
    trial_index: int,
    event_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    frequencies_hz: tuple[float, ...],
    gaussian_width: float,
    wavelet_window_length: float,
    precision: int,
    norm: str,
    output_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    minimum_relative_magnitude: float,
    plv_window_cycles: float,
    plv_min_window_s: float | None,
    plv_max_window_s: float | None,
    *,
    open_ephys_cache_token: tuple[OpenEphysCacheToken | None, OpenEphysCacheToken | None] | None = None,
) -> lfp_phase_clustering.SingleTrialRelativePhaseResult:
    """Compute one pairwise cached relative-phase result.

    Parameters
    ----------
    lfp_format : str
        Shared acquisition format for both sources.
    lfp_path_a, lfp_path_b : str
        Source-A/B LFP binary filesystem paths with no numerical array axis.
    aligned_sync_path_a, aligned_sync_path_b : str or None
        Source-A/B Open Ephys synchronization paths, required for Open Ephys
        and unused for SpikeGLX.
    lfp_mtime_ns_a, lfp_mtime_ns_b, aligned_sync_mtime_ns_a, aligned_sync_mtime_ns_b : int
        Compatibility cache identities in nanoseconds.
    channel_a, channel_b, trial_index : int
        Zero-based saved-channel and trial-row identifiers with no physical unit.
    site_a_label, site_b_label : str
        Ordered human-readable source labels defining phase A minus B; they
        are categorical values with no physical unit or numerical axis.
    event_time_s, window_start_s, window_end_s : float
        Absolute and event-relative timestamps in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization selectors, ignored by Open Ephys.
    bit_period_s, utc_offset_hours : float
        SpikeGLX IRIG period in seconds and synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        One-dimensional ascending frequency axis in Hz.
    gaussian_width, wavelet_window_length, notch_quality_factor : float
        Dimensionless Morlet/notch settings.
    precision : int
        Base-2 Morlet evaluation precision.
    norm : str
        Morlet normalization label.
    output_sample_rate_hz : float
        Positive common phase-grid rate in Hz.
    notch_60_hz : bool
        Whether to apply the existing notch filter.
    minimum_relative_magnitude : float
        Dimensionless wavelet numerical-validity threshold.
    plv_window_cycles : float
        Positive local PLV window width in cycles.
    plv_min_window_s, plv_max_window_s : float or None
        Optional PLV duration bounds in seconds.
    open_ephys_cache_token : tuple[OpenEphysCacheToken, OpenEphysCacheToken] or None
        Optional complete identities for sources A/B. Valid supplied instances
        are path-validated without rebuilding or rehashing.

    Returns
    -------
    lfp_phase_clustering.SingleTrialRelativePhaseResult
        Relative-phase arrays with frequency-by-time axes; phase is in radians,
        source trace values remain uV for Open Ephys, and time is seconds.

    Raises
    ------
    ValueError
        If either source/synchronization/token binding, the paired trial axes,
        or transform/PLV settings are invalid before cache execution.
    OSError
        If either production LFP source cannot be read.
    """
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        if open_ephys_cache_token is None:
            tokens = (
                _open_ephys_cache_token_or_none(lfp_format, lfp_path_a, aligned_sync_path_a, None),
                _open_ephys_cache_token_or_none(lfp_format, lfp_path_b, aligned_sync_path_b, None),
            )
        else:
            if len(open_ephys_cache_token) != 2:
                raise ValueError("relative-phase Open Ephys cache token must contain source A and B")
            token_a, token_b = open_ephys_cache_token
            if token_a is None or token_b is None or aligned_sync_path_a is None or aligned_sync_path_b is None:
                raise ValueError("relative-phase Open Ephys cache tokens require both aligned sync paths")
            tokens = (
                _validate_supplied_open_ephys_cache_token(token_a, lfp_path_a, aligned_sync_path_a),
                _validate_supplied_open_ephys_cache_token(token_b, lfp_path_b, aligned_sync_path_b),
            )
    else:
        tokens = None
    return _compute_single_trial_relative_phase_cached(
        lfp_format, lfp_path_a, lfp_mtime_ns_a, channel_a, site_a_label,
        aligned_sync_path_a, aligned_sync_mtime_ns_a, lfp_path_b, lfp_mtime_ns_b,
        channel_b, site_b_label, aligned_sync_path_b, aligned_sync_mtime_ns_b,
        trial_index, event_time_s, window_start_s, window_end_s, digital_word,
        irig_line, bit_period_s, utc_offset_hours, frequencies_hz, gaussian_width,
        wavelet_window_length, precision, norm, output_sample_rate_hz, notch_60_hz,
        notch_quality_factor, minimum_relative_magnitude, plv_window_cycles,
        plv_min_window_s, plv_max_window_s, tokens,
    )

compute_single_trial_relative_phase_cached.clear = _compute_single_trial_relative_phase_cached.clear

def render_single_trial_relative_phase_view(
    trial_df: pd.DataFrame,
    event_df: pd.DataFrame,
    session: spike_behavior_pynapple.Session,
    hpc_v1_lfp_path: str = "",
    pfc_lfp_path: str = "",
    hpc_v1_aligned_spike_path: str = "",
    pfc_aligned_spike_path: str = "",
    *,
    metadata_lfp_sites: tuple[MetadataLFPSiteInputs, ...] = (),
) -> None:
    """
    Render and optionally save one trial's two-site relative-phase view.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Session trial table with shape ``(n_trials, n_columns)`` and absolute
        event times in seconds.
    event_df : pd.DataFrame
        Session event table with shape ``(n_events, n_columns)`` used to build
        left/right lick timestamps in seconds.
    session : spike_behavior_pynapple.Session
        Session metadata supplying identifier and output directories.
    hpc_v1_lfp_path, pfc_lfp_path : str
        Probe-specific continuous LFP binary paths.
    hpc_v1_aligned_spike_path, pfc_aligned_spike_path : str
        Probe-specific aligned sync paths for Open Ephys-derived LFP.
    metadata_lfp_sites : tuple[MetadataLFPSiteInputs, ...]
        Direct version-2 site records. When nonempty, their probe identities,
        saved channels, LFP paths, and synchronization paths replace the
        legacy manual path arguments.

    Returns
    -------
    None
        Streamlit renders a four-panel selected-trial figure and optional
        timestamped NPZ/PNG outputs.
    """

    st.sidebar.header("Single-Trial Relative Phase")
    condition = st.sidebar.selectbox("Trial condition", options=CONDITION_OPTIONS)
    action_options = {label: value for label, value in ACTION_OPTIONS.items() if value != COMPARE_LEFT_RIGHT_ACTION}
    action_label = st.sidebar.selectbox("Action", options=list(action_options))
    alignment_event = st.sidebar.selectbox("Alignment event", options=ALIGNMENT_OPTIONS)
    window_start_s = float(
        st.sidebar.number_input("Window start (s)", value=float(RELATIVE_PHASE_DEFAULT_WINDOW[0]), step=0.1)
    )
    window_end_s = float(
        st.sidebar.number_input("Window end (s)", value=float(RELATIVE_PHASE_DEFAULT_WINDOW[1]), step=0.1)
    )
    if window_start_s >= window_end_s:
        st.error("Window start must be less than window end.")
        st.stop()

    condition_masks = lfp_phase_clustering.make_phase_condition_masks(trial_df)
    selected_mask = condition_masks[condition].copy()
    selected_action = action_options[action_label]
    if selected_action != "all":
        actions = pd.to_numeric(trial_df["action"], errors="coerce").to_numpy(dtype=float)
        selected_mask &= actions == int(selected_action)
    alignment_times = pd.to_numeric(trial_df[alignment_event], errors="coerce").to_numpy(dtype=float)
    selected_mask &= np.isfinite(alignment_times)
    selected_trial_indices = np.flatnonzero(selected_mask)
    if selected_trial_indices.size == 0:
        st.warning("No trials match the selected condition, action, and alignment event.")
        st.stop()
    trial_position = int(
        st.sidebar.number_input(
            "Trial position",
            min_value=0,
            max_value=int(selected_trial_indices.size - 1),
            value=0,
            step=1,
        )
    )
    trial_index = int(selected_trial_indices[trial_position])
    event_time_s = float(alignment_times[trial_index])
    st.sidebar.caption(f"Trial row {trial_index}; {selected_trial_indices.size} matching trials")

    lfp_format = st.sidebar.selectbox("LFP format", options=LFP_FORMAT_OPTIONS)
    utc_offset_hours = float(
        st.sidebar.number_input("LFP UTC offset (hours)", value=0, step=1, format="%d")
    )
    lfp_sites = resolve_lfp_view_sites(
        metadata_sites=metadata_lfp_sites,
        hpc_v1_lfp_path=hpc_v1_lfp_path,
        pfc_lfp_path=pfc_lfp_path,
        hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
        pfc_aligned_spike_path=pfc_aligned_spike_path,
    )
    probe_sources = {site.display_label: site for site in lfp_sites}
    probe_options = list(probe_sources)
    if len(probe_options) < 2:
        st.warning("At least two LFP sites are required for relative phase.")
        st.stop()
    probe_a = st.sidebar.selectbox("Site A probe", options=probe_options, index=0)
    channel_a = int(
        st.sidebar.number_input(
            "Site A saved channel",
            min_value=0,
            value=int(probe_sources[probe_a].saved_channel_index),
            step=1,
        )
    )
    probe_b = st.sidebar.selectbox("Site B probe", options=probe_options, index=1)
    channel_b = int(
        st.sidebar.number_input(
            "Site B saved channel",
            min_value=0,
            value=int(probe_sources[probe_b].saved_channel_index),
            step=1,
        )
    )
    if (probe_a, channel_a) == (probe_b, channel_b):
        st.error("Site A and site B must be distinct probe/channel selections.")
        st.stop()

    minimum_frequency_hz = float(
        st.sidebar.number_input(
            "Minimum frequency (Hz)",
            min_value=0.1,
            value=float(RELATIVE_PHASE_MIN_FREQUENCY_HZ),
            step=0.5,
        )
    )
    maximum_frequency_hz = float(
        st.sidebar.number_input(
            "Maximum frequency (Hz)",
            min_value=0.2,
            value=float(RELATIVE_PHASE_MAX_FREQUENCY_HZ),
            step=5.0,
        )
    )
    frequency_count = int(
        st.sidebar.number_input(
            "Frequency count",
            min_value=2,
            value=int(RELATIVE_PHASE_FREQUENCY_COUNT),
            step=1,
        )
    )
    gaussian_width = float(
        st.sidebar.number_input(
            "Morlet Gaussian width",
            min_value=0.1,
            value=float(LFP_SPECTROGRAM_GAUSSIAN_WIDTH),
            step=0.1,
        )
    )
    wavelet_window_length = float(
        st.sidebar.number_input(
            "Morlet window length",
            min_value=0.1,
            value=float(LFP_SPECTROGRAM_WINDOW_LENGTH),
            step=0.1,
        )
    )
    wavelet_norm = st.sidebar.selectbox("Morlet normalization", options=["l1", "l2"])
    notch_60_hz = st.sidebar.checkbox("Apply 60 Hz notch to heatmap", value=False)
    if minimum_frequency_hz >= maximum_frequency_hz:
        st.error("Maximum frequency must be greater than minimum frequency.")
        st.stop()
    frequencies_hz = np.geomspace(minimum_frequency_hz, maximum_frequency_hz, frequency_count)

    display_mode = st.sidebar.selectbox(
        "Phase analysis display",
        options=RELATIVE_PHASE_DISPLAY_OPTIONS,
        index=RELATIVE_PHASE_DISPLAY_OPTIONS.index(RELATIVE_PHASE_DEFAULT_DISPLAY),
    )
    plv_window_cycles = float(
        st.sidebar.number_input(
            "PLV window (cycles)",
            min_value=0.1,
            value=float(WITHIN_TRIAL_PLV_DEFAULT_WINDOW_CYCLES),
            step=0.5,
        )
    )
    use_plv_min_window = st.sidebar.checkbox(
        "Minimum PLV window duration",
        value=WITHIN_TRIAL_PLV_MIN_WINDOW_DEFAULT_ENABLED,
    )
    plv_min_window_s = None
    if use_plv_min_window:
        plv_min_window_s = float(
            st.sidebar.number_input(
                "Minimum PLV window (s)",
                min_value=0.001,
                value=float(WITHIN_TRIAL_PLV_DEFAULT_MIN_WINDOW_S),
                step=0.01,
            )
        )
    use_plv_max_window = st.sidebar.checkbox(
        "Maximum PLV window duration",
        value=WITHIN_TRIAL_PLV_MAX_WINDOW_DEFAULT_ENABLED,
    )
    plv_max_window_s = None
    if use_plv_max_window:
        plv_max_window_s = float(
            st.sidebar.number_input(
                "Maximum PLV window (s)",
                min_value=0.001,
                value=float(WITHIN_TRIAL_PLV_DEFAULT_MAX_WINDOW_S),
                step=0.05,
            )
        )
    if plv_min_window_s is not None and plv_max_window_s is not None and plv_min_window_s > plv_max_window_s:
        st.error("Minimum PLV window duration cannot exceed the maximum.")
        st.stop()
    plv_min_valid_fraction = float(
        st.sidebar.slider(
            "Minimum valid PLV fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(WITHIN_TRIAL_PLV_DEFAULT_MIN_VALID_FRACTION),
            step=0.05,
        )
    )

    amplitude_mask_mode = st.sidebar.selectbox(
        "Low-amplitude masking",
        options=RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS,
    )
    amplitude_percentile = 10.0
    absolute_threshold_a = 0.0
    absolute_threshold_b = 0.0
    if amplitude_mask_mode == "Per-frequency percentile":
        amplitude_percentile = float(
            st.sidebar.slider("Amplitude percentile", min_value=0, max_value=100, value=10, step=1)
        )
    elif amplitude_mask_mode == "Absolute magnitude":
        absolute_threshold_a = float(
            st.sidebar.number_input("Site A magnitude threshold", min_value=0.0, value=0.0, format="%.6g")
        )
        absolute_threshold_b = float(
            st.sidebar.number_input("Site B magnitude threshold", min_value=0.0, value=0.0, format="%.6g")
        )

    source_site_a = probe_sources[probe_a]
    source_site_b = probe_sources[probe_b]
    source_path_a = str(source_site_a.lfp_file or "")
    sync_path_a = str(source_site_a.synchronization_file or "")
    source_path_b = str(source_site_b.lfp_file or "")
    sync_path_b = str(source_site_b.synchronization_file or "")
    source_a = Path(str(source_path_a)).expanduser()
    source_b = Path(str(source_path_b)).expanduser()
    for probe_label, raw_source, source in (
        (probe_a, source_path_a, source_a),
        (probe_b, source_path_b, source_b),
    ):
        if not str(raw_source).strip() or not source.is_file():
            st.error(f"LFP path for {probe_label} does not exist: {source}")
            st.stop()
    if source_a.resolve() == source_b.resolve() and channel_a == channel_b:
        st.error("Site A and site B resolve to the same LFP file and channel.")
        st.stop()
    aligned_sync_a = None
    aligned_sync_b = None
    aligned_sync_mtime_a = -1
    aligned_sync_mtime_b = -1
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        sync_a = Path(str(sync_path_a)).expanduser()
        sync_b = Path(str(sync_path_b)).expanduser()
        for probe_label, raw_sync, sync in (
            (probe_a, sync_path_a, sync_a),
            (probe_b, sync_path_b, sync_b),
        ):
            if not str(raw_sync).strip() or not sync.is_file():
                st.error(f"Aligned sync path for {probe_label} does not exist: {sync}")
                st.stop()
        aligned_sync_a = str(sync_a)
        aligned_sync_b = str(sync_b)
        aligned_sync_mtime_a = sync_a.stat().st_mtime_ns
        aligned_sync_mtime_b = sync_b.stat().st_mtime_ns

    site_a_label = f"{probe_a}, channel {channel_a}"
    site_b_label = f"{probe_b}, channel {channel_b}"
    open_ephys_tokens: tuple[OpenEphysCacheToken, ...] = ()
    try:
        if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
            if aligned_sync_a is None or aligned_sync_b is None:
                raise ValueError("Open Ephys relative phase requires both aligned sync paths")
            tokens_by_source: dict[tuple[str, str], OpenEphysCacheToken] = {}
            selected_tokens = []
            for source, sync in ((source_a, aligned_sync_a), (source_b, aligned_sync_b)):
                source_identity = (str(source.resolve()), str(Path(sync).resolve()))
                token = tokens_by_source.get(source_identity)
                if token is None:
                    token = build_open_ephys_cache_token(source, sync)
                    tokens_by_source[source_identity] = token
                selected_tokens.append(token)
            open_ephys_tokens = tuple(selected_tokens)
        result = compute_single_trial_relative_phase_cached(
            lfp_format=lfp_format,
            lfp_path_a=str(source_a),
            lfp_mtime_ns_a=source_a.stat().st_mtime_ns,
            channel_a=channel_a,
            site_a_label=site_a_label,
            aligned_sync_path_a=aligned_sync_a,
            aligned_sync_mtime_ns_a=aligned_sync_mtime_a,
            lfp_path_b=str(source_b),
            lfp_mtime_ns_b=source_b.stat().st_mtime_ns,
            channel_b=channel_b,
            site_b_label=site_b_label,
            aligned_sync_path_b=aligned_sync_b,
            aligned_sync_mtime_ns_b=aligned_sync_mtime_b,
            trial_index=trial_index,
            event_time_s=event_time_s,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            digital_word=0,
            irig_line=6,
            bit_period_s=1.0,
            utc_offset_hours=utc_offset_hours,
            frequencies_hz=tuple(frequencies_hz.tolist()),
            gaussian_width=gaussian_width,
            wavelet_window_length=wavelet_window_length,
            precision=LFP_SPECTROGRAM_PRECISION,
            norm=wavelet_norm,
            output_sample_rate_hz=RELATIVE_PHASE_OUTPUT_SAMPLE_RATE_HZ,
            notch_60_hz=notch_60_hz,
            notch_quality_factor=LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
            minimum_relative_magnitude=LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE,
            plv_window_cycles=plv_window_cycles,
            plv_min_window_s=plv_min_window_s,
            plv_max_window_s=plv_max_window_s,
            open_ephys_cache_token=(
                (open_ephys_tokens[0], open_ephys_tokens[1])
                if open_ephys_tokens
                else None
            ),
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should report pair-specific loading failures.
        st.error(f"Could not compute single-trial relative phase: {error}")
        st.stop()

    display_valid = lfp_phase_clustering.make_relative_phase_display_mask(
        result,
        mode=amplitude_mask_mode,
        percentile=amplitude_percentile,
        absolute_threshold_a=absolute_threshold_a,
        absolute_threshold_b=absolute_threshold_b,
    )
    support_valid = lfp_phase_clustering.make_relative_phase_support_mask(
        result,
        mode=amplitude_mask_mode,
        percentile=amplitude_percentile,
        absolute_threshold_a=absolute_threshold_a,
        absolute_threshold_b=absolute_threshold_b,
    )
    plv_result = lfp_phase_clustering.compute_within_trial_plv(
        relative_phase_complex=result.support_relative_phase_complex,
        valid_mask=support_valid,
        frequencies_hz=result.frequencies_hz,
        relative_time_s=result.support_relative_time_s,
        visible_window=(window_start_s, window_end_s),
        window_cycles=plv_window_cycles,
        min_window_s=plv_min_window_s,
        max_window_s=plv_max_window_s,
        min_valid_fraction=plv_min_valid_fraction,
        trial_index=trial_index,
        site_a_label=site_a_label,
        site_b_label=site_b_label,
    )
    lick_times = spike_behavior_pynapple.build_lick_time_dict(event_df)
    lfp_y_label = "LFP (uV)"
    figure, _axes = unit_spike_plotting.plot_trial_lfp_phase_analysis_and_behavior(
        trial_df=trial_df,
        trial_index=trial_index,
        lick_times=lick_times,
        phase_result=result,
        plv_result=plv_result,
        phase_display_valid_mask=display_valid,
        display_mode=display_mode,
        alignment_event=alignment_event,
        window=(window_start_s, window_end_s),
        lfp_y_label=lfp_y_label,
    )

    metadata_column, figure_column = st.columns([1, 3])
    with metadata_column:
        st.subheader("Single-Trial Phase Analysis")
        st.write(f"Trial row: {trial_index}")
        st.write(f"Condition: {condition}")
        st.write(f"Action: {action_label}")
        st.write(f"Alignment: {alignment_event}")
        st.write(f"Site A: {site_a_label}")
        st.write(f"Site B: {site_b_label}")
        st.write(f"Display: {display_mode}")
        st.write(f"PLV window: {plv_window_cycles:g} cycles")
        st.write(f"Masking: {amplitude_mask_mode}")
        st.write(f"Displayed pixels: {100.0 * float(np.mean(display_valid)):.1f}%")
    with figure_column:
        st.pyplot(figure, width="stretch")
        st.caption(
            "Phase color is instantaneous phase A minus phase B. PLV is local phase stability, "
            "not a significance test, and short windows have finite-sample upward bias. Trace panels "
            "show unprocessed source-rate LFP; notch and decimation apply only to the heatmaps."
        )

    if st.button("Save single-trial phase-analysis result"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        numeric_output_dir = (
            Path(session.session_data_home) / "ephys" / "derived" / "single_trial_phase_analysis" / timestamp
        )
        run_output_dir = (
            Path(session.session_data_home) / "analysis_runs" / f"single_trial_phase_analysis_{timestamp}"
        )
        metadata = {
            "generator": "src.neural_analysis.lfp_phase_clustering",
            "analysis_version": lfp_phase_clustering.ANALYSIS_VERSION,
            "session_id": session.sess_id_full,
            "trial_index": trial_index,
            "condition": condition,
            "action": action_label,
            "alignment_event": alignment_event,
            "window_s": [window_start_s, window_end_s],
            "site_a": site_a_label,
            "site_b": site_b_label,
            "source_path_a": str(source_a),
            "source_path_b": str(source_b),
            "lfp_format": lfp_format,
            "frequencies_hz": frequencies_hz.tolist(),
            "gaussian_width": gaussian_width,
            "window_length": wavelet_window_length,
            "normalization": wavelet_norm,
            "notch_60_hz": notch_60_hz,
            "comparison_sample_rate_hz": RELATIVE_PHASE_OUTPUT_SAMPLE_RATE_HZ,
            "display_mode": display_mode,
            "plv_window_cycles": plv_window_cycles,
            "plv_min_window_s": plv_min_window_s,
            "plv_max_window_s": plv_max_window_s,
            "plv_min_valid_fraction": plv_min_valid_fraction,
            "plv_interpretation": "within-trial local phase stability; finite-sample biased",
            "mask_mode": amplitude_mask_mode,
            "mask_percentile": amplitude_percentile,
            "absolute_threshold_a": absolute_threshold_a,
            "absolute_threshold_b": absolute_threshold_b,
            "axis_order": ["frequency", "time"],
            "phase_units": "radians, A minus B",
            "frequency_units": "Hz",
            "time_units": "s relative to alignment event",
            "random_seed": None,
        }
        if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
            metadata["source_lfp_units"] = "uV"
        metadata.update(_open_ephys_exploratory_provenance(lfp_format, open_ephys_tokens))
        lfp_phase_clustering.save_single_trial_phase_analysis_result(
            numeric_output_dir / "phase_and_plv.npz",
            phase_result=result,
            plv_result=plv_result,
            phase_display_valid_mask=display_valid,
            metadata=metadata,
        )
        run_output_dir.mkdir(parents=True, exist_ok=False)
        figure.savefig(run_output_dir / "phase_and_plv.png", dpi=200, bbox_inches="tight")
        (run_output_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# Single-trial phase analysis",
                    "",
                    f"Session: {session.sess_id_full}",
                    f"Trial row: {trial_index}",
                    f"Goal: inspect instantaneous phase and local PLV for {site_a_label} minus {site_b_label}.",
                    f"Numeric result: {numeric_output_dir / 'phase_and_plv.npz'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_output_dir / "run.log").write_text(
            f"Completed {datetime.now().isoformat()}\nParameters: {metadata!r}\n",
            encoding="utf-8",
        )
        st.success(f"Saved numeric result to {numeric_output_dir} and figure run to {run_output_dir}")
    plt.close(figure)

def render_lfp_phase_clustering_view(
    trial_df: pd.DataFrame,
    session: spike_behavior_pynapple.Session,
    hpc_v1_lfp_path: str = "",
    pfc_lfp_path: str = "",
    hpc_v1_aligned_spike_path: str = "",
    pfc_aligned_spike_path: str = "",
    *,
    metadata_lfp_sites: tuple[MetadataLFPSiteInputs, ...] = (),
) -> None:
    """
    Render and optionally save one-session LFP phase-clustering results.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Session trial table with shape ``(n_trials, n_columns)``. Event times
        are absolute seconds and trial conditions follow project conventions.
    session : spike_behavior_pynapple.Session
        Session metadata; ``session_data_home`` supplies the output root and
        ``sess_id_full`` identifies saved results.
    hpc_v1_lfp_path, pfc_lfp_path : str
        Probe-specific continuous LFP binary paths.
    hpc_v1_aligned_spike_path, pfc_aligned_spike_path : str
        Probe-specific aligned sync paths used for Open Ephys-derived LFP.
    metadata_lfp_sites : tuple[MetadataLFPSiteInputs, ...]
        Direct version-2 site records. Nonempty records replace the legacy
        manual path arguments without translating probe or site identities.

    Returns
    -------
    None
        Streamlit renders one selected ITPC site or ISPC pair. On request, all
        currently calculated frequency-by-time matrices are saved as NPZ files
        with frequencies in Hz and event-relative times in seconds.
    """

    st.sidebar.header("LFP Phase Clustering")
    metric = st.sidebar.selectbox("Analysis", options=LFP_PHASE_CLUSTERING_ANALYSIS_OPTIONS)
    condition = st.sidebar.selectbox("Trial condition", options=CONDITION_OPTIONS)
    alignment_event = st.sidebar.selectbox("Alignment event", options=ALIGNMENT_OPTIONS)
    window_start_s = float(
        st.sidebar.number_input(
            "Window start (s)", value=float(LFP_PHASE_CLUSTERING_DEFAULT_WINDOW[0]), step=0.1
        )
    )
    window_end_s = float(
        st.sidebar.number_input(
            "Window end (s)", value=float(LFP_PHASE_CLUSTERING_DEFAULT_WINDOW[1]), step=0.1
        )
    )
    lfp_format = st.sidebar.selectbox("LFP format", options=LFP_FORMAT_OPTIONS)
    utc_offset_hours = float(
        st.sidebar.number_input("LFP UTC offset (hours)", value=0, step=1, format="%d")
    )

    minimum_frequency_hz = float(
        st.sidebar.number_input(
            "Minimum frequency (Hz)",
            min_value=0.1,
            value=float(LFP_PHASE_CLUSTERING_MIN_FREQUENCY_HZ),
            step=0.5,
        )
    )
    maximum_frequency_hz = float(
        st.sidebar.number_input(
            "Maximum frequency (Hz)",
            min_value=0.2,
            value=float(LFP_PHASE_CLUSTERING_MAX_FREQUENCY_HZ),
            step=5.0,
        )
    )
    frequency_count = int(
        st.sidebar.number_input(
            "Frequency count",
            min_value=2,
            value=int(LFP_PHASE_CLUSTERING_FREQUENCY_COUNT),
            step=1,
        )
    )
    gaussian_width = float(
        st.sidebar.number_input(
            "Morlet Gaussian width",
            min_value=0.1,
            value=float(LFP_SPECTROGRAM_GAUSSIAN_WIDTH),
            step=0.1,
        )
    )
    wavelet_window_length = float(
        st.sidebar.number_input(
            "Morlet window length",
            min_value=0.1,
            value=float(LFP_SPECTROGRAM_WINDOW_LENGTH),
            step=0.1,
        )
    )
    wavelet_norm = st.sidebar.selectbox("Morlet normalization", options=["l1", "l2"])
    notch_60_hz = st.sidebar.checkbox("Apply 60 Hz notch", value=False)

    if window_start_s >= window_end_s:
        st.error("Window start must be less than window end.")
        st.stop()
    if minimum_frequency_hz >= maximum_frequency_hz:
        st.error("Maximum frequency must be greater than minimum frequency.")
        st.stop()

    lfp_sites = resolve_lfp_view_sites(
        metadata_sites=metadata_lfp_sites,
        hpc_v1_lfp_path=hpc_v1_lfp_path,
        pfc_lfp_path=pfc_lfp_path,
        hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
        pfc_aligned_spike_path=pfc_aligned_spike_path,
    )
    probe_sources = {site.display_label: site for site in lfp_sites}
    probe_options = list(probe_sources)
    requested_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []
    requested_sites: list[tuple[str, int]] = []
    if metric == "ITPC":
        site_count = int(st.sidebar.number_input("ITPC site count", min_value=1, max_value=8, value=1, step=1))
        for site_number in range(site_count):
            probe_label = st.sidebar.selectbox(
                f"Site {site_number + 1} probe",
                options=probe_options,
                key=f"phase_itpc_probe_{site_number}",
            )
            saved_channel_index = int(
                st.sidebar.number_input(
                    f"Site {site_number + 1} saved channel",
                    min_value=0,
                    value=int(probe_sources[probe_label].saved_channel_index),
                    step=1,
                    key=f"phase_itpc_channel_{site_number}",
                )
            )
            requested_sites.append((probe_label, saved_channel_index))
    else:
        pair_count = int(st.sidebar.number_input("ISPC pair count", min_value=1, max_value=8, value=1, step=1))
        for pair_number in range(pair_count):
            pair_sites = []
            for member_label, default_probe_index in (("A", 0), ("B", 1)):
                probe_label = st.sidebar.selectbox(
                    f"Pair {pair_number + 1} site {member_label} probe",
                    options=probe_options,
                    index=default_probe_index,
                    key=f"phase_ispc_probe_{pair_number}_{member_label}",
                )
                saved_channel_index = int(
                    st.sidebar.number_input(
                        f"Pair {pair_number + 1} site {member_label} saved channel",
                        min_value=0,
                        value=int(probe_sources[probe_label].saved_channel_index),
                        step=1,
                        key=f"phase_ispc_channel_{pair_number}_{member_label}",
                    )
                )
                pair_sites.append((probe_label, saved_channel_index))
            if pair_sites[0] == pair_sites[1]:
                st.error(f"ISPC pair {pair_number + 1} must contain two distinct sites.")
                st.stop()
            requested_pairs.append((pair_sites[0], pair_sites[1]))
            requested_sites.extend(pair_sites)

    unique_sites = list(dict.fromkeys(requested_sites))
    condition_masks = lfp_phase_clustering.make_phase_condition_masks(trial_df)
    event_values = pd.to_numeric(trial_df[alignment_event], errors="coerce").to_numpy(dtype=float)
    preprocessing_mask = condition_masks["all"] & np.isfinite(event_values)
    preprocessing_trial_indices = np.flatnonzero(preprocessing_mask)
    if preprocessing_trial_indices.size == 0:
        st.warning("No valid trials have a finite selected alignment time.")
        st.stop()
    frequencies_hz = np.geomspace(minimum_frequency_hz, maximum_frequency_hz, frequency_count)

    site_tensors = []
    site_labels = []
    source_paths = []
    open_ephys_tokens_by_source: dict[tuple[str, str], OpenEphysCacheToken] = {}
    for probe_label, saved_channel_index in unique_sites:
        source_site = probe_sources[probe_label]
        source_path = str(source_site.lfp_file or "")
        aligned_sync_path = str(source_site.synchronization_file or "")
        source = Path(str(source_path)).expanduser()
        if not str(source_path).strip() or not source.exists():
            st.error(f"LFP path for {probe_label} does not exist: {source_path}")
            st.stop()
        aligned_sync_argument = None
        open_ephys_cache_token = None
        if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
            sync = Path(str(aligned_sync_path)).expanduser()
            if not str(aligned_sync_path).strip() or not sync.is_file():
                st.error(f"Aligned sync path for {probe_label} does not exist: {sync}")
                st.stop()
            aligned_sync_argument = str(sync)
            source_identity = (str(source.resolve()), str(sync.resolve()))
            open_ephys_cache_token = open_ephys_tokens_by_source.get(source_identity)
            if open_ephys_cache_token is None:
                open_ephys_cache_token = build_open_ephys_cache_token(source, sync)
                open_ephys_tokens_by_source[source_identity] = open_ephys_cache_token
        site_label = f"{probe_label}, channel {saved_channel_index}"
        try:
            site_tensor = compute_lfp_phase_site_cached(
                lfp_format=lfp_format,
                lfp_path=str(source),
                lfp_file_mtime_ns=source.stat().st_mtime_ns,
                saved_channel_index=saved_channel_index,
                event_times_s=tuple(event_values[preprocessing_trial_indices].tolist()),
                trial_indices=tuple(preprocessing_trial_indices.tolist()),
                window_start_s=window_start_s,
                window_end_s=window_end_s,
                digital_word=0,
                irig_line=6,
                bit_period_s=1.0,
                utc_offset_hours=utc_offset_hours,
                frequencies_hz=tuple(frequencies_hz.tolist()),
                gaussian_width=gaussian_width,
                window_length=wavelet_window_length,
                precision=LFP_SPECTROGRAM_PRECISION,
                norm=wavelet_norm,
                output_sample_rate_hz=LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ,
                notch_60_hz=notch_60_hz,
                notch_quality_factor=LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
                minimum_relative_magnitude=LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE,
                maximum_core_duration_s=LFP_PHASE_CLUSTERING_MAXIMUM_CORE_DURATION_S,
                aligned_sync_npz_path=aligned_sync_argument,
                open_ephys_cache_token=open_ephys_cache_token,
            )
        except Exception as error:  # noqa: BLE001 - Streamlit should report site-specific loading failures.
            st.error(f"Could not preprocess {site_label}: {error}")
            st.stop()
        site_tensors.append(site_tensor)
        site_labels.append(site_label)
        source_paths.append(str(source))

    try:
        combined_tensor = lfp_phase_clustering.combine_phase_trial_tensors(site_tensors)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    selected_trial_mask = lfp_phase_clustering.select_tensor_trial_mask(
        combined_tensor.trial_indices,
        condition_masks[condition],
    )
    selected_trial_indices = combined_tensor.trial_indices[selected_trial_mask]
    if selected_trial_indices.size == 0:
        st.warning(f"No retained trials match condition {condition!r}.")
        st.stop()

    calculated_results: list[tuple[str, lfp_phase_clustering.PhaseClusteringResult]] = []
    if metric == "ITPC":
        all_site_result = lfp_phase_clustering.compute_itpc(
            combined_tensor.phase,
            trial_mask=selected_trial_mask,
            valid_mask=combined_tensor.valid,
        )
        for site_index, site_label in enumerate(site_labels):
            calculated_results.append(
                (
                    site_label,
                    lfp_phase_clustering.PhaseClusteringResult(
                        values=all_site_result.values[site_index],
                        effective_trial_count=all_site_result.effective_trial_count[site_index],
                        n_trials=all_site_result.n_trials,
                    ),
                )
            )
    else:
        site_index_by_spec = {site_spec: index for index, site_spec in enumerate(unique_sites)}
        for site_a, site_b in requested_pairs:
            pair_label = f"{site_labels[site_index_by_spec[site_a]]} vs {site_labels[site_index_by_spec[site_b]]}"
            pair_result = lfp_phase_clustering.compute_ispc(
                combined_tensor.phase,
                site_a=site_index_by_spec[site_a],
                site_b=site_index_by_spec[site_b],
                trial_mask=selected_trial_mask,
                valid_mask=combined_tensor.valid,
            )
            calculated_results.append((pair_label, pair_result))

    result_labels = [label for label, _result in calculated_results]
    displayed_label = st.selectbox("Displayed site/pair", options=result_labels)
    displayed_result = calculated_results[result_labels.index(displayed_label)][1]
    figure, _axis = lfp_phase_clustering.plot_phase_clustering(
        displayed_result.values,
        frequencies_hz=frequencies_hz,
        relative_time_s=combined_tensor.relative_time_s,
        metric_label=metric,
        site_label=displayed_label,
        condition_label=condition,
        n_trials=displayed_result.n_trials,
        figure_size=(9.0, 4.8),
    )

    metadata_column, figure_column = st.columns([1, 3])
    with metadata_column:
        st.subheader("LFP Phase Clustering")
        st.write(f"Metric: {metric}")
        st.write(f"Condition: {condition}")
        st.write(f"Alignment: {alignment_event}")
        st.write(f"Selected trials: {selected_trial_indices.size}")
        st.write(f"Excluded during preprocessing: {combined_tensor.excluded_trial_indices.size}")
        st.write(
            "Effective trials per pixel: "
            f"{int(displayed_result.effective_trial_count.min())}-"
            f"{int(displayed_result.effective_trial_count.max())}"
        )
        st.caption(
            "Switch/stay use the established current unrewarded-trial restriction and compare "
            "the current choice with the next row's choice."
        )
    with figure_column:
        st.pyplot(figure, width="stretch")
        st.caption(
            "Phase consistency is bounded from 0 to 1. The continuous LFP is transformed before "
            "event-aligned trial extraction; no power baseline normalization is applied."
        )

    if st.button("Save phase-clustering results"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        numeric_output_dir = (
            Path(session.session_data_home) / "ephys" / "derived" / "phase_clustering" / timestamp
        )
        run_output_dir = Path(session.session_data_home) / "analysis_runs" / f"lfp_phase_clustering_{timestamp}"
        common_metadata = {
            "generator": "src.neural_analysis.lfp_phase_clustering",
            "analysis_version": lfp_phase_clustering.ANALYSIS_VERSION,
            "session_id": session.sess_id_full,
            "metric": metric,
            "condition": condition,
            "alignment_event": alignment_event,
            "window_s": [window_start_s, window_end_s],
            "frequencies_hz": frequencies_hz.tolist(),
            "gaussian_width": gaussian_width,
            "window_length": wavelet_window_length,
            "normalization": wavelet_norm,
            "notch_60_hz": notch_60_hz,
            "output_sample_rate_hz": LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ,
            "lfp_format": lfp_format,
            "utc_offset_hours": utc_offset_hours,
            "site_labels": site_labels,
            "source_paths": source_paths,
            "excluded_trial_indices": combined_tensor.excluded_trial_indices.tolist(),
            "axis_order": ["frequency", "time"],
            "value_units": "dimensionless phase consistency",
            "frequency_units": "Hz",
            "time_units": "s relative to alignment event",
            "random_seed": None,
        }
        common_metadata.update(
            _open_ephys_exploratory_provenance(
                lfp_format,
                tuple(open_ephys_tokens_by_source.values()),
            )
        )
        for result_number, (result_label, result) in enumerate(calculated_results, start=1):
            result_metadata = dict(common_metadata)
            result_metadata["site_or_pair"] = result_label
            lfp_phase_clustering.save_phase_clustering_result(
                numeric_output_dir / f"result_{result_number:02d}_{metric.lower()}.npz",
                values=result.values,
                effective_trial_count=result.effective_trial_count,
                frequencies_hz=frequencies_hz,
                relative_time_s=combined_tensor.relative_time_s,
                trial_indices=selected_trial_indices,
                metadata=result_metadata,
            )
        run_output_dir.mkdir(parents=True, exist_ok=False)
        figure.savefig(run_output_dir / f"{metric.lower()}_heatmap.png", dpi=200, bbox_inches="tight")
        (run_output_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# LFP phase clustering",
                    "",
                    f"Session: {session.sess_id_full}",
                    f"Goal: {metric} around {alignment_event} for {condition} trials.",
                    f"Numeric results: {numeric_output_dir}",
                    f"Sites/pairs: {', '.join(result_labels)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_output_dir / "run.log").write_text(
            f"Completed {datetime.now().isoformat()}\nParameters: {common_metadata!r}\n",
            encoding="utf-8",
        )
        st.success(f"Saved numeric results to {numeric_output_dir} and figure run to {run_output_dir}")
    plt.close(figure)

def build_lfp_dropdown_options(hpc_v1_lfp_path: str, pfc_lfp_path: str) -> dict[str, str]:
    """
    Build explicit LFP-file choices from the two probe path inputs.

    Parameters
    ----------
    hpc_v1_lfp_path : str
        User-entered HPC/V1 LFP ``.lf.bin`` path. Empty strings are preserved
        so the UI can warn without changing the user's input.
    pfc_lfp_path : str
        User-entered PFC LFP ``.lf.bin`` path. Empty strings are preserved so
        the UI can warn without changing the user's input.

    Returns
    -------
    dict[str, str]
        Mapping from dropdown label to the exact path string entered by the
        user. No directory discovery is performed.
    """

    return {
        LFP_DROPDOWN_LABEL_HPC_V1: str(hpc_v1_lfp_path),
        LFP_DROPDOWN_LABEL_PFC: str(pfc_lfp_path),
    }

def resolve_lfp_filter_band(filter_label: str) -> tuple[float, float] | None:
    """
    Convert one LFP filter display label into cutoff frequencies.

    Parameters
    ----------
    filter_label : str
        Display label from ``LFP_FILTER_BANDS``.

    Returns
    -------
    tuple[float, float] | None
        Bandpass cutoffs in Hz as ``(low_hz, high_hz)``. ``None`` means the
        gain-corrected LFP trace should be shown unfiltered.
    """

    if filter_label not in LFP_FILTER_BANDS:
        raise ValueError(f"Unknown LFP filter label: {filter_label}")
    frequency_band_hz = LFP_FILTER_BANDS[filter_label]
    if frequency_band_hz is None:
        return None
    return (float(frequency_band_hz[0]), float(frequency_band_hz[1]))
