"""Cached Spike-LFP calculations and interactive Spike-LFP views."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import unit_spike_plotting
from src.neural_analysis.lfp import spectrogram as lfp_spectrogram
from src.neural_analysis.spike_behavior import loading as spike_behavior_pynapple
from src.neural_analysis.spike_behavior import loading as unit_spike_loading
from src.neural_analysis.spike_lfp import hilbert as spike_lfp_hilbert_phase
from src.neural_analysis.spike_lfp import phase_locking as spike_lfp_phase_locking
from src.neural_analysis.webapp.data_loading import (
    OpenEphysCacheToken,
    _open_ephys_cache_token_or_none,
    _open_ephys_exploratory_provenance,
    build_open_ephys_cache_token,
    load_trial_lfp_trace_for_format_with_sample_rate,
)
from src.neural_analysis.webapp.lfp_views import (
    LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE,
    LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ,
    LFP_SPECTROGRAM_GAUSSIAN_WIDTH,
    LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
    LFP_SPECTROGRAM_PRECISION,
    LFP_SPECTROGRAM_WINDOW_LENGTH,
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
)


SPIKE_LFP_PHASE_DEFAULT_WINDOW = (-0.5, 0.5)

SPIKE_LFP_PHASE_MIN_FREQUENCY_HZ = 2.0

SPIKE_LFP_PHASE_MAX_FREQUENCY_HZ = 100.0

SPIKE_LFP_PHASE_FREQUENCY_COUNT = 50

SPIKE_LFP_PHASE_DEFAULT_POLAR_FREQUENCY_HZ = 8.0

SPIKE_LFP_PHASE_BIN_COUNT = 24

SPIKE_LFP_PHASE_AMPLITUDE_MASK_OPTIONS = ("Off", "Absolute magnitude")

SPIKE_LFP_PHASE_MAXIMUM_CORE_DURATION_S = 120.0

SPIKE_LFP_PHASE_CACHE_MAX_ENTRIES = 6

SPIKE_LFP_HILBERT_DEFAULT_WINDOW = (-1.0, 2.0)

SPIKE_LFP_HILBERT_PHASE_BAND_HZ = (6.0, 10.0)

SPIKE_LFP_HILBERT_FILTER_PADDING_S = 1.0

SPIKE_LFP_HILBERT_MINIMUM_ENVELOPE = 0.0

SPIKE_LFP_HILBERT_CACHE_MAX_ENTRIES = 12

@st.cache_data(
    show_spinner="Computing spike-LFP phase locking...",
    max_entries=SPIKE_LFP_PHASE_CACHE_MAX_ENTRIES,
)
def _compute_spike_lfp_phase_locking_cached(
    lfp_format: str,
    lfp_path: str,
    lfp_mtime_ns: int,
    saved_channel_index: int,
    lfp_site_label: str,
    aligned_sync_path: str | None,
    aligned_sync_mtime_ns: int,
    unit_id: int,
    unit_spike_times_s: tuple[float, ...],
    trial_indices: tuple[int, ...],
    event_times_s: tuple[float, ...],
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
    target_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    minimum_relative_magnitude: float,
    absolute_amplitude_threshold: float,
    maximum_core_duration_s: float,
    phase_bin_count: int,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> spike_lfp_phase_locking.SpikePhaseLockingResult:
    """
    Load bounded continuous LFP blocks and pool one unit's trial-window phases.

    Parameters
    ----------
    lfp_format : str
        Shared acquisition format from ``LFP_FORMAT_OPTIONS``.
    lfp_path : str
        SpikeGLX ``.lf.bin`` or Open Ephys-derived ``lfp.dat`` path.
    lfp_mtime_ns : int
        LFP modification time in nanoseconds used for cache invalidation.
    saved_channel_index : int
        Zero-based LFP saved-channel index.
    lfp_site_label : str
        Human-readable independent probe/channel label.
    aligned_sync_path : str | None
        Open Ephys aligned sync path; ``None`` for SpikeGLX.
    aligned_sync_mtime_ns : int
        Sync-file modification time in nanoseconds or ``-1`` when unused.
    unit_id : int
        Source sorter cluster identifier.
    unit_spike_times_s : tuple[float, ...]
        Unit spike timestamps in synchronized absolute seconds.
    trial_indices : tuple[int, ...]
        Selected integer source trial rows.
    event_times_s : tuple[float, ...]
        Selected chronological alignment timestamps in absolute seconds.
    window_start_s, window_end_s : float
        Event-relative half-open analysis bounds in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization word and IRIG line.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds.
    utc_offset_hours : float
        Constant synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        Positive Morlet frequencies in Hz.
    gaussian_width, wavelet_window_length : float
        Dimensionless Pynapple Morlet parameters.
    precision : int
        Base-2 Morlet precision.
    norm : str
        Pynapple wavelet normalization.
    target_sample_rate_hz : float
        Preferred post-decimation wavelet sample rate in Hz.
    notch_60_hz : bool
        Whether to apply the existing 60 Hz notch before decimation.
    notch_quality_factor : float
        Dimensionless notch quality factor.
    minimum_relative_magnitude : float
        Relative numerical coefficient threshold.
    absolute_amplitude_threshold : float
        Source-dependent wavelet magnitude threshold applied to continuous
        phase occupancy and interpolated spike samples.
    maximum_core_duration_s : float
        Maximum unpadded transform block span in seconds.
    phase_bin_count : int
        Number of equal-width phase bins spanning ``[-pi, pi]``. This is an
        explicit cache parameter so phase-tuning results cannot be reused
        across different binning settings.
    open_ephys_cache_token : OpenEphysCacheToken or None, optional
        Complete physical-uV source identity used only in this decorated cache
        key. It contains no retained LFP samples or phase values.

    Returns
    -------
    spike_lfp_phase_locking.SpikePhaseLockingResult
        Frequency metrics, retained spike-sample arrays with shape
        ``(frequency, selected_spike)``, and phase-tuning arrays with shape
        ``(frequency, phase_bin)``.
    """

    del lfp_mtime_ns, aligned_sync_mtime_ns, open_ephys_cache_token
    frequencies = np.asarray(frequencies_hz, dtype=float)
    padding_s = lfp_spectrogram.compute_wavelet_padding_s(
        minimum_frequency_hz=float(np.min(frequencies)),
        window_length=float(wavelet_window_length),
    )

    def load_absolute_block(start_s: float, end_s: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Load one synchronized absolute LFP interval and preserve source units."""

        center_s = (float(start_s) + float(end_s)) / 2.0
        relative_time_s, lfp_values, sample_rate_hz = load_trial_lfp_trace_for_format_with_sample_rate(
            lfp_format=lfp_format,
            lfp_path=lfp_path,
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=center_s,
            window_start_s=float(start_s) - center_s,
            window_end_s=float(end_s) - center_s,
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
            aligned_sync_npz_path=aligned_sync_path,
        )
        return center_s + relative_time_s, lfp_values, float(sample_rate_hz)

    return spike_lfp_phase_locking.compute_trial_aligned_spike_phase_locking(
        unit_spike_times_s=np.asarray(unit_spike_times_s, dtype=float),
        event_times_s=np.asarray(event_times_s, dtype=float),
        trial_indices=np.asarray(trial_indices, dtype=int),
        window=(float(window_start_s), float(window_end_s)),
        frequencies_hz=frequencies,
        wavelet_padding_s=padding_s,
        maximum_core_duration_s=float(maximum_core_duration_s),
        block_loader=load_absolute_block,
        unit_id=int(unit_id),
        lfp_site_label=lfp_site_label,
        gaussian_width=float(gaussian_width),
        wavelet_window_length=float(wavelet_window_length),
        precision=int(precision),
        norm=norm,
        target_sample_rate_hz=float(target_sample_rate_hz),
        notch_60_hz=bool(notch_60_hz),
        notch_quality_factor=float(notch_quality_factor),
        minimum_relative_magnitude=float(minimum_relative_magnitude),
        absolute_amplitude_threshold=float(absolute_amplitude_threshold),
        phase_bin_count=int(phase_bin_count),
    )

def compute_spike_lfp_phase_locking_cached(
    lfp_format: str,
    lfp_path: str,
    lfp_mtime_ns: int,
    saved_channel_index: int,
    lfp_site_label: str,
    aligned_sync_path: str | None,
    aligned_sync_mtime_ns: int,
    unit_id: int,
    unit_spike_times_s: tuple[float, ...],
    trial_indices: tuple[int, ...],
    event_times_s: tuple[float, ...],
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
    target_sample_rate_hz: float,
    notch_60_hz: bool,
    notch_quality_factor: float,
    minimum_relative_magnitude: float,
    absolute_amplitude_threshold: float,
    maximum_core_duration_s: float,
    phase_bin_count: int,
    *,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> spike_lfp_phase_locking.SpikePhaseLockingResult:
    """Compute one unit/site phase-locking result through a source cache.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label selecting Open Ephys or SpikeGLX behavior.
    lfp_path : str
        Selected LFP binary filesystem path with no numerical array axis.
    aligned_sync_path : str or None
        Open Ephys synchronization path, required for that format and unused
        for SpikeGLX.
    lfp_mtime_ns, aligned_sync_mtime_ns : int
        Existing compatibility cache identities in nanoseconds.
    saved_channel_index, unit_id : int
        Zero-based LFP saved channel and sorter unit identifiers.
    lfp_site_label : str
        Human-readable probe/channel label with no physical unit.
    unit_spike_times_s, event_times_s : tuple[float, ...]
        One-dimensional absolute timestamps in seconds.
    trial_indices : tuple[int, ...]
        One-dimensional source trial-row axis aligned with event times.
    window_start_s, window_end_s, bit_period_s, maximum_core_duration_s : float
        Event-relative bounds, IRIG period, and block-duration limit in
        seconds.
    digital_word, irig_line, phase_bin_count : int
        SpikeGLX synchronization selectors and phase-bin count.
    utc_offset_hours : float
        Synchronization offset in hours.
    frequencies_hz : tuple[float, ...]
        One-dimensional positive Morlet frequency axis in Hz.
    gaussian_width, wavelet_window_length, notch_quality_factor : float
        Dimensionless transform/filter settings.
    precision : int
        Base-2 Morlet precision.
    norm : str
        Morlet normalization label.
    target_sample_rate_hz : float
        Positive phase time-axis rate in Hz.
    notch_60_hz : bool
        Whether to apply the existing 60 Hz notch.
    minimum_relative_magnitude, absolute_amplitude_threshold : float
        Dimensionless numerical-validity and source-amplitude thresholds.
    open_ephys_cache_token : OpenEphysCacheToken or None, keyword-only
        Optional complete physical-uV source identity, constructed before
        cache keying only when omitted.

    Returns
    -------
    spike_lfp_phase_locking.SpikePhaseLockingResult
        Frequency metrics and retained phase observations. Frequency is Hz,
        phase is radians/dimensionless, and source LFP remains uV for Open
        Ephys before phase conversion.

    Raises
    ------
    ValueError
        If source/synchronization/token bindings, spike/trial axes, or phase
        transform/threshold settings are invalid before cache execution.
    OSError
        If production LFP loading fails.
    """
    token = _open_ephys_cache_token_or_none(
        lfp_format, lfp_path, aligned_sync_path, open_ephys_cache_token
    )
    return _compute_spike_lfp_phase_locking_cached(
        lfp_format, lfp_path, lfp_mtime_ns, saved_channel_index, lfp_site_label,
        aligned_sync_path, aligned_sync_mtime_ns, unit_id, unit_spike_times_s,
        trial_indices, event_times_s, window_start_s, window_end_s, digital_word,
        irig_line, bit_period_s, utc_offset_hours, frequencies_hz, gaussian_width,
        wavelet_window_length, precision, norm, target_sample_rate_hz,
        notch_60_hz, notch_quality_factor, minimum_relative_magnitude,
        absolute_amplitude_threshold, maximum_core_duration_s, phase_bin_count, token,
    )

compute_spike_lfp_phase_locking_cached.clear = _compute_spike_lfp_phase_locking_cached.clear

@st.cache_data(
    show_spinner="Computing single-trial spike-LFP Hilbert phase...",
    max_entries=SPIKE_LFP_HILBERT_CACHE_MAX_ENTRIES,
)
def _compute_single_trial_spike_lfp_hilbert_cached(
    lfp_format: str,
    lfp_path: str,
    lfp_mtime_ns: int,
    saved_channel_index: int,
    lfp_site_label: str,
    aligned_sync_path: str | None,
    aligned_sync_mtime_ns: int,
    unit_id: int,
    unit_spike_times_s: tuple[float, ...],
    trial_index: int,
    event_time_s: float,
    window_start_s: float,
    window_end_s: float,
    band_low_hz: float,
    band_high_hz: float,
    filter_padding_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    minimum_envelope: float,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> spike_lfp_hilbert_phase.SingleTrialSpikeLFPHilbertResult:
    """Load one padded raw LFP interval and compute visible Hilbert phase.

    Parameters
    ----------
    lfp_format : str
        Shared acquisition format from ``LFP_FORMAT_OPTIONS``.
    lfp_path : str
        Continuous LFP source file.
    lfp_mtime_ns : int
        LFP modification timestamp in nanoseconds used only to invalidate this
        Streamlit cache entry.
    saved_channel_index : int
        Zero-based channel index in the LFP source file.
    lfp_site_label : str
        Human-readable independent LFP probe/channel label.
    aligned_sync_path : str | None
        Open Ephys synchronization path, or ``None`` for SpikeGLX.
    aligned_sync_mtime_ns : int
        Synchronization-file modification timestamp used only for cache
        invalidation, or ``-1`` when no file is needed.
    unit_id : int
        Selected sorter cluster identifier.
    unit_spike_times_s : tuple[float, ...]
        Synchronized absolute unit spike timestamps in seconds.
    trial_index : int
        Source trial-table row.
    event_time_s : float
        Absolute synchronized alignment-event timestamp in seconds.
    window_start_s, window_end_s : float
        Visible half-open event-relative interval in seconds.
    band_low_hz, band_high_hz : float
        Fixed Hilbert bandpass cutoffs in Hz.
    filter_padding_s : float
        Continuous raw-LFP padding on both sides in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization settings; ignored by Open Ephys.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds; ignored by Open Ephys.
    utc_offset_hours : float
        Constant synchronization offset in hours.
    minimum_envelope : float
        Nonnegative source-unit Hilbert envelope validity threshold.
    open_ephys_cache_token : OpenEphysCacheToken or None, optional
        Complete physical-uV source identity used only in this decorated cache
        key. The public wrapper validates/binds it before the raw trace loads.

    Returns
    -------
    spike_lfp_hilbert_phase.SingleTrialSpikeLFPHilbertResult
        Visible source-rate LFP, 6--10 Hz Hilbert phase, and spike-phase
        observations. No Morlet transform, notch, or decimation is used.
    """

    # These values are cache-key inputs, not numerical computation inputs.
    del lfp_mtime_ns, aligned_sync_mtime_ns, open_ephys_cache_token
    padded_start_s = float(window_start_s) - float(filter_padding_s)
    padded_end_s = float(window_end_s) + float(filter_padding_s)
    relative_time_s, raw_lfp, source_sample_rate_hz = load_trial_lfp_trace_for_format_with_sample_rate(
        lfp_format=lfp_format,
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        alignment_time_s=float(event_time_s),
        window_start_s=padded_start_s,
        window_end_s=padded_end_s,
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
        aligned_sync_npz_path=aligned_sync_path,
    )
    return spike_lfp_hilbert_phase.compute_single_trial_spike_lfp_hilbert(
        padded_relative_time_s=relative_time_s,
        padded_raw_lfp=raw_lfp,
        source_sample_rate_hz=float(source_sample_rate_hz),
        unit_spike_times_absolute_s=np.asarray(unit_spike_times_s, dtype=float),
        event_time_s=float(event_time_s),
        visible_window=(float(window_start_s), float(window_end_s)),
        frequency_band_hz=(float(band_low_hz), float(band_high_hz)),
        filter_padding_s=float(filter_padding_s),
        minimum_envelope=float(minimum_envelope),
        trial_index=int(trial_index),
        unit_id=int(unit_id),
        lfp_site_label=lfp_site_label,
    )

def compute_single_trial_spike_lfp_hilbert_cached(
    lfp_format: str,
    lfp_path: str,
    lfp_mtime_ns: int,
    saved_channel_index: int,
    lfp_site_label: str,
    aligned_sync_path: str | None,
    aligned_sync_mtime_ns: int,
    unit_id: int,
    unit_spike_times_s: tuple[float, ...],
    trial_index: int,
    event_time_s: float,
    window_start_s: float,
    window_end_s: float,
    band_low_hz: float,
    band_high_hz: float,
    filter_padding_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    minimum_envelope: float,
    *,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
) -> spike_lfp_hilbert_phase.SingleTrialSpikeLFPHilbertResult:
    """Compute one trial's Hilbert phase through a source-identity cache.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label selecting Open Ephys or SpikeGLX behavior.
    lfp_path : str
        Selected LFP binary filesystem path with no numerical array axis.
    aligned_sync_path : str or None
        Open Ephys synchronization path, required for that format and unused
        for SpikeGLX.
    lfp_mtime_ns, aligned_sync_mtime_ns : int
        Existing compatibility cache identities in nanoseconds.
    saved_channel_index, unit_id, trial_index : int
        Zero-based saved-channel, sorter-unit, and trial-row identifiers.
    lfp_site_label : str
        Human-readable source site label with no physical unit.
    unit_spike_times_s : tuple[float, ...]
        One-dimensional absolute spike timestamps in seconds.
    event_time_s, window_start_s, window_end_s, filter_padding_s, bit_period_s : float
        Absolute alignment, relative visible window, filter padding, and
        SpikeGLX IRIG period in seconds.
    band_low_hz, band_high_hz : float
        Positive bandpass edges in Hz.
    digital_word, irig_line : int
        SpikeGLX synchronization selectors.
    utc_offset_hours : float
        Synchronization offset in hours.
    minimum_envelope : float
        Nonnegative physical-uV Hilbert-envelope validity threshold for Open
        Ephys source values.
    open_ephys_cache_token : OpenEphysCacheToken or None, keyword-only
        Optional complete source identity. Omission builds it before cache
        keying; a supplied valid token is reused without sidecar hashing.

    Returns
    -------
    spike_lfp_hilbert_phase.SingleTrialSpikeLFPHilbertResult
        One-dimensional visible raw/filtered LFP arrays in uV for Open Ephys,
        plus phase in radians and spike observations on seconds-based axes.

    Raises
    ------
    ValueError
        If source/synchronization/token bindings, selected trial/spike inputs,
        Hilbert band, or validity threshold are invalid before cache execution.
    OSError
        If production LFP loading fails.
    """
    token = _open_ephys_cache_token_or_none(
        lfp_format, lfp_path, aligned_sync_path, open_ephys_cache_token
    )
    return _compute_single_trial_spike_lfp_hilbert_cached(
        lfp_format, lfp_path, lfp_mtime_ns, saved_channel_index, lfp_site_label,
        aligned_sync_path, aligned_sync_mtime_ns, unit_id, unit_spike_times_s,
        trial_index, event_time_s, window_start_s, window_end_s, band_low_hz,
        band_high_hz, filter_padding_s, digital_word, irig_line, bit_period_s,
        utc_offset_hours, minimum_envelope, token,
    )

compute_single_trial_spike_lfp_hilbert_cached.clear = _compute_single_trial_spike_lfp_hilbert_cached.clear

def render_spike_lfp_phase_locking_view(
    trial_df: pd.DataFrame,
    session: spike_behavior_pynapple.Session,
    spike_group: object,
    unit_ids: np.ndarray,
    active_probe_label: str,
    hpc_v1_lfp_path: str,
    pfc_lfp_path: str,
    hpc_v1_aligned_spike_path: str,
    pfc_aligned_spike_path: str,
) -> None:
    """
    Render and optionally save pooled trial-aligned spike-LFP phase locking.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Session trial table with shape ``(n_trials, n_columns)`` and absolute
        synchronized event times in seconds.
    session : spike_behavior_pynapple.Session
        Session metadata whose data root receives timestamped outputs.
    spike_group : object
        Pynapple ``TsGroup`` keyed by integer cluster id, with timestamps in
        synchronized absolute seconds.
    unit_ids : np.ndarray
        Selectable integer unit ids with shape ``(n_units,)`` from the active
        probe's current region and quality filters.
    active_probe_label : str
        Probe owning the selected units. The LFP site remains independently
        selectable.
    hpc_v1_lfp_path, pfc_lfp_path : str
        Probe-specific continuous LFP binary paths.
    hpc_v1_aligned_spike_path, pfc_aligned_spike_path : str
        Probe-specific aligned sync paths for Open Ephys-derived LFP.

    Returns
    -------
    None
        Streamlit renders frequency metrics, a polar occupancy-normalized
        firing-rate histogram in Hz, and
        optional timestamped NPZ/PNG outputs.
    """

    st.sidebar.header("Spike-LFP Phase Locking")
    available_unit_ids = np.asarray(unit_ids, dtype=int).reshape(-1)
    selected_unit_id = int(st.sidebar.selectbox("Unit", options=available_unit_ids.tolist()))
    condition = st.sidebar.selectbox("Trial condition", options=CONDITION_OPTIONS)
    action_options = {label: value for label, value in ACTION_OPTIONS.items() if value != COMPARE_LEFT_RIGHT_ACTION}
    action_label = st.sidebar.selectbox("Action", options=list(action_options))
    alignment_event = st.sidebar.selectbox("Alignment event", options=ALIGNMENT_OPTIONS)
    window_start_s = float(
        st.sidebar.number_input(
            "Window start (s)",
            value=float(SPIKE_LFP_PHASE_DEFAULT_WINDOW[0]),
            step=0.1,
        )
    )
    window_end_s = float(
        st.sidebar.number_input(
            "Window end (s)",
            value=float(SPIKE_LFP_PHASE_DEFAULT_WINDOW[1]),
            step=0.1,
        )
    )
    if window_start_s >= window_end_s:
        st.error("Window start must be less than window end.")
        st.stop()

    selected_trial_indices = unit_spike_plotting.filter_trials_for_unit_plot(
        trial_df=trial_df,
        condition=condition,
        action=action_options[action_label],
    )
    alignment_times = pd.to_numeric(trial_df[alignment_event], errors="coerce").to_numpy(dtype=float)
    finite_alignment = np.isfinite(alignment_times[selected_trial_indices])
    selected_trial_indices = selected_trial_indices[finite_alignment]
    if selected_trial_indices.size == 0:
        st.warning("No trials match the selected condition, action, and alignment event.")
        st.stop()

    lfp_format = st.sidebar.selectbox("LFP format", options=LFP_FORMAT_OPTIONS)
    utc_offset_hours = float(
        st.sidebar.number_input("LFP UTC offset (hours)", value=0, step=1, format="%d")
    )
    probe_options = [LFP_DROPDOWN_LABEL_HPC_V1, LFP_DROPDOWN_LABEL_PFC]
    default_probe_index = 0 if active_probe_label == unit_spike_loading.PROBE_LABEL_HPC_V1 else 1
    lfp_probe_label = st.sidebar.selectbox("LFP probe", options=probe_options, index=default_probe_index)
    lfp_saved_channel_index = int(
        st.sidebar.number_input("LFP saved channel", min_value=0, value=0, step=1)
    )

    minimum_frequency_hz = float(
        st.sidebar.number_input(
            "Minimum frequency (Hz)",
            min_value=0.1,
            value=float(SPIKE_LFP_PHASE_MIN_FREQUENCY_HZ),
            step=0.5,
        )
    )
    maximum_frequency_hz = float(
        st.sidebar.number_input(
            "Maximum frequency (Hz)",
            min_value=0.2,
            value=float(SPIKE_LFP_PHASE_MAX_FREQUENCY_HZ),
            step=5.0,
        )
    )
    frequency_count = int(
        st.sidebar.number_input(
            "Frequency count",
            min_value=2,
            value=int(SPIKE_LFP_PHASE_FREQUENCY_COUNT),
            step=1,
        )
    )
    if minimum_frequency_hz >= maximum_frequency_hz:
        st.error("Maximum frequency must be greater than minimum frequency.")
        st.stop()
    frequencies_hz = np.geomspace(minimum_frequency_hz, maximum_frequency_hz, frequency_count)
    polar_default_index = int(np.argmin(np.abs(frequencies_hz - SPIKE_LFP_PHASE_DEFAULT_POLAR_FREQUENCY_HZ)))
    polar_frequency_hz = float(
        st.sidebar.selectbox(
            "Polar frequency (Hz)",
            options=frequencies_hz.tolist(),
            index=polar_default_index,
            format_func=lambda value: f"{float(value):.3g}",
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
    amplitude_mask_mode = st.sidebar.selectbox(
        "Low-amplitude masking",
        options=SPIKE_LFP_PHASE_AMPLITUDE_MASK_OPTIONS,
    )
    absolute_amplitude_threshold = 0.0
    if amplitude_mask_mode == "Absolute magnitude":
        absolute_amplitude_threshold = float(
            st.sidebar.number_input(
                "Wavelet magnitude threshold",
                min_value=0.0,
                value=0.0,
                format="%.6g",
            )
        )

    probe_sources = {
        LFP_DROPDOWN_LABEL_HPC_V1: (hpc_v1_lfp_path, hpc_v1_aligned_spike_path),
        LFP_DROPDOWN_LABEL_PFC: (pfc_lfp_path, pfc_aligned_spike_path),
    }
    raw_lfp_path, raw_sync_path = probe_sources[lfp_probe_label]
    lfp_source = Path(str(raw_lfp_path)).expanduser()
    if not str(raw_lfp_path).strip() or not lfp_source.is_file():
        st.error(f"LFP path for {lfp_probe_label} does not exist: {lfp_source}")
        st.stop()
    aligned_sync_path = None
    aligned_sync_mtime_ns = -1
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        sync_source = Path(str(raw_sync_path)).expanduser()
        if not str(raw_sync_path).strip() or not sync_source.is_file():
            st.error(f"Aligned sync path for {lfp_probe_label} does not exist: {sync_source}")
            st.stop()
        aligned_sync_path = str(sync_source)
        aligned_sync_mtime_ns = sync_source.stat().st_mtime_ns

    unit_spike_times_s = unit_spike_loading.get_unit_spike_times(spike_group, selected_unit_id)
    lfp_site_label = f"{lfp_probe_label}, channel {lfp_saved_channel_index}"
    open_ephys_tokens: tuple[OpenEphysCacheToken, ...] = ()
    try:
        if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
            if aligned_sync_path is None:
                raise ValueError("Open Ephys phase locking requires an aligned sync path")
            open_ephys_tokens = (build_open_ephys_cache_token(lfp_source, aligned_sync_path),)
        result = compute_spike_lfp_phase_locking_cached(
            lfp_format=lfp_format,
            lfp_path=str(lfp_source),
            lfp_mtime_ns=lfp_source.stat().st_mtime_ns,
            saved_channel_index=lfp_saved_channel_index,
            lfp_site_label=lfp_site_label,
            aligned_sync_path=aligned_sync_path,
            aligned_sync_mtime_ns=aligned_sync_mtime_ns,
            unit_id=selected_unit_id,
            unit_spike_times_s=tuple(unit_spike_times_s.tolist()),
            trial_indices=tuple(selected_trial_indices.tolist()),
            event_times_s=tuple(alignment_times[selected_trial_indices].tolist()),
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
            target_sample_rate_hz=LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ,
            notch_60_hz=notch_60_hz,
            notch_quality_factor=LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
            minimum_relative_magnitude=LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE,
            absolute_amplitude_threshold=absolute_amplitude_threshold,
            maximum_core_duration_s=SPIKE_LFP_PHASE_MAXIMUM_CORE_DURATION_S,
            phase_bin_count=SPIKE_LFP_PHASE_BIN_COUNT,
            open_ephys_cache_token=open_ephys_tokens[0] if open_ephys_tokens else None,
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should report unit/site-specific failures.
        st.error(f"Could not compute spike-LFP phase locking: {error}")
        st.stop()

    figure, _axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=result,
        polar_frequency_hz=polar_frequency_hz,
        polar_bin_count=SPIKE_LFP_PHASE_BIN_COUNT,
    )
    polar_frequency_index = int(np.argmin(np.abs(frequencies_hz - polar_frequency_hz)))
    actual_polar_frequency_hz = float(frequencies_hz[polar_frequency_index])
    phase_rate_sparsity = spike_lfp_phase_locking.assess_phase_rate_sparsity(
        phase_spike_counts=result.phase_spike_counts[polar_frequency_index],
        phase_occupancy_s=result.phase_occupancy_s[polar_frequency_index],
    )
    metadata_column, figure_column = st.columns([1, 3])
    with metadata_column:
        st.subheader("Spike-LFP Phase Locking")
        st.write(f"Unit: {selected_unit_id} ({active_probe_label})")
        st.write(f"LFP site: {lfp_site_label}")
        st.write(f"Condition: {condition}")
        st.write(f"Action: {action_label}")
        st.write(f"Trials: {selected_trial_indices.size}")
        st.write(f"Selected spikes: {result.spike_times_s.size}")
        retained_bytes = (
            result.spike_phase_vectors.nbytes
            + result.spike_phase_valid.nbytes
            + result.amplitude_at_spikes.nbytes
            + result.phase_bin_edges_rad.nbytes
            + result.phase_spike_counts.nbytes
            + result.phase_occupancy_s.nbytes
            + result.phase_firing_rate_hz.nbytes
        )
        st.write(f"Retained phase data: {retained_bytes / (1024.0**2):.4f} MB")
        if phase_rate_sparsity.is_sparse:
            warning_reasons = []
            if phase_rate_sparsity.low_spike_count:
                warning_reasons.append(
                    f"{phase_rate_sparsity.valid_spike_count} valid spikes "
                    f"({phase_rate_sparsity.mean_spikes_per_bin:.1f} per bin; "
                    f"heuristic target: {phase_rate_sparsity.minimum_recommended_spikes} total)"
                )
            if phase_rate_sparsity.has_zero_occupancy:
                warning_reasons.append(
                    f"{phase_rate_sparsity.zero_occupancy_bin_count} phase bins have no valid LFP occupancy"
                )
            st.warning(
                "Sparse phase-rate estimate at "
                f"{actual_polar_frequency_hz:.3g} Hz: {'; '.join(warning_reasons)}. "
                "This sampling heuristic is not a significance test."
            )
    with figure_column:
        st.pyplot(figure, width="stretch")
        st.caption(
            "The polar panel shows occupancy-normalized phase firing rate in Hz. "
            "PPC is not clipped and may be negative. It is descriptive here: spikes can be temporally "
            "correlated, and high-frequency phase from a nearby LFP site can contain spike contamination."
        )

    if st.button("Save spike-LFP phase-locking result"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        numeric_output_dir = (
            Path(session.session_data_home) / "ephys" / "derived" / "spike_lfp_phase_locking" / timestamp
        )
        run_output_dir = (
            Path(session.session_data_home) / "analysis_runs" / f"spike_lfp_phase_locking_{timestamp}"
        )
        metadata = {
            "generator": "src.neural_analysis.spike_lfp_phase_locking",
            "analysis_version": spike_lfp_phase_locking.ANALYSIS_VERSION,
            "session_id": session.sess_id_full,
            "unit_id": selected_unit_id,
            "unit_probe": active_probe_label,
            "lfp_site": lfp_site_label,
            "lfp_path": str(lfp_source),
            "lfp_format": lfp_format,
            "condition": condition,
            "action": action_label,
            "alignment_event": alignment_event,
            "window_s": [window_start_s, window_end_s],
            "frequencies_hz": frequencies_hz.tolist(),
            "gaussian_width": gaussian_width,
            "window_length": wavelet_window_length,
            "normalization": wavelet_norm,
            "notch_60_hz": notch_60_hz,
            "target_sample_rate_hz": LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ,
            "amplitude_mask_mode": amplitude_mask_mode,
            "absolute_amplitude_threshold": absolute_amplitude_threshold,
            "amplitude_units": (
                "uV"
                if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED
                else "source-dependent wavelet magnitude"
            ),
            "phase_units": "radians",
            "phase_bin_count": int(SPIKE_LFP_PHASE_BIN_COUNT),
            "phase_bin_range_rad": [-float(np.pi), float(np.pi)],
            "phase_occupancy_units": "seconds",
            "phase_firing_rate_units": "Hz",
            "phase_tuning_axis_order": ["frequency", "phase_bin"],
            "phase_mask_policy": {
                "minimum_relative_magnitude": float(LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE),
                "absolute_amplitude_threshold": float(absolute_amplitude_threshold),
                "amplitude_mask_mode": amplitude_mask_mode,
                "scope": "continuous occupancy and spike phase samples",
            },
            "time_units": "absolute synchronized seconds",
            "axis_order": ["frequency", "spike"],
            "polar_frequency_requested_hz": float(polar_frequency_hz),
            "polar_frequency_actual_hz": actual_polar_frequency_hz,
            "phase_rate_sparsity": {
                "is_sparse": bool(phase_rate_sparsity.is_sparse),
                "valid_spike_count": int(phase_rate_sparsity.valid_spike_count),
                "phase_bin_count": int(phase_rate_sparsity.phase_bin_count),
                "mean_spikes_per_bin": float(phase_rate_sparsity.mean_spikes_per_bin),
                "minimum_mean_spikes_per_bin": float(
                    phase_rate_sparsity.minimum_mean_spikes_per_bin
                ),
                "minimum_recommended_spikes": int(
                    phase_rate_sparsity.minimum_recommended_spikes
                ),
                "zero_occupancy_bin_count": int(
                    phase_rate_sparsity.zero_occupancy_bin_count
                ),
                "criterion": "descriptive sampling heuristic, not a significance test",
            },
            "random_seed": None,
        }
        metadata.update(_open_ephys_exploratory_provenance(lfp_format, open_ephys_tokens))
        spike_lfp_phase_locking.save_spike_lfp_phase_locking_result(
            output_path=numeric_output_dir / "spike_lfp_phase_locking.npz",
            result=result,
            metadata=metadata,
        )
        run_output_dir.mkdir(parents=True, exist_ok=False)
        figure.savefig(run_output_dir / "spike_lfp_phase_locking.png", dpi=200, bbox_inches="tight")
        (run_output_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# Spike-LFP phase locking",
                    "",
                    f"Session: {session.sess_id_full}",
                    f"Unit: {selected_unit_id} ({active_probe_label})",
                    f"LFP site: {lfp_site_label}",
                    f"Selected trials: {selected_trial_indices.size}",
                    f"Selected spikes: {result.spike_times_s.size}",
                    f"Numeric result: {numeric_output_dir / 'spike_lfp_phase_locking.npz'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        st.success(f"Saved analysis to {run_output_dir}")
    plt.close(figure)

def render_single_trial_spike_lfp_hilbert_view(
    trial_df: pd.DataFrame,
    event_df: pd.DataFrame,
    session: spike_behavior_pynapple.Session,
    spike_group: object,
    unit_ids: np.ndarray,
    active_probe_label: str,
    hpc_v1_lfp_path: str,
    pfc_lfp_path: str,
    hpc_v1_aligned_spike_path: str,
    pfc_aligned_spike_path: str,
) -> None:
    """Render and optionally save one unit's trial-level Hilbert phase trace.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Alignment columns
        contain synchronized absolute timestamps in seconds.
    event_df : pd.DataFrame
        Event table used to build left/right lick timestamps in seconds.
    session : spike_behavior_pynapple.Session
        Session metadata containing output root and full session identifier.
    spike_group : object
        Pynapple ``TsGroup`` keyed by unit id with synchronized absolute spike
        timestamps in seconds.
    unit_ids : np.ndarray
        Eligible integer unit ids with shape ``(n_units,)``.
    active_probe_label : str
        Probe owning the selected unit. The LFP probe remains independent.
    hpc_v1_lfp_path, pfc_lfp_path : str
        Probe-specific continuous LFP source paths.
    hpc_v1_aligned_spike_path, pfc_aligned_spike_path : str
        Probe-specific aligned sync paths required by derived Open Ephys LFP.

    Returns
    -------
    None
        Streamlit renders raw source-rate LFP, fixed 6--10 Hz Hilbert phase,
        spike observations, behavior events, and optional timestamped outputs.
    """

    st.sidebar.header("Single-Trial Spike-LFP Phase")
    available_unit_ids = np.asarray(unit_ids, dtype=int).reshape(-1)
    selected_unit_id = int(st.sidebar.selectbox("Unit", options=available_unit_ids.tolist()))
    condition = st.sidebar.selectbox("Trial condition", options=CONDITION_OPTIONS)
    action_options = {label: value for label, value in ACTION_OPTIONS.items() if value != COMPARE_LEFT_RIGHT_ACTION}
    action_label = st.sidebar.selectbox("Action", options=list(action_options))
    alignment_event = st.sidebar.selectbox("Alignment event", options=ALIGNMENT_OPTIONS)
    window_start_s = float(
        st.sidebar.number_input(
            "Window start (s)",
            value=float(SPIKE_LFP_HILBERT_DEFAULT_WINDOW[0]),
            step=0.1,
        )
    )
    window_end_s = float(
        st.sidebar.number_input(
            "Window end (s)",
            value=float(SPIKE_LFP_HILBERT_DEFAULT_WINDOW[1]),
            step=0.1,
        )
    )
    if window_start_s >= window_end_s:
        st.error("Window start must be less than window end.")
        st.stop()

    selected_trial_indices = unit_spike_plotting.filter_trials_for_unit_plot(
        trial_df=trial_df,
        condition=condition,
        action=action_options[action_label],
    )
    alignment_times_s = pd.to_numeric(trial_df[alignment_event], errors="coerce").to_numpy(dtype=float)
    selected_trial_indices = selected_trial_indices[np.isfinite(alignment_times_s[selected_trial_indices])]
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
    event_time_s = float(alignment_times_s[trial_index])
    st.sidebar.caption(f"Trial row {trial_index}; {selected_trial_indices.size} matching trials")

    lfp_format = st.sidebar.selectbox("LFP format", options=LFP_FORMAT_OPTIONS)
    utc_offset_hours = float(
        st.sidebar.number_input("LFP UTC offset (hours)", value=0, step=1, format="%d")
    )
    probe_options = [LFP_DROPDOWN_LABEL_HPC_V1, LFP_DROPDOWN_LABEL_PFC]
    default_probe_index = 0 if active_probe_label == unit_spike_loading.PROBE_LABEL_HPC_V1 else 1
    lfp_probe_label = st.sidebar.selectbox("LFP probe", options=probe_options, index=default_probe_index)
    lfp_saved_channel_index = int(
        st.sidebar.number_input("LFP saved channel", min_value=0, value=0, step=1)
    )
    st.sidebar.caption(
        "Phase band is fixed at 6--10 Hz: Pynapple Butterworth bandpass followed by SciPy Hilbert."
    )

    probe_sources = {
        LFP_DROPDOWN_LABEL_HPC_V1: (hpc_v1_lfp_path, hpc_v1_aligned_spike_path),
        LFP_DROPDOWN_LABEL_PFC: (pfc_lfp_path, pfc_aligned_spike_path),
    }
    raw_lfp_path, raw_sync_path = probe_sources[lfp_probe_label]
    lfp_source = Path(str(raw_lfp_path)).expanduser()
    if not str(raw_lfp_path).strip() or not lfp_source.is_file():
        st.error(f"LFP path for {lfp_probe_label} does not exist: {lfp_source}")
        st.stop()
    aligned_sync_path = None
    aligned_sync_mtime_ns = -1
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        sync_source = Path(str(raw_sync_path)).expanduser()
        if not str(raw_sync_path).strip() or not sync_source.is_file():
            st.error(f"Aligned sync path for {lfp_probe_label} does not exist: {sync_source}")
            st.stop()
        aligned_sync_path = str(sync_source)
        aligned_sync_mtime_ns = sync_source.stat().st_mtime_ns

    unit_spike_times_s = unit_spike_loading.get_unit_spike_times(spike_group, selected_unit_id)
    lfp_site_label = f"{lfp_probe_label}, channel {lfp_saved_channel_index}"
    open_ephys_tokens: tuple[OpenEphysCacheToken, ...] = ()
    try:
        if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
            if aligned_sync_path is None:
                raise ValueError("Open Ephys Hilbert phase requires an aligned sync path")
            open_ephys_tokens = (build_open_ephys_cache_token(lfp_source, aligned_sync_path),)
        result = compute_single_trial_spike_lfp_hilbert_cached(
            lfp_format=lfp_format,
            lfp_path=str(lfp_source),
            lfp_mtime_ns=lfp_source.stat().st_mtime_ns,
            saved_channel_index=lfp_saved_channel_index,
            lfp_site_label=lfp_site_label,
            aligned_sync_path=aligned_sync_path,
            aligned_sync_mtime_ns=aligned_sync_mtime_ns,
            unit_id=selected_unit_id,
            unit_spike_times_s=tuple(unit_spike_times_s.tolist()),
            trial_index=trial_index,
            event_time_s=event_time_s,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            band_low_hz=float(SPIKE_LFP_HILBERT_PHASE_BAND_HZ[0]),
            band_high_hz=float(SPIKE_LFP_HILBERT_PHASE_BAND_HZ[1]),
            filter_padding_s=float(SPIKE_LFP_HILBERT_FILTER_PADDING_S),
            digital_word=0,
            irig_line=6,
            bit_period_s=1.0,
            utc_offset_hours=utc_offset_hours,
            minimum_envelope=float(SPIKE_LFP_HILBERT_MINIMUM_ENVELOPE),
            open_ephys_cache_token=open_ephys_tokens[0] if open_ephys_tokens else None,
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should display unit/site loading failures cleanly.
        st.error(f"Could not compute single-trial spike-LFP phase: {error}")
        st.stop()

    lick_times = spike_behavior_pynapple.build_lick_time_dict(event_df)
    lfp_y_label = "LFP (uV)"
    figure, _axes = unit_spike_plotting.plot_trial_spike_lfp_hilbert_phase_and_behavior(
        trial_df=trial_df,
        trial_index=trial_index,
        lick_times=lick_times,
        result=result,
        alignment_event=alignment_event,
        window=(window_start_s, window_end_s),
        lfp_y_label=lfp_y_label,
    )

    metadata_column, figure_column = st.columns([1, 3])
    with metadata_column:
        st.subheader("Single-Trial Spike-LFP Phase")
        st.write(f"Trial row: {trial_index}")
        st.write(f"Matching trials: {selected_trial_indices.size}")
        st.write(f"Unit: {selected_unit_id} ({active_probe_label})")
        st.write(f"LFP site: {lfp_site_label}")
        st.write(f"Condition: {condition}")
        st.write(f"Action: {action_label}")
        st.write(f"Alignment: {alignment_event}")
        st.write(f"Visible spikes: {result.spike_times_relative_s.size}")
        st.write(f"Phase-valid spikes: {int(np.count_nonzero(result.spike_phase_valid))}")
        st.write(f"Source sample rate: {result.source_sample_rate_hz:g} Hz")
        st.caption(
            "This 6--10 Hz Hilbert phase is a trial-inspection signal and is not identical to "
            "the frequency-specific Morlet phase used by the pooled PPC view."
        )
    with figure_column:
        st.pyplot(figure, width="stretch")
        st.caption(
            "Raw LFP remains at source rate. The filtered and phase traces use padded continuous LFP before "
            "cropping; phase points are sampled at the unit's exact spike timestamps."
        )

    if st.button("Save single-trial spike-LFP phase result"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        numeric_output_dir = (
            Path(session.session_data_home) / "ephys" / "derived" / "single_trial_spike_lfp_hilbert" / timestamp
        )
        run_output_dir = (
            Path(session.session_data_home) / "analysis_runs" / f"single_trial_spike_lfp_hilbert_{timestamp}"
        )
        metadata = {
            "generator": "src.neural_analysis.spike_lfp_hilbert_phase",
            "analysis_version": spike_lfp_hilbert_phase.ANALYSIS_VERSION,
            "session_id": session.sess_id_full,
            "trial_index": trial_index,
            "trial_position": trial_position,
            "matching_trial_count": int(selected_trial_indices.size),
            "unit_id": selected_unit_id,
            "unit_probe": active_probe_label,
            "lfp_site": lfp_site_label,
            "lfp_path": str(lfp_source),
            "lfp_format": lfp_format,
            "condition": condition,
            "action": action_label,
            "alignment_event": alignment_event,
            "window_s": [window_start_s, window_end_s],
            "phase_band_hz": list(SPIKE_LFP_HILBERT_PHASE_BAND_HZ),
            "filter_method": "Pynapple Butterworth bandpass",
            "phase_method": "scipy.signal.hilbert",
            "phase_convention": "angle(hilbert(Pynapple Butterworth bandpass))",
            "phase_units": "radians",
            "phase_range_rad": [-float(np.pi), float(np.pi)],
            "raw_lfp_units": "uV",
            "filtered_lfp_units": "uV",
            "time_units": "seconds relative to alignment event",
            "spike_time_units": "seconds relative to alignment event",
            "spike_phase_units": "radians",
            "axis_order": ["time"],
            "filter_padding_s": float(SPIKE_LFP_HILBERT_FILTER_PADDING_S),
            "decimation": "none",
            "source_sample_rate_hz": float(result.source_sample_rate_hz),
            "minimum_envelope": float(SPIKE_LFP_HILBERT_MINIMUM_ENVELOPE),
            "visible_spike_count": int(result.spike_times_relative_s.size),
            "phase_valid_spike_count": int(np.count_nonzero(result.spike_phase_valid)),
            "lfp_mtime_ns": int(lfp_source.stat().st_mtime_ns),
            "aligned_sync_path": aligned_sync_path,
            "aligned_sync_mtime_ns": int(aligned_sync_mtime_ns),
            "hilbert_vs_pooled_morlet_caveat": (
                "The 6-10 Hz Hilbert phase shown here is not identical to the pooled "
                "frequency-specific Morlet phase used for PPC."
            ),
            "random_seed": None,
        }
        metadata.update(_open_ephys_exploratory_provenance(lfp_format, open_ephys_tokens))
        spike_lfp_hilbert_phase.save_single_trial_spike_lfp_hilbert_result(
            output_path=numeric_output_dir / "single_trial_spike_lfp_hilbert.npz",
            result=result,
            metadata=metadata,
        )
        run_output_dir.mkdir(parents=True, exist_ok=False)
        figure.savefig(run_output_dir / "single_trial_spike_lfp_hilbert.png", dpi=200, bbox_inches="tight")
        (run_output_dir / "summary.md").write_text(
            "\n".join(
                [
                    "# Single-trial spike-LFP Hilbert phase",
                    "",
                    f"Session: {session.sess_id_full}",
                    f"Trial row: {trial_index} ({trial_position} of {selected_trial_indices.size} matching trials)",
                    f"Unit: {selected_unit_id} ({active_probe_label})",
                    f"LFP site: {lfp_site_label}",
                    f"Visible spikes: {result.spike_times_relative_s.size}; phase-valid: "
                    f"{int(np.count_nonzero(result.spike_phase_valid))}",
                    f"Numeric result: {numeric_output_dir / 'single_trial_spike_lfp_hilbert.npz'}",
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
