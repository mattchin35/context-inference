from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import (
    lfp_loading,
    lfp_phase_clustering,
    lfp_spectrogram,
    population_pca,
    population_pca_decoding,
    population_pca_switch_trajectories,
    spike_behavior_pynapple,
    spike_lfp_phase_locking,
    unit_spike_loading,
    unit_spike_plotting,
)


CONDITION_OPTIONS = [
    "all",
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
]
ACTION_OPTIONS = {
    "all": "all",
    "right (0)": 0,
    "left (1)": 1,
    "compare left vs right": "compare_lr",
}
ALIGNMENT_OPTIONS = ["choice_time", "start_time"]
PAGE_SIZE_OPTIONS = [25, 50, 100]
PSTH_BIN_OPTIONS = [0.05, 0.1]
PLOT_VIEW_UNIT_RASTER = "Unit raster/PSTH"
PLOT_VIEW_TRIAL_SPIKES = "Trial spikes/licks/choices"
PLOT_VIEW_PCA_DECODING = "Population PCA decoding"
PLOT_VIEW_PCA_SWITCH_TRAJECTORIES = "Population PCA switch trajectories"
PLOT_VIEW_LFP_PHASE_CLUSTERING = "LFP phase clustering"
PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE = "Single-trial relative phase"
PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING = "Spike-LFP phase locking"
PLOT_VIEW_OPTIONS = [
    PLOT_VIEW_UNIT_RASTER,
    PLOT_VIEW_TRIAL_SPIKES,
    PLOT_VIEW_PCA_DECODING,
    PLOT_VIEW_PCA_SWITCH_TRAJECTORIES,
    PLOT_VIEW_LFP_PHASE_CLUSTERING,
    PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE,
    PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING,
]
UNIT_PLOT_TYPE_OPTIONS = [
    "PSTH",
    "Binned rate: trials + mean",
    "Binned rate: mean +/- SD",
]
NEURAL_DISPLAY_SPIKE_RASTER = "Spike raster"
NEURAL_DISPLAY_LFP_SPECTROGRAM = "LFP spectrogram + trace"
NEURAL_DISPLAY_POPULATION_PCA = "Population PCA"
NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED = "Population PCA: concatenated trials"
NEURAL_DISPLAY_OPTIONS = [
    NEURAL_DISPLAY_SPIKE_RASTER,
    NEURAL_DISPLAY_LFP_SPECTROGRAM,
    NEURAL_DISPLAY_POPULATION_PCA,
    NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED,
]
PCA_BIN_SIZE_OPTIONS = [0.1, 0.05, 0.02]
DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT = 5
DEFAULT_PCA_VARIANCE_COMPONENT_COUNT = 50
DEFAULT_PCA_COMPONENT_COUNT = DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT
DEFAULT_CONCATENATED_PCA_VISIBLE_TRIAL_COUNT = 20
CONCATENATED_PCA_VISIBLE_TRIAL_OPTIONS = [10, 20, 40]
DEFAULT_CONCATENATED_PCA_COMPONENT_COUNT = 3
CONCATENATED_PCA_FIGURE_SIZE = (11.0, 5.0)
PCA_DECODING_DEFAULT_COMPONENT_COUNT = 5
PCA_DECODING_DEFAULT_CV_FOLDS = 5
PCA_DECODING_DEFAULT_PERMUTATIONS = 100
PCA_DECODING_DISPLAY_PERFORMANCE = "Decoding performance"
PCA_DECODING_DISPLAY_AVERAGE_PC = "Average PC by condition/target"
PCA_DECODING_DISPLAY_OPTIONS = [
    PCA_DECODING_DISPLAY_PERFORMANCE,
    PCA_DECODING_DISPLAY_AVERAGE_PC,
]
PCA_DECODING_SHOW_RAW_PC_SCORES_DEFAULT = False
CHANNEL_SOURCE_MANUAL = "Manual / preset"
CHANNEL_SOURCE_CHANNEL_QUALITY = "channel_quality"
CHANNEL_SOURCE_OPTIONS = [CHANNEL_SOURCE_MANUAL, CHANNEL_SOURCE_CHANNEL_QUALITY]
COMPARE_LEFT_RIGHT_ACTION = "compare_lr"
POPULATION_PSTH_UNIT_SCOPE_OPTIONS = ["Visible page units", "All selected units"]
LFP_DROPDOWN_LABEL_HPC_V1 = "HPC/V1 LFP"
LFP_DROPDOWN_LABEL_PFC = "PFC LFP"
LFP_FORMAT_SPIKEGLX = "SpikeGLX"
LFP_FORMAT_OPEN_EPHYS_DERIVED = "Open Ephys derived"
LFP_FORMAT_OPTIONS = [LFP_FORMAT_SPIKEGLX, LFP_FORMAT_OPEN_EPHYS_DERIVED]
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
SPIKE_LFP_PHASE_DEFAULT_WINDOW = (-0.5, 0.5)
SPIKE_LFP_PHASE_MIN_FREQUENCY_HZ = 2.0
SPIKE_LFP_PHASE_MAX_FREQUENCY_HZ = 100.0
SPIKE_LFP_PHASE_FREQUENCY_COUNT = 50
SPIKE_LFP_PHASE_DEFAULT_POLAR_FREQUENCY_HZ = 8.0
SPIKE_LFP_PHASE_AMPLITUDE_MASK_OPTIONS = ("Off", "Absolute magnitude")
SPIKE_LFP_PHASE_MAXIMUM_CORE_DURATION_S = 120.0
SPIKE_LFP_PHASE_CACHE_MAX_ENTRIES = 6
DEFAULT_BROWSER_ROOT = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026")
RASTER_LAYOUT_OPTIONS = {
    "Compact": {"row_spacing": 1.0, "figure_size": (12.0, 7.0)},
    "Separated": {"row_spacing": 1.5, "figure_size": (12.0, 10.0)},
    "Wide": {"row_spacing": 2.0, "figure_size": (12.0, 13.0)},
}
EVENT_MARKER_STYLES = {
    "start_time": {"label": "trial start", "color": "black"},
    "choice_time": {"label": "choice", "color": "tab:purple"},
    "led_on_time": {"label": "LED", "color": "tab:green"},
}


def is_population_pca_display(neural_display: str) -> bool:
    """
    Return whether a trial-view neural display uses population PCA.

    Parameters
    ----------
    neural_display : str
        Display label selected from ``NEURAL_DISPLAY_OPTIONS``.

    Returns
    -------
    bool
        ``True`` for single-trial and concatenated population PCA displays;
        ``False`` for spike-raster displays.
    """

    return neural_display in {
        NEURAL_DISPLAY_POPULATION_PCA,
        NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED,
    }


def resolve_pca_decoding_component_minimum(display_option: str) -> int:
    """
    Resolve the minimum PCA component count for a PCA decoding display.

    Parameters
    ----------
    display_option : str
        Selected PCA decoding display mode. Expected values are entries in
        ``PCA_DECODING_DISPLAY_OPTIONS``.

    Returns
    -------
    int
        Minimum number of PCA components. The average PC score display requires
        two components because it plots mean PC1 against mean PC2; decoding
        performance can use one component.
    """

    if display_option == PCA_DECODING_DISPLAY_AVERAGE_PC:
        return 2
    return 1


def select_visible_concatenated_trial_indices(
    trial_indices: np.ndarray,
    page_index: int,
    visible_trial_count: int,
) -> tuple[np.ndarray, int]:
    """
    Select one bounded page of trial indices for concatenated PCA plotting.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Values are
        row indices into the trial table.
    page_index : int
        Zero-based requested page index. Values beyond the last page are
        clamped to the last available page.
    visible_trial_count : int
        Maximum number of trials shown in one concatenated figure.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(visible_trial_indices, page_count)``. ``visible_trial_indices`` has
        shape ``(n_visible_trials,)`` and preserves the input order.
        ``page_count`` is at least one for nonempty inputs and zero for empty
        inputs.
    """

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if int(visible_trial_count) <= 0:
        raise ValueError("visible_trial_count must be positive.")
    if normalized_trial_indices.size == 0:
        return np.array([], dtype=int), 0

    page_count = int(math.ceil(normalized_trial_indices.size / int(visible_trial_count)))
    clamped_page_index = min(max(int(page_index), 0), page_count - 1)
    start_index = clamped_page_index * int(visible_trial_count)
    end_index = start_index + int(visible_trial_count)
    return normalized_trial_indices[start_index:end_index], page_count


def select_concatenated_trial_viewport(
    trial_indices: np.ndarray,
    start_position: int,
    visible_trial_count: int,
) -> tuple[np.ndarray, int]:
    """
    Select one fixed-size chronological viewport of concatenated PCA trials.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Values are
        row indices into the trial table.
    start_position : int
        Zero-based requested start position within ``trial_indices``.
    visible_trial_count : int
        Number of trials shown in the fixed viewport. If enough trials are
        available, oversized start positions are clamped to keep the viewport
        full.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(visible_trial_indices, clamped_start_position)``.
        ``visible_trial_indices`` has shape ``(n_visible_trials,)`` and
        preserves the input order. ``clamped_start_position`` is the zero-based
        position actually used after bounds checking.
    """

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if int(visible_trial_count) <= 0:
        raise ValueError("visible_trial_count must be positive.")
    if normalized_trial_indices.size == 0:
        return np.array([], dtype=int), 0

    visible_count = int(visible_trial_count)
    max_start_position = max(0, normalized_trial_indices.size - visible_count)
    clamped_start_position = min(max(int(start_position), 0), max_start_position)
    end_position = clamped_start_position + visible_count
    return normalized_trial_indices[clamped_start_position:end_position], clamped_start_position


@st.cache_resource(show_spinner="Loading session data...")
def load_viewer_data_cached(
    session_data_home: str,
    sess_id_full: str,
    active_probe_label: str,
    sorter_output_path: str | None,
    aligned_spike_path: str | None,
    lfp_path: str | None,
):
    """
    Load one session for Streamlit with cache persistence until app restart.

    Parameters
    ----------
    session_data_home : str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    active_probe_label : str
        Active probe label, such as ``"HPC/V1"`` or ``"PFC"``.
    sorter_output_path : str | None
        Active probe sorter output directory path.
    aligned_spike_path : str | None
        Active probe aligned spike ``.npz`` path.
    lfp_path : str | None
        Active probe LFP ``.lf.bin`` path. Not required for spike-only plots.

    Returns
    -------
    dict
        Loaded viewer data from ``unit_spike_loading.load_viewer_data_for_probe``.
    """

    probe_paths = unit_spike_loading.ProbeDataPaths(
        label=active_probe_label,
        sorter_output_path=sorter_output_path,
        aligned_spike_path=aligned_spike_path,
        lfp_path=lfp_path,
    )
    return unit_spike_loading.load_viewer_data_for_probe(
        session_data_home=Path(session_data_home),
        sess_id_full=sess_id_full,
        probe_paths=probe_paths,
    )


@st.cache_resource(show_spinner="Loading behavior data...")
def load_phase_clustering_session_cached(
    session_data_home: str,
    sess_id_full: str,
) -> tuple[spike_behavior_pynapple.Session, pd.DataFrame, pd.DataFrame]:
    """
    Load session metadata and trials without requiring sorted spike data.

    Parameters
    ----------
    session_data_home : str
        Root directory for one experimental session.
    sess_id_full : str
        Session identifier formatted as ``mouse_YYYY-MM-DD_hhmmss``.

    Returns
    -------
    tuple[spike_behavior_pynapple.Session, pd.DataFrame, pd.DataFrame]
        Session metadata, event table with shape ``(n_events, n_event_columns)``,
        and trial table with shape ``(n_trials, n_trial_columns)``. Event and
        trial times are in absolute seconds.
    """

    session_home = Path(session_data_home)
    mouse, date, timestamp = spike_behavior_pynapple.parse_session_id(sess_id_full)
    session = spike_behavior_pynapple.Session(
        session_data_home=session_home,
        sess_id_full=sess_id_full,
        sess_id_abbreviated=f"{mouse}_{date}",
        raw_behavior_folder=session_home / "rpi" / sess_id_full,
        processed_data_path=session_home / "processed",
        figure_path=session_home / "figures",
        mouse=mouse,
        date=date,
        timestamp=timestamp,
    )
    event_df, trial_df = spike_behavior_pynapple.load_session_tables(session)
    return session, event_df, trial_df


@st.cache_data(show_spinner="Loading channel quality...")
def load_channel_quality_cached(channel_quality_path: str):
    """
    Load normalized channel-quality metadata with Streamlit caching.

    Parameters
    ----------
    channel_quality_path : str
        Path to ``channel_quality.csv``, ``channel_quality.json``, or a probe
        directory containing one of those files. Units: filesystem path.

    Returns
    -------
    pd.DataFrame
        Normalized channel-quality dataframe from
        ``unit_spike_loading.load_channel_quality``. Rows are channels; ``ch``
        is a zero-based channel index.
    """

    return unit_spike_loading.load_channel_quality(Path(channel_quality_path))


@st.cache_resource(show_spinner="Decoding LFP sync...")
def decode_lfp_sync_cached(
    lfp_path: str,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
):
    """
    Decode and cache LFP IRIG sync for one file/settings combination.

    Parameters
    ----------
    lfp_path : str
        SpikeGLX ``.lf.bin`` file path.
    digital_word : int
        Digital word index.
    irig_line : int
        IRIG-H digital line. Current IMEC default is line 6.
    bit_period_s : float
        IRIG-H bit period in seconds.
    utc_offset_hours : float
        Constant UTC offset in hours.

    Returns
    -------
    tuple
        ``(lfp_irig_df, sample_rate_hz)`` from ``lfp_loading.decode_lfp_sync``.
    """

    return lfp_loading.decode_lfp_sync(
        lfp_path=Path(lfp_path),
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
    )


@st.cache_data(show_spinner="Loading LFP trace...")
def load_trial_lfp_trace_cached(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    filter_low_hz: float | None,
    filter_high_hz: float | None,
    filter_padding_s: float,
    aligned_sync_npz_path: str | None = None,
):
    """
    Load and cache one trial-aligned LFP channel window.

    Parameters
    ----------
    lfp_format : str
        LFP file format. Supported values are ``LFP_FORMAT_SPIKEGLX`` and
        ``LFP_FORMAT_OPEN_EPHYS_DERIVED``.
    lfp_path : str
        LFP binary file path. Units: filesystem path.
    saved_channel_index : int
        Zero-based saved channel index into the binary rows.
    alignment_time_s : float
        Absolute trial alignment time in seconds.
    window_start_s, window_end_s : float
        Relative window bounds in seconds.
    digital_word : int
        Digital word index for sync decoding.
    irig_line : int
        IRIG-H digital line.
    bit_period_s : float
        IRIG-H bit period in seconds.
    utc_offset_hours : float
        Constant UTC offset in hours.
    filter_low_hz, filter_high_hz : float | None
        Optional bandpass cutoff frequencies in Hz. Both values must be
        provided to filter; ``None`` for either value keeps the trace
        unfiltered.
    filter_padding_s : float
        Seconds added to both sides of the requested window before filtering.
    aligned_sync_npz_path : str | None, optional
        Open Ephys probe sync ``.npz`` path. Required for
        ``LFP_FORMAT_OPEN_EPHYS_DERIVED`` and ignored for SpikeGLX.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_values)`` for plotting. Time is in seconds
        relative to alignment.
    """

    return load_trial_lfp_trace_for_format(
        lfp_format=lfp_format,
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        alignment_time_s=float(alignment_time_s),
        window_start_s=float(window_start_s),
        window_end_s=float(window_end_s),
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
        filter_padding_s=float(filter_padding_s),
        aligned_sync_npz_path=aligned_sync_npz_path,
    )


def load_trial_lfp_trace_for_format(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    filter_low_hz: float | None,
    filter_high_hz: float | None,
    filter_padding_s: float,
    aligned_sync_npz_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one LFP trace using the selected acquisition format.

    Parameters
    ----------
    lfp_format : str
        LFP file format. Supported values are ``LFP_FORMAT_SPIKEGLX`` and
        ``LFP_FORMAT_OPEN_EPHYS_DERIVED``.
    lfp_path : str
        LFP binary file path. SpikeGLX expects ``.lf.bin`` with a sibling
        ``.meta`` file; Open Ephys expects derived ``lfp.dat`` with a sibling
        ``lfp_preprocessing.json``. Units: filesystem path.
    saved_channel_index : int
        Zero-based saved channel index. Units: channel index.
    alignment_time_s : float
        Absolute trial alignment timestamp in UTC Unix seconds.
    window_start_s, window_end_s : float
        Relative window bounds in seconds around ``alignment_time_s``.
    digital_word : int
        SpikeGLX digital word index. Ignored for Open Ephys derived LFP.
    irig_line : int
        SpikeGLX IRIG-H digital line. Ignored for Open Ephys derived LFP.
    bit_period_s : float
        SpikeGLX IRIG-H bit period in seconds. Ignored for Open Ephys derived
        LFP.
    utc_offset_hours : float
        Constant UTC offset in hours applied to decoded sync anchors.
    filter_low_hz, filter_high_hz : float | None
        Optional bandpass cutoff frequencies in Hz. Both values must be
        provided to filter; ``None`` for either value keeps the trace
        unfiltered.
    filter_padding_s : float
        Seconds added to both sides of the requested window before filtering.
    aligned_sync_npz_path : str | None, optional
        Open Ephys probe sync ``.npz`` path containing IRIG anchors. Required
        for Open Ephys derived LFP and ignored for SpikeGLX.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_values)``. Both arrays have shape
        ``(n_samples,)``. Time is in seconds relative to alignment; LFP units
        depend on the selected file format.
    """

    frequency_band_hz = (
        (float(filter_low_hz), float(filter_high_hz))
        if filter_low_hz is not None and filter_high_hz is not None
        else None
    )
    if lfp_format == LFP_FORMAT_SPIKEGLX:
        lfp_irig_df, sample_rate_hz = lfp_loading.decode_lfp_sync(
            lfp_path=Path(lfp_path),
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
        )
        return lfp_loading.load_trial_lfp_trace(
            lfp_path=Path(lfp_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=float(sample_rate_hz),
            frequency_band_hz=frequency_band_hz,
            filter_padding_s=float(filter_padding_s),
        )
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        if aligned_sync_npz_path is None or str(aligned_sync_npz_path).strip() == "":
            raise ValueError("Open Ephys derived LFP requires an aligned sync .npz path.")
        return lfp_loading.load_open_ephys_trial_lfp_trace(
            lfp_path=Path(lfp_path),
            aligned_sync_npz_path=Path(aligned_sync_npz_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            utc_offset_hours=float(utc_offset_hours),
            frequency_band_hz=frequency_band_hz,
            filter_padding_s=float(filter_padding_s),
        )
    raise ValueError(f"Unsupported LFP format: {lfp_format!r}")


def load_trial_lfp_trace_for_format_with_sample_rate(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    aligned_sync_npz_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load an unfiltered LFP window and sample rate for spectrogram analysis.

    Parameters
    ----------
    lfp_format : str
        LFP format from ``LFP_FORMAT_OPTIONS``.
    lfp_path : str
        SpikeGLX ``.lf.bin`` or derived Open Ephys ``lfp.dat`` path.
    saved_channel_index : int
        Zero-based saved-channel index.
    alignment_time_s : float
        Absolute trial alignment timestamp in UTC Unix seconds.
    window_start_s, window_end_s : float
        Relative LFP bounds in seconds around ``alignment_time_s``.
    digital_word : int
        SpikeGLX sync digital word. Ignored for Open Ephys.
    irig_line : int
        SpikeGLX IRIG line. Ignored for Open Ephys.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds. Ignored for Open Ephys.
    utc_offset_hours : float
        Constant offset applied to sync timestamps in hours.
    aligned_sync_npz_path : str | None, optional
        Open Ephys aligned sync ``.npz`` path; required for that format.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Relative times and LFP values with shape ``(n_samples,)``, followed by
        sample rate in Hz. SpikeGLX values are microvolts; Open Ephys values
        retain their derived-file units.
    """

    if lfp_format == LFP_FORMAT_SPIKEGLX:
        lfp_irig_df, sample_rate_hz = decode_lfp_sync_cached(
            lfp_path=str(lfp_path),
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
        )
        return lfp_loading.load_trial_lfp_trace_with_sample_rate(
            lfp_path=Path(lfp_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=float(sample_rate_hz),
        )
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        if aligned_sync_npz_path is None or str(aligned_sync_npz_path).strip() == "":
            raise ValueError("Open Ephys derived LFP requires an aligned sync .npz path.")
        metadata = lfp_loading.load_open_ephys_lfp_metadata(Path(lfp_path))
        sample_rate_hz = float(metadata["sampling_frequency_hz"])
        relative_time_s, lfp_values = lfp_loading.load_open_ephys_trial_lfp_trace(
            lfp_path=Path(lfp_path),
            aligned_sync_npz_path=Path(aligned_sync_npz_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            utc_offset_hours=float(utc_offset_hours),
        )
        return relative_time_s, lfp_values, sample_rate_hz
    raise ValueError(f"Unsupported LFP format: {lfp_format!r}")


@st.cache_data(show_spinner="Computing trial LFP spectrogram...")
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

    Returns
    -------
    lfp_spectrogram.LFPSpectrogramResult
        Visible trace and absolute power. Power shape is
        ``(n_visible_samples, n_frequencies)`` in dB relative to one squared
        input unit; time is in relative seconds and frequencies are in Hz.
    """

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


@st.cache_data(show_spinner="Estimating shared LFP power scale...")
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


@st.cache_data(show_spinner="Computing continuous LFP phase tensors...")
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

    Returns
    -------
    lfp_phase_clustering.PhaseTrialTensor
        One-site phase tensor with shape ``(1, frequency, trial, time)``;
        phase is unitless complex phase and time is in event-relative seconds.
    """

    del lfp_file_mtime_ns

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


@st.cache_data(
    show_spinner="Computing single-trial relative phase...",
    max_entries=RELATIVE_PHASE_CACHE_MAX_ENTRIES,
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


@st.cache_data(
    show_spinner="Computing spike-LFP phase locking...",
    max_entries=SPIKE_LFP_PHASE_CACHE_MAX_ENTRIES,
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
        Source-dependent wavelet magnitude threshold at spike timestamps.
    maximum_core_duration_s : float
        Maximum unpadded transform block span in seconds.

    Returns
    -------
    spike_lfp_phase_locking.SpikePhaseLockingResult
        Frequency metrics and retained arrays with shape
        ``(frequency, selected_spike)``.
    """

    del lfp_mtime_ns, aligned_sync_mtime_ns
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
    )


def _build_channel_text(region_name: str) -> str:
    """Return editable default channel text for one preset region."""

    presets = unit_spike_loading.get_ct014_region_channel_presets()
    return unit_spike_loading.format_channel_list(presets.get(region_name, np.array([], dtype=int)))


def resolve_region_channels_for_source(
    channel_source: str,
    manual_channel_text: str,
    channel_quality: pd.DataFrame | None,
    require_inside_brain: bool,
    channel_quality_labels: tuple[str, ...] | list[str] | None,
) -> tuple[np.ndarray, str]:
    """
    Resolve active region channels from manual text or channel-quality metadata.

    Parameters
    ----------
    channel_source : str
        Channel source label. Supported values are ``CHANNEL_SOURCE_MANUAL`` and
        ``CHANNEL_SOURCE_CHANNEL_QUALITY``.
    manual_channel_text : str
        Editable channel text. Used only for ``CHANNEL_SOURCE_MANUAL``. Values
        are zero-based channel ids separated by commas or whitespace.
    channel_quality : pd.DataFrame | None
        Normalized channel-quality table with shape ``(n_channels, n_columns)``.
        Required for ``CHANNEL_SOURCE_CHANNEL_QUALITY``. ``None`` is allowed
        only for manual channels.
    require_inside_brain : bool
        If ``True``, channel-quality selection keeps only channels with
        ``inside_brain == True``. Ignored for manual channels.
    channel_quality_labels : tuple[str, ...] | list[str] | None
        Channel-quality labels to include, such as ``("good",)``. ``None``
        disables label filtering. Ignored for manual channels.

    Returns
    -------
    tuple[np.ndarray, str]
        ``(region_channels, summary)``. ``region_channels`` is a one-dimensional
        integer array of zero-based channel ids. ``summary`` is a concise
        human-readable description of the selection.
    """

    if channel_source == CHANNEL_SOURCE_MANUAL:
        region_channels = unit_spike_loading.parse_channel_list(manual_channel_text)
        return region_channels, f"Manual channel selection: {region_channels.size} channels."
    if channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY:
        if channel_quality is None:
            raise ValueError("channel_quality metadata is required for channel_quality channel source.")
        region_channels = unit_spike_loading.select_channels_from_quality(
            channel_quality=channel_quality,
            require_inside_brain=bool(require_inside_brain),
            labels=channel_quality_labels,
        )
        return region_channels, f"channel_quality selected {region_channels.size} / {channel_quality.shape[0]} channels."
    raise ValueError(f"Unsupported channel source: {channel_source!r}")


def render_single_trial_relative_phase_view(
    trial_df: pd.DataFrame,
    event_df: pd.DataFrame,
    session: spike_behavior_pynapple.Session,
    hpc_v1_lfp_path: str,
    pfc_lfp_path: str,
    hpc_v1_aligned_spike_path: str,
    pfc_aligned_spike_path: str,
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
    probe_options = [LFP_DROPDOWN_LABEL_HPC_V1, LFP_DROPDOWN_LABEL_PFC]
    probe_a = st.sidebar.selectbox("Site A probe", options=probe_options, index=0)
    channel_a = int(st.sidebar.number_input("Site A saved channel", min_value=0, value=0, step=1))
    probe_b = st.sidebar.selectbox("Site B probe", options=probe_options, index=1)
    channel_b = int(st.sidebar.number_input("Site B saved channel", min_value=0, value=0, step=1))
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

    probe_sources = {
        LFP_DROPDOWN_LABEL_HPC_V1: (hpc_v1_lfp_path, hpc_v1_aligned_spike_path),
        LFP_DROPDOWN_LABEL_PFC: (pfc_lfp_path, pfc_aligned_spike_path),
    }
    source_path_a, sync_path_a = probe_sources[probe_a]
    source_path_b, sync_path_b = probe_sources[probe_b]
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
    try:
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
    lfp_y_label = "LFP (uV)" if lfp_format == LFP_FORMAT_SPIKEGLX else "LFP"
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
    hpc_v1_lfp_path: str,
    pfc_lfp_path: str,
    hpc_v1_aligned_spike_path: str,
    pfc_aligned_spike_path: str,
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

    probe_options = [LFP_DROPDOWN_LABEL_HPC_V1, LFP_DROPDOWN_LABEL_PFC]
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
                    value=site_number,
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
                        value=pair_number,
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
    probe_sources = {
        LFP_DROPDOWN_LABEL_HPC_V1: (hpc_v1_lfp_path, hpc_v1_aligned_spike_path),
        LFP_DROPDOWN_LABEL_PFC: (pfc_lfp_path, pfc_aligned_spike_path),
    }
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
    for probe_label, saved_channel_index in unique_sites:
        source_path, aligned_sync_path = probe_sources[probe_label]
        source = Path(str(source_path)).expanduser()
        if not str(source_path).strip() or not source.exists():
            st.error(f"LFP path for {probe_label} does not exist: {source_path}")
            st.stop()
        aligned_sync_argument = str(aligned_sync_path) if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED else None
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


def resolve_population_pca_unit_ids(
    selected_unit_metadata: pd.DataFrame,
    page_unit_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    Resolve unit ids used for population PCA in the trial-view webapp.

    Parameters
    ----------
    selected_unit_metadata : pd.DataFrame
        Filtered unit metadata table with one row per selected unit and a
        ``cluster_id`` column. Rows are the current region/channel/quality
        selection.
    page_unit_ids : np.ndarray | None, optional
        Paginated visible unit ids from raster mode. This input is accepted to
        make the contract explicit but is not used for PCA.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_selected_units,)``. These
        are all selected region units, not only a displayed page.
    """

    if "cluster_id" not in selected_unit_metadata.columns:
        raise ValueError("selected_unit_metadata is missing cluster_id column.")
    return selected_unit_metadata["cluster_id"].to_numpy(dtype=int)


def filter_trial_indices_for_valid_alignment(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split selected trials by whether they have a finite alignment time.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and an ``alignment_event`` column in
        seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)``.
    alignment_event : str
        Trial time column used as time zero, such as ``"choice_time"`` or
        ``"start_time"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(valid_trial_indices, invalid_trial_indices)``. Both arrays are
        one-dimensional integer arrays of trial row indices. Valid trials have
        finite alignment times in seconds.

    Raises
    ------
    ValueError
        If ``alignment_event`` is missing, no trial indices are supplied, or no
        selected trials have valid alignment times.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if normalized_trial_indices.size == 0:
        raise ValueError("trial_indices must contain at least one trial.")
    alignment_times = pd.to_numeric(
        trial_df.loc[normalized_trial_indices, alignment_event],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid_mask = np.isfinite(alignment_times)
    valid_trial_indices = normalized_trial_indices[valid_mask]
    invalid_trial_indices = normalized_trial_indices[~valid_mask]
    if valid_trial_indices.size == 0:
        raise ValueError(f"No selected trials have valid {alignment_event} alignment times.")
    return valid_trial_indices, invalid_trial_indices


@st.cache_data(show_spinner="Computing population PCA...")
def compute_population_pca_cached(
    session_key: str,
    aligned_spike_path: str,
    unit_ids: tuple[int, ...],
    trial_indices: tuple[int, ...],
    alignment_event: str,
    window_start_s: float,
    window_end_s: float,
    bin_size_s: float,
    normalization: str,
    trajectory_component_count: int,
    variance_component_count: int,
    _spike_group,
    _trial_df: pd.DataFrame,
):
    """
    Compute and cache population PCA for one webapp trial selection.

    Parameters
    ----------
    session_key : str
        Session identifier included in the Streamlit cache key.
    aligned_spike_path : str
        Aligned spike file path included in the cache key. Units: filesystem
        path.
    unit_ids : tuple[int, ...]
        Unit ids included as PCA features, shape ``(n_units,)``.
    trial_indices : tuple[int, ...]
        Filtered trial row indices used to fit PCA, shape ``(n_trials,)``.
    alignment_event : str
        Trial time column used as time zero.
    window_start_s, window_end_s : float
        Relative window bounds in seconds.
    bin_size_s : float
        PCA bin width in seconds.
    normalization : str
        PCA normalization mode.
    trajectory_component_count : int
        Requested number of leading PCs to display in the trial trajectory plot.
    variance_component_count : int
        Requested number of leading PCs to display in the cumulative explained
        variance plot.
    _spike_group
        Pynapple spike group keyed by unit id. Leading underscore excludes this
        potentially large object from Streamlit hashing.
    _trial_df : pd.DataFrame
        Trial table. Leading underscore excludes this dataframe from Streamlit
        hashing; ``session_key`` and ``trial_indices`` carry the cache identity.

    Returns
    -------
    tuple[np.ndarray, population_pca.PopulationPCAResult]
        ``(pca_time_s, pca_result)``. ``pca_time_s`` has shape ``(n_bins,)`` in
        seconds relative to alignment. ``pca_result.scores`` has shape
        ``(n_trials, n_bins, n_fit_components)``. ``n_fit_components`` is the
        larger requested display count capped inside ``fit_population_pca``.
    """

    trajectory_component_count = int(trajectory_component_count)
    variance_component_count = int(variance_component_count)
    if trajectory_component_count < 1 or variance_component_count < 1:
        raise ValueError("PCA component counts must be positive.")
    fit_component_count = max(trajectory_component_count, variance_component_count)
    rate_tensor_hz, pca_time_s = population_pca.build_trial_unit_rate_tensor(
        spike_group=_spike_group,
        unit_ids=np.asarray(unit_ids, dtype=int),
        trial_df=_trial_df,
        trial_indices=np.asarray(trial_indices, dtype=int),
        alignment_event=alignment_event,
        window=(float(window_start_s), float(window_end_s)),
        bin_size_s=float(bin_size_s),
    )
    pca_result = population_pca.fit_population_pca(
        rate_tensor_hz=rate_tensor_hz,
        n_components=fit_component_count,
        normalization=normalization,
    )
    return pca_time_s, pca_result


@st.cache_data(show_spinner="Computing PCA decoding...")
def compute_population_pca_decoding_cached(
    session_key: str,
    aligned_spike_path: str,
    unit_ids: tuple[int, ...],
    condition_names: tuple[str, ...],
    target: str,
    mode: str,
    n_components: int,
    cv: int,
    n_permutations: int,
    normalization: str,
    include_score_summary: bool,
    include_raw_score_points: bool,
    _spike_group,
    _trial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, pd.DataFrame, pd.DataFrame]:
    """
    Compute and cache choice-aligned PCA decoding for the webapp.

    Parameters
    ----------
    session_key : str
        Session/probe identifier included in the Streamlit cache key.
    aligned_spike_path : str
        Aligned spike file path included in the cache key. Units: filesystem
        path.
    unit_ids : tuple[int, ...]
        Unit ids used as raw PCA features, shape ``(n_units,)``.
    condition_names : tuple[str, ...]
        Base decoding condition names to evaluate.
    target : str
        Decode target, ``"state_int"`` or ``"action"``.
    mode : str
        PCA fitting mode from ``population_pca_decoding.PCA_DECODING_MODE_OPTIONS``.
    n_components : int
        Requested number of leading PCs.
    cv : int
        Number of cross-validation folds.
    n_permutations : int
        Number of label permutations for null scoring.
    normalization : str
        Exploratory PCA unit-normalization mode. Rigorous mode fits a scaler in
        each CV split and ignores this value.
    include_score_summary : bool
        If ``True``, also compute average PC1/PC2 score summaries in one shared
        exploratory PCA coordinate system.
    include_raw_score_points : bool
        If ``True``, also compute raw trial-level PC1/PC2 score points in the
        same shared exploratory PCA coordinate system. Rows are condition
        memberships, so overlapping base conditions can duplicate a trial index.
    _spike_group
        Pynapple spike group keyed by unit id. Leading underscore excludes this
        potentially large object from Streamlit hashing.
    _trial_df : pd.DataFrame
        Trial table. Leading underscore excludes this dataframe from Streamlit
        hashing; ``session_key`` and decoding settings carry cache identity.

    Returns
    -------
    tuple[pd.DataFrame, int, pd.DataFrame, pd.DataFrame]
        ``(results_df, n_pca_trials, score_summary_df, raw_score_df)``.
        ``results_df`` is long-form decoding performance. ``n_pca_trials`` is
        the number of base-condition trials used to fit/extract the PCA decoding
        rates. ``score_summary_df`` and ``raw_score_df`` are empty unless their
        corresponding include flags are true.
    """

    if mode not in population_pca_decoding.PCA_DECODING_MODE_OPTIONS:
        raise ValueError(f"Unknown PCA decoding mode {mode!r}.")
    if len(condition_names) == 0:
        raise ValueError("Select at least one PCA decoding base condition.")

    trial_indices = population_pca_decoding.select_pca_decoding_trial_indices(
        trial_df=_trial_df,
        condition_names=condition_names,
    )
    if trial_indices.size == 0:
        raise ValueError("No valid trials match the selected PCA decoding base conditions.")
    rate_tensor_hz, _bin_centers_s = population_pca_decoding.build_choice_aligned_rate_tensor(
        spike_group=_spike_group,
        unit_ids=np.asarray(unit_ids, dtype=int),
        trial_df=_trial_df,
        trial_indices=trial_indices,
        bin_size_s=population_pca_decoding.PCA_DECODING_BIN_SIZE_S,
    )
    score_summary_df = pd.DataFrame(columns=population_pca_decoding.PCA_SCORE_SUMMARY_COLUMNS)
    raw_score_df = pd.DataFrame(columns=population_pca_decoding.PCA_RAW_SCORE_COLUMNS)
    if include_score_summary or include_raw_score_points:
        _trial_bins, exploratory_pca_result = population_pca_decoding.build_exploratory_pca_decoder_trial_bins(
            rate_tensor_hz=rate_tensor_hz,
            trial_df=_trial_df,
            trial_indices=trial_indices,
            n_components=int(n_components),
            normalization=normalization,
        )
        if include_score_summary:
            score_summary_df = population_pca_decoding.summarize_pca_scores_by_condition_and_target(
                pca_scores=exploratory_pca_result.scores,
                trial_df=_trial_df,
                trial_indices=trial_indices,
                condition_names=condition_names,
                target=target,
            )
        if include_raw_score_points:
            raw_score_df = population_pca_decoding.extract_pca_score_points_by_condition_and_target(
                pca_scores=exploratory_pca_result.scores,
                trial_df=_trial_df,
                trial_indices=trial_indices,
                condition_names=condition_names,
                target=target,
            )
    if mode == population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY:
        results_df = population_pca_decoding.run_exploratory_pca_choice_decoding(
            rate_tensor_hz=rate_tensor_hz,
            trial_df=_trial_df,
            trial_indices=trial_indices,
            n_components=int(n_components),
            condition_names=condition_names,
            target=target,
            cv=int(cv),
            n_permutations=int(n_permutations),
            random_state=42,
            normalization=normalization,
        )
    else:
        results_df = population_pca_decoding.run_rigorous_pca_choice_decoding(
            rate_tensor_hz=rate_tensor_hz,
            trial_df=_trial_df,
            trial_indices=trial_indices,
            n_components=int(n_components),
            condition_names=condition_names,
            target=target,
            cv=int(cv),
            n_permutations=int(n_permutations),
            random_state=42,
        )
    return results_df, int(trial_indices.size), score_summary_df, raw_score_df


@st.cache_data(show_spinner="Computing PCA switch trajectories...")
def compute_population_pca_switch_trajectories_cached(
    session_key: str,
    aligned_spike_path: str,
    unit_ids: tuple[int, ...],
    pre_switch_filter: str,
    normalization: str,
    _spike_group,
    _trial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Fit choice-aligned PCA on all valid trials and extract switch trajectories.

    Parameters
    ----------
    session_key : str
        Session/probe identifier included in the Streamlit cache key.
    aligned_spike_path : str
        Aligned spike file path included in the cache key. Units: filesystem
        path.
    unit_ids : tuple[int, ...]
        Unit ids used as PCA features, shape ``(n_units,)``.
    pre_switch_filter : str
        Previous-trial outcome filter from
        ``population_pca_switch_trajectories.SWITCH_PRE_FILTER_OPTIONS``.
    normalization : str
        Unit normalization mode passed to ``fit_population_pca``.
    _spike_group
        Pynapple spike group keyed by unit id. The leading underscore excludes
        this potentially large object from Streamlit hashing.
    _trial_df : pd.DataFrame
        Full trial table with choice times in seconds and scalar action,
        correctness, and reward labels. The leading underscore excludes the
        dataframe from Streamlit hashing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, int]
        ``(trajectory_df, event_count_df, n_pca_trials)``. ``trajectory_df``
        has four PC1/PC2 rows per plotted switch event. ``event_count_df`` has
        one row per supported correctness transition. ``n_pca_trials`` is the
        number of all-valid trials used for the PCA fit.
    """

    trial_indices = population_pca_switch_trajectories.select_valid_choice_trial_indices(
        _trial_df
    )
    if trial_indices.size == 0:
        raise ValueError("No valid trials have finite choice times and actions.")
    rate_tensor_hz, _bin_centers_s = population_pca_decoding.build_choice_aligned_rate_tensor(
        spike_group=_spike_group,
        unit_ids=np.asarray(unit_ids, dtype=int),
        trial_df=_trial_df,
        trial_indices=trial_indices,
        bin_size_s=population_pca_decoding.PCA_DECODING_BIN_SIZE_S,
    )
    pca_result = population_pca.fit_population_pca(
        rate_tensor_hz=rate_tensor_hz,
        n_components=2,
        normalization=normalization,
    )
    switch_events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df=_trial_df,
        pre_switch_filter=pre_switch_filter,
    )
    trajectory_df = population_pca_switch_trajectories.extract_switch_event_pca_trajectories(
        pca_scores=pca_result.scores,
        pca_trial_indices=trial_indices,
        switch_events=switch_events,
    )
    event_count_df = population_pca_switch_trajectories.summarize_switch_event_counts(
        switch_events
    )
    return trajectory_df, event_count_df, int(trial_indices.size)


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


def build_trial_view_plot_save_path(
    figure_path: Path | str,
    session_id: str,
    region_name: str,
    trial_index: int,
    condition: str,
    action_label: str,
    alignment_event: str,
    window: tuple[float, float],
    unit_page_index: int,
    population_psth_unit_scope: str,
    population_psth_bin_size: float,
    lfp_label: str | None = None,
    lfp_utc_offset_hours: float | None = None,
) -> Path:
    """
    Build a setting-specific PNG path for one trial-view webapp plot.

    Parameters
    ----------
    figure_path : Path | str
        Session figure directory. The returned path is inside its
        ``unit_spike_viewer`` subdirectory.
    session_id : str
        Full session identifier.
    region_name : str
        Active region label.
    trial_index : int
        Trial row index shown in the plot.
    condition : str
        Active trial-condition filter.
    action_label : str
        Active action filter label.
    alignment_event : str
        Event column used as time zero.
    window : tuple[float, float]
        Relative plot window in seconds as ``(start_s, end_s)``.
    unit_page_index : int
        Zero-based unit page index.
    population_psth_unit_scope : str
        Population PSTH unit-scope setting.
    population_psth_bin_size : float
        Population PSTH bin size in seconds.
    lfp_label : str | None, optional
        LFP trace label, including selected LFP source, saved channel, and
        filter band when applicable. ``None`` means no LFP trace was plotted.
    lfp_utc_offset_hours : float | None, optional
        LFP UTC offset in hours when an LFP trace was plotted.

    Returns
    -------
    Path
        Output PNG path. Parent directories are not created by this helper.
    """

    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    filename_parts = [
        session_id,
        region_name,
        f"trial{int(trial_index)}",
        condition,
        action_label,
        alignment_event,
        f"window{float(window[0])}-to-{float(window[1])}s",
        f"unitpage{int(unit_page_index)}",
        population_psth_unit_scope,
        f"popbin{float(population_psth_bin_size)}s",
    ]
    if lfp_label is not None:
        filename_parts.append(lfp_label)
        if lfp_utc_offset_hours is not None:
            filename_parts.append(f"lfputc{float(lfp_utc_offset_hours):g}h")

    filename = "_".join(
        unit_spike_plotting._sanitize_filename_part(filename_part)
        for filename_part in filename_parts
    ) + ".png"
    return Path(figure_path) / "unit_spike_viewer" / filename


def format_metadata_row_for_display(metadata_row: pd.Series) -> pd.DataFrame:
    """
    Convert one mixed-type metadata row into a Streamlit display dataframe.

    Parameters
    ----------
    metadata_row : pd.Series
        One unit's metadata values indexed by metadata field name. Values may
        mix numeric entries and manual labels such as ``"mua"``.

    Returns
    -------
    pd.DataFrame
        One-column dataframe with the same index as ``metadata_row`` and a
        string-valued ``"value"`` column. This avoids PyArrow numeric coercion
        errors when Streamlit renders mixed-type metadata.
    """

    display_values = []
    for value in metadata_row.to_list():
        if value is None:
            display_values.append("")
            continue
        try:
            missing_value = pd.isna(value)
        except (TypeError, ValueError):
            missing_value = False
        if isinstance(missing_value, (bool, np.bool_)) and missing_value:
            display_values.append("")
            continue
        if isinstance(value, np.generic):
            value = value.item()
        display_values.append(str(value))
    return pd.DataFrame({"value": display_values}, index=metadata_row.index)


def _spikeglx_meta_path(binary_path: Path) -> Path:
    """
    Return the SpikeGLX metadata path expected by the local readSGLX helper.

    Parameters
    ----------
    binary_path : Path
        SpikeGLX binary path, typically ending in ``.lf.bin``.

    Returns
    -------
    Path
        Sibling metadata path with ``binary_path.stem + ".meta"`` naming.
    """

    return binary_path.parent / f"{binary_path.stem}.meta"


def _initialize_path_input_state() -> None:
    """Initialize editable path text boxes once per Streamlit session."""

    path_defaults = {
        "session_data_home_input": str(unit_spike_loading.DEFAULT_SESSION_DATA_HOME),
        "hpc_v1_sorter_output_path_input": str(unit_spike_loading.DEFAULT_HPC_SORTER_OUTPUT_PATH),
        "hpc_v1_aligned_spike_path_input": str(unit_spike_loading.DEFAULT_HPC_V1_ALIGNED_SPIKE_PATH),
        "hpc_v1_lfp_path_input": str(unit_spike_loading.DEFAULT_HPC_V1_LFP_PATH),
        "pfc_sorter_output_path_input": str(unit_spike_loading.DEFAULT_PFC_SORTER_OUTPUT_PATH),
        "pfc_aligned_spike_path_input": str(unit_spike_loading.DEFAULT_PFC_ALIGNED_SPIKE_PATH),
        "pfc_lfp_path_input": str(unit_spike_loading.DEFAULT_PFC_LFP_PATH),
        "path_browser_current_dir": str(DEFAULT_BROWSER_ROOT),
    }
    for state_key, default_value in path_defaults.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = default_value


def _render_path_browser() -> None:
    """
    Render an optional server-side filesystem browser for path text inputs.

    Parameters
    ----------
    None
        Uses and mutates Streamlit session state keys for path inputs.

    Returns
    -------
    None
        Browser selections update existing text-input state.
    """

    with st.sidebar.expander("Browse paths"):
        path_targets = {
            "Session data home": {"key": "session_data_home_input", "kind": "directory"},
            "HPC/V1 sorter output": {"key": "hpc_v1_sorter_output_path_input", "kind": "directory"},
            "HPC/V1 aligned spikes": {"key": "hpc_v1_aligned_spike_path_input", "kind": "file"},
            "HPC/V1 LFP": {"key": "hpc_v1_lfp_path_input", "kind": "file"},
            "PFC sorter output": {"key": "pfc_sorter_output_path_input", "kind": "directory"},
            "PFC aligned spikes": {"key": "pfc_aligned_spike_path_input", "kind": "file"},
            "PFC LFP": {"key": "pfc_lfp_path_input", "kind": "file"},
        }
        current_dir = Path(st.session_state["path_browser_current_dir"]).expanduser()
        if not current_dir.exists() or not current_dir.is_dir():
            st.warning(f"Browser path is invalid; resetting to {DEFAULT_BROWSER_ROOT}")
            current_dir = DEFAULT_BROWSER_ROOT
            st.session_state["path_browser_current_dir"] = str(current_dir)

        st.caption(f"Current directory: {current_dir}")
        browser_root = st.text_input(
            "Browser root/current directory",
            key="path_browser_current_dir",
        )
        current_dir = Path(browser_root).expanduser()

        try:
            entries = unit_spike_loading.list_path_browser_entries(
                current_dir,
                file_suffix=None,
            )
        except (FileNotFoundError, NotADirectoryError, PermissionError) as error:
            st.warning(str(error))
            return

        selected_target_label = st.selectbox(
            "Set path field",
            options=list(path_targets.keys()),
            key="path_browser_target_select",
        )
        selected_target = path_targets[selected_target_label]

        if st.button("Up", key="path_browser_up"):
            st.session_state["path_browser_current_dir"] = str(current_dir.parent)
            st.rerun()

        directory_options = [path.name for path in entries["directories"]]
        selected_directory_name = st.selectbox(
            "Subdirectories",
            options=directory_options,
            index=0 if directory_options else None,
            placeholder="No subdirectories",
            key="path_browser_directory_select",
        )
        if st.button("Open directory", key="path_browser_open_directory") and selected_directory_name:
            st.session_state["path_browser_current_dir"] = str(current_dir / selected_directory_name)
            st.rerun()

        if selected_target["kind"] == "directory" and st.button("Use current directory", key="path_browser_use_dir"):
            st.session_state[selected_target["key"]] = str(current_dir)
            st.rerun()

        file_options = [path.name for path in entries["files"]]
        selected_file_name = st.selectbox(
            "Files",
            options=file_options,
            index=0 if file_options else None,
            placeholder="No files",
            key="path_browser_file_select",
        )
        if selected_target["kind"] == "file" and st.button("Use selected file", key="path_browser_use_file") and selected_file_name:
            st.session_state[selected_target["key"]] = str(current_dir / selected_file_name)
            st.rerun()


def _select_unit_metadata(cluster_info, region_channels):
    """Render unit-quality controls and return filtered unit metadata."""

    quality_column_options = [column for column in ("group", "KSLabel") if column in cluster_info.columns]
    if not quality_column_options:
        st.warning("cluster_info.tsv must contain a quality column such as group or KSLabel.")
        return cluster_info.iloc[0:0].copy()
    quality_column = st.sidebar.selectbox(
        "Quality column",
        options=quality_column_options,
        index=0,
        help="Use group for manual Phy labels or KSLabel for Kilosort labels.",
    )

    quality_labels = unit_spike_loading.normalize_quality_labels(cluster_info[quality_column])
    available_quality_labels = sorted(quality_labels.unique().tolist())
    default_quality_labels = [label for label in ("good", "mua") if label in available_quality_labels]
    if not default_quality_labels:
        default_quality_labels = available_quality_labels
    selected_quality_labels = st.sidebar.multiselect(
        "Quality labels",
        options=available_quality_labels,
        default=default_quality_labels,
        help="Quality labels come from cluster_info.tsv. String 'nan' labels are shown as mua.",
    )
    if not selected_quality_labels:
        st.warning("Select at least one quality label to show units.")
        return cluster_info.iloc[0:0].copy()

    return unit_spike_loading.filter_cluster_metadata(
        cluster_info,
        region_channels=region_channels,
        quality_labels=tuple(selected_quality_labels),
        quality_column=quality_column,
    )


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
        Streamlit renders frequency metrics, a polar phase histogram, and
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
    try:
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
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should report unit/site-specific failures.
        st.error(f"Could not compute spike-LFP phase locking: {error}")
        st.stop()

    figure, _axes = unit_spike_plotting.plot_spike_lfp_phase_locking(
        result=result,
        polar_frequency_hz=polar_frequency_hz,
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
        )
        st.write(f"Retained phase data: {retained_bytes / (1024.0**2):.1f} MB")
    with figure_column:
        st.pyplot(figure, width="stretch")
        st.caption(
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
            "amplitude_units": "source-dependent wavelet magnitude",
            "phase_units": "radians",
            "time_units": "absolute synchronized seconds",
            "axis_order": ["frequency", "spike"],
            "random_seed": None,
        }
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


def main() -> None:
    """
    Run the local Streamlit unit raster/PSTH browser.

    Parameters
    ----------
    None
        Streamlit controls provide session paths, unit selection, and plotting
        settings.

    Returns
    -------
    None
        The Streamlit app renders plots and optional saved PNGs.
    """

    st.set_page_config(page_title="Unit PSTH Viewer", layout="wide")
    st.title("Unit Raster and PSTH Viewer")

    if st.sidebar.button("Clear cached data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    _initialize_path_input_state()
    st.sidebar.header("Session")
    session_data_home = st.sidebar.text_input(
        "Session data home",
        key="session_data_home_input",
    )
    sess_id_full = st.sidebar.text_input(
        "Session id",
        value=unit_spike_loading.DEFAULT_SESSION_ID,
    )

    st.sidebar.header("Probe Paths")
    hpc_v1_sorter_output_path = st.sidebar.text_input(
        "HPC/V1 sorter output path",
        key="hpc_v1_sorter_output_path_input",
    )
    hpc_v1_aligned_spike_path = st.sidebar.text_input(
        "HPC/V1 aligned spike path",
        key="hpc_v1_aligned_spike_path_input",
    )
    hpc_v1_lfp_path = st.sidebar.text_input(
        "HPC/V1 LFP path",
        key="hpc_v1_lfp_path_input",
    )
    pfc_sorter_output_path = st.sidebar.text_input(
        "PFC sorter output path",
        key="pfc_sorter_output_path_input",
    )
    pfc_aligned_spike_path = st.sidebar.text_input(
        "PFC aligned spike path",
        key="pfc_aligned_spike_path_input",
    )
    pfc_lfp_path = st.sidebar.text_input(
        "PFC LFP path",
        key="pfc_lfp_path_input",
    )
    _render_path_browser()
    plot_view = st.sidebar.selectbox("Plot view", options=PLOT_VIEW_OPTIONS)

    if plot_view in {PLOT_VIEW_LFP_PHASE_CLUSTERING, PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE}:
        try:
            phase_session, phase_event_df, phase_trial_df = load_phase_clustering_session_cached(
                session_data_home=session_data_home,
                sess_id_full=sess_id_full,
            )
        except Exception as error:  # noqa: BLE001 - Streamlit should display behavior loading failures cleanly.
            st.error(f"Could not load session behavior data: {error}")
            st.stop()
        if plot_view == PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE:
            render_single_trial_relative_phase_view(
                trial_df=phase_trial_df,
                event_df=phase_event_df,
                session=phase_session,
                hpc_v1_lfp_path=hpc_v1_lfp_path,
                pfc_lfp_path=pfc_lfp_path,
                hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
                pfc_aligned_spike_path=pfc_aligned_spike_path,
            )
        else:
            render_lfp_phase_clustering_view(
                trial_df=phase_trial_df,
                session=phase_session,
                hpc_v1_lfp_path=hpc_v1_lfp_path,
                pfc_lfp_path=pfc_lfp_path,
                hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
                pfc_aligned_spike_path=pfc_aligned_spike_path,
            )
        return

    st.sidebar.header("Region and Units")
    preset_options = list(unit_spike_loading.get_ct014_region_channel_presets().keys())
    region_name = st.sidebar.selectbox("Region preset", options=preset_options, index=0)
    custom_probe_label = None
    if region_name == "Custom":
        custom_probe_label = st.sidebar.selectbox(
            "Custom probe source",
            options=list(unit_spike_loading.SUPPORTED_PROBE_LABELS),
        )
    try:
        active_probe_label = unit_spike_loading.get_probe_label_for_region(
            region_name,
            custom_probe_label=custom_probe_label,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.sidebar.caption(f"Active probe: {active_probe_label}")

    if active_probe_label == unit_spike_loading.PROBE_LABEL_HPC_V1:
        active_sorter_output_path = hpc_v1_sorter_output_path
        active_aligned_spike_path = hpc_v1_aligned_spike_path
        active_lfp_path = hpc_v1_lfp_path
    else:
        active_sorter_output_path = pfc_sorter_output_path
        active_aligned_spike_path = pfc_aligned_spike_path
        active_lfp_path = pfc_lfp_path

    inferred_probe_derived_dir = unit_spike_loading.infer_probe_derived_dir(
        sorter_output_path=active_sorter_output_path,
        lfp_path=active_lfp_path,
    )
    channel_quality_path = None
    if inferred_probe_derived_dir is not None:
        try:
            channel_quality_path = unit_spike_loading.resolve_channel_quality_path(inferred_probe_derived_dir)
        except (FileNotFoundError, ValueError):
            channel_quality_path = None
    channel_source_options = [CHANNEL_SOURCE_MANUAL]
    if channel_quality_path is not None:
        channel_source_options.append(CHANNEL_SOURCE_CHANNEL_QUALITY)
    channel_source = st.sidebar.selectbox(
        "Channel source",
        options=channel_source_options,
        index=1 if CHANNEL_SOURCE_CHANNEL_QUALITY in channel_source_options else 0,
        help="Use channel_quality when available to select good in-brain probe sites.",
    )

    channel_quality = None
    channel_quality_labels = ("good",)
    require_inside_brain = True
    if channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY:
        try:
            channel_quality = load_channel_quality_cached(str(channel_quality_path))
        except Exception as error:  # noqa: BLE001 - Streamlit should show metadata failures cleanly.
            st.error(f"Could not load channel quality: {error}")
            st.stop()
        st.sidebar.caption(f"Channel quality: {channel_quality_path}")
        available_channel_labels = sorted(channel_quality["label"].astype(str).str.strip().str.lower().unique().tolist())
        default_channel_labels = ["good"] if "good" in available_channel_labels else available_channel_labels
        channel_quality_labels = tuple(
            st.sidebar.multiselect(
                "Channel labels",
                options=available_channel_labels,
                default=default_channel_labels,
                help="Good-site selection uses channel_quality label values; label='good' is the default.",
            )
        )
        require_inside_brain = st.sidebar.checkbox("Inside brain only", value=True)
        label_good_mask = channel_quality["label"].astype(str).str.strip().str.lower().eq("good")
        is_good_mask = channel_quality["is_good"].astype(bool)
        disagreement_count = int((label_good_mask != is_good_mask).sum())
        if disagreement_count > 0:
            st.sidebar.warning(f"{disagreement_count} channels disagree between label == 'good' and is_good.")
        manual_channel_text = _build_channel_text(region_name)
    else:
        manual_channel_text = st.sidebar.text_area(
            "Region channels",
            value=_build_channel_text(region_name),
            key=f"channel_text_{region_name}",
            height=140,
            help="CT014 presets are editable defaults. Replace with custom channel ids when needed.",
        )

    try:
        region_channels, channel_summary = resolve_region_channels_for_source(
            channel_source=channel_source,
            manual_channel_text=manual_channel_text,
            channel_quality=channel_quality,
            require_inside_brain=require_inside_brain,
            channel_quality_labels=channel_quality_labels,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()
    st.sidebar.caption(channel_summary)
    if region_channels.size == 0:
        st.warning("No region channels are selected.")
        st.stop()

    try:
        viewer_data = load_viewer_data_cached(
            session_data_home=session_data_home,
            sess_id_full=sess_id_full,
            active_probe_label=active_probe_label,
            sorter_output_path=active_sorter_output_path,
            aligned_spike_path=active_aligned_spike_path,
            lfp_path=active_lfp_path,
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should display load failures without a traceback wall.
        st.error(f"Could not load {active_probe_label} spike data: {error}")
        st.stop()

    session = viewer_data["session"]
    event_df = viewer_data["event_df"]
    trial_df = viewer_data["trial_df"]
    cluster_info = viewer_data["cluster_info"]
    spike_group = viewer_data["spike_group"]

    selected_unit_metadata = _select_unit_metadata(cluster_info, region_channels)
    if selected_unit_metadata.empty:
        st.warning("No units match the selected channels and quality filters.")
        st.stop()

    unit_ids = selected_unit_metadata["cluster_id"].to_numpy(dtype=int)
    if plot_view == PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING:
        render_spike_lfp_phase_locking_view(
            trial_df=trial_df,
            session=session,
            spike_group=spike_group,
            unit_ids=unit_ids,
            active_probe_label=active_probe_label,
            hpc_v1_lfp_path=hpc_v1_lfp_path,
            pfc_lfp_path=pfc_lfp_path,
            hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
            pfc_aligned_spike_path=pfc_aligned_spike_path,
        )
        return
    if plot_view == PLOT_VIEW_PCA_SWITCH_TRAJECTORIES:
        st.sidebar.header("PCA Switch Trajectories")
        pre_switch_filter = st.sidebar.selectbox(
            "Pre-switch trials",
            options=list(population_pca_switch_trajectories.SWITCH_PRE_FILTER_OPTIONS),
            index=0,
        )
        switch_pca_normalization = st.sidebar.selectbox(
            "PCA normalization",
            options=list(population_pca.PCA_NORMALIZATION_OPTIONS),
            index=0,
        )
        pca_unit_ids = resolve_population_pca_unit_ids(
            selected_unit_metadata=selected_unit_metadata,
            page_unit_ids=None,
        )
        try:
            switch_trajectory_df, switch_event_counts, switch_pca_trial_count = (
                compute_population_pca_switch_trajectories_cached(
                    session_key=f"{session.sess_id_full}:{active_probe_label}",
                    aligned_spike_path=str(active_aligned_spike_path),
                    unit_ids=tuple(int(unit_id) for unit_id in pca_unit_ids),
                    pre_switch_filter=pre_switch_filter,
                    normalization=switch_pca_normalization,
                    _spike_group=spike_group,
                    _trial_df=trial_df,
                )
            )
        except Exception as error:  # noqa: BLE001 - Streamlit should show analysis failures cleanly.
            st.error(f"Could not compute PCA switch trajectories: {error}")
            st.stop()

        st.subheader("Population PCA Switch Trajectories")
        st.caption(
            f"PCA fit on {switch_pca_trial_count} valid trials from {pca_unit_ids.size} units; "
            "each point is one 0.5 s choice-aligned bin."
        )
        st.dataframe(
            switch_event_counts[["label", "n_events"]],
            width="content",
            hide_index=True,
        )
        switch_figure, _switch_axes = (
            population_pca_switch_trajectories.plot_switch_event_pca_trajectories(
                switch_trajectory_df
            )
        )
        st.pyplot(switch_figure)
        plt.close(switch_figure)
        return

    if plot_view == PLOT_VIEW_PCA_DECODING:
        st.sidebar.header("PCA Decoding")
        decode_target = st.sidebar.selectbox(
            "Decode target",
            options=["state_int", "action"],
            index=0,
        )
        pca_decoding_mode = st.sidebar.selectbox(
            "PCA fitting mode",
            options=list(population_pca_decoding.PCA_DECODING_MODE_OPTIONS),
            index=0,
        )
        pca_decoding_display = st.sidebar.selectbox(
            "Display",
            options=PCA_DECODING_DISPLAY_OPTIONS,
            index=0,
        )
        pca_decoding_component_minimum = resolve_pca_decoding_component_minimum(pca_decoding_display)
        pca_decoding_component_count = int(
            st.sidebar.number_input(
                "PCA component count",
                min_value=pca_decoding_component_minimum,
                value=max(PCA_DECODING_DEFAULT_COMPONENT_COUNT, pca_decoding_component_minimum),
                step=1,
            )
        )
        selected_base_conditions = tuple(
            st.sidebar.multiselect(
                "Base conditions",
                options=population_pca_decoding.PCA_DECODING_BASE_CONDITIONS,
                default=population_pca_decoding.PCA_DECODING_BASE_CONDITIONS,
            )
        )
        if not selected_base_conditions:
            st.warning("Select at least one base condition for PCA decoding.")
            st.stop()
        pca_decoding_cv = int(
            st.sidebar.number_input(
                "CV folds",
                min_value=2,
                value=PCA_DECODING_DEFAULT_CV_FOLDS,
                step=1,
            )
        )
        pca_decoding_permutations = int(
            st.sidebar.number_input(
                "Permutations",
                min_value=1,
                value=PCA_DECODING_DEFAULT_PERMUTATIONS,
                step=1,
            )
        )
        pca_decoding_normalization = population_pca.PCA_NORMALIZATION_ZSCORE
        if pca_decoding_mode == population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY:
            pca_decoding_normalization = st.sidebar.selectbox(
                "Exploratory PCA normalization",
                options=list(population_pca.PCA_NORMALIZATION_OPTIONS),
                index=0,
            )
        if (
            pca_decoding_display == PCA_DECODING_DISPLAY_AVERAGE_PC
            and pca_decoding_mode == population_pca_decoding.PCA_DECODING_MODE_RIGOROUS
        ):
            st.sidebar.warning("Average PC score plots use exploratory PCA coordinates.")
        show_raw_pc_scores = False
        if pca_decoding_display == PCA_DECODING_DISPLAY_AVERAGE_PC:
            show_raw_pc_scores = st.sidebar.checkbox(
                "Show raw trial PC scores",
                value=PCA_DECODING_SHOW_RAW_PC_SCORES_DEFAULT,
            )

        pca_unit_ids = resolve_population_pca_unit_ids(
            selected_unit_metadata=selected_unit_metadata,
            page_unit_ids=None,
        )
        try:
            (
                pca_decoding_results,
                pca_decoding_trial_count,
                pca_score_summary,
                pca_raw_scores,
            ) = compute_population_pca_decoding_cached(
                session_key=f"{session.sess_id_full}:{active_probe_label}",
                aligned_spike_path=str(active_aligned_spike_path),
                unit_ids=tuple(int(unit_id) for unit_id in pca_unit_ids),
                condition_names=selected_base_conditions,
                target=decode_target,
                mode=pca_decoding_mode,
                n_components=int(pca_decoding_component_count),
                cv=int(pca_decoding_cv),
                n_permutations=int(pca_decoding_permutations),
                normalization=pca_decoding_normalization,
                include_score_summary=pca_decoding_display == PCA_DECODING_DISPLAY_AVERAGE_PC,
                include_raw_score_points=show_raw_pc_scores,
                _spike_group=spike_group,
                _trial_df=trial_df,
            )
        except Exception as error:  # noqa: BLE001 - Streamlit should show analysis failures cleanly.
            st.error(f"Could not compute PCA decoding: {error}")
            st.stop()

        decoding_display = population_pca_decoding.summarize_pca_decoding_results(pca_decoding_results)
        metadata_column, result_column = st.columns([1, 3])
        with metadata_column:
            st.subheader("PCA Decoding")
            st.write(f"Active probe: {active_probe_label}")
            st.write(f"Filtered units: {selected_unit_metadata.shape[0]}")
            st.write(f"PCA units: {pca_unit_ids.size}")
            st.write(f"Base-condition trials: {pca_decoding_trial_count}")
            st.write(f"Target: {decode_target}")
            st.write(f"Mode: {pca_decoding_mode}")
            st.write(f"Requested PCs: {int(pca_decoding_component_count)}")
            st.write(f"Bin size: {population_pca_decoding.PCA_DECODING_BIN_SIZE_S:g} s")
            st.write("Choice windows: -0.5 to 0 s, 0 to 0.5 s")
        with result_column:
            if pca_decoding_display == PCA_DECODING_DISPLAY_AVERAGE_PC:
                st.subheader("Average PC Score by Condition and Target")
                st.dataframe(pca_score_summary, width="stretch")
                if show_raw_pc_scores:
                    st.caption(
                        "Raw PC points are condition memberships; a trial can appear more than once "
                        "when selected base conditions overlap."
                    )
                score_figure, _score_axes = population_pca_decoding.plot_average_pca_scores_by_condition_and_target(
                    pca_score_summary,
                    raw_score_df=pca_raw_scores if show_raw_pc_scores else None,
                )
                st.pyplot(score_figure)
                plt.close(score_figure)
            else:
                st.subheader("Before/After Choice Decoding Performance")
                st.dataframe(decoding_display, width="stretch")
                decoding_figure, _decoding_axis = population_pca_decoding.plot_pca_decoding_pre_post_scores(
                    decoding_display
                )
                st.pyplot(decoding_figure)
                plt.close(decoding_figure)
        return

    st.sidebar.header("Trials")
    condition = st.sidebar.selectbox("Condition", options=CONDITION_OPTIONS)
    action_label = st.sidebar.selectbox("Action", options=list(ACTION_OPTIONS.keys()))
    alignment_event = st.sidebar.selectbox("Alignment event", options=ALIGNMENT_OPTIONS)
    raster_layout_label = st.sidebar.selectbox(
        "Raster row spacing",
        options=list(RASTER_LAYOUT_OPTIONS.keys()),
        index=1,
    )
    raster_layout = RASTER_LAYOUT_OPTIONS[raster_layout_label]
    window_start = st.sidebar.number_input("Window start (s)", value=-2.0, step=0.1)
    window_end = st.sidebar.number_input("Window end (s)", value=2.0, step=0.1)
    if float(window_start) >= float(window_end):
        st.error("Window start must be less than window end.")
        st.stop()
    window = (float(window_start), float(window_end))

    selected_action = ACTION_OPTIONS[action_label]
    compare_left_right = selected_action == COMPARE_LEFT_RIGHT_ACTION
    selected_trial_indices = unit_spike_plotting.filter_trials_for_unit_plot(
        trial_df,
        condition=condition,
        action="all" if compare_left_right else selected_action,
    )
    if selected_trial_indices.size == 0:
        st.warning("No trials match the selected filters.")
        st.stop()

    if plot_view == "Unit raster/PSTH":
        unit_id = st.sidebar.selectbox("Cluster id", options=unit_ids.tolist())
        selected_unit_row = selected_unit_metadata.loc[selected_unit_metadata["cluster_id"] == unit_id].iloc[0]
        unit_plot_type = st.sidebar.selectbox("Unit plot type", options=UNIT_PLOT_TYPE_OPTIONS)
        unit_spike_times = unit_spike_loading.get_unit_spike_times(spike_group, unit_id=int(unit_id))
        title_suffix = (
            f"{region_name}; {active_probe_label}; {condition}; {action_label}; "
            f"n={selected_trial_indices.size}"
        )
        page_size = st.sidebar.selectbox("Raster trial page size", options=PAGE_SIZE_OPTIONS, index=1)
        event_marker_columns: list[str] = []
        if alignment_event == "start_time":
            if st.sidebar.checkbox("Show choice markers", value=True):
                event_marker_columns.append("choice_time")
        elif st.sidebar.checkbox("Show trial start markers", value=True):
            event_marker_columns.append("start_time")
        if st.sidebar.checkbox("Show LED markers", value=True):
            event_marker_columns.append("led_on_time")

        plot_type_token = "raster-psth"
        summary_plot_type = "psth"
        binned_rate_bin_size = 0.1
        if unit_plot_type == "PSTH":
            bin_size = st.sidebar.selectbox("PSTH bin size (s)", options=PSTH_BIN_OPTIONS, index=1)
        else:
            bin_size = 0.1
            firing_rate_bin_size = st.sidebar.number_input(
                "Firing rate bin size (s)",
                min_value=0.001,
                value=0.1,
                step=0.01,
                format="%.3f",
            )
            binned_rate_bin_size = float(firing_rate_bin_size)
            if unit_plot_type == "Binned rate: trials + mean":
                plot_type_token = "binned-rate-trials-mean"
                summary_plot_type = "binned-rate-trials-mean"
            else:
                plot_type_token = "binned-rate-mean-sd"
                summary_plot_type = "binned-rate-mean-sd"

        comparison_counts = None
        if compare_left_right:
            (
                left_summary_trial_indices,
                right_summary_trial_indices,
            ) = unit_spike_plotting.split_trial_indices_by_action(
                trial_df=trial_df,
                trial_indices=selected_trial_indices,
            )
            if left_summary_trial_indices.size == 0 and right_summary_trial_indices.size == 0:
                st.warning("No left or right choice trials match the selected condition.")
                st.stop()
            left_pages = (
                math.ceil(left_summary_trial_indices.size / int(page_size))
                if left_summary_trial_indices.size
                else 0
            )
            right_pages = (
                math.ceil(right_summary_trial_indices.size / int(page_size))
                if right_summary_trial_indices.size
                else 0
            )
            n_pages = max(1, left_pages, right_pages)
            page_index = int(
                st.sidebar.number_input(
                    "Raster trial page",
                    min_value=0,
                    max_value=n_pages - 1,
                    value=0,
                    step=1,
                )
            )
            left_raster_trial_indices = unit_spike_plotting.paginate_trial_indices(
                left_summary_trial_indices,
                page_index=int(page_index),
                page_size=int(page_size),
            )
            right_raster_trial_indices = unit_spike_plotting.paginate_trial_indices(
                right_summary_trial_indices,
                page_index=int(page_index),
                page_size=int(page_size),
            )
            comparison_counts = {
                "left_summary": left_summary_trial_indices.size,
                "right_summary": right_summary_trial_indices.size,
                "left_raster": left_raster_trial_indices.size,
                "right_raster": right_raster_trial_indices.size,
            }
            plot_type_token = f"choice-comparison-{summary_plot_type}"
            comparison_figure_size = (
                max(float(raster_layout["figure_size"][0]) * 1.45, 14.0),
                float(raster_layout["figure_size"][1]),
            )
            figure, axes = unit_spike_plotting.plot_unit_left_right_choice_comparison(
                unit_spike_times=unit_spike_times,
                trial_df=trial_df,
                left_raster_trial_indices=left_raster_trial_indices,
                right_raster_trial_indices=right_raster_trial_indices,
                left_summary_trial_indices=left_summary_trial_indices,
                right_summary_trial_indices=right_summary_trial_indices,
                alignment_event=alignment_event,
                window=window,
                bin_size=float(bin_size),
                unit_id=int(unit_id),
                summary_plot_type=summary_plot_type,
                binned_rate_bin_size=float(binned_rate_bin_size),
                title_suffix=f"{title_suffix}; page {int(page_index) + 1}/{n_pages}",
                event_marker_columns=tuple(event_marker_columns),
                event_marker_styles=EVENT_MARKER_STYLES,
                raster_row_spacing=raster_layout["row_spacing"],
                figure_size=comparison_figure_size,
            )
        else:
            n_pages = max(1, math.ceil(selected_trial_indices.size / int(page_size)))
            page_index = int(
                st.sidebar.number_input(
                    "Raster trial page",
                    min_value=0,
                    max_value=n_pages - 1,
                    value=0,
                    step=1,
                )
            )
            raster_trial_indices = unit_spike_plotting.paginate_trial_indices(
                selected_trial_indices,
                page_index=int(page_index),
                page_size=int(page_size),
            )
            raster_title_suffix = (
                f"{title_suffix}; page {int(page_index) + 1}/{n_pages}; "
                f"summary n={selected_trial_indices.size}"
            )
            figure, axes = unit_spike_plotting.plot_unit_raster_and_psth(
                unit_spike_times=unit_spike_times,
                trial_df=trial_df,
                raster_trial_indices=raster_trial_indices,
                psth_trial_indices=selected_trial_indices,
                alignment_event=alignment_event,
                window=window,
                bin_size=float(bin_size),
                unit_id=int(unit_id),
                title_suffix=raster_title_suffix,
                event_marker_columns=tuple(event_marker_columns),
                event_marker_styles=EVENT_MARKER_STYLES,
                raster_row_spacing=raster_layout["row_spacing"],
                figure_size=raster_layout["figure_size"],
                summary_plot_type=summary_plot_type,
                binned_rate_bin_size=float(binned_rate_bin_size),
            )
            comparison_counts = {
                "raster": raster_trial_indices.size,
            }

        metadata_column, plot_column = st.columns([1, 3])
        with metadata_column:
            st.subheader("Unit Metadata")
            st.dataframe(format_metadata_row_for_display(selected_unit_row), width="stretch")
            st.write(f"Active probe: {active_probe_label}")
            st.write(f"Unit plot type: {unit_plot_type}")
            st.write(f"Filtered units: {selected_unit_metadata.shape[0]}")
            st.write(f"Filtered trials: {selected_trial_indices.size}")
            if compare_left_right and comparison_counts is not None:
                st.write(f"Left summary trials: {comparison_counts['left_summary']}")
                st.write(f"Right summary trials: {comparison_counts['right_summary']}")
                st.write(f"Left raster trials on page: {comparison_counts['left_raster']}")
                st.write(f"Right raster trials on page: {comparison_counts['right_raster']}")
            elif comparison_counts is not None:
                st.write(f"Raster trials on page: {comparison_counts['raster']}")
        with plot_column:
            st.pyplot(figure)

        if st.button("Save current plot"):
            save_path = unit_spike_plotting.save_unit_plot_figure(
                figure=figure,
                figure_path=session.figure_path,
                session_id=session.sess_id_full,
                unit_id=int(unit_id),
                region_name=region_name,
                condition=condition,
                action_label=action_label,
                alignment_event=alignment_event,
                page_index=int(page_index),
                plot_type=plot_type_token,
            )
            st.success(f"Saved plot to {save_path}")

    else:
        neural_display = st.sidebar.selectbox("Neural display", options=NEURAL_DISPLAY_OPTIONS)
        plot_trial_indices = selected_trial_indices
        invalid_alignment_trial_indices = np.array([], dtype=int)
        pca_display = is_population_pca_display(neural_display)
        lfp_spectrogram_display = neural_display == NEURAL_DISPLAY_LFP_SPECTROGRAM
        if pca_display or lfp_spectrogram_display:
            try:
                plot_trial_indices, invalid_alignment_trial_indices = filter_trial_indices_for_valid_alignment(
                    trial_df=trial_df,
                    trial_indices=selected_trial_indices,
                    alignment_event=alignment_event,
                )
            except ValueError as error:
                st.error(str(error))
                st.stop()
            if invalid_alignment_trial_indices.size > 0:
                preview = invalid_alignment_trial_indices[:10].tolist()
                st.warning(
                    f"Omitting {invalid_alignment_trial_indices.size} trials without valid "
                    f"{alignment_event} alignment times from this view. First omitted trials: {preview}"
                )
        concatenated_visible_trial_indices = np.array([], dtype=int)
        concatenated_visible_trial_positions = np.array([], dtype=int)
        concatenated_start_position = 0
        if neural_display == NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED:
            concatenated_visible_trial_count = st.sidebar.selectbox(
                "Visible concatenated trials",
                options=CONCATENATED_PCA_VISIBLE_TRIAL_OPTIONS,
                index=CONCATENATED_PCA_VISIBLE_TRIAL_OPTIONS.index(DEFAULT_CONCATENATED_PCA_VISIBLE_TRIAL_COUNT),
            )
            max_start_position = max(0, plot_trial_indices.size - int(concatenated_visible_trial_count))
            requested_start_position = int(
                st.sidebar.slider(
                    "Start trial position",
                    min_value=0,
                    max_value=int(max_start_position),
                    value=0,
                    step=1,
                )
            )
            concatenated_visible_trial_indices, concatenated_start_position = select_concatenated_trial_viewport(
                trial_indices=plot_trial_indices,
                start_position=requested_start_position,
                visible_trial_count=int(concatenated_visible_trial_count),
            )
            trial_position_by_index = {
                int(trial_index_value): trial_position_value
                for trial_position_value, trial_index_value in enumerate(plot_trial_indices)
            }
            concatenated_visible_trial_positions = np.asarray(
                [trial_position_by_index[int(trial_index_value)] for trial_index_value in concatenated_visible_trial_indices],
                dtype=int,
            )
            trial_position = int(concatenated_visible_trial_positions[0])
        else:
            trial_position = st.sidebar.number_input(
                "Trial position",
                min_value=0,
                max_value=int(plot_trial_indices.size - 1),
                value=0,
                step=1,
            )
        trial_index = int(plot_trial_indices[int(trial_position)])
        pca_bin_size_s = PCA_BIN_SIZE_OPTIONS[0]
        pca_normalization = population_pca.PCA_NORMALIZATION_ZSCORE
        pca_trajectory_component_count = DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT
        pca_variance_component_count = DEFAULT_PCA_VARIANCE_COMPONENT_COUNT
        if neural_display == NEURAL_DISPLAY_SPIKE_RASTER:
            unit_page_size = st.sidebar.selectbox("Units per page", options=PAGE_SIZE_OPTIONS, index=0)
            n_unit_pages = max(1, math.ceil(unit_ids.size / int(unit_page_size)))
            unit_page_index = st.sidebar.number_input(
                "Unit page",
                min_value=0,
                max_value=n_unit_pages - 1,
                value=0,
                step=1,
            )
            page_unit_ids = unit_spike_plotting.paginate_unit_ids(
                unit_ids,
                page_index=int(unit_page_index),
                page_size=int(unit_page_size),
            )
            population_psth_unit_scope = st.sidebar.selectbox(
                "Population PSTH units",
                options=POPULATION_PSTH_UNIT_SCOPE_OPTIONS,
            )
            population_psth_bin_size = st.sidebar.selectbox(
                "Population PSTH bin size (s)",
                options=PSTH_BIN_OPTIONS,
                index=0,
            )
            psth_unit_ids = page_unit_ids if population_psth_unit_scope == "Visible page units" else unit_ids
        elif pca_display:
            pca_bin_size_s = st.sidebar.selectbox(
                "PCA bin size (s)",
                options=PCA_BIN_SIZE_OPTIONS,
                index=0,
            )
            pca_normalization = st.sidebar.selectbox(
                "PCA normalization",
                options=list(population_pca.PCA_NORMALIZATION_OPTIONS),
                index=0,
            )
            pca_trajectory_component_count = int(
                st.sidebar.number_input(
                    "Trajectory PC count",
                    min_value=1,
                    value=(
                        DEFAULT_CONCATENATED_PCA_COMPONENT_COUNT
                        if neural_display == NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED
                        else DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT
                    ),
                    step=1,
                )
            )
            pca_variance_component_count = int(
                st.sidebar.number_input(
                    "Variance PC count",
                    min_value=1,
                    value=DEFAULT_PCA_VARIANCE_COMPONENT_COUNT,
                    step=1,
                )
            )
            unit_page_index = 0
            n_unit_pages = 1
            page_unit_ids = np.array([], dtype=int)
            population_psth_unit_scope = "Omitted in PCA mode"
            population_psth_bin_size = float(pca_bin_size_s)
            psth_unit_ids = np.array([], dtype=int)
        else:
            unit_page_index = 0
            n_unit_pages = 1
            page_unit_ids = np.array([], dtype=int)
            population_psth_unit_scope = "Omitted in LFP spectrogram mode"
            population_psth_bin_size = float(PSTH_BIN_OPTIONS[0])
            psth_unit_ids = np.array([], dtype=int)
        if lfp_spectrogram_display:
            show_lfp_trace = True
        elif neural_display == NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED:
            show_lfp_trace = False
        else:
            show_lfp_trace = st.sidebar.checkbox("Show LFP trace", value=False)
        lfp_time_s = None
        lfp_uv = None
        lfp_label = None
        lfp_y_label = "LFP (uV)"
        lfp_power_unit_label = "dB re 1 uV^2"
        lfp_utc_offset_hours = None
        lfp_spectrogram_result = None
        lfp_spectrogram_power_limits = None
        lfp_spectrogram_reference_count = 0
        if show_lfp_trace:
            lfp_digital_word = 0
            lfp_irig_line = 6
            lfp_bit_period_s = 1.0
            lfp_format = st.sidebar.selectbox(
                "LFP format",
                options=LFP_FORMAT_OPTIONS,
                index=0,
            )
            lfp_utc_offset_hours = st.sidebar.number_input(
                "LFP UTC offset (hours)",
                value=0,
                step=1,
                format="%d",
                help=(
                    "Whole-hour offset added to decoded LFP IRIG timestamps before mapping "
                    "trial times to LFP samples. Use -1 if decoded LFP times are one hour too late."
                ),
            )
            if lfp_format == LFP_FORMAT_SPIKEGLX:
                st.sidebar.caption(f"LFP sync digital line: {lfp_irig_line}")
            st.sidebar.caption(f"LFP UTC offset: {int(lfp_utc_offset_hours)} h")
            if lfp_spectrogram_display:
                spectrogram_min_frequency_hz = float(
                    st.sidebar.number_input(
                        "Minimum frequency (Hz)",
                        min_value=0.1,
                        value=float(LFP_SPECTROGRAM_MIN_FREQUENCY_HZ),
                        step=0.5,
                    )
                )
                spectrogram_max_frequency_hz = float(
                    st.sidebar.number_input(
                        "Maximum frequency (Hz)",
                        min_value=0.2,
                        value=float(LFP_SPECTROGRAM_MAX_FREQUENCY_HZ),
                        step=5.0,
                    )
                )
                spectrogram_frequency_count = int(
                    st.sidebar.number_input(
                        "Frequency count",
                        min_value=2,
                        value=int(LFP_SPECTROGRAM_FREQUENCY_COUNT),
                        step=1,
                    )
                )
                spectrogram_gaussian_width = float(
                    st.sidebar.number_input(
                        "Morlet Gaussian width",
                        min_value=0.1,
                        value=float(LFP_SPECTROGRAM_GAUSSIAN_WIDTH),
                        step=0.1,
                    )
                )
                spectrogram_window_length = float(
                    st.sidebar.number_input(
                        "Morlet window length",
                        min_value=0.1,
                        value=float(LFP_SPECTROGRAM_WINDOW_LENGTH),
                        step=0.1,
                    )
                )
                spectrogram_norm = st.sidebar.selectbox(
                    "Morlet normalization",
                    options=["l1", "l2"],
                    index=0,
                )
                spectrogram_notch_60_hz = st.sidebar.checkbox(
                    "Apply 60 Hz notch",
                    value=LFP_SPECTROGRAM_NOTCH_DEFAULT,
                )
                if spectrogram_max_frequency_hz <= spectrogram_min_frequency_hz:
                    st.error("Maximum frequency must be greater than minimum frequency.")
                    st.stop()
                spectrogram_frequencies_hz = tuple(
                    np.geomspace(
                        spectrogram_min_frequency_hz,
                        spectrogram_max_frequency_hz,
                        spectrogram_frequency_count,
                    ).tolist()
                )
                lfp_filter_label = "Unfiltered input"
                lfp_filter_band = None
            else:
                lfp_filter_label = st.sidebar.selectbox(
                    "LFP filter band",
                    options=list(LFP_FILTER_BANDS.keys()),
                    index=0,
                )
                lfp_filter_band = resolve_lfp_filter_band(lfp_filter_label)
            lfp_dropdown_options = build_lfp_dropdown_options(
                hpc_v1_lfp_path=hpc_v1_lfp_path,
                pfc_lfp_path=pfc_lfp_path,
            )
            default_lfp_label = (
                LFP_DROPDOWN_LABEL_HPC_V1
                if active_probe_label == unit_spike_loading.PROBE_LABEL_HPC_V1
                else LFP_DROPDOWN_LABEL_PFC
            )
            selected_lfp_label = st.sidebar.selectbox(
                "Active LFP file",
                options=list(lfp_dropdown_options.keys()),
                index=list(lfp_dropdown_options.keys()).index(default_lfp_label),
            )
            selected_lfp_path = lfp_dropdown_options[selected_lfp_label]
            default_aligned_sync_path = (
                hpc_v1_aligned_spike_path
                if selected_lfp_label == LFP_DROPDOWN_LABEL_HPC_V1
                else pfc_aligned_spike_path
            )
            selected_aligned_sync_path = default_aligned_sync_path
            if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
                selected_aligned_sync_path = st.sidebar.text_input(
                    "Open Ephys aligned sync path",
                    value=str(default_aligned_sync_path),
                    help="Use the probe sync .npz produced by the Open Ephys spike synchronization workflow.",
                )
                lfp_y_label = "LFP"
                lfp_power_unit_label = "dB re 1 input-unit^2"
            st.sidebar.caption(selected_lfp_path or "No LFP path entered for this selection.")
            if channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY and region_channels.size > 0:
                lfp_saved_channel_index = st.sidebar.selectbox(
                    "LFP saved channel index",
                    options=region_channels.tolist(),
                    index=0,
                    help="Restricted to the selected channel_quality channels.",
                )
            else:
                lfp_saved_channel_index = st.sidebar.number_input(
                    "LFP saved channel index",
                    min_value=0,
                    value=0,
                    step=1,
                )
            if str(selected_lfp_path).strip() == "":
                st.warning("LFP trace requested, but no active LFP path is set.")
            else:
                if lfp_format == LFP_FORMAT_SPIKEGLX:
                    lfp_meta_path = _spikeglx_meta_path(Path(selected_lfp_path))
                    if not lfp_meta_path.exists():
                        st.warning(f"Expected LFP metadata file was not found: {lfp_meta_path}")
                try:
                    alignment_time_s = unit_spike_plotting.get_trial_alignment_time(
                        trial_df=trial_df,
                        trial_index=trial_index,
                        alignment_event=alignment_event,
                    )
                    aligned_sync_path_argument = (
                        str(selected_aligned_sync_path)
                        if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED
                        else None
                    )
                    lfp_label = f"{selected_lfp_label}, {lfp_format}, saved channel {int(lfp_saved_channel_index)}"
                    if lfp_spectrogram_display:
                        lfp_spectrogram_result = compute_trial_lfp_spectrogram_cached(
                            lfp_format=lfp_format,
                            lfp_path=str(selected_lfp_path),
                            saved_channel_index=int(lfp_saved_channel_index),
                            alignment_time_s=float(alignment_time_s),
                            visible_window_start_s=float(window[0]),
                            visible_window_end_s=float(window[1]),
                            digital_word=lfp_digital_word,
                            irig_line=lfp_irig_line,
                            bit_period_s=lfp_bit_period_s,
                            utc_offset_hours=float(lfp_utc_offset_hours),
                            frequencies_hz=spectrogram_frequencies_hz,
                            gaussian_width=spectrogram_gaussian_width,
                            window_length=spectrogram_window_length,
                            precision=LFP_SPECTROGRAM_PRECISION,
                            norm=spectrogram_norm,
                            target_sample_rate_hz=LFP_SPECTROGRAM_TARGET_SAMPLE_RATE_HZ,
                            notch_60_hz=spectrogram_notch_60_hz,
                            notch_quality_factor=LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
                            aligned_sync_npz_path=aligned_sync_path_argument,
                        )
                        reference_trial_indices = lfp_spectrogram.select_reference_trial_indices(
                            trial_df=trial_df,
                            alignment_event=alignment_event,
                            maximum_trial_count=LFP_SPECTROGRAM_REFERENCE_TRIAL_COUNT,
                        )
                        reference_alignment_times_s = tuple(
                            pd.to_numeric(
                                trial_df.loc[reference_trial_indices, alignment_event],
                                errors="coerce",
                            ).to_numpy(dtype=float)
                        )
                        (
                            lfp_spectrogram_power_limits,
                            lfp_spectrogram_reference_count,
                        ) = compute_shared_lfp_power_limits_cached(
                            reference_alignment_times_s=reference_alignment_times_s,
                            lfp_format=lfp_format,
                            lfp_path=str(selected_lfp_path),
                            saved_channel_index=int(lfp_saved_channel_index),
                            visible_window_start_s=float(window[0]),
                            visible_window_end_s=float(window[1]),
                            digital_word=lfp_digital_word,
                            irig_line=lfp_irig_line,
                            bit_period_s=lfp_bit_period_s,
                            utc_offset_hours=float(lfp_utc_offset_hours),
                            frequencies_hz=spectrogram_frequencies_hz,
                            gaussian_width=spectrogram_gaussian_width,
                            window_length=spectrogram_window_length,
                            precision=LFP_SPECTROGRAM_PRECISION,
                            norm=spectrogram_norm,
                            target_sample_rate_hz=LFP_SPECTROGRAM_TARGET_SAMPLE_RATE_HZ,
                            notch_60_hz=spectrogram_notch_60_hz,
                            notch_quality_factor=LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
                            lower_percentile=LFP_SPECTROGRAM_COLOR_PERCENTILES[0],
                            upper_percentile=LFP_SPECTROGRAM_COLOR_PERCENTILES[1],
                            aligned_sync_npz_path=aligned_sync_path_argument,
                        )
                        lfp_time_s = lfp_spectrogram_result.time_s
                        lfp_uv = lfp_spectrogram_result.lfp_values
                    else:
                        lfp_time_s, lfp_uv = load_trial_lfp_trace_cached(
                            lfp_format=lfp_format,
                            lfp_path=str(selected_lfp_path),
                            saved_channel_index=int(lfp_saved_channel_index),
                            alignment_time_s=float(alignment_time_s),
                            window_start_s=float(window[0]),
                            window_end_s=float(window[1]),
                            digital_word=lfp_digital_word,
                            irig_line=lfp_irig_line,
                            bit_period_s=lfp_bit_period_s,
                            utc_offset_hours=lfp_utc_offset_hours,
                            filter_low_hz=lfp_filter_band[0] if lfp_filter_band is not None else None,
                            filter_high_hz=lfp_filter_band[1] if lfp_filter_band is not None else None,
                            filter_padding_s=LFP_FILTER_PADDING_S,
                            aligned_sync_npz_path=aligned_sync_path_argument,
                        )
                    if lfp_filter_band is not None:
                        lfp_label = f"{lfp_label}, {lfp_filter_label}"
                except Exception as error:  # noqa: BLE001 - Optional LFP should not block raster plotting.
                    if lfp_spectrogram_display:
                        st.error(f"Could not build LFP spectrogram: {error}")
                    else:
                        st.warning(f"Could not load LFP trace; plotting rasters without LFP. {error}")
        variance_figure = None
        pca_result = None
        pca_unit_ids = np.array([], dtype=int)
        try:
            lick_times = spike_behavior_pynapple.build_lick_time_dict(event_df)
            if lfp_spectrogram_display:
                if lfp_spectrogram_result is None or lfp_spectrogram_power_limits is None or lfp_label is None:
                    raise ValueError("LFP spectrogram data are unavailable for the selected trial and channel.")
                figure, axes = unit_spike_plotting.plot_trial_lfp_spectrogram_and_behavior(
                    trial_df=trial_df,
                    trial_index=trial_index,
                    lick_times=lick_times,
                    spectrogram_time_s=lfp_spectrogram_result.time_s,
                    frequencies_hz=lfp_spectrogram_result.frequencies_hz,
                    log_power_db=lfp_spectrogram_result.log_power_db,
                    lfp_time_s=lfp_spectrogram_result.time_s,
                    lfp_values=lfp_spectrogram_result.lfp_values,
                    alignment_event=alignment_event,
                    window=window,
                    power_limits_db=lfp_spectrogram_power_limits,
                    lfp_label=lfp_label,
                    power_unit_label=lfp_power_unit_label,
                    reference_trial_count=lfp_spectrogram_reference_count,
                    lfp_y_label=lfp_y_label,
                )
            elif pca_display:
                pca_unit_ids = resolve_population_pca_unit_ids(
                    selected_unit_metadata=selected_unit_metadata,
                    page_unit_ids=page_unit_ids,
                )
                pca_time_s, pca_result = compute_population_pca_cached(
                    session_key=f"{session.sess_id_full}:{active_probe_label}",
                    aligned_spike_path=str(active_aligned_spike_path),
                    unit_ids=tuple(int(unit_id) for unit_id in pca_unit_ids),
                    trial_indices=tuple(int(trial_index_value) for trial_index_value in plot_trial_indices),
                    alignment_event=alignment_event,
                    window_start_s=float(window[0]),
                    window_end_s=float(window[1]),
                    bin_size_s=float(pca_bin_size_s),
                    normalization=pca_normalization,
                    trajectory_component_count=int(pca_trajectory_component_count),
                    variance_component_count=int(pca_variance_component_count),
                    _spike_group=spike_group,
                    _trial_df=trial_df,
                )
                if neural_display == NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED:
                    figure, axes, _axis_data = unit_spike_plotting.plot_concatenated_trial_behavior_and_population_pca(
                        trial_df=trial_df,
                        trial_indices=concatenated_visible_trial_indices,
                        lick_times=lick_times,
                        pca_time_s=pca_time_s,
                        pca_scores=pca_result.scores[concatenated_visible_trial_positions, :, :],
                        alignment_event=alignment_event,
                        window=window,
                        pc_count=int(pca_trajectory_component_count),
                        figure_size=CONCATENATED_PCA_FIGURE_SIZE,
                        axis_mode="pseudo_time",
                    )
                else:
                    figure, axes = unit_spike_plotting.plot_trial_behavior_and_population_pca(
                        trial_df=trial_df,
                        trial_index=trial_index,
                        lick_times=lick_times,
                        pca_time_s=pca_time_s,
                        pca_scores=pca_result.scores,
                        trial_position=int(trial_position),
                        alignment_event=alignment_event,
                        window=window,
                        pc_count=int(pca_trajectory_component_count),
                        lfp_time_s=lfp_time_s,
                        lfp_uv=lfp_uv,
                        lfp_label=lfp_label,
                        lfp_y_label=lfp_y_label,
                        figure_size=raster_layout["figure_size"],
                    )
                variance_figure, _ = unit_spike_plotting.plot_pca_cumulative_explained_variance(
                    cumulative_explained_variance=pca_result.cumulative_explained_variance,
                    pc_count=int(pca_variance_component_count),
                )
            else:
                figure, axes = unit_spike_plotting.plot_trial_behavior_and_spike_raster(
                    trial_df=trial_df,
                    trial_index=trial_index,
                    lick_times=lick_times,
                    spike_group=spike_group,
                    raster_unit_ids=page_unit_ids,
                    psth_unit_ids=psth_unit_ids,
                    alignment_event=alignment_event,
                    window=window,
                    psth_bin_size=float(population_psth_bin_size),
                    lfp_time_s=lfp_time_s,
                    lfp_uv=lfp_uv,
                    lfp_label=lfp_label,
                    lfp_y_label=lfp_y_label,
                    figure_size=raster_layout["figure_size"],
                    spike_row_spacing=raster_layout["row_spacing"],
                )
        except Exception as error:  # noqa: BLE001 - Streamlit should show plot failures cleanly.
            st.error(f"Could not build trial behavior/neural plot: {error}")
            st.stop()

        metadata_column, plot_column = st.columns([1, 3])
        with metadata_column:
            st.subheader("Trial and Unit Page")
            st.write(f"Active probe: {active_probe_label}")
            st.write(f"Filtered units: {selected_unit_metadata.shape[0]}")
            st.write(f"Filtered trials: {selected_trial_indices.size}")
            st.write(f"Trial index: {trial_index}")
            st.write(f"Neural display: {neural_display}")
            if lfp_spectrogram_display:
                st.write(
                    f"Frequencies: {spectrogram_min_frequency_hz:g}-"
                    f"{spectrogram_max_frequency_hz:g} Hz ({spectrogram_frequency_count})"
                )
                st.write(f"Morlet Gaussian width: {spectrogram_gaussian_width:g}")
                st.write(f"Morlet window length: {spectrogram_window_length:g}")
                st.write(f"Morlet normalization: {spectrogram_norm}")
                st.write(f"60 Hz notch: {'on' if spectrogram_notch_60_hz else 'off'}")
                st.write(f"Shared power reference trials: {lfp_spectrogram_reference_count}")
                if lfp_spectrogram_power_limits is not None:
                    st.write(
                        "Shared power limits: "
                        f"{lfp_spectrogram_power_limits[0]:.1f} to "
                        f"{lfp_spectrogram_power_limits[1]:.1f} dB"
                    )
            elif pca_display:
                if invalid_alignment_trial_indices.size > 0:
                    st.write(
                        f"PCA trials omitted for missing {alignment_event}: "
                        f"{invalid_alignment_trial_indices.size}"
                    )
                st.write(f"PCA units: {pca_unit_ids.size}")
                st.write(f"PCA bin size: {float(pca_bin_size_s):g} s")
                st.write(f"PCA normalization: {pca_normalization}")
                if pca_result is not None:
                    trajectory_pcs_shown = min(
                        int(pca_trajectory_component_count),
                        pca_result.scores.shape[2],
                    )
                    st.write(f"Trajectory PCs shown: {trajectory_pcs_shown}")
                    st.write(f"Variance PCs fit: {pca_result.cumulative_explained_variance.size}")
                if neural_display == NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED:
                    st.write(
                        "Concatenated trial start position: "
                        f"{int(concatenated_start_position) + 1}/{plot_trial_indices.size}"
                    )
                    st.write(f"Visible concatenated trials: {concatenated_visible_trial_indices.size}")
            else:
                st.write(f"Unit page: {int(unit_page_index) + 1}/{n_unit_pages}")
                st.write(f"Units on page: {page_unit_ids.size}")
                st.write(f"Population PSTH units: {population_psth_unit_scope}")
                st.write(f"Population PSTH unit count: {psth_unit_ids.size}")
            if lfp_label is not None:
                st.write(lfp_label)
        with plot_column:
            st.pyplot(figure)
            if variance_figure is not None:
                st.pyplot(variance_figure)

        if st.button("Save current plot"):
            save_path = build_trial_view_plot_save_path(
                figure_path=session.figure_path,
                session_id=session.sess_id_full,
                region_name=region_name,
                trial_index=trial_index,
                condition=condition,
                action_label=action_label,
                alignment_event=alignment_event,
                window=window,
                unit_page_index=int(unit_page_index),
                population_psth_unit_scope=population_psth_unit_scope,
                population_psth_bin_size=float(population_psth_bin_size),
                lfp_label=lfp_label,
                lfp_utc_offset_hours=lfp_utc_offset_hours,
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_path, format="png", dpi=300)
            st.success(f"Saved plot to {save_path}")

    plt.close(figure)
    if "variance_figure" in locals() and variance_figure is not None:
        plt.close(variance_figure)


if __name__ == "__main__":
    main()
