from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import (
    lfp_loading,
    population_pca,
    spike_behavior_pynapple,
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
PLOT_VIEW_OPTIONS = ["Unit raster/PSTH", "Trial spikes/licks/choices"]
UNIT_PLOT_TYPE_OPTIONS = [
    "PSTH",
    "Binned rate: trials + mean",
    "Binned rate: mean +/- SD",
]
NEURAL_DISPLAY_SPIKE_RASTER = "Spike raster"
NEURAL_DISPLAY_POPULATION_PCA = "Population PCA"
NEURAL_DISPLAY_OPTIONS = [NEURAL_DISPLAY_SPIKE_RASTER, NEURAL_DISPLAY_POPULATION_PCA]
PCA_BIN_SIZE_OPTIONS = [0.1, 0.05, 0.02]
DEFAULT_PCA_COMPONENT_COUNT = 5
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
DEFAULT_BROWSER_ROOT = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014")
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
    component_count: int,
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
    component_count : int
        Requested number of PCs.
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
        ``(n_trials, n_bins, n_components)``.
    """

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
        n_components=int(component_count),
        normalization=normalization,
    )
    return pca_time_s, pca_result


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
    plot_view = st.sidebar.selectbox("Plot view", options=PLOT_VIEW_OPTIONS)

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
            st.dataframe(format_metadata_row_for_display(selected_unit_row), use_container_width=True)
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
        trial_position = st.sidebar.number_input(
            "Trial position",
            min_value=0,
            max_value=int(selected_trial_indices.size - 1),
            value=0,
            step=1,
        )
        trial_index = int(selected_trial_indices[int(trial_position)])
        neural_display = st.sidebar.selectbox("Neural display", options=NEURAL_DISPLAY_OPTIONS)
        pca_bin_size_s = PCA_BIN_SIZE_OPTIONS[0]
        pca_normalization = population_pca.PCA_NORMALIZATION_ZSCORE
        pca_component_count = DEFAULT_PCA_COMPONENT_COUNT
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
        else:
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
            pca_component_count = int(
                st.sidebar.number_input(
                    "PC count",
                    min_value=1,
                    value=DEFAULT_PCA_COMPONENT_COUNT,
                    step=1,
                )
            )
            unit_page_index = 0
            n_unit_pages = 1
            page_unit_ids = np.array([], dtype=int)
            population_psth_unit_scope = "Omitted in PCA mode"
            population_psth_bin_size = float(pca_bin_size_s)
            psth_unit_ids = np.array([], dtype=int)
        show_lfp_trace = st.sidebar.checkbox("Show LFP trace", value=False)
        lfp_time_s = None
        lfp_uv = None
        lfp_label = None
        lfp_y_label = "LFP (uV)"
        lfp_utc_offset_hours = None
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
                        aligned_sync_npz_path=(
                            str(selected_aligned_sync_path)
                            if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED
                            else None
                        ),
                    )
                    lfp_label = f"{selected_lfp_label}, {lfp_format}, saved channel {int(lfp_saved_channel_index)}"
                    if lfp_filter_band is not None:
                        lfp_label = f"{lfp_label}, {lfp_filter_label}"
                except Exception as error:  # noqa: BLE001 - Optional LFP should not block raster plotting.
                    st.warning(f"Could not load LFP trace; plotting rasters without LFP. {error}")
        variance_figure = None
        pca_result = None
        pca_unit_ids = np.array([], dtype=int)
        try:
            lick_times = spike_behavior_pynapple.build_lick_time_dict(event_df)
            if neural_display == NEURAL_DISPLAY_POPULATION_PCA:
                pca_unit_ids = resolve_population_pca_unit_ids(
                    selected_unit_metadata=selected_unit_metadata,
                    page_unit_ids=page_unit_ids,
                )
                pca_time_s, pca_result = compute_population_pca_cached(
                    session_key=f"{session.sess_id_full}:{active_probe_label}",
                    aligned_spike_path=str(active_aligned_spike_path),
                    unit_ids=tuple(int(unit_id) for unit_id in pca_unit_ids),
                    trial_indices=tuple(int(trial_index_value) for trial_index_value in selected_trial_indices),
                    alignment_event=alignment_event,
                    window_start_s=float(window[0]),
                    window_end_s=float(window[1]),
                    bin_size_s=float(pca_bin_size_s),
                    normalization=pca_normalization,
                    component_count=int(pca_component_count),
                    _spike_group=spike_group,
                    _trial_df=trial_df,
                )
                figure, axes = unit_spike_plotting.plot_trial_behavior_and_population_pca(
                    trial_df=trial_df,
                    trial_index=trial_index,
                    lick_times=lick_times,
                    pca_time_s=pca_time_s,
                    pca_scores=pca_result.scores,
                    trial_position=int(trial_position),
                    alignment_event=alignment_event,
                    window=window,
                    pc_count=int(pca_component_count),
                    lfp_time_s=lfp_time_s,
                    lfp_uv=lfp_uv,
                    lfp_label=lfp_label,
                    lfp_y_label=lfp_y_label,
                    figure_size=raster_layout["figure_size"],
                )
                variance_figure, _ = unit_spike_plotting.plot_pca_cumulative_explained_variance(
                    cumulative_explained_variance=pca_result.cumulative_explained_variance,
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
            if neural_display == NEURAL_DISPLAY_POPULATION_PCA:
                st.write(f"PCA units: {pca_unit_ids.size}")
                st.write(f"PCA bin size: {float(pca_bin_size_s):g} s")
                st.write(f"PCA normalization: {pca_normalization}")
                if pca_result is not None:
                    st.write(f"PCs shown: {pca_result.scores.shape[2]}")
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
