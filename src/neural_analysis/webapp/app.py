from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import (
    lfp_phase_clustering,
    lfp_summary_webapp,
    population_pca_decoding,
    population_pca_switch_trajectories,
    unit_spike_plotting,
)
from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp import spectrogram as lfp_spectrogram
from src.neural_analysis.lfp.loading import (
    OPEN_EPHYS_AFFINE_UV_SEMANTICS,
    sha256_file_content,
)
from src.neural_analysis.population import pca as population_pca
from src.neural_analysis.spike_behavior import loading as spike_behavior_pynapple
from src.neural_analysis.spike_behavior import loading as unit_spike_loading
from src.neural_analysis.spike_lfp import hilbert as spike_lfp_hilbert_phase
from src.neural_analysis.spike_lfp import phase_locking as spike_lfp_phase_locking
from src.neural_analysis.lfp_summary.session import (
    LFPSummarySessionRequest,
    build_lfp_summary_config,
)
from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
)


from src.neural_analysis.webapp.data_loading import (
    OpenEphysCacheToken,
    build_open_ephys_cache_token,
    decode_lfp_sync_cached,
    load_channel_quality_cached,
    load_cluster_info_cached,
    load_metadata_behavior_session_cached,
    load_metadata_viewer_data_cached,
    load_phase_clustering_session_cached,
    load_summary_channel_metadata,
    load_summary_cluster_metadata,
    load_trial_lfp_trace_cached,
    load_trial_lfp_trace_for_format,
    load_trial_lfp_trace_for_format_with_sample_rate,
    load_viewer_data_cached,
)
from src.neural_analysis.webapp.lfp_views import (
    EVENT_MARKER_STYLES,
    LFP_FILTER_BANDS,
    LFP_FILTER_PADDING_S,
    LFP_PHASE_CLUSTERING_ANALYSIS_OPTIONS,
    LFP_PHASE_CLUSTERING_DEFAULT_WINDOW,
    LFP_PHASE_CLUSTERING_FREQUENCY_COUNT,
    LFP_PHASE_CLUSTERING_MAXIMUM_CORE_DURATION_S,
    LFP_PHASE_CLUSTERING_MAX_FREQUENCY_HZ,
    LFP_PHASE_CLUSTERING_MINIMUM_RELATIVE_MAGNITUDE,
    LFP_PHASE_CLUSTERING_MIN_FREQUENCY_HZ,
    LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ,
    LFP_SPECTROGRAM_COLOR_PERCENTILES,
    LFP_SPECTROGRAM_FREQUENCY_COUNT,
    LFP_SPECTROGRAM_GAUSSIAN_WIDTH,
    LFP_SPECTROGRAM_MAX_FREQUENCY_HZ,
    LFP_SPECTROGRAM_MIN_FREQUENCY_HZ,
    LFP_SPECTROGRAM_NOTCH_DEFAULT,
    LFP_SPECTROGRAM_NOTCH_QUALITY_FACTOR,
    LFP_SPECTROGRAM_PRECISION,
    LFP_SPECTROGRAM_REFERENCE_TRIAL_COUNT,
    LFP_SPECTROGRAM_TARGET_SAMPLE_RATE_HZ,
    LFP_SPECTROGRAM_WINDOW_LENGTH,
    RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS,
    RELATIVE_PHASE_CACHE_MAX_ENTRIES,
    RELATIVE_PHASE_DEFAULT_DISPLAY,
    RELATIVE_PHASE_DEFAULT_WINDOW,
    RELATIVE_PHASE_DISPLAY_OPTIONS,
    RELATIVE_PHASE_FREQUENCY_COUNT,
    RELATIVE_PHASE_MAX_FREQUENCY_HZ,
    RELATIVE_PHASE_MIN_FREQUENCY_HZ,
    RELATIVE_PHASE_OUTPUT_SAMPLE_RATE_HZ,
    WITHIN_TRIAL_PLV_DEFAULT_MAX_WINDOW_S,
    WITHIN_TRIAL_PLV_DEFAULT_MIN_VALID_FRACTION,
    WITHIN_TRIAL_PLV_DEFAULT_MIN_WINDOW_S,
    WITHIN_TRIAL_PLV_DEFAULT_WINDOW_CYCLES,
    WITHIN_TRIAL_PLV_MAX_WINDOW_DEFAULT_ENABLED,
    WITHIN_TRIAL_PLV_MIN_WINDOW_DEFAULT_ENABLED,
    build_lfp_dropdown_options,
    compute_lfp_phase_site_cached,
    compute_shared_lfp_power_limits_cached,
    compute_single_trial_relative_phase_cached,
    compute_trial_lfp_spectrogram_cached,
    render_lfp_phase_clustering_view,
    render_single_trial_relative_phase_view,
    resolve_lfp_filter_band,
    resolve_lfp_view_sites,
)
from src.neural_analysis.webapp.population_views import (
    CONCATENATED_PCA_FIGURE_SIZE,
    CONCATENATED_PCA_VISIBLE_TRIAL_OPTIONS,
    DEFAULT_CONCATENATED_PCA_COMPONENT_COUNT,
    DEFAULT_CONCATENATED_PCA_VISIBLE_TRIAL_COUNT,
    DEFAULT_PCA_COMPONENT_COUNT,
    DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT,
    DEFAULT_PCA_VARIANCE_COMPONENT_COUNT,
    NEURAL_DISPLAY_LFP_SPECTROGRAM,
    NEURAL_DISPLAY_OPTIONS,
    NEURAL_DISPLAY_POPULATION_PCA,
    NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED,
    NEURAL_DISPLAY_SPIKE_RASTER,
    PCA_BIN_SIZE_OPTIONS,
    PCA_DECODING_DEFAULT_COMPONENT_COUNT,
    PCA_DECODING_DEFAULT_CV_FOLDS,
    PCA_DECODING_DEFAULT_PERMUTATIONS,
    PCA_DECODING_DISPLAY_AVERAGE_PC,
    PCA_DECODING_DISPLAY_OPTIONS,
    PCA_DECODING_DISPLAY_PERFORMANCE,
    PCA_DECODING_SHOW_RAW_PC_SCORES_DEFAULT,
    compute_population_pca_cached,
    compute_population_pca_decoding_cached,
    compute_population_pca_switch_trajectories_cached,
    filter_trial_indices_for_valid_alignment,
    is_population_pca_display,
    resolve_pca_decoding_component_minimum,
    resolve_population_pca_unit_ids,
    select_concatenated_trial_viewport,
    select_visible_concatenated_trial_indices,
)
from src.neural_analysis.webapp.session_inputs import (
    ACTION_OPTIONS,
    ALIGNMENT_OPTIONS,
    CHANNEL_SOURCE_CHANNEL_QUALITY,
    CHANNEL_SOURCE_MANUAL,
    CHANNEL_SOURCE_OPTIONS,
    COMPARE_LEFT_RIGHT_ACTION,
    CONDITION_OPTIONS,
    LFP_DROPDOWN_LABEL_HPC_V1,
    LFP_DROPDOWN_LABEL_PFC,
    LFP_FORMAT_OPEN_EPHYS_DERIVED,
    LFP_FORMAT_OPTIONS,
    LFP_FORMAT_SPIKEGLX,
    MetadataLFPSiteInputs,
    MetadataViewAvailability,
    PLOT_VIEW_LFP_PHASE_CLUSTERING,
    PLOT_VIEW_LFP_SUMMARY,
    PLOT_VIEW_OPTIONS,
    PLOT_VIEW_PCA_DECODING,
    PLOT_VIEW_PCA_SWITCH_TRAJECTORIES,
    PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE,
    PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT,
    PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING,
    PLOT_VIEW_TRIAL_SPIKES,
    PLOT_VIEW_UNIT_RASTER,
    load_webapp_session,
    metadata_lfp_site_inputs,
    metadata_view_availability,
    resolve_region_channels_for_source,
)
from src.neural_analysis.webapp.spike_lfp_views import (
    SPIKE_LFP_HILBERT_CACHE_MAX_ENTRIES,
    SPIKE_LFP_HILBERT_DEFAULT_WINDOW,
    SPIKE_LFP_HILBERT_FILTER_PADDING_S,
    SPIKE_LFP_HILBERT_MINIMUM_ENVELOPE,
    SPIKE_LFP_HILBERT_PHASE_BAND_HZ,
    SPIKE_LFP_PHASE_AMPLITUDE_MASK_OPTIONS,
    SPIKE_LFP_PHASE_BIN_COUNT,
    SPIKE_LFP_PHASE_CACHE_MAX_ENTRIES,
    SPIKE_LFP_PHASE_DEFAULT_POLAR_FREQUENCY_HZ,
    SPIKE_LFP_PHASE_DEFAULT_WINDOW,
    SPIKE_LFP_PHASE_FREQUENCY_COUNT,
    SPIKE_LFP_PHASE_MAXIMUM_CORE_DURATION_S,
    SPIKE_LFP_PHASE_MAX_FREQUENCY_HZ,
    SPIKE_LFP_PHASE_MIN_FREQUENCY_HZ,
    compute_single_trial_spike_lfp_hilbert_cached,
    compute_spike_lfp_phase_locking_cached,
    render_single_trial_spike_lfp_hilbert_view,
    render_spike_lfp_phase_locking_view,
)
from src.neural_analysis.webapp.summary_view import (
    render_lfp_summary_view,
    render_metadata_lfp_summary_view,
)
from src.neural_analysis.webapp.unit_views import (
    PAGE_SIZE_OPTIONS,
    POPULATION_PSTH_UNIT_SCOPE_OPTIONS,
    PSTH_BIN_OPTIONS,
    RASTER_LAYOUT_OPTIONS,
    UNIT_PLOT_TYPE_OPTIONS,
    _select_unit_metadata,
    build_trial_view_plot_save_path,
    format_metadata_row_for_display,
)


@dataclass(frozen=True)
class MetadataPopulationInputs:
    """Resolved paths and zero-based channels for one metadata population."""

    population_id: str
    population_label: str
    probe_id: str
    probe_label: str
    channel_group_label: str
    channel_indices: tuple[int, ...]
    sorter_directory: Path | None
    aligned_spike_file: Path | None
    lfp_file: Path | None
    channel_quality_file: Path | None


def metadata_population_inputs(
    session: ResolvedSession,
    population_id: str,
) -> MetadataPopulationInputs:
    """Resolve one metadata population to its probe sources and anatomy.

    Parameters
    ----------
    session : ResolvedSession
        One metadata-resolved session.
    population_id : str
        Exact stable population identifier.

    Returns
    -------
    MetadataPopulationInputs
        Paths, labels, and zero-based channel indices. No unit IDs or spike
        arrays are derived.
    """
    population = next(
        (item for item in session.populations if item.population_id == population_id),
        None,
    )
    if population is None:
        raise ValueError(f"unknown population ID: {population_id}")
    group = next(
        item
        for item in session.channel_groups
        if item.channel_group_id == population.channel_group_id
    )
    probe = resolve_probe_sources(session, population.probe_id)
    return MetadataPopulationInputs(
        population_id=population.population_id,
        population_label=population.display_label,
        probe_id=probe.probe_id,
        probe_label=probe.display_label,
        channel_group_label=group.display_label,
        channel_indices=group.channel_indices,
        sorter_directory=probe.sorter_directory,
        aligned_spike_file=probe.aligned_spike_file,
        lfp_file=probe.lfp_file,
        channel_quality_file=probe.channel_quality_file,
    )


DEFAULT_BROWSER_ROOT = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026")


def parse_webapp_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the optional metadata path passed after Streamlit's ``--``.

    Parameters
    ----------
    argv : sequence of str or None
        Script arguments. ``None`` uses ``argparse``'s process arguments.

    Returns
    -------
    argparse.Namespace
        Namespace whose ``session_metadata`` value is a ``Path`` or ``None``.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session-metadata", type=Path)
    return parser.parse_args(argv)


def _build_channel_text(region_name: str) -> str:
    """Return editable default channel text for one preset region."""

    presets = unit_spike_loading.get_ct014_region_channel_presets()
    return unit_spike_loading.format_channel_list(presets.get(region_name, np.array([], dtype=int)))


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


def _legacy_population_controls(
    hpc_sorter_path: str,
    hpc_aligned_path: str,
    hpc_lfp_path: str,
    pfc_sorter_path: str,
    pfc_aligned_path: str,
    pfc_lfp_path: str,
) -> tuple[str, str, str, str, str, np.ndarray, str]:
    """Render the existing manual region controls and return their selection.

    All paths are user-entered filesystem strings. The returned channel array
    is one-dimensional, integer, and zero based. This helper preserves the
    legacy route while the metadata route supplies the same values directly.
    """
    st.sidebar.header("Region and Units")
    preset_options = list(unit_spike_loading.get_ct014_region_channel_presets().keys())
    region_name = st.sidebar.selectbox("Region preset", options=preset_options, index=0)
    custom_probe_label = None
    if region_name == "Custom":
        custom_probe_label = st.sidebar.selectbox(
            "Custom probe source",
            options=list(unit_spike_loading.SUPPORTED_PROBE_LABELS),
        )
    active_probe_label = unit_spike_loading.get_probe_label_for_region(
        region_name,
        custom_probe_label=custom_probe_label,
    )
    st.sidebar.caption(f"Active probe: {active_probe_label}")
    if active_probe_label == unit_spike_loading.PROBE_LABEL_HPC_V1:
        active_sorter_output_path = hpc_sorter_path
        active_aligned_spike_path = hpc_aligned_path
        active_lfp_path = hpc_lfp_path
    else:
        active_sorter_output_path = pfc_sorter_path
        active_aligned_spike_path = pfc_aligned_path
        active_lfp_path = pfc_lfp_path

    inferred_probe_derived_dir = unit_spike_loading.infer_probe_derived_dir(
        sorter_output_path=active_sorter_output_path,
        lfp_path=active_lfp_path,
    )
    channel_quality_path = None
    if inferred_probe_derived_dir is not None:
        try:
            channel_quality_path = unit_spike_loading.resolve_channel_quality_path(
                inferred_probe_derived_dir
            )
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
        channel_quality = load_channel_quality_cached(str(channel_quality_path))
        st.sidebar.caption(f"Channel quality: {channel_quality_path}")
        available_labels = sorted(
            channel_quality["label"].astype(str).str.strip().str.lower().unique().tolist()
        )
        default_labels = ["good"] if "good" in available_labels else available_labels
        channel_quality_labels = tuple(
            st.sidebar.multiselect(
                "Channel labels",
                options=available_labels,
                default=default_labels,
            )
        )
        require_inside_brain = st.sidebar.checkbox("Inside brain only", value=True)
        disagreement_count = int(
            (
                channel_quality["label"].astype(str).str.strip().str.lower().eq("good")
                != channel_quality["is_good"].astype(bool)
            ).sum()
        )
        if disagreement_count:
            st.sidebar.warning(
                f"{disagreement_count} channels disagree between label == 'good' and is_good."
            )
        manual_channel_text = _build_channel_text(region_name)
    else:
        manual_channel_text = st.sidebar.text_area(
            "Region channels",
            value=_build_channel_text(region_name),
            key=f"channel_text_{region_name}",
            height=140,
        )
    region_channels, channel_summary = resolve_region_channels_for_source(
        channel_source=channel_source,
        manual_channel_text=manual_channel_text,
        channel_quality=channel_quality,
        require_inside_brain=require_inside_brain,
        channel_quality_labels=channel_quality_labels,
    )
    st.sidebar.caption(channel_summary)
    return (
        region_name,
        active_probe_label,
        active_sorter_output_path,
        active_aligned_spike_path,
        active_lfp_path,
        region_channels,
        channel_source,
    )


def _metadata_population_controls(
    session: ResolvedSession,
) -> tuple[str, str, str, str, str, np.ndarray, str]:
    """Render the established scientific controls using metadata-defined paths.

    Parameters
    ----------
    session : ResolvedSession
        One resolved session. Probe paths are absolute filesystem paths;
        ``unit_channels`` values, when present, are zero-based channel ids.

    Returns
    -------
    tuple[str, str, str, str, str, numpy.ndarray, str]
        Region label, probe id, sorter path, alignment path, LFP path,
        one-dimensional zero-based channel ids, and channel-source label.
        This function reads only the selected probe's small quality table.
    """
    if not session.probes:
        raise ValueError("No probes are configured in session metadata.")

    st.sidebar.header("Region and Units")
    probe_ids = tuple(probe.probe_id for probe in session.probes)
    probe_id = st.sidebar.selectbox("Probe / region", options=probe_ids, index=0)
    probe = resolve_probe_sources(session, probe_id)
    st.sidebar.caption(f"Active probe: {probe.probe_id}")

    quality_path = probe.channel_quality_file
    channel_source_options = [CHANNEL_SOURCE_MANUAL]
    if quality_path is not None and quality_path.is_file():
        channel_source_options.append(CHANNEL_SOURCE_CHANNEL_QUALITY)
    channel_source = st.sidebar.selectbox(
        "Channel source",
        options=channel_source_options,
        index=1 if CHANNEL_SOURCE_CHANNEL_QUALITY in channel_source_options else 0,
        help="Use channel_quality when available to select good in-brain probe sites.",
    )

    channel_quality = None
    channel_quality_labels: tuple[str, ...] = ("good",)
    require_inside_brain = True
    manual_defaults = probe.unit_channels
    if manual_defaults is None:
        manual_defaults = tuple(site.saved_channel_index for site in probe.sites)
    manual_channel_text = unit_spike_loading.format_channel_list(manual_defaults)

    if channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY:
        channel_quality = load_channel_quality_cached(str(quality_path))
        st.sidebar.caption(f"Channel quality: {quality_path}")
        available_labels = sorted(
            channel_quality["label"].astype(str).str.strip().str.lower().unique().tolist()
        )
        default_labels = ["good"] if "good" in available_labels else available_labels
        channel_quality_labels = tuple(
            st.sidebar.multiselect(
                "Channel labels",
                options=available_labels,
                default=default_labels,
            )
        )
        require_inside_brain = st.sidebar.checkbox("Inside brain only", value=True)
        disagreement_count = int(
            (
                channel_quality["label"].astype(str).str.strip().str.lower().eq("good")
                != channel_quality["is_good"].astype(bool)
            ).sum()
        )
        if disagreement_count:
            st.sidebar.warning(
                f"{disagreement_count} channels disagree between label == 'good' and is_good."
            )
    else:
        manual_channel_text = st.sidebar.text_area(
            "Region channels",
            value=manual_channel_text,
            key=f"channel_text_{probe.probe_id}",
            height=140,
        )

    region_channels, channel_summary = resolve_region_channels_for_source(
        channel_source=channel_source,
        manual_channel_text=manual_channel_text,
        channel_quality=channel_quality,
        require_inside_brain=require_inside_brain,
        channel_quality_labels=channel_quality_labels,
    )
    if channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY and probe.unit_channels is not None:
        region_channels = np.intersect1d(
            region_channels,
            np.asarray(probe.unit_channels, dtype=int),
        )
        channel_summary += f" Metadata restriction kept {region_channels.size} channels."
    st.sidebar.caption(channel_summary)

    return (
        probe.probe_id,
        probe.probe_id,
        str(probe.sorter_directory or ""),
        str(probe.aligned_spike_file or ""),
        str(probe.lfp_file or ""),
        region_channels,
        channel_source,
    )


def _start_metadata_webapp(session: ResolvedSession) -> str | None:
    """Render metadata identity/view controls and return an available non-summary view.

    Parameters
    ----------
    session : ResolvedSession
        One resolved metadata session. No scientific array has been opened.

    Returns
    -------
    str or None
        Selected non-summary view, or ``None`` after rendering a summary view
        or explaining why the selected view is unavailable.
    """
    st.sidebar.caption(f"Session: {session.subject_id} / {session.session_id}")
    plot_view = st.sidebar.selectbox("Plot view", options=PLOT_VIEW_OPTIONS)
    availability = metadata_view_availability(session)[plot_view]
    if not availability.available:
        st.warning(availability.reason)
        return None
    if plot_view == PLOT_VIEW_LFP_SUMMARY:
        render_metadata_lfp_summary_view(st, session)
        return None
    return plot_view


def main(argv: Sequence[str] = ()) -> None:
    """
    Run the local Streamlit unit raster/PSTH browser.

    Parameters
    ----------
    argv : sequence of str
        Optional script arguments after Streamlit's ``--``. An explicit
        ``--session-metadata`` selects the metadata-driven route; an empty
        sequence preserves the existing direct/manual route.

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

    arguments = parse_webapp_arguments(argv)
    metadata_session: ResolvedSession | None = None
    metadata_lfp_sites: tuple[MetadataLFPSiteInputs, ...] = ()
    hpc_v1_sorter_output_path = ""
    hpc_v1_aligned_spike_path = ""
    hpc_v1_lfp_path = ""
    pfc_sorter_output_path = ""
    pfc_aligned_spike_path = ""
    pfc_lfp_path = ""
    if arguments.session_metadata is not None:
        try:
            metadata_session = load_webapp_session(arguments.session_metadata)
        except (OSError, ValueError) as error:
            st.error(f"Could not load session metadata: {error}")
            return
        selected_metadata_view = _start_metadata_webapp(metadata_session)
        if selected_metadata_view is None:
            return
        plot_view = selected_metadata_view
        session_data_home = str(metadata_session.session_root)
        sess_id_full = metadata_session.session_id
        metadata_lfp_sites = metadata_lfp_site_inputs(metadata_session)
    else:
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

    if plot_view == PLOT_VIEW_LFP_SUMMARY:
        render_lfp_summary_view(
            session_data_home=session_data_home,
            sess_id_full=sess_id_full,
            hpc_v1_lfp_path=hpc_v1_lfp_path,
            pfc_lfp_path=pfc_lfp_path,
            hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
            pfc_aligned_spike_path=pfc_aligned_spike_path,
            hpc_v1_sorter_output_path=hpc_v1_sorter_output_path,
            pfc_sorter_output_path=pfc_sorter_output_path,
        )
        return

    if plot_view in {PLOT_VIEW_LFP_PHASE_CLUSTERING, PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE}:
        try:
            if metadata_session is not None:
                phase_session, phase_event_df, phase_trial_df = (
                    load_metadata_behavior_session_cached(
                        str(metadata_session.session_root),
                        metadata_session.subject_id,
                        metadata_session.session_id,
                        metadata_session.session_date,
                        str(metadata_session.behavior.session_directory),
                        str(metadata_session.behavior.trial_table_file),
                        (
                            str(metadata_session.behavior.event_table_file)
                            if metadata_session.behavior.event_table_file is not None
                            else None
                        ),
                    )
                )
            else:
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
                metadata_lfp_sites=metadata_lfp_sites,
                hpc_v1_lfp_path=hpc_v1_lfp_path,
                pfc_lfp_path=pfc_lfp_path,
                hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
                pfc_aligned_spike_path=pfc_aligned_spike_path,
            )
        else:
            render_lfp_phase_clustering_view(
                trial_df=phase_trial_df,
                session=phase_session,
                metadata_lfp_sites=metadata_lfp_sites,
                hpc_v1_lfp_path=hpc_v1_lfp_path,
                pfc_lfp_path=pfc_lfp_path,
                hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
                pfc_aligned_spike_path=pfc_aligned_spike_path,
            )
        return

    try:
        if metadata_session is not None:
            (
                region_name,
                active_probe_label,
                active_sorter_output_path,
                active_aligned_spike_path,
                active_lfp_path,
                region_channels,
                channel_source,
            ) = _metadata_population_controls(metadata_session)
        else:
            (
                region_name,
                active_probe_label,
                active_sorter_output_path,
                active_aligned_spike_path,
                active_lfp_path,
                region_channels,
                channel_source,
            ) = _legacy_population_controls(
                hpc_v1_sorter_output_path,
                hpc_v1_aligned_spike_path,
                hpc_v1_lfp_path,
                pfc_sorter_output_path,
                pfc_aligned_spike_path,
                pfc_lfp_path,
            )
    except (OSError, ValueError) as error:
        st.error(str(error))
        st.stop()
    if region_channels.size == 0:
        st.warning("No region channels are selected.")
        st.stop()

    try:
        if metadata_session is not None:
            viewer_data = load_metadata_viewer_data_cached(
                str(metadata_session.session_root),
                metadata_session.subject_id,
                metadata_session.session_id,
                metadata_session.session_date,
                str(metadata_session.behavior.session_directory),
                str(metadata_session.behavior.trial_table_file),
                (
                    str(metadata_session.behavior.event_table_file)
                    if metadata_session.behavior.event_table_file is not None
                    else None
                ),
                active_probe_label,
                active_sorter_output_path,
                active_aligned_spike_path,
                active_lfp_path or None,
            )
        else:
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
            metadata_lfp_sites=metadata_lfp_sites,
            hpc_v1_lfp_path=hpc_v1_lfp_path,
            pfc_lfp_path=pfc_lfp_path,
            hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
            pfc_aligned_spike_path=pfc_aligned_spike_path,
        )
        return
    if plot_view == PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT:
        render_single_trial_spike_lfp_hilbert_view(
            trial_df=trial_df,
            event_df=event_df,
            session=session,
            spike_group=spike_group,
            unit_ids=unit_ids,
            active_probe_label=active_probe_label,
            metadata_lfp_sites=metadata_lfp_sites,
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
            lfp_view_sites = resolve_lfp_view_sites(
                metadata_sites=metadata_lfp_sites,
                hpc_v1_lfp_path=hpc_v1_lfp_path,
                pfc_lfp_path=pfc_lfp_path,
                hpc_v1_aligned_spike_path=hpc_v1_aligned_spike_path,
                pfc_aligned_spike_path=pfc_aligned_spike_path,
            )
            lfp_view_sites_by_label = {
                site.display_label: site for site in lfp_view_sites
            }
            lfp_dropdown_options = list(lfp_view_sites_by_label)
            default_lfp_index = next(
                (
                    index
                    for index, site in enumerate(lfp_view_sites)
                    if site.probe_id == active_probe_label
                ),
                (
                    0
                    if active_probe_label == unit_spike_loading.PROBE_LABEL_HPC_V1
                    else min(1, len(lfp_view_sites) - 1)
                ),
            )
            selected_lfp_label = st.sidebar.selectbox(
                "Active LFP file",
                options=lfp_dropdown_options,
                index=default_lfp_index,
            )
            selected_lfp_site = lfp_view_sites_by_label[selected_lfp_label]
            selected_lfp_path = str(selected_lfp_site.lfp_file or "")
            default_aligned_sync_path = str(
                selected_lfp_site.synchronization_file or ""
            )
            selected_aligned_sync_path = default_aligned_sync_path
            if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
                selected_aligned_sync_path = st.sidebar.text_input(
                    "Open Ephys aligned sync path",
                    value=str(default_aligned_sync_path),
                    help="Use the probe sync .npz produced by the Open Ephys spike synchronization workflow.",
                )
                lfp_y_label = "LFP (uV)"
                lfp_power_unit_label = "dB re 1 uV^2"
            st.sidebar.caption(selected_lfp_path or "No LFP path entered for this selection.")
            if metadata_lfp_sites:
                lfp_saved_channel_index = st.sidebar.number_input(
                    "LFP saved channel index",
                    min_value=0,
                    value=int(selected_lfp_site.saved_channel_index),
                    step=1,
                )
            elif channel_source == CHANNEL_SOURCE_CHANNEL_QUALITY and region_channels.size > 0:
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
                    # Build one complete identity at this selected-source boundary.
                    # The same instance keys the displayed trial and every nested
                    # reference-trial spectrogram calculation below.
                    selected_open_ephys_cache_token = (
                        build_open_ephys_cache_token(
                            selected_lfp_path,
                            aligned_sync_path_argument,
                        )
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
                            open_ephys_cache_token=selected_open_ephys_cache_token,
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
                            open_ephys_cache_token=selected_open_ephys_cache_token,
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
                            open_ephys_cache_token=selected_open_ephys_cache_token,
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
    main(sys.argv[1:])
