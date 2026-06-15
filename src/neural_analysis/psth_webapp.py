from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.neural_analysis import unit_spike_loading, unit_spike_plotting


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
}
ALIGNMENT_OPTIONS = ["choice_time", "start_time"]
PAGE_SIZE_OPTIONS = [25, 50, 100]
PSTH_BIN_OPTIONS = [0.05, 0.1]


@st.cache_resource(show_spinner="Loading session data...")
def load_viewer_data_cached(
    session_data_home: str,
    sess_id_full: str,
    sorter_output_path: str,
    aligned_spike_path: str,
):
    """
    Load one session for Streamlit with cache persistence until app restart.

    Parameters
    ----------
    session_data_home : str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    sorter_output_path : str
        Sorter output directory path.
    aligned_spike_path : str
        Aligned spike ``.npz`` path.

    Returns
    -------
    dict
        Loaded viewer data from ``unit_spike_loading.load_viewer_data``.
    """

    return unit_spike_loading.load_viewer_data(
        session_data_home=Path(session_data_home),
        sess_id_full=sess_id_full,
        sorter_output_path=Path(sorter_output_path),
        aligned_spike_path=Path(aligned_spike_path),
    )


def _build_channel_text(region_name: str) -> str:
    """Return editable default channel text for one preset region."""

    presets = unit_spike_loading.get_ct014_region_channel_presets()
    return unit_spike_loading.format_channel_list(presets.get(region_name, np.array([], dtype=int)))


def _select_unit_metadata(cluster_info, region_channels):
    """Render unit-quality controls and return filtered unit metadata."""

    quality_labels = unit_spike_loading.normalize_quality_labels(cluster_info["group"])
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

    if st.sidebar.button("Clear cached session data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.sidebar.header("Session")
    session_data_home = st.sidebar.text_input(
        "Session data home",
        value=str(unit_spike_loading.DEFAULT_SESSION_DATA_HOME),
    )
    sess_id_full = st.sidebar.text_input(
        "Session id",
        value=unit_spike_loading.DEFAULT_SESSION_ID,
    )
    sorter_output_path = st.sidebar.text_input(
        "Sorter output path",
        value=str(unit_spike_loading.DEFAULT_HPC_SORTER_OUTPUT_PATH),
    )
    aligned_spike_path = st.sidebar.text_input(
        "Aligned spike path",
        value=str(unit_spike_loading.DEFAULT_ALIGNED_SPIKE_PATH),
    )

    try:
        viewer_data = load_viewer_data_cached(
            session_data_home=session_data_home,
            sess_id_full=sess_id_full,
            sorter_output_path=sorter_output_path,
            aligned_spike_path=aligned_spike_path,
        )
    except Exception as error:  # noqa: BLE001 - Streamlit should display load failures without a traceback wall.
        st.error(f"Could not load session data: {error}")
        st.stop()

    session = viewer_data["session"]
    trial_df = viewer_data["trial_df"]
    cluster_info = viewer_data["cluster_info"]
    spike_group = viewer_data["spike_group"]

    st.sidebar.header("Region and Units")
    preset_options = list(unit_spike_loading.get_ct014_region_channel_presets().keys())
    region_name = st.sidebar.selectbox("Region preset", options=preset_options, index=0)
    channel_text = st.sidebar.text_area(
        "Region channels",
        value=_build_channel_text(region_name),
        key=f"channel_text_{region_name}",
        height=140,
        help="CT014 presets are editable defaults. Replace with custom channel ids when needed.",
    )
    try:
        region_channels = unit_spike_loading.parse_channel_list(channel_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    if region_channels.size == 0:
        st.warning("No region channels are selected.")
        st.stop()

    selected_unit_metadata = _select_unit_metadata(cluster_info, region_channels)
    if selected_unit_metadata.empty:
        st.warning("No units match the selected channels and quality filters.")
        st.stop()

    unit_ids = selected_unit_metadata["cluster_id"].to_numpy(dtype=int)
    unit_id = st.sidebar.selectbox("Cluster id", options=unit_ids.tolist())
    selected_unit_row = selected_unit_metadata.loc[selected_unit_metadata["cluster_id"] == unit_id].iloc[0]

    st.sidebar.header("Trials")
    condition = st.sidebar.selectbox("Condition", options=CONDITION_OPTIONS)
    action_label = st.sidebar.selectbox("Action", options=list(ACTION_OPTIONS.keys()))
    alignment_event = st.sidebar.selectbox("Alignment event", options=ALIGNMENT_OPTIONS)
    page_size = st.sidebar.selectbox("Raster page size", options=PAGE_SIZE_OPTIONS, index=1)
    bin_size = st.sidebar.selectbox("PSTH bin size (s)", options=PSTH_BIN_OPTIONS, index=1)
    window_start = st.sidebar.number_input("Window start (s)", value=-2.0, step=0.1)
    window_end = st.sidebar.number_input("Window end (s)", value=2.0, step=0.1)
    if float(window_start) >= float(window_end):
        st.error("Window start must be less than window end.")
        st.stop()
    window = (float(window_start), float(window_end))

    selected_trial_indices = unit_spike_plotting.filter_trials_for_unit_plot(
        trial_df,
        condition=condition,
        action=ACTION_OPTIONS[action_label],
    )
    if selected_trial_indices.size == 0:
        st.warning("No trials match the selected filters.")
        st.stop()

    n_pages = max(1, math.ceil(selected_trial_indices.size / int(page_size)))
    page_index = st.sidebar.number_input(
        "Raster page",
        min_value=0,
        max_value=n_pages - 1,
        value=0,
        step=1,
    )
    raster_trial_indices = unit_spike_plotting.paginate_trial_indices(
        selected_trial_indices,
        page_index=int(page_index),
        page_size=int(page_size),
    )

    unit_spike_times = unit_spike_loading.get_unit_spike_times(spike_group, unit_id=int(unit_id))
    title_suffix = (
        f"{region_name}; {condition}; {action_label}; "
        f"page {int(page_index) + 1}/{n_pages}; PSTH n={selected_trial_indices.size}"
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
        title_suffix=title_suffix,
    )

    metadata_column, plot_column = st.columns([1, 3])
    with metadata_column:
        st.subheader("Unit Metadata")
        st.dataframe(selected_unit_row.to_frame(name="value"), use_container_width=True)
        st.write(f"Filtered units: {selected_unit_metadata.shape[0]}")
        st.write(f"Filtered trials: {selected_trial_indices.size}")
        st.write(f"Raster trials on page: {raster_trial_indices.size}")
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
        )
        st.success(f"Saved plot to {save_path}")

    plt.close(figure)


if __name__ == "__main__":
    main()
