"""Unit-selection presentation and trial-view publication helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import unit_spike_plotting
from src.neural_analysis.spike_behavior import loading as unit_spike_loading


PAGE_SIZE_OPTIONS = [25, 50, 100]

PSTH_BIN_OPTIONS = [0.05, 0.1]

UNIT_PLOT_TYPE_OPTIONS = [
    "PSTH",
    "Binned rate: trials + mean",
    "Binned rate: mean +/- SD",
]

POPULATION_PSTH_UNIT_SCOPE_OPTIONS = ["Visible page units", "All selected units"]

RASTER_LAYOUT_OPTIONS = {
    "Compact": {"row_spacing": 1.0, "figure_size": (12.0, 7.0)},
    "Separated": {"row_spacing": 1.5, "figure_size": (12.0, 10.0)},
    "Wide": {"row_spacing": 2.0, "figure_size": (12.0, 13.0)},
}

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
