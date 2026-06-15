from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.spike_behavior_pynapple import make_trial_type_masks


def filter_trials_for_unit_plot(
    trial_df: pd.DataFrame,
    condition: str = "all",
    action: int | str | None = "all",
) -> np.ndarray:
    """
    Select trial indices for unit raster/PSTH plotting.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape ``(n_trials, n_columns)``. Condition filters use
        columns required by ``make_trial_type_masks``: ``give_reward``,
        ``correct``, ``reward``, and ``action``.
    condition : str, default="all"
        Trial condition. Supported values are ``"all"`` and keys returned by
        ``make_trial_type_masks`` such as ``"correct_rewarded"``, ``"incorrect"``,
        ``"omission"``, ``"switch"``, and ``"stay"``.
    action : int | str | None, default="all"
        Optional action selector. ``"all"`` or ``None`` disables action
        filtering; otherwise values are compared numerically to ``trial_df["action"]``.

    Returns
    -------
    np.ndarray
        Chronologically ordered trial indices with shape ``(n_selected_trials,)``.
    """

    if condition == "all":
        selected_mask = pd.Series(True, index=trial_df.index)
    else:
        trial_masks = make_trial_type_masks(trial_df)
        if condition not in trial_masks:
            raise ValueError(f"Unknown trial condition {condition!r}.")
        selected_mask = trial_masks[condition].copy()

    if action is not None and str(action).lower() != "all":
        action_value = int(action)
        selected_mask = selected_mask & pd.to_numeric(trial_df["action"], errors="coerce").eq(action_value)

    return trial_df.index[np.asarray(selected_mask, dtype=bool)].to_numpy(dtype=int)


def paginate_trial_indices(trial_indices: np.ndarray, page_index: int, page_size: int) -> np.ndarray:
    """
    Return one chronological page of trial indices.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``.
    page_index : int
        Zero-based page number.
    page_size : int
        Number of trials per page.

    Returns
    -------
    np.ndarray
        Trial indices for the requested page.
    """

    if page_size <= 0:
        raise ValueError("page_size must be positive.")
    if page_index < 0:
        raise ValueError("page_index must be nonnegative.")
    start_index = int(page_index) * int(page_size)
    end_index = start_index + int(page_size)
    return np.asarray(trial_indices, dtype=int)[start_index:end_index]


def extract_relative_unit_spikes(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
) -> list[np.ndarray]:
    """
    Extract spike times relative to an event for selected trials.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table. ``alignment_event`` must name a numeric time column in
        seconds.
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``.
    alignment_event : str
        Event column used as time zero, typically ``"choice_time"`` or
        ``"start_time"``.
    window : tuple[float, float]
        Inclusive lower and upper plotting bounds in seconds relative to the
        alignment event.

    Returns
    -------
    list[np.ndarray]
        One relative spike-time array per requested trial. Each array is
        one-dimensional and in seconds relative to ``alignment_event``.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

    unit_spike_times = np.asarray(unit_spike_times, dtype=float)
    relative_spikes_by_trial: list[np.ndarray] = []
    for trial_index in np.asarray(trial_indices, dtype=int):
        event_time = pd.to_numeric(pd.Series([trial_df.loc[trial_index, alignment_event]]), errors="coerce").iloc[0]
        if pd.isna(event_time):
            relative_spikes_by_trial.append(np.array([], dtype=float))
            continue
        event_time = float(event_time)
        absolute_start = event_time + float(window[0])
        absolute_end = event_time + float(window[1])
        in_window = (unit_spike_times >= absolute_start) & (unit_spike_times <= absolute_end)
        relative_spikes_by_trial.append(unit_spike_times[in_window] - event_time)
    return relative_spikes_by_trial


def compute_psth_hz(
    relative_spikes_by_trial: list[np.ndarray],
    window: tuple[float, float],
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a peri-event time histogram in firing-rate units.

    Parameters
    ----------
    relative_spikes_by_trial : list[np.ndarray]
        One relative spike-time array per trial. Times are in seconds relative
        to the selected event.
    window : tuple[float, float]
        Histogram bounds in seconds relative to the event.
    bin_size : float
        Histogram bin width in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(bin_centers, firing_rate_hz)``. Both arrays have shape ``(n_bins,)``.
        Firing rate is normalized by ``n_trials * bin_size``.
    """

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")

    bin_edges = np.arange(float(window[0]), float(window[1]) + float(bin_size), float(bin_size))
    if bin_edges[-1] < float(window[1]):
        bin_edges = np.append(bin_edges, float(window[1]))
    all_relative_spikes = (
        np.concatenate(relative_spikes_by_trial)
        if len(relative_spikes_by_trial) > 0 and any(spikes.size for spikes in relative_spikes_by_trial)
        else np.array([], dtype=float)
    )
    spike_counts, _ = np.histogram(all_relative_spikes, bins=bin_edges)
    n_trials = max(len(relative_spikes_by_trial), 1)
    firing_rate_hz = spike_counts.astype(float) / (n_trials * float(bin_size))
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    return bin_centers, firing_rate_hz


def plot_unit_raster_and_psth(
    unit_spike_times: np.ndarray,
    trial_df: pd.DataFrame,
    raster_trial_indices: np.ndarray,
    psth_trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size: float,
    unit_id: int,
    title_suffix: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one unit's page-level raster with an all-filtered-trial PSTH.

    Parameters
    ----------
    unit_spike_times : np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    trial_df : pd.DataFrame
        Trial table with event-time columns in seconds.
    raster_trial_indices : np.ndarray
        Trial indices for the displayed raster page, shape ``(n_page_trials,)``.
    psth_trial_indices : np.ndarray
        Trial indices used for the PSTH, shape ``(n_filtered_trials,)``.
    alignment_event : str
        Event column used as time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    bin_size : float
        PSTH bin width in seconds.
    unit_id : int
        Cluster id shown in the plot title.
    title_suffix : str | None, optional
        Optional suffix appended to the title.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes array with shape ``(2,)``. Axis 0 is the
        raster and axis 1 is the PSTH.
    """

    raster_spikes = extract_relative_unit_spikes(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=raster_trial_indices,
        alignment_event=alignment_event,
        window=window,
    )
    psth_spikes = extract_relative_unit_spikes(
        unit_spike_times=unit_spike_times,
        trial_df=trial_df,
        trial_indices=psth_trial_indices,
        alignment_event=alignment_event,
        window=window,
    )
    bin_centers, firing_rate_hz = compute_psth_hz(
        psth_spikes,
        window=window,
        bin_size=bin_size,
    )

    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 7),
        height_ratios=[3, 1],
    )
    raster_axis, psth_axis = axes
    for row_index, relative_spikes in enumerate(raster_spikes):
        if relative_spikes.size == 0:
            continue
        raster_axis.scatter(
            relative_spikes,
            np.full(relative_spikes.shape, row_index, dtype=float),
            marker="|",
            color="black",
            s=80,
        )
    raster_axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    raster_axis.set_ylabel("Trial")
    raster_axis.set_ylim(-0.5, max(len(raster_spikes) - 0.5, 0.5))
    raster_axis.set_xlim(float(window[0]), float(window[1]))
    title = f"Unit {unit_id} aligned to {alignment_event}"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    raster_axis.set_title(title)

    if bin_centers.size > 0:
        psth_axis.bar(
            bin_centers,
            firing_rate_hz,
            width=float(bin_size),
            align="center",
            color="0.35",
            edgecolor="0.35",
        )
    psth_axis.axvline(0.0, color="tab:red", linestyle="--", linewidth=1.2)
    psth_axis.set_ylabel("Firing rate (Hz)")
    psth_axis.set_xlabel(f"Time from {alignment_event} (s)")
    figure.tight_layout()
    return figure, axes


def _sanitize_filename_part(value: str | int | float | None) -> str:
    """Return one filesystem-safe filename token."""

    if value is None:
        return "none"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def save_unit_plot_figure(
    figure: plt.Figure,
    figure_path: Path | str,
    session_id: str,
    unit_id: int,
    region_name: str,
    condition: str,
    action_label: str,
    alignment_event: str,
    page_index: int,
) -> Path:
    """
    Save the current unit raster/PSTH figure in the viewer output folder.

    Parameters
    ----------
    figure : plt.Figure
        Matplotlib figure to save.
    figure_path : Path | str
        Session figure directory.
    session_id : str
        Full session identifier.
    unit_id : int
        Cluster id shown in the figure.
    region_name : str
        Active region label.
    condition : str
        Active condition filter.
    action_label : str
        Active action filter label.
    alignment_event : str
        Alignment event column.
    page_index : int
        Zero-based raster page index.

    Returns
    -------
    Path
        Saved PNG path.
    """

    output_dir = Path(figure_path) / "unit_spike_viewer"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_parts = [
        session_id,
        region_name,
        f"unit{int(unit_id)}",
        condition,
        action_label,
        alignment_event,
        f"page{int(page_index)}",
    ]
    filename = "_".join(_sanitize_filename_part(part) for part in filename_parts) + ".png"
    save_path = output_dir / filename
    figure.savefig(save_path, format="png", dpi=300)
    return save_path
