"""Reusable population PCA, decoding, trajectory, and cross-session plots."""

from __future__ import annotations

from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.population.cross_session import BASE_CONDITIONS
from src.neural_analysis.population.decoding import (
    PCA_DECODING_BASE_CONDITIONS,
    PCA_RAW_SCORE_COLUMNS,
    PCA_SCORE_SUMMARY_COLUMNS,
)
from src.neural_analysis.population.switch_trajectories import (
    SWITCH_DIRECTION_ORDER,
    SWITCH_DIRECTION_TITLES,
    SWITCH_TYPE_ORDER,
    SWITCH_TYPE_TITLES,
)
from src.neural_analysis.spike_behavior.plotting import (
    LEFT_CHOICE_ACTION,
    LEFT_LICK_EVENT,
    LICK_RASTER_STYLES,
    RIGHT_LICK_EVENT,
    extract_relative_events_for_trial,
    get_trial_alignment_time,
)


BEFORE_AFTER_XLIM = (-0.2, 1.2)


def plot_trial_behavior_and_population_pca(
    trial_df: pd.DataFrame,
    trial_index: int,
    lick_times: Mapping[str, nap.Ts],
    pca_time_s: np.ndarray,
    pca_scores: np.ndarray,
    trial_position: int,
    alignment_event: str,
    window: tuple[float, float],
    pc_count: int = 5,
    lfp_time_s: np.ndarray | None = None,
    lfp_uv: np.ndarray | None = None,
    lfp_label: str | None = None,
    lfp_y_label: str = "LFP (uV)",
    figure_size: tuple[float, float] = (12.0, 10.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot one trial's behavior events and population PCA time courses.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``start_time``, ``choice_time``, ``led_on_time``, and ``action``.
        Times are in seconds.
    trial_index : int
        Trial row index used to extract behavior events from ``trial_df``.
    lick_times : Mapping[str, nap.Ts]
        Mapping with ``"left_entry"`` and ``"right_entry"`` keys. Each value
        is a one-dimensional Pynapple ``Ts`` of lick timestamps in seconds.
    pca_time_s : np.ndarray
        One-dimensional PCA bin centers with shape ``(n_bins,)`` in seconds
        relative to ``alignment_event``.
    pca_scores : np.ndarray
        PCA score tensor with shape ``(n_trials, n_bins, n_components)``.
        ``trial_position`` selects the first axis.
    trial_position : int
        Zero-based position of ``trial_index`` within the PCA fit trial set.
    alignment_event : str
        Event column used as relative time zero, typically ``"start_time"`` or
        ``"choice_time"``.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    pc_count : int, default=5
        Number of leading PCs to plot. The plotted count is capped by the
        number of fitted components.
    lfp_time_s : np.ndarray | None, optional
        Optional LFP time axis with shape ``(n_lfp_samples,)`` in seconds
        relative to alignment. If provided, ``lfp_uv`` must also be provided.
    lfp_uv : np.ndarray | None, optional
        Optional LFP trace with shape ``(n_lfp_samples,)``.
    lfp_label : str | None, optional
        Title label for the optional LFP axis.
    lfp_y_label : str, default="LFP (uV)"
        Y-axis label for the optional LFP trace.
    figure_size : tuple[float, float], default=(12.0, 10.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Matplotlib figure and axes in top-to-bottom order: optional LFP,
        behavior events, and PCA traces. X-axis units are seconds relative to
        ``alignment_event``.
    """

    required_columns = {"start_time", "choice_time", "led_on_time", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if int(pc_count) < 1:
        raise ValueError("pc_count must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    pca_time_s = np.asarray(pca_time_s, dtype=float).reshape(-1)
    pca_scores = np.asarray(pca_scores, dtype=float)
    if pca_scores.ndim != 3:
        raise ValueError("pca_scores must have shape (n_trials, n_bins, n_components).")
    if int(trial_position) < 0 or int(trial_position) >= pca_scores.shape[0]:
        raise ValueError("trial_position is outside the PCA score trial axis.")
    if pca_time_s.shape[0] != pca_scores.shape[1]:
        raise ValueError("pca_time_s length must match the PCA score time axis.")

    show_lfp = lfp_time_s is not None or lfp_uv is not None
    if show_lfp:
        if lfp_time_s is None or lfp_uv is None:
            raise ValueError("lfp_time_s and lfp_uv must be provided together.")
        lfp_time_s = np.asarray(lfp_time_s, dtype=float).reshape(-1)
        lfp_uv = np.asarray(lfp_uv, dtype=float).reshape(-1)
        if lfp_time_s.shape != lfp_uv.shape:
            raise ValueError("lfp_time_s and lfp_uv must have the same one-dimensional shape.")

    reference_time = get_trial_alignment_time(
        trial_df=trial_df,
        trial_index=int(trial_index),
        alignment_event=alignment_event,
    )
    left_licks = extract_relative_events_for_trial(
        event_times=lick_times[LEFT_LICK_EVENT],
        reference_time=reference_time,
        window=window,
    )
    right_licks = extract_relative_events_for_trial(
        event_times=lick_times[RIGHT_LICK_EVENT],
        reference_time=reference_time,
        window=window,
    )

    n_axes = 3 if show_lfp else 2
    height_ratios = [1, 1, 3] if show_lfp else [1, 3]
    figure, axes = plt.subplots(
        n_axes,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=height_ratios,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    axis_index = 0
    if show_lfp:
        lfp_axis = axes[axis_index]
        axis_index += 1
        lfp_axis.plot(lfp_time_s, lfp_uv, color="black", linewidth=0.8)
        lfp_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
        lfp_axis.set_ylabel(str(lfp_y_label))
        lfp_axis.set_xlim(float(window[0]), float(window[1]))
        lfp_axis.set_title(lfp_label or "LFP")

    behavior_axis = axes[axis_index]
    pca_axis = axes[axis_index + 1]

    behavior_axis.eventplot(
        [right_licks],
        orientation="horizontal",
        lineoffsets=[0.0],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["label"],
    )
    behavior_axis.eventplot(
        [left_licks],
        orientation="horizontal",
        lineoffsets=[1.0],
        linelengths=0.7,
        linewidths=1.0,
        colors=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["color"],
        label=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["label"],
    )
    trial_row = trial_df.loc[int(trial_index)]
    choice_time = pd.to_numeric(pd.Series([trial_row["choice_time"]]), errors="coerce").iloc[0]
    if not pd.isna(choice_time):
        choice_offset = float(choice_time) - reference_time
        action_value = pd.to_numeric(pd.Series([trial_row["action"]]), errors="coerce").iloc[0]
        choice_y = 0.5
        if not pd.isna(action_value):
            choice_y = 1.0 if int(action_value) == LEFT_CHOICE_ACTION else 0.0
        behavior_axis.vlines(
            choice_offset,
            choice_y - 0.35,
            choice_y + 0.35,
            colors="tab:purple",
            linewidth=1.4,
            label="Choice",
        )
    led_time = pd.to_numeric(pd.Series([trial_row["led_on_time"]]), errors="coerce").iloc[0]
    if not pd.isna(led_time):
        behavior_axis.vlines(
            float(led_time) - reference_time,
            -0.35,
            1.35,
            colors="tab:green",
            linewidth=1.1,
            label="LED",
        )
    behavior_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, label=alignment_event)
    behavior_axis.set_yticks([0.0, 1.0])
    behavior_axis.set_yticklabels(["Right licks", "Left licks"])
    behavior_axis.set_ylim(-0.7, 1.7)
    behavior_axis.set_ylabel("Behavior")
    behavior_axis.set_xlim(float(window[0]), float(window[1]))
    behavior_axis.set_title(f"Trial {trial_index} behavior aligned to {alignment_event}")
    behavior_axis.legend(loc="upper right", fontsize="small")

    plotted_pc_count = min(int(pc_count), pca_scores.shape[2])
    for pc_index in range(plotted_pc_count):
        pca_axis.plot(
            pca_time_s,
            pca_scores[int(trial_position), :, pc_index],
            linewidth=1.2,
            label=f"PC{pc_index + 1}",
        )
    pca_axis.axvline(0.0, color="gray", linestyle="--", linewidth=1.2)
    pca_axis.set_xlim(float(window[0]), float(window[1]))
    pca_axis.set_ylabel("PC score")
    pca_axis.set_xlabel(f"Time from {alignment_event} (s)")
    pca_axis.set_title(f"Population PCA trajectory, first {plotted_pc_count} PCs")
    pca_axis.legend(loc="upper right", fontsize="small", ncol=min(plotted_pc_count, 5))

    figure.tight_layout()
    return figure, axes


def build_concatenated_trial_time_axis(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    pca_time_s: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    axis_mode: str = "auto",
) -> dict[str, object]:
    """
    Build x-axis coordinates for concatenated trial-window PCA inspection.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and an ``alignment_event`` column in
        seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)``.
    pca_time_s : np.ndarray
        One-dimensional PCA bin centers with shape ``(n_bins,)`` in seconds
        relative to ``alignment_event``.
    alignment_event : str
        Trial time column used as time zero.
    window : tuple[float, float]
        Relative window bounds in seconds as ``(start_s, end_s)``.
    axis_mode : str, default="auto"
        X-axis layout. ``"auto"`` preserves real elapsed time when trial
        windows are non-overlapping and otherwise falls back to pseudo-time.
        ``"pseudo_time"`` forces equal-width side-by-side trial windows.
        ``"real_time"`` requires finite, non-overlapping trial windows.

    Returns
    -------
    dict[str, object]
        Dictionary containing ``x_by_trial`` with shape ``(n_trials, n_bins)``,
        ``trial_start_x`` and ``trial_end_x`` with shape ``(n_trials,)``,
        ``trial_break_x`` with shape ``(max(n_trials - 1, 0),)``, and ``mode``
        as either ``"real_time"`` or ``"pseudo_time"``. X values are seconds
        from the first visible trial-window start in real-time mode, or seconds
        in concatenated trial-window coordinates in pseudo-time mode.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    if axis_mode not in {"auto", "real_time", "pseudo_time"}:
        raise ValueError("axis_mode must be 'auto', 'real_time', or 'pseudo_time'.")
    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if normalized_trial_indices.size == 0:
        raise ValueError("trial_indices must contain at least one trial.")
    pca_time_s = np.asarray(pca_time_s, dtype=float).reshape(-1)
    if pca_time_s.size == 0:
        raise ValueError("pca_time_s must contain at least one time bin.")

    alignment_times = pd.to_numeric(
        trial_df.loc[normalized_trial_indices, alignment_event],
        errors="coerce",
    ).to_numpy(dtype=float)
    window_start_s = float(window[0])
    window_end_s = float(window[1])
    window_width_s = window_end_s - window_start_s
    absolute_start_s = alignment_times + window_start_s
    absolute_end_s = alignment_times + window_end_s

    can_use_real_time = np.isfinite(alignment_times).all()
    if can_use_real_time and absolute_start_s.size > 1:
        can_use_real_time = bool(np.all(absolute_start_s[1:] >= absolute_end_s[:-1]))
    if axis_mode == "real_time" and not can_use_real_time:
        raise ValueError("axis_mode='real_time' requires finite, non-overlapping trial windows.")

    if axis_mode != "pseudo_time" and can_use_real_time:
        first_window_start_s = float(absolute_start_s[0])
        x_by_trial = alignment_times[:, np.newaxis] + pca_time_s[np.newaxis, :] - first_window_start_s
        trial_start_x = absolute_start_s - first_window_start_s
        trial_end_x = absolute_end_s - first_window_start_s
        mode = "real_time"
    else:
        trial_start_x = np.arange(normalized_trial_indices.size, dtype=float) * window_width_s
        trial_end_x = trial_start_x + window_width_s
        x_by_trial = trial_start_x[:, np.newaxis] + (pca_time_s[np.newaxis, :] - window_start_s)
        mode = "pseudo_time"
    trial_break_x = trial_start_x[1:].copy()

    return {
        "x_by_trial": x_by_trial,
        "trial_start_x": trial_start_x,
        "trial_end_x": trial_end_x,
        "trial_break_x": trial_break_x,
        "mode": mode,
    }


def plot_concatenated_trial_behavior_and_population_pca(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    lick_times: Mapping[str, nap.Ts],
    pca_time_s: np.ndarray,
    pca_scores: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    pc_count: int = 5,
    figure_size: tuple[float, float] = (11.0, 5.0),
    axis_mode: str = "auto",
) -> tuple[plt.Figure, np.ndarray, dict[str, object]]:
    """
    Plot behavior events and PCA trajectories across concatenated trial windows.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are
        ``start_time``, ``choice_time``, ``reward_time``, ``led_on_time``,
        ``action``, and ``reward``. Times are in seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)`` matching
        the first axis of ``pca_scores``.
    lick_times : Mapping[str, nap.Ts]
        Mapping with ``"left_entry"`` and ``"right_entry"`` keys. Each value is
        a one-dimensional Pynapple ``Ts`` of lick timestamps in seconds.
    pca_time_s : np.ndarray
        One-dimensional PCA bin centers with shape ``(n_bins,)`` in seconds
        relative to ``alignment_event``.
    pca_scores : np.ndarray
        PCA score tensor with shape ``(n_trials, n_bins, n_components)``.
    alignment_event : str
        Event column used as relative time zero.
    window : tuple[float, float]
        Plot bounds in seconds relative to ``alignment_event``.
    pc_count : int, default=5
        Number of leading PCs to plot. The plotted count is capped by fitted
        components.
    figure_size : tuple[float, float], default=(11.0, 5.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.
    axis_mode : str, default="auto"
        X-axis layout passed to ``build_concatenated_trial_time_axis``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray, dict[str, object]]
        Figure, two axes in top-to-bottom order, and axis-coordinate metadata
        from ``build_concatenated_trial_time_axis``.
    """

    required_columns = {"start_time", "choice_time", "reward_time", "led_on_time", "action", "reward"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")
    if LEFT_LICK_EVENT not in lick_times or RIGHT_LICK_EVENT not in lick_times:
        raise ValueError("lick_times must contain 'left_entry' and 'right_entry' keys.")
    if int(pc_count) < 1:
        raise ValueError("pc_count must be positive.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    pca_time_s = np.asarray(pca_time_s, dtype=float).reshape(-1)
    pca_scores = np.asarray(pca_scores, dtype=float)
    if pca_scores.ndim != 3:
        raise ValueError("pca_scores must have shape (n_trials, n_bins, n_components).")
    if pca_scores.shape[0] != normalized_trial_indices.size:
        raise ValueError("pca_scores trial axis must match trial_indices length.")
    if pca_scores.shape[1] != pca_time_s.size:
        raise ValueError("pca_scores time axis must match pca_time_s length.")

    axis_data = build_concatenated_trial_time_axis(
        trial_df=trial_df,
        trial_indices=normalized_trial_indices,
        pca_time_s=pca_time_s,
        alignment_event=alignment_event,
        window=window,
        axis_mode=axis_mode,
    )
    x_by_trial = np.asarray(axis_data["x_by_trial"], dtype=float)
    trial_start_x = np.asarray(axis_data["trial_start_x"], dtype=float)
    trial_end_x = np.asarray(axis_data["trial_end_x"], dtype=float)
    trial_break_x = np.asarray(axis_data["trial_break_x"], dtype=float)

    figure, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
        height_ratios=[0.9, 2.6],
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    behavior_axis = axes[0]
    pca_axis = axes[1]

    for trial_position, trial_index in enumerate(normalized_trial_indices):
        trial_row = trial_df.loc[int(trial_index)]
        reward_value = pd.to_numeric(pd.Series([trial_row["reward"]]), errors="coerce").iloc[0]
        shade_color = "tab:green" if not pd.isna(reward_value) and int(reward_value) == 1 else "tab:red"
        for axis in (behavior_axis, pca_axis):
            axis.axvspan(
                trial_start_x[trial_position],
                trial_end_x[trial_position],
                color=shade_color,
                alpha=0.08,
                linewidth=0,
            )
            axis.axvline(trial_start_x[trial_position], color="0.6", linewidth=0.8, alpha=0.35)
        pca_axis.axvline(trial_end_x[trial_position], color="0.6", linewidth=0.8, alpha=0.2)

        reference_time = get_trial_alignment_time(
            trial_df=trial_df,
            trial_index=int(trial_index),
            alignment_event=alignment_event,
        )
        trial_offset_x = x_by_trial[trial_position, 0] - pca_time_s[0]
        for event_column, y_value, color, label in (
            ("led_on_time", 0.5, "tab:green", "LED"),
            ("choice_time", 1.5, "tab:purple", "Choice"),
            ("reward_time", 2.5, "tab:green", "Reward"),
        ):
            event_time = pd.to_numeric(pd.Series([trial_row[event_column]]), errors="coerce").iloc[0]
            if pd.isna(event_time):
                continue
            relative_event_time = float(event_time) - reference_time
            if float(window[0]) <= relative_event_time <= float(window[1]):
                event_x = trial_offset_x + relative_event_time
                behavior_axis.plot(
                    [event_x, event_x],
                    [y_value - 0.3, y_value + 0.3],
                    color=color,
                    linewidth=1.2,
                    label=label,
                )

        left_licks = extract_relative_events_for_trial(
            event_times=lick_times[LEFT_LICK_EVENT],
            reference_time=reference_time,
            window=window,
        )
        right_licks = extract_relative_events_for_trial(
            event_times=lick_times[RIGHT_LICK_EVENT],
            reference_time=reference_time,
            window=window,
        )
        if right_licks.size:
            behavior_axis.eventplot(
                [trial_offset_x + right_licks],
                orientation="horizontal",
                lineoffsets=[3.5],
                linelengths=0.4,
                linewidths=0.8,
                colors=LICK_RASTER_STYLES[RIGHT_LICK_EVENT]["color"],
            )
        if left_licks.size:
            behavior_axis.eventplot(
                [trial_offset_x + left_licks],
                orientation="horizontal",
                lineoffsets=[4.5],
                linelengths=0.4,
                linewidths=0.8,
                colors=LICK_RASTER_STYLES[LEFT_LICK_EVENT]["color"],
            )

    plotted_pc_count = min(int(pc_count), pca_scores.shape[2])
    pc_color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    if not pc_color_cycle:
        pc_color_cycle = ["C0"]
    for pc_index in range(plotted_pc_count):
        pc_color = pc_color_cycle[pc_index % len(pc_color_cycle)]
        for trial_position in range(normalized_trial_indices.size):
            pca_axis.plot(
                x_by_trial[trial_position],
                pca_scores[trial_position, :, pc_index],
                color=pc_color,
                linewidth=1.0,
                label=f"PC{pc_index + 1}",
            )

    for break_position_x in trial_break_x:
        for axis in (behavior_axis, pca_axis):
            axis.axvline(
                float(break_position_x),
                color="black",
                linewidth=1.6,
                alpha=0.25,
                label="Trial break",
            )

    behavior_axis.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    behavior_axis.set_yticklabels(["LED", "Choice", "Reward", "Right licks", "Left licks"])
    behavior_axis.set_ylim(0.0, 5.0)
    behavior_axis.set_ylabel("Behavior")
    behavior_axis.set_title("Concatenated trial behavior and PCA")

    handles, labels = behavior_axis.get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique_labels.setdefault(label, handle)
    if unique_labels:
        behavior_axis.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize="small")

    pca_axis.set_xlim(float(trial_start_x[0]), float(trial_end_x[-1]))
    pca_axis.set_ylabel("PC score")
    pca_axis.set_xlabel(
        "Time from first visible trial window start (s)"
        if axis_data["mode"] == "real_time"
        else "Concatenated trial-window time (s)"
    )
    pca_axis.set_title(f"Population PCA trajectories, first {plotted_pc_count} PCs")
    handles, labels = pca_axis.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        unique_labels.setdefault(label, handle)
    pca_axis.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize="small", ncol=min(plotted_pc_count, 5))

    figure.tight_layout()
    return figure, axes, axis_data


def plot_pca_cumulative_explained_variance(
    cumulative_explained_variance: np.ndarray,
    pc_count: int | None = None,
    figure_size: tuple[float, float] = (5.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cumulative explained variance against the number of PCs.

    Parameters
    ----------
    cumulative_explained_variance : np.ndarray
        One-dimensional cumulative explained variance ratio with shape
        ``(n_components,)``. Values are fractions between 0 and 1.
    pc_count : int | None, optional
        Maximum number of leading PCs to display. ``None`` displays all
        supplied components. The displayed count is capped by available
        components.
    figure_size : tuple[float, float], default=(5.0, 3.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axis. X-axis is one-based PC count; y-axis is
        cumulative explained variance ratio.
    """

    cumulative_explained_variance = np.asarray(cumulative_explained_variance, dtype=float).reshape(-1)
    if cumulative_explained_variance.size == 0:
        raise ValueError("cumulative_explained_variance must contain at least one value.")
    if not np.isfinite(cumulative_explained_variance).all():
        raise ValueError("cumulative_explained_variance must contain only finite values.")
    if pc_count is not None and int(pc_count) < 1:
        raise ValueError("pc_count must be positive when provided.")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    displayed_pc_count = cumulative_explained_variance.size
    if pc_count is not None:
        displayed_pc_count = min(displayed_pc_count, int(pc_count))
    cumulative_explained_variance = cumulative_explained_variance[:displayed_pc_count]
    pc_numbers = np.arange(1, cumulative_explained_variance.size + 1, dtype=int)
    figure, axis = plt.subplots(1, 1, figsize=(float(figure_size[0]), float(figure_size[1])))
    axis.plot(pc_numbers, cumulative_explained_variance, marker="o", color="black", linewidth=1.2)
    axis.set_xlabel("Number of PCs")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0.0, min(1.05, max(1.0, float(cumulative_explained_variance.max()) * 1.05)))
    if displayed_pc_count <= 20:
        axis.set_xticks(pc_numbers)
    else:
        tick_interval = max(1, int(np.ceil(displayed_pc_count / 10)))
        sparse_ticks = np.unique(
            np.concatenate(
                [
                    np.array([1], dtype=int),
                    np.arange(tick_interval, displayed_pc_count + 1, tick_interval, dtype=int),
                    np.array([displayed_pc_count], dtype=int),
                ]
            )
        )
        axis.set_xticks(sparse_ticks)
    figure.tight_layout()
    return figure, axis


def plot_pca_decoding_pre_post_scores(
    results_df: pd.DataFrame,
    figure_size: tuple[float, float] = (7.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot side-by-side pre/post choice PCA decoding accuracy by condition.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form PCA decoding result table with ``condition``, ``window``,
        ``status``, and ``cv_score`` columns. Only rows with ``status == "ok"``
        and finite ``cv_score`` are plotted.
    figure_size : tuple[float, float], default=(7.0, 3.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing grouped bars. X-axis groups are base
        conditions; pre/post bars are horizontally adjacent within each group.
    """

    required_columns = {"condition", "window", "status", "cv_score"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"results_df is missing required columns: {sorted(missing_columns)}")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    ok_rows = results_df.loc[results_df["status"] == "ok"].copy()
    ok_rows["cv_score"] = pd.to_numeric(ok_rows["cv_score"], errors="coerce")
    ok_rows = ok_rows.loc[ok_rows["cv_score"].notna()]

    figure, axis = plt.subplots(1, 1, figsize=(float(figure_size[0]), float(figure_size[1])))
    if ok_rows.empty:
        axis.set_ylabel("CV decoding accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("PCA decoding before/after choice")
        figure.tight_layout()
        return figure, axis

    condition_order = [
        condition_name
        for condition_name in PCA_DECODING_BASE_CONDITIONS
        if condition_name in set(ok_rows["condition"])
    ]
    condition_order.extend(
        condition_name
        for condition_name in ok_rows["condition"].tolist()
        if condition_name not in condition_order
    )
    condition_order = list(dict.fromkeys(condition_order))
    window_offsets = {"pre_choice": -0.18, "post_choice": 0.18}
    window_colors = {"pre_choice": "tab:blue", "post_choice": "tab:orange"}
    bar_width = 0.32
    group_spacing = 1.35
    group_centers = np.arange(len(condition_order), dtype=float) * group_spacing

    for condition_position, condition_name in enumerate(condition_order):
        condition_rows = ok_rows.loc[ok_rows["condition"] == condition_name]
        for window_name in ("pre_choice", "post_choice"):
            window_rows = condition_rows.loc[condition_rows["window"] == window_name]
            if window_rows.empty:
                continue
            bar_x = group_centers[condition_position] + window_offsets[window_name]
            axis.bar(
                bar_x,
                float(window_rows.iloc[0]["cv_score"]),
                width=bar_width,
                color=window_colors[window_name],
                label=window_name,
            )

    handles, labels = axis.get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique_labels.setdefault(label, handle)
    if unique_labels:
        axis.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize="small")
    axis.set_xticks(group_centers)
    axis.set_xticklabels(condition_order, rotation=25, ha="right")
    axis.set_ylabel("CV decoding accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("PCA decoding before/after choice")
    figure.tight_layout()
    return figure, axis


def plot_average_pca_scores_by_condition_and_target(
    score_summary_df: pd.DataFrame,
    raw_score_df: pd.DataFrame | None = None,
    figure_size: tuple[float, float] = (8.0, 4.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot average PC1/PC2 score points by condition and target value.

    Parameters
    ----------
    score_summary_df : pd.DataFrame
        Output from ``summarize_pca_scores_by_condition_and_target`` with
        columns ``PCA_SCORE_SUMMARY_COLUMNS``.
    raw_score_df : pd.DataFrame | None, default=None
        Optional output from ``extract_pca_score_points_by_condition_and_target``.
        Rows are plotted as transparent condition-membership points beneath the
        group means. A trial can appear more than once if selected conditions
        overlap.
    figure_size : tuple[float, float], default=(8.0, 4.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and two axes. The first axis shows pre-choice group means; the
        second axis shows post-choice group means. X is mean PC1 and Y is mean
        PC2.
    """

    missing_columns = set(PCA_SCORE_SUMMARY_COLUMNS) - set(score_summary_df.columns)
    if missing_columns:
        raise ValueError(f"score_summary_df is missing required columns: {sorted(missing_columns)}")
    if raw_score_df is not None:
        missing_raw_columns = set(PCA_RAW_SCORE_COLUMNS) - set(raw_score_df.columns)
        if missing_raw_columns:
            raise ValueError(f"raw_score_df is missing required columns: {sorted(missing_raw_columns)}")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    figure, axes = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)

    condition_values = score_summary_df["condition"].tolist()
    if raw_score_df is not None:
        condition_values.extend(raw_score_df["condition"].tolist())
    condition_order = [
        condition_name
        for condition_name in PCA_DECODING_BASE_CONDITIONS
        if condition_name in set(condition_values)
    ]
    condition_order.extend(
        condition_name
        for condition_name in condition_values
        if condition_name not in condition_order
    )
    condition_order = list(dict.fromkeys(condition_order))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    condition_colors = {
        condition_name: color_cycle[index % len(color_cycle)]
        for index, condition_name in enumerate(condition_order)
    }
    target_label_values = score_summary_df["target_label"].tolist()
    if raw_score_df is not None:
        target_label_values.extend(raw_score_df["target_label"].tolist())
    target_labels = list(dict.fromkeys(target_label_values))
    marker_cycle = ["o", "^", "s", "D", "P", "X"]
    target_markers = {
        target_label: marker_cycle[index % len(marker_cycle)]
        for index, target_label in enumerate(target_labels)
    }

    for axis, window_name, title in zip(
        axes,
        ("pre_choice", "post_choice"),
        ("Pre-choice (-0.5 to 0 s)", "Post-choice (0 to 0.5 s)"),
        strict=True,
    ):
        if raw_score_df is not None:
            raw_window_rows = raw_score_df.loc[raw_score_df["window"] == window_name]
            for _, row in raw_window_rows.iterrows():
                axis.scatter(
                    float(row["pc1"]),
                    float(row["pc2"]),
                    color=condition_colors[str(row["condition"])],
                    marker=target_markers[str(row["target_label"])],
                    s=18,
                    alpha=0.18,
                    edgecolor="none",
                    linewidth=0.0,
                )
        window_rows = score_summary_df.loc[score_summary_df["window"] == window_name]
        for _, row in window_rows.iterrows():
            axis.scatter(
                float(row["pc1_mean"]),
                float(row["pc2_mean"]),
                color=condition_colors[str(row["condition"])],
                marker=target_markers[str(row["target_label"])],
                s=70,
                edgecolor="black",
                linewidth=0.6,
                alpha=1.0,
            )
        axis.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
        axis.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
        axis.set_title(title)
        axis.set_xlabel("Mean PC1 score")
    axes[0].set_ylabel("Mean PC2 score")

    condition_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=condition_colors[condition_name],
            label=condition_name,
        )
        for condition_name in condition_order
    ]
    target_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=target_markers[target_label],
            linestyle="",
            color="black",
            markerfacecolor="white",
            label=target_label,
        )
        for target_label in target_labels
    ]
    if condition_handles:
        axes[1].legend(
            handles=[*condition_handles, *target_handles],
            loc="best",
            fontsize="small",
            title="Condition / target",
        )
    figure.tight_layout()
    return figure, axes


def plot_switch_event_pca_trajectories(
    trajectory_df: pd.DataFrame,
    figure_size: tuple[float, float] = (11.0, 7.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot individual and mean four-point switch trajectories in PC1/PC2 space.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Long-form trajectory table returned by
        ``extract_switch_event_pca_trajectories``. PC coordinates are in PCA
        score units.
    figure_size : tuple[float, float], default=(11, 7)
        Matplotlib figure width and height in inches.

    Returns
    -------
    tuple[matplotlib.figure.Figure, np.ndarray]
        Figure and object array with shape ``(2, 3)``. Rows follow
        ``SWITCH_DIRECTION_ORDER`` and columns follow ``SWITCH_TYPE_ORDER``.
        All six axes use identical PC1 and PC2 limits.
    """

    required_columns = {
        "event_id",
        "switch_type",
        "switch_direction",
        "point_order",
        "pc1",
        "pc2",
    }
    missing_columns = required_columns - set(trajectory_df.columns)
    if missing_columns:
        raise ValueError(f"trajectory_df is missing required columns: {sorted(missing_columns)}")

    figure, axes = plt.subplots(2, 3, figsize=figure_size, sharex=True, sharey=True)
    axes = np.asarray(axes, dtype=object).reshape(2, 3)
    panel_colors = ("tab:green", "tab:red", "tab:blue")
    point_markers = ("o", "s", "^", "D")

    for row_index, switch_direction in enumerate(SWITCH_DIRECTION_ORDER):
        for column_index, (switch_type, color) in enumerate(
            zip(SWITCH_TYPE_ORDER, panel_colors, strict=True)
        ):
            axis = axes[row_index, column_index]
            panel_mask = trajectory_df["switch_type"].eq(switch_type) & trajectory_df[
                "switch_direction"
            ].eq(switch_direction)
            panel_df = trajectory_df.loc[panel_mask]
            for _event_id, event_df in panel_df.groupby("event_id", sort=False):
                ordered_event = event_df.sort_values("point_order")
                x_values = ordered_event["pc1"].to_numpy(dtype=float)
                y_values = ordered_event["pc2"].to_numpy(dtype=float)
                axis.plot(x_values, y_values, color=color, linewidth=0.8, alpha=0.2, zorder=1)
                _draw_directed_segments(
                    axis,
                    x_values,
                    y_values,
                    color=color,
                    alpha=0.2,
                    linewidth=0.8,
                )
                for point_order, marker in enumerate(point_markers):
                    point = ordered_event.loc[ordered_event["point_order"] == point_order]
                    if point.empty:
                        continue
                    axis.scatter(
                        point["pc1"],
                        point["pc2"],
                        color=color,
                        marker=marker,
                        s=18,
                        alpha=0.2,
                        linewidths=0,
                        zorder=2,
                    )

            if not panel_df.empty:
                mean_points = (
                    panel_df.groupby("point_order", sort=True)[["pc1", "pc2"]]
                    .mean()
                    .reindex(range(4))
                )
                mean_x = mean_points["pc1"].to_numpy(dtype=float)
                mean_y = mean_points["pc2"].to_numpy(dtype=float)
                axis.plot(mean_x, mean_y, color=color, linewidth=2.8, alpha=1.0, zorder=4)
                _draw_directed_segments(
                    axis,
                    mean_x,
                    mean_y,
                    color=color,
                    alpha=1.0,
                    linewidth=2.2,
                )
                for point_order, marker in enumerate(point_markers):
                    axis.scatter(
                        mean_x[point_order],
                        mean_y[point_order],
                        color=color,
                        edgecolor="white",
                        linewidth=0.8,
                        marker=marker,
                        s=75,
                        zorder=5,
                    )
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No events",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

            event_count = int(panel_df["event_id"].nunique())
            if row_index == 0:
                axis.set_title(SWITCH_TYPE_TITLES[switch_type])
            if row_index == len(SWITCH_DIRECTION_ORDER) - 1:
                axis.set_xlabel("PC1 score")
            axis.text(
                0.03,
                0.97,
                f"n = {event_count}",
                ha="left",
                va="top",
                transform=axis.transAxes,
            )
            axis.grid(alpha=0.2)
        axes[row_index, 0].set_ylabel(
            f"{SWITCH_DIRECTION_TITLES[switch_direction]}\nPC2 score"
        )

    x_limits, y_limits = _compute_shared_axis_limits(trajectory_df)
    for axis in axes.reshape(-1):
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=marker,
            linestyle="none",
            markersize=6,
            label=label,
        )
        for marker, label in zip(
            point_markers,
            ("Previous pre", "Previous post", "Next pre", "Next post"),
            strict=True,
        )
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("Choice-switch population trajectories", y=0.99)
    figure.text(
        0.5,
        0.015,
        "Thin paths are individual events; thick paths are condition means. Arrows show temporal order.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.88))
    return figure, axes


def _draw_directed_segments(
    axis: plt.Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    """
    Draw arrowheads over consecutive segments of one PC path.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis receiving the directed segments.
    x_values, y_values : np.ndarray
        One-dimensional PC coordinate arrays with matching shape ``(n_points,)``.
        Values are in PCA score units.
    color : str
        Matplotlib color specification.
    alpha : float
        Arrow opacity from zero to one.
    linewidth : float
        Arrow line width in points.

    Returns
    -------
    None
        The supplied axis is modified in place.
    """

    for point_index in range(len(x_values) - 1):
        axis.annotate(
            "",
            xy=(x_values[point_index + 1], y_values[point_index + 1]),
            xytext=(x_values[point_index], y_values[point_index]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "alpha": alpha,
                "linewidth": linewidth,
                "mutation_scale": 8,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=3,
        )


def _compute_shared_axis_limits(
    trajectory_df: pd.DataFrame,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute padded PC1 and PC2 limits shared by every switch panel.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        Long-form trajectory table with numeric ``pc1`` and ``pc2`` columns in
        PCA score units.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        PC1 and PC2 ``(minimum, maximum)`` limits in PCA score units.
    """

    if trajectory_df.empty:
        return (-1.0, 1.0), (-1.0, 1.0)
    pc1 = pd.to_numeric(trajectory_df["pc1"], errors="coerce").to_numpy(dtype=float)
    pc2 = pd.to_numeric(trajectory_df["pc2"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(pc1) & np.isfinite(pc2)
    if not finite_mask.any():
        return (-1.0, 1.0), (-1.0, 1.0)
    return _padded_limits(pc1[finite_mask]), _padded_limits(pc2[finite_mask])


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    """
    Return finite min/max limits with eight-percent visual padding.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional finite coordinate values in PCA score units.

    Returns
    -------
    tuple[float, float]
        Padded ``(minimum, maximum)`` limits in the input coordinate units.
    """

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    padding = 0.08 * span if span > 0.0 else max(1.0, abs(minimum) * 0.08)
    return minimum - padding, maximum + padding


def append_plot_title_suffix(axis: plt.Axes, title_suffix: str | None) -> None:
    """
    Append a date-selection label to an existing plot title.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis whose title should be updated.
    title_suffix : str | None
        Optional text appended on a new title line.

    Returns
    -------
    None
        The axis title is modified in place when ``title_suffix`` is supplied.
    """

    if title_suffix is None:
        return
    axis.set_title(f"{axis.get_title()}\n{title_suffix}")


def _format_condition_label(trial_condition: str) -> str:
    """Convert underscore-separated trial-condition names into plot-friendly labels."""
    return trial_condition.replace("_", " ")


def _format_session_date_label(session_row: pd.Series) -> str:
    """
    Format a session row into a date-only label for plot legends.

    Parameters
    ----------
    session_row : pd.Series
        One row containing at least a ``date`` field.

    Returns
    -------
    str
        Session date label for the legend.
    """

    if "date" not in session_row or pd.isna(session_row["date"]):
        raise ValueError("Session-mean decoder plots require a non-null 'date' column for legend labels.")
    return str(session_row["date"])


def _plot_paired_before_after(
    values_df: pd.DataFrame,
    trial_condition: str,
    before_column: str,
    after_column: str,
    ylabel: str,
    title_prefix: str,
    *,
    y_limits: tuple[float, float] = (0.0, 1.0),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot paired before/after values across sessions in the legacy style.

    Parameters
    ----------
    values_df : pd.DataFrame
        Cross-session table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    before_column : str
        Column containing before-choice values.
    after_column : str
        Column containing after-choice values.
    ylabel : str
        Y-axis label for the metric being plotted.
    title_prefix : str
        Prefix used in the figure title before the condition label.
    y_limits : tuple[float, float], optional
        Lower and upper y-axis limits for the plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the paired before/after plot.
    """

    condition_df = values_df.loc[values_df["trial_condition"] == trial_condition].copy()
    if condition_df.empty:
        raise ValueError(f"No rows found for trial_condition={trial_condition!r}.")

    figure, axis = plt.subplots(figsize=(8, 6))
    before_values = condition_df[before_column].to_numpy(dtype=float)
    after_values = condition_df[after_column].to_numpy(dtype=float)

    for _, row in condition_df.iterrows():
        axis.plot(
            [0, 1],
            [float(row[before_column]), float(row[after_column])],
            color="gray",
            alpha=0.5,
        )

    axis.scatter(np.zeros(before_values.shape[0]), before_values, color="C0", alpha=0.5)
    axis.scatter(np.ones(after_values.shape[0]), after_values, color="C1", alpha=0.5)
    axis.plot(
        [0, 1],
        [float(np.nanmean(before_values)), float(np.nanmean(after_values))],
        color="k",
        linestyle="--",
        linewidth=2,
    )
    axis.set_xlim(*BEFORE_AFTER_XLIM)
    axis.set_ylim(*y_limits)
    axis.set_xticks([0, 1], ["Before\nchoice", "After\nchoice"])
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title_prefix}, {_format_condition_label(trial_condition)} trials")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis


def plot_cross_session_decodability_scores(
    decodability_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cross-session before/after CV scores for one trial condition.

    Parameters
    ----------
    decodability_df : pd.DataFrame
        Cross-session decodability table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the CV-score plot.
    """

    return _plot_paired_before_after(
        decodability_df,
        trial_condition=trial_condition,
        before_column="cv_score_before",
        after_column="cv_score_after",
        ylabel="CV score",
        title_prefix="Decoder performance",
        show=show,
    )


def plot_cross_session_decodability_pvalues(
    decodability_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cross-session before/after decodability p-values for one trial condition.

    Parameters
    ----------
    decodability_df : pd.DataFrame
        Cross-session decodability table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the p-value plot.
    """

    return _plot_paired_before_after(
        decodability_df,
        trial_condition=trial_condition,
        before_column="p_value_before",
        after_column="p_value_after",
        ylabel="P value",
        title_prefix="Decoder performance",
        show=show,
    )


def plot_cross_session_decoder_accuracy(
    decoder_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cross-session before/after decoder test accuracy for one trial condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the test-accuracy plot.
    """

    return _plot_paired_before_after(
        decoder_df,
        trial_condition=trial_condition,
        before_column="test_accuracy_before",
        after_column="test_accuracy_after",
        ylabel="Test accuracy",
        title_prefix="Decoder performance",
        show=show,
    )


def _compute_session_mean_decoder_table(
    decoder_df: pd.DataFrame,
    trial_condition: str,
) -> pd.DataFrame:
    """
    Compute per-session decoder means for one trial condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session, condition, and decoder run.
    trial_condition : str
        Condition name whose per-session decoder means should be computed.

    Returns
    -------
    pd.DataFrame
        One row per session with columns ``session``, ``date``, ``test_accuracy_before``,
        and ``test_accuracy_after`` representing the mean across decoder runs.
    """

    condition_df = decoder_df.loc[decoder_df["trial_condition"] == trial_condition].copy()
    if condition_df.empty:
        raise ValueError(f"No rows found for trial_condition={trial_condition!r}.")
    if "date" not in condition_df.columns:
        condition_df["date"] = condition_df["session"].astype(str)

    return (
        condition_df.groupby("session", as_index=False).agg(
            date=("date", "first"),
            test_accuracy_before=("test_accuracy_before", "mean"),
            test_accuracy_after=("test_accuracy_after", "mean"),
        )
        .sort_values("session")
    )


def plot_cross_session_decoder_superplot(
    decoder_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot all decoder runs, per-session means, and the mean of session means for one condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session, condition, and decoder run.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the mixed-model style decoder super-plot.
    """

    condition_df = decoder_df.loc[decoder_df["trial_condition"] == trial_condition].copy()
    if condition_df.empty:
        raise ValueError(f"No rows found for trial_condition={trial_condition!r}.")

    figure, axis = plt.subplots(figsize=(8, 6))
    session_names = sorted(condition_df["session"].unique().tolist())
    color_values = np.linspace(0.0, 1.0, max(len(session_names), 2))
    session_colors = {
        session_name: plt.cm.tab10(color_values[min(ix, 9)])
        for ix, session_name in enumerate(session_names)
    }

    for session_name in session_names:
        session_df = condition_df.loc[condition_df["session"] == session_name]
        color = session_colors[session_name]
        for _, decoder_row in session_df.iterrows():
            axis.plot(
                [0, 1],
                [
                    float(decoder_row["test_accuracy_before"]),
                    float(decoder_row["test_accuracy_after"]),
                ],
                color=color,
                alpha=0.25,
                linewidth=1.0,
            )

    session_mean_df = _compute_session_mean_decoder_table(decoder_df, trial_condition=trial_condition)
    for _, session_row in session_mean_df.iterrows():
        axis.plot(
            [0, 1],
            [
                float(session_row["test_accuracy_before"]),
                float(session_row["test_accuracy_after"]),
            ],
            color=session_colors[str(session_row["session"])],
            alpha=0.9,
            linewidth=2.5,
        )

    overall_mean_before = float(session_mean_df["test_accuracy_before"].mean())
    overall_mean_after = float(session_mean_df["test_accuracy_after"].mean())
    axis.plot(
        [0, 1],
        [overall_mean_before, overall_mean_after],
        color="k",
        linestyle="--",
        linewidth=2,
    )
    axis.set_xlim(*BEFORE_AFTER_XLIM)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0, 1], ["Before\nchoice", "After\nchoice"])
    axis.set_ylabel("Test accuracy")
    axis.set_title(f"Decoder performance, {_format_condition_label(trial_condition)} trials")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis


def plot_cross_session_decoder_session_means(
    decoder_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot per-session decoder means and the mean of session means for one condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session, condition, and decoder run.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the session-mean decoder plot with a date legend.
    """

    session_mean_df = _compute_session_mean_decoder_table(decoder_df, trial_condition=trial_condition)

    figure, axis = plt.subplots(figsize=(8, 6))
    session_names = sorted(session_mean_df["session"].unique().tolist())
    color_values = np.linspace(0.0, 1.0, max(len(session_names), 2))
    session_colors = {
        session_name: plt.cm.tab10(color_values[min(ix, 9)])
        for ix, session_name in enumerate(session_names)
    }
    for _, session_row in session_mean_df.iterrows():
        axis.plot(
            [0, 1],
            [
                float(session_row["test_accuracy_before"]),
                float(session_row["test_accuracy_after"]),
            ],
            color=session_colors[str(session_row["session"])],
            alpha=0.9,
            linewidth=2.0,
            label=_format_session_date_label(session_row),
        )

    overall_mean_before = float(session_mean_df["test_accuracy_before"].mean())
    overall_mean_after = float(session_mean_df["test_accuracy_after"].mean())
    axis.plot(
        [0, 1],
        [overall_mean_before, overall_mean_after],
        color="k",
        linestyle="--",
        linewidth=2,
        label="Overall mean",
    )
    axis.set_xlim(*BEFORE_AFTER_XLIM)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0, 1], ["Before\nchoice", "After\nchoice"])
    axis.set_ylabel("Test accuracy")
    axis.set_title(f"Decoder performance, {_format_condition_label(trial_condition)} trials")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis
