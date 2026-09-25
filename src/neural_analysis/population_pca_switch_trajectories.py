"""Compatibility facade retaining population-switch plots until R3."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.neural_analysis.population.switch_trajectories import (
    SWITCH_PRE_FILTER_ALL,
    SWITCH_PRE_FILTER_CORRECT_REWARDED,
    SWITCH_PRE_FILTER_OMISSION,
    SWITCH_PRE_FILTER_OPTIONS,
    SWITCH_TYPE_ORDER,
    SWITCH_TYPE_TITLES,
    SWITCH_DIRECTION_ORDER,
    SWITCH_DIRECTION_TITLES,
    SWITCH_EVENT_COLUMNS,
    SWITCH_TRAJECTORY_COLUMNS,
    select_valid_choice_trial_indices,
    select_choice_switch_events,
    extract_switch_event_pca_trajectories,
    summarize_switch_event_counts,
)


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
