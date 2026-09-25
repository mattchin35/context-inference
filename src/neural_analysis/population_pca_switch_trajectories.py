from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.neural_analysis.spike_behavior import trials as spike_behavior_trials


SWITCH_PRE_FILTER_ALL = "All choice switches"
SWITCH_PRE_FILTER_CORRECT_REWARDED = "Correct rewarded before switch"
SWITCH_PRE_FILTER_OMISSION = "Omission before switch"
SWITCH_PRE_FILTER_OPTIONS = (
    SWITCH_PRE_FILTER_ALL,
    SWITCH_PRE_FILTER_CORRECT_REWARDED,
    SWITCH_PRE_FILTER_OMISSION,
)
SWITCH_TYPE_ORDER = (
    "incorrect_to_correct",
    "correct_to_incorrect",
    "correct_to_correct",
)
SWITCH_TYPE_TITLES = {
    "incorrect_to_correct": "Incorrect to correct",
    "correct_to_incorrect": "Correct to incorrect",
    "correct_to_correct": "Correct to correct",
}
SWITCH_DIRECTION_ORDER = ("left_to_right", "right_to_left")
SWITCH_DIRECTION_TITLES = {
    "left_to_right": "Left to right",
    "right_to_left": "Right to left",
}
SWITCH_EVENT_COLUMNS = [
    "event_id",
    "previous_trial_index",
    "next_trial_index",
    "previous_correct",
    "next_correct",
    "previous_reward",
    "switch_type",
    "switch_direction",
]
SWITCH_TRAJECTORY_COLUMNS = [
    "event_id",
    "switch_type",
    "switch_direction",
    "previous_trial_index",
    "next_trial_index",
    "point_order",
    "point_label",
    "trial_index",
    "choice_window",
    "pc1",
    "pc2",
]


def select_valid_choice_trial_indices(trial_df: pd.DataFrame) -> np.ndarray:
    """
    Select all trials eligible for the choice-aligned population PCA fit.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. It must contain the columns required
        by ``make_trial_type_masks`` plus ``choice_time`` in seconds. ``action``
        contains scalar choice labels.

    Returns
    -------
    np.ndarray
        One-dimensional integer row positions with shape ``(n_valid_trials,)``.
        Selected trials are behaviorally valid and have finite choice times and
        nonmissing actions.
    """

    if "choice_time" not in trial_df.columns:
        raise ValueError("trial_df is missing required choice_time column.")
    valid_mask = spike_behavior_trials.make_trial_type_masks(trial_df)["valid"]
    finite_choice_mask = pd.to_numeric(trial_df["choice_time"], errors="coerce").notna()
    valid_choice_mask = valid_mask & finite_choice_mask & trial_df["action"].notna()
    return np.flatnonzero(np.asarray(valid_choice_mask, dtype=bool))


def select_choice_switch_events(
    trial_df: pd.DataFrame,
    pre_switch_filter: str = SWITCH_PRE_FILTER_ALL,
) -> pd.DataFrame:
    """
    Find adjacent valid trials whose choices differ and classify their outcomes.

    This definition is independent of the established ``switch`` trial mask,
    which is restricted to unrewarded trials. Here, any adjacent valid pair can
    be a switch event.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Required columns are those used by
        ``make_trial_type_masks`` plus ``choice_time`` in seconds. ``correct``,
        ``reward``, and ``action`` are scalar trial labels.
    pre_switch_filter : str, default="All choice switches"
        One value from ``SWITCH_PRE_FILTER_OPTIONS``. Outcome-specific filters
        apply only to the previous (pre-switch) trial.

    Returns
    -------
    pd.DataFrame
        One row per included switch event with columns
        ``SWITCH_EVENT_COLUMNS``. Trial indices are zero-based integer row
        positions. Incorrect-to-incorrect switches are excluded.
    """

    if pre_switch_filter not in SWITCH_PRE_FILTER_OPTIONS:
        raise ValueError(f"Unknown pre-switch filter {pre_switch_filter!r}.")
    required_columns = {"choice_time", "correct", "reward", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")

    valid_positions = select_valid_choice_trial_indices(trial_df)
    valid_position_mask = np.zeros(trial_df.shape[0], dtype=bool)
    valid_position_mask[valid_positions] = True
    correct_values = pd.to_numeric(trial_df["correct"], errors="coerce").to_numpy(dtype=float)
    reward_values = pd.to_numeric(trial_df["reward"], errors="coerce").to_numpy(dtype=float)
    action_values = trial_df["action"].to_numpy()

    event_rows: list[dict[str, int | float | str]] = []
    for previous_trial_index in range(max(0, trial_df.shape[0] - 1)):
        next_trial_index = previous_trial_index + 1
        if not (valid_position_mask[previous_trial_index] and valid_position_mask[next_trial_index]):
            continue
        if action_values[previous_trial_index] == action_values[next_trial_index]:
            continue

        previous_correct = correct_values[previous_trial_index]
        next_correct = correct_values[next_trial_index]
        if previous_correct not in (0.0, 1.0) or next_correct not in (0.0, 1.0):
            continue
        switch_type = _classify_switch_type(previous_correct, next_correct)
        if switch_type not in SWITCH_TYPE_ORDER:
            continue
        switch_direction = _classify_switch_direction(
            previous_action=action_values[previous_trial_index],
            next_action=action_values[next_trial_index],
        )
        if switch_direction is None:
            continue
        if not _matches_pre_switch_filter(
            previous_correct=previous_correct,
            previous_reward=reward_values[previous_trial_index],
            pre_switch_filter=pre_switch_filter,
        ):
            continue

        event_rows.append(
            {
                "event_id": len(event_rows),
                "previous_trial_index": previous_trial_index,
                "next_trial_index": next_trial_index,
                "previous_correct": int(previous_correct),
                "next_correct": int(next_correct),
                "previous_reward": reward_values[previous_trial_index],
                "switch_type": switch_type,
                "switch_direction": switch_direction,
            }
        )
    return pd.DataFrame(event_rows, columns=SWITCH_EVENT_COLUMNS)


def extract_switch_event_pca_trajectories(
    pca_scores: np.ndarray,
    pca_trial_indices: np.ndarray,
    switch_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract four ordered PC1/PC2 points for each switch event.

    Parameters
    ----------
    pca_scores : np.ndarray
        PCA score tensor with shape ``(n_pca_trials, 2, n_components)``. The
        second axis is the pre-choice ``[-0.5, 0]`` bin followed by the
        post-choice ``[0, 0.5]`` bin. Scores are in PCA coordinate units.
    pca_trial_indices : np.ndarray
        One-dimensional full-table trial row positions represented by the first
        axis of ``pca_scores``, shape ``(n_pca_trials,)``.
    switch_events : pd.DataFrame
        Switch-event table returned by ``select_choice_switch_events``.

    Returns
    -------
    pd.DataFrame
        Long-form table with four rows per event and columns
        ``SWITCH_TRAJECTORY_COLUMNS``. Points are ordered as previous pre,
        previous post, next pre, and next post.
    """

    scores = np.asarray(pca_scores, dtype=float)
    trial_indices = np.asarray(pca_trial_indices, dtype=int).reshape(-1)
    if scores.ndim != 3 or scores.shape[1] != 2:
        raise ValueError("pca_scores must have shape (n_trials, 2, n_components).")
    if scores.shape[0] != trial_indices.size:
        raise ValueError("pca_scores trial axis must match pca_trial_indices length.")
    if scores.shape[2] < 2:
        raise ValueError("Switch trajectories require at least two PCs.")
    if np.unique(trial_indices).size != trial_indices.size:
        raise ValueError("pca_trial_indices must contain unique trial positions.")
    missing_event_columns = {
        "event_id",
        "previous_trial_index",
        "next_trial_index",
        "switch_type",
        "switch_direction",
    } - set(switch_events.columns)
    if missing_event_columns:
        raise ValueError(f"switch_events is missing required columns: {sorted(missing_event_columns)}")

    score_index_by_trial = {
        int(trial_index): score_index for score_index, trial_index in enumerate(trial_indices)
    }
    point_definitions = (
        (0, "previous_pre_choice", "previous_trial_index", "pre_choice", 0),
        (1, "previous_post_choice", "previous_trial_index", "post_choice", 1),
        (2, "next_pre_choice", "next_trial_index", "pre_choice", 0),
        (3, "next_post_choice", "next_trial_index", "post_choice", 1),
    )
    trajectory_rows: list[dict[str, int | float | str]] = []
    for event in switch_events.itertuples(index=False):
        previous_trial_index = int(event.previous_trial_index)
        next_trial_index = int(event.next_trial_index)
        missing_trials = [
            trial_index
            for trial_index in (previous_trial_index, next_trial_index)
            if trial_index not in score_index_by_trial
        ]
        if missing_trials:
            raise ValueError(f"Switch event trials are absent from PCA scores: {missing_trials}")
        for point_order, point_label, trial_field, choice_window, bin_index in point_definitions:
            trial_index = int(getattr(event, trial_field))
            score = scores[score_index_by_trial[trial_index], bin_index, :2]
            trajectory_rows.append(
                {
                    "event_id": int(event.event_id),
                    "switch_type": str(event.switch_type),
                    "switch_direction": str(event.switch_direction),
                    "previous_trial_index": previous_trial_index,
                    "next_trial_index": next_trial_index,
                    "point_order": point_order,
                    "point_label": point_label,
                    "trial_index": trial_index,
                    "choice_window": choice_window,
                    "pc1": float(score[0]),
                    "pc2": float(score[1]),
                }
            )
    return pd.DataFrame(trajectory_rows, columns=SWITCH_TRAJECTORY_COLUMNS)


def summarize_switch_event_counts(switch_events: pd.DataFrame) -> pd.DataFrame:
    """
    Count plotted events in the fixed switch-type panel order.

    Parameters
    ----------
    switch_events : pd.DataFrame
        Switch-event table with a ``switch_type`` column and one row per event.

    Returns
    -------
    pd.DataFrame
        Table with columns ``switch_type``, ``label``, and ``n_events``. It has
        one row for each supported switch type, including zero-count types.
    """

    if "switch_type" not in switch_events.columns:
        raise ValueError("switch_events is missing required switch_type column.")
    counts = switch_events["switch_type"].value_counts()
    return pd.DataFrame(
        {
            "switch_type": list(SWITCH_TYPE_ORDER),
            "label": [SWITCH_TYPE_TITLES[switch_type] for switch_type in SWITCH_TYPE_ORDER],
            "n_events": [int(counts.get(switch_type, 0)) for switch_type in SWITCH_TYPE_ORDER],
        }
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


def _classify_switch_type(previous_correct: float, next_correct: float) -> str:
    """
    Return the correctness-transition key for one adjacent trial pair.

    Parameters
    ----------
    previous_correct, next_correct : float
        Scalar binary correctness labels for the ordered trial pair.

    Returns
    -------
    str
        Transition key formatted as ``previous_to_next``.
    """

    previous_label = "correct" if previous_correct == 1.0 else "incorrect"
    next_label = "correct" if next_correct == 1.0 else "incorrect"
    return f"{previous_label}_to_{next_label}"


def _classify_switch_direction(previous_action: object, next_action: object) -> str | None:
    """
    Classify one binary choice switch using the established action convention.

    Parameters
    ----------
    previous_action, next_action : object
        Scalar action labels. Action ``1`` is left and action ``0`` is right.

    Returns
    -------
    str | None
        ``"left_to_right"`` for ``1 -> 0``, ``"right_to_left"`` for
        ``0 -> 1``, or ``None`` when the labels are not a supported switch.
    """

    if previous_action == 1 and next_action == 0:
        return "left_to_right"
    if previous_action == 0 and next_action == 1:
        return "right_to_left"
    return None


def _matches_pre_switch_filter(
    previous_correct: float,
    previous_reward: float,
    pre_switch_filter: str,
) -> bool:
    """
    Return whether one pre-switch outcome matches the selected UI filter.

    Parameters
    ----------
    previous_correct : float
        Scalar binary correctness label for the previous trial.
    previous_reward : float
        Scalar binary reward label for the previous trial.
    pre_switch_filter : str
        One value from ``SWITCH_PRE_FILTER_OPTIONS``.

    Returns
    -------
    bool
        ``True`` when the previous trial belongs to the selected outcome group.
    """

    if pre_switch_filter == SWITCH_PRE_FILTER_ALL:
        return True
    if pre_switch_filter == SWITCH_PRE_FILTER_CORRECT_REWARDED:
        return previous_correct == 1.0 and previous_reward == 1.0
    return previous_correct == 1.0 and previous_reward == 0.0


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
