"""Compatibility facade retaining population-decoding plots until R3."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.population.decoding import (
    PCA_DECODING_MODE_EXPLORATORY,
    PCA_DECODING_MODE_RIGOROUS,
    PCA_DECODING_MODE_OPTIONS,
    PCA_DECODING_BASE_CONDITIONS,
    PCA_DECODING_WINDOWS,
    PCA_DECODING_BIN_SIZE_S,
    PCA_DECODING_WINDOW,
    PCA_DECODING_DISPLAY_COLUMNS,
    PCA_SCORE_SUMMARY_COLUMNS,
    PCA_RAW_SCORE_COLUMNS,
    build_choice_aligned_rate_tensor,
    select_pca_decoding_trial_indices,
    build_valid_base_condition_masks,
    build_exploratory_pca_decoder_trial_bins,
    run_exploratory_pca_choice_decoding,
    run_rigorous_pca_choice_decoding,
    cv_decodeability_score_with_pca_pipeline,
    summarize_pca_decoding_results,
    summarize_pca_scores_by_condition_and_target,
    extract_pca_score_points_by_condition_and_target,
    format_pca_score_target_label,
)


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
