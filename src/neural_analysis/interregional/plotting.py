"""Read-only figures built exclusively from saved inter-regional result tables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from .records import InterregionalResults


_DIRECTIONS = ("HPC_to_PFC", "PFC_to_HPC")
_DIRECTION_COLORS = {"HPC_to_PFC": "#1f77b4", "PFC_to_HPC": "#d62728"}


def _filtered_targets(
    target_summaries: pd.DataFrame,
    *,
    representation: str,
    model_family: str,
    window: str,
    metric_names: tuple[str, ...],
    condition: str | None = None,
) -> pd.DataFrame:
    """Return copied saved rows matching one display-only selection."""
    required = {
        "direction",
        "representation",
        "model_family",
        "condition",
        "window",
        "target_id",
        "metric_name",
        "status",
        "mean_value",
    }
    missing = required - set(target_summaries.columns)
    if missing:
        raise ValueError(f"target_summaries is missing columns: {sorted(missing)}")
    mask = (
        target_summaries["representation"].eq(representation)
        & target_summaries["model_family"].eq(model_family)
        & target_summaries["window"].eq(window)
        & target_summaries["metric_name"].isin(metric_names)
    )
    if condition is not None:
        mask &= target_summaries["condition"].eq(condition)
    selected = target_summaries.loc[mask].copy()
    if selected.empty:
        raise ValueError("No saved target summaries match the display selection.")
    return selected


def _caption(coverage_assumption_version: str, detail: str) -> str:
    """Return the shared prediction and coverage caveat for a figure."""
    return (
        f"{detail} Values are predictive, not causal. Loaded spike coverage is assumed "
        f"complete under {coverage_assumption_version}."
    )


def _style_light_figure(figure: plt.Figure, axis: plt.Axes) -> None:
    """Apply the approved opaque light-mode presentation."""
    figure.patch.set_facecolor("white")
    figure.patch.set_alpha(1.0)
    axis.set_facecolor("white")
    axis.tick_params(colors="black", labelsize=10)
    axis.xaxis.label.set_color("black")
    axis.yaxis.label.set_color("black")
    axis.title.set_color("black")
    for spine in axis.spines.values():
        spine.set_color("black")
    axis.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)


def plot_cv_increment_summary(
    target_summaries: pd.DataFrame,
    population_summaries: pd.DataFrame,
    *,
    representation: str,
    model_family: str,
    window: str,
    coverage_assumption_version: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot saved target increments with population median and IQR overlays.

    Parameters
    ----------
    target_summaries : pandas.DataFrame
        Frozen target-summary rows. ``mean_value`` is dimensionless held-out
        incremental R-squared; one row represents one stable target identity.
    population_summaries : pandas.DataFrame
        Frozen population-summary rows containing target quartiles.
    representation, model_family, window : str
        Display-only labels that must already be present in the saved tables.
    coverage_assumption_version : str
        Saved coverage declaration shown verbatim in the caption.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Light-mode figure and primary axis. Target artists carry ``target_id``
        as their Matplotlib ``gid`` for stable inspection.
    """
    selected = _filtered_targets(
        target_summaries,
        representation=representation,
        model_family=model_family,
        window=window,
        metric_names=("delta_r2",),
    )
    conditions = tuple(dict.fromkeys(selected["condition"].astype(str)))
    figure, axis = plt.subplots(figsize=(max(8.0, 2.6 * len(conditions)), 6.5))
    _style_light_figure(figure, axis)
    direction_offsets = {"HPC_to_PFC": -0.35, "PFC_to_HPC": 0.35}
    cell_positions: list[float] = []
    all_finite = pd.to_numeric(selected["mean_value"], errors="coerce").dropna()
    text_y = float(all_finite.max()) if not all_finite.empty else 0.0
    text_y += max(float(all_finite.max() - all_finite.min()), 0.1) * 0.12
    for condition_index, condition in enumerate(conditions):
        center = condition_index * 3.0
        for direction in _DIRECTIONS:
            position = center + direction_offsets[direction]
            cell_positions.append(position)
            rows = selected.loc[
                selected["condition"].eq(condition)
                & selected["direction"].eq(direction)
            ]
            complete = rows.loc[rows["status"].eq("ok")]
            unavailable_count = int(len(rows) - len(complete))
            for target_index, row in enumerate(complete.itertuples(index=False)):
                jitter = (target_index - (len(complete) - 1) / 2.0) * 0.06
                (artist,) = axis.plot(
                    position + jitter,
                    float(row.mean_value),
                    marker="o",
                    linestyle="none",
                    markersize=5.5,
                    color=_DIRECTION_COLORS[direction],
                    alpha=0.8,
                )
                artist.set_gid(str(row.target_id))
            summary = population_summaries.loc[
                population_summaries["representation"].eq(representation)
                & population_summaries["model_family"].eq(model_family)
                & population_summaries["window"].eq(window)
                & population_summaries["metric_name"].eq("delta_r2")
                & population_summaries["condition"].eq(condition)
                & population_summaries["direction"].eq(direction)
                & population_summaries["status"].eq("ok")
            ]
            if len(summary) == 1:
                item = summary.iloc[0]
                axis.vlines(
                    position,
                    float(item["q25"]),
                    float(item["q75"]),
                    color="black",
                    linewidth=3.0,
                    zorder=3,
                )
                axis.plot(
                    [position - 0.13, position + 0.13],
                    [float(item["median"]), float(item["median"])],
                    color="black",
                    linewidth=2.0,
                    zorder=4,
                )
            axis.text(
                position,
                text_y,
                f"{len(complete)} contributing\n{unavailable_count} unavailable",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )
    axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axis.set_xticks([index * 3.0 for index in range(len(conditions))], conditions)
    axis.set_ylabel("Incremental CV R-squared")
    axis.set_xlabel("Condition (blue: HPC to PFC; red: PFC to HPC)")
    axis.set_title(f"Held-out incremental prediction: {representation}, {window}")
    figure.text(
        0.5,
        0.01,
        _caption(
            coverage_assumption_version,
            "Points are complete target means; black marks show population median and IQR.",
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.18, top=0.88)
    return figure, axis


def plot_absolute_cv_scores(
    target_summaries: pd.DataFrame,
    *,
    representation: str,
    model_family: str,
    condition: str,
    window: str,
    coverage_assumption_version: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot saved restricted/full absolute held-out R-squared values.

    Input rows are frozen target summaries and values are dimensionless. The
    returned figure contains no estimator or refitting behavior.
    """
    selected = _filtered_targets(
        target_summaries,
        representation=representation,
        model_family=model_family,
        condition=condition,
        window=window,
        metric_names=("r2_restricted", "r2_full"),
    )
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    _style_light_figure(figure, axis)
    metric_positions = {"r2_restricted": 0.0, "r2_full": 1.0}
    direction_offsets = {"HPC_to_PFC": -0.08, "PFC_to_HPC": 0.08}
    for row in selected.loc[selected["status"].eq("ok")].itertuples(index=False):
        x_value = metric_positions[str(row.metric_name)] + direction_offsets[str(row.direction)]
        (artist,) = axis.plot(
            x_value,
            float(row.mean_value),
            marker="o",
            linestyle="none",
            color=_DIRECTION_COLORS[str(row.direction)],
            alpha=0.75,
        )
        artist.set_gid(str(row.target_id))
    axis.set_xticks([0.0, 1.0], ["Restricted", "Full"])
    axis.set_ylabel("Absolute held-out CV R-squared")
    axis.set_title(f"Absolute prediction scores: {condition}, {window}, {representation}")
    figure.text(
        0.5,
        0.01,
        _caption(
            coverage_assumption_version,
            "No model was refit; points are complete means read from the saved result.",
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )
    figure.subplots_adjust(left=0.14, right=0.97, bottom=0.17, top=0.88)
    return figure, axis


def save_standard_regression_figures(
    result: InterregionalResults,
    figures_directory: Path | str,
) -> tuple[Path, ...]:
    """Save all applicable standard OLS inspection figures as opaque PNGs.

    Parameters
    ----------
    result : InterregionalResults
        Valid saved-result record. Only target and population summary tables are
        read; no model is reconstructed or fit.
    figures_directory : pathlib.Path or str
        Existing output directory. PNG dimensions are controlled by the
        plotting functions' figure sizes and saved at 150 dots per inch.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Saved PNG paths in deterministic representation/model/window/condition
        order. An empty tuple means no applicable summary rows were present.
    """
    destination = Path(figures_directory)
    if not destination.is_dir():
        raise ValueError("figures_directory must be an existing directory.")
    summaries = result.target_summaries
    if summaries.empty:
        return ()
    saved: list[Path] = []
    combinations = (
        summaries[["representation", "model_family", "window"]]
        .drop_duplicates()
        .sort_values(["representation", "model_family", "window"])
    )
    for combination in combinations.itertuples(index=False):
        cell = summaries.loc[
            summaries["representation"].eq(combination.representation)
            & summaries["model_family"].eq(combination.model_family)
            & summaries["window"].eq(combination.window)
        ]
        if cell["metric_name"].eq("delta_r2").any() and not result.population_summaries.empty:
            figure, _ = plot_cv_increment_summary(
                summaries,
                result.population_summaries,
                representation=str(combination.representation),
                model_family=str(combination.model_family),
                window=str(combination.window),
                coverage_assumption_version=result.coverage_assumption_version,
            )
            path = destination / (
                f"cv_increment_{combination.representation}_"
                f"{combination.model_family}_{combination.window}.png"
            )
            figure.savefig(path, dpi=150, facecolor="white", transparent=False)
            plt.close(figure)
            saved.append(path)
        absolute_metrics = set(cell["metric_name"].astype(str))
        if {"r2_restricted", "r2_full"} <= absolute_metrics:
            for condition in sorted(set(cell["condition"].astype(str))):
                figure, _ = plot_absolute_cv_scores(
                    summaries,
                    representation=str(combination.representation),
                    model_family=str(combination.model_family),
                    condition=condition,
                    window=str(combination.window),
                    coverage_assumption_version=result.coverage_assumption_version,
                )
                path = destination / (
                    f"cv_absolute_{condition}_{combination.representation}_"
                    f"{combination.model_family}_{combination.window}.png"
                )
                figure.savefig(path, dpi=150, facecolor="white", transparent=False)
                plt.close(figure)
                saved.append(path)
    return tuple(saved)
