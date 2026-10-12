"""Light-mode figures for deterministic behavior model benchmark reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.behavior_modeling.benchmark_models import ModelReplay
from src.behavior_modeling.benchmark_scenarios import BehaviorScenario


def plot_model_behavior_atlas(
    scenarios: Sequence[BehaviorScenario],
    replays: Mapping[tuple[str, str], ModelReplay],
    model_id: str,
    display_label: str,
    parameters: Mapping[str, float | int | str],
) -> tuple[Figure, ...]:
    """Plot observed behavior and one model's value for every selected sample.

    Parameters
    ----------
    scenarios : sequence of BehaviorScenario
        One to 22 prescribed samples with aligned trial arrays.
    replays : mapping
        Benchmark replays keyed by ``(model_id, scenario_id)``.
    model_id, display_label : str
        Stable identifier and human-readable model label.
    parameters : mapping
        Fixed unit-bearing or unitless model parameter values for the caption.

    Returns
    -------
    tuple of matplotlib.figure.Figure
        One wide page for at most 12 samples, otherwise two balanced pages.
        Each sample occupies one axis; no additional diagnostic panels exist.
    """
    scenario_tuple = tuple(scenarios)
    if not scenario_tuple:
        raise ValueError("At least one behavior scenario is required.")
    if len(scenario_tuple) > 24:
        raise ValueError("The initial atlas supports at most 24 selected samples.")
    for scenario in scenario_tuple:
        if (model_id, scenario.scenario_id) not in replays:
            raise ValueError(
                f"Missing replay for model {model_id!r} and "
                f"scenario {scenario.scenario_id!r}."
            )

    pages = _balanced_pages(scenario_tuple)
    figures = []
    for page_index, page_scenarios in enumerate(pages, start=1):
        figure = _plot_atlas_page(
            page_scenarios,
            replays,
            model_id,
            display_label,
            parameters,
            page_index,
            len(pages),
        )
        figures.append(figure)
    return tuple(figures)


def plot_combined_sample_agreement(
    score_table: pd.DataFrame,
    model_labels: Mapping[str, str],
) -> Figure:
    """Plot sample-as-block greedy agreement for all selected models.

    Parameters
    ----------
    score_table : pandas.DataFrame
        One row per model/sample pair with ``model_id``, ``scenario_id``, and
        unitless ``greedy_agreement`` in ``[0, 1]``.
    model_labels : mapping from str to str
        Human-readable label for every model ID in the table.

    Returns
    -------
    matplotlib.figure.Figure
        One opaque light-mode figure. Samples follow first-seen table order.
    """
    required = {"model_id", "scenario_id", "greedy_agreement"}
    missing = required.difference(score_table.columns)
    if missing:
        raise ValueError(f"Score table is missing columns: {sorted(missing)}")
    if score_table.empty:
        raise ValueError("Score table must contain at least one model/sample row.")

    scenario_ids = pd.unique(score_table["scenario_id"]).tolist()
    model_ids = pd.unique(score_table["model_id"]).tolist()
    figure, axis = plt.subplots(figsize=(24, 7), facecolor="white")
    axis.set_facecolor("white")
    x_positions = np.arange(len(scenario_ids), dtype=float)
    for model_id in model_ids:
        if model_id not in model_labels:
            raise ValueError(f"Missing display label for model {model_id!r}.")
        model_rows = score_table.loc[score_table["model_id"] == model_id]
        values_by_scenario = model_rows.set_index("scenario_id")[
            "greedy_agreement"
        ]
        values = values_by_scenario.reindex(scenario_ids).to_numpy(dtype=float)
        axis.plot(
            x_positions,
            values,
            marker="o",
            linewidth=1.0,
            markersize=4,
            label=model_labels[model_id],
        )
    axis.axhline(
        0.5,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label="0.5 reference",
    )
    axis.set_xticks(x_positions, scenario_ids, rotation=45, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Sample-as-block greedy agreement")
    axis.set_xlabel("Behavior sample")
    axis.set_title("Fixed-model correspondence across prescribed behaviors")
    axis.grid(axis="y", color="0.9", linewidth=0.6)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    figure.text(
        0.01,
        0.01,
        "Each complete behavior sample is one block; exact policy ties "
        "contribute 0.5.",
        fontsize=9,
        color="black",
    )
    figure.tight_layout(rect=(0.0, 0.04, 0.84, 1.0))
    return figure


def save_benchmark_figure(figure: Figure, output_path: Path) -> None:
    """Save one benchmark page as a 300-dpi PNG and close only that figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    output_path : pathlib.Path
        Existing-parent PNG path selected by the report runner.

    Returns
    -------
    None
    """
    path = Path(output_path)
    try:
        figure.savefig(
            path,
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# Helpers


def _balanced_pages(
    scenarios: tuple[BehaviorScenario, ...],
) -> tuple[tuple[BehaviorScenario, ...], ...]:
    """Split at most 24 samples into one or two nearly equal pages."""
    if len(scenarios) <= 12:
        return (scenarios,)
    first_page_size = ceil(len(scenarios) / 2)
    return (scenarios[:first_page_size], scenarios[first_page_size:])


def _plot_atlas_page(
    scenarios: tuple[BehaviorScenario, ...],
    replays: Mapping[tuple[str, str], ModelReplay],
    model_id: str,
    display_label: str,
    parameters: Mapping[str, float | int | str],
    page_index: int,
    page_count: int,
) -> Figure:
    """Create one behavior-atlas page for a single fixed model."""
    n_columns = min(4, len(scenarios))
    n_rows = ceil(len(scenarios) / n_columns)
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(24, 12),
        squeeze=False,
        facecolor="white",
    )
    flat_axes = axes.ravel()
    for axis, scenario in zip(flat_axes, scenarios):
        replay = replays[(model_id, scenario.scenario_id)]
        _plot_sample_axis(axis, scenario, replay)
    for unused_axis in flat_axes[len(scenarios) :]:
        figure.delaxes(unused_axis)

    parameter_text = ", ".join(
        f"{name}={value}" for name, value in parameters.items()
    )
    if not parameter_text:
        parameter_text = "no free parameters"
    figure.suptitle(
        f"{display_label} ({model_id}) - page {page_index}/{page_count}",
        fontsize=18,
        color="black",
    )
    figure.text(
        0.01,
        0.01,
        "Thin line: P(left)-P(right). Circles: observed choices; larger circles "
        f"were rewarded. Dashed lines mark context changes. {parameter_text}",
        fontsize=9,
        color="black",
    )
    legend_entries = [
        (handle, label)
        for handle, label in zip(*figure.axes[0].get_legend_handles_labels())
        if label != "Model value (single trial)"
    ]
    handles = [entry[0] for entry in legend_entries]
    labels = [entry[1] for entry in legend_entries]
    figure.legend(handles, labels, loc="upper right", frameon=False)
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    return figure


def _plot_sample_axis(
    axis: plt.Axes,
    scenario: BehaviorScenario,
    replay: ModelReplay,
) -> None:
    """Overlay one aligned behavior sample and model value on one axis."""
    axis.set_facecolor("white")
    trial_index = np.arange(scenario.action.size)
    axis.plot(
        trial_index,
        replay.model_value,
        color="#355f8a",
        linewidth=1.0,
        label="Model value",
        zorder=2,
    )
    if scenario.action.size == 1 and scenario.valid[0]:
        axis.scatter(
            trial_index,
            replay.model_value,
            marker="_",
            s=90,
            color="#355f8a",
            linewidths=1.2,
            label="Model value (single trial)",
            zorder=3,
        )
    signed_choice = np.where(scenario.action == 1, 1.0, -1.0)
    marker_sizes = np.where(scenario.reward > 0, 52.0, 20.0)
    axis.scatter(
        trial_index[scenario.valid],
        signed_choice[scenario.valid],
        s=marker_sizes[scenario.valid],
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
        label="Observed choice (larger = rewarded)",
        zorder=4,
    )
    _mark_context(axis, scenario)
    axis.set_xlim(-0.5, max(0.5, scenario.action.size - 0.5))
    axis.set_ylim(-1.1, 1.1)
    axis.set_yticks([-1.0, 0.0, 1.0], ["R", "0", "L"])
    axis.set_xlabel("Trial")
    axis.set_ylabel("Choice / model value")
    axis.set_title(scenario.scenario_id)
    axis.grid(axis="y", color="0.9", linewidth=0.5)


def _mark_context(axis: plt.Axes, scenario: BehaviorScenario) -> None:
    """Label the initial hidden context and each transition without a trace."""
    valid_indices = np.flatnonzero(scenario.valid)
    if valid_indices.size == 0:
        return
    first_index = int(valid_indices[0])
    previous_side = int(scenario.correct_side[first_index])
    axis.text(
        first_index,
        0.88,
        "L" if previous_side == 1 else "R",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        zorder=5,
    )
    for index in valid_indices[1:]:
        side = int(scenario.correct_side[index])
        if side == previous_side:
            continue
        axis.axvline(
            index - 0.5,
            color="0.35",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )
        axis.text(
            index,
            0.88,
            "L" if side == 1 else "R",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=5,
        )
        previous_side = side
