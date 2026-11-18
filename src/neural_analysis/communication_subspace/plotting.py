"""Descriptive held-out rank curves from a saved communication result."""

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .results import CommunicationResult


_DIRECTION_STYLE = {
    "PFC->PFC": ("Within PFC", "#176B87", "-"),
    "PFC->HPC": ("Cross to HPC", "#CC6B35", "--"),
    "HPC->HPC": ("Within HPC", "#176B87", "-"),
    "HPC->PFC": ("Cross to PFC", "#CC6B35", "--"),
}


def plot_rank_curves(result: "CommunicationResult") -> Figure:
    """Plot saved mean/draw R2 from a CommunicationResult; return a Figure.

    result.computation.draw_scores and curve_scores have one dimensionless
    held-out mean-R2 row per complete draw or rank curve. Source direction
    determines a panel; RRR ranks are integer coefficient ranks, while ridge
    has no rank. No fit or file I/O occurs. The caller owns the returned figure
    and should close it after display.
    """
    computation = result.computation
    draw_scores = computation.draw_scores
    curve_scores = computation.curve_scores
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True,
                                facecolor="white", constrained_layout=False)
    for axis, source_region in zip(axes, ("PFC", "HPC")):
        axis.set_facecolor("white")
        axis.set_title(f"{source_region} source", fontsize=15, color="black")
        axis.set_xlabel("Coefficient rank", fontsize=12, color="black")
        axis.set_ylabel(r"Held-out mean per-neuron $R^2$", fontsize=12, color="black")
        axis.tick_params(colors="black", labelsize=11)
        for spine in axis.spines.values():
            spine.set_color("black")
        axis.axhline(0, color="black", linewidth=0.9, alpha=0.6, label="_nolegend_")
        for direction in (f"{source_region}->{source_region}",
                          f"{source_region}->{'HPC' if source_region == 'PFC' else 'PFC'}"):
            _plot_direction(axis, draw_scores, curve_scores, direction)
        all_ranks = pd.to_numeric(curve_scores.loc[
            (curve_scores["direction"].str.startswith(source_region + "->")) &
            (curve_scores["model"] == "rrr"), "rank"], errors="coerce") if not curve_scores.empty else []
        finite_ranks = sorted({int(value) for value in all_ranks if pd.notna(value)})
        # Keep all curve points but limit labels for larger recorded populations.
        tick_indices = np.linspace(0, len(finite_ranks) - 1, min(9, len(finite_ranks)), dtype=int)
        axis.set_xticks([finite_ranks[index] for index in tick_indices] or [1])
        axis.set_xlim(0.75, max(finite_ranks or [1]) + 0.25)
        axis.grid(axis="y", color="#E4E4E4", linewidth=0.7)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, loc="upper left", fontsize=8, frameon=False)

    caption = _caption(result)
    figure.text(0.03, 0.025, caption, fontsize=9, color="black", va="bottom",
                ha="left", wrap=True)
    figure.subplots_adjust(left=0.075, right=0.98, top=0.91, bottom=0.29, wspace=0.12)
    return figure


def save_rank_figure(result: "CommunicationResult", path: Path) -> Path:
    """Save a result's two-panel PNG to path and close the temporary Figure.

    result supplies dimensionless rank R2 tables; path is a local filesystem
    Path to a PNG file. Return the same Path after a 180-dpi opaque export.
    The caller creates the parent directory and owns overwrite policy.
    """
    output_path = Path(path)
    figure = plot_rank_curves(result)
    try:
        figure.savefig(output_path, dpi=180, facecolor="white", transparent=False)
    finally:
        plt.close(figure)
    return output_path


# Helper functions
def _plot_direction(axis, draw_scores: pd.DataFrame, curve_scores: pd.DataFrame,
                    direction: str) -> None:
    """Add one within/cross direction's dimensionless draw and mean R2 lines."""
    if draw_scores.empty or curve_scores.empty:
        return
    title, color, line_style = _DIRECTION_STYLE[direction]
    draw_rows = draw_scores.loc[draw_scores["direction"] == direction]
    curve_rows = curve_scores.loc[curve_scores["direction"] == direction]
    rrr_draws = draw_rows.loc[(draw_rows["model"] == "rrr") &
                               (draw_rows["status"] == "ok")]
    for _, one_draw in rrr_draws.groupby("draw", sort=True):
        one_draw = one_draw.sort_values("rank")
        axis.plot(one_draw["rank"].to_numpy(dtype=float),
                  one_draw["mean_r2"].to_numpy(dtype=float), color=color,
                  linestyle=line_style, linewidth=1, alpha=0.25, label="_nolegend_")

    rrr_curves = curve_rows.loc[curve_rows["model"] == "rrr"].sort_values("rank")
    if not rrr_curves.empty:
        complete_means = rrr_curves["mean_r2"].to_numpy(dtype=float).copy()
        complete_means[rrr_curves["status"].to_numpy() != "ok"] = np.nan
        axis.plot(rrr_curves["rank"].to_numpy(dtype=float),
                  complete_means, color=color,
                  linestyle=line_style, linewidth=2.5, marker="o", markersize=5,
                  label=f"{title} RRR")
    baseline = curve_rows.loc[curve_rows["model"] == "ridge"]
    if not baseline.empty and np.isfinite(baseline["mean_r2"].iloc[0]):
        max_rank = int(rrr_curves["rank"].max()) if not rrr_curves.empty else 1
        axis.plot([1, max_rank], [baseline["mean_r2"].iloc[0]] * 2,
                  color=color, linestyle=":", linewidth=1.8,
                  label=f"{title} ridge ({int(baseline['n_draws'].iloc[0])} draws)")
    _plot_incomplete_means(axis, rrr_curves, title, color)
    _plot_paired_incomplete(axis, draw_rows, rrr_curves, direction, title, color)


def _plot_incomplete_means(axis, rrr_curves: pd.DataFrame, title: str, color: str) -> None:
    """Draw incomplete means on a Matplotlib Axes; return None.

    ``rrr_curves`` is a DataFrame with one row per integer coefficient rank,
    containing status, dimensionless mean_r2, and n_draws. ``title`` and
    ``color`` are strings used for legend labels and Matplotlib styling.
    """
    labeled = False
    for curve in rrr_curves.itertuples():
        if curve.status == "ok" or not np.isfinite(curve.mean_r2):
            continue
        rank = int(curve.rank)
        label = (f"{title} incomplete mean rank {rank} ({int(curve.n_draws)} draws)"
                 if not labeled else "_nolegend_")
        axis.plot([rank], [float(curve.mean_r2)], marker="o", linestyle="None",
                  color=color, markerfacecolor="white", markeredgewidth=1.5,
                  markersize=9, label=label)
        labeled = True


def _plot_paired_incomplete(axis, draw_rows: pd.DataFrame, rrr_curves: pd.DataFrame,
                            direction: str, title: str, color: str) -> None:
    """Mark incomplete ranks using exact common complete-draw IDs for both models."""
    ridge_rows = draw_rows.loc[(draw_rows["model"] == "ridge") &
                                (draw_rows["status"] == "ok")]
    ridge_by_draw = {int(row.draw): float(row.mean_r2) for row in ridge_rows.itertuples()
                     if np.isfinite(row.mean_r2)}
    labeled = False
    for curve in rrr_curves.itertuples():
        if curve.status == "ok":
            continue
        rank = int(curve.rank)
        rank_rows = draw_rows.loc[(draw_rows["model"] == "rrr") &
                                  (draw_rows["rank"] == rank) &
                                  (draw_rows["status"] == "ok")]
        rank_by_draw = {int(row.draw): float(row.mean_r2) for row in rank_rows.itertuples()
                        if np.isfinite(row.mean_r2)}
        common_draws = tuple(sorted(rank_by_draw.keys() & ridge_by_draw.keys()))
        if not common_draws:
            continue
        paired_rrr = float(np.mean([rank_by_draw[draw] for draw in common_draws]))
        paired_ridge = float(np.mean([ridge_by_draw[draw] for draw in common_draws]))
        count = len(common_draws)
        suffix = f"rank {rank} ({count} draws)"
        rrr_label = f"{title} paired RRR {suffix}" if not labeled else "_nolegend_"
        ridge_label = f"{title} paired ridge {suffix}" if not labeled else "_nolegend_"
        axis.plot([rank], [paired_rrr], marker="D", linestyle="None", color=color,
                  markersize=8, markerfacecolor="white", markeredgewidth=1.5,
                  label=rrr_label)
        axis.plot([rank], [paired_ridge], marker="s", linestyle="None", color=color,
                  markersize=7, markerfacecolor="white", markeredgewidth=1.5,
                  label=ridge_label)
        labeled = True


def _caption(result: "CommunicationResult") -> str:
    """Return a plain-language caption for dimensionless held-out R2 panels."""
    meta = result.meta
    session_id = getattr(result, "session_id", None) or meta.get("session_id", "unknown session")
    configuration = meta.get("configuration", {})
    bin_width = configuration.get("bin_width_ms", meta.get("bin_width_ms", "unknown"))
    fold_scores = getattr(result.computation, "fold_scores", None)
    if fold_scores is None:
        valid_targets = "not recorded"
    elif "valid" in fold_scores:
        valid_targets = str(int(fold_scores["valid"].sum()))
    else:
        valid_targets = "0" if fold_scores.empty else "not recorded"
    choice_positions = getattr(result, "choice_positions", None)
    unit_ids = getattr(result, "unit_ids", None)
    if choice_positions is not None and unit_ids is not None:
        matched_units = min(len(unit_ids["PFC"]), len(unit_ids["HPC"])) // 2
        population_counts = (f"{len(choice_positions)} eligible choices, "
                             f"{matched_units} matched units per population. ")
    else:
        population_counts = ""
    status = getattr(result.computation, "status", "unavailable")
    detail = (f"{valid_targets} valid target scores. Status: {status}. "
              "Incomplete rank markers pair RRR and ridge over the same complete draws.")
    if result.computation.curve_scores.empty:
        detail = "Results unavailable: no complete rank curves or ridge baseline. " + detail
    return (f"{session_id} | -2 to +2 s choice window | {bin_width} ms bins | "
            "raw activity (no condition mean subtraction). " + population_counts +
            "Five chronological folds "
            "and five population draws; faint lines show draws, not confidence intervals.\n"
            "Negative R2 means worse prediction than the held-out mean. " + detail)
