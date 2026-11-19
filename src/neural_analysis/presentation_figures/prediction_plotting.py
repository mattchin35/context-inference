"""Render the saved regional population-prediction results in Figures 5 and 8."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.neural_analysis.communication_subspace.configuration import DRAW_COUNT


PANEL_LABELS = {"A": "Source units", "B": "Source PCs",
                "C": "Rank with source units"}
MODEL_STYLES = {"ridge": ("Ridge", "-"), "rrr": ("Reduced rank", "-"),
                "ridge_units": ("All-unit ridge", "--"), "ridge_10pc": ("Ten-PC ridge", ":"),
                "ridge_30pc": ("30-PC ridge", ":")}
PC_REFERENCE_COUNTS = {"ridge_10pc": 10, "ridge_30pc": 30}
DIRECTION_COLORS = {"PFC->PFC": "#246C9A", "HPC->PFC": "#D87439",
                    "HPC->HPC": "#188274", "PFC->HPC": "#9860AE"}
TARGET_COLORS = {"PFC": "#246C9A", "HPC": "#188274"}


def render_prediction(fields: dict, meta: dict, theme: str) -> dict[str, Figure]:
    """Return Figure 5 from a saved session result with dimensionless R2 curves.

    fields contains score/range/support rows at integer counts or ranks and
    choice/unit counts. meta identifies the session. theme names the caller's
    active Matplotlib theme. No fits or I/O occur.
    """
    curves = fields["curves"]
    panel_d_pcs = _panel_d_count(fields, meta, curves)
    figure = plt.figure(figsize=(13.5, 9))
    grid = figure.add_gridspec(nrows=2, ncols=2, hspace=0.48, wspace=0.22)
    for location, panel in enumerate("ABCD"):
        label = _panel_label(panel, panel_d_pcs)
        axis, support_axis = _paired_axes(figure, grid[location // 2, location % 2], panel)
        panel_rows = curves.loc[curves.panel.eq(panel)]
        for (direction, model), points in panel_rows.groupby(["direction", "model"], sort=False):
            points = points.sort_values("count")
            color = DIRECTION_COLORS[direction]
            main = model == _main_model(panel)
            _draw_line(axis, points, model, color, gid=f"{panel}:{direction}:{model}", primary=main)
            axis.fill_between(points["count"].to_numpy(), points.min_r2.to_numpy(dtype=float),
                              points.max_r2.to_numpy(dtype=float), color=color, alpha=0.13 if main else 0.035)
        main_rows = panel_rows.loc[panel_rows.model.eq(_main_model(panel))]
        for target, points in main_rows.groupby("target_region", sort=False):
            points = points.drop_duplicates("count").sort_values("count")
            support_axis.plot(points["count"], points.n_draws, color=TARGET_COLORS[target],
                              linestyle="--" if target == "PFC" else "-",
                              marker=".", markersize=3, gid=target,
                              zorder=3 if target == "PFC" else 2)
        _label_axes(axis, support_axis, f"{panel}  {label}", _x_label(panel), "Draws", DRAW_COUNT)
        tested_counts = sorted(main_rows["count"].unique())
        if 0 < len(tested_counts) <= 10:
            support_axis.set_xticks(tested_counts)
        _mark_unavailable(axis, main_rows)

    session_id = meta.get("session_id", "Unknown session")
    figure.suptitle(f"Figure 5. Regional population prediction | {session_id}", y=0.985, fontsize=19)
    handles = [Line2D([], [], color=color, label=_direction_text(direction), linewidth=2)
               for direction, color in DIRECTION_COLORS.items()]
    handles.extend(_model_handles(panel_d_pcs))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                  ncols=7, fontsize=9, frameon=False)
    available = ", ".join(f"{row.region} {row.n_units}" for row in fields["available_units"].itertuples())
    n_choices = len(fields["row_positions"])
    n_units = fields["source_pool_size"]
    neuron_support = ""
    scored = curves.loc[curves.n_draws.gt(0)]
    if not scored.empty and {"min_neurons", "max_neurons"} <= set(scored):
        least = int(scored.min_neurons.min())
        most = int(scored.max_neurons.max())
        support_label = str(least) if least == most else f"{least}-{most}"
        neuron_support = f" Defined R\u00b2 uses {support_label} targets per draw."
    caption = (f"{n_choices} choices aligned from -2 to +2 s in 100 ms bins. Good units available: "
               f"{available}. Each draw uses {n_units} source and {n_units} target units per region.\n"
               "Held-out per-neuron R\u00b2 compares same-bin spike-count prediction with the training-fold mean. "
               "Curves show draw means and shaded ranges.\n"
               "Each rank and its references use the same draws. Support strips show matched draw counts "
               "for the PFC (blue) and HPC (teal) target pairs." + neuron_support)
    figure.text(0.08, 0.025, textwrap.fill(caption.replace("\n", " "), width=145),
                fontsize=10, va="bottom")
    figure.subplots_adjust(left=0.10, right=0.985, top=0.86, bottom=0.17)
    return {"figure5": figure}


def render_prediction_cohort(fields: dict, meta: dict, theme: str) -> dict[str, Figure]:
    """Return two Figure 8 Figures with supported saved session R2 and roster medians.

    fields contains per-session dimensionless curves, the count of sessions
    contributing at each integer x, roster medians and the intended session
    roster. theme names the caller's style. No fits or I/O occur.
    """
    curves = fields.get("curves", pd.DataFrame(columns=[
        "panel", "direction", "model", "session_id", "source_pool_size", "mean_r2", "reason"]))
    medians = fields.get("prediction_medians", pd.DataFrame(columns=["panel", "direction", "model"]))
    roster = fields["roster"]
    panel_d_pcs = _panel_d_count(fields, meta, curves)
    session_ids = tuple(roster.session_id)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    session_colors = {session: colors[index % len(colors)] for index, session in enumerate(session_ids)}
    figures = {}
    for panels, kind in (("AB", "counts"), ("CD", "ranks")):
        figure = plt.figure(figsize=(13.8, 13))
        grid = figure.add_gridspec(nrows=4, ncols=2, hspace=0.63, wspace=0.20)
        for row, direction in enumerate(DIRECTION_COLORS):
            for column, panel in enumerate(panels):
                axis, support_axis = _paired_axes(figure, grid[row, column], f"{panel}:{direction}")
                points = curves.loc[curves.panel.eq(panel) & curves.direction.eq(direction)]
                support = medians.loc[medians.panel.eq(panel) & medians.direction.eq(direction)]
                _cohort_curves(axis, support_axis, points, support, session_colors, _main_model(panel))
                title = f"{panel}  {_panel_label(panel, panel_d_pcs)} | {_direction_text(direction)}"
                _label_axes(axis, support_axis, title, _x_label(panel), "Sessions", len(session_ids))
                _mark_unavailable(axis, points.loc[points.model.eq(_main_model(panel))])

        title = "source counts" if kind == "counts" else "regression rank"
        subject = meta.get("mouse_id") or meta.get("subject_id") or "Cohort"
        figure.suptitle(f"Figure 8. Regional prediction by {title} | {subject}", y=0.985, fontsize=19)
        _cohort_legend(figure, curves, session_colors, panel_d_pcs, include_ranks=kind == "ranks")
        _cohort_caption(figure, roster, curves, panels, panel_d_pcs)
        figure.subplots_adjust(left=0.10, right=0.985, top=0.89, bottom=0.13)
        figures[f"figure8_{kind}"] = figure
    return figures


def _cohort_curves(
    axis: Axes, support_axis: Axes, points: pd.DataFrame, medians: pd.DataFrame,
    session_colors: dict[str, str], main_model: str,
) -> None:
    """Plot saved per-session and full-roster R2 plus pointwise integer session support."""
    for (session_id, model), series in points.groupby(["session_id", "model"], sort=False):
        _draw_line(axis, series.sort_values("count"), model, session_colors[session_id],
                   gid=f"{session_id}:{model}", primary=model == main_model, thick=False)
    median_color = plt.rcParams["axes.labelcolor"]
    for model, series in medians.groupby("model", sort=False):
        series = series.sort_values("count").copy()
        if model == main_model:
            support_axis.plot(series["count"], series.n_sessions, marker=".", markersize=3,
                              color=median_color)
        series.loc[series.n_sessions.ne(series.roster_size), "mean_r2"] = np.nan
        _draw_line(axis, series, model, median_color, gid=f"median:{model}", primary=model == main_model)


def _draw_line(
    axis: Axes, points: pd.DataFrame, model: str, color: str, *, gid: str,
    primary: bool, thick: bool = True,
) -> None:
    """Plot integer x and dimensionless R2, retaining gaps as NaN."""
    axis.plot(points["count"].to_numpy(), points.mean_r2.to_numpy(dtype=float),
              color=color, linestyle=MODEL_STYLES[model][1], gid=gid,
              linewidth=2.2 if thick and primary else 1.1, alpha=1 if primary else 0.85,
              marker="o" if primary else None, markersize=2.8)


def _cohort_legend(
    figure: Figure, curves: pd.DataFrame, session_colors: dict[str, str], panel_d_pcs: int,
    *, include_ranks: bool,
) -> None:
    """Add a stable session-ID legend with matched source pool sizes on rank slides."""
    handles = []
    for session_id, color in session_colors.items():
        pools = curves.loc[curves.session_id.eq(session_id), "source_pool_size"].unique()
        label = f"{session_id} (n={pools[0]})" if include_ranks and len(pools) else str(session_id)
        handles.append(Line2D([], [], color=color, label=label, linewidth=1.3))
    handles.append(Line2D([], [], color=plt.rcParams["axes.labelcolor"],
                          label="Full-roster median", linewidth=2.2))
    uses_long_labels = sum(len(handle.get_label()) for handle in handles) > 140
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.956),
                  ncols=min(len(handles), 3 if uses_long_labels else 5), fontsize=9, frameon=False)
    if include_ranks:
        model_y = 0.911 if uses_long_labels else 0.93
        figure.legend(handles=_model_handles(panel_d_pcs), loc="upper center", bbox_to_anchor=(0.5, model_y),
                      ncols=3, fontsize=9, frameon=False)


def _cohort_caption(
    figure: Figure, roster: pd.DataFrame, curves: pd.DataFrame, panels: str, panel_d_pcs: int,
) -> None:
    """Caption the held-out scores, roster failures and unavailable session panels."""
    caption = ("Same-bin prediction in 100 ms bins from -2 to +2 s around choices. Held-out per-neuron "
               "R\u00b2 uses each training fold's mean count as its baseline.\n"
               "Thin traces show each session mean across matched draws. Support strips show sessions "
               "contributing at each count or rank.")
    missing = [f"{row.session_id} ({row.reason})" for row in roster.itertuples() if row.reason]
    panel_rows = curves.loc[curves.panel.isin(list(panels))]
    for (session_id, panel), points in panel_rows.groupby(["session_id", "panel"], sort=False):
        if not np.isfinite(points.mean_r2.to_numpy(dtype=float)).any():
            if panel == "D" and points.source_pool_size.iloc[0] < panel_d_pcs:
                reason = f"only {points.source_pool_size.iloc[0]} sampled source units"
            else:
                reason = next((str(reason) for reason in points.reason if reason), "no supported scores")
            missing.append(f"{session_id}, panel {panel} ({reason})")
    if missing:
        caption += "\n" + textwrap.fill("Roster notes. " + ". ".join(missing), width=145)
    figure.text(0.08, 0.025, caption, fontsize=9, va="bottom")


def _paired_axes(figure: Figure, slot: SubplotSpec, label: str) -> tuple[Axes, Axes]:
    """Return linked R2 and support-strip Axes inside a Matplotlib SubplotSpec."""
    grid = slot.subgridspec(2, 1, height_ratios=(3.8, 1), hspace=0.17)
    axis = figure.add_subplot(grid[0], gid=label)
    support = figure.add_subplot(grid[1], sharex=axis, gid=f"{label}:support")
    return axis, support


def _label_axes(axis: Axes, support: Axes, title: str, x_label: str, name: str, total: int) -> None:
    """Label an R2 axis and its aligned support strip with a zero-to-total count scale."""
    axis.set(title=title, ylabel="Held-out R\u00b2")
    axis.tick_params(axis="x", bottom=False, labelbottom=False)
    axis.grid(axis="y", alpha=0.2)
    axis.axhline(0, color=plt.rcParams["axes.labelcolor"], alpha=0.35, linewidth=0.8)
    support.set(xlabel=x_label, ylabel=f"{name} / {total}", ylim=(-0.4, max(total, 1) + 0.4))
    observed = {int(value) for line in support.lines for value in line.get_ydata() if np.isfinite(value)}
    ticks = sorted(observed | {total})
    support.set_yticks(ticks if len(ticks) <= 3 else [ticks[0], ticks[-1]])
    support.tick_params(axis="both", labelsize=8)
    support.yaxis.label.set_size(8)
    support.yaxis.set_label_coords(-0.067, 0.43)
    support.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    support.spines[["top", "right"]].set_visible(False)


def _mark_unavailable(axis: Axes, points: pd.DataFrame) -> None:
    """Annotate saved reasons when a curve panel has no finite scores."""
    if not np.isfinite(points.mean_r2.to_numpy(dtype=float)).any():
        reasons = [str(reason) for reason in points.reason.dropna().unique() if reason]
        message = next(iter(reasons), "No supported scores.")
        axis.text(0.5, 0.55, textwrap.fill(message, width=47), transform=axis.transAxes,
                  va="center", ha="center", fontsize=9)


def _model_handles(panel_d_pcs: int) -> list[Line2D]:
    """Return rank-model legend handles using the saved Panel D PC input count."""
    models = ("rrr", "ridge_units", f"ridge_{panel_d_pcs}pc")
    return [Line2D([], [], color=plt.rcParams["axes.labelcolor"], label=label, linestyle=style)
            for label, style in (MODEL_STYLES[model] for model in models)]


def _panel_d_count(fields: dict, meta: dict, curves: pd.DataFrame) -> int:
    """Read the integer Panel D input count, inferring it from references in older archives."""
    count = fields.get("panel_d_pcs", meta.get("parameters", {}).get("panel_d_pcs"))
    if count is not None:
        return int(count)
    saved_models = set(curves.model)
    return next((size for model, size in PC_REFERENCE_COUNTS.items() if model in saved_models), 30)


def _panel_label(panel: str, panel_d_pcs: int) -> str:
    """Return the label for one panel and its saved number of Panel D input PCs."""
    if panel == "D":
        count = "ten" if panel_d_pcs == 10 else str(panel_d_pcs)
        return f"Rank with {count} source PCs"
    return PANEL_LABELS[panel]


def _main_model(panel: str) -> str:
    """Return the saved primary model label for a Figure 5 or 8 panel."""
    return "ridge" if panel in "AB" else "rrr"


def _direction_text(direction: str) -> str:
    """Convert an ASCII saved direction ID to a readable source-to-target label."""
    return direction.replace("->", " \u2192 ")


def _x_label(panel: str) -> str:
    """Return the x-axis label for the panel's integer source count or rank."""
    return PANEL_LABELS[panel] if panel in "AB" else "Regression rank"
