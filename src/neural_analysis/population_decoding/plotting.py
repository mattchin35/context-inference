"""Saved OOF probabilities on actual elapsed time; no source loading or fitting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.population.timeline_plotting import (
    add_timeline_behavior, available_event_tracks, session_time_bounds,
)

COLORS = ("#2563eb", "#c2410c", "#08916d", "#a855f7")

if TYPE_CHECKING:
    from .preparation import PreparedPopulation


def plot_decoding_timeline(prepared: PreparedPopulation, cell: dict, representation: str,
                           time_range_s: tuple[float, float] | None = None,
                           event_tracks: tuple[str, ...] = (), viewport_revision: int = 0) -> go.Figure:
    """Plot saved preparation/cell probabilities with actual labels on elapsed-second axes.

    Cell probabilities are (W_target,B_scope,C), dimensionless. Optional range
    selects elapsed seconds; the neural payload includes one viewport of padding
    on either side. Only each target's supervised bins are drawn, with explicit
    line breaks between trials, preserving gaps and overlapping trial windows.
    Return Plotly Figure; no arrays, sources, or estimators are modified.
    """
    if representation not in ("units", "pcs"):
        raise ValueError("Unknown population representation")
    tracks = tuple(track for track in event_tracks if track in available_event_tracks(prepared.behavior.events))
    actual_row, probability_row = 2 + len(tracks), 3 + len(tracks)
    titles = ["Recorded choices", *tracks, f"Actual {cell['meta']['target']}", "Held-out population probabilities"]
    figure = make_subplots(rows=probability_row, cols=1, shared_xaxes=True,
                           vertical_spacing=.04, subplot_titles=titles)
    add_timeline_behavior(figure, prepared.behavior, tracks)
    bounds = session_time_bounds(prepared.behavior)
    viewport = time_range_s or bounds
    span = viewport[1] - viewport[0]
    loaded = (max(bounds[0], viewport[0] - span), min(bounds[1], viewport[1] + span))
    classes = cell["meta"]["class_names"]
    target = cell["meta"]["target"]
    bins = len(cell["bin_indices"])
    actual = np.repeat(cell["labels"][:, None], bins, axis=1)
    x, y, hover = _step_track(prepared, cell, actual, loaded)
    figure.add_trace(go.Scatter(x=x, y=y, customdata=hover, name="Actual target", mode="lines",
        line={"color": "#111827"}, connectgaps=False,
        hovertemplate="Trial %{customdata[0]}: class %{y}<extra>Recorded label</extra>"), row=actual_row, col=1)
    probabilities = cell[f"{representation}_probabilities"]
    shown = [1] if target.endswith("choice_switch") else range(len(classes))
    for class_index in shown:
        x, y, hover = _step_track(prepared, cell, probabilities[:, :, class_index], loaded)
        figure.add_trace(go.Scatter(x=x, y=y, customdata=hover, name=f"P({classes[class_index]})",
            mode="lines", connectgaps=False, line={"color": COLORS[class_index]},
            hovertemplate="Trial %{customdata[0]}<br>Relative [%{customdata[1]},%{customdata[2]}) s"
                          "<br>Held-out fold %{customdata[3]}<br>P=%{y:.3f}<extra>%{fullData.name}</extra>"),
            row=probability_row, col=1)
    figure.update_yaxes(tickvals=list(range(len(classes))), ticktext=classes, range=[-.5, len(classes)-.5], row=actual_row, col=1)
    figure.update_yaxes(range=[0, 1], title_text="Probability", row=probability_row, col=1)
    figure.update_xaxes(range=list(viewport))
    figure.update_xaxes(title_text="Session elapsed time (s)", row=probability_row, col=1)
    summary = cell["meta"]["representations"][representation]["pooled"]
    caption = (f"{cell['meta']['region']} {target}, {representation}; block-held-out bins; "
               f"balanced accuracy={summary['balanced_accuracy']}, log loss={summary['log_loss']}.")
    figure.add_annotation(text=caption, x=0, y=-.14, xref="paper", yref="paper", showarrow=False, xanchor="left")
    figure.update_layout(template="plotly_white", height=650 + 70*len(tracks), font={"size": 15},
        margin={"b": 100}, uirevision=f"population-{prepared.session_id}-{viewport_revision}",
        meta={"target": target, "representation": representation, "primary_time_axis": "xaxis",
              "time_axes": ["xaxis" if i == 1 else f"xaxis{i}" for i in range(1, probability_row+1)],
              "loaded_range_s": list(loaded), "caption": caption})
    return figure


def export_png(path: Path, prepared: PreparedPopulation, cell: dict, representation: str,
               time_range_s: tuple[float, float] | None = None) -> None:
    """Export saved probabilities to new scalar PNG Path; axes use elapsed seconds/[0,1].

    Opaque white background and a caption include target/scope/held-out scores.
    Uses Matplotlib directly without browser/Kaleido dependencies. Existing
    files raise FileExistsError; no fitting/source loading or input mutation.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    path = Path(path)
    bounds = time_range_s or session_time_bounds(prepared.behavior)
    figure = Figure(figsize=(15, 7), facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    target = cell["meta"]["target"]
    classes = cell["meta"]["class_names"]
    for index in ([1] if target.endswith("choice_switch") else range(len(classes))):
        x, y, _ = _step_track(prepared, cell, cell[f"{representation}_probabilities"][:, :, index], bounds)
        axis.plot(x, y, label=f"P({classes[index]})", color=COLORS[index])
    axis.set(xlim=bounds, ylim=(0, 1), xlabel="Session elapsed time (s)", ylabel="Held-out probability")
    axis.tick_params(labelsize=14)
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.legend(fontsize=12)
    score = cell["meta"]["representations"][representation]["pooled"]
    caption = (f"{prepared.session_id}: {cell['meta']['region']} {target}, {representation}. "
        f"Choice-relative {cell['meta']['supervised_interval_s']} s; block-held-out predictions.\n"
        f"Balanced accuracy={score['balanced_accuracy']}; log loss={score['log_loss']}. "
        "Outcome scores may include chosen-side information; inspect within-side reward summaries.")
    figure.text(.03, .03, caption, fontsize=11, color="black")
    figure.subplots_adjust(bottom=.23)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=140, facecolor="white", transparent=False)


def _step_track(prepared: PreparedPopulation, cell: dict, values: np.ndarray,
                 loaded_range_s: tuple[float, float]) -> tuple[list, list, list]:
    """Map saved (W_target,B_scope) values to step vertices and trial/bin/fold hover rows.

    X values are elapsed seconds, Y keeps input units. None separates trials so
    real gaps/overlaps never become compressed or interpolated observations.
    Return three aligned Python lists; no external sources are read.
    """
    region = prepared.regions.get(cell["meta"]["region"])
    if region is None:
        return [], [], []
    x, y, hover = [], [], []
    edges = region.bin_edges_s
    for window, original_row in enumerate(cell["row_positions"]):
        trial = prepared.trials.iloc[int(original_row)]
        choice = float(trial.choice_time) - prepared.behavior.time_origin_s
        if choice + edges[-1] < loaded_range_s[0] or choice + edges[0] > loaded_range_s[1]:
            continue
        for index, bin_index in enumerate(cell["bin_indices"]):
            start, end = float(edges[bin_index]), float(edges[bin_index+1])
            x.extend([choice + start, choice + end])
            y.extend([float(values[window, index])] * 2)
            hover.extend([(trial.cur_trial, start, end, int(cell["fold_ids"][window]))] * 2)
        x.append(None)
        y.append(None)
        hover.append((None, None, None, None))
    return x, y, hover
